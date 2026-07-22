#if canImport(Metal)
import Metal
import MetalKit
import simd

/// A renderer that uses Metal to render layer trees on Apple platforms.
///
/// This renderer is used for testing and verification on macOS/iOS.
/// In production WASM environments, `CAWebGPURenderer` is used instead.
///
/// ## Protocol Conformance
///
/// Conforms to the internal renderer-backend contract used by the animation engine.
public final class CAMetalRenderer: CARendererDelegate {

    // MARK: - Properties

    /// The Metal device.
    private var device: MTLDevice?

    /// The command queue.
    private var commandQueue: MTLCommandQueue?

    /// The render pipeline state.
    private var pipelineState: MTLRenderPipelineState?

    /// The vertex buffer for quad rendering.
    private var vertexBuffer: MTLBuffer?

    /// The uniform buffer.
    private var uniformBuffer: MTLBuffer?

    /// The current drawable size.
    public var size: CGSize = CGSize(width: 0, height: 0)

    /// The pixel format for rendering.
    private var pixelFormat: MTLPixelFormat = .bgra8Unorm

    /// The target texture for offscreen rendering.
    internal private(set) var targetTexture: MTLTexture?

    /// Whether the destination size is inferred from the root layer bounds.
    private var sizesTargetFromRootBounds = true

    /// The most recent submission, retained so native verification can wait for completion.
    internal private(set) var lastCommandBuffer: MTLCommandBuffer?

    /// The latest synchronous renderer failure, cleared after a successful submission.
    public private(set) var lastRenderError: CARendererError?

    // MARK: - Initialization

    public init() {}

    internal init(destination texture: any MTLTexture) throws {
        try configure(device: texture.device, destination: texture)
        sizesTargetFromRootBounds = false
    }

    // MARK: - CARenderer

    @MainActor public func initialize() async throws {
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererError.deviceNotAvailable
        }
        try configure(device: device, destination: nil)
        lastRenderError = nil
    }

    internal func setDestination(_ texture: any MTLTexture) throws {
        try configure(device: texture.device, destination: texture)
        sizesTargetFromRootBounds = false
        lastRenderError = nil
    }

    public func resize(width: Int, height: Int) {
        do {
            try resizeTarget(width: width, height: height)
            sizesTargetFromRootBounds = false
            lastRenderError = nil
        } catch let error as CARendererError {
            lastRenderError = error
        } catch {
            lastRenderError = .renderingFailed(error.localizedDescription)
        }
    }

    public func render(layer rootLayer: CALayer) {
        do {
            try prepareForRendering(rootLayer)
        } catch let error as CARendererError {
            lastRenderError = error
            return
        } catch {
            lastRenderError = .renderingFailed(error.localizedDescription)
            return
        }
        guard let commandQueue, let pipelineState, let targetTexture else {
            lastRenderError = .renderingFailed("Metal renderer configuration is incomplete")
            return
        }

        // Phase 1 (PERFORMANCE_DESIGN.md §3.6): mirror CAWebGPURenderer
        // and bump the per-frame token before any presentation cache lookup.
        CALayer.advanceFrameToken()

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            lastRenderError = .renderingFailed("Unable to create a Metal command buffer")
            return
        }

        // Create render pass descriptor
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = targetTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        // Create render encoder
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            lastRenderError = .renderingFailed("Unable to create a Metal render encoder")
            return
        }

        encoder.setRenderPipelineState(pipelineState)

        // Create projection matrix for SpriteKit/CoreAnimation coordinate system (Y+ up)
        // - y=0 maps to NDC=-1 (bottom of screen)
        // - y=height maps to NDC=+1 (top of screen)
        let projectionMatrix = simd_float4x4.orthographic(
            left: 0,
            right: Float(size.width),
            bottom: 0,
            top: Float(size.height),
            near: -1000,
            far: 1000
        )

        // Render layer tree
        renderLayer(rootLayer, encoder: encoder, parentMatrix: projectionMatrix)

        encoder.endEncoding()
        lastCommandBuffer = commandBuffer
        commandBuffer.commit()
        lastRenderError = nil

        // Phase 1 commit-end housekeeping (PERFORMANCE_DESIGN.md §3.8 / §6.5).
        // Mirror CAWebGPURenderer: clear after submit so any setter that
        // runs in the same tick re-marks for the NEXT frame, not this one.
        rootLayer.recursivelyClearDirtyAfterCommit()
    }

    public func invalidate() {
        pipelineState = nil
        vertexBuffer = nil
        uniformBuffer = nil
        targetTexture = nil
        commandQueue = nil
        device = nil
        lastCommandBuffer = nil
        lastRenderError = nil
    }

    // MARK: - Private Methods

    private func createPipeline() throws {
        guard let device = device else {
            throw CARendererError.deviceNotAvailable
        }

        // Shader source code
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexIn {
            float2 position [[attribute(0)]];
            float2 texCoord [[attribute(1)]];
            float4 color [[attribute(2)]];
        };

        struct VertexOut {
            float4 position [[position]];
            float2 texCoord;
            float4 color;
        };

        struct Uniforms {
            float4x4 mvpMatrix;
            float opacity;
            float cornerRadius;
            float2 padding;
        };

        vertex VertexOut vertex_main(
            VertexIn in [[stage_in]],
            constant Uniforms& uniforms [[buffer(1)]]
        ) {
            VertexOut out;
            out.position = uniforms.mvpMatrix * float4(in.position, 0.0, 1.0);
            out.texCoord = in.texCoord;
            out.color = in.color * uniforms.opacity;
            return out;
        }

        fragment float4 fragment_main(VertexOut in [[stage_in]]) {
            return in.color;
        }
        """

        // Compile shader
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            throw CARendererError.shaderCompilationFailed(error.localizedDescription)
        }

        guard let vertexFunction = library.makeFunction(name: "vertex_main"),
              let fragmentFunction = library.makeFunction(name: "fragment_main") else {
            throw CARendererError.shaderCompilationFailed("Failed to create shader functions")
        }

        // Create vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()

        // Position
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0

        // TexCoord
        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0

        // Color
        vertexDescriptor.attributes[2].format = .float4
        vertexDescriptor.attributes[2].offset = MemoryLayout<SIMD2<Float>>.stride * 2
        vertexDescriptor.attributes[2].bufferIndex = 0

        vertexDescriptor.layouts[0].stride = MemoryLayout<CAMetalRendererVertex>.stride

        // Create pipeline descriptor
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        pipelineDescriptor.colorAttachments[0].pixelFormat = pixelFormat

        // Enable blending
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            throw CARendererError.pipelineCreationFailed
        }
    }

    private func configure(
        device: any MTLDevice,
        destination: (any MTLTexture)?
    ) throws {
        guard let commandQueue = device.makeCommandQueue() else {
            throw CARendererError.deviceNotAvailable
        }
        self.device = device
        self.commandQueue = commandQueue
        targetTexture = destination
        if let destination {
            pixelFormat = destination.pixelFormat
            size = CGSize(width: destination.width, height: destination.height)
        }
        try createPipeline()
        createVertexBuffer()
        createUniformBuffer()
    }

    private func prepareForRendering(_ rootLayer: CALayer) throws {
        if device == nil {
            guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
                throw CARendererError.deviceNotAvailable
            }
            try configure(device: defaultDevice, destination: nil)
        }
        guard sizesTargetFromRootBounds else {
            guard targetTexture != nil else {
                throw CARendererError.textureCreationFailed
            }
            return
        }

        let scale = max(rootLayer.contentsScale, 1)
        let widthValue = rootLayer.bounds.size.width * scale
        let heightValue = rootLayer.bounds.size.height * scale
        guard widthValue.isFinite, heightValue.isFinite,
              widthValue > 0, heightValue > 0,
              widthValue <= CGFloat(Int.max),
              heightValue <= CGFloat(Int.max) else {
            throw CARendererError.renderingFailed(
                "The root layer must have finite, positive bounds"
            )
        }
        let width = Int(ceil(widthValue))
        let height = Int(ceil(heightValue))
        if targetTexture?.width != width || targetTexture?.height != height {
            try resizeTarget(width: width, height: height)
        }
    }

    private func resizeTarget(width: Int, height: Int) throws {
        guard width > 0, height > 0 else {
            throw CARendererError.renderingFailed(
                "The Metal destination must have positive dimensions"
            )
        }
        guard let device else {
            throw CARendererError.deviceNotAvailable
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw CARendererError.textureCreationFailed
        }
        size = CGSize(width: width, height: height)
        targetTexture = texture
    }

    private func createVertexBuffer() {
        guard let device = device else { return }

        // Quad vertices (two triangles)
        let vertices: [CAMetalRendererVertex] = [
            // Triangle 1
            CAMetalRendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: SIMD4(1, 1, 1, 1)),
            CAMetalRendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: SIMD4(1, 1, 1, 1)),
            CAMetalRendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: SIMD4(1, 1, 1, 1)),
            // Triangle 2
            CAMetalRendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: SIMD4(1, 1, 1, 1)),
            CAMetalRendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: SIMD4(1, 1, 1, 1)),
            CAMetalRendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: SIMD4(1, 1, 1, 1)),
        ]

        vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<CAMetalRendererVertex>.stride * vertices.count,
            options: .storageModeShared
        )
    }

    private func createUniformBuffer() {
        guard let device = device else { return }

        uniformBuffer = device.makeBuffer(
            length: MemoryLayout<CAMetalRendererUniforms>.stride,
            options: .storageModeShared
        )
    }

    private func renderLayer(
        _ layer: CALayer,
        encoder: MTLRenderCommandEncoder,
        parentMatrix: simd_float4x4
    ) {
        let presentationLayer = layer._renderTimePresentation()

        // Skip hidden layers
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }

        // Calculate model matrix
        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Create scale matrix for layer bounds (column-major order)
        let boundsSize = presentationLayer.bounds.size
        let w = Float(boundsSize.width)
        let h = Float(boundsSize.height)
        let col0 = SIMD4<Float>(w, 0, 0, 0)
        let col1 = SIMD4<Float>(0, h, 0, 0)
        let col2 = SIMD4<Float>(0, 0, 1, 0)
        let col3 = SIMD4<Float>(0, 0, 0, 1)
        let scaleMatrix = simd_float4x4(col0, col1, col2, col3)

        let finalMatrix = modelMatrix * scaleMatrix

        // Update uniforms
        var uniforms = CAMetalRendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: presentationLayer.opacity,
            cornerRadius: Float(presentationLayer.cornerRadius)
        )

        uniformBuffer?.contents().copyMemory(
            from: &uniforms,
            byteCount: MemoryLayout<CAMetalRendererUniforms>.stride
        )

        // Render background color if set
        if presentationLayer.backgroundColor != nil {
            // Update vertex colors with background color
            let color = presentationLayer.backgroundColorComponents

            var vertices: [CAMetalRendererVertex] = [
                CAMetalRendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
                CAMetalRendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CAMetalRendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
                CAMetalRendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CAMetalRendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
                CAMetalRendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            ]

            vertexBuffer?.contents().copyMemory(
                from: &vertices,
                byteCount: MemoryLayout<CAMetalRendererVertex>.stride * vertices.count
            )

            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }

        // Render sublayers
        if let sublayers = layer.sublayers {
            // Use sublayerMatrix helper to apply sublayerTransform and bounds.origin offset
            let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)

            for sublayer in sublayers {
                renderLayer(sublayer, encoder: encoder, parentMatrix: sublayerMatrix)
            }
        }
    }
}

#endif
