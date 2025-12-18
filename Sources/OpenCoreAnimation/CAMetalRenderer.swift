#if canImport(Metal)
import Metal
import MetalKit
import simd

/// A renderer that uses Metal to render layer trees on Apple platforms.
///
/// This renderer is used for testing and verification on macOS/iOS.
/// In production WASM environments, `CAWebGPURenderer` is used instead.
public final class CAMetalRenderer: CARenderer {

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
    public private(set) var size: CGSize = CGSize(width: 0, height: 0)

    /// The pixel format for rendering.
    private let pixelFormat: MTLPixelFormat = .bgra8Unorm

    /// The target texture for offscreen rendering.
    private var targetTexture: MTLTexture?

    // MARK: - Initialization

    public init() {}

    // MARK: - CARenderer

    public func initialize() async throws {
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererError.deviceNotAvailable
        }
        self.device = device

        // Create command queue
        guard let commandQueue = device.makeCommandQueue() else {
            throw CARendererError.deviceNotAvailable
        }
        self.commandQueue = commandQueue

        // Create render pipeline
        try createPipeline()

        // Create vertex buffer for a quad (two triangles)
        createVertexBuffer()

        // Create uniform buffer
        createUniformBuffer()
    }

    public func resize(width: Int, height: Int) {
        size = CGSize(width: width, height: height)

        // Recreate target texture with new size
        guard let device = device, width > 0, height > 0 else { return }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        targetTexture = device.makeTexture(descriptor: descriptor)
    }

    public func render(layer rootLayer: CALayer) {
        guard let _ = device,
              let commandQueue = commandQueue,
              let pipelineState = pipelineState,
              let targetTexture = targetTexture else { return }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // Create render pass descriptor
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = targetTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        // Create render encoder
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        encoder.setRenderPipelineState(pipelineState)

        // Create projection matrix
        let projectionMatrix = simd_float4x4.orthographic(
            left: 0,
            right: Float(size.width),
            bottom: Float(size.height),
            top: 0,
            near: -1000,
            far: 1000
        )

        // Render layer tree
        renderLayer(rootLayer, encoder: encoder, parentMatrix: projectionMatrix)

        encoder.endEncoding()
        commandBuffer.commit()
    }

    public func invalidate() {
        pipelineState = nil
        vertexBuffer = nil
        uniformBuffer = nil
        targetTexture = nil
        commandQueue = nil
        device = nil
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

        vertexDescriptor.layouts[0].stride = MemoryLayout<CARendererVertex>.stride

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

    private func createVertexBuffer() {
        guard let device = device else { return }

        // Quad vertices (two triangles)
        let vertices: [CARendererVertex] = [
            // Triangle 1
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: SIMD4(1, 1, 1, 1)),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: SIMD4(1, 1, 1, 1)),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: SIMD4(1, 1, 1, 1)),
            // Triangle 2
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: SIMD4(1, 1, 1, 1)),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: SIMD4(1, 1, 1, 1)),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: SIMD4(1, 1, 1, 1)),
        ]

        vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<CARendererVertex>.stride * vertices.count,
            options: .storageModeShared
        )
    }

    private func createUniformBuffer() {
        guard let device = device else { return }

        uniformBuffer = device.makeBuffer(
            length: MemoryLayout<CARendererUniforms>.stride,
            options: .storageModeShared
        )
    }

    private func renderLayer(
        _ layer: CALayer,
        encoder: MTLRenderCommandEncoder,
        parentMatrix: simd_float4x4
    ) {
        // Skip hidden layers
        guard !layer.isHidden && layer.opacity > 0 else { return }

        // Calculate model matrix
        let modelMatrix = layer.modelMatrix(parentMatrix: parentMatrix)

        // Create scale matrix for layer bounds (column-major order)
        let boundsSize = layer.bounds.size
        let w = Float(boundsSize.width)
        let h = Float(boundsSize.height)
        let col0 = SIMD4<Float>(w, 0, 0, 0)
        let col1 = SIMD4<Float>(0, h, 0, 0)
        let col2 = SIMD4<Float>(0, 0, 1, 0)
        let col3 = SIMD4<Float>(0, 0, 0, 1)
        let scaleMatrix = simd_float4x4(col0, col1, col2, col3)

        let finalMatrix = modelMatrix * scaleMatrix

        // Update uniforms
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: layer.opacity,
            cornerRadius: Float(layer.cornerRadius)
        )

        uniformBuffer?.contents().copyMemory(
            from: &uniforms,
            byteCount: MemoryLayout<CARendererUniforms>.stride
        )

        // Render background color if set
        if layer.backgroundColor != nil {
            // Update vertex colors with background color
            let color = layer.backgroundColorComponents

            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            ]

            vertexBuffer?.contents().copyMemory(
                from: &vertices,
                byteCount: MemoryLayout<CARendererVertex>.stride * vertices.count
            )

            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }

        // Render sublayers
        if let sublayers = layer.sublayers {
            // Apply sublayer transform
            var sublayerMatrix = modelMatrix
            if !CATransform3DIsIdentity(layer.sublayerTransform) {
                sublayerMatrix = sublayerMatrix * layer.sublayerTransform.simdMatrix
            }

            for sublayer in sublayers {
                renderLayer(sublayer, encoder: encoder, parentMatrix: sublayerMatrix)
            }
        }
    }
}

#endif
