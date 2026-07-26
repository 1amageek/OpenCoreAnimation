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
@MainActor public final class CAMetalRenderer: CARendererDelegate {

    // MARK: - Properties

    /// The Metal device.
    private var device: MTLDevice?

    /// The command queue.
    private var commandQueue: MTLCommandQueue?

    /// The client-owned command queue, when supplied through renderer options.
    private var clientCommandQueue: MTLCommandQueue?

    /// The render pipeline state.
    private var pipelineState: MTLRenderPipelineState?

    /// The current drawable size.
    public var size: CGSize = CGSize(width: 0, height: 0)

    /// The pixel format for rendering.
    private var pixelFormat: MTLPixelFormat = .bgra8Unorm

    /// The requested output color space.
    private var outputColorSpace: OpenCoreGraphics.CGColorSpace?

    /// The target texture for offscreen rendering.
    internal private(set) var targetTexture: MTLTexture?

    /// Whether the destination size is inferred from the root layer bounds.
    private var sizesTargetFromRootBounds = true

    /// The most recent submission, retained so native verification can wait for completion.
    internal private(set) var lastCommandBuffer: MTLCommandBuffer?

    /// The latest synchronous renderer failure, cleared after a successful submission.
    public private(set) var lastRenderError: CARendererError?

    internal var synchronousRenderError: CARendererError? {
        lastRenderError
    }

    internal var configuredCommandQueue:
        (any MTLCommandQueue)? {
        commandQueue
    }

    private var retainedAnimationEvaluator:
        CACommittedAnimationEvaluator?
    private var retainedAnimationFrameToken: UInt64?
    private var retainedAnimationRootIdentity:
        ObjectIdentifier?

    // MARK: - Initialization

    public init() {}

    internal init(
        destination texture: any MTLTexture,
        commandQueue: (any MTLCommandQueue)? = nil,
        outputColorSpace: OpenCoreGraphics.CGColorSpace? = nil
    ) {
        do {
            try configure(
                device: texture.device,
                destination: texture,
                clientCommandQueue: commandQueue,
                outputColorSpace: outputColorSpace
            )
            sizesTargetFromRootBounds = false
            lastRenderError = nil
        } catch let error as CARendererError {
            lastRenderError = error
        } catch {
            lastRenderError = .renderingFailed(
                String(describing: error)
            )
        }
    }

    // MARK: - CARenderer

    @MainActor public func initialize() async throws {
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererError.deviceNotAvailable
        }
        try configure(
            device: device,
            destination: nil,
            clientCommandQueue: nil,
            outputColorSpace: nil
        )
        lastRenderError = nil
    }

    internal func replaceDestination(
        _ texture: any MTLTexture
    ) {
        do {
            try configure(
                device: texture.device,
                destination: texture,
                clientCommandQueue: clientCommandQueue,
                outputColorSpace: outputColorSpace
            )
            sizesTargetFromRootBounds = false
            lastRenderError = nil
        } catch let error as CARendererError {
            lastRenderError = error
        } catch {
            lastRenderError = .renderingFailed(
                String(describing: error)
            )
        }
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
        let committedState = rootLayer.pendingCommittedRenderState
        let snapshot: CARenderSnapshot
        let committedFrameToken: UInt64?
        switch committedState {
        case .snapshot(let committedSnapshot):
            retainedAnimationEvaluator = nil
            retainedAnimationFrameToken = nil
            retainedAnimationRootIdentity = nil
            snapshot = committedSnapshot
            committedFrameToken = committedSnapshot.frameToken
        case .captureFailure(_, let error):
            retainedAnimationEvaluator = nil
            retainedAnimationFrameToken = nil
            retainedAnimationRootIdentity = nil
            lastRenderError = error
            return
        case .animationEvaluator(
            let frameToken,
            let evaluator
        ):
            retainedAnimationEvaluator = evaluator
            retainedAnimationFrameToken = frameToken
            retainedAnimationRootIdentity =
                ObjectIdentifier(rootLayer)
            CALayer.advanceFrameToken()
            do {
                snapshot = try evaluator.snapshot(
                    frameToken: frameToken
                )
            } catch {
                lastRenderError = error
                return
            }
            committedFrameToken = frameToken
        case nil:
            if retainedAnimationRootIdentity
                    == ObjectIdentifier(rootLayer),
               let retainedAnimationEvaluator,
               let retainedAnimationFrameToken {
                CALayer.advanceFrameToken()
                do {
                    snapshot =
                        try retainedAnimationEvaluator
                            .snapshot(
                                frameToken:
                                    retainedAnimationFrameToken
                            )
                } catch {
                    lastRenderError = error
                    return
                }
                committedFrameToken = nil
            } else {
                CALayer.advanceFrameToken()
                do {
                    snapshot = try CARenderSnapshot.capture(
                        rootLayer,
                        frameToken:
                            CALayer._currentFrameToken
                    )
                } catch {
                    lastRenderError = error
                    return
                }
                committedFrameToken = nil
            }
        }

        do {
            try prepareForRendering(snapshot)
        } catch let error as CARendererError {
            lastRenderError = error
            return
        } catch {
            lastRenderError = .renderingFailed(error.localizedDescription)
            return
        }

        guard render(snapshot: snapshot) else { return }

        // Phase 1 commit-end housekeeping (PERFORMANCE_DESIGN.md §3.8 / §6.5).
        // Mirror CAWebGPURenderer: clear after submit so any setter that
        // runs in the same tick re-marks for the NEXT frame, not this one.
        rootLayer.recursivelyClearDirtyAfterCommit(matching: snapshot)
        if let committedFrameToken {
            rootLayer.acknowledgeCommittedRenderState(
                frameToken: committedFrameToken
            )
        }
        rootLayer.completeTransactionsAfterRenderRecursively()
    }

    /// Encodes one immutable frame without consulting the mutable layer tree.
    @discardableResult
    internal func render(snapshot: CARenderSnapshot) -> Bool {
        if let unsupportedFeature = unsupportedFeature(in: snapshot) {
            lastRenderError = .unsupportedCommittedSnapshotFeature(
                unsupportedFeature
            )
            return false
        }
        guard let device, let commandQueue,
              let pipelineState, let targetTexture else {
            lastRenderError = .renderingFailed("Metal renderer configuration is incomplete")
            return false
        }

        let preparedDraws: [CAMetalPreparedDraw]
        do {
            preparedDraws = try prepareDraws(
                for: snapshot,
                device: device
            )
        } catch let error as CARendererError {
            lastRenderError = error
            return false
        } catch {
            lastRenderError = .renderingFailed(
                String(describing: error)
            )
            return false
        }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            lastRenderError = .renderingFailed("Unable to create a Metal command buffer")
            return false
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
            return false
        }

        encoder.setRenderPipelineState(pipelineState)

        for draw in preparedDraws {
            encoder.setVertexBuffer(
                draw.vertexBuffer,
                offset: 0,
                index: 0
            )
            encoder.setVertexBuffer(
                draw.uniformBuffer,
                offset: 0,
                index: 1
            )
            encoder.drawPrimitives(
                type: .triangle,
                vertexStart: 0,
                vertexCount: 6
            )
        }

        encoder.endEncoding()
        lastCommandBuffer = commandBuffer
        commandBuffer.commit()
        if clientCommandQueue == nil {
            commandBuffer.waitUntilScheduled()
        }
        lastRenderError = nil
        return true
    }

    public func invalidate() {
        pipelineState = nil
        targetTexture = nil
        commandQueue = nil
        clientCommandQueue = nil
        device = nil
        outputColorSpace = nil
        lastCommandBuffer = nil
        lastRenderError = nil
    }

    // MARK: - Private Methods

    private func makePipeline(
        device: any MTLDevice,
        pixelFormat: MTLPixelFormat
    ) throws -> any MTLRenderPipelineState {
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
            return try device.makeRenderPipelineState(
                descriptor: pipelineDescriptor
            )
        } catch {
            throw CARendererError.pipelineCreationFailed
        }
    }

    private func configure(
        device: any MTLDevice,
        destination: (any MTLTexture)?,
        clientCommandQueue:
            (any MTLCommandQueue)?,
        outputColorSpace:
            OpenCoreGraphics.CGColorSpace?
    ) throws {
        if let destination,
           !destination.usage.contains(.renderTarget) {
            throw CARendererError
                .metalDestinationMissingRenderTargetUsage
        }
        if let clientCommandQueue,
           ObjectIdentifier(clientCommandQueue.device)
                != ObjectIdentifier(device) {
            throw CARendererError
                .rendererCommandQueueDeviceMismatch
        }
        guard let configuredCommandQueue =
                clientCommandQueue
                ?? device.makeCommandQueue() else {
            throw CARendererError.deviceNotAvailable
        }
        let configuredPixelFormat =
            destination?.pixelFormat ?? .bgra8Unorm
        let configuredSize: CGSize
        if let destination {
            configuredSize = CGSize(
                width: destination.width,
                height: destination.height
            )
        } else {
            configuredSize = CGSize(width: 0, height: 0)
        }
        let configuredPipeline = try makePipeline(
            device: device,
            pixelFormat: configuredPixelFormat
        )

        self.device = device
        commandQueue = configuredCommandQueue
        self.clientCommandQueue = clientCommandQueue
        pipelineState = configuredPipeline
        targetTexture = destination
        pixelFormat = configuredPixelFormat
        size = configuredSize
        self.outputColorSpace = outputColorSpace
    }

    private func prepareForRendering(_ snapshot: CARenderSnapshot) throws {
        if let unsupportedFeature = unsupportedFeature(in: snapshot) {
            throw CARendererError.unsupportedCommittedSnapshotFeature(
                unsupportedFeature
            )
        }
        if device == nil {
            guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
                throw CARendererError.deviceNotAvailable
            }
            try configure(
                device: defaultDevice,
                destination: nil,
                clientCommandQueue: nil,
                outputColorSpace: nil
            )
        }
        guard sizesTargetFromRootBounds else {
            guard targetTexture != nil else {
                throw CARendererError.textureCreationFailed
            }
            return
        }

        let scale = max(snapshot.rootContentsScale, 1)
        let widthValue = snapshot.rootBounds.size.width * scale
        let heightValue = snapshot.rootBounds.size.height * scale
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

    private func unsupportedFeature(
        in snapshot: CARenderSnapshot
    ) -> CARenderSnapshotFeature? {
        if snapshot.nodes.contains(where: {
            $0.presentationValues.transition != nil
        }) {
            return .transition
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.tiled != nil
        }) {
            return .tiledLayer
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.emitter != nil
        }) {
            return .emitter
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.replicator != nil
        }) {
            return .replicatorInstances
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.isTransformLayer
                || !CATransform3DIsAffine(
                    $0.presentationValues.transform
                )
                || !CATransform3DIsAffine(
                    $0.presentationValues.sublayerTransform
                )
        }) {
            return .transformDepth
        }
        if snapshot.nodes.contains(where: { $0.maskIndex != nil }) {
            return .contentMask
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.masksToBounds
        }) {
            return .clipping
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.cornerRadius > 0
        }) {
            return .roundedCorners
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.borderWidth > 0
                && $0.presentationValues.borderColor != nil
        }) {
            return .border
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.isGeometryFlipped
        }) {
            return .geometryFlipped
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.edgeAntialiasingMask != 15
        }) {
            return .edgeAntialiasing
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.toneMapMode != .automatic
                || $0.presentationValues.preferredDynamicRange
                    != .standard
                || $0.presentationValues.contentsHeadroom != 0
        }) {
            return .dynamicRange
        }
        if snapshot.nodes.contains(where: {
            !$0.presentationValues.isDoubleSided
        }) {
            return .backfaceCulling
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.allowsGroupOpacity
                && $0.presentationValues.opacity < 1
                && !$0.childIndices.isEmpty
        }) {
            return .groupOpacity
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.shouldRasterize
        }) {
            return .rasterization
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.shadow != nil
        }) {
            return .shadow
        }
        if snapshot.nodes.contains(where: {
            !$0.presentationValues.filters.isEmpty
        }) {
            return .filters
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.compositingFilter != nil
                || !$0.presentationValues.backgroundFilters.isEmpty
        }) {
            return .backdropComposition
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.imageContents != nil
        }) {
            return .imageContents
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.gradient != nil
        }) {
            return .gradient
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.shape?.fill != nil
                || $0.presentationValues.shape?.stroke != nil
        }) {
            return .shape
        }
        if snapshot.nodes.contains(where: {
            $0.presentationValues.text?.configuration?.text.isEmpty == false
        }) {
            return .text
        }
        return nil
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

    private func prepareDraws(
        for snapshot: CARenderSnapshot,
        device: any MTLDevice
    ) throws -> [CAMetalPreparedDraw] {
        let projectionMatrix = simd_float4x4.orthographic(
            left: 0,
            right: Float(size.width),
            bottom: 0,
            top: Float(size.height),
            near: -1000,
            far: 1000
        )
        var draws: [CAMetalPreparedDraw] = []
        draws.reserveCapacity(snapshot.nodes.count)
        try prepareNode(
            at: snapshot.rootIndex,
            in: snapshot,
            device: device,
            parentMatrix: projectionMatrix,
            parentOpacity: 1,
            draws: &draws
        )
        return draws
    }

    private func prepareNode(
        at nodeIndex: Int,
        in snapshot: CARenderSnapshot,
        device: any MTLDevice,
        parentMatrix: simd_float4x4,
        parentOpacity: Float,
        draws: inout [CAMetalPreparedDraw]
    ) throws {
        let node = snapshot.nodes[nodeIndex]
        let values = node.presentationValues

        guard !values.isHidden && values.opacity > 0 else { return }
        let cumulativeOpacity = parentOpacity * values.opacity
        guard cumulativeOpacity > 0 else { return }

        let modelMatrix = values.modelMatrix(parentMatrix: parentMatrix)
        let w = values.boundsSize.x
        let h = values.boundsSize.y
        let col0 = SIMD4<Float>(w, 0, 0, 0)
        let col1 = SIMD4<Float>(0, h, 0, 0)
        let col2 = SIMD4<Float>(0, 0, 1, 0)
        let col3 = SIMD4<Float>(0, 0, 0, 1)
        let scaleMatrix = simd_float4x4(col0, col1, col2, col3)

        let finalMatrix = modelMatrix * scaleMatrix

        var uniforms = CAMetalRendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: cumulativeOpacity,
            cornerRadius: values.cornerRadius
        )

        if let backgroundColor = values.backgroundColor {
            let color = try outputColor(
                from: backgroundColor
            )
            let vertices: [CAMetalRendererVertex] = [
                CAMetalRendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
                CAMetalRendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CAMetalRendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
                CAMetalRendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CAMetalRendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
                CAMetalRendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            ]

            guard let vertexBuffer = device.makeBuffer(
                bytes: vertices,
                length:
                    MemoryLayout<CAMetalRendererVertex>.stride
                    * vertices.count,
                options: .storageModeShared
            ), let uniformBuffer = device.makeBuffer(
                bytes: &uniforms,
                length:
                    MemoryLayout<CAMetalRendererUniforms>.stride,
                options: .storageModeShared
            ) else {
                throw CARendererError.bufferCreationFailed
            }
            draws.append(
                CAMetalPreparedDraw(
                    vertexBuffer: vertexBuffer,
                    uniformBuffer: uniformBuffer
                )
            )
        }

        if !node.childIndices.isEmpty {
            let sublayerMatrix = values.sublayerMatrix(modelMatrix: modelMatrix)
            for childIndex in node.childIndices {
                try prepareNode(
                    at: childIndex,
                    in: snapshot,
                    device: device,
                    parentMatrix: sublayerMatrix,
                    parentOpacity: cumulativeOpacity,
                    draws: &draws
                )
            }
        }
    }

    private func outputColor(
        from color: SIMD4<Float>
    ) throws -> SIMD4<Float> {
        guard let outputColorSpace else {
            return color
        }
        let sourceColor = OpenCoreGraphics.CGColor(
            red: CGFloat(color.x),
            green: CGFloat(color.y),
            blue: CGFloat(color.z),
            alpha: CGFloat(color.w)
        )
        guard let converted = sourceColor.converted(
            to: outputColorSpace,
            intent: .defaultIntent,
            options: nil
        ), let components = converted.components,
           components.count == 4,
           components.allSatisfy(\.isFinite) else {
            throw CARendererError.rendererColorConversionFailed
        }
        return SIMD4(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
    }
}

private struct CAMetalPreparedDraw {
    internal let vertexBuffer: any MTLBuffer
    internal let uniformBuffer: any MTLBuffer
}

private extension CARenderSnapshot.PresentationValues {
    func modelMatrix(
        parentMatrix: simd_float4x4 = matrix_identity_float4x4
    ) -> simd_float4x4 {
        var matrix = parentMatrix
        if !CATransform3DIsIdentity(
            replicatorInstanceTransform
        ) {
            matrix =
                matrix * replicatorInstanceTransform.simdMatrix
        }
        let positionTranslation = simd_float4x4(translation: position)
        matrix = matrix * positionTranslation
        if !CATransform3DIsIdentity(transform) {
            matrix = matrix * transform.simdMatrix
        }
        let anchorTranslation = simd_float4x4(translation: anchorOffset)
        matrix = matrix * anchorTranslation
        return matrix
    }

    func sublayerMatrix(modelMatrix: simd_float4x4) -> simd_float4x4 {
        var result = modelMatrix
        if !CATransform3DIsIdentity(sublayerTransform) {
            result = result * sublayerTransform.simdMatrix
        }
        if boundsOrigin.x != 0 || boundsOrigin.y != 0 {
            let boundsTranslation = simd_float4x4(translation: SIMD3<Float>(
                -boundsOrigin.x,
                -boundsOrigin.y,
                0
            ))
            result = result * boundsTranslation
        }
        return result
    }
}

#endif
