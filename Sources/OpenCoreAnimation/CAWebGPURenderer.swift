#if arch(wasm32)
@_spi(SoftwareBitmapContext) import OpenCoreGraphics
@_spi(WebGPUInterop) import OpenCoreImage
import Foundation
import JavaScriptKit
import SwiftWebGPU

private let caRendererAlignedUniformSize: UInt64 = {
    let baseSize = UInt64(MemoryLayout<CARendererUniforms>.stride)
    let alignment: UInt64 = 256
    return ((baseSize + alignment - 1) / alignment) * alignment
}()

// MARK: - File Organization
//
// Types, helpers, and shaders are now organized in the Rendering/WebGPU directory:
// - Types/Matrix4x4.swift
// - Types/RendererTypes.swift
// - Types/ParticleTypes.swift
// - Types/GeometryTypes.swift
// - Internal/BufferPool.swift
// - Internal/TextureManager.swift
// - Internal/GeometryCache.swift
// - Shaders/CAWebGPUShaders.swift
// - Extensions/CALayerWebGPUExtensions.swift

// MARK: - Coordinate System Documentation
//
// OpenCoreAnimation uses a **SpriteKit-compatible coordinate system (Y-up)**:
//
//   ┌─────────────────────────────────┐
//   │ Origin: Bottom-left (0, 0)     │
//   │ X-axis: Positive X → RIGHT     │
//   │ Y-axis: Positive Y → UP        │
//   └─────────────────────────────────┘
//
// Reference: https://developer.apple.com/documentation/spritekit/about-spritekit-coordinate-systems
// "A positive x coordinate goes to the right and a positive y coordinate goes up the screen."
//
// This differs from iOS UIKit/CoreAnimation which uses Y-down (origin at top-left).
//
// ## Why Y-up?
//
// 1. SpriteKit compatibility - Same coordinate system as Apple's game framework
// 2. Mathematical convention - Standard Cartesian coordinate system
// 3. WebGPU NDC alignment - WebGPU NDC is Y-up (-1 at bottom, +1 at top)
//
// ## Texture Coordinate Handling
//
// Image and text textures store pixel data with row 0 at the TOP.
// For Y-up rendering, the V coordinate must be flipped:
//
//   Screen (Y-up)         Texture
//   Y=height ───          V=0 ─── (top of image)
//            │              │
//            │   flip V     │
//            │              │
//   Y=0      ───          V=1 ─── (bottom of image)
//
// V-flip is applied to:
// - Image rendering (renderImageContents)
// - Text rendering (renderTextLayer)
// - 9-patch rendering (render9PatchImage)
//
// V-flip is NOT applied to solid color rendering because texCoord is used
// for position-based calculations (corner radius, gradients), not texture sampling.

// MARK: - WebGPU Renderer

/// Typed cache key for the renderer's textured-content view / bind group caches.
///
/// `ObjectIdentifier` alone is just a heap address. The same renderer caches
/// textured draws keyed by `CGImage` identity, live `CALayer` identity for
/// rasterization, and immutable transition-snapshot identity. Using a raw
/// `ObjectIdentifier` would let those namespaces alias whenever an object
/// address is reused, returning a stale `GPUTextureView`. Tagging the kind
/// keeps the namespaces disjoint.
fileprivate enum EmitterTextureSampling: CaseIterable, Hashable {
    case nearestNearest
    case nearestLinear
    case nearestTrilinear
    case linearNearest
    case linearLinear
    case linearTrilinear

    init?(magnificationFilter: String, minificationFilter: String) {
        let magnificationIsNearest: Bool
        switch magnificationFilter {
        case CALayerContentsFilter.nearest.rawValue:
            magnificationIsNearest = true
        case CALayerContentsFilter.linear.rawValue, CALayerContentsFilter.trilinear.rawValue:
            magnificationIsNearest = false
        default:
            return nil
        }

        switch (magnificationIsNearest, minificationFilter) {
        case (true, CALayerContentsFilter.nearest.rawValue): self = .nearestNearest
        case (true, CALayerContentsFilter.linear.rawValue): self = .nearestLinear
        case (true, CALayerContentsFilter.trilinear.rawValue): self = .nearestTrilinear
        case (false, CALayerContentsFilter.nearest.rawValue): self = .linearNearest
        case (false, CALayerContentsFilter.linear.rawValue): self = .linearLinear
        case (false, CALayerContentsFilter.trilinear.rawValue): self = .linearTrilinear
        default: return nil
        }
    }

    var magnificationFilter: GPUFilterMode {
        switch self {
        case .nearestNearest, .nearestLinear, .nearestTrilinear: return .nearest
        case .linearNearest, .linearLinear, .linearTrilinear: return .linear
        }
    }

    var minificationFilter: GPUFilterMode {
        switch self {
        case .nearestNearest, .linearNearest: return .nearest
        case .nearestLinear, .nearestTrilinear, .linearLinear, .linearTrilinear: return .linear
        }
    }

    var usesMipmaps: Bool {
        switch self {
        case .nearestTrilinear, .linearTrilinear: return true
        default: return false
        }
    }
}

fileprivate enum TexturedCacheKey: Hashable {
    case image(ObjectIdentifier)
    case emitterImage(ObjectIdentifier, EmitterTextureSampling)
    case rasterizedLayer(LayerRenderKey, RasterizationCachePurpose)
    case transitionSource(ObjectIdentifier)
    case transitionTarget(ObjectIdentifier)
    case transitionFilter(ObjectIdentifier)
}

private struct PendingTileDraw {
    let tiledLayer: CATiledLayer
    let delegate: any CALayerDelegate
    let tileKey: CATiledLayer.TileKey
    let tileRect: CGRect
    let scale: CGFloat
    let pixelWidth: Int
    let pixelHeight: Int
}

private struct PrerasterizedTexture {
    let texture: GPUTexture
    let purpose: RasterizationCachePurpose
    let captureBounds: CGRect
}

private struct RasterShadowCompositeUniforms {
    var shadowColor: SIMD4<Float>
    var shadowOffsetUV: SIMD2<Float>
    var padding: SIMD2<Float> = .zero
}

private struct LayerPrepassTarget {
    let layer: CALayer
    let presentationLayer: CALayer
    let parentMatrix: Matrix4x4
    let renderKey: LayerRenderKey
    let timeOffset: CFTimeInterval
}

private struct BackdropCompositionTarget {
    let prepass: LayerPrepassTarget
    let scope: LayerPrepassTarget?
    let backgroundFilterExtent: LayerPrepassTarget?
    let ancestorRenderKeys: [LayerRenderKey]
    let depth: Int
    let clipAncestors: [LayerPrepassTarget]
    let contentMaskAncestors: [LayerPrepassTarget]
    let targetContentMask: LayerPrepassTarget?
    let sourceOpacity: Float
    let sourceColor: SIMD4<Float>
}

private struct BackdropCompositionRoot {
    let prepass: LayerPrepassTarget
    let clearColor: GPUColor
}

private enum CompositionMaskTarget {
    case clipShape(LayerPrepassTarget)
    case layerContent(LayerPrepassTarget)
}

private struct TransitionParticipantCapture {
    let texture: GPUTexture
    let compositeLayer: CALayer
    let pixelWidth: Int
    let pixelHeight: Int
}

private struct TransitionCapturePair {
    let source: TransitionParticipantCapture
    let target: TransitionParticipantCapture
    let filterExecution: CIWebGPUTransitionExecution?
}

private enum EmitterBirthRemainderKey: Hashable {
    case root(ObjectIdentifier)
    case child(parentBirthSequence: UInt64, cell: ObjectIdentifier)
}

/// GPU resources owned by one filtered layer.
///
/// Filter captures are viewport-sized because they preserve the complete transformed
/// subtree in renderer coordinates. Keeping a resource set per layer prevents sibling
/// and nested filters from overwriting a shared ping-pong chain before the command
/// buffer is submitted.
private final class FilterLayerResources {
    let width: Int
    let height: Int
    let sourceTexture: GPUTexture
    let sourceView: GPUTextureView
    let intermediateTexture: GPUTexture
    let intermediateView: GPUTextureView
    let resultTexture: GPUTexture
    let resultView: GPUTextureView
    let compositeUniformBuffer: GPUBuffer
    private(set) var operationUniformBuffers: [GPUBuffer] = []

    init(device: GPUDevice, width: Int, height: Int, format: GPUTextureFormat) {
        self.width = width
        self.height = height
        let descriptor = GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: format,
            usage: [.renderAttachment, .textureBinding]
        )
        sourceTexture = device.createTexture(descriptor: descriptor)
        sourceView = sourceTexture.createView()
        intermediateTexture = device.createTexture(descriptor: descriptor)
        intermediateView = intermediateTexture.createView()
        resultTexture = device.createTexture(descriptor: descriptor)
        resultView = resultTexture.createView()
        compositeUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(MemoryLayout<FilterCompositeUniforms>.stride),
            usage: [.uniform, .copyDst]
        ))
    }

    func uniformBuffer(forOperationAt index: Int, device: GPUDevice) -> GPUBuffer {
        while operationUniformBuffers.count <= index {
            operationUniformBuffers.append(device.createBuffer(descriptor: GPUBufferDescriptor(
                size: UInt64(max(
                    MemoryLayout<BlurUniforms>.stride,
                    MemoryLayout<FilterCompositeUniforms>.stride
                )),
                usage: [.uniform, .copyDst]
            )))
        }
        return operationUniformBuffers[index]
    }

    func view(for texture: GPUTexture) -> GPUTextureView? {
        if texture === sourceTexture { return sourceView }
        if texture === intermediateTexture { return intermediateView }
        if texture === resultTexture { return resultView }
        return nil
    }

    func destroy() {
        sourceTexture.destroy()
        intermediateTexture.destroy()
        resultTexture.destroy()
        compositeUniformBuffer.destroy()
        for buffer in operationUniformBuffers {
            buffer.destroy()
        }
        operationUniformBuffers.removeAll(keepingCapacity: false)
    }
}

private struct PrerenderedFilter {
    let resources: FilterLayerResources
    let outputTexture: GPUTexture
    let outputView: GPUTextureView
    let appliedContentMask: Bool
}

private final class CompositionLayerResources {
    let backdropTexture: GPUTexture
    let backdropView: GPUTextureView
    let sourcePremultipliedTexture: GPUTexture
    let sourcePremultipliedView: GPUTextureView
    let sourceStraightTexture: GPUTexture
    let sourceStraightView: GPUTextureView
    let backdropStraightTexture: GPUTexture
    let backdropStraightView: GPUTextureView
    let resultPremultipliedTexture: GPUTexture
    let resultPremultipliedView: GPUTextureView
    let backdropMaskTexture: GPUTexture
    let backdropMaskView: GPUTextureView
    let mixedBackdropStraightTexture: GPUTexture
    let mixedBackdropStraightView: GPUTextureView
    let clipShapeTexture: GPUTexture
    let clipShapeView: GPUTextureView
    let clipCumulativeTextureA: GPUTexture
    let clipCumulativeViewA: GPUTextureView
    let clipCumulativeTextureB: GPUTexture
    let clipCumulativeViewB: GPUTextureView
    let clippedSourceTexture: GPUTexture
    let clippedSourceView: GPUTextureView
    let combinedBackdropMaskTexture: GPUTexture
    let combinedBackdropMaskView: GPUTextureView
    let backgroundFilterResources: FilterLayerResources
    let sourceUniformBuffer: GPUBuffer
    let sourceOpacityUniformBuffer: GPUBuffer
    let backdropUniformBuffer: GPUBuffer
    let resultConversionUniformBuffer: GPUBuffer
    let displayUniformBuffer: GPUBuffer
    let transformedDisplayUniformBuffer: GPUBuffer
    let backdropMaskUniformBuffer: GPUBuffer
    let backdropMaskVertexBuffer: GPUBuffer
    private(set) var clipUniformBuffers: [GPUBuffer] = []
    private(set) var contentMaskFilterResources: [FilterLayerResources] = []

    init(device: GPUDevice, width: Int, height: Int, format: GPUTextureFormat) {
        func makeTexture(_ format: GPUTextureFormat) -> GPUTexture {
            device.createTexture(descriptor: GPUTextureDescriptor(
                size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
                format: format,
                usage: [.renderAttachment, .textureBinding]
            ))
        }
        func makeUniformBuffer() -> GPUBuffer {
            device.createBuffer(descriptor: GPUBufferDescriptor(
                size: UInt64(MemoryLayout<FilterCompositeUniforms>.stride),
                usage: [.uniform, .copyDst]
            ))
        }

        backdropTexture = makeTexture(format)
        backdropView = backdropTexture.createView()
        sourcePremultipliedTexture = makeTexture(format)
        sourcePremultipliedView = sourcePremultipliedTexture.createView()
        sourceStraightTexture = makeTexture(format)
        sourceStraightView = sourceStraightTexture.createView()
        backdropStraightTexture = makeTexture(format)
        backdropStraightView = backdropStraightTexture.createView()
        resultPremultipliedTexture = makeTexture(format)
        resultPremultipliedView = resultPremultipliedTexture.createView()
        backdropMaskTexture = makeTexture(format)
        backdropMaskView = backdropMaskTexture.createView()
        mixedBackdropStraightTexture = makeTexture(format)
        mixedBackdropStraightView = mixedBackdropStraightTexture.createView()
        clipShapeTexture = makeTexture(format)
        clipShapeView = clipShapeTexture.createView()
        clipCumulativeTextureA = makeTexture(format)
        clipCumulativeViewA = clipCumulativeTextureA.createView()
        clipCumulativeTextureB = makeTexture(format)
        clipCumulativeViewB = clipCumulativeTextureB.createView()
        clippedSourceTexture = makeTexture(format)
        clippedSourceView = clippedSourceTexture.createView()
        combinedBackdropMaskTexture = makeTexture(format)
        combinedBackdropMaskView = combinedBackdropMaskTexture.createView()
        backgroundFilterResources = FilterLayerResources(
            device: device,
            width: width,
            height: height,
            format: format
        )
        sourceUniformBuffer = makeUniformBuffer()
        sourceOpacityUniformBuffer = makeUniformBuffer()
        backdropUniformBuffer = makeUniformBuffer()
        resultConversionUniformBuffer = makeUniformBuffer()
        displayUniformBuffer = makeUniformBuffer()
        transformedDisplayUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(MemoryLayout<TexturedUniforms>.stride),
            usage: [.uniform, .copyDst]
        ))
        backdropMaskUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: caRendererAlignedUniformSize,
            usage: [.uniform, .copyDst]
        ))
        backdropMaskVertexBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(6 * MemoryLayout<CARendererVertex>.stride),
            usage: [.vertex, .copyDst]
        ))
    }

    func clipUniformBuffer(at index: Int, device: GPUDevice) -> GPUBuffer {
        while clipUniformBuffers.count <= index {
            clipUniformBuffers.append(device.createBuffer(descriptor: GPUBufferDescriptor(
                size: caRendererAlignedUniformSize,
                usage: [.uniform, .copyDst]
            )))
        }
        return clipUniformBuffers[index]
    }

    func contentMaskResources(
        at index: Int,
        device: GPUDevice,
        width: Int,
        height: Int,
        format: GPUTextureFormat
    ) -> FilterLayerResources {
        while contentMaskFilterResources.count <= index {
            contentMaskFilterResources.append(FilterLayerResources(
                device: device,
                width: width,
                height: height,
                format: format
            ))
        }
        return contentMaskFilterResources[index]
    }

    func destroy() {
        backdropTexture.destroy()
        sourcePremultipliedTexture.destroy()
        sourceStraightTexture.destroy()
        backdropStraightTexture.destroy()
        resultPremultipliedTexture.destroy()
        backdropMaskTexture.destroy()
        mixedBackdropStraightTexture.destroy()
        clipShapeTexture.destroy()
        clipCumulativeTextureA.destroy()
        clipCumulativeTextureB.destroy()
        clippedSourceTexture.destroy()
        combinedBackdropMaskTexture.destroy()
        backgroundFilterResources.destroy()
        sourceUniformBuffer.destroy()
        sourceOpacityUniformBuffer.destroy()
        backdropUniformBuffer.destroy()
        resultConversionUniformBuffer.destroy()
        displayUniformBuffer.destroy()
        transformedDisplayUniformBuffer.destroy()
        backdropMaskUniformBuffer.destroy()
        backdropMaskVertexBuffer.destroy()
        for buffer in clipUniformBuffers {
            buffer.destroy()
        }
        clipUniformBuffers.removeAll(keepingCapacity: false)
        for resources in contentMaskFilterResources {
            resources.destroy()
        }
        contentMaskFilterResources.removeAll(keepingCapacity: false)
    }
}

private struct PrerenderedComposition {
    let resources: CompositionLayerResources
    let outputTexture: GPUTexture
    let outputView: GPUTextureView
    let samplingModelMatrix: Matrix4x4
}

private enum LayerFilterStage {
    case renderer(CAFilterOperation)
    case coreImage(CIFilter)
}

private enum FilterTextureAlphaMode: Equatable {
    case premultiplied
    case straight
}

private struct ShadowCaptureState: Equatable {
    let matrixColumns: [SIMD4<Float>]
    let layerSize: SIMD2<Float>
    let cornerRadius: Float
    let cornerRadii: SIMD4<Float>
    let blurRadius: Float
    let detachedMaskRevisionHash: Int

    init(
        matrix: Matrix4x4,
        layerSize: SIMD2<Float>,
        cornerRadius: Float,
        cornerRadii: SIMD4<Float>,
        blurRadius: Float,
        detachedMaskRevisionHash: Int
    ) {
        matrixColumns = [
            matrix.columns.0,
            matrix.columns.1,
            matrix.columns.2,
            matrix.columns.3,
        ]
        self.layerSize = layerSize
        self.cornerRadius = cornerRadius
        self.cornerRadii = cornerRadii
        self.blurRadius = blurRadius
        self.detachedMaskRevisionHash = detachedMaskRevisionHash
    }
}

/// GPU resources owned by one shadow-producing layer.
private final class ShadowLayerResources {
    let maskTexture: GPUTexture
    let maskView: GPUTextureView
    let intermediateTexture: GPUTexture
    let intermediateView: GPUTextureView
    let blurUniformBuffer: GPUBuffer
    let maskUniformBuffer: GPUBuffer
    let compositeUniformBuffer: GPUBuffer
    private(set) var maskVertexBuffer: GPUBuffer
    private(set) var maskVertexCapacity: Int
    var hasRenderedContent = false
    var captureState: ShadowCaptureState?

    init(
        device: GPUDevice,
        width: Int,
        height: Int,
        format: GPUTextureFormat
    ) {
        let textureDescriptor = GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: format,
            usage: [.renderAttachment, .textureBinding]
        )
        maskTexture = device.createTexture(descriptor: textureDescriptor)
        maskView = maskTexture.createView()
        intermediateTexture = device.createTexture(descriptor: textureDescriptor)
        intermediateView = intermediateTexture.createView()
        blurUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(MemoryLayout<BlurUniforms>.stride),
            usage: [.uniform, .copyDst]
        ))
        maskUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: caRendererAlignedUniformSize,
            usage: [.uniform, .copyDst]
        ))
        compositeUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(MemoryLayout<ShadowUniforms>.stride),
            usage: [.uniform, .copyDst]
        ))
        maskVertexCapacity = 6
        maskVertexBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(maskVertexCapacity * MemoryLayout<CARendererVertex>.stride),
            usage: [.vertex, .copyDst]
        ))
    }

    func ensureMaskVertexCapacity(_ count: Int, device: GPUDevice) -> GPUBuffer {
        guard count > maskVertexCapacity else { return maskVertexBuffer }
        var newCapacity = maskVertexCapacity
        while newCapacity < count {
            newCapacity *= 2
        }
        maskVertexBuffer.destroy()
        maskVertexBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(newCapacity * MemoryLayout<CARendererVertex>.stride),
            usage: [.vertex, .copyDst]
        ))
        maskVertexCapacity = newCapacity
        return maskVertexBuffer
    }

    func destroy() {
        maskTexture.destroy()
        intermediateTexture.destroy()
        blurUniformBuffer.destroy()
        maskUniformBuffer.destroy()
        compositeUniformBuffer.destroy()
        maskVertexBuffer.destroy()
    }
}

private struct PrerenderedShadow {
    let resources: ShadowLayerResources
}

private final class EmitterLayerState {
    weak var owner: CAEmitterLayer?
    var particles: [EmitterParticle] = []
    var birthRemainders: [EmitterBirthRemainderKey: Float] = [:]
    var randomSource: EmitterRandomSource
    var configuredSeed: UInt32
    var lastUpdateTime: CFTimeInterval = 0
    var simulationTime: CFTimeInterval = 0
    var lastUpdatedFrame: UInt64 = 0
    var nextBirthSequence: UInt64 = 0
    var lastRenderedBirthSequences: [UInt64] = []
    var lastRenderUsedAdditiveBlending = false

    init(owner: CAEmitterLayer, seed: UInt32) {
        self.owner = owner
        randomSource = EmitterRandomSource(seed: seed)
        configuredSeed = seed
    }
}

/// A renderer that uses WebGPU to render layer trees in WASM/Web environments.
///
/// This is the primary renderer for OpenCoreAnimation in production.
/// It conforms to both `CARenderer` (public API) and `CARendererDelegate` (internal).
public final class CAWebGPURenderer: CARenderer, CARendererDelegate {

    // MARK: - Constants

    /// Maximum number of layers that can be rendered per frame.
    private static let maxLayers = 1024

    /// Size of aligned uniform data per layer.
    private static var alignedUniformSize: UInt64 {
        caRendererAlignedUniformSize
    }

    // MARK: - Properties

    /// The WebGPU device.
    private var device: GPUDevice?

    /// Retains the JavaScript callback used to surface uncaptured WebGPU failures.
    private var uncapturedGPUErrorHandler: JSClosure?

    /// First WebGPU failure that escaped an explicit error scope.
    @_spi(RendererDiagnostics)
    public private(set) var firstUncapturedGPUError: String?

    /// The canvas context.
    private var context: GPUCanvasContext?

    /// The render pipeline.
    private var pipeline: GPURenderPipeline?

    /// The vertex buffer pool (triple buffered).
    private var vertexBufferPool: VertexBufferPool?

    /// The uniform buffer pool (triple buffered).
    private var uniformBufferPool: UniformBufferPool?

    /// The vertex buffer (large enough for all layers).
    /// - Note: This is a computed property that returns the current buffer from the pool.
    private var vertexBuffer: GPUBuffer? {
        return vertexBufferPool?.currentBuffer
    }

    /// The uniform buffer (large enough for all layers with alignment).
    /// - Note: This is a computed property that returns the current buffer from the pool.
    private var uniformBuffer: GPUBuffer? {
        return uniformBufferPool?.currentBuffer
    }

    /// The bind group layout for dynamic uniform access.
    private var bindGroupLayout: GPUBindGroupLayout?

    /// The bind group.
    /// - Note: This is a computed property that returns the current bind group from the pool.
    private var bindGroup: GPUBindGroup? {
        return uniformBufferPool?.currentBindGroup
    }

    /// The depth texture.
    private var depthTexture: GPUTexture?

    /// Cached view for `depthTexture`.
    ///
    /// `createView()` crosses the JS bridge and allocates a fresh `GPUTextureView`
    /// every call. The depth texture changes only on `resize`, so we cache the
    /// view alongside it and invalidate together with the texture.
    private var depthTextureView: GPUTextureView?

    /// The current size.
    public private(set) var size: CGSize = CGSize(width: 0, height: 0)

    /// Number of transition source textures captured during this renderer's lifetime.
    @_spi(RendererDiagnostics)
    public private(set) var transitionSourceCaptureCount: Int = 0

    /// Number of transition target textures captured during this renderer's lifetime.
    @_spi(RendererDiagnostics)
    public private(set) var transitionTargetCaptureCount: Int = 0

    /// Number of Core Image transition dispatches encoded by this renderer.
    @_spi(RendererDiagnostics)
    public private(set) var transitionFilterDispatchCount: Int = 0

    /// Number of filter transitions rejected before GPU dispatch.
    @_spi(RendererDiagnostics)
    public private(set) var transitionFilterFailureCount: Int = 0

    /// Number of unknown built-in transition types rejected before capture.
    @_spi(RendererDiagnostics)
    public private(set) var transitionRenderFailureCount: Int = 0

    /// Number of requested layer filters rejected before GPU dispatch.
    @_spi(RendererDiagnostics)
    public private(set) var layerFilterFailureCount: Int = 0

    /// Number of frozen source and target textures currently retained by the renderer.
    @_spi(RendererDiagnostics)
    public var activeTransitionTextureCount: Int {
        transitionCaptures.values.reduce(into: 0) { count, capture in
            count += capture.filterExecution == nil ? 2 : 3
        }
    }

    /// Number of filtered-layer texture sets retained by the renderer.
    @_spi(RendererDiagnostics)
    public var activeFilterResourceCount: Int {
        filterLayerResources.count
    }

    /// Number of shadow texture sets retained by the renderer.
    @_spi(RendererDiagnostics)
    public var activeShadowResourceCount: Int {
        shadowLayerResources.count
    }

    /// Number of emitter-layer simulations retained after the latest frame.
    @_spi(RendererDiagnostics)
    public var activeEmitterStateCount: Int {
        emitterLayerStates.count
    }

    /// Number of live particles owned by one emitter layer.
    @_spi(RendererDiagnostics)
    public func activeParticleCount(for layer: CAEmitterLayer) -> Int {
        guard let state = emitterLayerStates[ObjectIdentifier(layer)], state.owner === layer else {
            return 0
        }
        return state.particles.count
    }

    /// Current local-space particle positions for one emitter layer.
    @_spi(RendererDiagnostics)
    public func activeParticlePositions(for layer: CAEmitterLayer) -> [SIMD3<Float>] {
        guard let state = emitterLayerStates[ObjectIdentifier(layer)], state.owner === layer else {
            return []
        }
        return state.particles.map(\.position)
    }

    /// Current local-space particle velocities for one emitter layer.
    @_spi(RendererDiagnostics)
    public func activeParticleVelocities(for layer: CAEmitterLayer) -> [SIMD3<Float>] {
        guard let state = emitterLayerStates[ObjectIdentifier(layer)], state.owner === layer else {
            return []
        }
        return state.particles.map(\.velocity)
    }

    /// Generation number for each live particle (zero for root cells).
    @_spi(RendererDiagnostics)
    public func activeParticleGenerations(for layer: CAEmitterLayer) -> [Int] {
        guard let state = emitterLayerStates[ObjectIdentifier(layer)], state.owner === layer else {
            return []
        }
        return state.particles.map(\.generation)
    }

    /// Current color for each live particle after inheritance and simulation.
    @_spi(RendererDiagnostics)
    public func activeParticleColors(for layer: CAEmitterLayer) -> [SIMD4<Float>] {
        guard let state = emitterLayerStates[ObjectIdentifier(layer)], state.owner === layer else {
            return []
        }
        return state.particles.map(\.color)
    }

    /// Current scale for each live particle after inheritance and simulation.
    @_spi(RendererDiagnostics)
    public func activeParticleScales(for layer: CAEmitterLayer) -> [Float] {
        guard let state = emitterLayerStates[ObjectIdentifier(layer)], state.owner === layer else {
            return []
        }
        return state.particles.map(\.scale)
    }

    /// Birth sequences in the order submitted to WebGPU during the latest frame.
    @_spi(RendererDiagnostics)
    public func lastRenderedParticleSequences(for layer: CAEmitterLayer) -> [UInt64] {
        guard let state = emitterLayerStates[ObjectIdentifier(layer)], state.owner === layer else {
            return []
        }
        return state.lastRenderedBirthSequences
    }

    /// Whether the latest particle submission used source-additive blending.
    @_spi(RendererDiagnostics)
    public func lastEmitterRenderUsedAdditiveBlending(for layer: CAEmitterLayer) -> Bool {
        guard let state = emitterLayerStates[ObjectIdentifier(layer)], state.owner === layer else {
            return false
        }
        return state.lastRenderUsedAdditiveBlending
    }

    /// Number of particles rejected because their emitter configuration was unsupported or non-finite.
    @_spi(RendererDiagnostics)
    public private(set) var emitterSpawnFailureCount: Int = 0

    /// Number of emitter frames rejected because their render mode was unsupported.
    @_spi(RendererDiagnostics)
    public private(set) var emitterRenderFailureCount: Int = 0

    /// Number of visible shadows that could not complete the GPU render path.
    @_spi(RendererDiagnostics)
    public private(set) var shadowRenderFailureCount: Int = 0

    /// Number of rasterization captures rejected because their extent or scale was invalid.
    @_spi(RendererDiagnostics)
    public private(set) var rasterizationFailureCount: Int = 0

    /// Number of requested delegate backing-store draws that could not produce an image.
    @_spi(RendererDiagnostics)
    public private(set) var delegateDrawFailureCount: Int = 0

    /// Number of live delegate-generated backing stores retained by the renderer.
    @_spi(RendererDiagnostics)
    public var activeDelegateBackingStoreCount: Int {
        delegateBackingStores.count
    }

    /// The preferred texture format.
    private var preferredFormat: GPUTextureFormat = .bgra8unorm

    /// The swap-chain texture most recently submitted for presentation.
    private var lastRenderedTexture: GPUTexture?

    /// Software-rasterized backing stores for ordinary CALayer delegates.
    private var delegateBackingStores: [ObjectIdentifier: CGImage] = [:]

    /// The canvas element (JavaScript object).
    private let canvas: JSObject

    /// Current layer index during rendering (used for uniform buffer indexing).
    private var currentLayerIndex: Int = 0

    /// Current vertex buffer offset during rendering (dynamic allocation).
    private var currentVertexOffset: UInt64 = 0

    /// Number of layers dropped due to buffer capacity limits in the current frame.
    private var droppedLayerCount: Int = 0

    /// Root layer reference during rendering (to skip its backgroundColor rendering).
    /// SKScene's backgroundColor is rendered via clear color instead.
    private weak var currentRootLayer: CALayer?

    /// Maximum vertex buffer size (4MB to accommodate complex scenes).
    private static let maxVertexBufferSize: UInt64 = 4 * 1024 * 1024

    // MARK: - Dynamic Vertex Allocation

    /// Allocates space for vertices in the vertex buffer.
    ///
    /// - Parameter count: The number of vertices to allocate.
    /// - Returns: A tuple of (vertexOffset, uniformIndex) if allocation succeeds, nil if buffer is full.
    private func allocateVertices(count: Int) -> (vertexOffset: UInt64, uniformIndex: Int)? {
        let requiredSize = UInt64(count * MemoryLayout<CARendererVertex>.stride)

        // Check vertex buffer capacity
        guard currentVertexOffset + requiredSize <= Self.maxVertexBufferSize else {
            droppedLayerCount += 1
            return nil
        }

        // Check uniform count limit
        guard currentLayerIndex < Self.maxLayers else {
            droppedLayerCount += 1
            return nil
        }

        let result = (vertexOffset: currentVertexOffset, uniformIndex: currentLayerIndex)

        // Advance pointers
        currentVertexOffset += requiredSize
        currentLayerIndex += 1

        return result
    }

    // MARK: - Clipping (masksToBounds)

    /// Represents a clip rectangle in screen coordinates.
    private struct ClipRect {
        var x: Int32
        var y: Int32
        var width: UInt32
        var height: UInt32

        /// Returns the intersection of two clip rects.
        func intersection(with other: ClipRect) -> ClipRect {
            let x1 = max(self.x, other.x)
            let y1 = max(self.y, other.y)
            let x2 = min(self.x + Int32(self.width), other.x + Int32(other.width))
            let y2 = min(self.y + Int32(self.height), other.y + Int32(other.height))

            let w = max(0, x2 - x1)
            let h = max(0, y2 - y1)

            return ClipRect(x: x1, y: y1, width: UInt32(w), height: UInt32(h))
        }
    }

    /// Stack of clip rectangles for nested masksToBounds.
    private var clipRectStack: [ClipRect] = []

    // MARK: - Opacity Propagation

    /// Stack of effective opacity values for parent-to-child opacity propagation.
    /// CoreAnimation multiplies opacity through the tree:
    /// effectiveOpacity = parent.effectiveOpacity * layer.opacity
    private var opacityStack: [Float] = []

    /// Multiplicative color inherited from enclosing replicator instances.
    private var replicatorColorStack: [SIMD4<Float>] = []

    /// Animation-time offset inherited from enclosing replicator instances.
    private var replicatorTimeOffsetStack: [CFTimeInterval] = []

    /// Stable instance path used to separate per-instance offscreen resources.
    private var replicatorInstancePath: [ReplicatorInstancePathComponent] = []

    /// Tile delegate calls are deferred until the next frame boundary so layer-tree
    /// rendering never re-enters CGContext setup from inside an active render pass.
    private var pendingTileDraws: [PendingTileDraw] = []

    /// Prevents the live side of a transition from recursively re-entering its compositor.
    private var transitionSuppressedLayer: CALayer?

    /// Overrides viewport-derived calculations while a layer subtree is captured offscreen.
    private var renderTargetSizeOverride: CGSize?

    private var currentRenderTargetSize: CGSize {
        renderTargetSizeOverride ?? size
    }

    /// Returns the current effective opacity from the stack.
    /// Defaults to 1.0 when empty (root level).
    private var currentEffectiveOpacity: Float {
        opacityStack.last ?? 1.0
    }

    private var currentReplicatorColor: SIMD4<Float> {
        replicatorColorStack.last ?? SIMD4(1, 1, 1, 1)
    }

    private var currentReplicatorTimeOffset: CFTimeInterval {
        replicatorTimeOffsetStack.last ?? 0
    }

    private func renderKey(for layer: CALayer) -> LayerRenderKey {
        LayerRenderKey(
            layer: ObjectIdentifier(layer),
            replicatorPath: replicatorInstancePath
        )
    }

    private func renderPresentation(for layer: CALayer) -> CALayer {
        let timeOffset = currentReplicatorTimeOffset
        return timeOffset == 0
            ? layer._renderTimePresentation()
            : layer.presentationAtTimeOffset(timeOffset)
    }

    private func replicatedColor(_ color: SIMD4<Float>) -> SIMD4<Float> {
        color * currentReplicatorColor
    }

    private func requiresGroupOpacity(_ presentationLayer: CALayer) -> Bool {
        presentationLayer.allowsGroupOpacity && presentationLayer.opacity < 1
    }

    private func requiresTransformFlattening(
        modelLayer: CALayer,
        presentationLayer: CALayer
    ) -> Bool {
        modelLayer.sublayers?.isEmpty == false
            || presentationLayer.filters?.isEmpty == false
            || requiresGroupOpacity(presentationLayer)
            || presentationLayer.mask != nil
            || presentationLayer.masksToBounds
            || hasVisibleShadow(presentationLayer)
    }

    private func requiresEffectFlattening(_ presentationLayer: CALayer) -> Bool {
        presentationLayer.filters?.isEmpty == false
            || requiresGroupOpacity(presentationLayer)
            || hasVisibleShadow(presentationLayer)
    }

    private func hasVisibleShadow(_ presentationLayer: CALayer) -> Bool {
        presentationLayer.shadowOpacity > 0 && presentationLayer.shadowColor != nil
    }

    private func orderedSublayers(for layer: CALayer) -> [CALayer] {
        guard currentReplicatorTimeOffset != 0 else {
            return layer.sortedSublayers()
        }
        return (layer.sublayers ?? []).enumerated().sorted { lhs, rhs in
            let lhsZ = renderPresentation(for: lhs.element).zPosition
            let rhsZ = renderPresentation(for: rhs.element).zPosition
            return lhsZ == rhsZ ? lhs.offset < rhs.offset : lhsZ < rhsZ
        }.map(\.element)
    }

    /// Orders transform-layer children by their projected center depth so blended
    /// pixels see the already-rendered farther surface before depth-tested nearer ones.
    private func depthOrderedSublayers(
        for layer: CALayer,
        parentMatrix: Matrix4x4
    ) -> [CALayer] {
        (layer.sublayers ?? []).enumerated().sorted { lhs, rhs in
            let lhsDepth = projectedCenterDepth(of: lhs.element, parentMatrix: parentMatrix)
            let rhsDepth = projectedCenterDepth(of: rhs.element, parentMatrix: parentMatrix)
            return lhsDepth == rhsDepth ? lhs.offset < rhs.offset : lhsDepth < rhsDepth
        }.map(\.element)
    }

    private func projectedCenterDepth(
        of layer: CALayer,
        parentMatrix: Matrix4x4
    ) -> Float {
        let presentation = renderPresentation(for: layer)
        let matrix = presentation.modelMatrix(parentMatrix: parentMatrix)
        let center = SIMD4<Float>(
            Float(presentation.bounds.width * 0.5),
            Float(presentation.bounds.height * 0.5),
            0,
            1
        )
        let projected = matrix * center
        guard projected.z.isFinite, projected.w.isFinite, projected.w != 0 else {
            return 0
        }
        return projected.z / projected.w
    }

    private func forEachPrepassSublayer(
        of layer: CALayer,
        presentationLayer: CALayer,
        modelMatrix: Matrix4x4,
        _ body: (CALayer, Matrix4x4) -> Void
    ) {
        guard layer.sublayers?.isEmpty == false else { return }
        let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)
        guard let replicatorPresentation = presentationLayer as? CAReplicatorLayer,
              let replicatorModel = layer as? CAReplicatorLayer else {
            for sublayer in orderedSublayers(for: layer) {
                body(sublayer, sublayerMatrix)
            }
            return
        }

        let instanceCount = max(0, replicatorPresentation.instanceCount)
        let baseColor = replicatorPresentation.instanceColor.map(rgbaComponents)
            ?? SIMD4<Float>(repeating: 1)
        var cumulativeTransform = CATransform3DIdentity
        for instanceIndex in 0..<instanceCount {
            let inheritedTimeOffset = currentReplicatorTimeOffset
            let inheritedColor = currentReplicatorColor
            let instanceColor = SIMD4<Float>(
                clamp(baseColor.x + Float(instanceIndex) * replicatorPresentation.instanceRedOffset, 0, 1),
                clamp(baseColor.y + Float(instanceIndex) * replicatorPresentation.instanceGreenOffset, 0, 1),
                clamp(baseColor.z + Float(instanceIndex) * replicatorPresentation.instanceBlueOffset, 0, 1),
                clamp(baseColor.w + Float(instanceIndex) * replicatorPresentation.instanceAlphaOffset, 0, 1)
            )
            replicatorColorStack.append(inheritedColor * instanceColor)
            replicatorTimeOffsetStack.append(
                inheritedTimeOffset
                    + CFTimeInterval(instanceIndex) * replicatorPresentation.instanceDelay
            )
            replicatorInstancePath.append(ReplicatorInstancePathComponent(
                replicator: ObjectIdentifier(replicatorModel),
                instanceIndex: instanceIndex
            ))

            let instanceMatrix = sublayerMatrix * cumulativeTransform.matrix4x4
            for sublayer in orderedSublayers(for: layer) {
                body(sublayer, instanceMatrix)
            }

            _ = replicatorInstancePath.popLast()
            _ = replicatorTimeOffsetStack.popLast()
            _ = replicatorColorStack.popLast()
            cumulativeTransform = CATransform3DConcat(
                cumulativeTransform,
                replicatorPresentation.instanceTransform
            )
        }
    }

    private func withPrepassContext<T>(
        _ target: LayerPrepassTarget,
        _ body: () -> T
    ) -> T {
        let savedColorStack = replicatorColorStack
        let savedTimeOffsetStack = replicatorTimeOffsetStack
        let savedInstancePath = replicatorInstancePath

        replicatorColorStack.removeAll(keepingCapacity: true)
        replicatorTimeOffsetStack = target.timeOffset == 0 ? [] : [target.timeOffset]
        replicatorInstancePath = target.renderKey.replicatorPath
        defer {
            replicatorColorStack = savedColorStack
            replicatorTimeOffsetStack = savedTimeOffsetStack
            replicatorInstancePath = savedInstancePath
        }
        return body()
    }

    /// The full viewport clip rect.
    private var viewportClipRect: ClipRect {
        let renderSize = currentRenderTargetSize
        return ClipRect(x: 0, y: 0, width: UInt32(renderSize.width), height: UInt32(renderSize.height))
    }

    /// Calculates the screen-space clip rect for a layer.
    ///
    /// Transforms the layer's bounds corners through the model matrix and
    /// returns an axis-aligned bounding box in screen coordinates.
    private func calculateClipRect(layer: CALayer, modelMatrix: Matrix4x4) -> ClipRect {
        let bounds = layer.bounds

        // Transform the four corners of the content area.
        // Rendering uses (0,0)-(bounds.width, bounds.height) via scaleMatrix.
        // bounds.origin only affects sublayer positioning (sublayerMatrix), not
        // the layer's own rendered area, so we use (0,0) as the origin.
        let corners: [SIMD4<Float>] = [
            SIMD4(0, 0, 0, 1),
            SIMD4(Float(bounds.width), 0, 0, 1),
            SIMD4(0, Float(bounds.height), 0, 1),
            SIMD4(Float(bounds.width), Float(bounds.height), 0, 1)
        ]

        var minX: Float = .greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude
        var maxX: Float = -.greatestFiniteMagnitude
        var maxY: Float = -.greatestFiniteMagnitude

        // Viewport dimensions for NDC-to-screen conversion
        let renderSize = currentRenderTargetSize
        let viewWidth = Float(renderSize.width)
        let viewHeight = Float(renderSize.height)

        for corner in corners {
            // Apply model matrix (which includes projection)
            let transformed = modelMatrix * corner
            // Perspective divide (w should be 1 for orthographic, but handle it anyway)
            let w = transformed.w != 0 ? transformed.w : 1
            let ndcX = transformed.x / w
            let ndcY = transformed.y / w

            // Convert NDC [-1,1] to screen coordinates [0, viewWidth/viewHeight]
            let screenX = (ndcX + 1) * 0.5 * viewWidth
            let screenY = (ndcY + 1) * 0.5 * viewHeight

            minX = min(minX, screenX)
            minY = min(minY, screenY)
            maxX = max(maxX, screenX)
            maxY = max(maxY, screenY)
        }

        // Clamp to viewport bounds
        let clampedMinX = max(0, minX)
        let clampedMinY = max(0, minY)
        let clampedMaxX = min(viewWidth, maxX)
        let clampedMaxY = min(viewHeight, maxY)

        let width = max(0, clampedMaxX - clampedMinX)
        let height = max(0, clampedMaxY - clampedMinY)

        return ClipRect(
            x: Int32(clampedMinX),
            y: Int32(clampedMinY),
            width: UInt32(width),
            height: UInt32(height)
        )
    }

    /// Returns the current effective clip rect (intersection of all stacked clip rects).
    private var currentClipRect: ClipRect {
        clipRectStack.last ?? viewportClipRect
    }

    /// Applies the current clip rect to the render pass.
    private func applyScissorRect(_ renderPass: GPURenderPassEncoder) {
        let clip = currentClipRect
        let x = UInt32(max(0, clip.x))
        let w = clip.width
        let h = clip.height
        // WebGPU scissor rect uses Y-down convention (y=0 at top of framebuffer).
        // Internal clip rects use Y-up convention (y=0 at bottom, matching SpriteKit).
        // Flip Y here so the conversion is centralized in one place.
        let viewH = Int32(currentRenderTargetSize.height)
        let flippedY = UInt32(max(0, viewH - clip.y - Int32(h)))
        renderPass.setScissorRect(x: x, y: flippedY, width: w, height: h)
    }

    // MARK: - Texture Rendering

    /// The textured render pipeline.
    private var texturedPipeline: GPURenderPipeline?

    /// Textured pipeline used while traversing a true-3D transform hierarchy.
    private var texturedDepthPipeline: GPURenderPipeline?

    /// Composites premultiplied-alpha offscreen captures.
    private var premultipliedTexturedPipeline: GPURenderPipeline?

    /// Premultiplied textured pipeline with true-3D depth testing.
    private var premultipliedTexturedDepthPipeline: GPURenderPipeline?

    /// Stencil-aware variant for premultiplied-alpha captures.
    private var premultipliedTexturedStencilPipeline: GPURenderPipeline?

    /// Stencil-aware premultiplied textured pipeline with true-3D depth testing.
    private var premultipliedTexturedDepthStencilPipeline: GPURenderPipeline?

    /// Interpolates two premultiplied transition captures in one fragment pass.
    private var transitionFadePipeline: GPURenderPipeline?

    /// True-3D variant of the premultiplied transition fade pipeline.
    private var transitionFadeDepthPipeline: GPURenderPipeline?

    /// Stencil-aware transition fade pipeline.
    private var transitionFadeStencilPipeline: GPURenderPipeline?

    /// Stencil-aware transition fade pipeline with true-3D depth testing.
    private var transitionFadeDepthStencilPipeline: GPURenderPipeline?

    /// Bindings for fade uniforms and the frozen source/target textures.
    private var transitionFadeBindGroupLayout: GPUBindGroupLayout?

    /// The bind group layout for textured rendering.
    private var texturedBindGroupLayout: GPUBindGroupLayout?

    /// The texture sampler.
    private var textureSampler: GPUSampler?

    /// Samplers covering every supported CAEmitterCell magnification/minification pair.
    private var emitterTextureSamplers: [EmitterTextureSampling: GPUSampler] = [:]

    /// Source-additive textured particle pipeline without stencil testing.
    private var emitterTexturedAdditivePipeline: GPURenderPipeline?

    /// Additive particle pipeline with true-3D depth testing.
    private var emitterTexturedAdditiveDepthPipeline: GPURenderPipeline?

    /// Source-additive textured particle pipeline constrained by the active stencil mask.
    private var emitterTexturedAdditiveStencilPipeline: GPURenderPipeline?

    /// Stencil-aware additive particle pipeline with true-3D depth testing.
    private var emitterTexturedAdditiveDepthStencilPipeline: GPURenderPipeline?

    /// Texture manager with LRU cache for efficient texture memory management.
    private var textureManager: GPUTextureManager?

    /// Geometry cache for tessellated path data.
    private var geometryCache: GeometryCache?

    // MARK: - Shadow Rendering

    /// Shadow blur pipeline (horizontal pass).
    private var shadowBlurHorizontalPipeline: GPURenderPipeline?

    /// Shadow blur pipeline (vertical pass).
    private var shadowBlurVerticalPipeline: GPURenderPipeline?

    /// Horizontal blur pipeline targeting the renderer's preferred color format.
    private var filterBlurHorizontalPipeline: GPURenderPipeline?

    /// Vertical blur pipeline targeting the renderer's preferred color format.
    private var filterBlurVerticalPipeline: GPURenderPipeline?

    /// Shadow composite pipeline.
    private var shadowCompositePipeline: GPURenderPipeline?

    /// Combines a local shadow mask and local premultiplied layer capture.
    private var rasterizedShadowCompositePipeline: GPURenderPipeline?

    /// Shadow mask pipeline.
    private var shadowMaskPipeline: GPURenderPipeline?

    /// Shadow bind group layout.
    private var shadowBindGroupLayout: GPUBindGroupLayout?

    /// Full-screen quad sampler for blur.
    private var blurSampler: GPUSampler?

    /// Persistent viewport-sized resources keyed by layer identity and replicator path.
    private var shadowLayerResources: [LayerRenderKey: ShadowLayerResources] = [:]

    /// Shadow outputs available to the main pass this frame.
    private var prerenderedShadows: [LayerRenderKey: PrerenderedShadow] = [:]

    // MARK: - Filter Rendering (CAFilter)

    /// Filter composite pipeline for blending filtered result with optional tint.
    private var filterCompositePipeline: GPURenderPipeline?

    /// Replaces the current framebuffer with an already-composited backdrop snapshot.
    private var filterReplacementPipeline: GPURenderPipeline?

    /// Replaces a true-3D layer plane while preserving its projected depth.
    private var transformedCompositionPipeline: GPURenderPipeline?

    /// Stencil-aware variant of the true-3D composition replacement pipeline.
    private var transformedCompositionStencilPipeline: GPURenderPipeline?

    /// Plane replacement used inside a flat rasterization capture.
    private var capturedCompositionPipeline: GPURenderPipeline?

    /// Stencil-aware plane replacement used inside a flat rasterization capture.
    private var capturedCompositionStencilPipeline: GPURenderPipeline?

    /// Restricts filtered backdrop pixels to the transformed layer shape.
    private var backdropFilterMixPipeline: GPURenderPipeline?

    /// Multiplies two full-viewport alpha coverage masks.
    private var compositionMaskIntersectPipeline: GPURenderPipeline?

    /// Applies a full-viewport alpha coverage mask to premultiplied color.
    private var compositionMaskApplyPipeline: GPURenderPipeline?

    /// Stencil-aware filter composite pipeline.
    private var filterCompositeStencilPipeline: GPURenderPipeline?

    /// Executes color-adjustment operations into preferred-format offscreen textures.
    private var filterOperationPipeline: GPURenderPipeline?

    /// Persistent viewport-sized resources keyed by layer identity and replicator path.
    private var filterLayerResources: [LayerRenderKey: FilterLayerResources] = [:]

    /// Filter outputs available to capture passes and the main pass this frame.
    private var prerenderedFilters: [LayerRenderKey: PrerenderedFilter] = [:]

    /// Executes Core Image layer filters on this renderer's GPU device.
    private var layerFilterProcessor: CIWebGPUFilterProcessor?

    /// Executions encoded in the current frame and retained through the next submission.
    private var activeLayerFilterExecutions: [CIWebGPUFilterExecution] = []

    /// Executions from the preceding frame, released after one complete frame of GPU lifetime.
    private var retiringLayerFilterExecutions: [CIWebGPUFilterExecution] = []

    /// Requested layer-filter paths whose configuration is currently not executable.
    private var failedLayerFilterKeys: Set<LayerRenderKey> = []

    // MARK: - Backdrop Composition

    private var compositionLayerResources: [LayerRenderKey: CompositionLayerResources] = [:]
    private var prerenderedCompositions: [LayerRenderKey: PrerenderedComposition] = [:]
    private var activeCompositionExecutions: [CIWebGPUFilterExecution] = []
    private var retiringCompositionExecutions: [CIWebGPUFilterExecution] = []
    private var compositionCaptureStopKey: LayerRenderKey?
    private var compositionCaptureDidReachStop = false
    private var compositionCapturePassThroughKeys: Set<LayerRenderKey> = []
    private var deferredCompositionRasterizationKeys: Set<LayerRenderKey> = []
    private var capturesOnlyDeferredCompositionRasterizations = false
    private var failedCompositionKeys: Set<LayerRenderKey> = []

    /// Number of requested backdrop compositions rejected before GPU dispatch.
    @_spi(RendererDiagnostics)
    public private(set) var compositionFilterFailureCount: Int = 0

    @_spi(RendererDiagnostics)
    public var activeCompositionResourceCount: Int {
        compositionLayerResources.count
    }

    /// The root layer currently being rendered into the offscreen filter source texture.
    /// While set, filter compositing is suppressed so the subtree can be captured raw.
    private weak var filterPrerenderRootLayer: CALayer?

    /// The root layer whose own content mask is deferred until its captured subtree and
    /// filter chain have been combined. Descendant masks remain active during capture.
    private weak var contentMaskCaptureSuppressedRootLayer: CALayer?

    /// The root layer whose rendered alpha is being captured as a shadow silhouette.
    private weak var shadowCaptureRootLayer: CALayer?

    /// Suppresses shadow draws while mask-dependent filter resources are prepared
    /// ahead of the shadow silhouette pass.
    private var suppressShadowRendering = false

    // MARK: - Rasterization Cache (R3.2 / R3.4)

    /// LRU + byte-budget cache of captured `shouldRasterize` subtrees.
    /// Allocated on first `resize` once the viewport size is known so the
    /// budget can be sized to `viewport × 4 × 2.5` per WWDC 2014 #419.
    private var rasterizationCache: RasterizationCache<GPUTexture>?

    /// Frozen source/target pairs keyed by the immutable transition source snapshot.
    private var transitionCaptures: [ObjectIdentifier: TransitionCapturePair] = [:]

    /// Executes Core Image transition shaders on this renderer's GPU device.
    private var transitionFilterProcessor: CIWebGPUTransitionProcessor?

    /// Source snapshots referenced by an active transition during the current frame.
    private var activeTransitionSourceIDs: Set<ObjectIdentifier> = []

    /// Active transitions rejected because their filter cannot execute.
    private var failedTransitionSourceIDs: Set<ObjectIdentifier> = []

    /// Capture-only depth textures retained until their command buffer has been submitted.
    private var transientCaptureDepthTextures: [GPUTexture] = []

    /// Raw captures and filter ping-pong resources retained until their command buffer
    /// has been submitted. The final filtered texture is owned by rasterizationCache.
    private var transientRasterizationTextures: [GPUTexture] = []
    private var transientRasterizationFilterResources: [FilterLayerResources] = []
    private var transientRasterizationShadowResources: [ShadowLayerResources] = []

    /// Per-frame scratch storage: layers whose subtree has been captured
    /// (or had a fresh cache hit) this frame, mapped to the texture used
    /// for compositing. Populated by `prerenderRasterizedLayers`,
    /// consumed by `renderLayer`, cleared after submit.
    private var prerasterizedTextures: [LayerRenderKey: PrerasterizedTexture] = [:]

    @_spi(RendererDiagnostics)
    public private(set) var transformFlatteningCaptureCount: Int = 0

    /// Number of explicit `shouldRasterize` subtree captures encoded in the latest frame.
    @_spi(RendererDiagnostics)
    public private(set) var explicitRasterizationCaptureCount: Int = 0

    @_spi(RendererDiagnostics)
    public private(set) var transformFlatteningCompositeCount: Int = 0

    /// Distinguishes the user-visible render pass from offscreen prefix captures.
    private var isRenderingMainPass = false


    /// Persistent cache of `GPUTextureView`s keyed by `TexturedCacheKey`.
    ///
    /// `gpuTexture.createView()` is a JS round-trip to allocate a fresh
    /// `GPUTextureView`. The view is invariant for a given texture, so we
    /// keep it across frames.
    ///
    /// **Identity contract**: keyed by either `.image(OID(cgImage))` for
    /// regular content layers (eviction wired to `GPUTextureManager.onEvict`)
    /// or `.rasterizedLayer(renderKey, purpose)` for rasterized subtree
    /// composites. Tagging the kind keeps the two
    /// namespaces disjoint — a raw `ObjectIdentifier` would let a freed
    /// layer's address be reused by a fresh `CGImage` (or vice versa) and
    /// return a stale view. See `TextureManager.swift` for the broader
    /// identity-ownership contract.
    private var texturedTextureViewCache: [TexturedCacheKey: GPUTextureView] = [:]

    /// Per-frame cache of textured-content bind groups keyed by
    /// `TexturedCacheKey`.
    ///
    /// The bind group binds (uniformBuffer, textureSampler, textureView).
    /// `uniformBuffer` rotates through the pool every frame, so the bind
    /// group cannot survive across frames — but it is constant within a
    /// single frame for any layer drawing the same texture. The wall
    /// shrapnel burst spawns ~24 sprites sharing one atlas in a single
    /// frame; without this cache we paid 24 `device.createBindGroup` JS
    /// round-trips on that frame alone, which empirically lined up with
    /// the 25 ms rAF jank measured in `_frame_drop_probe.spec.ts`.
    /// Cleared at the start of each `render(layer:)` call.
    ///
    /// **Identity contract**: same disjoint-namespace tagging as
    /// `texturedTextureViewCache` above.
    private var perFrameTexturedBindGroupCache: [TexturedCacheKey: GPUBindGroup] = [:]

    /// Persistent pool of JS `Float32Array`s keyed by element count.
    ///
    /// `JSTypedArray<Float32>(buffer:)` lowers to `swjs_create_typed_array`
    /// which calls `array.slice()` JS-side — every call allocates a fresh
    /// JS `ArrayBuffer`. At ~88 writeBuffer calls/frame this generated
    /// ~0.75 MB/sec of JS heap garbage; the resulting GC pauses jumped
    /// `dt` from 33 → 67 ms on 30 Hz baseline (Battery-Saver throttling).
    /// The pool reuses one `Float32Array` per distinct float count so
    /// per-frame allocation drops to zero after warm-up; the cost is
    /// `floatCount` indexed-property bridge calls per write rather than
    /// one `slice()` allocation, which V8 pipelines efficiently.
    /// Cleared on `invalidate()` so JS handles are released alongside
    /// the renderer's other GPU resources.
    private var float32StagingPool: [Int: JSObject] = [:]

    /// While set, the named layer is being rendered into its rasterization
    /// capture texture. The renderer must (a) skip its own re-composition
    /// at the matching cache-hit branch in `renderLayer`, and (b) treat
    /// the layer as "root" for opacity (clear-α 1.0 per R3.3, layer.opacity
    /// applied at composite, not at capture).
    private weak var rasterizePrerenderRootLayer: CALayer?

    // MARK: - Mask Rendering

    /// Pipeline for writing mask to stencil buffer (stencil-based approach).
    /// Writes to stencil where mask alpha > 0, no color output.
    private var stencilWritePipeline: GPURenderPipeline?

    /// Pipeline for writing rounded rectangle to stencil buffer.
    /// Uses stencilClipFragment shader which discards fragments outside the rounded rect,
    /// preventing stencil writes in the discarded regions.
    private var stencilWriteRoundedPipeline: GPURenderPipeline?

    /// Pipeline for rendering with stencil test.
    /// Only renders where stencil value matches.
    private var stencilTestPipeline: GPURenderPipeline?

    /// Solid-color stencil pipeline with true-3D depth testing.
    private var depthStencilTestPipeline: GPURenderPipeline?

    /// Solid-color pipeline with true-3D depth testing.
    private var depthPipeline: GPURenderPipeline?

    /// Writes the farthest depth without changing color or stencil.
    private var depthClearPipeline: GPURenderPipeline?

    /// Current stencil reference value for nested masks.
    private var currentStencilValue: UInt32 = 0

    /// Tracks mask nesting depth for stencil-based masking.
    /// When > 0, textured and composite pipelines must use stencil-aware variants.
    /// Supports nested masks (e.g., child B with mask inside parent A with mask).
    private var maskNestingDepth: Int = 0

    /// Number of active CATransformLayer ancestors in the current traversal.
    private var transformDepthNesting: Int = 0

    /// Stencil-aware textured pipeline (tests stencil buffer for mask).
    private var texturedStencilPipeline: GPURenderPipeline?

    /// Stencil-aware textured pipeline with true-3D depth testing.
    private var texturedDepthStencilPipeline: GPURenderPipeline?

    /// Opaque textured pipeline (no alpha blend). R3.5 — selected when
    /// `RasterizationDecisions.blendEnabled(for:)` returns false.
    private var texturedOpaquePipeline: GPURenderPipeline?

    /// Opaque textured pipeline with true-3D depth testing.
    private var texturedDepthOpaquePipeline: GPURenderPipeline?

    /// Opaque stencil-aware textured pipeline (R3.5 + mask).
    private var texturedStencilOpaquePipeline: GPURenderPipeline?

    /// Opaque stencil-aware textured pipeline with true-3D depth testing.
    private var texturedDepthStencilOpaquePipeline: GPURenderPipeline?

    /// Stencil-aware shadow composite pipeline (tests stencil buffer for mask).
    private var shadowCompositeStencilPipeline: GPURenderPipeline?

    // MARK: - Particle System (CAEmitterLayer)

    /// Maximum number of particles.
    private static let maxParticles = 10000

    /// Simulation state owned independently by each emitter layer.
    private var emitterLayerStates: [ObjectIdentifier: EmitterLayerState] = [:]

    /// Emitter layers reached by rendering or offscreen capture this frame.
    private var activeEmitterLayerIDs: Set<ObjectIdentifier> = []

    // MARK: - Initialization

    /// Creates a new WebGPU renderer with the specified canvas element.
    ///
    /// - Parameter canvas: The JavaScript canvas element to render to.
    public init(canvas: JSObject) {
        self.canvas = canvas
    }

    // MARK: - CARenderer

    @MainActor public func initialize() async throws {
        // Get GPU
        guard let gpu = GPU.shared else {
            throw CARendererError.deviceNotAvailable
        }

        // Request adapter
        guard let adapter = await gpu.requestAdapter() else {
            throw CARendererError.deviceNotAvailable
        }

        // Request device
        let device = try await adapter.requestDevice()
        self.device = device
        uncapturedGPUErrorHandler = device.onUncapturedError { [weak self] event in
            guard self?.firstUncapturedGPUError == nil else { return }
            self?.firstUncapturedGPUError = event.error?.message
        }
        transitionFilterProcessor = CIWebGPUTransitionProcessor(device: device)
        layerFilterProcessor = CIWebGPUFilterProcessor(device: device)

        // Get preferred format
        preferredFormat = gpu.preferredCanvasFormat

        // Configure canvas context
        let ctx = canvas.getContext!("webgpu")
        guard let ctxObject = ctx.object else {
            throw CARendererError.canvasNotConfigured
        }
        context = GPUCanvasContext(jsObject: ctxObject)

        context?.configure(GPUCanvasConfiguration(
            device: device,
            format: preferredFormat,
            usage: [.renderAttachment, .copySrc]
        ))

        // Create shader module
        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.main
        ))

        // Create bind group layout with dynamic uniform buffer
        let layout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(
                        type: .uniform,
                        hasDynamicOffset: true,
                        minBindingSize: UInt64(MemoryLayout<CARendererUniforms>.stride)
                    )
                )
            ]
        ))
        bindGroupLayout = layout

        // Create pipeline layout
        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [layout]
        ))

        // Create render pipeline with depth testing
        pipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))
        depthPipeline = createLayerPipeline(
            device: device,
            shaderModule: shaderModule,
            pipelineLayout: pipelineLayout,
            blend: GPUBlendState(
                color: GPUBlendComponent(
                    srcFactor: .srcAlpha,
                    dstFactor: .oneMinusSrcAlpha,
                    operation: .add
                ),
                alpha: GPUBlendComponent(
                    srcFactor: .one,
                    dstFactor: .oneMinusSrcAlpha,
                    operation: .add
                )
            ),
            stencilCompare: .always,
            depthEnabled: true
        )

        let depthClearShader = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.depthClear
        ))
        depthClearPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(module: depthClearShader, entryPoint: "vertexMain"),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: true,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: depthClearShader,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(format: preferredFormat, writeMask: [])
                ]
            ),
            layout: .auto
        ))
        // Create triple-buffered vertex buffer pool
        // Use maxVertexBufferSize (1MB) to accommodate complex shapes with many vertices
        let vertexBufferSize = Self.maxVertexBufferSize
        vertexBufferPool = VertexBufferPool(
            device: device,
            bufferSize: vertexBufferSize,
            bufferCount: 3
        )

        // Create triple-buffered uniform buffer pool with bind groups
        let uniformBufferSize = Self.alignedUniformSize * UInt64(Self.maxLayers)
        uniformBufferPool = UniformBufferPool(
            device: device,
            bufferSize: uniformBufferSize,
            bindGroupLayout: layout,
            bindingSize: UInt64(MemoryLayout<CARendererUniforms>.stride),
            bufferCount: 3
        )

        // Create texture manager with LRU cache
        let manager = GPUTextureManager(
            device: device,
            maxTextures: 256,
            maxMemoryBytes: 256 * 1024 * 1024  // 256 MB
        )
        // When the texture manager evicts a CGImage's GPU texture, the
        // matching GPUTextureView (and any per-frame bind group built on
        // it) become orphaned. Drop them here so a future CGImage that
        // happens to alias the evicted image's heap address cannot
        // inherit the stale view/bind group.
        manager.onEvict = { [weak self] cgImage in
            guard let self = self else { return }
            let imageID = ObjectIdentifier(cgImage)
            let viewKeys = self.texturedTextureViewCache.keys.filter {
                switch $0 {
                case .image(let id), .emitterImage(let id, _): return id == imageID
                default: return false
                }
            }
            let bindGroupKeys = self.perFrameTexturedBindGroupCache.keys.filter {
                switch $0 {
                case .image(let id), .emitterImage(let id, _): return id == imageID
                default: return false
                }
            }
            for key in viewKeys {
                self.texturedTextureViewCache.removeValue(forKey: key)
            }
            for key in bindGroupKeys {
                self.perFrameTexturedBindGroupCache.removeValue(forKey: key)
            }
        }
        textureManager = manager

        // Create geometry cache for path tessellation
        geometryCache = GeometryCache(
            maxEntries: 256,
            maxMemoryBytes: 64 * 1024 * 1024  // 64 MB
        )

        // Get initial canvas size
        let width = canvas.width.number ?? 800
        let height = canvas.height.number ?? 600
        size = CGSize(width: width, height: height)

        // Create depth texture
        createDepthTexture(width: Int(width), height: Int(height))
        configureRasterizationCache(width: Int(width), height: Int(height))

        // Create textured pipeline
        try createTexturedPipeline(device: device)

        // Create shadow/blur pipelines
        try createShadowPipelines(device: device)

        // Create stencil pipelines for CALayer.mask
        try createStencilPipelines(device: device)
    }

    /// Creates the standard vertex/color pipeline with optional transform-layer depth testing.
    private func createLayerPipeline(
        device: GPUDevice,
        shaderModule: GPUShaderModule,
        pipelineLayout: GPUPipelineLayout,
        blend: GPUBlendState?,
        stencilCompare: GPUCompareFunction,
        depthEnabled: Bool
    ) -> GPURenderPipeline {
        device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(
                                format: .float32x2,
                                offset: UInt64(MemoryLayout<SIMD2<Float>>.stride),
                                shaderLocation: 1
                            ),
                            GPUVertexAttribute(
                                format: .float32x4,
                                offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2),
                                shaderLocation: 2
                            )
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: depthEnabled,
                depthCompare: depthEnabled ? .greaterEqual : .always,
                stencilFront: GPUStencilFaceState(
                    compare: stencilCompare,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: stencilCompare,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [GPUColorTargetState(format: preferredFormat, blend: blend)]
            ),
            layout: .layout(pipelineLayout)
        ))
    }

    /// Creates stencil pipelines for CALayer.mask functionality.
    ///
    /// - stencilWritePipeline: Writes mask alpha to stencil buffer (increment stencil where alpha > 0)
    /// - stencilTestPipeline: Only renders where stencil equals reference value
    private func createStencilPipelines(device: GPUDevice) throws {
        guard let bindGroupLayout = bindGroupLayout else {
            throw CARendererError.pipelineCreationFailed
        }

        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [bindGroupLayout]
        ))

        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.main
        ))

        // Stencil write pipeline: writes to stencil where fragment alpha > 0
        // Uses increment operation to support nested masks
        stencilWritePipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .incrementClamp  // Increment stencil on pass
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .incrementClamp
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        writeMask: []  // No color output, only stencil
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        // Stencil write rounded pipeline: writes to stencil only within rounded rectangle.
        // Uses stencilClipFragment which discards fragments outside the rounded rect,
        // preventing stencil increment in clipped corners.
        stencilWriteRoundedPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .incrementClamp
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .incrementClamp
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "stencilClipFragment",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        writeMask: []  // No color output, only stencil
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        // Stencil test pipeline: only renders where stencil equals reference
        stencilTestPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .equal,  // Only pass where stencil == reference
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))
        depthStencilTestPipeline = createLayerPipeline(
            device: device,
            shaderModule: shaderModule,
            pipelineLayout: pipelineLayout,
            blend: GPUBlendState(
                color: GPUBlendComponent(
                    srcFactor: .srcAlpha,
                    dstFactor: .oneMinusSrcAlpha,
                    operation: .add
                ),
                alpha: GPUBlendComponent(
                    srcFactor: .one,
                    dstFactor: .oneMinusSrcAlpha,
                    operation: .add
                )
            ),
            stencilCompare: .equal,
            depthEnabled: true
        )
    }

    /// Creates the textured render pipeline for layer.contents rendering.
    private func createTexturedPipeline(device: GPUDevice) throws {
        // Create shader module
        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.textured
        ))

        // Create sampler
        textureSampler = device.createSampler(descriptor: GPUSamplerDescriptor(
            addressModeU: .clampToEdge,
            addressModeV: .clampToEdge,
            addressModeW: .clampToEdge,
            magFilter: .linear,
            minFilter: .linear,
            mipmapFilter: .nearest,
            lodMinClamp: 0,
            lodMaxClamp: 0
        ))
        emitterTextureSamplers = Dictionary(uniqueKeysWithValues: EmitterTextureSampling.allCases.map {
            sampling in
            let sampler = device.createSampler(descriptor: GPUSamplerDescriptor(
                addressModeU: .clampToEdge,
                addressModeV: .clampToEdge,
                addressModeW: .clampToEdge,
                magFilter: sampling.magnificationFilter,
                minFilter: sampling.minificationFilter,
                mipmapFilter: sampling.usesMipmaps ? .linear : .nearest,
                lodMinClamp: 0,
                lodMaxClamp: sampling.usesMipmaps ? 32 : 0
            ))
            return (sampling, sampler)
        })

        // Create bind group layout with uniform, sampler, and texture
        texturedBindGroupLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(
                        type: .uniform,
                        hasDynamicOffset: true,
                        minBindingSize: UInt64(MemoryLayout<TexturedUniforms>.stride)
                    )
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(
                        sampleType: .float,
                        viewDimension: .type2D
                    )
                )
            ]
        ))

        guard let texturedBindGroupLayout = texturedBindGroupLayout else {
            throw CARendererError.resourceCreationFailed
        }

        // Create pipeline layout
        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [texturedBindGroupLayout]
        ))

        // Create textured render pipeline
        texturedPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        // Create stencil-aware textured pipeline (tests stencil buffer for mask).
        // Same as texturedPipeline but with stencilFront.compare: .equal instead of .always.
        texturedStencilPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        // R3.5 (PERFORMANCE_DESIGN.md §5.3): opaque variants — `blend: nil`
        // means the fragment output replaces the destination instead of being
        // alpha-composited. Selected via `selectTexturedPipeline(for:)` when
        // `RasterizationDecisions.blendEnabled(for:)` returns false (i.e.
        // layer.isOpaque && layer.opacity >= 1.0). This skips ROP blending,
        // saving bandwidth on opaque images covering most of the layer.
        texturedOpaquePipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: nil
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        texturedStencilOpaquePipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: nil
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        let alphaBlend = GPUBlendState(
            color: GPUBlendComponent(
                srcFactor: .srcAlpha,
                dstFactor: .oneMinusSrcAlpha,
                operation: .add
            ),
            alpha: GPUBlendComponent(
                srcFactor: .one,
                dstFactor: .oneMinusSrcAlpha,
                operation: .add
            )
        )
        texturedDepthPipeline = createLayerPipeline(
            device: device,
            shaderModule: shaderModule,
            pipelineLayout: pipelineLayout,
            blend: alphaBlend,
            stencilCompare: .always,
            depthEnabled: true
        )
        texturedDepthStencilPipeline = createLayerPipeline(
            device: device,
            shaderModule: shaderModule,
            pipelineLayout: pipelineLayout,
            blend: alphaBlend,
            stencilCompare: .equal,
            depthEnabled: true
        )
        texturedDepthOpaquePipeline = createLayerPipeline(
            device: device,
            shaderModule: shaderModule,
            pipelineLayout: pipelineLayout,
            blend: nil,
            stencilCompare: .always,
            depthEnabled: true
        )
        texturedDepthStencilOpaquePipeline = createLayerPipeline(
            device: device,
            shaderModule: shaderModule,
            pipelineLayout: pipelineLayout,
            blend: nil,
            stencilCompare: .equal,
            depthEnabled: true
        )

        createEmitterTexturedAdditivePipelines(
            device: device,
            shaderModule: shaderModule,
            pipelineLayout: pipelineLayout
        )

        createPremultipliedTexturedPipelines(
            device: device,
            bindGroupLayout: texturedBindGroupLayout
        )
        createTransitionFadePipelines(device: device)
    }

    private func createTransitionFadePipelines(device: GPUDevice) {
        let bindGroupLayout = device.createBindGroupLayout(
            descriptor: GPUBindGroupLayoutDescriptor(entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(
                        type: .uniform,
                        hasDynamicOffset: true,
                        minBindingSize: UInt64(MemoryLayout<TransitionFadeUniforms>.stride)
                    )
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 3,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
            ])
        )
        transitionFadeBindGroupLayout = bindGroupLayout
        let pipelineLayout = device.createPipelineLayout(
            descriptor: GPUPipelineLayoutDescriptor(bindGroupLayouts: [bindGroupLayout])
        )
        let shaderModule = device.createShaderModule(
            descriptor: GPUShaderModuleDescriptor(code: CAWebGPUShaders.transitionFade)
        )
        let vertexState = GPUVertexState(
            module: shaderModule,
            entryPoint: "vertexMain",
            buffers: [
                GPUVertexBufferLayout(
                    arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                    attributes: [
                        GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                        GPUVertexAttribute(
                            format: .float32x2,
                            offset: UInt64(MemoryLayout<SIMD2<Float>>.stride),
                            shaderLocation: 1
                        ),
                        GPUVertexAttribute(
                            format: .float32x4,
                            offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2),
                            shaderLocation: 2
                        ),
                    ]
                )
            ]
        )
        let blend = GPUBlendState(
            color: GPUBlendComponent(
                srcFactor: .one,
                dstFactor: .oneMinusSrcAlpha,
                operation: .add
            ),
            alpha: GPUBlendComponent(
                srcFactor: .one,
                dstFactor: .oneMinusSrcAlpha,
                operation: .add
            )
        )

        func makePipeline(
            stencilCompare: GPUCompareFunction,
            depthEnabled: Bool
        ) -> GPURenderPipeline {
            device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
                vertex: vertexState,
                depthStencil: GPUDepthStencilState(
                    format: .depth24plusStencil8,
                    depthWriteEnabled: depthEnabled,
                    depthCompare: depthEnabled ? .greaterEqual : .always,
                    stencilFront: GPUStencilFaceState(
                        compare: stencilCompare,
                        failOp: .keep,
                        depthFailOp: .keep,
                        passOp: .keep
                    ),
                    stencilBack: GPUStencilFaceState(
                        compare: stencilCompare,
                        failOp: .keep,
                        depthFailOp: .keep,
                        passOp: .keep
                    )
                ),
                fragment: GPUFragmentState(
                    module: shaderModule,
                    entryPoint: "fragmentMain",
                    targets: [GPUColorTargetState(format: preferredFormat, blend: blend)]
                ),
                layout: .layout(pipelineLayout)
            ))
        }

        transitionFadePipeline = makePipeline(stencilCompare: .always, depthEnabled: false)
        transitionFadeStencilPipeline = makePipeline(stencilCompare: .equal, depthEnabled: false)
        transitionFadeDepthPipeline = makePipeline(stencilCompare: .always, depthEnabled: true)
        transitionFadeDepthStencilPipeline = makePipeline(stencilCompare: .equal, depthEnabled: true)
    }

    private func createEmitterTexturedAdditivePipelines(
        device: GPUDevice,
        shaderModule: GPUShaderModule,
        pipelineLayout: GPUPipelineLayout
    ) {
        let vertexState = GPUVertexState(
            module: shaderModule,
            entryPoint: "vertexMain",
            buffers: [
                GPUVertexBufferLayout(
                    arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                    attributes: [
                        GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                        GPUVertexAttribute(
                            format: .float32x2,
                            offset: UInt64(MemoryLayout<SIMD2<Float>>.stride),
                            shaderLocation: 1
                        ),
                        GPUVertexAttribute(
                            format: .float32x4,
                            offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2),
                            shaderLocation: 2
                        ),
                    ]
                )
            ]
        )
        let additiveBlend = GPUBlendState(
            color: GPUBlendComponent(
                srcFactor: .srcAlpha,
                dstFactor: .one,
                operation: .add
            ),
            alpha: GPUBlendComponent(
                srcFactor: .one,
                dstFactor: .one,
                operation: .add
            )
        )

        func makePipeline(
            stencilCompare: GPUCompareFunction,
            depthEnabled: Bool
        ) -> GPURenderPipeline {
            device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
                vertex: vertexState,
                depthStencil: GPUDepthStencilState(
                    format: .depth24plusStencil8,
                    depthWriteEnabled: depthEnabled,
                    depthCompare: depthEnabled ? .greaterEqual : .always,
                    stencilFront: GPUStencilFaceState(
                        compare: stencilCompare,
                        failOp: .keep,
                        depthFailOp: .keep,
                        passOp: .keep
                    ),
                    stencilBack: GPUStencilFaceState(
                        compare: stencilCompare,
                        failOp: .keep,
                        depthFailOp: .keep,
                        passOp: .keep
                    )
                ),
                fragment: GPUFragmentState(
                    module: shaderModule,
                    entryPoint: "fragmentMain",
                    targets: [GPUColorTargetState(format: preferredFormat, blend: additiveBlend)]
                ),
                layout: .layout(pipelineLayout)
            ))
        }

        emitterTexturedAdditivePipeline = makePipeline(stencilCompare: .always, depthEnabled: false)
        emitterTexturedAdditiveStencilPipeline = makePipeline(stencilCompare: .equal, depthEnabled: false)
        emitterTexturedAdditiveDepthPipeline = makePipeline(stencilCompare: .always, depthEnabled: true)
        emitterTexturedAdditiveDepthStencilPipeline = makePipeline(stencilCompare: .equal, depthEnabled: true)
    }

    private func createPremultipliedTexturedPipelines(
        device: GPUDevice,
        bindGroupLayout: GPUBindGroupLayout
    ) {
        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.premultipliedTextured
        ))
        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [bindGroupLayout]
        ))
        let vertexState = GPUVertexState(
            module: shaderModule,
            entryPoint: "vertexMain",
            buffers: [
                GPUVertexBufferLayout(
                    arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                    attributes: [
                        GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                        GPUVertexAttribute(
                            format: .float32x2,
                            offset: UInt64(MemoryLayout<SIMD2<Float>>.stride),
                            shaderLocation: 1
                        ),
                        GPUVertexAttribute(
                            format: .float32x4,
                            offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2),
                            shaderLocation: 2
                        )
                    ]
                )
            ]
        )
        let blend = GPUBlendState(
            color: GPUBlendComponent(
                srcFactor: .one,
                dstFactor: .oneMinusSrcAlpha,
                operation: .add
            ),
            alpha: GPUBlendComponent(
                srcFactor: .one,
                dstFactor: .oneMinusSrcAlpha,
                operation: .add
            )
        )

        func makePipeline(
            stencilCompare: GPUCompareFunction,
            depthEnabled: Bool
        ) -> GPURenderPipeline {
            device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
                vertex: vertexState,
                depthStencil: GPUDepthStencilState(
                    format: .depth24plusStencil8,
                    depthWriteEnabled: depthEnabled,
                    depthCompare: depthEnabled ? .greaterEqual : .always,
                    stencilFront: GPUStencilFaceState(
                        compare: stencilCompare,
                        failOp: .keep,
                        depthFailOp: .keep,
                        passOp: .keep
                    ),
                    stencilBack: GPUStencilFaceState(
                        compare: stencilCompare,
                        failOp: .keep,
                        depthFailOp: .keep,
                        passOp: .keep
                    )
                ),
                fragment: GPUFragmentState(
                    module: shaderModule,
                    entryPoint: "fragmentMain",
                    targets: [
                        GPUColorTargetState(format: preferredFormat, blend: blend)
                    ]
                ),
                layout: .layout(pipelineLayout)
            ))
        }

        premultipliedTexturedPipeline = makePipeline(stencilCompare: .always, depthEnabled: false)
        premultipliedTexturedStencilPipeline = makePipeline(stencilCompare: .equal, depthEnabled: false)
        premultipliedTexturedDepthPipeline = makePipeline(stencilCompare: .always, depthEnabled: true)
        premultipliedTexturedDepthStencilPipeline = makePipeline(stencilCompare: .equal, depthEnabled: true)
    }

    /// R3.5: pick the textured pipeline based on stencil state and the
    /// `blendEnabled` decision for the layer. Falls back to the alpha-blended
    /// variant when an opaque pipeline isn't created (test fallback).
    private func selectTexturedPipeline(
        for layer: CALayer,
        forceBlending: Bool = false
    ) -> GPURenderPipeline? {
        let blendOff = !forceBlending && !RasterizationDecisions.blendEnabled(for: layer)
        if transformDepthNesting > 0 {
            if maskNestingDepth > 0 {
                if blendOff, let opaque = texturedDepthStencilOpaquePipeline { return opaque }
                return texturedDepthStencilPipeline ?? texturedDepthPipeline
            }
            if blendOff, let opaque = texturedDepthOpaquePipeline { return opaque }
            return texturedDepthPipeline
        }
        if maskNestingDepth > 0 {
            if blendOff, let opaque = texturedStencilOpaquePipeline { return opaque }
            return texturedStencilPipeline ?? texturedPipeline
        }
        if blendOff, let opaque = texturedOpaquePipeline { return opaque }
        return texturedPipeline
    }

    /// Selects the source-over pipeline for textures whose color channels
    /// already contain alpha. Captured layer and transition textures are
    /// premultiplied render targets, so sampling them through the regular
    /// textured pipeline would multiply RGB by alpha a second time.
    private func selectPremultipliedTexturedPipeline() -> GPURenderPipeline? {
        if transformDepthNesting > 0 {
            return maskNestingDepth > 0
                ? premultipliedTexturedDepthStencilPipeline
                : premultipliedTexturedDepthPipeline
        }
        return maskNestingDepth > 0
            ? premultipliedTexturedStencilPipeline
            : premultipliedTexturedPipeline
    }

    /// Creates shadow and blur pipelines.
    private func createShadowPipelines(device: GPUDevice) throws {
        // Create blur sampler
        blurSampler = device.createSampler(descriptor: GPUSamplerDescriptor(
            addressModeU: .clampToEdge,
            addressModeV: .clampToEdge,
            addressModeW: .clampToEdge,
            magFilter: .linear,
            minFilter: .linear
        ))

        // Create blur bind group layout
        let blurBindGroupLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(type: .uniform)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                )
            ]
        ))

        shadowBindGroupLayout = blurBindGroupLayout

        let blurPipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [blurBindGroupLayout]
        ))

        // Create horizontal blur pipeline
        let shadowBlurHShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.shadowAlphaBlurHorizontal
        ))
        let blurHShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.blurHorizontal
        ))

        shadowBlurHorizontalPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shadowBlurHShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: shadowBlurHShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(format: preferredFormat)
                ]
            ),
            layout: .layout(blurPipelineLayout)
        ))

        // Create vertical blur pipeline
        let blurVShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.blurVertical
        ))

        shadowBlurVerticalPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: blurVShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: blurVShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(format: preferredFormat)
                ]
            ),
            layout: .layout(blurPipelineLayout)
        ))

        filterBlurHorizontalPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: blurHShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: blurHShaderModule,
                entryPoint: "fragmentMain",
                targets: [GPUColorTargetState(format: preferredFormat)]
            ),
            layout: .layout(blurPipelineLayout)
        ))

        filterBlurVerticalPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: blurVShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: blurVShaderModule,
                entryPoint: "fragmentMain",
                targets: [GPUColorTargetState(format: preferredFormat)]
            ),
            layout: .layout(blurPipelineLayout)
        ))

        let rasterizedShadowCompositeModule = device.createShaderModule(
            descriptor: GPUShaderModuleDescriptor(
                code: CAWebGPUShaders.rasterizedShadowComposite
            )
        )
        rasterizedShadowCompositePipeline = device.createRenderPipeline(
            descriptor: GPURenderPipelineDescriptor(
                vertex: GPUVertexState(
                    module: rasterizedShadowCompositeModule,
                    entryPoint: "vertexMain"
                ),
                fragment: GPUFragmentState(
                    module: rasterizedShadowCompositeModule,
                    entryPoint: "fragmentMain",
                    targets: [GPUColorTargetState(format: preferredFormat)]
                ),
                layout: .auto
            )
        )

        // Create shadow composite pipeline
        let shadowCompositeLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(type: .uniform)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                )
            ]
        ))

        let shadowCompositePipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [shadowCompositeLayout]
        ))

        let shadowCompositeShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.shadowComposite
        ))

        shadowCompositePipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shadowCompositeShaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shadowCompositeShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(shadowCompositePipelineLayout)
        ))

        // Create stencil-aware shadow composite pipeline (tests stencil buffer for mask).
        // Same as shadowCompositePipeline but with stencilFront.compare: .equal instead of .always.
        shadowCompositeStencilPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shadowCompositeShaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shadowCompositeShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(shadowCompositePipelineLayout)
        ))

        let filterCompositeShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.filterComposite
        ))

        filterCompositePipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: filterCompositeShaderModule,
                entryPoint: "vertexMain"
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: filterCompositeShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(shadowCompositePipelineLayout)
        ))

        filterReplacementPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: filterCompositeShaderModule,
                entryPoint: "vertexMain"
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: filterCompositeShaderModule,
                entryPoint: "fragmentMain",
                targets: [GPUColorTargetState(format: preferredFormat)]
            ),
            layout: .layout(shadowCompositePipelineLayout)
        ))

        let transformedCompositionShaderModule = device.createShaderModule(
            descriptor: GPUShaderModuleDescriptor(code: CAWebGPUShaders.transformedComposition)
        )
        let capturedCompositionShaderModule = device.createShaderModule(
            descriptor: GPUShaderModuleDescriptor(code: CAWebGPUShaders.capturedComposition)
        )
        let compositionVertexLayout = GPUVertexBufferLayout(
            arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
            attributes: [
                GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                GPUVertexAttribute(
                    format: .float32x2,
                    offset: UInt64(MemoryLayout<SIMD2<Float>>.stride),
                    shaderLocation: 1
                ),
                GPUVertexAttribute(
                    format: .float32x4,
                    offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2),
                    shaderLocation: 2
                ),
            ]
        )
        func makeTransformedCompositionPipeline(
            shaderModule: GPUShaderModule,
            stencilCompare: GPUCompareFunction,
            depthEnabled: Bool
        ) -> GPURenderPipeline {
            device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
                vertex: GPUVertexState(
                    module: shaderModule,
                    entryPoint: "vertexMain",
                    buffers: [compositionVertexLayout]
                ),
                depthStencil: GPUDepthStencilState(
                    format: .depth24plusStencil8,
                    depthWriteEnabled: depthEnabled,
                    depthCompare: depthEnabled ? .greaterEqual : .always,
                    stencilFront: GPUStencilFaceState(
                        compare: stencilCompare,
                        failOp: .keep,
                        depthFailOp: .keep,
                        passOp: .keep
                    ),
                    stencilBack: GPUStencilFaceState(
                        compare: stencilCompare,
                        failOp: .keep,
                        depthFailOp: .keep,
                        passOp: .keep
                    )
                ),
                fragment: GPUFragmentState(
                    module: shaderModule,
                    entryPoint: "fragmentMain",
                    targets: [GPUColorTargetState(format: preferredFormat)]
                ),
                layout: .auto
            ))
        }
        transformedCompositionPipeline = makeTransformedCompositionPipeline(
            shaderModule: transformedCompositionShaderModule,
            stencilCompare: .always,
            depthEnabled: true
        )
        transformedCompositionStencilPipeline = makeTransformedCompositionPipeline(
            shaderModule: transformedCompositionShaderModule,
            stencilCompare: .equal,
            depthEnabled: true
        )
        capturedCompositionPipeline = makeTransformedCompositionPipeline(
            shaderModule: capturedCompositionShaderModule,
            stencilCompare: .always,
            depthEnabled: false
        )
        capturedCompositionStencilPipeline = makeTransformedCompositionPipeline(
            shaderModule: capturedCompositionShaderModule,
            stencilCompare: .equal,
            depthEnabled: false
        )

        let backdropMixBindGroupLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 3,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                ),
            ]
        ))
        let backdropMixPipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [backdropMixBindGroupLayout]
        ))
        let backdropMixShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.backdropFilterMix
        ))
        backdropFilterMixPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: backdropMixShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: backdropMixShaderModule,
                entryPoint: "fragmentMain",
                targets: [GPUColorTargetState(format: preferredFormat)]
            ),
            layout: .layout(backdropMixPipelineLayout)
        ))

        let compositionMaskBindGroupLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .type2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                ),
            ]
        ))
        let compositionMaskPipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [compositionMaskBindGroupLayout]
        ))
        let compositionMaskShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.compositionMaskOperation
        ))
        compositionMaskIntersectPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(module: compositionMaskShaderModule, entryPoint: "vertexMain"),
            fragment: GPUFragmentState(
                module: compositionMaskShaderModule,
                entryPoint: "intersectMasks",
                targets: [GPUColorTargetState(format: preferredFormat)]
            ),
            layout: .layout(compositionMaskPipelineLayout)
        ))
        compositionMaskApplyPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(module: compositionMaskShaderModule, entryPoint: "vertexMain"),
            fragment: GPUFragmentState(
                module: compositionMaskShaderModule,
                entryPoint: "applyMask",
                targets: [GPUColorTargetState(format: preferredFormat)]
            ),
            layout: .layout(compositionMaskPipelineLayout)
        ))

        filterOperationPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: filterCompositeShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: filterCompositeShaderModule,
                entryPoint: "fragmentMain",
                targets: [GPUColorTargetState(format: preferredFormat)]
            ),
            layout: .layout(shadowCompositePipelineLayout)
        ))

        filterCompositeStencilPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: filterCompositeShaderModule,
                entryPoint: "vertexMain"
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .always,
                stencilFront: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: filterCompositeShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(shadowCompositePipelineLayout)
        ))

        // Create shadow mask pipeline (same as main pipeline but WITHOUT depth/stencil).
        // The shadow mask render pass has no depth attachment, so using the main pipeline
        // (which has depthStencil state) would cause a mismatch.
        guard let bindGroupLayout = bindGroupLayout else { return }
        let maskPipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [bindGroupLayout]
        ))
        let maskShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.main
        ))
        shadowMaskPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: maskShaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            fragment: GPUFragmentState(
                module: maskShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(format: preferredFormat)
                ]
            ),
            layout: .layout(maskPipelineLayout)
        ))
    }

    /// Creates or recreates the depth texture for the given size.
    /// Uses depth24plus-stencil8 format to support both depth testing and stencil-based masking.
    private func createDepthTexture(width: Int, height: Int) {
        guard let device = device, width > 0, height > 0 else { return }

        depthTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .depth24plusStencil8,
            usage: .renderAttachment
        ))
        depthTextureView = depthTexture?.createView()
    }

    public func resize(width: Int, height: Int) {
        size = CGSize(width: width, height: height)

        // Update canvas size
        canvas.width = .number(Double(width))
        canvas.height = .number(Double(height))

        // Reconfigure context for new size
        if let device = device {
            context?.configure(GPUCanvasConfiguration(
                device: device,
                format: preferredFormat,
                usage: [.renderAttachment, .copySrc]
            ))
        }

        // Recreate depth texture
        createDepthTexture(width: width, height: height)

        // Shadow captures are viewport-sized. A resize invalidates every per-layer set.
        for resources in shadowLayerResources.values {
            resources.destroy()
        }
        shadowLayerResources.removeAll(keepingCapacity: true)
        prerenderedShadows.removeAll(keepingCapacity: true)

        // Filter captures are viewport-sized. A resize invalidates every per-layer set.
        for resources in filterLayerResources.values {
            resources.destroy()
        }
        filterLayerResources.removeAll(keepingCapacity: true)
        prerenderedFilters.removeAll(keepingCapacity: true)

        // R3.2/R3.4: size the rasterization cache to viewport × 4 × 2.5
        // per PERFORMANCE_DESIGN.md §5.2 (WWDC 2014 #419 budget).
        // Recreating the cache on resize applies the new viewport-derived
        // immutable byte budget. The eviction callback releases every old
        // capture and its dependent texture view before replacement.
        configureRasterizationCache(width: width, height: height)
        prerasterizedTextures.removeAll(keepingCapacity: true)
        rasterizePrerenderRootLayer = nil
    }

    private func configureRasterizationCache(width: Int, height: Int) {
        let budget = max(0, Int((Double(width) * Double(height) * 4.0 * 2.5).rounded()))
        rasterizationCache?.removeAll()
        let cache = RasterizationCache<GPUTexture>(maxBytes: budget)
        cache.onEvict = { [weak self] key, texture in
            if let identity = key.layerIdentity, let self = self {
                let cacheKey = TexturedCacheKey.rasterizedLayer(
                    identity.renderKey,
                    identity.purpose
                )
                self.texturedTextureViewCache.removeValue(forKey: cacheKey)
                self.perFrameTexturedBindGroupCache.removeValue(forKey: cacheKey)
            }
            texture.destroy()
        }
        rasterizationCache = cache
    }

    public func render(layer rootLayer: CALayer) {
        guard let device = device,
              let context = context,
              let pipeline = pipeline,
              bindGroup != nil,
              depthTexture != nil else { return }

        processPendingTileDraws()

        // Phase 1 (PERFORMANCE_DESIGN.md §3.6): bump the per-frame token
        // before any presentation cache lookup so this frame is distinct
        // from the previous one. The process-wide token is synchronized.
        CALayer.advanceFrameToken()

        // Reset per-frame state
        currentLayerIndex = 0
        currentVertexOffset = 0
        droppedLayerCount = 0
        opacityStack.removeAll()
        replicatorColorStack.removeAll()
        replicatorTimeOffsetStack.removeAll()
        replicatorInstancePath.removeAll()
        transitionSuppressedLayer = nil
        renderTargetSizeOverride = nil
        activeTransitionSourceIDs.removeAll(keepingCapacity: true)
        for texture in transientCaptureDepthTextures {
            texture.destroy()
        }
        transientCaptureDepthTextures.removeAll(keepingCapacity: true)
        for texture in transientRasterizationTextures {
            texture.destroy()
        }
        transientRasterizationTextures.removeAll(keepingCapacity: true)
        for resources in transientRasterizationFilterResources {
            resources.destroy()
        }
        transientRasterizationFilterResources.removeAll(keepingCapacity: true)
        for resources in transientRasterizationShadowResources {
            resources.destroy()
        }
        transientRasterizationShadowResources.removeAll(keepingCapacity: true)
        currentRootLayer = nil
        filterPrerenderRootLayer = nil
        contentMaskCaptureSuppressedRootLayer = nil
        shadowCaptureRootLayer = nil
        suppressShadowRendering = false
        activeEmitterLayerIDs.removeAll(keepingCapacity: true)
        collectEmitterLayerIDs(rootLayer, into: &activeEmitterLayerIDs)
        updateDelegateBackingStores(
            in: rootLayer,
            maximumTextureDimension: max(1, Int(device.limits.maxTextureDimension2D))
        )

        // Reset clip rect stack for this frame
        clipRectStack.removeAll()

        // Reset per-frame shadow outputs; persistent resource ownership remains cached.
        prerenderedShadows.removeAll(keepingCapacity: true)

        // Reset per-frame filter outputs; persistent resource ownership remains cached.
        prerenderedFilters.removeAll(keepingCapacity: true)
        for execution in retiringLayerFilterExecutions {
            execution.invalidate()
        }
        retiringLayerFilterExecutions = activeLayerFilterExecutions
        activeLayerFilterExecutions.removeAll(keepingCapacity: true)
        prerenderedCompositions.removeAll(keepingCapacity: true)
        for execution in retiringCompositionExecutions {
            execution.invalidate()
        }
        retiringCompositionExecutions = activeCompositionExecutions
        activeCompositionExecutions.removeAll(keepingCapacity: true)
        compositionCaptureStopKey = nil
        compositionCaptureDidReachStop = false
        compositionCapturePassThroughKeys.removeAll(keepingCapacity: true)
        deferredCompositionRasterizationKeys.removeAll(keepingCapacity: true)
        capturesOnlyDeferredCompositionRasterizations = false
        transformDepthNesting = 0
        transformFlatteningCaptureCount = 0
        explicitRasterizationCaptureCount = 0
        transformFlatteningCompositeCount = 0
        isRenderingMainPass = false

        // Reset rasterization pre-rendering state (R3.2)
        prerasterizedTextures.removeAll(keepingCapacity: true)
        rasterizePrerenderRootLayer = nil

        // Drop the previous frame's textured bind groups: the buffer pool has
        // already rotated `uniformBuffer` so any cached bind group from frame
        // N-1 references a stale buffer. The persistent texture-view cache
        // does not need clearing here — views remain valid until the texture
        // is evicted.
        perFrameTexturedBindGroupCache.removeAll(keepingCapacity: true)

        // Get current texture
        // The swap-chain texture is recreated every frame, so its view must be
        // freshly created. The depth texture only changes on resize and its
        // view is cached in `depthTextureView`.
        let currentTexture = context.getCurrentTexture()
        let textureView = currentTexture.createView()
        guard let depthTextureView = depthTextureView else { return }

        // Create command encoder
        let encoder = device.createCommandEncoder()

        // MARK: - SpriteKit-Compatible Coordinate System (Y-up)
        //
        // OpenCoreAnimation uses a SpriteKit-compatible coordinate system:
        // - Origin (0, 0) is at the BOTTOM-LEFT corner
        // - Positive X goes RIGHT
        // - Positive Y goes UP
        //
        // Reference: https://developer.apple.com/documentation/spritekit/about-spritekit-coordinate-systems
        // "A positive x coordinate goes to the right and a positive y coordinate goes up the screen."
        //
        // This projection matrix maps world coordinates to WebGPU NDC:
        // - World Y=0 → NDC Y=-1 (bottom of screen)
        // - World Y=height → NDC Y=+1 (top of screen)
        // - World X=0 → NDC X=-1 (left of screen)
        // - World X=width → NDC X=+1 (right of screen)
        //
        // Note: This differs from iOS UIKit/CoreAnimation which uses Y-down (origin at top-left).
        // OpenCoreAnimation intentionally follows SpriteKit's Y-up convention for consistency
        // with game development and standard mathematical coordinate systems.
        let projectionMatrix = Matrix4x4.orthographic(
            left: 0,
            right: Float(size.width),
            bottom: 0,
            top: Float(size.height),
            near: -1000,
            far: 1000
        )

        prerenderTransitions(rootLayer, encoder: encoder)

        // A shadow silhouette can depend on rendered mask-tree alpha. Prepare
        // those filtered/masked captures without recursively drawing shadows,
        // then run the normal shadow pass and rebuild final filter captures.
        if shadowPrepassRequiresContentMasks(rootLayer) {
            suppressShadowRendering = true
            prerenderFilteredLayers(
                rootLayer,
                encoder: encoder,
                projectionMatrix: projectionMatrix
            )
            suppressShadowRendering = false
        }

        // Pre-render shadows with 2-pass Gaussian blur.
        prerenderShadows(rootLayer, encoder: encoder, projectionMatrix: projectionMatrix)

        // Pre-render layers with blur filters
        prerenderFilteredLayers(rootLayer, encoder: encoder, projectionMatrix: projectionMatrix)

        prepareDeferredCompositionRasterizations(
            rootLayer,
            projectionMatrix: projectionMatrix
        )

        // Pre-render shouldRasterize subtrees (R3.2 / R3.3)
        prerenderRasterizedLayers(rootLayer, encoder: encoder, projectionMatrix: projectionMatrix)

        // Use root layer's backgroundColor as clear color (for SKScene background)
        // This prevents SKScene's backgroundColor from rendering at zPosition=0
        // which would occlude child layers with negative zPosition (like backgroundLayer)
        let rootPresentation = rootLayer._renderTimePresentation()
        let clearColor: GPUColor
        if rootPresentation.cornerRadius > 0 {
            // When root layer has cornerRadius, use transparent clear color.
            // The background will be rendered as geometry with corner radius SDF.
            clearColor = GPUColor(r: 0, g: 0, b: 0, a: 0)
        } else if rootPresentation.filters?.isEmpty == false
            || prerenderedFilters[renderKey(for: rootLayer)] != nil {
            // The filtered root layer is composited from the offscreen texture, so drawing
            // its background here would double-apply the root background color.
            clearColor = GPUColor(r: 0, g: 0, b: 0, a: 0)
        } else if let bgColor = rootLayer.backgroundColor,
           let components = bgColor.components,
           components.count >= 4 {
            let alpha = components[3]
            clearColor = GPUColor(
                r: components[0] * alpha,
                g: components[1] * alpha,
                b: components[2] * alpha,
                a: alpha
            )
        } else {
            clearColor = GPUColor(r: 0, g: 0, b: 0, a: 1)
        }

        // Store root layer to skip its backgroundColor rendering in renderLayer()
        currentRootLayer = rootLayer

        prerenderBackdropCompositions(
            rootLayer,
            clearColor: clearColor,
            encoder: encoder,
            projectionMatrix: projectionMatrix
        )
        if !deferredCompositionRasterizationKeys.isEmpty {
            capturesOnlyDeferredCompositionRasterizations = true
            prerenderRasterizedLayers(
                rootLayer,
                encoder: encoder,
                projectionMatrix: projectionMatrix
            )
            capturesOnlyDeferredCompositionRasterizations = false
        }

        // Begin render pass with depth attachment
        let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: textureView,
                    clearValue: clearColor,
                    loadOp: .clear,
                    storeOp: .store
                )
            ],
            depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                view: depthTextureView,
                depthClearValue: 0.0,
                depthLoadOp: .clear,
                depthStoreOp: .store,
                stencilClearValue: 0,
                stencilLoadOp: .clear,
                stencilStoreOp: .store
            )
        ))

        renderPass.setPipeline(pipeline)

        // Set viewport to full canvas size
        renderPass.setViewport(
            x: 0,
            y: 0,
            width: Float(size.width),
            height: Float(size.height),
            minDepth: 0,
            maxDepth: 1
        )

        // Render layer tree (projectionMatrix already created above for shadow pre-rendering)
        isRenderingMainPass = true
        renderLayer(rootLayer, renderPass: renderPass, parentMatrix: projectionMatrix)
        isRenderingMainPass = false

        renderPass.end()

        // Log warning if layers were dropped due to buffer capacity
        if droppedLayerCount > 0 {
            print("[CAWebGPURenderer] \(droppedLayerCount) layer(s) dropped: buffer capacity exceeded (maxLayers=\(Self.maxLayers))")
        }

        // Submit command buffer
        device.queue.submit([encoder.finish()])
        lastRenderedTexture = currentTexture

        // Phase 1 commit-end housekeeping (PERFORMANCE_DESIGN.md §3.8 / §6.5).
        // Order is mandated: submit → clear → user-visible completion blocks.
        // Clearing here means subsequent setters that reach this layer in the
        // SAME frame will mark it dirty for the NEXT frame, never for the one
        // that just left the renderer.
        rootLayer.recursivelyClearDirtyAfterCommit()

        // Advance buffer pools, texture manager, and geometry cache to the next frame
        vertexBufferPool?.advanceFrame()
        uniformBufferPool?.advanceFrame()
        textureManager?.advanceFrame()
        geometryCache?.advanceFrame()

        // Periodically evict stale resources (not used in last 300 frames = ~5 seconds at 60fps)
        textureManager?.evictStale(olderThan: 300)
        geometryCache?.evictStale(olderThan: 300)

        // R3.4: drop rasterization entries that have sat idle longer
        // than 6 frames (~100 ms @ 60 Hz) and any overflow above the
        // viewport-derived byte budget. Idle eviction first so the
        // budget pass operates on the trimmed live set.
        if let cache = rasterizationCache {
            cache.evictIdle(currentFrame: CALayer._currentFrameToken, olderThan: 6)
            cache.evictToBudget()
        }
        prerasterizedTextures.removeAll(keepingCapacity: true)
        emitterLayerStates = emitterLayerStates.filter {
            activeEmitterLayerIDs.contains($0.key)
        }
    }

    /// Resolves ordinary CALayer delegate drawing before any shadow, filter,
    /// rasterization, or composition prepass consumes the layer subtree.
    private func updateDelegateBackingStores(
        in rootLayer: CALayer,
        maximumTextureDimension: Int
    ) {
        var visited: Set<ObjectIdentifier> = []
        var activeLayerIDs: Set<ObjectIdentifier> = []

        func visit(_ layer: CALayer) {
            let identifier = ObjectIdentifier(layer)
            guard visited.insert(identifier).inserted else { return }
            activeLayerIDs.insert(identifier)

            if layer.needsDisplay(), supportsDelegateBackingStore(for: layer) {
                updateDelegateBackingStore(
                    for: layer,
                    identifier: identifier,
                    maximumTextureDimension: maximumTextureDimension
                )
            } else if layer._dirtyMask.contains(.contents) {
                // An explicit contents assignment supersedes the previously
                // rasterized delegate backing store.
                delegateBackingStores.removeValue(forKey: identifier)
            }
            if let mask = layer.mask {
                visit(mask)
            }
            for sublayer in layer.sublayers ?? [] {
                visit(sublayer)
            }
        }

        visit(rootLayer)
        delegateBackingStores = delegateBackingStores.filter {
            activeLayerIDs.contains($0.key)
        }
    }

    private func supportsDelegateBackingStore(for layer: CALayer) -> Bool {
        !(layer is CATiledLayer)
            && !(layer is CATransformLayer)
            && !(layer is CAEmitterLayer)
            && !(layer is CATextLayer)
            && !(layer is CAShapeLayer)
            && !(layer is CAGradientLayer)
    }

    private func updateDelegateBackingStore(
        for layer: CALayer,
        identifier: ObjectIdentifier,
        maximumTextureDimension: Int
    ) {
        let revisionBeforeDisplay = layer._contentRevision
        layer.displayIfNeeded()
        if layer._contentRevision != revisionBeforeDisplay {
            delegateBackingStores.removeValue(forKey: identifier)
            return
        }

        guard let delegate = layer.delegate else {
            delegateBackingStores.removeValue(forKey: identifier)
            return
        }
        let bounds = layer.bounds
        let scale = layer.contentsScale
        guard bounds.width.isFinite,
              bounds.height.isFinite,
              bounds.width > 0,
              bounds.height > 0,
              scale.isFinite,
              scale > 0 else {
            delegateBackingStores.removeValue(forKey: identifier)
            delegateDrawFailureCount += 1
            return
        }

        let pixelWidthValue = ceil(bounds.width * scale)
        let pixelHeightValue = ceil(bounds.height * scale)
        let maximumDimension = CGFloat(maximumTextureDimension)
        guard pixelWidthValue.isFinite,
              pixelHeightValue.isFinite,
              pixelWidthValue <= maximumDimension,
              pixelHeightValue <= maximumDimension else {
            delegateBackingStores.removeValue(forKey: identifier)
            delegateDrawFailureCount += 1
            return
        }
        let pixelWidth = Int(pixelWidthValue)
        let pixelHeight = Int(pixelHeightValue)
        guard let context = CGContext(
            softwareData: nil,
            width: pixelWidth,
            height: pixelHeight,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        ) else {
            delegateBackingStores.removeValue(forKey: identifier)
            delegateDrawFailureCount += 1
            return
        }

        context.scaleBy(x: scale, y: scale)
        if layer.contentsAreFlipped() {
            context.translateBy(x: -bounds.minX, y: -bounds.minY)
        } else {
            // CGImage row zero is the top row, while OpenCoreAnimation's
            // default layer geometry is Y-up. Write logical Y=0 into the
            // final bitmap row so textured display preserves that contract.
            context.translateBy(x: -bounds.minX, y: bounds.maxY)
            context.scaleBy(x: 1, y: -1)
        }
        delegate.layerWillDraw(layer)
        layer.draw(in: context)
        guard let image = context.makeImage() else {
            delegateBackingStores.removeValue(forKey: identifier)
            delegateDrawFailureCount += 1
            return
        }
        delegateBackingStores[identifier] = image
    }

    private func collectEmitterLayerIDs(
        _ layer: CALayer,
        into result: inout Set<ObjectIdentifier>
    ) {
        if layer is CAEmitterLayer {
            result.insert(ObjectIdentifier(layer))
        }
        for sublayer in layer.sublayers ?? [] {
            collectEmitterLayerIDs(sublayer, into: &result)
        }
    }

    /// Reads one pixel from the most recently submitted canvas texture.
    ///
    /// This is primarily useful for rendering conformance tests and diagnostics.
    /// The returned components are normalized to RGBA regardless of the canvas
    /// texture's native channel order.
    @MainActor
    public func readbackPixel(x: Int, y: Int) async throws -> [UInt8] {
        try await readbackPixels(at: [CGPoint(x: x, y: y)])[0]
    }

    /// Reads pixels from the most recently submitted canvas texture in one copy.
    ///
    /// A browser canvas texture is presentation-scoped. Copying the complete
    /// texture once keeps multi-point diagnostics deterministic after present.
    @MainActor
    public func readbackPixels(at points: [CGPoint]) async throws -> [[UInt8]] {
        guard let device, let texture = lastRenderedTexture else {
            throw CARendererError.renderingFailed("No rendered texture is available for readback")
        }
        guard !points.isEmpty else { return [] }
        guard points.allSatisfy({
            $0.x >= 0 && $0.y >= 0
                && $0.x < CGFloat(texture.width) && $0.y < CGFloat(texture.height)
        }) else {
            throw CARendererError.renderingFailed("Readback coordinate is outside the render target")
        }

        let unalignedBytesPerRow = UInt32(texture.width) * 4
        let bytesPerRow = ((unalignedBytesPerRow + 255) / 256) * 256
        let bufferSize = UInt64(bytesPerRow) * UInt64(texture.height)
        let stagingBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: bufferSize,
            usage: [.mapRead, .copyDst],
            label: "CAWebGPU Texture Readback"
        ))
        let encoder = device.createCommandEncoder()
        encoder.copyTextureToBuffer(
            source: GPUImageCopyTexture(
                texture: texture,
                origin: GPUOrigin3D(x: 0, y: 0, z: 0)
            ),
            destination: GPUImageCopyBuffer(
                buffer: stagingBuffer,
                offset: 0,
                bytesPerRow: bytesPerRow,
                rowsPerImage: UInt32(texture.height)
            ),
            copySize: GPUExtent3D(
                width: UInt32(texture.width),
                height: UInt32(texture.height),
                depthOrArrayLayers: 1
            )
        )
        device.queue.submit([encoder.finish()])

        do {
            try await stagingBuffer.mapAsync(mode: .read)
        } catch {
            stagingBuffer.destroy()
            throw CARendererError.renderingFailed("WebGPU pixel readback failed: \(error)")
        }

        let mappedRange = stagingBuffer.getMappedRange()
        let bytes = JSObject.global.Uint8Array.function!.new(mappedRange)
        let pixels = points.map { point -> [UInt8] in
            let offset = Int(point.y) * Int(bytesPerRow) + Int(point.x) * 4
            let first = UInt8(bytes[offset].number ?? 0)
            let second = UInt8(bytes[offset + 1].number ?? 0)
            let third = UInt8(bytes[offset + 2].number ?? 0)
            let alpha = UInt8(bytes[offset + 3].number ?? 0)
            if preferredFormat == .bgra8unorm {
                return [third, second, first, alpha]
            }
            return [first, second, third, alpha]
        }
        stagingBuffer.unmap()
        stagingBuffer.destroy()
        return pixels
    }

    public func invalidate() {
        if let device, let uncapturedGPUErrorHandler {
            device.removeUncapturedErrorHandler(uncapturedGPUErrorHandler)
        }
        uncapturedGPUErrorHandler = nil
        firstUncapturedGPUError = nil

        // Invalidate buffer pools
        vertexBufferPool?.invalidate()
        vertexBufferPool = nil
        uniformBufferPool?.invalidate()
        uniformBufferPool = nil

        // Invalidate texture manager
        textureManager?.invalidate()
        textureManager = nil

        // Invalidate geometry cache
        geometryCache?.invalidate()
        geometryCache = nil

        bindGroupLayout = nil
        depthTexture = nil
        pipeline = nil
        depthPipeline = nil
        depthClearPipeline = nil
        texturedPipeline = nil
        texturedDepthPipeline = nil
        texturedOpaquePipeline = nil
        texturedDepthOpaquePipeline = nil
        premultipliedTexturedPipeline = nil
        premultipliedTexturedStencilPipeline = nil
        premultipliedTexturedDepthPipeline = nil
        premultipliedTexturedDepthStencilPipeline = nil
        transitionFadePipeline = nil
        transitionFadeStencilPipeline = nil
        transitionFadeDepthPipeline = nil
        transitionFadeDepthStencilPipeline = nil
        transitionFadeBindGroupLayout = nil
        texturedBindGroupLayout = nil
        textureSampler = nil

        // Shadow resources
        for resources in shadowLayerResources.values {
            resources.destroy()
        }
        shadowLayerResources.removeAll(keepingCapacity: false)
        prerenderedShadows.removeAll(keepingCapacity: false)
        shadowRenderFailureCount = 0
        rasterizationFailureCount = 0
        delegateBackingStores.removeAll(keepingCapacity: false)
        delegateDrawFailureCount = 0
        depthTextureView = nil
        lastRenderedTexture = nil
        shadowBlurHorizontalPipeline = nil
        shadowBlurVerticalPipeline = nil
        filterBlurHorizontalPipeline = nil
        filterBlurVerticalPipeline = nil
        shadowCompositePipeline = nil
        rasterizedShadowCompositePipeline = nil
        shadowMaskPipeline = nil
        shadowBindGroupLayout = nil
        blurSampler = nil
        textTextureCache.removeAll()
        textTextureAccessOrder.removeAll()

        // Filter resources
        filterCompositePipeline = nil
        filterCompositeStencilPipeline = nil
        filterOperationPipeline = nil
        for resources in filterLayerResources.values {
            resources.destroy()
        }
        filterLayerResources.removeAll(keepingCapacity: false)
        prerenderedFilters.removeAll(keepingCapacity: false)
        for execution in activeLayerFilterExecutions {
            execution.invalidate()
        }
        for execution in retiringLayerFilterExecutions {
            execution.invalidate()
        }
        activeLayerFilterExecutions.removeAll(keepingCapacity: false)
        retiringLayerFilterExecutions.removeAll(keepingCapacity: false)
        failedLayerFilterKeys.removeAll(keepingCapacity: false)
        layerFilterFailureCount = 0
        layerFilterProcessor?.invalidate()
        layerFilterProcessor = nil
        for resources in compositionLayerResources.values {
            resources.destroy()
        }
        compositionLayerResources.removeAll(keepingCapacity: false)
        prerenderedCompositions.removeAll(keepingCapacity: false)
        for execution in activeCompositionExecutions {
            execution.invalidate()
        }
        for execution in retiringCompositionExecutions {
            execution.invalidate()
        }
        activeCompositionExecutions.removeAll(keepingCapacity: false)
        retiringCompositionExecutions.removeAll(keepingCapacity: false)
        compositionCaptureStopKey = nil
        compositionCaptureDidReachStop = false
        compositionCapturePassThroughKeys.removeAll(keepingCapacity: false)
        deferredCompositionRasterizationKeys.removeAll(keepingCapacity: false)
        capturesOnlyDeferredCompositionRasterizations = false
        failedCompositionKeys.removeAll(keepingCapacity: false)
        compositionFilterFailureCount = 0

        // Rasterization cache (R3.2 / R3.4)
        rasterizationCache?.removeAll()
        rasterizationCache = nil
        for capture in transitionCaptures.values {
            capture.filterExecution?.invalidate()
            capture.source.texture.destroy()
            capture.target.texture.destroy()
        }
        transitionCaptures.removeAll(keepingCapacity: false)
        activeTransitionSourceIDs.removeAll(keepingCapacity: false)
        failedTransitionSourceIDs.removeAll(keepingCapacity: false)
        for texture in transientCaptureDepthTextures {
            texture.destroy()
        }
        transientCaptureDepthTextures.removeAll(keepingCapacity: false)
        for texture in transientRasterizationTextures {
            texture.destroy()
        }
        transientRasterizationTextures.removeAll(keepingCapacity: false)
        for resources in transientRasterizationFilterResources {
            resources.destroy()
        }
        transientRasterizationFilterResources.removeAll(keepingCapacity: false)
        for resources in transientRasterizationShadowResources {
            resources.destroy()
        }
        transientRasterizationShadowResources.removeAll(keepingCapacity: false)
        transitionSourceCaptureCount = 0
        transitionTargetCaptureCount = 0
        transitionFilterDispatchCount = 0
        transitionFilterFailureCount = 0
        transitionRenderFailureCount = 0
        transitionFilterProcessor?.invalidate()
        transitionFilterProcessor = nil
        prerasterizedTextures.removeAll(keepingCapacity: false)
        rasterizePrerenderRootLayer = nil
        shadowCaptureRootLayer = nil
        suppressShadowRendering = false
        contentMaskCaptureSuppressedRootLayer = nil
        for request in pendingTileDraws {
            request.tiledLayer.loadingTiles.remove(request.tileKey)
        }
        pendingTileDraws.removeAll(keepingCapacity: false)

        filterReplacementPipeline = nil
        transformedCompositionPipeline = nil
        transformedCompositionStencilPipeline = nil
        capturedCompositionPipeline = nil
        capturedCompositionStencilPipeline = nil
        backdropFilterMixPipeline = nil
        compositionMaskIntersectPipeline = nil
        compositionMaskApplyPipeline = nil

        // Stencil resources
        stencilWritePipeline = nil
        stencilWriteRoundedPipeline = nil
        stencilTestPipeline = nil
        depthStencilTestPipeline = nil
        texturedStencilPipeline = nil
        texturedDepthStencilPipeline = nil
        texturedStencilOpaquePipeline = nil
        texturedDepthStencilOpaquePipeline = nil
        shadowCompositeStencilPipeline = nil
        maskNestingDepth = 0
        transformDepthNesting = 0
        currentStencilValue = 0

        // Particle resources
        emitterTextureSamplers.removeAll(keepingCapacity: false)
        emitterTexturedAdditivePipeline = nil
        emitterTexturedAdditiveStencilPipeline = nil
        emitterTexturedAdditiveDepthPipeline = nil
        emitterTexturedAdditiveDepthStencilPipeline = nil
        emitterLayerStates.removeAll(keepingCapacity: false)
        activeEmitterLayerIDs.removeAll(keepingCapacity: false)
        emitterSpawnFailureCount = 0
        emitterRenderFailureCount = 0

        // Textured bind group / view caches (release JS-side handles)
        perFrameTexturedBindGroupCache.removeAll(keepingCapacity: false)
        texturedTextureViewCache.removeAll(keepingCapacity: false)

        // Persistent JS Float32Array staging pool (release JS handles)
        float32StagingPool.removeAll(keepingCapacity: false)

        context = nil
        device = nil
    }

    // MARK: - Private Methods

    /// Returns a persistent JS `Float32Array` of exactly `floatCount` elements,
    /// allocating a new one on first use and reusing it for every subsequent
    /// call with the same size.
    ///
    /// The pool exists to eliminate per-frame JS heap allocation: the previous
    /// `JSTypedArray<Float32>(buffer:)` path created a fresh `ArrayBuffer`
    /// every call (via `swjs_create_typed_array` → `.slice()`), generating
    /// ~0.75 MB/sec of garbage on a typical scene. See `float32StagingPool`.
    private func stagingFloat32Array(floatCount: Int) -> JSObject {
        if let cached = float32StagingPool[floatCount] {
            return cached
        }
        let array = JSObject.global.Float32Array.function!.new(floatCount)
        float32StagingPool[floatCount] = array
        return array
    }

    /// Writes Swift bytes into a pooled persistent JS `Float32Array` for
    /// WebGPU buffer writes.
    ///
    /// Per-element subscript assignment crosses the JS bridge once per float
    /// (`swjs_set_indexed_property`). V8 pipelines indexed TypedArray writes
    /// efficiently, and at ~3,000 floats per frame the bridge cost is well
    /// under a millisecond. The win is zero JS heap churn: the same array
    /// is overwritten in place every frame.
    ///
    /// The returned `JSObject` is shared across calls with the same float
    /// count. WebGPU's `writeBuffer` copies the contents synchronously, so
    /// the next call may safely overwrite the array.
    private func createFloat32Array<T>(from data: inout T) -> JSObject {
        let floatCount = MemoryLayout<T>.stride / 4
        let array = stagingFloat32Array(floatCount: floatCount)
        withUnsafeBytes(of: &data) { rawBytes in
            let typed = rawBytes.bindMemory(to: Float32.self)
            for i in 0..<floatCount {
                array[i] = .number(Double(typed[i]))
            }
        }
        return array
    }

    /// Writes an array of vertices into a pooled persistent JS `Float32Array`.
    /// See ``createFloat32Array(from:)`` for the rationale.
    private func createFloat32Array(from vertices: inout [CARendererVertex]) -> JSObject {
        let floatCount = vertices.count * (MemoryLayout<CARendererVertex>.stride / 4)
        let array = stagingFloat32Array(floatCount: floatCount)
        vertices.withUnsafeBytes { rawBytes in
            let typed = rawBytes.bindMemory(to: Float32.self)
            for i in 0..<floatCount {
                array[i] = .number(Double(typed[i]))
            }
        }
        return array
    }

    private func renderLayer(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        guard let device = device,
              let bindGroup = bindGroup else { return }

        // Get the presentation layer for animated values, fall back to model layer
        // This is critical for animations to be visible - the presentation layer
        // reflects the current animated state of all properties
        let presentationLayer = renderPresentation(for: layer)

        // Skip hidden layers (using presentation layer values)
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }

        let currentRenderKey = renderKey(for: layer)
        if compositionCaptureDidReachStop {
            return
        }
        if compositionCaptureStopKey == currentRenderKey {
            compositionCaptureDidReachStop = true
            return
        }

        if transitionSuppressedLayer !== layer,
           let transitionState = presentationLayer._transitionRenderState {
            renderTransition(
                state: transitionState,
                renderPass: renderPass,
                parentMatrix: parentMatrix
            )
            return
        }

        // Push effective opacity (parent * layer) for this subtree.
        // R3.3: when capturing into a rasterization texture or filter
        // source texture, force the root's contribution to 1.0 so the
        // captured pixels are fully opaque; layer.opacity is reapplied
        // at composite time.
        let isCaptureRoot =
            filterPrerenderRootLayer === layer
            || shadowCaptureRootLayer === layer
            || rasterizePrerenderRootLayer === layer
        let opacityMultiplier: Float = isCaptureRoot ? 1 : presentationLayer.opacity
        let effectiveOpacity = currentEffectiveOpacity * opacityMultiplier
        opacityStack.append(effectiveOpacity)
        defer { _ = opacityStack.popLast() }

        // Calculate model matrix using presentation layer values.
        //
        // Capture-root shortcut (R3.2 / §5.2): when this very layer is
        // the root of an in-flight rasterization capture, `parentMatrix`
        // is already the bounds-local capture projection set up by
        // `captureRasterizedLayer`. Using the regular `modelMatrix`
        // would re-apply position/transform/anchor and shift the
        // captured pixels out of the texture. Those transforms belong
        // at composite time, not in the bake.
        let modelMatrix: Matrix4x4
        if rasterizePrerenderRootLayer === layer {
            modelMatrix = parentMatrix
        } else {
            modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)
        }

        // A transform layer contributes only its transform and per-child opacity.
        // Its own 2D compositing, mask, shadow, filter, rasterization, and backface
        // properties do not create a drawable plane.
        if layer is CATransformLayer {
            renderTransformLayerSublayers(
                layer,
                presentationLayer: presentationLayer,
                renderPass: renderPass,
                modelMatrix: modelMatrix
            )
            return
        }

        let hasBackdropComposition = presentationLayer.compositingFilter != nil
            || presentationLayer.backgroundFilters?.isEmpty == false
        if rasterizePrerenderRootLayer !== layer,
           !hasBackdropComposition,
           !compositionCapturePassThroughKeys.contains(currentRenderKey),
           let flattenedTexture = prerasterizedTextures[currentRenderKey] {
            if isRenderingMainPass,
               flattenedTexture.purpose == .transformFlattening {
                transformFlatteningCompositeCount += 1
            }
            renderRasterizedLayerComposite(
                layer,
                presentationLayer: presentationLayer,
                prerasterized: flattenedTexture,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix
            )
            return
        }

        if filterPrerenderRootLayer !== layer,
           rasterizePrerenderRootLayer !== layer,
           failedLayerFilterKeys.contains(renderKey(for: layer)) {
            return
        }
        // Backface culling: skip rendering when layer faces away from camera
        if !presentationLayer.isDoubleSided {
            // The 2x2 determinant of the upper-left submatrix indicates face direction.
            // Positive = front-facing, negative = back-facing (flipped by odd number of reflections).
            let col0 = modelMatrix.columns.0
            let col1 = modelMatrix.columns.1
            let det2x2 = col0.x * col1.y - col0.y * col1.x
            if det2x2 < 0 {
                return  // Back-facing: defer handles opacity pop
            }
        }

        // Handle CALayer.mask if set
        let preparedFilter = prerenderedFilters[currentRenderKey]
        let hasPreparedContentMask = filterPrerenderRootLayer !== layer
            && preparedFilter?.appliedContentMask == true
        let hasMask = presentationLayer.mask != nil
            && contentMaskCaptureSuppressedRootLayer !== layer
            && !hasPreparedContentMask
        if hasMask, let maskLayer = presentationLayer.mask {
            // Render mask to stencil buffer
            renderMaskToStencil(maskLayer, renderPass: renderPass, parentMatrix: modelMatrix)
        }
        defer {
            if hasMask {
                clearStencilMask(renderPass: renderPass)
            }
        }

        // Render shadow before layer content. Filters apply to the layer subtree capture, but
        // the shadow itself is composited separately in the main pass.
        if filterPrerenderRootLayer !== layer,
           rasterizePrerenderRootLayer !== layer,
           shadowCaptureRootLayer == nil,
           !suppressShadowRendering,
           presentationLayer.shadowOpacity > 0 && presentationLayer.shadowColor != nil {
            renderLayerShadow(
                layer,
                presentationLayer: presentationLayer,
                device: device,
                renderPass: renderPass
            )
        }

        // Handle special layer types first

        if filterPrerenderRootLayer !== layer, hasBackdropComposition {
            if let composition = prerenderedCompositions[currentRenderKey] {
                renderPreparedComposition(
                    composition,
                    presentationLayer: presentationLayer,
                    device: device,
                    renderPass: renderPass,
                    modelMatrix: modelMatrix
                )
            }
            // Requested backdrop effects never fall through to an unprocessed source-over draw.
            return
        }

        // Check for layer filters.
        // If this layer has supported filters and was pre-rendered, composite the filtered result.
        let hasRequestedFilters = presentationLayer.filters?.isEmpty == false
        let shouldCompositeAsGroup = requiresGroupOpacity(presentationLayer)
        if filterPrerenderRootLayer !== layer,
           rasterizePrerenderRootLayer !== layer,
           !compositionCapturePassThroughKeys.contains(currentRenderKey),
           (hasRequestedFilters || shouldCompositeAsGroup || hasPreparedContentMask) {
            if let prerendered = preparedFilter {
                // Composite the pre-rendered filtered texture.
                renderFilteredLayerComposite(
                    layer,
                    prerendered: prerendered,
                    device: device,
                    renderPass: renderPass,
                    modelMatrix: modelMatrix
                )
                return
            }
            // A requested filter path must not silently fall back to unfiltered rendering.
            if hasRequestedFilters {
                return
            }
        }

        // CAEmitterLayer: Render particle system
        if let emitterPresentation = presentationLayer as? CAEmitterLayer,
           let emitterModel = layer as? CAEmitterLayer {
            renderEmitterLayer(
                emitterModel,
                presentation: emitterPresentation,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix
            )
            // Render sublayers after particles
            if let sublayers = layer.sublayers, !sublayers.isEmpty {
                // Use sublayerMatrix helper to apply sublayerTransform and bounds.origin offset
                let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)
                for sublayer in orderedSublayers(for: layer) {
                    self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
                }
            }
            return
        }

        // CATiledLayer: Render tiled content
        if let tiledPresentation = presentationLayer as? CATiledLayer,
           let tiledModel = layer as? CATiledLayer {
            renderTiledLayer(
                tiledModel,
                presentation: tiledPresentation,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix
            )
            return
        }

        // Apple's CoreAnimation draw order for a single layer:
        //   1. shadow (already drawn above)
        //   2. backgroundColor
        //   3. contents / text / shape / gradient (foreground)
        //   4. sublayers
        //   5. border (topmost, drawn over sublayers)
        //
        // Render background color unconditionally (coexists with contents per Apple spec).
        // Skip root layer - its backgroundColor is rendered via clear color to avoid z-fighting,
        // unless cornerRadius > 0 (rounded root must be drawn as a rounded rect).
        if presentationLayer.backgroundColor != nil
            && presentationLayer.bounds.width > 0
            && presentationLayer.bounds.height > 0
            && (layer !== currentRootLayer
                || presentationLayer.cornerRadius > 0
                || shadowCaptureRootLayer === layer) {
            renderLayerBackgroundColor(
                presentationLayer,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix,
                bindGroup: bindGroup
            )
        }

        // Foreground — mutually exclusive subclass branches.
        // backgroundColor is drawn separately above, so it coexists with foreground per Apple spec.
        if let textLayer = presentationLayer as? CATextLayer {
            if textLayer.string != nil {
                renderTextLayer(textLayer, device: device, renderPass: renderPass,
                               modelMatrix: modelMatrix, bindGroup: bindGroup)
            }
        } else if let shapeLayer = presentationLayer as? CAShapeLayer {
            if shapeLayer.path != nil {
                renderShapeLayer(shapeLayer, device: device, renderPass: renderPass,
                                modelMatrix: modelMatrix, bindGroup: bindGroup)
            }
        } else if let gradientLayer = presentationLayer as? CAGradientLayer,
                  let colors = gradientLayer.colors, !colors.isEmpty {
            renderGradientLayer(gradientLayer, device: device, renderPass: renderPass,
                              modelMatrix: modelMatrix, bindGroup: bindGroup)
        } else if let contents = delegateBackingStores[ObjectIdentifier(layer)] {
            renderContentsLayer(presentationLayer, contents: contents, device: device,
                               renderPass: renderPass, modelMatrix: modelMatrix)
        } else if let contents = presentationLayer.contents as? CGImage {
            renderContentsLayer(presentationLayer, contents: contents, device: device,
                               renderPass: renderPass, modelMatrix: modelMatrix)
        }

        // Render sublayers (use model layer hierarchy, but presentation layer's sublayerTransform)
        if layer.sublayers != nil {
            // Use sublayerMatrix helper to apply sublayerTransform and bounds.origin offset
            let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)

            // Apply masksToBounds clipping if enabled
            let shouldClip = presentationLayer.masksToBounds
            let useStencilClip = shouldClip && presentationLayer.cornerRadius > 0
            if shouldClip {
                // Scissor rect for axis-aligned rectangular clipping (always applied as optimization)
                let layerClipRect = calculateClipRect(layer: presentationLayer, modelMatrix: modelMatrix)
                let newClipRect = currentClipRect.intersection(with: layerClipRect)
                clipRectStack.append(newClipRect)
                applyScissorRect(renderPass)

                // Rounded corner clipping via stencil buffer (only when cornerRadius > 0)
                if useStencilClip {
                    renderRoundedRectToStencil(presentationLayer, renderPass: renderPass,
                                               modelMatrix: modelMatrix, device: device)
                }
            }

            // Check if this is a replicator layer
            // `sortedSublayers()` is cached on the model parent (`layer`); a
            // CAReplicatorLayer's presentation copy has no sublayers of its
            // own, so the cache must be read off the model side.
            if let replicatorLayer = presentationLayer as? CAReplicatorLayer,
               let replicatorModelLayer = layer as? CAReplicatorLayer {
                renderReplicatorSublayers(
                    replicatorModelLayer: replicatorModelLayer,
                    replicatorLayer: replicatorLayer,
                    sublayers: layer.sublayers ?? [],
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix
                )
            } else {
                for sublayer in orderedSublayers(for: layer) {
                    self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
                }
            }

            // Restore clipping state
            if shouldClip {
                if useStencilClip {
                    clearStencilMask(renderPass: renderPass)
                }
                _ = clipRectStack.popLast()
                applyScissorRect(renderPass)
            }
        }

        // Render border AFTER sublayers so it sits on top (Apple spec).
        // Border is part of the layer itself, so it draws in the parent's scissor rect
        // (masksToBounds clipping for sublayers has already been restored above).
        if !compositionCaptureDidReachStop,
           presentationLayer.borderWidth > 0 && presentationLayer.borderColor != nil {
            renderLayerBorder(
                presentationLayer,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix,
                bindGroup: bindGroup
            )
        }

    }

    private func prerenderTransitions(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder
    ) {
        guard let device, let pipeline else { return }
        var visited: Set<ObjectIdentifier> = []

        func collect(_ layer: CALayer) {
            let identifier = ObjectIdentifier(layer)
            guard visited.insert(identifier).inserted else { return }

            // Capture descendants first so a nested active transition is available
            // when its parent subtree is frozen into the target texture.
            let presentation = layer._renderTimePresentation()
            if let mask = presentation.mask {
                collect(mask)
            }
            for sublayer in layer.sublayers ?? [] {
                collect(sublayer)
            }

            if let state = presentation._transitionRenderState {
                let sourceID = ObjectIdentifier(state.sourceLayer)
                activeTransitionSourceIDs.insert(sourceID)
                if transitionCaptures[sourceID] == nil,
                   !failedTransitionSourceIDs.contains(sourceID) {
                    let capture: TransitionCapturePair?
                    if let filterValue = state.filter {
                        guard let filter = filterValue as? CIFilter,
                              let processor = transitionFilterProcessor,
                              processor.supports(filter) else {
                            failedTransitionSourceIDs.insert(sourceID)
                            transitionFilterFailureCount += 1
                            return
                        }
                        capture = createFilteredTransitionCapture(
                            sourceLayer: state.sourceLayer,
                            targetLayer: layer,
                            filter: filter,
                            processor: processor,
                            device: device,
                            pipeline: pipeline,
                            encoder: encoder
                        )
                    } else if supportsBuiltInTransition(
                        type: state.type,
                        subtype: state.subtype
                    ) {
                        capture = createBuiltInTransitionCapture(
                            sourceLayer: state.sourceLayer,
                            targetLayer: layer,
                            device: device,
                            pipeline: pipeline,
                            encoder: encoder
                        )
                    } else {
                        failedTransitionSourceIDs.insert(sourceID)
                        transitionRenderFailureCount += 1
                        return
                    }
                    guard let capture else {
                        failedTransitionSourceIDs.insert(sourceID)
                        if state.filter != nil {
                            transitionFilterFailureCount += 1
                        }
                        return
                    }
                    transitionCaptures[sourceID] = capture
                    transitionSourceCaptureCount += 1
                    transitionTargetCaptureCount += 1
                }

                if let execution = transitionCaptures[sourceID]?.filterExecution {
                    do {
                        try execution.encode(
                            progress: Float(state.progress),
                            commandEncoder: encoder
                        )
                        transitionFilterDispatchCount += 1
                    } catch {
                        destroyTransitionCapture(for: sourceID)
                        failedTransitionSourceIDs.insert(sourceID)
                        transitionFilterFailureCount += 1
                    }
                }
            }
        }

        collect(rootLayer)

        let staleSourceIDs = transitionCaptures.keys.filter {
            !activeTransitionSourceIDs.contains($0)
        }
        for sourceID in staleSourceIDs {
            destroyTransitionCapture(for: sourceID)
        }
        failedTransitionSourceIDs = failedTransitionSourceIDs.intersection(activeTransitionSourceIDs)
    }

    private func supportsBuiltInTransition(
        type: CATransitionType,
        subtype: CATransitionSubtype?
    ) -> Bool {
        switch type {
        case .fade:
            return true
        case .moveIn, .push, .reveal:
            switch subtype {
            case .fromRight, .fromLeft, .fromTop, .fromBottom, nil:
                return true
            default:
                return false
            }
        default:
            return false
        }
    }

    private func createBuiltInTransitionCapture(
        sourceLayer: CALayer,
        targetLayer: CALayer,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        encoder: GPUCommandEncoder
    ) -> TransitionCapturePair? {
        guard let source = captureTransitionParticipant(
            sourceLayer,
            device: device,
            pipeline: pipeline,
            encoder: encoder
        ) else {
            return nil
        }
        guard let target = captureTransitionParticipant(
            targetLayer,
            device: device,
            pipeline: pipeline,
            encoder: encoder
        ) else {
            source.texture.destroy()
            return nil
        }
        return TransitionCapturePair(source: source, target: target, filterExecution: nil)
    }

    private func createFilteredTransitionCapture(
        sourceLayer: CALayer,
        targetLayer: CALayer,
        filter: CIFilter,
        processor: CIWebGPUTransitionProcessor,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        encoder: GPUCommandEncoder
    ) -> TransitionCapturePair? {
        guard let target = captureTransitionParticipant(
            targetLayer,
            device: device,
            pipeline: pipeline,
            encoder: encoder
        ) else {
            return nil
        }
        let sharedPixelSize = CGSize(width: target.pixelWidth, height: target.pixelHeight)
        guard let source = captureTransitionParticipant(
            sourceLayer,
            pixelSizeOverride: sharedPixelSize,
            device: device,
            pipeline: pipeline,
            encoder: encoder
        ) else {
            target.texture.destroy()
            return nil
        }

        do {
            let execution = try processor.makeExecution(
                filter: filter,
                sourceTexture: source.texture,
                targetTexture: target.texture,
                width: UInt32(target.pixelWidth),
                height: UInt32(target.pixelHeight)
            )
            return TransitionCapturePair(
                source: source,
                target: target,
                filterExecution: execution
            )
        } catch {
            source.texture.destroy()
            target.texture.destroy()
            return nil
        }
    }

    private func destroyTransitionCapture(for sourceID: ObjectIdentifier) {
        if let capture = transitionCaptures.removeValue(forKey: sourceID) {
            capture.filterExecution?.invalidate()
            capture.source.texture.destroy()
            capture.target.texture.destroy()
        }
        texturedTextureViewCache.removeValue(forKey: .transitionSource(sourceID))
        texturedTextureViewCache.removeValue(forKey: .transitionTarget(sourceID))
        texturedTextureViewCache.removeValue(forKey: .transitionFilter(sourceID))
        perFrameTexturedBindGroupCache.removeValue(forKey: .transitionSource(sourceID))
        perFrameTexturedBindGroupCache.removeValue(forKey: .transitionTarget(sourceID))
        perFrameTexturedBindGroupCache.removeValue(forKey: .transitionFilter(sourceID))
    }

    private func captureTransitionParticipant(
        _ layer: CALayer,
        pixelSizeOverride: CGSize? = nil,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        encoder: GPUCommandEncoder
    ) -> TransitionParticipantCapture? {
        let presentation = layer._renderTimePresentation()
        let bounds = presentation.bounds
        guard bounds.width.isFinite,
              bounds.height.isFinite,
              bounds.width > 0,
              bounds.height > 0 else {
            return nil
        }

        let pixelWidth: Int
        let pixelHeight: Int
        if let pixelSizeOverride {
            guard pixelSizeOverride.width.isFinite,
                  pixelSizeOverride.height.isFinite,
                  pixelSizeOverride.width > 0,
                  pixelSizeOverride.height > 0 else {
                return nil
            }
            pixelWidth = Int(pixelSizeOverride.width)
            pixelHeight = Int(pixelSizeOverride.height)
        } else {
            let requestedScale = presentation.contentsScale.isFinite
                ? max(presentation.contentsScale, 1)
                : 1
            let maximumDimension = CGFloat(max(1, Int(device.limits.maxTextureDimension2D)))
            let requestedWidth = bounds.width * requestedScale
            let requestedHeight = bounds.height * requestedScale
            guard requestedWidth.isFinite,
                  requestedHeight.isFinite,
                  requestedWidth > 0,
                  requestedHeight > 0 else {
                return nil
            }
            let fittingScale = min(
                1,
                maximumDimension / max(requestedWidth, requestedHeight)
            )
            pixelWidth = max(1, Int(ceil(requestedWidth * fittingScale)))
            pixelHeight = max(1, Int(ceil(requestedHeight * fittingScale)))
        }
        let pixelSize = CGSize(width: pixelWidth, height: pixelHeight)

        let texture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(
                width: UInt32(pixelWidth),
                height: UInt32(pixelHeight),
                depthOrArrayLayers: 1
            ),
            format: preferredFormat,
            usage: [.renderAttachment, .textureBinding]
        ))
        let depthTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(
                width: UInt32(pixelWidth),
                height: UInt32(pixelHeight),
                depthOrArrayLayers: 1
            ),
            format: .depth24plusStencil8,
            usage: .renderAttachment
        ))
        transientCaptureDepthTextures.append(depthTexture)
        let capturePass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: texture.createView(),
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ],
            depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                view: depthTexture.createView(),
                depthClearValue: 1,
                depthLoadOp: .clear,
                depthStoreOp: .store,
                stencilClearValue: 0,
                stencilLoadOp: .clear,
                stencilStoreOp: .store
            )
        ))
        capturePass.setPipeline(pipeline)
        capturePass.setViewport(
            x: 0,
            y: 0,
            width: Float(pixelWidth),
            height: Float(pixelHeight),
            minDepth: 0,
            maxDepth: 1
        )

        let captureProjection = Matrix4x4.orthographic(
            left: 0,
            right: Float(bounds.width),
            bottom: 0,
            top: Float(bounds.height),
            near: -1000,
            far: 1000
        )
        let previousCaptureRoot = rasterizePrerenderRootLayer
        let previousSuppression = transitionSuppressedLayer
        let previousRenderTargetSize = renderTargetSizeOverride
        let previousClipStack = clipRectStack
        let previousOpacityStack = opacityStack
        let previousMaskNestingDepth = maskNestingDepth
        let previousStencilValue = currentStencilValue

        rasterizePrerenderRootLayer = layer
        transitionSuppressedLayer = layer
        renderTargetSizeOverride = pixelSize
        clipRectStack.removeAll(keepingCapacity: true)
        opacityStack.removeAll(keepingCapacity: true)
        maskNestingDepth = 0
        currentStencilValue = 0

        renderLayer(layer, renderPass: capturePass, parentMatrix: captureProjection)

        rasterizePrerenderRootLayer = previousCaptureRoot
        transitionSuppressedLayer = previousSuppression
        renderTargetSizeOverride = previousRenderTargetSize
        clipRectStack = previousClipStack
        opacityStack = previousOpacityStack
        maskNestingDepth = previousMaskNestingDepth
        currentStencilValue = previousStencilValue
        capturePass.end()

        let compositeLayer = type(of: presentation).init(layer: presentation)
        compositeLayer.recursivelyClearDirtyAfterCommit()
        return TransitionParticipantCapture(
            texture: texture,
            compositeLayer: compositeLayer,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight
        )
    }

    /// Composites frozen source and target layer trees for a built-in or filtered transition.
    private func renderTransition(
        state: CATransitionRenderState,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let sourceID = ObjectIdentifier(state.sourceLayer)
        guard let capture = transitionCaptures[sourceID] else { return }
        let targetLayer = capture.target.compositeLayer
        let progress = CGFloat(max(0, min(1, state.progress)))
        let baseModelMatrix = targetLayer.modelMatrix(parentMatrix: parentMatrix)
        let transitionClip = calculateClipRect(layer: targetLayer, modelMatrix: baseModelMatrix)
        clipRectStack.append(currentClipRect.intersection(with: transitionClip))
        applyScissorRect(renderPass)
        defer {
            _ = clipRectStack.popLast()
            applyScissorRect(renderPass)
        }

        func renderParticipant(
            _ participant: TransitionParticipantCapture,
            cacheKey: TexturedCacheKey,
            offset: CGPoint,
            opacityMultiplier: Float
        ) {
            renderTransitionCapture(
                participant,
                cacheKey: cacheKey,
                offset: offset,
                opacityMultiplier: opacityMultiplier,
                renderPass: renderPass,
                parentMatrix: parentMatrix
            )
        }

        if let filterExecution = capture.filterExecution {
            renderParticipant(
                TransitionParticipantCapture(
                    texture: filterExecution.outputTexture,
                    compositeLayer: capture.target.compositeLayer,
                    pixelWidth: capture.target.pixelWidth,
                    pixelHeight: capture.target.pixelHeight
                ),
                cacheKey: .transitionFilter(sourceID),
                offset: .zero,
                opacityMultiplier: 1
            )
            return
        }

        switch state.type {
        case .fade:
            renderFadeTransition(
                capture,
                sourceID: sourceID,
                progress: Float(progress),
                renderPass: renderPass,
                parentMatrix: parentMatrix
            )

        case .moveIn:
            guard let direction = transitionDirection(
                subtype: state.subtype,
                bounds: targetLayer.bounds
            ) else { return }
            renderParticipant(capture.source, cacheKey: .transitionSource(sourceID), offset: .zero, opacityMultiplier: 1)
            renderParticipant(
                capture.target,
                cacheKey: .transitionTarget(sourceID),
                offset: CGPoint(x: direction.x * (1 - progress), y: direction.y * (1 - progress)),
                opacityMultiplier: 1
            )

        case .push:
            guard let direction = transitionDirection(
                subtype: state.subtype,
                bounds: targetLayer.bounds
            ) else { return }
            renderParticipant(
                capture.source,
                cacheKey: .transitionSource(sourceID),
                offset: CGPoint(x: -direction.x * progress, y: -direction.y * progress),
                opacityMultiplier: 1
            )
            renderParticipant(
                capture.target,
                cacheKey: .transitionTarget(sourceID),
                offset: CGPoint(x: direction.x * (1 - progress), y: direction.y * (1 - progress)),
                opacityMultiplier: 1
            )

        case .reveal:
            guard let direction = transitionDirection(
                subtype: state.subtype,
                bounds: targetLayer.bounds
            ) else { return }
            renderParticipant(capture.target, cacheKey: .transitionTarget(sourceID), offset: .zero, opacityMultiplier: 1)
            renderParticipant(
                capture.source,
                cacheKey: .transitionSource(sourceID),
                offset: CGPoint(x: -direction.x * progress, y: -direction.y * progress),
                opacityMultiplier: 1
            )

        default:
            return
        }
    }

    private func renderFadeTransition(
        _ capture: TransitionCapturePair,
        sourceID: ObjectIdentifier,
        progress: Float,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        guard let device,
              let transitionFadeBindGroupLayout,
              let textureSampler,
              let vertexBuffer,
              let uniformBuffer,
              let selectedPipeline = selectedTransitionFadePipeline() else {
            return
        }

        let presentation = capture.target.compositeLayer
        let bounds = presentation.bounds
        guard bounds.width > 0, bounds.height > 0 else { return }

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 0), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: SIMD4(repeating: 1)),
        ]
        guard let (vertexOffset, uniformIndex) = allocateVertices(count: vertices.count) else {
            return
        }

        var uniforms = TransitionFadeUniforms(
            mvpMatrix: presentation.modelMatrix(parentMatrix: parentMatrix) * scaleMatrix,
            colorMultiplier: currentReplicatorColor,
            opacity: currentEffectiveOpacity * presentation.opacity,
            progress: progress
        )
        let uniformOffset = UInt64(uniformIndex) * Self.alignedUniformSize
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: createFloat32Array(from: &uniforms)
        )
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: createFloat32Array(from: &vertices)
        )

        let sourceView = cachedTransitionTextureView(
            key: .transitionSource(sourceID),
            texture: capture.source.texture
        )
        let targetView = cachedTransitionTextureView(
            key: .transitionTarget(sourceID),
            texture: capture.target.texture
        )
        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: transitionFadeBindGroupLayout,
            entries: [
                GPUBindGroupEntry(
                    binding: 0,
                    resource: .bufferBinding(GPUBufferBinding(
                        buffer: uniformBuffer,
                        size: UInt64(MemoryLayout<TransitionFadeUniforms>.stride)
                    ))
                ),
                GPUBindGroupEntry(binding: 1, resource: .sampler(textureSampler)),
                GPUBindGroupEntry(binding: 2, resource: .textureView(sourceView)),
                GPUBindGroupEntry(binding: 3, resource: .textureView(targetView)),
            ]
        ))

        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
        if let pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    private func selectedTransitionFadePipeline() -> GPURenderPipeline? {
        if transformDepthNesting > 0 {
            return maskNestingDepth > 0
                ? transitionFadeDepthStencilPipeline
                : transitionFadeDepthPipeline
        }
        return maskNestingDepth > 0
            ? transitionFadeStencilPipeline
            : transitionFadePipeline
    }

    private func cachedTransitionTextureView(
        key: TexturedCacheKey,
        texture: GPUTexture
    ) -> GPUTextureView {
        if let cached = texturedTextureViewCache[key] {
            return cached
        }
        let view = texture.createView()
        texturedTextureViewCache[key] = view
        return view
    }

    private func renderTransitionCapture(
        _ capture: TransitionParticipantCapture,
        cacheKey: TexturedCacheKey,
        offset: CGPoint,
        opacityMultiplier: Float,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        guard opacityMultiplier > 0,
              let device,
              let texturedBindGroupLayout,
              let textureSampler,
              let vertexBuffer,
              let uniformBuffer else {
            return
        }

        let presentation = capture.compositeLayer
        let bounds = presentation.bounds
        guard bounds.width > 0, bounds.height > 0 else { return }

        let originalPosition = presentation.position
        presentation.position = CGPoint(
            x: originalPosition.x + offset.x,
            y: originalPosition.y + offset.y
        )
        let modelMatrix = presentation.modelMatrix(parentMatrix: parentMatrix)
        presentation.position = originalPosition

        let white = currentReplicatorColor
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
        ]
        guard let (vertexOffset, uniformIndex) = allocateVertices(count: vertices.count) else {
            return
        }

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        var uniforms = TexturedUniforms(
            mvpMatrix: modelMatrix * scaleMatrix,
            opacity: currentEffectiveOpacity * presentation.opacity * opacityMultiplier,
            cornerRadius: 0,
            layerSize: SIMD2<Float>(Float(bounds.width), Float(bounds.height)),
            cornerRadii: .zero
        )
        let uniformOffset = UInt64(uniformIndex) * Self.alignedUniformSize
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: createFloat32Array(from: &uniforms)
        )
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: createFloat32Array(from: &vertices)
        )

        let bindGroup = cachedTexturedBindGroup(
            cacheKey: cacheKey,
            gpuTexture: capture.texture,
            device: device,
            layout: texturedBindGroupLayout,
            sampler: textureSampler,
            uniformBuffer: uniformBuffer,
            uniformStride: UInt64(MemoryLayout<TexturedUniforms>.stride)
        )
        guard let selectedPipeline = selectPremultipliedTexturedPipeline() else { return }
        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        if let pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    private func transitionDirection(
        subtype: CATransitionSubtype?,
        bounds: CGRect
    ) -> CGPoint? {
        switch subtype {
        case .fromRight:
            return CGPoint(x: bounds.width, y: 0)
        case .fromTop:
            return CGPoint(x: 0, y: -bounds.height)
        case .fromBottom:
            return CGPoint(x: 0, y: bounds.height)
        case .fromLeft, nil:
            return CGPoint(x: -bounds.width, y: 0)
        default:
            return nil
        }
    }

    /// Renders the layer's backgroundColor as a rounded-rect quad.
    /// Drawn BEFORE contents per Apple's CoreAnimation spec (backgroundColor coexists with contents).
    private func renderLayerBackgroundColor(
        _ presentationLayer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        let color = replicatedColor(presentationLayer.backgroundColorComponents)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
        ]

        guard let (vertexOffset, uniformIndex) = allocateVertices(count: vertices.count) else {
            return
        }

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: Float(presentationLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height)),
            edgeAntialiasingMask: presentationLayer.edgeAntialiasingMaskValue,
            cornerRadii: presentationLayer.cornerRadiiComponents
        )

        let uniformOffset = UInt64(uniformIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Stencil-aware pipeline selection: if a mask or rounded-corner stencil is active,
        // use the stencil-testing variant so the bg respects the mask shape.
        renderPass.setPipeline(stencilAwarePipeline())
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    /// Renders the layer's border as a rounded-rect stroke quad (renderMode = 1).
    /// Drawn AFTER sublayers per Apple's CoreAnimation spec (border is topmost).
    private func renderLayerBorder(
        _ presentationLayer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        let color = replicatedColor(presentationLayer.borderColorComponents)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
        ]

        guard let (vertexOffset, uniformIndex) = allocateVertices(count: vertices.count) else {
            return
        }

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: Float(presentationLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height)),
            borderWidth: Float(presentationLayer.borderWidth),
            renderMode: 1.0,  // Border mode
            edgeAntialiasingMask: presentationLayer.edgeAntialiasingMaskValue,
            cornerRadii: presentationLayer.cornerRadiiComponents
        )

        let uniformOffset = UInt64(uniformIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Stencil-aware pipeline selection so border respects any active mask/rounded clip.
        renderPass.setPipeline(stencilAwarePipeline())
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    /// Returns the stencil-testing variant when a mask/clip stencil is active,
    /// otherwise the default pipeline with stencil compare == .always.
    /// Used by background and border rendering so solid-colored quads respect CALayer.mask
    /// and rounded-corner masksToBounds clipping.
    private func stencilAwarePipeline() -> GPURenderPipeline {
        if transformDepthNesting > 0 {
            if maskNestingDepth > 0, let depthStencilTestPipeline {
                return depthStencilTestPipeline
            }
            if let depthPipeline {
                return depthPipeline
            }
        }
        if maskNestingDepth > 0, let stencilTestPipeline = stencilTestPipeline {
            return stencilTestPipeline
        }
        return pipeline!
    }

    // MARK: - Mask Rendering (Stencil)

    /// Renders a mask layer to the stencil buffer.
    ///
    /// This writes to the stencil buffer where the mask layer has visible content.
    /// Subsequent layer rendering will only appear where the stencil value matches.
    private func renderMaskToStencil(
        _ maskLayer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        guard let device = device,
              let stencilWritePipeline = stencilWritePipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        // Increment stencil reference for nested masks
        currentStencilValue += 1

        // Switch to stencil write pipeline
        renderPass.setPipeline(stencilWritePipeline)
        renderPass.setStencilReference(currentStencilValue)

        // Render the mask layer (this writes to stencil buffer)
        let maskPresentationLayer = renderPresentation(for: maskLayer)
        let maskModelMatrix = maskPresentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Render mask layer content to stencil - use dynamic vertex allocation
        let maskColor = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: maskColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: maskColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: maskColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: maskColor),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: maskColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: maskColor),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else { return }
        let (vertexOffset, layerIndex) = allocation

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(maskPresentationLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(maskPresentationLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = maskModelMatrix * scaleMatrix

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: 1.0,  // Full opacity for stencil write
            cornerRadius: Float(maskPresentationLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(maskPresentationLayer.bounds.width), Float(maskPresentationLayer.bounds.height)),
            cornerRadii: maskPresentationLayer.cornerRadiiComponents
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: uniformOffset, data: uniformData)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch to stencil test pipeline for subsequent rendering
        if let stencilTestPipeline = stencilTestPipeline {
            renderPass.setPipeline(stencilTestPipeline)
        }
        maskNestingDepth += 1
    }

    /// Clears the stencil mask after layer rendering is complete.
    /// Supports nesting: if still inside a parent mask, restores the stencil test
    /// pipeline with the parent's stencil reference value instead of fully disabling.
    private func clearStencilMask(renderPass: GPURenderPassEncoder) {
        // Decrement stencil reference for nested masks
        if currentStencilValue > 0 {
            currentStencilValue -= 1
        }
        maskNestingDepth = max(0, maskNestingDepth - 1)

        if maskNestingDepth > 0 {
            // Still inside a parent mask — restore stencil test with parent's reference
            if let stencilTestPipeline = stencilTestPipeline {
                renderPass.setPipeline(stencilTestPipeline)
                renderPass.setStencilReference(currentStencilValue)
            }
        } else {
            // No more masks — switch back to main pipeline
            if let pipeline = pipeline {
                renderPass.setPipeline(pipeline)
                renderPass.setStencilReference(0)
            }
        }
    }

    /// Writes a rounded rectangle to the stencil buffer for masksToBounds + cornerRadius clipping.
    ///
    /// This method renders the layer's bounds as a rounded rectangle into the stencil buffer,
    /// then switches to stencil test mode so that subsequent sublayer rendering is clipped
    /// to the rounded rect region. Uses `stencilWriteRoundedPipeline` with `stencilClipFragment`
    /// shader which discards fragments outside the rounded corners.
    private func renderRoundedRectToStencil(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        device: GPUDevice
    ) {
        guard let stencilWriteRoundedPipeline = stencilWriteRoundedPipeline,
              let stencilTestPipeline = stencilTestPipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        currentStencilValue += 1

        renderPass.setPipeline(stencilWriteRoundedPipeline)
        renderPass.setStencilReference(currentStencilValue)

        // Build transform that maps the unit quad to the layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(layer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(layer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix

        // Create vertices for the quad
        let color = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else { return }
        let (vertexOffset, layerIndex) = allocation

        // Set uniforms with corner radius info for SDF calculation in the shader
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: 1.0,
            cornerRadius: Float(layer.cornerRadius),
            layerSize: SIMD2<Float>(Float(layer.bounds.width), Float(layer.bounds.height))
        )

        uniforms.cornerRadii = layer.cornerRadiiComponents

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: uniformOffset, data: uniformData)

        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch to stencil test mode for subsequent content rendering
        maskNestingDepth += 1
        renderPass.setPipeline(stencilTestPipeline)
        renderPass.setStencilReference(currentStencilValue)
    }

    // MARK: - Replicator Layer Rendering

    /// Renders the sublayers of a CAReplicatorLayer with instance transformations.
    ///
    /// Supports the following CAReplicatorLayer features:
    /// - instanceCount: Number of copies to render
    /// - instanceTransform: Cumulative transform applied to each instance
    /// - instanceDelay: Animation time offset between instances (staggered animations)
    /// - instanceColor/instanceRedOffset/etc.: Color variations per instance
    private func renderReplicatorSublayers(
        replicatorModelLayer: CAReplicatorLayer,
        replicatorLayer: CAReplicatorLayer,
        sublayers: [CALayer],
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let instanceCount = max(0, replicatorLayer.instanceCount)
        guard instanceCount > 0 else { return }
        let instanceTransform = replicatorLayer.instanceTransform
        let instanceDelay = replicatorLayer.instanceDelay

        let baseColor = replicatorLayer.instanceColor.map(rgbaComponents)
            ?? SIMD4<Float>(1, 1, 1, 1)

        // Color offsets per instance
        let redOffset = replicatorLayer.instanceRedOffset
        let greenOffset = replicatorLayer.instanceGreenOffset
        let blueOffset = replicatorLayer.instanceBlueOffset
        let alphaOffset = replicatorLayer.instanceAlphaOffset

        // Render each instance
        var cumulativeTransform = CATransform3DIdentity
        for instanceIndex in 0..<instanceCount {
            // Calculate color multiplier for this instance
            let instanceColor = SIMD4<Float>(
                clamp(baseColor.x + Float(instanceIndex) * redOffset, 0, 1),
                clamp(baseColor.y + Float(instanceIndex) * greenOffset, 0, 1),
                clamp(baseColor.z + Float(instanceIndex) * blueOffset, 0, 1),
                clamp(baseColor.w + Float(instanceIndex) * alphaOffset, 0, 1)
            )
            // Calculate instance matrix
            let instanceMatrix = parentMatrix * cumulativeTransform.matrix4x4

            // Calculate time offset for this instance
            // Positive instanceDelay means later instances start their animations later
            // So instance N's animations are evaluated at (currentTime - N * instanceDelay)
            let timeOffset = CFTimeInterval(instanceIndex) * instanceDelay

            let inheritedColor = currentReplicatorColor
            let inheritedTimeOffset = currentReplicatorTimeOffset
            replicatorColorStack.append(inheritedColor * instanceColor)
            replicatorTimeOffsetStack.append(inheritedTimeOffset + timeOffset)
            replicatorInstancePath.append(ReplicatorInstancePathComponent(
                replicator: ObjectIdentifier(replicatorModelLayer),
                instanceIndex: instanceIndex
            ))

            let orderedSublayers = sublayers.enumerated().sorted { lhs, rhs in
                let lhsZ = renderPresentation(for: lhs.element).zPosition
                let rhsZ = renderPresentation(for: rhs.element).zPosition
                return lhsZ == rhsZ ? lhs.offset < rhs.offset : lhsZ < rhsZ
            }.map(\.element)
            for sublayer in orderedSublayers {
                renderLayer(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: instanceMatrix
                )
            }

            _ = replicatorInstancePath.popLast()
            _ = replicatorTimeOffsetStack.popLast()
            _ = replicatorColorStack.popLast()

            // Apply instance transform for next iteration
            cumulativeTransform = CATransform3DConcat(cumulativeTransform, instanceTransform)
        }
    }

    /// Clamps a float value between min and max.
    private func clamp(_ value: Float, _ minVal: Float, _ maxVal: Float) -> Float {
        return min(max(value, minVal), maxVal)
    }

    private func rgbaComponents(_ color: CGColor) -> SIMD4<Float> {
        let components = color.components ?? []
        switch components.count {
        case 4...:
            return SIMD4(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
        case 3:
            return SIMD4(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                1
            )
        case 2:
            let gray = Float(components[0])
            return SIMD4(gray, gray, gray, Float(components[1]))
        case 1:
            let gray = Float(components[0])
            return SIMD4(gray, gray, gray, 1)
        default:
            return SIMD4(0, 0, 0, 0)
        }
    }

    // MARK: - Contents Layer Rendering (CGImage)

    /// Calculates the destination rectangle for contents based on contentsGravity.
    ///
    /// This determines where and how the image is positioned within the layer's bounds.
    private func calculateContentsDestRect(
        layer: CALayer,
        imageWidth: Int,
        imageHeight: Int
    ) -> CGRect {
        // Convert pixel dimensions to point dimensions using contentsScale.
        // A @2x image (200x200 pixels, contentsScale=2) displays as 100x100 points.
        let contentsScale = max(layer.contentsScale, 1.0)
        let imageSize = CGSize(
            width: CGFloat(imageWidth) / contentsScale,
            height: CGFloat(imageHeight) / contentsScale
        )
        let boundsSize = layer.bounds.size

        switch layer.contentsGravity {
        case .center:
            return CGRect(
                x: (boundsSize.width - imageSize.width) / 2,
                y: (boundsSize.height - imageSize.height) / 2,
                width: imageSize.width,
                height: imageSize.height
            )
        case .resize:
            return CGRect(origin: .zero, size: boundsSize)
        case .resizeAspect:
            let scale = min(boundsSize.width / imageSize.width, boundsSize.height / imageSize.height)
            let scaledSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
            return CGRect(
                x: (boundsSize.width - scaledSize.width) / 2,
                y: (boundsSize.height - scaledSize.height) / 2,
                width: scaledSize.width,
                height: scaledSize.height
            )
        case .resizeAspectFill:
            let scale = max(boundsSize.width / imageSize.width, boundsSize.height / imageSize.height)
            let scaledSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
            return CGRect(
                x: (boundsSize.width - scaledSize.width) / 2,
                y: (boundsSize.height - scaledSize.height) / 2,
                width: scaledSize.width,
                height: scaledSize.height
            )
        case .top:
            // Y-up: top = high Y value
            return CGRect(x: (boundsSize.width - imageSize.width) / 2, y: boundsSize.height - imageSize.height, width: imageSize.width, height: imageSize.height)
        case .bottom:
            // Y-up: bottom = Y=0
            return CGRect(x: (boundsSize.width - imageSize.width) / 2, y: 0, width: imageSize.width, height: imageSize.height)
        case .left:
            return CGRect(x: 0, y: (boundsSize.height - imageSize.height) / 2, width: imageSize.width, height: imageSize.height)
        case .right:
            return CGRect(x: boundsSize.width - imageSize.width, y: (boundsSize.height - imageSize.height) / 2, width: imageSize.width, height: imageSize.height)
        case .topLeft:
            // Y-up: top-left = (0, boundsHeight - imageHeight)
            return CGRect(x: 0, y: boundsSize.height - imageSize.height, width: imageSize.width, height: imageSize.height)
        case .topRight:
            // Y-up: top-right = (boundsWidth - imageWidth, boundsHeight - imageHeight)
            return CGRect(x: boundsSize.width - imageSize.width, y: boundsSize.height - imageSize.height, width: imageSize.width, height: imageSize.height)
        case .bottomLeft:
            // Y-up: bottom-left = origin
            return CGRect(origin: .zero, size: imageSize)
        case .bottomRight:
            // Y-up: bottom-right = (boundsWidth - imageWidth, 0)
            return CGRect(x: boundsSize.width - imageSize.width, y: 0, width: imageSize.width, height: imageSize.height)
        default:
            return CGRect(origin: .zero, size: boundsSize)
        }
    }

    /// Checks if 9-patch (contentsCenter) scaling is needed.
    ///
    /// Returns true if contentsCenter is not the default (0, 0, 1, 1),
    /// indicating that 9-slice scaling should be applied.
    private func needs9PatchScaling(_ layer: CALayer) -> Bool {
        let center = layer.contentsCenter
        return center.origin.x != 0 || center.origin.y != 0 ||
               center.size.width != 1 || center.size.height != 1
    }

    /// Renders contents using 9-patch (9-slice) scaling.
    ///
    /// The contentsCenter property defines a rectangle in unit coordinates (0-1) that
    /// specifies the center region. The image is divided into 9 regions:
    ///
    /// ```
    /// +-------+---------------+-------+
    /// |   1   |       2       |   3   |  <- corners don't stretch
    /// +-------+---------------+-------+
    /// |   4   |       5       |   6   |  <- edges stretch in one direction
    /// +-------+---------------+-------+     center stretches in both
    /// |   7   |       8       |   9   |
    /// +-------+---------------+-------+
    /// ```
    ///
    /// - Corners (1, 3, 7, 9): Fixed size, no stretching
    /// - Horizontal edges (2, 8): Stretch horizontally only
    /// - Vertical edges (4, 6): Stretch vertically only
    /// - Center (5): Stretches in both directions
    private func render9PatchContents(
        layer: CALayer,
        contents: CGImage,
        gpuTexture: GPUTexture,
        imageWidth: Int,
        imageHeight: Int,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let texturedPipeline = texturedPipeline,
              let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        let boundsWidth = layer.bounds.width
        let boundsHeight = layer.bounds.height
        guard boundsWidth > 0 && boundsHeight > 0 else { return }

        let imgW = CGFloat(imageWidth)
        let imgH = CGFloat(imageHeight)

        // contentsCenter defines the stretchable center region in normalized coordinates (0-1)
        let center = layer.contentsCenter

        // Calculate UV coordinates for the 9-patch grid
        // UV coordinates (texture space, 0-1)
        let uvLeft: Float = 0
        let uvCenterLeft: Float = Float(center.origin.x)
        let uvCenterRight: Float = Float(center.origin.x + center.size.width)
        let uvRight: Float = 1

        let uvTop: Float = 0
        let uvCenterTop: Float = Float(center.origin.y)
        let uvCenterBottom: Float = Float(center.origin.y + center.size.height)
        let uvBottom: Float = 1

        // Calculate destination positions in layer bounds
        // The corner regions maintain their pixel size from the source image
        // The center region stretches to fill the remaining space

        // Source pixel sizes for corners/edges
        let srcLeftWidth = center.origin.x * imgW
        let srcCenterWidth = center.size.width * imgW
        let srcRightWidth = (1 - center.origin.x - center.size.width) * imgW

        let srcTopHeight = center.origin.y * imgH
        let srcCenterHeight = center.size.height * imgH
        let srcBottomHeight = (1 - center.origin.y - center.size.height) * imgH

        // Calculate destination contentsScale factor
        let contentsScale = layer.contentsScale > 0 ? layer.contentsScale : 1

        // Destination sizes - corners/edges use source pixel size (accounting for contentsScale)
        let destLeftWidth = srcLeftWidth / contentsScale
        let destRightWidth = srcRightWidth / contentsScale
        let destTopHeight = srcTopHeight / contentsScale
        let destBottomHeight = srcBottomHeight / contentsScale

        // Center region fills remaining space
        let destCenterWidth = boundsWidth - destLeftWidth - destRightWidth
        let destCenterHeight = boundsHeight - destTopHeight - destBottomHeight

        // Handle case where layer is smaller than the fixed regions
        // In this case, scale down proportionally
        var scaleX: CGFloat = 1
        var scaleY: CGFloat = 1

        if destCenterWidth < 0 {
            let totalFixedWidth = destLeftWidth + destRightWidth
            scaleX = boundsWidth / totalFixedWidth
        }
        if destCenterHeight < 0 {
            let totalFixedHeight = destTopHeight + destBottomHeight
            scaleY = boundsHeight / totalFixedHeight
        }

        // Apply scale if needed
        let finalLeftWidth = destLeftWidth * scaleX
        let finalRightWidth = destRightWidth * scaleX
        let finalCenterWidth = max(0, boundsWidth - finalLeftWidth - finalRightWidth)

        let finalTopHeight = destTopHeight * scaleY
        let finalBottomHeight = destBottomHeight * scaleY
        let finalCenterHeight = max(0, boundsHeight - finalTopHeight - finalBottomHeight)

        // Position coordinates in layer bounds (normalized 0-1)
        // X coordinates (left to right)
        let posLeft: Float = 0
        let posCenterLeft: Float = Float(finalLeftWidth / boundsWidth)
        let posCenterRight: Float = Float((finalLeftWidth + finalCenterWidth) / boundsWidth)
        let posRight: Float = 1

        // Y coordinates for Y-up system (Y=0 at bottom, Y=1 at top)
        // Image top border maps to layer top, image bottom border maps to layer bottom
        let posBottom: Float = 0
        let posCenterBottom: Float = Float(finalBottomHeight / boundsHeight)
        let posCenterTop: Float = Float((boundsHeight - finalTopHeight) / boundsHeight)
        let posTop: Float = 1

        // Create scale matrix for layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(boundsWidth), 0, 0, 0),
            SIMD4<Float>(0, Float(boundsHeight), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix

        // Apple's spec: contents are only clipped to cornerRadius when masksToBounds == true.
        let effectiveCornerRadius: Float = layer.masksToBounds ? Float(layer.cornerRadius) : 0
        let effectiveCornerRadii: SIMD4<Float> = layer.masksToBounds ? layer.cornerRadiiComponents : .zero

        // Create uniforms (shared for all 9 patches)
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: effectiveCornerRadius,
            layerSize: SIMD2<Float>(Float(boundsWidth), Float(boundsHeight)),
            cornerRadii: effectiveCornerRadii,
            edgeAntialiasingMask: layer.edgeAntialiasingMaskValue
        )

        let white = currentReplicatorColor

        // Define the 9 patches as (posXMin, posXMax, posYMin, posYMax, uvXMin, uvXMax, uvYMin, uvYMax)
        // Note: In Y-up screen coordinates, posYMin < posYMax (bottom to top)
        // In texture coordinates, uvYMin < uvYMax (top to bottom, V=0 at top)
        let patches: [(Float, Float, Float, Float, Float, Float, Float, Float)] = [
            // Row 1: Top (high Y positions in screen space, low V in texture space)
            (posLeft, posCenterLeft, posCenterTop, posTop, uvLeft, uvCenterLeft, uvTop, uvCenterTop),           // 1: Top-left corner
            (posCenterLeft, posCenterRight, posCenterTop, posTop, uvCenterLeft, uvCenterRight, uvTop, uvCenterTop), // 2: Top edge
            (posCenterRight, posRight, posCenterTop, posTop, uvCenterRight, uvRight, uvTop, uvCenterTop),       // 3: Top-right corner

            // Row 2: Middle
            (posLeft, posCenterLeft, posCenterBottom, posCenterTop, uvLeft, uvCenterLeft, uvCenterTop, uvCenterBottom),           // 4: Left edge
            (posCenterLeft, posCenterRight, posCenterBottom, posCenterTop, uvCenterLeft, uvCenterRight, uvCenterTop, uvCenterBottom), // 5: Center
            (posCenterRight, posRight, posCenterBottom, posCenterTop, uvCenterRight, uvRight, uvCenterTop, uvCenterBottom),       // 6: Right edge

            // Row 3: Bottom (low Y positions in screen space, high V in texture space)
            (posLeft, posCenterLeft, posBottom, posCenterBottom, uvLeft, uvCenterLeft, uvCenterBottom, uvBottom),           // 7: Bottom-left corner
            (posCenterLeft, posCenterRight, posBottom, posCenterBottom, uvCenterLeft, uvCenterRight, uvCenterBottom, uvBottom), // 8: Bottom edge
            (posCenterRight, posRight, posBottom, posCenterBottom, uvCenterRight, uvRight, uvCenterBottom, uvBottom),       // 9: Bottom-right corner
        ]

        // Render each patch
        for (index, patch) in patches.enumerated() {
            let (pMinX, pMaxX, pMinY, pMaxY, uMinX, uMaxX, uMinY, uMaxY) = patch

            // Skip patches with zero size (can happen when center is at edges)
            if pMaxX <= pMinX || pMaxY <= pMinY {
                continue
            }

            // Create vertices for this patch (2 triangles = 6 vertices)
            // Note: Screen coordinates have Y=0 at bottom (pMinY), but texture coordinates have V=0 at top
            // So we flip V: position bottom (pMinY) uses texture bottom (uMaxY)
            var vertices: [CARendererVertex] = [
                // Triangle 1: bottom-left, bottom-right, top-left
                CARendererVertex(position: SIMD2(pMinX, pMinY), texCoord: SIMD2(uMinX, uMaxY), color: white),
                CARendererVertex(position: SIMD2(pMaxX, pMinY), texCoord: SIMD2(uMaxX, uMaxY), color: white),
                CARendererVertex(position: SIMD2(pMinX, pMaxY), texCoord: SIMD2(uMinX, uMinY), color: white),
                // Triangle 2: bottom-right, top-right, top-left
                CARendererVertex(position: SIMD2(pMaxX, pMinY), texCoord: SIMD2(uMaxX, uMaxY), color: white),
                CARendererVertex(position: SIMD2(pMaxX, pMaxY), texCoord: SIMD2(uMaxX, uMinY), color: white),
                CARendererVertex(position: SIMD2(pMinX, pMaxY), texCoord: SIMD2(uMinX, uMinY), color: white),
            ]

            guard let allocation = allocateVertices(count: vertices.count) else { continue }
            let (vertexOffset, layerIndex) = allocation

            // Write uniforms for this patch
            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            var patchUniforms = uniforms
            let uniformData = createFloat32Array(from: &patchUniforms)
            device.queue.writeBuffer(
                uniformBuffer,
                bufferOffset: uniformOffset,
                data: uniformData
            )
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            // Create bind group with texture (per-frame cached, keyed by
            // CGImage identity — the texture manager retains `contents`
            // for the cached lifetime so the key stays unique).
            let texturedBindGroup = cachedTexturedBindGroup(
                cacheKey: .image(ObjectIdentifier(contents)),
                gpuTexture: gpuTexture,
                device: device,
                layout: texturedBindGroupLayout,
                sampler: textureSampler,
                uniformBuffer: uniformBuffer,
                uniformStride: UInt64(MemoryLayout<TexturedUniforms>.stride)
            )

            // Render this patch
            if let selected = selectTexturedPipeline(for: layer) {
                renderPass.setPipeline(selected)
            }
            renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Switch back to non-textured pipeline for subsequent rendering
        if let pipeline = pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    /// Renders a layer with CGImage contents.
    private func renderContentsLayer(
        _ layer: CALayer,
        contents: CGImage,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let texturedPipeline = texturedPipeline,
              let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        // Get or create GPU texture from CGImage using the texture manager.
        // The manager retains `contents` for as long as the texture is
        // cached so `ObjectIdentifier(contents)` stays unique across the
        // dependent view / bind group caches keyed by the same identity.
        let imageWidth = contents.width
        let imageHeight = contents.height
        guard let gpuTexture = textureManager?.getOrCreateTexture(
            for: contents,
            width: imageWidth,
            height: imageHeight,
            memorySizeBytes: mipmappedRGBAByteCount(width: imageWidth, height: imageHeight),
            factory: { [weak self] in
                self?.createGPUTexture(from: contents, device: device)
            }
        ) else { return }

        // Check if 9-patch scaling is needed
        if needs9PatchScaling(layer) {
            render9PatchContents(
                layer: layer,
                contents: contents,
                gpuTexture: gpuTexture,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix
            )
            return
        }

        // Standard single-quad rendering
        // Calculate destination rectangle based on contentsGravity
        let destRect = calculateContentsDestRect(layer: layer, imageWidth: imageWidth, imageHeight: imageHeight)

        // Calculate source UV coordinates based on contentsRect
        // contentsRect is in unit coordinate space (0-1), defining which portion of the image to use
        let srcRect = layer.contentsRect
        let uvMinX = Float(srcRect.origin.x)
        let uvMinY = Float(srcRect.origin.y)
        let uvMaxX = Float(srcRect.origin.x + srcRect.size.width)
        let uvMaxY = Float(srcRect.origin.y + srcRect.size.height)

        // Calculate normalized destination positions (0-1 in layer bounds)
        let boundsWidth = layer.bounds.width
        let boundsHeight = layer.bounds.height
        guard boundsWidth > 0 && boundsHeight > 0 else { return }

        let posMinX = Float(destRect.origin.x / boundsWidth)
        let posMinY = Float(destRect.origin.y / boundsHeight)
        let posMaxX = Float((destRect.origin.x + destRect.size.width) / boundsWidth)
        let posMaxY = Float((destRect.origin.y + destRect.size.height) / boundsHeight)

        // MARK: Texture V-Coordinate Flipping for Y-up Coordinate System
        //
        // OpenCoreAnimation uses SpriteKit-compatible Y-up coordinates (origin at bottom-left),
        // but image textures store pixel data with row 0 at the TOP (standard image format).
        //
        // Coordinate systems:
        //   Screen (Y-up):              Texture:
        //   Y=height ─────────          V=0 ─────────  (top of image)
        //            │ TOP   │              │ TOP   │
        //            │       │              │       │
        //            │BOTTOM │              │BOTTOM │
        //   Y=0     ─────────              V=1 ─────────  (bottom of image)
        //   (bottom of screen)
        //
        // To display an image right-side up, we must flip the V coordinate:
        //   - Screen bottom (posMinY) → Texture bottom (uvMaxY)
        //   - Screen top (posMaxY) → Texture top (uvMinY)
        //
        // Reference: https://developer.apple.com/documentation/spritekit/about-spritekit-coordinate-systems
        let white = currentReplicatorColor
        var vertices: [CARendererVertex] = [
            // Triangle 1: bottom-left, bottom-right, top-left
            // posMinY (screen bottom) uses uvMaxY (texture bottom) - V-flipped
            CARendererVertex(position: SIMD2(posMinX, posMinY), texCoord: SIMD2(uvMinX, uvMaxY), color: white),
            CARendererVertex(position: SIMD2(posMaxX, posMinY), texCoord: SIMD2(uvMaxX, uvMaxY), color: white),
            CARendererVertex(position: SIMD2(posMinX, posMaxY), texCoord: SIMD2(uvMinX, uvMinY), color: white),
            // Triangle 2: bottom-right, top-right, top-left
            // posMaxY (screen top) uses uvMinY (texture top) - V-flipped
            CARendererVertex(position: SIMD2(posMaxX, posMinY), texCoord: SIMD2(uvMaxX, uvMaxY), color: white),
            CARendererVertex(position: SIMD2(posMaxX, posMaxY), texCoord: SIMD2(uvMaxX, uvMinY), color: white),
            CARendererVertex(position: SIMD2(posMinX, posMaxY), texCoord: SIMD2(uvMinX, uvMinY), color: white),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else { return }
        let (vertexOffset, layerIndex) = allocation

        // Create scale matrix for layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(boundsWidth), 0, 0, 0),
            SIMD4<Float>(0, Float(boundsHeight), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Apple's spec: contents are only clipped to cornerRadius when masksToBounds == true.
        // Without this gate, images would always be corner-clipped, diverging from CoreAnimation.
        let effectiveCornerRadius: Float = layer.masksToBounds ? Float(layer.cornerRadius) : 0
        let effectiveCornerRadii: SIMD4<Float> = layer.masksToBounds ? layer.cornerRadiiComponents : .zero

        // Create uniforms for textured rendering
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: effectiveCornerRadius,
            layerSize: SIMD2<Float>(Float(boundsWidth), Float(boundsHeight)),
            cornerRadii: effectiveCornerRadii,
            edgeAntialiasingMask: layer.edgeAntialiasingMaskValue
        )

        // Write uniforms
        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Create bind group with texture (per-frame cached, keyed by
        // CGImage identity — the texture manager owns `contents` for the
        // cached lifetime so the key stays unique).
        let texturedBindGroup = cachedTexturedBindGroup(
            cacheKey: .image(ObjectIdentifier(contents)),
            gpuTexture: gpuTexture,
            device: device,
            layout: texturedBindGroupLayout,
            sampler: textureSampler,
            uniformBuffer: uniformBuffer,
            uniformStride: UInt64(MemoryLayout<TexturedUniforms>.stride)
        )

        // Switch to textured pipeline and render
        if let selected = selectTexturedPipeline(for: layer) {
            renderPass.setPipeline(selected)
        }
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch back to non-textured pipeline for subsequent rendering
        if let pipeline = pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    /// Returns a textured-content bind group for the given texture, reusing
    /// one from the per-frame cache if possible.
    ///
    /// The bind group binds (uniformBuffer @ stride, textureSampler, textureView).
    /// `uniformBuffer` rotates per frame so the bind group itself cannot live
    /// past a frame, but every layer drawing the same texture in the same frame
    /// can share the same bind group. The texture view is invariant for the
    /// life of the texture and is cached across frames in
    /// `texturedTextureViewCache`.
    ///
    /// - Parameter cacheKey: The typed identity used to key both the
    ///   persistent `GPUTextureView` cache and the per-frame
    ///   `GPUBindGroup` cache. Use `.image(OID(cgImage))` for content
    ///   textures (the `GPUTextureManager` retains the CGImage and
    ///   notifies eviction via `onEvict`), or
    ///   `.rasterizedLayer(renderKey, purpose)` for
    ///   layer-owned rasterization composites. The case discriminator
    ///   keeps the two identity namespaces disjoint so a freed layer's
    ///   reused address cannot alias a fresh CGImage.
    private func cachedTexturedBindGroup(
        cacheKey: TexturedCacheKey,
        gpuTexture: GPUTexture,
        device: GPUDevice,
        layout: GPUBindGroupLayout,
        sampler: GPUSampler,
        uniformBuffer: GPUBuffer,
        uniformStride: UInt64
    ) -> GPUBindGroup {
        if let cached = perFrameTexturedBindGroupCache[cacheKey] {
            return cached
        }
        let view: GPUTextureView
        if let cachedView = texturedTextureViewCache[cacheKey] {
            view = cachedView
        } else {
            view = gpuTexture.createView()
            texturedTextureViewCache[cacheKey] = view
        }
        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: layout,
            entries: [
                GPUBindGroupEntry(
                    binding: 0,
                    resource: .bufferBinding(GPUBufferBinding(
                        buffer: uniformBuffer,
                        size: uniformStride
                    ))
                ),
                GPUBindGroupEntry(
                    binding: 1,
                    resource: .sampler(sampler)
                ),
                GPUBindGroupEntry(
                    binding: 2,
                    resource: .textureView(view)
                )
            ]
        ))
        perFrameTexturedBindGroupCache[cacheKey] = bindGroup
        return bindGroup
    }

    /// Creates a GPU texture from a CGImage.
    private func createGPUTexture(from cgImage: CGImage, device: GPUDevice) -> GPUTexture? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0 && height > 0 else { return nil }

        // Get RGBA data
        guard let rgbaData = getRGBAData(from: cgImage) else { return nil }

        var mipLevelCount: UInt32 = 1
        var mipWidth = width
        var mipHeight = height
        while mipWidth > 1 || mipHeight > 1 {
            mipWidth = max(1, mipWidth / 2)
            mipHeight = max(1, mipHeight / 2)
            mipLevelCount += 1
        }

        // Every cached image receives a complete mip chain. Linear samplers clamp
        // to level zero, while CAEmitterCell.trilinear selects between these levels.
        let textureDescriptor = GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            mipLevelCount: mipLevelCount,
            format: .rgba8unorm,
            usage: [.textureBinding, .copyDst, .renderAttachment]
        )

        let texture = device.createTexture(descriptor: textureDescriptor)

        var levelData = rgbaData
        mipWidth = width
        mipHeight = height
        for level in 0..<mipLevelCount {
            device.queue.writeTexture(
                destination: GPUImageCopyTexture(texture: texture, mipLevel: level),
                data: createUint8Array(from: levelData),
                dataLayout: GPUImageDataLayout(
                    offset: 0,
                    bytesPerRow: UInt32(mipWidth * 4),
                    rowsPerImage: UInt32(mipHeight)
                ),
                size: GPUExtent3D(width: UInt32(mipWidth), height: UInt32(mipHeight))
            )
            guard level + 1 < mipLevelCount else { continue }
            levelData = downsampleRGBA8(
                levelData,
                width: mipWidth,
                height: mipHeight
            )
            mipWidth = max(1, mipWidth / 2)
            mipHeight = max(1, mipHeight / 2)
        }

        return texture
    }

    private func mipmappedRGBAByteCount(width: Int, height: Int) -> UInt64 {
        var levelWidth = max(1, width)
        var levelHeight = max(1, height)
        var byteCount: UInt64 = 0
        while true {
            byteCount += UInt64(levelWidth * levelHeight * 4)
            guard levelWidth > 1 || levelHeight > 1 else { return byteCount }
            levelWidth = max(1, levelWidth / 2)
            levelHeight = max(1, levelHeight / 2)
        }
    }

    /// Builds the next RGBA8 mip level with a box filter.
    private func downsampleRGBA8(_ source: Data, width: Int, height: Int) -> Data {
        let destinationWidth = max(1, width / 2)
        let destinationHeight = max(1, height / 2)
        var destination = Data(count: destinationWidth * destinationHeight * 4)
        source.withUnsafeBytes { sourceBytes in
            destination.withUnsafeMutableBytes { destinationBytes in
                guard let sourceBase = sourceBytes.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let destinationBase = destinationBytes.baseAddress?.assumingMemoryBound(
                        to: UInt8.self
                      ) else {
                    return
                }
                for destinationY in 0..<destinationHeight {
                    for destinationX in 0..<destinationWidth {
                        let sourceMinX = destinationX * width / destinationWidth
                        let sourceMinY = destinationY * height / destinationHeight
                        let sourceMaxX = max(
                            sourceMinX + 1,
                            (destinationX + 1) * width / destinationWidth
                        )
                        let sourceMaxY = max(
                            sourceMinY + 1,
                            (destinationY + 1) * height / destinationHeight
                        )
                        let destinationOffset = (destinationY * destinationWidth + destinationX) * 4
                        for component in 0..<4 {
                            var sum: UInt32 = 0
                            var sampleCount: UInt32 = 0
                            for sourceY in sourceMinY..<sourceMaxY {
                                for sourceX in sourceMinX..<sourceMaxX {
                                    sum += UInt32(
                                        sourceBase[(sourceY * width + sourceX) * 4 + component]
                                    )
                                    sampleCount += 1
                                }
                            }
                            destinationBase[destinationOffset + component] = UInt8(
                                (sum + sampleCount / 2) / sampleCount
                            )
                        }
                    }
                }
            }
        }
        return destination
    }

    /// Converts CGImage data to RGBA format.
    private func getRGBAData(from cgImage: CGImage) -> Data? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0 && height > 0 else { return nil }

        // Try to get source data from cgImage.data or dataProvider
        let sourceData: Data
        if let directData = cgImage.data {
            sourceData = directData
        } else if let provider = cgImage.dataProvider, let providerData = provider.data {
            sourceData = providerData
        } else {
            print("CAWebGPURenderer: ERROR - CGImage has no data (width=\(width), height=\(height))")
            return nil
        }

        let bytesPerPixel = cgImage.bitsPerPixel / 8
        let sourceBytesPerRow = cgImage.bytesPerRow
        let destBytesPerRow = width * 4
        let totalBytes = destBytesPerRow * height

        // If already RGBA8, use directly
        if bytesPerPixel == 4 && sourceBytesPerRow == destBytesPerRow {
            let isRGBA = cgImage.alphaInfo == .premultipliedLast ||
                        cgImage.alphaInfo == .last ||
                        cgImage.alphaInfo == .noneSkipLast
            if isRGBA {
                return sourceData
            }
        }

        // Need to convert
        var destData = Data(count: totalBytes)

        sourceData.withUnsafeBytes { sourceBuffer in
            destData.withUnsafeMutableBytes { destBuffer in
                guard let sourceBase = sourceBuffer.baseAddress,
                      let destBase = destBuffer.baseAddress else { return }

                for y in 0..<height {
                    for x in 0..<width {
                        let destOffset = y * destBytesPerRow + x * 4
                        let d = destBase.advanced(by: destOffset).assumingMemoryBound(to: UInt8.self)

                        if bytesPerPixel == 4 {
                            let sourceOffset = y * sourceBytesPerRow + x * 4
                            let s = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)

                            // Handle different alpha positions
                            switch cgImage.alphaInfo {
                            case .premultipliedFirst, .first:
                                // ARGB -> RGBA
                                d[0] = s[1]; d[1] = s[2]; d[2] = s[3]; d[3] = s[0]
                            case .noneSkipFirst:
                                // xRGB -> RGBA
                                d[0] = s[1]; d[1] = s[2]; d[2] = s[3]; d[3] = 255
                            case .noneSkipLast:
                                // RGBx -> RGBA
                                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255
                            default:
                                // Assume RGBA
                                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3]
                            }
                        } else if bytesPerPixel == 3 {
                            let sourceOffset = y * sourceBytesPerRow + x * 3
                            let s = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)
                            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255
                        } else if bytesPerPixel == 1 {
                            let sourceOffset = y * sourceBytesPerRow + x
                            let gray = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)[0]
                            d[0] = gray; d[1] = gray; d[2] = gray; d[3] = 255
                        }
                    }
                }
            }
        }

        return destData
    }

    /// Creates a JavaScript Uint8Array from Data.
    ///
    /// Bulk-copies via `JSTypedArray<UInt8>(buffer:)` so a 32×32 RGBA texture
    /// upload is one JS round-trip rather than 4096.
    private func createUint8Array(from data: Data) -> JSObject {
        return data.withUnsafeBytes { rawBytes -> JSObject in
            let typed = rawBytes.bindMemory(to: UInt8.self)
            let buffer = UnsafeBufferPointer<UInt8>(start: typed.baseAddress, count: data.count)
            return JSTypedArray<UInt8>(buffer: buffer).jsObject
        }
    }

    /// Clears the texture cache for a specific CGImage.
    ///
    /// The texture manager fires `onEvict(cgImage)` when removing the
    /// entry, which propagates to `texturedTextureViewCache` /
    /// `perFrameTexturedBindGroupCache` automatically.
    public func removeTexture(for cgImage: CGImage) {
        textureManager?.removeTexture(for: cgImage)
    }

    /// Clears all cached textures.
    ///
    /// `textureManager?.clearAll()` fires `onEvict` for every entry,
    /// draining `texturedTextureViewCache` /
    /// `perFrameTexturedBindGroupCache` along the way. The explicit
    /// `removeAll()` calls below are defensive for entries that were
    /// inserted via call paths that don't go through the texture
    /// manager (e.g. layer-keyed rasterization composites).
    public func clearTextureCache() {
        textureManager?.clearAll()
        rasterizationCache?.removeAll()
        prerasterizedTextures.removeAll(keepingCapacity: true)
        for resources in shadowLayerResources.values {
            resources.destroy()
        }
        shadowLayerResources.removeAll(keepingCapacity: true)
        prerenderedShadows.removeAll(keepingCapacity: true)
        for resources in filterLayerResources.values {
            resources.destroy()
        }
        filterLayerResources.removeAll(keepingCapacity: true)
        prerenderedFilters.removeAll(keepingCapacity: true)
        textTextureCache.removeAll()
        textTextureAccessOrder.removeAll()
        texturedTextureViewCache.removeAll(keepingCapacity: true)
        perFrameTexturedBindGroupCache.removeAll(keepingCapacity: true)
    }

    // MARK: - Text Layer Rendering

    /// Cache for text textures to avoid recreating them every frame.
    private var textTextureCache: [String: GPUTexture] = [:]

    /// LRU order for text textures. Dynamic labels can otherwise create one
    /// GPU texture per distinct string for the lifetime of the renderer.
    private var textTextureAccessOrder: [String] = []

    private let maxTextTextureCacheEntries = 128

    private func touchTextTexture(_ key: String) {
        textTextureAccessOrder.removeAll { $0 == key }
        textTextureAccessOrder.append(key)
    }

    private func cacheTextTexture(_ texture: GPUTexture, for key: String) {
        textTextureCache[key] = texture
        touchTextTexture(key)
        while textTextureCache.count > maxTextTextureCacheEntries,
              let oldest = textTextureAccessOrder.first {
            textTextureAccessOrder.removeFirst()
            textTextureCache.removeValue(forKey: oldest)
        }
    }

    /// Renders a CATextLayer using Canvas2D for text rasterization and texture-based rendering.
    ///
    /// This method uses an offscreen Canvas2D to render the text, creates a WebGPU texture
    /// from the canvas data, and displays it using the textured pipeline.
    private func renderTextLayer(
        _ textLayer: CATextLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let string = textLayer.string else {
            return
        }
        guard let texturedPipeline = texturedPipeline,
              let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let pipeline = pipeline else {
            return
        }

        let text: String
        if let str = string as? String {
            text = str
        } else {
            text = String(describing: string)
        }

        guard !text.isEmpty else {
            return
        }
        guard textLayer.fontSize.isFinite, textLayer.fontSize > 0 else {
            return
        }

        // Determine text size: use bounds if set, otherwise measure with Canvas2D
        let width: Int
        let height: Int
        if textLayer.bounds.width > 0 && textLayer.bounds.height > 0 {
            width = Int(textLayer.bounds.width)
            height = Int(textLayer.bounds.height)
        } else {
            // Measure text size using Canvas2D
            let measuredSize = measureTextSize(
                text: text,
                font: textLayer.font as? String,
                fontSize: textLayer.fontSize,
                isWrapped: textLayer.isWrapped,
                maxWidth: textLayer.bounds.width > 0 ? textLayer.bounds.width : nil
            )
            width = Int(ceil(measuredSize.width))
            height = Int(ceil(measuredSize.height))
        }

        guard width > 0 && height > 0 else {
            return
        }

        // Create cache key based on text content and properties.
        // Includes font fingerprint and foreground color so that two layers
        // with the same text but different font / color do not collide.
        let cacheKey = textCacheKey(
            text: text,
            width: width,
            height: height,
            layer: textLayer
        )

        // Check cache first
        let gpuTexture: GPUTexture
        if let cached = textTextureCache[cacheKey] {
            touchTextTexture(cacheKey)
            gpuTexture = cached
        } else {
            // Create offscreen canvas for text rendering
            let document = JSObject.global.document
            let offscreenCanvas = document.createElement("canvas")
            offscreenCanvas.width = .number(Double(width))
            offscreenCanvas.height = .number(Double(height))

            guard let ctx = offscreenCanvas.getContext("2d").object else { return }

            // Clear canvas with background color if set
            if let bgColor = textLayer.backgroundColor {
                let components = bgColor.components ?? [0, 0, 0, 1]
                let r = Int((components.count > 0 ? components[0] : 0) * 255)
                let g = Int((components.count > 1 ? components[1] : 0) * 255)
                let b = Int((components.count > 2 ? components[2] : 0) * 255)
                let a = components.count > 3 ? components[3] : 1.0
                ctx.fillStyle = .string("rgba(\(r),\(g),\(b),\(a))")
                _ = ctx.fillRect!(0, 0, width, height)
            } else {
                _ = ctx.clearRect!(0, 0, width, height)
            }

            // Set font
            let fontName: String
            if let font = textLayer.font as? String {
                fontName = font
            } else {
                fontName = "sans-serif"
            }
            ctx.font = .string("\(Int(textLayer.fontSize))px \(fontName)")

            // Set text color
            if let fgColor = textLayer.foregroundColor {
                let components = fgColor.components ?? [0, 0, 0, 1]
                let r = Int((components.count > 0 ? components[0] : 0) * 255)
                let g = Int((components.count > 1 ? components[1] : 0) * 255)
                let b = Int((components.count > 2 ? components[2] : 0) * 255)
                let a = components.count > 3 ? components[3] : 1.0
                ctx.fillStyle = .string("rgba(\(r),\(g),\(b),\(a))")
            } else {
                ctx.fillStyle = .string("rgba(255,255,255,1)")
            }

            // Set text alignment
            switch textLayer.alignmentMode {
            case .left:
                ctx.textAlign = .string("left")
            case .right:
                ctx.textAlign = .string("right")
            case .center:
                ctx.textAlign = .string("center")
            case .justified, .natural:
                ctx.textAlign = .string("start")
            default:
                ctx.textAlign = .string("start")
            }

            ctx.textBaseline = .string("top")

            // Calculate text position based on alignment
            let x: Double
            switch textLayer.alignmentMode {
            case .center:
                x = Double(width) / 2
            case .right:
                x = Double(width)
            default:
                x = 0
            }

            // Draw measured single-line or multiline text into the layer texture.
            if textLayer.isWrapped || CATextLayoutEngine.containsParagraphBreak(text) {
                drawMultilineText(
                    ctx: ctx,
                    text: text,
                    x: x,
                    y: 0,
                    maxWidth: Double(width),
                    maxHeight: Double(height),
                    lineHeight: textLayer.fontSize * 1.2,
                    truncationMode: textLayer.truncationMode,
                    alignmentMode: textLayer.alignmentMode,
                    wrapsToWidth: textLayer.isWrapped
                )
            } else {
                let displayedText = CATextLayoutEngine.truncatedText(
                    text,
                    mode: textLayer.truncationMode,
                    maximumWidth: CGFloat(width),
                    measure: { candidate in
                        CGFloat(self.measureWidth(ctx: ctx, candidate))
                    }
                )
                _ = ctx.fillText!(displayedText, x, Double(textLayer.fontSize * 0.1))
            }

            // Get image data from canvas
            guard let imageData = ctx.getImageData!(0, 0, width, height).object else { return }
            guard let dataArray = imageData.data.object else { return }

            // Create WebGPU texture
            let textureDescriptor = GPUTextureDescriptor(
                size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
                format: .rgba8unorm,
                usage: [.textureBinding, .copyDst, .renderAttachment]
            )

            let texture = device.createTexture(descriptor: textureDescriptor)

            // Copy image data to texture
            device.queue.writeTexture(
                destination: GPUImageCopyTexture(texture: texture),
                data: dataArray,
                dataLayout: GPUImageDataLayout(
                    offset: 0,
                    bytesPerRow: UInt32(width * 4),
                    rowsPerImage: UInt32(height)
                ),
                size: GPUExtent3D(width: UInt32(width), height: UInt32(height))
            )

            cacheTextTexture(texture, for: cacheKey)
            gpuTexture = texture
        }

        // Create texture view
        let textureView = gpuTexture.createView()

        // Create bind group with the texture
        let texturedBindGroup = device.createBindGroup(
            descriptor: GPUBindGroupDescriptor(
                layout: texturedBindGroupLayout,
                entries: [
                    GPUBindGroupEntry(binding: 0, resource: .buffer(uniformBuffer, offset: 0, size: UInt64(MemoryLayout<TexturedUniforms>.stride))),
                    GPUBindGroupEntry(binding: 1, resource: .sampler(textureSampler)),
                    GPUBindGroupEntry(binding: 2, resource: .textureView(textureView))
                ]
            )
        )

        // MARK: Text Texture V-Coordinate Flipping for Y-up Coordinate System
        //
        // OpenCoreAnimation uses SpriteKit-compatible Y-up coordinates (origin at bottom-left),
        // but HTML Canvas (used for text rendering) uses Y-down coordinates (origin at top-left).
        //
        // Coordinate systems:
        //   Screen (Y-up):              Canvas/Texture:
        //   Y=1   ─────────             V=0 ─────────  (Canvas Y=0, top)
        //         │ TOP   │                 │ TOP   │
        //         │ text  │                 │ text  │
        //         │BOTTOM │                 │BOTTOM │
        //   Y=0   ─────────             V=1 ─────────  (Canvas Y=height, bottom)
        //   (bottom of screen)
        //
        // To display text right-side up, we must flip the V coordinate:
        //   - Screen bottom (Y=0) → Texture bottom (V=1)
        //   - Screen top (Y=1) → Texture top (V=0)
        //
        // Reference: https://developer.apple.com/documentation/spritekit/about-spritekit-coordinate-systems
        let white = currentReplicatorColor
        var vertices: [CARendererVertex] = [
            // Triangle 1: bottom-left, bottom-right, top-left
            // Y=0 (screen bottom) uses V=1 (texture bottom) - V-flipped
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
            // Triangle 2: bottom-right, top-right, top-left
            // Y=1 (screen top) uses V=0 (texture top) - V-flipped
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else { return }
        let (vertexOffset, layerIndex) = allocation

        // Setup matrices using measured/actual text size
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(width), 0, 0, 0),
            SIMD4<Float>(0, Float(height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Apple's spec: text contents are only clipped to cornerRadius when masksToBounds == true.
        let effectiveCornerRadius: Float = textLayer.masksToBounds ? Float(textLayer.cornerRadius) : 0
        let effectiveCornerRadii: SIMD4<Float> = textLayer.masksToBounds ? textLayer.cornerRadiiComponents : .zero

        // Update uniforms
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: effectiveCornerRadius,
            layerSize: SIMD2<Float>(Float(width), Float(height)),
            cornerRadii: effectiveCornerRadii,
            edgeAntialiasingMask: textLayer.edgeAntialiasingMaskValue
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Switch to textured pipeline and draw
        if let selected = selectTexturedPipeline(for: textLayer) {
            renderPass.setPipeline(selected)
        }
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch back to standard pipeline
        renderPass.setPipeline(pipeline)
    }

    /// Builds a cache key for a CATextLayer's rendered text texture.
    ///
    /// The key must vary on anything that changes the rendered pixels:
    /// text content, layer size, font (family / size), alignment, and
    /// foreground color. Earlier versions omitted font and color, so two
    /// layers with the same string but different styling silently shared
    /// the same texture.
    private func textCacheKey(
        text: String,
        width: Int,
        height: Int,
        layer: CATextLayer
    ) -> String {
        let fontFingerprint: String
        if let fontString = layer.font as? String {
            fontFingerprint = fontString
        } else if let anyFont = layer.font {
            // Any other font representation — rely on its reflection / description.
            // This is deterministic for the same object reference during a frame.
            fontFingerprint = String(describing: anyFont)
        } else {
            fontFingerprint = "sans-serif"
        }

        let colorHex: String
        if let fg = layer.foregroundColor, let components = fg.components {
            let r = Int((components.indices.contains(0) ? components[0] : 0) * 255) & 0xFF
            let g = Int((components.indices.contains(1) ? components[1] : 0) * 255) & 0xFF
            let b = Int((components.indices.contains(2) ? components[2] : 0) * 255) & 0xFF
            let a = Int((components.indices.contains(3) ? components[3] : 1) * 255) & 0xFF
            colorHex = String(format: "%02X%02X%02X%02X", r, g, b, a)
        } else {
            colorHex = "FFFFFFFF"
        }

        return "\(text)_\(width)x\(height)_\(layer.fontSize)_\(layer.alignmentMode.rawValue)_\(layer.truncationMode.rawValue)_\(fontFingerprint)_\(colorHex)_\(layer.isWrapped ? "w" : "n")"
    }

    /// Canvas2D text width for a string.
    private func measureWidth(ctx: JSObject, _ s: String) -> Double {
        let metrics = ctx.measureText!(s)
        return metrics.width.number ?? 0
    }

    /// Draws paragraph and width-driven line breaks, truncating visible lines when requested.
    private func drawMultilineText(
        ctx: JSObject,
        text: String,
        x: Double,
        y: Double,
        maxWidth: Double,
        maxHeight: Double,
        lineHeight: CGFloat,
        truncationMode: CATextLayerTruncationMode,
        alignmentMode: CATextLayerAlignmentMode,
        wrapsToWidth: Bool
    ) {
        guard lineHeight.isFinite, lineHeight > 0 else { return }
        let lines = CATextLayoutEngine.wrappedLines(
            text,
            maximumWidth: CGFloat(maxWidth),
            wrapsToWidth: wrapsToWidth,
            measure: { candidate in CGFloat(self.measureWidth(ctx: ctx, candidate)) }
        )

        let visibleLineCount = max(1, Int(floor(maxHeight / Double(lineHeight))))
        var displayedLines = lines
        let shouldTruncate = truncationMode == .start
            || truncationMode == .middle
            || truncationMode == .end
        if shouldTruncate, !wrapsToWidth {
            displayedLines = lines.map { line in
                CATextLayoutLine(
                    text: CATextLayoutEngine.truncatedText(
                        line.text,
                        mode: truncationMode,
                        maximumWidth: CGFloat(maxWidth),
                        measure: { candidate in
                            CGFloat(self.measureWidth(ctx: ctx, candidate))
                        }
                    ),
                    separatorAfter: line.separatorAfter,
                    isParagraphFinal: line.isParagraphFinal
                )
            }
        }
        if shouldTruncate, lines.count > visibleLineCount {
            displayedLines = Array(lines.prefix(visibleLineCount))
            let overflowText = CATextLayoutEngine.joinedText(
                lines[(visibleLineCount - 1)...]
            )
            displayedLines[visibleLineCount - 1] = CATextLayoutLine(
                text: CATextLayoutEngine.truncatedText(
                    overflowText,
                    mode: truncationMode,
                    maximumWidth: CGFloat(maxWidth),
                    measure: { candidate in
                        CGFloat(self.measureWidth(ctx: ctx, candidate))
                    }
                ),
                separatorAfter: "",
                isParagraphFinal: true
            )
        }

        for (index, displayedLine) in displayedLines.enumerated() {
            let lineY = y + Double(index) * Double(lineHeight)
            if alignmentMode == .justified, !displayedLine.isParagraphFinal {
                drawJustifiedText(
                    ctx: ctx,
                    text: displayedLine.text,
                    y: lineY,
                    maximumWidth: maxWidth
                )
            } else {
                _ = ctx.fillText!(displayedLine.text, x, lineY)
            }
        }
    }

    private func drawJustifiedText(
        ctx: JSObject,
        text: String,
        y: Double,
        maximumWidth: Double
    ) {
        let segments = CATextLayoutEngine.justificationSegments(for: text)
        guard segments.count > 1 else {
            _ = ctx.fillText!(text, 0, y)
            return
        }
        let contentWidth = segments.reduce(0) { partialResult, segment in
            partialResult + measureWidth(ctx: ctx, segment)
        }
        let spacing = max(0, maximumWidth - contentWidth) / Double(segments.count - 1)
        var cursor = 0.0
        for segment in segments {
            _ = ctx.fillText!(segment, cursor, y)
            cursor += measureWidth(ctx: ctx, segment) + spacing
        }
    }

    /// Measures text size using Canvas2D measureText API.
    /// - Parameters:
    ///   - text: The text to measure
    ///   - font: Font family name (nil for system default)
    ///   - fontSize: Font size in points
    ///   - isWrapped: Whether text should wrap
    ///   - maxWidth: Maximum width for wrapping (nil for single line)
    /// - Returns: The measured size
    private func measureTextSize(
        text: String,
        font: String?,
        fontSize: CGFloat,
        isWrapped: Bool,
        maxWidth: CGFloat?
    ) -> CGSize {
        // Use shared measurement canvas or create one
        let document = JSObject.global.document
        let measureCanvas = document.createElement("canvas")
        guard let ctx = measureCanvas.getContext("2d").object else {
            // Fallback to estimation
            let estimatedWidth = CGFloat(text.count) * fontSize * 0.6
            let estimatedHeight = fontSize * 1.2
            return CGSize(width: estimatedWidth, height: estimatedHeight)
        }

        // Set font
        let fontFamily = font ?? "sans-serif"
        ctx.font = .string("\(Int(fontSize))px \(fontFamily)")

        // Measure text
        let metrics = ctx.measureText!(text)
        let measuredWidth = metrics.width.number ?? Double(text.count) * Double(fontSize) * 0.6

        // Get height from font metrics if available, otherwise estimate
        let ascent = metrics.actualBoundingBoxAscent.number ?? Double(fontSize) * 0.8
        let descent = metrics.actualBoundingBoxDescent.number ?? Double(fontSize) * 0.2
        let lineHeight = ascent + descent

        if isWrapped, let maxWidth {
            let lines = CATextLayoutEngine.wrappedLines(
                text,
                maximumWidth: maxWidth,
                measure: { candidate in CGFloat(self.measureWidth(ctx: ctx, candidate)) }
            )
            return CGSize(
                width: maxWidth,
                height: fontSize * 1.2 * CGFloat(lines.count)
            )
        } else if CATextLayoutEngine.containsParagraphBreak(text) {
            let lines = CATextLayoutEngine.wrappedLines(
                text,
                maximumWidth: maxWidth ?? .greatestFiniteMagnitude,
                wrapsToWidth: false,
                measure: { candidate in CGFloat(self.measureWidth(ctx: ctx, candidate)) }
            )
            let widestLine = lines.reduce(CGFloat.zero) { width, line in
                max(width, CGFloat(measureWidth(ctx: ctx, line.text)))
            }
            return CGSize(
                width: maxWidth ?? widestLine + fontSize * 0.1,
                height: fontSize * 1.2 * CGFloat(lines.count)
            )
        } else if let maxWidth = maxWidth, CGFloat(measuredWidth) > maxWidth {
            // Not wrapped but has maxWidth constraint - text will be clipped
            // Return maxWidth as the size to match the canvas dimensions
            return CGSize(
                width: maxWidth,
                height: CGFloat(lineHeight) * 1.2
            )
        } else {
            // Single line, no constraints - add some padding
            return CGSize(
                width: CGFloat(measuredWidth) + fontSize * 0.1,
                height: CGFloat(lineHeight) * 1.2
            )
        }
    }

    // MARK: - Shape Layer Rendering

    /// Flattens a CGPath into an array of polylines (arrays of points).
    /// Each subpath becomes a separate polyline.
    private func flattenPath(_ path: CGPath, flatness: CGFloat = 0.5) -> [[CGPoint]] {
        var polylines: [[CGPoint]] = []
        var currentPolyline: [CGPoint] = []
        var currentPoint = CGPoint.zero
        var subpathStart = CGPoint.zero

        path.applyWithBlock { elementPtr in
            let element = elementPtr.pointee
            switch element.type {
            case .moveToPoint:
                if !currentPolyline.isEmpty {
                    polylines.append(currentPolyline)
                }
                currentPolyline = []
                let point = element.points![0]
                currentPolyline.append(point)
                currentPoint = point
                subpathStart = point

            case .addLineToPoint:
                let point = element.points![0]
                currentPolyline.append(point)
                currentPoint = point

            case .addQuadCurveToPoint:
                let control = element.points![0]
                let end = element.points![1]
                // Flatten quadratic bezier
                self.flattenQuadBezier(from: currentPoint, control: control, to: end,
                                       flatness: flatness, into: &currentPolyline)
                currentPoint = end

            case .addCurveToPoint:
                let control1 = element.points![0]
                let control2 = element.points![1]
                let end = element.points![2]
                // Flatten cubic bezier
                self.flattenCubicBezier(from: currentPoint, control1: control1,
                                        control2: control2, to: end,
                                        flatness: flatness, into: &currentPolyline)
                currentPoint = end

            case .closeSubpath:
                if !currentPolyline.isEmpty && currentPolyline.first != currentPolyline.last {
                    currentPolyline.append(subpathStart)
                }
                currentPoint = subpathStart

            @unknown default:
                break
            }
        }

        if !currentPolyline.isEmpty {
            polylines.append(currentPolyline)
        }

        return polylines
    }

    /// Flattens a quadratic bezier curve into line segments.
    private func flattenQuadBezier(from p0: CGPoint, control p1: CGPoint, to p2: CGPoint,
                                   flatness: CGFloat, into polyline: inout [CGPoint]) {
        // Simple recursive subdivision
        let midX = (p0.x + 2 * p1.x + p2.x) / 4
        let midY = (p0.y + 2 * p1.y + p2.y) / 4
        let dx = (p0.x + p2.x) / 2 - midX
        let dy = (p0.y + p2.y) / 2 - midY

        if dx * dx + dy * dy <= flatness * flatness {
            polyline.append(p2)
        } else {
            let p01 = CGPoint(x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2)
            let p12 = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
            let mid = CGPoint(x: (p01.x + p12.x) / 2, y: (p01.y + p12.y) / 2)
            flattenQuadBezier(from: p0, control: p01, to: mid, flatness: flatness, into: &polyline)
            flattenQuadBezier(from: mid, control: p12, to: p2, flatness: flatness, into: &polyline)
        }
    }

    /// Flattens a cubic bezier curve into line segments.
    private func flattenCubicBezier(from p0: CGPoint, control1 p1: CGPoint,
                                    control2 p2: CGPoint, to p3: CGPoint,
                                    flatness: CGFloat, into polyline: inout [CGPoint]) {
        // Check if curve is flat enough
        let dx1 = p1.x - p0.x
        let dy1 = p1.y - p0.y
        let dx2 = p2.x - p3.x
        let dy2 = p2.y - p3.y

        let d = sqrt(max(dx1 * dx1 + dy1 * dy1, dx2 * dx2 + dy2 * dy2))

        if d <= flatness {
            polyline.append(p3)
        } else {
            // Subdivide
            let p01 = CGPoint(x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2)
            let p12 = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
            let p23 = CGPoint(x: (p2.x + p3.x) / 2, y: (p2.y + p3.y) / 2)
            let p012 = CGPoint(x: (p01.x + p12.x) / 2, y: (p01.y + p12.y) / 2)
            let p123 = CGPoint(x: (p12.x + p23.x) / 2, y: (p12.y + p23.y) / 2)
            let mid = CGPoint(x: (p012.x + p123.x) / 2, y: (p012.y + p123.y) / 2)

            flattenCubicBezier(from: p0, control1: p01, control2: p012, to: mid,
                               flatness: flatness, into: &polyline)
            flattenCubicBezier(from: mid, control1: p123, control2: p23, to: p3,
                               flatness: flatness, into: &polyline)
        }
    }

    /// Triangulates a simple polygon using ear-clipping algorithm.
    /// Returns array of vertex indices forming triangles.
    private func triangulatePolygon(_ polygon: [CGPoint]) -> [Int] {
        guard polygon.count >= 3 else { return [] }

        // Remove duplicate last point if closed
        var points = polygon
        if let first = points.first, let last = points.last,
           first.x == last.x && first.y == last.y && points.count > 1 {
            points.removeLast()
        }

        guard points.count >= 3 else { return [] }

        var indices: [Int] = []
        var remaining = Array(0..<points.count)

        // Ensure polygon is counter-clockwise
        let area = signedArea(points)
        if area < 0 {
            remaining.reverse()
        }

        var safetyCounter = remaining.count * remaining.count

        while remaining.count > 3 && safetyCounter > 0 {
            safetyCounter -= 1
            var earFound = false

            for i in 0..<remaining.count {
                let prev = (i + remaining.count - 1) % remaining.count
                let next = (i + 1) % remaining.count

                let a = points[remaining[prev]]
                let b = points[remaining[i]]
                let c = points[remaining[next]]

                if isEar(a: a, b: b, c: c, polygon: points, remaining: remaining) {
                    indices.append(remaining[prev])
                    indices.append(remaining[i])
                    indices.append(remaining[next])
                    remaining.remove(at: i)
                    earFound = true
                    break
                }
            }

            if !earFound {
                // Fallback: just use first three vertices as triangle
                if remaining.count >= 3 {
                    indices.append(remaining[0])
                    indices.append(remaining[1])
                    indices.append(remaining[2])
                    remaining.remove(at: 1)
                }
            }
        }

        // Add final triangle
        if remaining.count == 3 {
            indices.append(remaining[0])
            indices.append(remaining[1])
            indices.append(remaining[2])
        }

        return indices
    }

    /// Calculates the signed area of a polygon.
    private func signedArea(_ polygon: [CGPoint]) -> CGFloat {
        var area: CGFloat = 0
        let n = polygon.count
        for i in 0..<n {
            let j = (i + 1) % n
            area += polygon[i].x * polygon[j].y
            area -= polygon[j].x * polygon[i].y
        }
        return area / 2
    }

    /// Checks if vertex b is an ear (can be clipped).
    private func isEar(a: CGPoint, b: CGPoint, c: CGPoint,
                       polygon: [CGPoint], remaining: [Int]) -> Bool {
        // Check if triangle is convex (counter-clockwise)
        let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if cross <= 0 {
            return false
        }

        // Check if any other point is inside the triangle
        for idx in remaining {
            let p = polygon[idx]
            if (p.x == a.x && p.y == a.y) ||
               (p.x == b.x && p.y == b.y) ||
               (p.x == c.x && p.y == c.y) {
                continue
            }
            if pointInTriangle(p: p, a: a, b: b, c: c) {
                return false
            }
        }

        return true
    }

    /// Checks if point p is inside triangle abc.
    private func pointInTriangle(p: CGPoint, a: CGPoint, b: CGPoint, c: CGPoint) -> Bool {
        let v0x = c.x - a.x
        let v0y = c.y - a.y
        let v1x = b.x - a.x
        let v1y = b.y - a.y
        let v2x = p.x - a.x
        let v2y = p.y - a.y

        let dot00 = v0x * v0x + v0y * v0y
        let dot01 = v0x * v1x + v0y * v1y
        let dot02 = v0x * v2x + v0y * v2y
        let dot11 = v1x * v1x + v1y * v1y
        let dot12 = v1x * v2x + v1y * v2y

        let invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        let u = (dot11 * dot02 - dot01 * dot12) * invDenom
        let v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= 0) && (v >= 0) && (u + v <= 1)
    }

    /// Generates stroke geometry for a polyline.
    /// Returns vertices forming triangle strips along the path.
    private func generateStrokeGeometry(polyline: [CGPoint], lineWidth: CGFloat,
                                        lineCap: CAShapeLayerLineCap,
                                        lineJoin: CAShapeLayerLineJoin) -> [CGPoint] {
        guard polyline.count >= 2 else { return [] }

        let halfWidth = lineWidth / 2
        var leftPoints: [CGPoint] = []
        var rightPoints: [CGPoint] = []

        for i in 0..<polyline.count {
            let curr = polyline[i]

            // Calculate tangent
            var tangent: CGPoint
            if i == 0 {
                let next = polyline[i + 1]
                tangent = CGPoint(x: next.x - curr.x, y: next.y - curr.y)
            } else if i == polyline.count - 1 {
                let prev = polyline[i - 1]
                tangent = CGPoint(x: curr.x - prev.x, y: curr.y - prev.y)
            } else {
                let prev = polyline[i - 1]
                let next = polyline[i + 1]
                tangent = CGPoint(x: next.x - prev.x, y: next.y - prev.y)
            }

            // Normalize tangent
            let len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y)
            if len > 0.0001 {
                tangent.x /= len
                tangent.y /= len
            }

            // Calculate normal (perpendicular)
            let normal = CGPoint(x: -tangent.y, y: tangent.x)

            // Offset points
            leftPoints.append(CGPoint(x: curr.x + normal.x * halfWidth,
                                      y: curr.y + normal.y * halfWidth))
            rightPoints.append(CGPoint(x: curr.x - normal.x * halfWidth,
                                       y: curr.y - normal.y * halfWidth))
        }

        // Build triangle strip as triangles
        var vertices: [CGPoint] = []
        for i in 0..<(leftPoints.count - 1) {
            // Triangle 1
            vertices.append(leftPoints[i])
            vertices.append(rightPoints[i])
            vertices.append(leftPoints[i + 1])

            // Triangle 2
            vertices.append(rightPoints[i])
            vertices.append(rightPoints[i + 1])
            vertices.append(leftPoints[i + 1])
        }

        // Add end caps
        if lineCap == .round {
            // Round cap at start
            let startCenter = polyline[0]
            let startTangent = CGPoint(x: polyline[1].x - polyline[0].x,
                                       y: polyline[1].y - polyline[0].y)
            addRoundCap(center: startCenter, tangent: startTangent, halfWidth: halfWidth,
                        isStart: true, into: &vertices)

            // Round cap at end
            let endCenter = polyline[polyline.count - 1]
            let endTangent = CGPoint(x: polyline[polyline.count - 1].x - polyline[polyline.count - 2].x,
                                     y: polyline[polyline.count - 1].y - polyline[polyline.count - 2].y)
            addRoundCap(center: endCenter, tangent: endTangent, halfWidth: halfWidth,
                        isStart: false, into: &vertices)
        } else if lineCap == .square {
            // Square caps extend by halfWidth
            let startCenter = polyline[0]
            let startDir = CGPoint(x: polyline[0].x - polyline[1].x,
                                   y: polyline[0].y - polyline[1].y)
            let startLen = sqrt(startDir.x * startDir.x + startDir.y * startDir.y)
            if startLen > 0 {
                let startNorm = CGPoint(x: startDir.x / startLen, y: startDir.y / startLen)
                let extendedStart = CGPoint(x: startCenter.x + startNorm.x * halfWidth,
                                            y: startCenter.y + startNorm.y * halfWidth)
                let perpStart = CGPoint(x: -startNorm.y, y: startNorm.x)

                vertices.append(leftPoints[0])
                vertices.append(CGPoint(x: extendedStart.x + perpStart.x * halfWidth,
                                        y: extendedStart.y + perpStart.y * halfWidth))
                vertices.append(rightPoints[0])

                vertices.append(rightPoints[0])
                vertices.append(CGPoint(x: extendedStart.x + perpStart.x * halfWidth,
                                        y: extendedStart.y + perpStart.y * halfWidth))
                vertices.append(CGPoint(x: extendedStart.x - perpStart.x * halfWidth,
                                        y: extendedStart.y - perpStart.y * halfWidth))
            }

            let endCenter = polyline[polyline.count - 1]
            let endDir = CGPoint(x: polyline[polyline.count - 1].x - polyline[polyline.count - 2].x,
                                 y: polyline[polyline.count - 1].y - polyline[polyline.count - 2].y)
            let endLen = sqrt(endDir.x * endDir.x + endDir.y * endDir.y)
            if endLen > 0 {
                let endNorm = CGPoint(x: endDir.x / endLen, y: endDir.y / endLen)
                let extendedEnd = CGPoint(x: endCenter.x + endNorm.x * halfWidth,
                                          y: endCenter.y + endNorm.y * halfWidth)
                let perpEnd = CGPoint(x: -endNorm.y, y: endNorm.x)

                vertices.append(leftPoints[leftPoints.count - 1])
                vertices.append(CGPoint(x: extendedEnd.x + perpEnd.x * halfWidth,
                                        y: extendedEnd.y + perpEnd.y * halfWidth))
                vertices.append(rightPoints[rightPoints.count - 1])

                vertices.append(rightPoints[rightPoints.count - 1])
                vertices.append(CGPoint(x: extendedEnd.x + perpEnd.x * halfWidth,
                                        y: extendedEnd.y + perpEnd.y * halfWidth))
                vertices.append(CGPoint(x: extendedEnd.x - perpEnd.x * halfWidth,
                                        y: extendedEnd.y - perpEnd.y * halfWidth))
            }
        }

        return vertices
    }

    /// Adds a round cap to the stroke geometry.
    private func addRoundCap(center: CGPoint, tangent: CGPoint, halfWidth: CGFloat,
                             isStart: Bool, into vertices: inout [CGPoint]) {
        let len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y)
        guard len > 0 else { return }

        let dir = CGPoint(x: tangent.x / len, y: tangent.y / len)
        let normal = CGPoint(x: -dir.y, y: dir.x)

        // Create semicircle with 8 segments
        let segments = 8
        let startAngle: CGFloat = isStart ? CGFloat.pi / 2 : -CGFloat.pi / 2
        let endAngle: CGFloat = isStart ? 3 * CGFloat.pi / 2 : CGFloat.pi / 2

        for i in 0..<segments {
            let angle1 = startAngle + CGFloat(i) * (endAngle - startAngle) / CGFloat(segments)
            let angle2 = startAngle + CGFloat(i + 1) * (endAngle - startAngle) / CGFloat(segments)

            let p1 = CGPoint(
                x: center.x + halfWidth * (cos(angle1) * normal.x - sin(angle1) * dir.x),
                y: center.y + halfWidth * (cos(angle1) * normal.y - sin(angle1) * dir.y)
            )
            let p2 = CGPoint(
                x: center.x + halfWidth * (cos(angle2) * normal.x - sin(angle2) * dir.x),
                y: center.y + halfWidth * (cos(angle2) * normal.y - sin(angle2) * dir.y)
            )

            vertices.append(center)
            vertices.append(p1)
            vertices.append(p2)
        }
    }

    /// Renders a CAShapeLayer with its path.
    private func renderShapeLayer(
        _ shapeLayer: CAShapeLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let path = shapeLayer.path else {
            return
        }
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else {
            return
        }

        // Flatten the path
        let polylines = flattenPath(path)
        guard !polylines.isEmpty else {
            return
        }

        // Render fill if fillColor is set
        if let fillColor = shapeLayer.fillColor {
            for polyline in polylines {
                guard polyline.count >= 3 else { continue }

                // Triangulate the polygon
                let indices = triangulatePolygon(polyline)
                guard !indices.isEmpty else { continue }

                // Create vertices from triangulation
                var vertices: [CARendererVertex] = []
                let colorComponents = cgColorToSIMD4(fillColor)
                let bounds = shapeLayer.bounds
                let hasValidBounds = bounds.width > 0 && bounds.height > 0

                for idx in indices {
                    let point = polyline[idx]
                    let layerCoordinate = hasValidBounds
                        ? SIMD2(
                            Float((point.x - bounds.minX) / bounds.width),
                            Float((point.y - bounds.minY) / bounds.height)
                        )
                        : .zero
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: layerCoordinate,
                        color: colorComponents
                    ))
                }

                guard !vertices.isEmpty else { continue }

                // Allocate vertices dynamically
                guard let (vertexOffset, uniformIndex) = allocateVertices(count: vertices.count) else {
                    continue  // Buffer full, skip this polyline
                }

                // Update uniforms
                var uniforms = CARendererUniforms(
                    mvpMatrix: modelMatrix,
                    opacity: currentEffectiveOpacity,
                    cornerRadius: 0,
                    layerSize: SIMD2(Float(bounds.width), Float(bounds.height)),
                    edgeAntialiasingMask: hasValidBounds ? shapeLayer.edgeAntialiasingMaskValue : 0
                )

                let uniformOffset = UInt64(uniformIndex) * Self.alignedUniformSize
                let uniformData = createFloat32Array(from: &uniforms)
                device.queue.writeBuffer(
                    uniformBuffer,
                    bufferOffset: uniformOffset,
                    data: uniformData
                )

                // Write vertices at dynamically allocated offset
                let vertexData = createFloat32Array(from: &vertices)
                device.queue.writeBuffer(
                    vertexBuffer,
                    bufferOffset: vertexOffset,
                    data: vertexData
                )

                // Draw
                renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
                renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
                renderPass.draw(vertexCount: UInt32(vertices.count))
            }
        }

        // Render stroke if strokeColor is set
        if let strokeColor = shapeLayer.strokeColor, shapeLayer.lineWidth > 0 {
            for polyline in polylines {
                guard polyline.count >= 2 else { continue }

                // Generate stroke geometry
                let strokeVertices = generateStrokeGeometry(
                    polyline: polyline,
                    lineWidth: shapeLayer.lineWidth,
                    lineCap: shapeLayer.lineCap,
                    lineJoin: shapeLayer.lineJoin
                )
                guard !strokeVertices.isEmpty else { continue }

                // Create vertices
                var vertices: [CARendererVertex] = []
                let colorComponents = cgColorToSIMD4(strokeColor)
                let bounds = shapeLayer.bounds
                let hasValidBounds = bounds.width > 0 && bounds.height > 0

                for point in strokeVertices {
                    let layerCoordinate = hasValidBounds
                        ? SIMD2(
                            Float((point.x - bounds.minX) / bounds.width),
                            Float((point.y - bounds.minY) / bounds.height)
                        )
                        : .zero
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: layerCoordinate,
                        color: colorComponents
                    ))
                }

                // Allocate vertices dynamically
                guard let (vertexOffset, uniformIndex) = allocateVertices(count: vertices.count) else {
                    continue  // Buffer full, skip this polyline
                }

                // Update uniforms
                var uniforms = CARendererUniforms(
                    mvpMatrix: modelMatrix,
                    opacity: currentEffectiveOpacity,
                    cornerRadius: 0,
                    layerSize: SIMD2(Float(bounds.width), Float(bounds.height)),
                    edgeAntialiasingMask: hasValidBounds ? shapeLayer.edgeAntialiasingMaskValue : 0
                )

                let uniformOffset = UInt64(uniformIndex) * Self.alignedUniformSize
                let uniformData = createFloat32Array(from: &uniforms)
                device.queue.writeBuffer(
                    uniformBuffer,
                    bufferOffset: uniformOffset,
                    data: uniformData
                )

                // Write vertices at dynamically allocated offset
                let vertexData = createFloat32Array(from: &vertices)
                device.queue.writeBuffer(
                    vertexBuffer,
                    bufferOffset: vertexOffset,
                    data: vertexData
                )

                // Draw
                renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
                renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
                renderPass.draw(vertexCount: UInt32(vertices.count))
            }
        }
    }

    /// Converts CGColor to SIMD4<Float>.
    private func cgColorToSIMD4(_ color: CGColor) -> SIMD4<Float> {
        replicatedColor(rgbaComponents(color))
    }

    // MARK: - Gradient Layer Rendering

    /// Renders a CAGradientLayer with its gradient colors.
    private func renderGradientLayer(
        _ gradientLayer: CAGradientLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let colors = gradientLayer.colors, !colors.isEmpty else { return }

        // Create scale matrix for layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(gradientLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(gradientLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Prepare gradient uniforms
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: Float(gradientLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(gradientLayer.bounds.width), Float(gradientLayer.bounds.height)),
            borderWidth: 0,
            renderMode: 2.0,  // Gradient mode
            gradientStartPoint: SIMD2<Float>(Float(gradientLayer.startPoint.x), Float(gradientLayer.startPoint.y)),
            gradientEndPoint: SIMD2<Float>(Float(gradientLayer.endPoint.x), Float(gradientLayer.endPoint.y)),
            gradientColorCount: Float(min(colors.count, kMaxGradientStops)),
            edgeAntialiasingMask: gradientLayer.edgeAntialiasingMaskValue,
            cornerRadii: gradientLayer.cornerRadiiComponents
        )

        // Extract gradient colors
        var gradientColors: [SIMD4<Float>] = []
        for colorAny in colors.prefix(kMaxGradientStops) {
            if let cgColor = colorAny as? CGColor {
                gradientColors.append(replicatedColor(rgbaComponents(cgColor)))
            } else {
                gradientColors.append(SIMD4<Float>(0, 0, 0, 1))
            }
        }

        // Pad to 8 colors
        while gradientColors.count < kMaxGradientStops {
            gradientColors.append(.zero)
        }

        uniforms.gradientColors = (
            gradientColors[0], gradientColors[1], gradientColors[2], gradientColors[3],
            gradientColors[4], gradientColors[5], gradientColors[6], gradientColors[7]
        )

        // Extract or generate locations
        let colorCount = min(colors.count, kMaxGradientStops)
        var locations: [Float] = []
        if let providedLocations = gradientLayer.locations, !providedLocations.isEmpty {
            locations = providedLocations.prefix(kMaxGradientStops).map { Float($0) }
        } else {
            // Generate evenly spaced locations
            for i in 0..<colorCount {
                locations.append(Float(i) / Float(max(colorCount - 1, 1)))
            }
        }

        // Pad locations to 8
        while locations.count < kMaxGradientStops {
            locations.append(1.0)
        }

        uniforms.gradientLocations = SIMD4<Float>(locations[0], locations[1], locations[2], locations[3])
        uniforms.gradientLocations2 = SIMD4<Float>(locations[4], locations[5], locations[6], locations[7])

        // Create vertices (color doesn't matter for gradient, shader uses gradient uniforms)
        let dummyColor = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: dummyColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: dummyColor),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else { return }
        let (vertexOffset, layerIndex) = allocation

        // Write uniforms
        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Draw
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    // MARK: - Shadow Rendering (2-Pass Gaussian Blur)

    private func shadowPrepassRequiresContentMasks(_ rootLayer: CALayer) -> Bool {
        var visited: Set<ObjectIdentifier> = []

        func subtreeContainsMask(_ layer: CALayer) -> Bool {
            if layer.mask != nil { return true }
            return (layer.sublayers ?? []).contains(where: subtreeContainsMask)
        }

        func visit(_ layer: CALayer) -> Bool {
            let identifier = ObjectIdentifier(layer)
            guard visited.insert(identifier).inserted else { return false }
            let presentation = renderPresentation(for: layer)
            guard !presentation.isHidden, presentation.opacity > 0 else { return false }
            if presentation.shadowOpacity > 0,
               presentation.shadowColor != nil,
               presentation.shadowPath == nil,
               subtreeContainsMask(layer) {
                return true
            }
            return (layer.sublayers ?? []).contains(where: visit)
        }

        return visit(rootLayer)
    }

    private func shadowDetachedMaskRevisionHash(of rootLayer: CALayer) -> Int {
        var hasher = Hasher()
        var mainTreeVisited: Set<ObjectIdentifier> = []
        var detachedTreeVisited: Set<ObjectIdentifier> = []

        func visit(_ layer: CALayer) {
            let identifier = ObjectIdentifier(layer)
            guard mainTreeVisited.insert(identifier).inserted else { return }
            if let mask = layer.mask {
                combineDetachedContentRevision(
                    of: mask,
                    into: &hasher,
                    visited: &detachedTreeVisited
                )
            }
            for sublayer in layer.sublayers ?? [] {
                visit(sublayer)
            }
        }

        visit(rootLayer)
        return detachedTreeVisited.isEmpty ? 0 : hasher.finalize()
    }

    /// Pre-renders every visible shadow with independently-owned GPU resources.
    private func prerenderShadows(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        var targets: [LayerPrepassTarget] = []
        collectShadowLayers(rootLayer, parentMatrix: projectionMatrix, into: &targets)

        guard !targets.isEmpty else {
            evictShadowResources(except: [])
            return
        }

        guard let device = device,
              let pipeline = pipeline,
              let depthTextureView = depthTextureView,
              let bindGroupLayout = bindGroupLayout,
              let shadowMaskPipeline = shadowMaskPipeline,
              let shadowBlurHorizontalPipeline = shadowBlurHorizontalPipeline,
              let shadowBlurVerticalPipeline = shadowBlurVerticalPipeline,
              let shadowBindGroupLayout = shadowBindGroupLayout,
              let blurSampler = blurSampler else {
            shadowRenderFailureCount += targets.count
            return
        }

        var activeRenderKeys: Set<LayerRenderKey> = []
        activeRenderKeys.reserveCapacity(targets.count)

        for target in targets {
            let shadowLayer = target.layer
            let shadowRenderKey = target.renderKey
            activeRenderKeys.insert(shadowRenderKey)

            let resources: ShadowLayerResources
            if let existing = shadowLayerResources[shadowRenderKey] {
                resources = existing
            } else {
                resources = ShadowLayerResources(
                    device: device,
                    width: Int(size.width),
                    height: Int(size.height),
                    format: preferredFormat
                )
                shadowLayerResources[shadowRenderKey] = resources
            }

            let presentationLayer = target.presentationLayer
            // Capture the silhouette at the layer's actual position. The display
            // shader applies `shadowOffset` uniformly for raw, filtered, masked,
            // and explicit-path silhouettes.
            let finalMatrix = presentationLayer.modelMatrix(parentMatrix: target.parentMatrix)
            let layerSize = SIMD2<Float>(
                Float(presentationLayer.bounds.width),
                Float(presentationLayer.bounds.height)
            )
            let cornerRadius = Float(presentationLayer.cornerRadius)
            let cornerRadii = presentationLayer.cornerRadiiComponents
            let blurRadius = max(0, Float(presentationLayer.shadowRadius)) * 0.5
            let captureState = ShadowCaptureState(
                matrix: finalMatrix,
                layerSize: layerSize,
                cornerRadius: cornerRadius,
                cornerRadii: cornerRadii,
                blurRadius: blurRadius,
                detachedMaskRevisionHash: shadowDetachedMaskRevisionHash(
                    of: shadowLayer
                )
            )

            let usesShadowPath = RasterizationDecisions.useShadowPathFastPath(
                for: presentationLayer
            )
            let contentHasActiveAnimations = !usesShadowPath
                && subtreeHasAnimations(shadowLayer)

            if !usesShadowPath,
               !contentHasActiveAnimations,
               resources.captureState == captureState,
               RasterizationDecisions.canReusePrerenderCache(
                   contributorLayer: shadowLayer,
                   hasCachedTexture: resources.hasRenderedContent
               ) {
                prerenderedShadows[shadowRenderKey] = PrerenderedShadow(resources: resources)
                continue
            }

            if usesShadowPath, let shadowPath = presentationLayer.shadowPath {
                // Opacity is applied once during composite. Baking it into the mask
                // would square the effective opacity for every shadow.
                var maskUniforms = CARendererUniforms(
                    mvpMatrix: finalMatrix,
                    opacity: 1,
                    cornerRadius: cornerRadius,
                    layerSize: layerSize,
                    cornerRadii: cornerRadii
                )
                let maskUniformData = createFloat32Array(from: &maskUniforms)
                device.queue.writeBuffer(
                    resources.maskUniformBuffer,
                    bufferOffset: 0,
                    data: maskUniformData
                )

                let maskBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                    layout: bindGroupLayout,
                    entries: [
                        GPUBindGroupEntry(
                            binding: 0,
                            resource: .buffer(
                                resources.maskUniformBuffer,
                                offset: 0,
                                size: Self.alignedUniformSize
                            )
                        )
                    ]
                ))

                let whiteColor = SIMD4<Float>(1, 1, 1, 1)
                var maskVertices: [CARendererVertex] = []
                for polyline in flattenPath(shadowPath) where polyline.count >= 3 {
                    for index in triangulatePolygon(polyline) {
                        let point = polyline[index]
                        maskVertices.append(CARendererVertex(
                            position: SIMD2(Float(point.x), Float(point.y)),
                            texCoord: .zero,
                            color: whiteColor
                        ))
                    }
                }

                let maskVertexBuffer = resources.ensureMaskVertexCapacity(
                    maskVertices.count,
                    device: device
                )
                if !maskVertices.isEmpty {
                    let maskVertexData = createFloat32Array(from: &maskVertices)
                    device.queue.writeBuffer(
                        maskVertexBuffer,
                        bufferOffset: 0,
                        data: maskVertexData
                    )
                }

                let maskRenderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
                    colorAttachments: [
                        GPURenderPassColorAttachment(
                            view: resources.maskView,
                            clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                            loadOp: .clear,
                            storeOp: .store
                        )
                    ]
                ))
                maskRenderPass.setPipeline(shadowMaskPipeline)
                maskRenderPass.setBindGroup(0, bindGroup: maskBindGroup, dynamicOffsets: [0])
                if !maskVertices.isEmpty {
                    maskRenderPass.setVertexBuffer(0, buffer: maskVertexBuffer, offset: 0)
                    maskRenderPass.draw(vertexCount: UInt32(maskVertices.count))
                }
                maskRenderPass.end()
            } else {
                withPrepassContext(target) {
                    captureShadowContent(
                        shadowLayer,
                        parentMatrix: target.parentMatrix,
                        resources: resources,
                        pipeline: pipeline,
                        depthTextureView: depthTextureView,
                        encoder: encoder
                    )
                }
            }

            var blurUniforms = BlurUniforms(
                texelSize: SIMD2<Float>(1 / Float(size.width), 1 / Float(size.height)),
                blurRadius: blurRadius
            )
            let blurUniformData = createFloat32Array(from: &blurUniforms)
            device.queue.writeBuffer(
                resources.blurUniformBuffer,
                bufferOffset: 0,
                data: blurUniformData
            )

            let horizontalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                layout: shadowBindGroupLayout,
                entries: [
                    GPUBindGroupEntry(binding: 0, resource: .buffer(
                        resources.blurUniformBuffer,
                        offset: 0,
                        size: UInt64(MemoryLayout<BlurUniforms>.stride)
                    )),
                    GPUBindGroupEntry(binding: 1, resource: .textureView(resources.maskView)),
                    GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
                ]
            ))
            let verticalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                layout: shadowBindGroupLayout,
                entries: [
                    GPUBindGroupEntry(binding: 0, resource: .buffer(
                        resources.blurUniformBuffer,
                        offset: 0,
                        size: UInt64(MemoryLayout<BlurUniforms>.stride)
                    )),
                    GPUBindGroupEntry(binding: 1, resource: .textureView(resources.intermediateView)),
                    GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
                ]
            ))

            let horizontalPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
                colorAttachments: [
                    GPURenderPassColorAttachment(
                        view: resources.intermediateView,
                        clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                        loadOp: .clear,
                        storeOp: .store
                    )
                ]
            ))
            horizontalPass.setPipeline(shadowBlurHorizontalPipeline)
            horizontalPass.setBindGroup(0, bindGroup: horizontalBindGroup)
            horizontalPass.draw(vertexCount: 6)
            horizontalPass.end()

            let verticalPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
                colorAttachments: [
                    GPURenderPassColorAttachment(
                        view: resources.maskView,
                        clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                        loadOp: .clear,
                        storeOp: .store
                    )
                ]
            ))
            verticalPass.setPipeline(shadowBlurVerticalPipeline)
            verticalPass.setBindGroup(0, bindGroup: verticalBindGroup)
            verticalPass.draw(vertexCount: 6)
            verticalPass.end()

            resources.hasRenderedContent = true
            resources.captureState = captureState
            prerenderedShadows[shadowRenderKey] = PrerenderedShadow(resources: resources)
        }

        evictShadowResources(except: activeRenderKeys)
    }

    /// Captures the actual rendered alpha of a layer subtree for the default shadow shape.
    private func captureShadowContent(
        _ layer: CALayer,
        parentMatrix: Matrix4x4,
        resources: ShadowLayerResources,
        pipeline: GPURenderPipeline,
        depthTextureView: GPUTextureView,
        encoder: GPUCommandEncoder
    ) {
        let contentRenderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: resources.maskView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ],
            depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                view: depthTextureView,
                depthClearValue: 1,
                depthLoadOp: .clear,
                depthStoreOp: .store,
                stencilClearValue: 0,
                stencilLoadOp: .clear,
                stencilStoreOp: .store
            )
        ))
        contentRenderPass.setPipeline(pipeline)
        contentRenderPass.setViewport(
            x: 0,
            y: 0,
            width: Float(size.width),
            height: Float(size.height),
            minDepth: 0,
            maxDepth: 1
        )

        let previousCaptureRoot = shadowCaptureRootLayer
        let previousRenderTargetSize = renderTargetSizeOverride
        let previousClipStack = clipRectStack
        let previousOpacityStack = opacityStack
        let previousMaskNestingDepth = maskNestingDepth
        let previousStencilValue = currentStencilValue

        shadowCaptureRootLayer = layer
        renderTargetSizeOverride = size
        clipRectStack.removeAll(keepingCapacity: true)
        opacityStack.removeAll(keepingCapacity: true)
        maskNestingDepth = 0
        currentStencilValue = 0

        renderLayer(layer, renderPass: contentRenderPass, parentMatrix: parentMatrix)

        shadowCaptureRootLayer = previousCaptureRoot
        renderTargetSizeOverride = previousRenderTargetSize
        clipRectStack = previousClipStack
        opacityStack = previousOpacityStack
        maskNestingDepth = previousMaskNestingDepth
        currentStencilValue = previousStencilValue
        contentRenderPass.end()
    }

    private func subtreeHasAnimations(_ rootLayer: CALayer) -> Bool {
        var visited: Set<ObjectIdentifier> = []

        func visit(_ layer: CALayer) -> Bool {
            let identifier = ObjectIdentifier(layer)
            guard visited.insert(identifier).inserted else { return false }
            if layer.animationKeys()?.isEmpty == false { return true }
            if let mask = layer.mask, visit(mask) { return true }
            return layer.sublayers?.contains(where: visit) == true
        }

        return visit(rootLayer)
    }

    private func evictShadowResources(except activeRenderKeys: Set<LayerRenderKey>) {
        let staleRenderKeys = shadowLayerResources.keys.filter { !activeRenderKeys.contains($0) }
        for renderKey in staleRenderKeys {
            shadowLayerResources.removeValue(forKey: renderKey)?.destroy()
            prerenderedShadows.removeValue(forKey: renderKey)
        }
    }

    /// Pre-renders every visible filtered layer into independently-owned textures.
    ///
    /// Descendants are captured before ancestors so a filtered parent includes the
    /// already-filtered output of its filtered descendants.
    private func prerenderFilteredLayers(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        guard let device = device,
              let pipeline = pipeline,
              let depthTextureView = depthTextureView else { return }

        var targets: [LayerPrepassTarget] = []
        var visitedRenderKeys: Set<LayerRenderKey> = []
        collectFilteredLayers(
            rootLayer,
            parentMatrix: projectionMatrix,
            visitedRenderKeys: &visitedRenderKeys,
            into: &targets
        )
        var activeRenderKeys: Set<LayerRenderKey> = []
        var failedRenderKeys: Set<LayerRenderKey> = []

        for target in targets {
            let filteredLayer = target.layer
            let requestedFilters = target.presentationLayer.filters ?? []
            let requiresGroup = requiresGroupOpacity(target.presentationLayer)
            let requiresCompositionSource = target.presentationLayer.compositingFilter != nil
                || target.presentationLayer.backgroundFilters?.isEmpty == false
            let requiresContentMask = target.presentationLayer.mask != nil
            let compositionOwnsAncestorMask = requiresContentMask
                && requestedFilters.isEmpty
                && !requiresGroup
                && !requiresCompositionSource
                && descendantsContainBackdropComposition(filteredLayer)
            if compositionOwnsAncestorMask {
                // Descendant backdrop-composition targets already receive this
                // ancestor mask through `contentMaskAncestors`. Capturing the
                // whole ancestor again would either hide the composition before
                // it is prepared or multiply partial mask alpha twice.
                continue
            }
            guard !requestedFilters.isEmpty
                    || requiresGroup
                    || requiresCompositionSource
                    || requiresContentMask else { continue }

            let filteredRenderKey = target.renderKey
            guard let stages = layerFilterStages(from: requestedFilters) else {
                failedRenderKeys.insert(filteredRenderKey)
                if failedLayerFilterKeys.insert(filteredRenderKey).inserted {
                    layerFilterFailureCount += 1
                }
                continue
            }
            activeRenderKeys.insert(filteredRenderKey)
            let resources: FilterLayerResources
            if let existing = filterLayerResources[filteredRenderKey] {
                resources = existing
            } else {
                resources = FilterLayerResources(
                    device: device,
                    width: Int(size.width),
                    height: Int(size.height),
                    format: preferredFormat
                )
                filterLayerResources[filteredRenderKey] = resources
            }

            let contentRenderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
                colorAttachments: [
                    GPURenderPassColorAttachment(
                        view: resources.sourceView,
                        clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                        loadOp: .clear,
                        storeOp: .store
                    )
                ],
                depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                    view: depthTextureView,
                    depthClearValue: 1.0,
                    depthLoadOp: .clear,
                    depthStoreOp: .store,
                    stencilClearValue: 0,
                    stencilLoadOp: .clear,
                    stencilStoreOp: .store
                )
            ))
            contentRenderPass.setPipeline(pipeline)
            contentRenderPass.setViewport(
                x: 0,
                y: 0,
                width: Float(size.width),
                height: Float(size.height),
                minDepth: 0,
                maxDepth: 1
            )

            let previousFilterRoot = filterPrerenderRootLayer
            let previousSuppressedMaskRoot = contentMaskCaptureSuppressedRootLayer
            withPrepassContext(target) {
                filterPrerenderRootLayer = filteredLayer
                if requiresContentMask {
                    contentMaskCaptureSuppressedRootLayer = filteredLayer
                }
                renderLayer(
                    filteredLayer,
                    renderPass: contentRenderPass,
                    parentMatrix: target.parentMatrix
                )
            }
            filterPrerenderRootLayer = previousFilterRoot
            contentMaskCaptureSuppressedRootLayer = previousSuppressedMaskRoot
            contentRenderPass.end()

            var currentTexture = resources.sourceTexture
            var currentView = resources.sourceView
            var currentAlphaMode = FilterTextureAlphaMode.premultiplied
            var conversionIndex = 0
            var didFail = false

            func convert(to targetMode: FilterTextureAlphaMode) -> Bool {
                guard currentAlphaMode != targetMode else { return true }
                let outputTexture = currentTexture === resources.resultTexture
                    ? resources.sourceTexture
                    : resources.resultTexture
                guard let outputView = resources.view(for: outputTexture) else { return false }
                let uniformBuffer = resources.uniformBuffer(
                    forOperationAt: stages.count + conversionIndex,
                    device: device
                )
                conversionIndex += 1
                guard applyAlphaConversion(
                    from: currentAlphaMode,
                    inputTexture: currentTexture,
                    outputTexture: outputTexture,
                    uniformBuffer: uniformBuffer,
                    resources: resources,
                    encoder: encoder
                ) else { return false }
                currentTexture = outputTexture
                currentView = outputView
                currentAlphaMode = targetMode
                return true
            }

            for (stageIndex, stage) in stages.enumerated() {
                switch stage {
                case let .renderer(operation):
                    guard convert(to: .premultiplied) else {
                        didFail = true
                        break
                    }
                    let outputTexture = currentTexture === resources.resultTexture
                        ? resources.sourceTexture
                        : resources.resultTexture
                    guard let outputView = resources.view(for: outputTexture) else {
                        didFail = true
                        break
                    }
                    let operationUniformBuffer = resources.uniformBuffer(
                        forOperationAt: stageIndex,
                        device: device
                    )

                    let applied: Bool
                    switch operation {
                    case let .gaussianBlur(radius):
                        guard radius > 0 else { continue }
                        applied = applyBlurFilter(
                            inputTexture: currentTexture,
                            intermediateTexture: resources.intermediateTexture,
                            outputTexture: outputTexture,
                            radius: radius,
                            uniformBuffer: operationUniformBuffer,
                            resources: resources,
                            encoder: encoder
                        )
                    case .brightness, .contrast, .saturation, .colorInvert:
                        applied = applyColorFilter(
                            operation,
                            inputTexture: currentTexture,
                            outputTexture: outputTexture,
                            uniformBuffer: operationUniformBuffer,
                            resources: resources,
                            encoder: encoder
                        )
                    }

                    guard applied else {
                        didFail = true
                        break
                    }
                    currentTexture = outputTexture
                    currentView = outputView

                case let .coreImage(filter):
                    guard convert(to: .straight) else {
                        didFail = true
                        break
                    }
                    guard let processor = layerFilterProcessor else {
                        didFail = true
                        break
                    }
                    do {
                        let execution = try processor.makeExecution(
                            filter: filter,
                            inputMode: .singleInput,
                            inputTexture: currentTexture,
                            width: UInt32(size.width),
                            height: UInt32(size.height)
                        )
                        try execution.encode(commandEncoder: encoder)
                        activeLayerFilterExecutions.append(execution)
                        currentTexture = execution.outputTexture
                        currentView = execution.outputTexture.createView()
                        currentAlphaMode = .straight
                    } catch {
                        didFail = true
                    }
                }

                if didFail {
                    break
                }
            }

            if didFail {
                failedRenderKeys.insert(filteredRenderKey)
                if failedLayerFilterKeys.insert(filteredRenderKey).inserted {
                    layerFilterFailureCount += 1
                }
                continue
            }

            guard convert(to: .premultiplied) else {
                failedRenderKeys.insert(filteredRenderKey)
                if failedLayerFilterKeys.insert(filteredRenderKey).inserted {
                    layerFilterFailureCount += 1
                }
                continue
            }

            if requiresContentMask {
                guard let maskLayer = target.presentationLayer.mask,
                      let compositionMaskApplyPipeline else {
                    failedRenderKeys.insert(filteredRenderKey)
                    if failedLayerFilterKeys.insert(filteredRenderKey).inserted {
                        layerFilterFailureCount += 1
                    }
                    continue
                }
                let maskTarget = LayerPrepassTarget(
                    layer: maskLayer,
                    presentationLayer: renderPresentation(for: maskLayer),
                    parentMatrix: target.presentationLayer.modelMatrix(
                        parentMatrix: target.parentMatrix
                    ),
                    renderKey: renderKey(for: maskLayer),
                    timeOffset: target.timeOffset
                )
                guard renderRawCompositionContentMask(
                    maskTarget,
                    outputView: resources.intermediateView,
                    suppressRootFilters: false,
                    encoder: encoder
                ) else {
                    failedRenderKeys.insert(filteredRenderKey)
                    if failedLayerFilterKeys.insert(filteredRenderKey).inserted {
                        layerFilterFailureCount += 1
                    }
                    continue
                }
                let maskedTexture = currentTexture === resources.resultTexture
                    ? resources.sourceTexture
                    : resources.resultTexture
                guard let maskedView = resources.view(for: maskedTexture),
                      encodeCompositionMaskOperation(
                        pipeline: compositionMaskApplyPipeline,
                        firstView: currentView,
                        secondView: resources.intermediateView,
                        outputView: maskedView,
                        encoder: encoder
                      ) else {
                    failedRenderKeys.insert(filteredRenderKey)
                    if failedLayerFilterKeys.insert(filteredRenderKey).inserted {
                        layerFilterFailureCount += 1
                    }
                    continue
                }
                currentTexture = maskedTexture
                currentView = maskedView
            }

            failedLayerFilterKeys.remove(filteredRenderKey)
            prerenderedFilters[filteredRenderKey] = PrerenderedFilter(
                resources: resources,
                outputTexture: currentTexture,
                outputView: currentView,
                appliedContentMask: requiresContentMask
            )
        }

        failedLayerFilterKeys.formIntersection(failedRenderKeys)
        evictFilterResources(except: activeRenderKeys)
    }

    private func descendantsContainBackdropComposition(_ layer: CALayer) -> Bool {
        var visited: Set<ObjectIdentifier> = []

        func visit(_ candidate: CALayer) -> Bool {
            let identifier = ObjectIdentifier(candidate)
            guard visited.insert(identifier).inserted else { return false }
            let presentation = renderPresentation(for: candidate)
            if presentation.compositingFilter != nil
                || presentation.backgroundFilters?.isEmpty == false {
                return true
            }
            return (candidate.sublayers ?? []).contains(where: visit)
        }

        return (layer.sublayers ?? []).contains(where: visit)
    }

    private func layerFilterStages(from filters: [Any]) -> [LayerFilterStage]? {
        var stages: [LayerFilterStage] = []
        stages.reserveCapacity(filters.count)

        for value in filters {
            if let filter = value as? CAFilter {
                if let operation = filter.operation {
                    stages.append(.renderer(operation))
                    continue
                }
                guard let coreImageFilter = CIFilter(name: filter.name) else {
                    return nil
                }
                for (key, parameter) in filter.parameters {
                    coreImageFilter.setValue(parameter, forKey: key)
                }
                stages.append(.coreImage(coreImageFilter))
            } else if let filter = value as? CIFilter {
                if filter.isEnabled {
                    stages.append(.coreImage(filter))
                }
            } else {
                return nil
            }
        }

        return stages
    }

    private func evictFilterResources(except activeRenderKeys: Set<LayerRenderKey>) {
        let staleRenderKeys = filterLayerResources.keys.filter {
            !activeRenderKeys.contains($0)
        }
        for renderKey in staleRenderKeys {
            filterLayerResources.removeValue(forKey: renderKey)?.destroy()
            prerenderedFilters.removeValue(forKey: renderKey)
        }
    }

    private func executeBackdropFilterStages(
        _ stages: [LayerFilterStage],
        inputTexture: GPUTexture,
        inputView: GPUTextureView,
        inputAlphaMode: FilterTextureAlphaMode = .straight,
        resources: FilterLayerResources,
        encoder: GPUCommandEncoder
    ) -> (texture: GPUTexture, view: GPUTextureView)? {
        guard let device else { return nil }

        var currentTexture = inputTexture
        var currentView = inputView
        var currentAlphaMode = inputAlphaMode
        var conversionIndex = 0

        func convert(to targetMode: FilterTextureAlphaMode) -> Bool {
            guard currentAlphaMode != targetMode else { return true }
            let outputTexture = currentTexture === resources.resultTexture
                ? resources.sourceTexture
                : resources.resultTexture
            guard let outputView = resources.view(for: outputTexture) else { return false }
            let uniformBuffer = resources.uniformBuffer(
                forOperationAt: stages.count + conversionIndex,
                device: device
            )
            conversionIndex += 1
            guard applyAlphaConversion(
                from: currentAlphaMode,
                inputTexture: currentTexture,
                outputTexture: outputTexture,
                uniformBuffer: uniformBuffer,
                resources: resources,
                encoder: encoder
            ) else { return false }
            currentTexture = outputTexture
            currentView = outputView
            currentAlphaMode = targetMode
            return true
        }

        for (stageIndex, stage) in stages.enumerated() {
            switch stage {
            case let .renderer(operation):
                guard convert(to: .premultiplied) else { return nil }
                let outputTexture = currentTexture === resources.resultTexture
                    ? resources.sourceTexture
                    : resources.resultTexture
                guard let outputView = resources.view(for: outputTexture) else { return nil }
                let operationUniformBuffer = resources.uniformBuffer(
                    forOperationAt: stageIndex,
                    device: device
                )

                let applied: Bool
                switch operation {
                case let .gaussianBlur(radius):
                    if radius <= 0 { continue }
                    applied = applyBlurFilter(
                        inputTexture: currentTexture,
                        intermediateTexture: resources.intermediateTexture,
                        outputTexture: outputTexture,
                        radius: radius,
                        uniformBuffer: operationUniformBuffer,
                        resources: resources,
                        encoder: encoder
                    )
                case .brightness, .contrast, .saturation, .colorInvert:
                    applied = applyColorFilter(
                        operation,
                        inputTexture: currentTexture,
                        outputTexture: outputTexture,
                        uniformBuffer: operationUniformBuffer,
                        resources: resources,
                        encoder: encoder
                    )
                }
                guard applied else { return nil }
                currentTexture = outputTexture
                currentView = outputView

            case let .coreImage(filter):
                guard convert(to: .straight),
                      let processor = layerFilterProcessor else { return nil }
                do {
                    let execution = try processor.makeExecution(
                        filter: filter,
                        inputMode: .singleInput,
                        inputTexture: currentTexture,
                        width: UInt32(size.width),
                        height: UInt32(size.height)
                    )
                    try execution.encode(commandEncoder: encoder)
                    activeCompositionExecutions.append(execution)
                    currentTexture = execution.outputTexture
                    currentView = execution.outputTexture.createView()
                    currentAlphaMode = .straight
                } catch {
                    return nil
                }
            }
        }

        guard convert(to: .straight) else { return nil }
        return (currentTexture, currentView)
    }

    private func renderBackdropFilterMask(
        for target: BackdropCompositionTarget,
        resources: CompositionLayerResources,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let extent = target.backgroundFilterExtent else {
            let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
                colorAttachments: [
                    GPURenderPassColorAttachment(
                        view: resources.backdropMaskView,
                        clearValue: GPUColor(r: 1, g: 1, b: 1, a: 1),
                        loadOp: .clear,
                        storeOp: .store
                    ),
                ]
            ))
            renderPass.end()
            return true
        }
        return renderCompositionClipShape(
            extent,
            outputView: resources.backdropMaskView,
            uniformBuffer: resources.backdropMaskUniformBuffer,
            vertexBuffer: resources.backdropMaskVertexBuffer,
            encoder: encoder
        )
    }

    private func renderCompositionClipShape(
        _ target: LayerPrepassTarget,
        outputView: GPUTextureView,
        uniformBuffer: GPUBuffer,
        vertexBuffer: GPUBuffer,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let device,
              let bindGroupLayout,
              let shadowMaskPipeline else { return false }

        let presentation = target.presentationLayer
        let bounds = presentation.bounds
        guard bounds.width > 0, bounds.height > 0 else { return false }

        let modelMatrix = presentation.modelMatrix(parentMatrix: target.parentMatrix)
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        var uniforms = CARendererUniforms(
            mvpMatrix: modelMatrix * scaleMatrix,
            opacity: 1,
            cornerRadius: Float(presentation.cornerRadius),
            layerSize: SIMD2<Float>(Float(bounds.width), Float(bounds.height)),
            cornerRadii: presentation.cornerRadiiComponents
        )
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: 0,
            data: createFloat32Array(from: &uniforms)
        )

        let white = SIMD4<Float>(repeating: 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
        ]
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: 0,
            data: createFloat32Array(from: &vertices)
        )
        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: bindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(
                    uniformBuffer,
                    offset: 0,
                    size: Self.alignedUniformSize
                )),
            ]
        ))
        let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: outputView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                ),
            ]
        ))
        renderPass.setPipeline(shadowMaskPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [0])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: 0)
        renderPass.draw(vertexCount: 6)
        renderPass.end()
        return true
    }

    private func encodeCompositionMaskOperation(
        pipeline: GPURenderPipeline,
        firstView: GPUTextureView,
        secondView: GPUTextureView,
        outputView: GPUTextureView,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let device, let blurSampler else { return false }
        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: pipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .textureView(firstView)),
                GPUBindGroupEntry(binding: 1, resource: .textureView(secondView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler)),
            ]
        ))
        let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: outputView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                ),
            ]
        ))
        renderPass.setPipeline(pipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup)
        renderPass.draw(vertexCount: 6)
        renderPass.end()
        return true
    }

    private func compositionContentMaskTreeIsDirectlyRenderable(
        _ layer: CALayer,
        isRoot: Bool = true
    ) -> Bool {
        let presentation = renderPresentation(for: layer)
        let key = renderKey(for: layer)
        let hasPreparedFilteredContent = prerenderedFilters[key] != nil
        let hasRequestedBackdropComposition = presentation.compositingFilter != nil
            || presentation.backgroundFilters?.isEmpty == false
        let hasUnpreparedBackdropComposition = hasRequestedBackdropComposition
            && prerenderedCompositions[key] == nil
        let hasUnpreparedTransition: Bool
        if let transitionState = presentation._transitionRenderState {
            hasUnpreparedTransition = transitionCaptures[
                ObjectIdentifier(transitionState.sourceLayer)
            ] == nil
        } else {
            hasUnpreparedTransition = false
        }
        if (!isRoot
                && presentation.filters?.isEmpty == false
                && !hasPreparedFilteredContent)
            || hasUnpreparedBackdropComposition
            || hasUnpreparedTransition {
            return false
        }
        if let nestedMask = presentation.mask,
           !compositionContentMaskTreeIsDirectlyRenderable(nestedMask, isRoot: false) {
            return false
        }
        return (layer.sublayers ?? []).allSatisfy {
            compositionContentMaskTreeIsDirectlyRenderable($0, isRoot: false)
        }
    }

    private func renderRawCompositionContentMask(
        _ target: LayerPrepassTarget,
        outputView: GPUTextureView,
        suppressRootFilters: Bool,
        renderSize: CGSize? = nil,
        depthStencilView: GPUTextureView? = nil,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let pipeline,
              let activeDepthStencilView = depthStencilView ?? depthTextureView,
              compositionContentMaskTreeIsDirectlyRenderable(target.layer) else { return false }

        let activeRenderSize = renderSize ?? size

        let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: outputView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                ),
            ],
            depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                view: activeDepthStencilView,
                depthClearValue: 1,
                depthLoadOp: .clear,
                depthStoreOp: .store,
                stencilClearValue: 0,
                stencilLoadOp: .clear,
                stencilStoreOp: .store
            )
        ))
        renderPass.setPipeline(pipeline)
        renderPass.setViewport(
            x: 0,
            y: 0,
            width: Float(activeRenderSize.width),
            height: Float(activeRenderSize.height),
            minDepth: 0,
            maxDepth: 1
        )

        let savedOpacityStack = opacityStack
        let savedColorStack = replicatorColorStack
        let savedTimeStack = replicatorTimeOffsetStack
        let savedInstancePath = replicatorInstancePath
        let savedClipStack = clipRectStack
        let savedMaskDepth = maskNestingDepth
        let savedStencilValue = currentStencilValue
        let savedStopKey = compositionCaptureStopKey
        let savedDidReachStop = compositionCaptureDidReachStop
        let savedFilterPrerenderRoot = filterPrerenderRootLayer
        let savedRenderTargetSize = renderTargetSizeOverride

        opacityStack.removeAll(keepingCapacity: true)
        replicatorColorStack.removeAll(keepingCapacity: true)
        replicatorTimeOffsetStack.removeAll(keepingCapacity: true)
        replicatorInstancePath.removeAll(keepingCapacity: true)
        clipRectStack.removeAll(keepingCapacity: true)
        maskNestingDepth = 0
        currentStencilValue = 0
        compositionCaptureStopKey = nil
        compositionCaptureDidReachStop = false
        renderTargetSizeOverride = activeRenderSize
        if suppressRootFilters {
            filterPrerenderRootLayer = target.layer
        }

        withPrepassContext(target) {
            renderLayer(
                target.layer,
                renderPass: renderPass,
                parentMatrix: target.parentMatrix
            )
        }

        opacityStack = savedOpacityStack
        replicatorColorStack = savedColorStack
        replicatorTimeOffsetStack = savedTimeStack
        replicatorInstancePath = savedInstancePath
        clipRectStack = savedClipStack
        maskNestingDepth = savedMaskDepth
        currentStencilValue = savedStencilValue
        compositionCaptureStopKey = savedStopKey
        compositionCaptureDidReachStop = savedDidReachStop
        filterPrerenderRootLayer = savedFilterPrerenderRoot
        renderTargetSizeOverride = savedRenderTargetSize
        renderPass.end()
        return true
    }

    private func renderCompositionContentMask(
        _ target: LayerPrepassTarget,
        outputView: GPUTextureView,
        resources: CompositionLayerResources,
        contentMaskIndex: Int,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let device,
              let stages = layerFilterStages(from: target.presentationLayer.filters ?? []) else {
            return false
        }
        guard !stages.isEmpty else {
            return renderRawCompositionContentMask(
                target,
                outputView: outputView,
                suppressRootFilters: false,
                encoder: encoder
            )
        }

        let filterResources = resources.contentMaskResources(
            at: contentMaskIndex,
            device: device,
            width: Int(size.width),
            height: Int(size.height),
            format: preferredFormat
        )
        guard renderRawCompositionContentMask(
            target,
            outputView: filterResources.sourceView,
            suppressRootFilters: true,
            encoder: encoder
        ) else { return false }

        var inputTexture = filterResources.sourceTexture
        var inputView = filterResources.sourceView
        if target.presentationLayer.opacity < 1 {
            let opacityUniformBuffer = filterResources.uniformBuffer(
                forOperationAt: stages.count + 16,
                device: device
            )
            guard applyFilterOperation(
                uniforms: FilterCompositeUniforms(opacity: target.presentationLayer.opacity),
                inputTexture: inputTexture,
                outputView: filterResources.resultView,
                uniformBuffer: opacityUniformBuffer,
                encoder: encoder
            ) else { return false }
            inputTexture = filterResources.resultTexture
            inputView = filterResources.resultView
        }

        guard let filteredMask = executeBackdropFilterStages(
            stages,
            inputTexture: inputTexture,
            inputView: inputView,
            inputAlphaMode: .premultiplied,
            resources: filterResources,
            encoder: encoder
        ) else { return false }
        let outputUniformBuffer = filterResources.uniformBuffer(
            forOperationAt: stages.count + 17,
            device: device
        )
        return applyAlphaConversion(
            from: .straight,
            inputTexture: filteredMask.texture,
            outputView: outputView,
            uniformBuffer: outputUniformBuffer,
            encoder: encoder
        )
    }

    private func buildCompositionClipMask(
        clipTargets: [LayerPrepassTarget],
        contentMaskTargets: [LayerPrepassTarget],
        resources: CompositionLayerResources,
        encoder: GPUCommandEncoder
    ) -> GPUTextureView? {
        let maskTargets = clipTargets.map(CompositionMaskTarget.clipShape)
            + contentMaskTargets.map(CompositionMaskTarget.layerContent)
        guard let device,
              let compositionMaskIntersectPipeline,
              !maskTargets.isEmpty else { return nil }

        var currentView = resources.clipCumulativeViewA
        for (index, maskTarget) in maskTargets.enumerated() {
            let outputView = index == 0
                ? resources.clipCumulativeViewA
                : resources.clipShapeView
            let didRender: Bool
            switch maskTarget {
            case let .clipShape(clipTarget):
                didRender = renderCompositionClipShape(
                    clipTarget,
                    outputView: outputView,
                    uniformBuffer: resources.clipUniformBuffer(at: index, device: device),
                    vertexBuffer: resources.backdropMaskVertexBuffer,
                    encoder: encoder
                )
            case let .layerContent(contentTarget):
                didRender = renderCompositionContentMask(
                    contentTarget,
                    outputView: outputView,
                    resources: resources,
                    contentMaskIndex: index,
                    encoder: encoder
                )
            }
            guard didRender else { return nil }

            guard index > 0 else { continue }
            let intersectionView = currentView === resources.clipCumulativeViewA
                ? resources.clipCumulativeViewB
                : resources.clipCumulativeViewA
            guard encodeCompositionMaskOperation(
                pipeline: compositionMaskIntersectPipeline,
                firstView: currentView,
                secondView: resources.clipShapeView,
                outputView: intersectionView,
                encoder: encoder
            ) else { return nil }
            currentView = intersectionView
        }
        return currentView
    }

    private func mixFilteredBackdrop(
        originalView: GPUTextureView,
        filteredView: GPUTextureView,
        maskView: GPUTextureView,
        resources: CompositionLayerResources,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let device,
              let backdropFilterMixPipeline,
              let blurSampler else { return false }

        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: backdropFilterMixPipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .textureView(originalView)),
                GPUBindGroupEntry(binding: 1, resource: .textureView(filteredView)),
                GPUBindGroupEntry(binding: 2, resource: .textureView(maskView)),
                GPUBindGroupEntry(binding: 3, resource: .sampler(blurSampler)),
            ]
        ))
        let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: resources.mixedBackdropStraightView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                ),
            ]
        ))
        renderPass.setPipeline(backdropFilterMixPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup)
        renderPass.draw(vertexCount: 6)
        renderPass.end()
        return true
    }

    private func prepareDeferredCompositionRasterizations(
        _ rootLayer: CALayer,
        projectionMatrix: Matrix4x4
    ) {
        // Backdrop composition must be resolved before any ancestor capture that
        // bakes the composition target into a flattened texture. Record those
        // ancestors so the first capture phase skips them and the second phase
        // runs after `prerenderBackdropCompositions` has produced their inputs.
        let roots = backdropCompositionRoots(
            rootLayer,
            projectionMatrix: projectionMatrix,
            mainClearColor: GPUColor(r: 0, g: 0, b: 0, a: 0)
        )
        var ancestorKeys: Set<LayerRenderKey> = []
        for root in roots {
            var targets: [BackdropCompositionTarget] = []
            withPrepassContext(root.prepass) {
                collectBackdropCompositionTargets(
                    root.prepass.layer,
                    parentMatrix: root.prepass.parentMatrix,
                    parentBackdropTarget: nil,
                    ancestorRenderKeys: [],
                    ancestorBackdropScopes: [],
                    ancestorClipTargets: [],
                    ancestorContentMaskTargets: [],
                    inheritedOpacity: 1,
                    targets: &targets
                )
            }
            ancestorKeys.formUnion(targets.flatMap(\.ancestorRenderKeys))
        }
        deferredCompositionRasterizationKeys = ancestorKeys
    }

    private func backdropCompositionRoots(
        _ rootLayer: CALayer,
        projectionMatrix: Matrix4x4,
        mainClearColor: GPUColor
    ) -> [BackdropCompositionRoot] {
        var detachedRoots: [BackdropCompositionRoot] = []
        var visitedRootKeys: Set<LayerRenderKey> = []

        func visit(_ layer: CALayer, parentMatrix: Matrix4x4) {
            let presentation = renderPresentation(for: layer)
            guard !presentation.isHidden, presentation.opacity > 0 else { return }
            let modelMatrix = presentation.modelMatrix(parentMatrix: parentMatrix)

            if let mask = presentation.mask {
                let maskTarget = LayerPrepassTarget(
                    layer: mask,
                    presentationLayer: renderPresentation(for: mask),
                    parentMatrix: modelMatrix,
                    renderKey: renderKey(for: mask),
                    timeOffset: currentReplicatorTimeOffset
                )
                if visitedRootKeys.insert(maskTarget.renderKey).inserted {
                    withPrepassContext(maskTarget) {
                        visit(mask, parentMatrix: modelMatrix)
                    }
                    detachedRoots.append(BackdropCompositionRoot(
                        prepass: maskTarget,
                        clearColor: GPUColor(r: 0, g: 0, b: 0, a: 0)
                    ))
                }
            }

            forEachPrepassSublayer(
                of: layer,
                presentationLayer: presentation,
                modelMatrix: modelMatrix
            ) { sublayer, sublayerParentMatrix in
                visit(sublayer, parentMatrix: sublayerParentMatrix)
            }
        }

        visit(rootLayer, parentMatrix: projectionMatrix)
        let mainTarget = LayerPrepassTarget(
            layer: rootLayer,
            presentationLayer: renderPresentation(for: rootLayer),
            parentMatrix: projectionMatrix,
            renderKey: renderKey(for: rootLayer),
            timeOffset: currentReplicatorTimeOffset
        )
        detachedRoots.append(BackdropCompositionRoot(
            prepass: mainTarget,
            clearColor: mainClearColor
        ))
        return detachedRoots
    }

    private func prerenderBackdropCompositions(
        _ rootLayer: CALayer,
        clearColor: GPUColor,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        guard let device,
              let pipeline,
              let depthTextureView,
              let processor = layerFilterProcessor else { return }

        let roots = backdropCompositionRoots(
            rootLayer,
            projectionMatrix: projectionMatrix,
            mainClearColor: clearColor
        )
        var activeKeys: Set<LayerRenderKey> = []
        var failedKeys: Set<LayerRenderKey> = []

        for root in roots {
            var targets: [BackdropCompositionTarget] = []
            withPrepassContext(root.prepass) {
                collectBackdropCompositionTargets(
                    root.prepass.layer,
                    parentMatrix: root.prepass.parentMatrix,
                    parentBackdropTarget: nil,
                    ancestorRenderKeys: [],
                    ancestorBackdropScopes: [],
                    ancestorClipTargets: [],
                    ancestorContentMaskTargets: [],
                    inheritedOpacity: 1,
                    targets: &targets
                )
            }
            var prefixIsValid = true
            let recordFailure: (LayerRenderKey) -> Void = { key in
                failedKeys.insert(key)
                if self.failedCompositionKeys.insert(key).inserted {
                    self.compositionFilterFailureCount += 1
                }
                prefixIsValid = false
            }

            let processingTargets = targets.enumerated().sorted { lhs, rhs in
                if lhs.element.depth != rhs.element.depth {
                    return lhs.element.depth > rhs.element.depth
                }
                return lhs.offset < rhs.offset
            }.map(\.element)
            var previousDepth = processingTargets.first?.depth

            for compositionTarget in processingTargets {
                if let previousDepth, compositionTarget.depth < previousDepth {
                    prerenderFilteredLayers(
                        rootLayer,
                        encoder: encoder,
                        projectionMatrix: projectionMatrix
                    )
                }
                previousDepth = compositionTarget.depth
                let target = compositionTarget.prepass
                let key = target.renderKey
                let presentation = target.presentationLayer
                let requestedBackgroundFilters = presentation.backgroundFilters ?? []
                guard let backgroundStages = layerFilterStages(from: requestedBackgroundFilters) else {
                    recordFailure(key)
                    continue
                }
                let compositionFilter: CIFilter
                if let requestedFilter = presentation.compositingFilter as? CIFilter {
                    compositionFilter = requestedFilter
                } else if presentation.compositingFilter != nil {
                    recordFailure(key)
                    continue
                } else if let sourceOver = CIFilter(name: "CISourceOverCompositing") {
                    compositionFilter = sourceOver
                } else {
                    recordFailure(key)
                    continue
                }
                guard prefixIsValid,
                      processor.supports(compositionFilter, inputMode: .foregroundAndBackground),
                      let source = prerenderedFilters[key] else {
                    recordFailure(key)
                    continue
                }

                let resources: CompositionLayerResources
                if let existing = compositionLayerResources[key] {
                    resources = existing
                } else {
                    resources = CompositionLayerResources(
                        device: device,
                        width: Int(size.width),
                        height: Int(size.height),
                        format: preferredFormat
                    )
                    compositionLayerResources[key] = resources
                }
                activeKeys.insert(key)

                let backdropClearColor = compositionTarget.scope == nil
                    ? root.clearColor
                    : GPUColor(r: 0, g: 0, b: 0, a: 0)
                let backdropPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
                    colorAttachments: [
                        GPURenderPassColorAttachment(
                            view: resources.backdropView,
                            clearValue: backdropClearColor,
                            loadOp: .clear,
                            storeOp: .store
                        )
                    ],
                    depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                        view: depthTextureView,
                        depthClearValue: 1,
                        depthLoadOp: .clear,
                        depthStoreOp: .store,
                        stencilClearValue: 0,
                        stencilLoadOp: .clear,
                        stencilStoreOp: .store
                    )
                ))
                backdropPass.setPipeline(pipeline)
                backdropPass.setViewport(
                    x: 0,
                    y: 0,
                    width: Float(size.width),
                    height: Float(size.height),
                    minDepth: 0,
                    maxDepth: 1
                )

                let savedOpacityStack = opacityStack
                let savedColorStack = replicatorColorStack
                let savedTimeStack = replicatorTimeOffsetStack
                let savedInstancePath = replicatorInstancePath
                let savedClipStack = clipRectStack
                let savedMaskDepth = maskNestingDepth
                let savedStencilValue = currentStencilValue
                let savedStopKey = compositionCaptureStopKey
                let savedDidReachStop = compositionCaptureDidReachStop
                let savedPassThroughKeys = compositionCapturePassThroughKeys
                let savedFilterPrerenderRoot = filterPrerenderRootLayer

                opacityStack.removeAll(keepingCapacity: true)
                replicatorColorStack.removeAll(keepingCapacity: true)
                replicatorTimeOffsetStack.removeAll(keepingCapacity: true)
                replicatorInstancePath.removeAll(keepingCapacity: true)
                clipRectStack.removeAll(keepingCapacity: true)
                maskNestingDepth = 0
                currentStencilValue = 0
                compositionCaptureStopKey = key
                compositionCaptureDidReachStop = false
                compositionCapturePassThroughKeys = Set(compositionTarget.ancestorRenderKeys)

                if let scope = compositionTarget.scope {
                    filterPrerenderRootLayer = scope.layer
                    withPrepassContext(scope) {
                        renderLayer(
                            scope.layer,
                            renderPass: backdropPass,
                            parentMatrix: scope.parentMatrix
                        )
                    }
                } else {
                    withPrepassContext(root.prepass) {
                        renderLayer(
                            root.prepass.layer,
                            renderPass: backdropPass,
                            parentMatrix: root.prepass.parentMatrix
                        )
                    }
                }
                let didReachTarget = compositionCaptureDidReachStop

                opacityStack = savedOpacityStack
                replicatorColorStack = savedColorStack
                replicatorTimeOffsetStack = savedTimeStack
                replicatorInstancePath = savedInstancePath
                clipRectStack = savedClipStack
                maskNestingDepth = savedMaskDepth
                currentStencilValue = savedStencilValue
                compositionCaptureStopKey = savedStopKey
                compositionCaptureDidReachStop = savedDidReachStop
                compositionCapturePassThroughKeys = savedPassThroughKeys
                filterPrerenderRootLayer = savedFilterPrerenderRoot
                backdropPass.end()

                let clipMaskView = buildCompositionClipMask(
                    clipTargets: compositionTarget.clipAncestors,
                    contentMaskTargets: compositionTarget.contentMaskAncestors,
                    resources: resources,
                    encoder: encoder
                )
                var premultipliedSourceTexture = source.outputTexture
                if !compositionTarget.clipAncestors.isEmpty
                    || !compositionTarget.contentMaskAncestors.isEmpty {
                    guard let clipMaskView,
                          let compositionMaskApplyPipeline,
                          encodeCompositionMaskOperation(
                            pipeline: compositionMaskApplyPipeline,
                            firstView: source.outputView,
                            secondView: clipMaskView,
                            outputView: resources.clippedSourceView,
                            encoder: encoder
                          ) else {
                        recordFailure(key)
                        continue
                    }
                    premultipliedSourceTexture = resources.clippedSourceTexture
                }
                if compositionTarget.sourceOpacity < 1
                    || compositionTarget.sourceColor != SIMD4<Float>(repeating: 1) {
                    guard applyFilterOperation(
                        uniforms: FilterCompositeUniforms(
                            opacity: compositionTarget.sourceOpacity,
                            colorMultiplier: compositionTarget.sourceColor
                        ),
                        inputTexture: premultipliedSourceTexture,
                        outputView: resources.sourcePremultipliedView,
                        uniformBuffer: resources.sourceOpacityUniformBuffer,
                        encoder: encoder
                    ) else {
                        recordFailure(key)
                        continue
                    }
                    premultipliedSourceTexture = resources.sourcePremultipliedTexture
                }

                guard didReachTarget,
                      applyAlphaConversion(
                        from: .premultiplied,
                        inputTexture: premultipliedSourceTexture,
                        outputView: resources.sourceStraightView,
                        uniformBuffer: resources.sourceUniformBuffer,
                        encoder: encoder
                      ),
                      applyAlphaConversion(
                        from: .premultiplied,
                        inputTexture: resources.backdropTexture,
                        outputView: resources.backdropStraightView,
                        uniformBuffer: resources.backdropUniformBuffer,
                        encoder: encoder
                      ) else {
                    recordFailure(key)
                    continue
                }

                var compositionBackdropTexture = resources.backdropStraightTexture
                if !backgroundStages.isEmpty {
                    guard let filteredBackdrop = executeBackdropFilterStages(
                        backgroundStages,
                        inputTexture: resources.backdropStraightTexture,
                        inputView: resources.backdropStraightView,
                        resources: resources.backgroundFilterResources,
                        encoder: encoder
                    ),
                    renderBackdropFilterMask(
                        for: compositionTarget,
                        resources: resources,
                        encoder: encoder
                    ) else {
                        recordFailure(key)
                        continue
                    }
                    var backgroundMaskView = resources.backdropMaskView
                    let backgroundClipMaskView: GPUTextureView?
                    if let targetContentMask = compositionTarget.targetContentMask {
                        backgroundClipMaskView = buildCompositionClipMask(
                            clipTargets: compositionTarget.clipAncestors,
                            contentMaskTargets: compositionTarget.contentMaskAncestors
                                + [targetContentMask],
                            resources: resources,
                            encoder: encoder
                        )
                    } else {
                        backgroundClipMaskView = clipMaskView
                    }
                    if let backgroundClipMaskView {
                        guard let compositionMaskIntersectPipeline,
                              encodeCompositionMaskOperation(
                                pipeline: compositionMaskIntersectPipeline,
                                firstView: resources.backdropMaskView,
                                secondView: backgroundClipMaskView,
                                outputView: resources.combinedBackdropMaskView,
                                encoder: encoder
                              ) else {
                            recordFailure(key)
                            continue
                        }
                        backgroundMaskView = resources.combinedBackdropMaskView
                    }
                    guard mixFilteredBackdrop(
                        originalView: resources.backdropStraightView,
                        filteredView: filteredBackdrop.view,
                        maskView: backgroundMaskView,
                        resources: resources,
                        encoder: encoder
                    ) else {
                        recordFailure(key)
                        continue
                    }
                    compositionBackdropTexture = resources.mixedBackdropStraightTexture
                }

                do {
                    let execution = try processor.makeExecution(
                        filter: compositionFilter,
                        inputMode: .foregroundAndBackground,
                        inputTexture: resources.sourceStraightTexture,
                        backgroundTexture: compositionBackdropTexture,
                        width: UInt32(size.width),
                        height: UInt32(size.height)
                    )
                    try execution.encode(commandEncoder: encoder)
                    activeCompositionExecutions.append(execution)
                    guard applyAlphaConversion(
                        from: .straight,
                        inputTexture: execution.outputTexture,
                        outputView: resources.resultPremultipliedView,
                        uniformBuffer: resources.resultConversionUniformBuffer,
                        encoder: encoder
                    ) else {
                        recordFailure(key)
                        continue
                    }
                    failedCompositionKeys.remove(key)
                    prerenderedCompositions[key] = PrerenderedComposition(
                        resources: resources,
                        outputTexture: resources.resultPremultipliedTexture,
                        outputView: resources.resultPremultipliedView,
                        samplingModelMatrix: presentation.modelMatrix(
                            parentMatrix: target.parentMatrix
                        )
                    )
                } catch {
                    recordFailure(key)
                }
            }

            if previousDepth != nil {
                // Refresh every context so resources referenced by commands
                // encoded for sibling mask roots remain alive until submission.
                // The refreshed main tree also picks up depth-zero compositions
                // that became available inside detached content masks.
                prerenderFilteredLayers(
                    rootLayer,
                    encoder: encoder,
                    projectionMatrix: projectionMatrix
                )
            }
        }

        failedCompositionKeys.formIntersection(failedKeys)
        let staleKeys = compositionLayerResources.keys.filter { !activeKeys.contains($0) }
        for key in staleKeys {
            compositionLayerResources.removeValue(forKey: key)?.destroy()
            prerenderedCompositions.removeValue(forKey: key)
        }
    }

    private func collectBackdropCompositionTargets(
        _ layer: CALayer,
        parentMatrix: Matrix4x4,
        parentBackdropTarget: LayerPrepassTarget?,
        ancestorRenderKeys: [LayerRenderKey],
        ancestorBackdropScopes: [LayerPrepassTarget],
        ancestorClipTargets: [LayerPrepassTarget],
        ancestorContentMaskTargets: [LayerPrepassTarget],
        inheritedOpacity: Float,
        targets: inout [BackdropCompositionTarget]
    ) {
        let presentation = renderPresentation(for: layer)
        guard !presentation.isHidden && presentation.opacity > 0 else { return }

        let key = renderKey(for: layer)
        let modelMatrix = presentation.modelMatrix(parentMatrix: parentMatrix)
        if layer is CATransformLayer {
            forEachPrepassSublayer(
                of: layer,
                presentationLayer: presentation,
                modelMatrix: modelMatrix
            ) { sublayer, sublayerParentMatrix in
                collectBackdropCompositionTargets(
                    sublayer,
                    parentMatrix: sublayerParentMatrix,
                    parentBackdropTarget: parentBackdropTarget,
                    ancestorRenderKeys: ancestorRenderKeys,
                    ancestorBackdropScopes: ancestorBackdropScopes,
                    ancestorClipTargets: ancestorClipTargets,
                    ancestorContentMaskTargets: ancestorContentMaskTargets,
                    inheritedOpacity: inheritedOpacity * presentation.opacity,
                    targets: &targets
                )
            }
            return
        }
        let hasComposition = presentation.compositingFilter != nil
            || presentation.backgroundFilters?.isEmpty == false
        let createsBackdropScope = hasComposition
            || requiresGroupOpacity(presentation)
            || presentation.filters?.isEmpty == false
        let prepass = LayerPrepassTarget(
            layer: layer,
            presentationLayer: presentation,
            parentMatrix: parentMatrix,
            renderKey: key,
            timeOffset: currentReplicatorTimeOffset
        )
        var descendantScopes = ancestorBackdropScopes
        var descendantClipTargets = ancestorClipTargets
        var descendantContentMaskTargets = ancestorContentMaskTargets
        let targetContentMask: LayerPrepassTarget?
        if let maskLayer = presentation.mask {
            targetContentMask = LayerPrepassTarget(
                layer: maskLayer,
                presentationLayer: renderPresentation(for: maskLayer),
                parentMatrix: modelMatrix,
                renderKey: renderKey(for: maskLayer),
                timeOffset: currentReplicatorTimeOffset
            )
        } else {
            targetContentMask = nil
        }
        if hasComposition {
            targets.append(BackdropCompositionTarget(
                prepass: prepass,
                scope: ancestorBackdropScopes.last,
                backgroundFilterExtent: presentation.masksToBounds
                    ? prepass
                    : parentBackdropTarget,
                ancestorRenderKeys: ancestorRenderKeys,
                depth: ancestorBackdropScopes.count,
                clipAncestors: ancestorClipTargets,
                contentMaskAncestors: ancestorContentMaskTargets,
                targetContentMask: targetContentMask,
                sourceOpacity: inheritedOpacity * presentation.opacity,
                sourceColor: ancestorBackdropScopes.isEmpty
                    ? currentReplicatorColor
                    : SIMD4<Float>(repeating: 1)
            ))
        }
        if createsBackdropScope {
            descendantScopes.append(prepass)
        }
        if presentation.masksToBounds {
            descendantClipTargets.append(prepass)
        }
        if let targetContentMask {
            descendantContentMaskTargets.append(targetContentMask)
        }

        forEachPrepassSublayer(
            of: layer,
            presentationLayer: presentation,
            modelMatrix: modelMatrix
        ) { sublayer, sublayerParentMatrix in
            collectBackdropCompositionTargets(
                sublayer,
                parentMatrix: sublayerParentMatrix,
                parentBackdropTarget: prepass,
                ancestorRenderKeys: ancestorRenderKeys + [key],
                ancestorBackdropScopes: descendantScopes,
                ancestorClipTargets: descendantClipTargets,
                ancestorContentMaskTargets: descendantContentMaskTargets,
                inheritedOpacity: createsBackdropScope
                    ? 1
                    : inheritedOpacity * presentation.opacity,
                targets: &targets
            )
        }
    }

    // MARK: - Rasterized Layer Pre-render (R3.2 / R3.3)

    /// Walks the tree and captures every `shouldRasterize` subtree into
    /// its own offscreen texture, populating `prerasterizedTextures` so
    /// the main pass can composite the captured pixels as a quad.
    ///
    /// On a cache hit the existing texture is reused — no GPU work — and
    /// `lookup` updates the entry's `lastUsedFrame` to keep the idle
    /// eviction pass honest. On a miss the renderer allocates a fresh
    /// layer-sized texture, redirects `renderLayer` into it with
    /// `rasterizePrerenderRootLayer` set so this same layer's composite
    /// branch in `renderLayer` is suppressed during the capture pass,
    /// and inserts the new entry. Explicit rasterization preserves its
    /// opaque-clear contract; automatic transform flattening clears transparent.
    private func prerenderRasterizedLayers(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        guard let device = device,
              let pipeline = pipeline,
              let cache = rasterizationCache else { return }

        // Frame token already bumped at the top of `render`; use it for
        // both lookup `lastUsedFrame` updates and insert tagging.
        let frameToken = CALayer._currentFrameToken

        collectAndCaptureRasterizedLayers(
            rootLayer,
            parentMatrix: projectionMatrix,
            parentIsTransformLayer: false,
            isInsideFlattenedSubtree: false,
            device: device,
            pipeline: pipeline,
            cache: cache,
            encoder: encoder,
            frameToken: frameToken
        )
    }

    /// Recursive worker for `prerenderRasterizedLayers`. Walks the layer
    /// tree, decides per-layer reuse via `RasterizationDecisions`, and
    /// dispatches the capture pass on miss. Walks descendants of
    /// `shouldRasterize` layers too — nested `shouldRasterize` layers
    /// each get their own cache entry, just as CoreAnimation would.
    private func collectAndCaptureRasterizedLayers(
        _ layer: CALayer,
        parentMatrix: Matrix4x4,
        parentIsTransformLayer: Bool,
        isInsideFlattenedSubtree: Bool,
        isMaskTreeRoot: Bool = false,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        cache: RasterizationCache<GPUTexture>,
        encoder: GPUCommandEncoder,
        frameToken: UInt64
    ) {
        let presentationLayer = renderPresentation(for: layer)
        guard !presentationLayer.isHidden, presentationLayer.opacity > 0 else {
            return
        }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)
        let isTransformLayer = layer is CATransformLayer
        let requiresTransformFlattening = parentIsTransformLayer
            && !isTransformLayer
            && requiresTransformFlattening(
                modelLayer: layer,
                presentationLayer: presentationLayer
            )
        let requiresEffectFlattening = isInsideFlattenedSubtree
            && !isMaskTreeRoot
            && !isTransformLayer
            && !presentationLayer.shouldRasterize
            && !requiresTransformFlattening
            && requiresEffectFlattening(presentationLayer)
        let descendantsAreInsideFlattenedSubtree = isInsideFlattenedSubtree
            || isMaskTreeRoot
            || presentationLayer.shouldRasterize
            || requiresTransformFlattening

        if let maskLayer = presentationLayer.mask {
            collectAndCaptureRasterizedLayers(
                maskLayer,
                parentMatrix: modelMatrix,
                parentIsTransformLayer: false,
                isInsideFlattenedSubtree: true,
                isMaskTreeRoot: true,
                device: device,
                pipeline: pipeline,
                cache: cache,
                encoder: encoder,
                frameToken: frameToken
            )
        }

        // Capture descendants first so a parent capture composites already-finalized
        // nested rasterization entries instead of baking their uncached live paths.
        forEachPrepassSublayer(
            of: layer,
            presentationLayer: presentationLayer,
            modelMatrix: modelMatrix
        ) { sublayer, sublayerParentMatrix in
            collectAndCaptureRasterizedLayers(
                sublayer,
                parentMatrix: sublayerParentMatrix,
                parentIsTransformLayer: isTransformLayer,
                isInsideFlattenedSubtree: descendantsAreInsideFlattenedSubtree,
                device: device,
                pipeline: pipeline,
                cache: cache,
                encoder: encoder,
                frameToken: frameToken
            )
        }

        if !isTransformLayer,
           presentationLayer.shouldRasterize
            || requiresTransformFlattening
            || requiresEffectFlattening {
            let purpose: RasterizationCachePurpose
            if requiresTransformFlattening {
                purpose = .transformFlattening
            } else if presentationLayer.shouldRasterize {
                purpose = .explicit
            } else {
                purpose = .effectFlattening
            }
            let isDeferredCompositionRasterization =
                deferredCompositionRasterizationKeys.contains(renderKey(for: layer))
            guard capturesOnlyDeferredCompositionRasterizations
                == isDeferredCompositionRasterization else {
                return
            }
            captureRasterizedLayer(
                layer,
                purpose: purpose,
                device: device,
                pipeline: pipeline,
                cache: cache,
                encoder: encoder,
                frameToken: frameToken
            )
        }
    }

    /// Captures or reuses the rasterized texture for one
    /// `shouldRasterize` layer.
    ///
    /// Per PERFORMANCE_DESIGN.md §5.2 the offscreen texture is sized to
    /// the layer's local capture extent × `rasterizationScale`, not the canvas.
    /// The extent unions visible unmasked descendants through their projective
    /// transforms and expands for shadows, so the composite path preserves
    /// out-of-bounds pixels without burning canvas-sized memory per entry. The
    /// capture pass uses a local orthographic projection so the layer's own
    /// `position`, `transform`
    /// and `anchorPoint` are *excluded* from the bake — those land at
    /// composite time. `renderLayer` recognises the capture root via
    /// `rasterizePrerenderRootLayer` and skips its `modelMatrix`
    /// computation accordingly.
    private func captureRasterizedLayer(
        _ layer: CALayer,
        purpose: RasterizationCachePurpose,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        cache: RasterizationCache<GPUTexture>,
        encoder: GPUCommandEncoder,
        frameToken: UInt64
    ) {
        let layerRenderKey = renderKey(for: layer)
        let key = RasterizationCacheKey(layerRenderKey, purpose: purpose)
        let presentationLayer = renderPresentation(for: layer)
        guard let captureBounds = rasterizationCaptureBounds(for: layer) else {
            rasterizationFailureCount += 1
            return
        }
        let contentBoundsHash = rasterizationContentBoundsHash(
            for: layer,
            captureBounds: captureBounds
        )

        // Backdrop-dependent captures cannot reuse ordinary layer dirty state:
        // pixels outside the captured subtree can change their result. Other
        // cache hits update `lastUsedFrame` so idle eviction treats them as live.
        if !deferredCompositionRasterizationKeys.contains(layerRenderKey),
           let entry = cache.lookup(key, atFrame: frameToken),
           canReuseRasterizedTexture(
               layer: layer,
               entry: entry,
               purpose: purpose,
               contentBoundsHash: contentBoundsHash
           ) {
            prerasterizedTextures[layerRenderKey] = PrerasterizedTexture(
                texture: entry.texture,
                purpose: purpose,
                captureBounds: captureBounds
            )
            return
        }

        // Miss — allocate a visible-subtree texture (`captureBounds × scale`)
        // and render the subtree into it under a bounds-local projection.
        let requestedScale = CGFloat(presentationLayer.rasterizationScale)
        guard requestedScale.isFinite, requestedScale > 0 else {
            rasterizationFailureCount += 1
            return
        }
        let scaledWidth = captureBounds.width * requestedScale
        let scaledHeight = captureBounds.height * requestedScale
        guard scaledWidth.isFinite,
              scaledHeight.isFinite,
              scaledWidth > 0,
              scaledHeight > 0 else {
            rasterizationFailureCount += 1
            return
        }
        let maximumDimension = CGFloat(max(1, Int(device.limits.maxTextureDimension2D)))
        let fittingScale = min(1, maximumDimension / max(scaledWidth, scaledHeight))
        let pixelWidth = max(1, Int((scaledWidth * fittingScale).rounded(.up)))
        let pixelHeight = max(1, Int((scaledHeight * fittingScale).rounded(.up)))

        let requestedFilters = presentationLayer.filters ?? []
        let filterStages: [LayerFilterStage]
        if requestedFilters.isEmpty {
            filterStages = []
        } else {
            guard let stages = layerFilterStages(from: requestedFilters) else {
                return
            }
            filterStages = stages
        }

        let captureTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(pixelWidth), height: UInt32(pixelHeight)),
            format: preferredFormat,
            usage: [.renderAttachment, .textureBinding]
        ))
        let captureView = captureTexture.createView()
        let captureDepthTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(pixelWidth), height: UInt32(pixelHeight)),
            format: .depth24plusStencil8,
            usage: .renderAttachment
        ))
        transientCaptureDepthTextures.append(captureDepthTexture)
        let captureDepthView = captureDepthTexture.createView()

        // R3.3 / R3.4a: clear α is 1.0 regardless of `layer.opacity`; the
        // composite pass applies `layer.opacity` at draw time. With the
        // texture now matching the layer's own bounds, the layer's
        // backgroundColor / sublayers fill the texture and this clear
        // value only shows through where the layer doesn't draw at all
        // (a flat-rasterized `shouldRasterize` layer is treated as an
        // opaque image, mirroring CoreAnimation's WWDC 2014 #419 model).
        let clearAlpha: Float = purpose == .explicit && !hasVisibleShadow(presentationLayer)
            ? RasterizationDecisions.captureClearAlpha()
            : 0
        let capturePass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: captureView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: Double(clearAlpha)),
                    loadOp: .clear,
                    storeOp: .store
                )
            ],
            depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                view: captureDepthView,
                depthClearValue: 0.0,
                depthLoadOp: .clear,
                depthStoreOp: .store,
                stencilClearValue: 0,
                stencilLoadOp: .clear,
                stencilStoreOp: .store
            )
        ))
        capturePass.setPipeline(pipeline)
        capturePass.setViewport(
            x: 0,
            y: 0,
            width: Float(pixelWidth),
            height: Float(pixelHeight),
            minDepth: 0,
            maxDepth: 1
        )

        // Bounds-local projection: maps `[bounds.minX, bounds.maxX]` ×
        // `[bounds.minY, bounds.maxY]` to NDC. Combined with the
        // capture-root `modelMatrix` shortcut in `renderLayer` (which
        // bypasses position/transform/anchor for `rasterizePrerenderRootLayer`),
        // this places the layer's bounds-local content directly into the
        // texture's pixel grid.
        let captureProjection = Matrix4x4.orthographic(
            left: Float(captureBounds.minX),
            right: Float(captureBounds.maxX),
            bottom: Float(captureBounds.minY),
            top: Float(captureBounds.maxY),
            near: -1000,
            far: 1000
        )

        let captureSize = CGSize(width: CGFloat(pixelWidth), height: CGFloat(pixelHeight))
        let previousCaptureRoot = rasterizePrerenderRootLayer
        let previousRenderTargetSize = renderTargetSizeOverride
        let previousClipStack = clipRectStack
        let previousOpacityStack = opacityStack
        let previousMaskNestingDepth = maskNestingDepth
        let previousTransformDepthNesting = transformDepthNesting
        let previousStencilValue = currentStencilValue

        rasterizePrerenderRootLayer = layer
        renderTargetSizeOverride = captureSize
        clipRectStack.removeAll(keepingCapacity: true)
        opacityStack.removeAll(keepingCapacity: true)
        maskNestingDepth = 0
        transformDepthNesting = 0
        currentStencilValue = 0
        renderLayer(layer, renderPass: capturePass, parentMatrix: captureProjection)

        rasterizePrerenderRootLayer = previousCaptureRoot
        renderTargetSizeOverride = previousRenderTargetSize
        clipRectStack = previousClipStack
        opacityStack = previousOpacityStack
        maskNestingDepth = previousMaskNestingDepth
        transformDepthNesting = previousTransformDepthNesting
        currentStencilValue = previousStencilValue

        capturePass.end()

        var cachedTexture: GPUTexture
        if filterStages.isEmpty {
            cachedTexture = captureTexture
        } else {
            transientRasterizationTextures.append(captureTexture)
            guard let filteredTexture = executeRasterizedFilterStages(
                filterStages,
                inputTexture: captureTexture,
                width: pixelWidth,
                height: pixelHeight,
                encoder: encoder
            ) else {
                if failedLayerFilterKeys.insert(layerRenderKey).inserted {
                    layerFilterFailureCount += 1
                }
                return
            }
            failedLayerFilterKeys.remove(layerRenderKey)
            cachedTexture = filteredTexture
        }

        if let maskLayer = presentationLayer.mask {
            guard let maskedTexture = executeRasterizedContentMask(
                maskLayer,
                contentTexture: cachedTexture,
                projectionMatrix: captureProjection,
                renderSize: captureSize,
                depthStencilView: captureDepthView,
                width: pixelWidth,
                height: pixelHeight,
                encoder: encoder
            ) else {
                if failedLayerFilterKeys.insert(layerRenderKey).inserted {
                    layerFilterFailureCount += 1
                }
                return
            }
            failedLayerFilterKeys.remove(layerRenderKey)
            cachedTexture = maskedTexture
        }

        if hasVisibleShadow(presentationLayer) {
            transientRasterizationTextures.append(cachedTexture)
            guard let shadowedTexture = executeRasterizedShadow(
                layer: presentationLayer,
                contentTexture: cachedTexture,
                captureBounds: captureBounds,
                width: pixelWidth,
                height: pixelHeight,
                encoder: encoder
            ) else {
                shadowRenderFailureCount += 1
                return
            }
            cachedTexture = shadowedTexture
        }

        // Insert the entry. The cache enforces the byte budget only
        // when `evictToBudget` is called (post-submit), so an oversized
        // single insert is allowed to land here.
        let pixelSize = captureSize
        cache.insert(
            key,
            texture: cachedTexture,
            pixelSize: pixelSize,
            contentBoundsHash: contentBoundsHash,
            atFrame: frameToken
        )
        prerasterizedTextures[layerRenderKey] = PrerasterizedTexture(
            texture: cachedTexture,
            purpose: purpose,
            captureBounds: captureBounds
        )
        if purpose == .transformFlattening {
            transformFlatteningCaptureCount += 1
        } else if purpose == .explicit {
            explicitRasterizationCaptureCount += 1
        }
    }

    private func canReuseRasterizedTexture(
        layer: CALayer,
        entry: RasterizedEntry<GPUTexture>,
        purpose: RasterizationCachePurpose,
        contentBoundsHash: Int
    ) -> Bool {
        if let mask = layer.mask, subtreeHasAnimations(mask) {
            return false
        }
        if purpose == .explicit {
            return RasterizationDecisions.canReuseRasterizedTexture(
                layer: layer,
                cached: entry,
                currentContentBoundsHash: contentBoundsHash
            )
        }
        guard !layer._dirtyMask.contains(.rasterization),
              layer._subtreeDirtyCount == 0 else {
            return false
        }
        return entry.contentBoundsHash == contentBoundsHash
    }

    private func rasterizationCaptureBounds(for layer: CALayer) -> CGRect? {
        let bounds = rasterizationSubtreeBounds(for: layer)
        guard !bounds.isNull,
              !bounds.isInfinite,
              bounds.minX.isFinite,
              bounds.minY.isFinite,
              bounds.width.isFinite,
              bounds.height.isFinite,
              bounds.width > 0,
              bounds.height > 0 else {
            return nil
        }
        return bounds
    }

    private func rasterizationSubtreeBounds(for layer: CALayer) -> CGRect {
        let presentation = renderPresentation(for: layer)
        guard !presentation.isHidden, presentation.opacity > 0 else {
            return .null
        }

        let contentBounds = CGRect(origin: .zero, size: presentation.bounds.size)
        var subtreeBounds = contentBounds
        if !presentation.masksToBounds {
            forEachPrepassSublayer(
                of: layer,
                presentationLayer: presentation,
                modelMatrix: .identity
            ) { sublayer, sublayerParentMatrix in
                let sublayerPresentation = renderPresentation(for: sublayer)
                let sublayerBounds = rasterizationSubtreeBounds(for: sublayer)
                if sublayerBounds.isNull { return }
                guard let projectedBounds = projectedRasterizationBounds(
                    sublayerBounds,
                    matrix: sublayerPresentation.modelMatrix(
                        parentMatrix: sublayerParentMatrix
                    )
                ) else {
                    subtreeBounds = .infinite
                    return
                }
                subtreeBounds = subtreeBounds.union(projectedBounds)
            }
        }

        guard hasVisibleShadow(presentation) else { return subtreeBounds }
        let shadowSourceBounds = presentation.shadowPath?.boundingBox ?? subtreeBounds
        let blurPadding = max(0, presentation.shadowRadius * 2)
        let shadowBounds = shadowSourceBounds
            .offsetBy(dx: presentation.shadowOffset.width, dy: presentation.shadowOffset.height)
            .insetBy(dx: -blurPadding, dy: -blurPadding)
        return subtreeBounds.union(shadowBounds)
    }

    private func projectedRasterizationBounds(
        _ bounds: CGRect,
        matrix: Matrix4x4
    ) -> CGRect? {
        let corners = [
            SIMD4<Float>(Float(bounds.minX), Float(bounds.minY), 0, 1),
            SIMD4<Float>(Float(bounds.maxX), Float(bounds.minY), 0, 1),
            SIMD4<Float>(Float(bounds.minX), Float(bounds.maxY), 0, 1),
            SIMD4<Float>(Float(bounds.maxX), Float(bounds.maxY), 0, 1),
        ]
        var minimum = SIMD2<Float>(repeating: .infinity)
        var maximum = SIMD2<Float>(repeating: -.infinity)
        var positiveW: Bool?

        for corner in corners {
            let projected = matrix * corner
            guard projected.x.isFinite,
                  projected.y.isFinite,
                  projected.w.isFinite,
                  abs(projected.w) > 0.000001 else {
                return nil
            }
            let isPositive = projected.w > 0
            if let positiveW, positiveW != isPositive {
                return nil
            }
            positiveW = isPositive
            let point = SIMD2<Float>(projected.x / projected.w, projected.y / projected.w)
            minimum = SIMD2<Float>(Swift.min(minimum.x, point.x), Swift.min(minimum.y, point.y))
            maximum = SIMD2<Float>(Swift.max(maximum.x, point.x), Swift.max(maximum.y, point.y))
        }

        let result = CGRect(
            x: CGFloat(minimum.x),
            y: CGFloat(minimum.y),
            width: CGFloat(maximum.x - minimum.x),
            height: CGFloat(maximum.y - minimum.y)
        )
        guard !result.isInfinite,
              result.minX.isFinite,
              result.minY.isFinite,
              result.width.isFinite,
              result.height.isFinite else {
            return nil
        }
        return result
    }

    private func executeRasterizedFilterStages(
        _ stages: [LayerFilterStage],
        inputTexture: GPUTexture,
        width: Int,
        height: Int,
        encoder: GPUCommandEncoder
    ) -> GPUTexture? {
        guard let device else { return nil }

        let resources = FilterLayerResources(
            device: device,
            width: width,
            height: height,
            format: preferredFormat
        )
        transientRasterizationFilterResources.append(resources)

        var currentTexture = inputTexture
        var currentAlphaMode = FilterTextureAlphaMode.premultiplied
        var conversionIndex = 0

        func convert(to targetMode: FilterTextureAlphaMode) -> Bool {
            guard currentAlphaMode != targetMode else { return true }
            let outputTexture = currentTexture === resources.resultTexture
                ? resources.sourceTexture
                : resources.resultTexture
            let uniformBuffer = resources.uniformBuffer(
                forOperationAt: stages.count + conversionIndex,
                device: device
            )
            conversionIndex += 1
            guard applyAlphaConversion(
                from: currentAlphaMode,
                inputTexture: currentTexture,
                outputTexture: outputTexture,
                uniformBuffer: uniformBuffer,
                resources: resources,
                encoder: encoder
            ) else { return false }
            currentTexture = outputTexture
            currentAlphaMode = targetMode
            return true
        }

        for (stageIndex, stage) in stages.enumerated() {
            let uniformBuffer = resources.uniformBuffer(
                forOperationAt: stageIndex,
                device: device
            )

            switch stage {
            case let .renderer(operation):
                guard convert(to: .premultiplied) else { return nil }
                let outputTexture = currentTexture === resources.resultTexture
                    ? resources.sourceTexture
                    : resources.resultTexture
                let applied: Bool
                switch operation {
                case let .gaussianBlur(radius):
                    if radius <= 0 { continue }
                    applied = applyBlurFilter(
                        inputTexture: currentTexture,
                        intermediateTexture: resources.intermediateTexture,
                        outputTexture: outputTexture,
                        radius: radius,
                        uniformBuffer: uniformBuffer,
                        resources: resources,
                        encoder: encoder
                    )
                case .brightness, .contrast, .saturation, .colorInvert:
                    applied = applyColorFilter(
                        operation,
                        inputTexture: currentTexture,
                        outputTexture: outputTexture,
                        uniformBuffer: uniformBuffer,
                        resources: resources,
                        encoder: encoder
                    )
                }
                guard applied else { return nil }
                currentTexture = outputTexture

            case let .coreImage(filter):
                guard convert(to: .straight), let processor = layerFilterProcessor else {
                    return nil
                }
                do {
                    let execution = try processor.makeExecution(
                        filter: filter,
                        inputMode: .singleInput,
                        inputTexture: currentTexture,
                        width: UInt32(width),
                        height: UInt32(height)
                    )
                    try execution.encode(commandEncoder: encoder)
                    activeLayerFilterExecutions.append(execution)
                    currentTexture = execution.outputTexture
                    currentAlphaMode = .straight
                } catch {
                    return nil
                }
            }
        }

        guard convert(to: .premultiplied) else { return nil }

        let finalTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: preferredFormat,
            usage: [.renderAttachment, .textureBinding]
        ))
        guard applyFilterOperation(
            uniforms: FilterCompositeUniforms(opacity: 1),
            inputTexture: currentTexture,
            outputView: finalTexture.createView(),
            uniformBuffer: resources.compositeUniformBuffer,
            encoder: encoder
        ) else {
            transientRasterizationTextures.append(finalTexture)
            return nil
        }
        return finalTexture
    }

    private func executeRasterizedContentMask(
        _ maskLayer: CALayer,
        contentTexture: GPUTexture,
        projectionMatrix: Matrix4x4,
        renderSize: CGSize,
        depthStencilView: GPUTextureView,
        width: Int,
        height: Int,
        encoder: GPUCommandEncoder
    ) -> GPUTexture? {
        guard let device,
              let compositionMaskApplyPipeline,
              let maskStages = layerFilterStages(
                from: renderPresentation(for: maskLayer).filters ?? []
              ) else { return nil }

        let resources = FilterLayerResources(
            device: device,
            width: width,
            height: height,
            format: preferredFormat
        )
        transientRasterizationFilterResources.append(resources)
        let target = LayerPrepassTarget(
            layer: maskLayer,
            presentationLayer: renderPresentation(for: maskLayer),
            parentMatrix: projectionMatrix,
            renderKey: renderKey(for: maskLayer),
            timeOffset: currentReplicatorTimeOffset
        )
        let usesPrerasterizedMaskRoot = prerasterizedTextures[target.renderKey] != nil
        let executesRootMaskStages = !usesPrerasterizedMaskRoot && !maskStages.isEmpty
        guard renderRawCompositionContentMask(
            target,
            outputView: resources.sourceView,
            suppressRootFilters: executesRootMaskStages,
            renderSize: renderSize,
            depthStencilView: depthStencilView,
            encoder: encoder
        ) else { return nil }

        let maskTexture: GPUTexture
        if !executesRootMaskStages {
            maskTexture = resources.sourceTexture
        } else {
            guard let filteredMask = executeRasterizedFilterStages(
                maskStages,
                inputTexture: resources.sourceTexture,
                width: width,
                height: height,
                encoder: encoder
            ) else { return nil }
            transientRasterizationTextures.append(filteredMask)
            maskTexture = filteredMask
        }

        let outputTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: preferredFormat,
            usage: [.renderAttachment, .textureBinding]
        ))
        guard encodeCompositionMaskOperation(
            pipeline: compositionMaskApplyPipeline,
            firstView: contentTexture.createView(),
            secondView: maskTexture.createView(),
            outputView: outputTexture.createView(),
            encoder: encoder
        ) else {
            outputTexture.destroy()
            return nil
        }
        transientRasterizationTextures.append(contentTexture)
        return outputTexture
    }

    private func executeRasterizedShadow(
        layer: CALayer,
        contentTexture: GPUTexture,
        captureBounds: CGRect,
        width: Int,
        height: Int,
        encoder: GPUCommandEncoder
    ) -> GPUTexture? {
        guard let device,
              let bindGroupLayout,
              let shadowMaskPipeline,
              let shadowBlurHorizontalPipeline,
              let shadowBlurVerticalPipeline,
              let shadowBindGroupLayout,
              let rasterizedShadowCompositePipeline,
              let blurSampler else { return nil }

        let resources = ShadowLayerResources(
            device: device,
            width: width,
            height: height,
            format: preferredFormat
        )
        transientRasterizationShadowResources.append(resources)

        let horizontalInputView: GPUTextureView
        if let shadowPath = layer.shadowPath {
            var maskUniforms = CARendererUniforms(
                mvpMatrix: Matrix4x4.orthographic(
                    left: Float(captureBounds.minX),
                    right: Float(captureBounds.maxX),
                    bottom: Float(captureBounds.minY),
                    top: Float(captureBounds.maxY),
                    near: -1000,
                    far: 1000
                ),
                opacity: 1,
                layerSize: SIMD2(Float(captureBounds.width), Float(captureBounds.height))
            )
            device.queue.writeBuffer(
                resources.maskUniformBuffer,
                bufferOffset: 0,
                data: createFloat32Array(from: &maskUniforms)
            )
            let maskBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                layout: bindGroupLayout,
                entries: [
                    GPUBindGroupEntry(
                        binding: 0,
                        resource: .buffer(
                            resources.maskUniformBuffer,
                            offset: 0,
                            size: Self.alignedUniformSize
                        )
                    )
                ]
            ))
            let white = SIMD4<Float>(repeating: 1)
            var vertices: [CARendererVertex] = []
            for polyline in flattenPath(shadowPath) where polyline.count >= 3 {
                for index in triangulatePolygon(polyline) {
                    let point = polyline[index]
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: .zero,
                        color: white
                    ))
                }
            }
            let vertexBuffer = resources.ensureMaskVertexCapacity(vertices.count, device: device)
            if !vertices.isEmpty {
                device.queue.writeBuffer(
                    vertexBuffer,
                    bufferOffset: 0,
                    data: createFloat32Array(from: &vertices)
                )
            }
            let maskPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
                colorAttachments: [
                    GPURenderPassColorAttachment(
                        view: resources.maskView,
                        clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                        loadOp: .clear,
                        storeOp: .store
                    )
                ]
            ))
            maskPass.setPipeline(shadowMaskPipeline)
            maskPass.setBindGroup(0, bindGroup: maskBindGroup, dynamicOffsets: [0])
            if !vertices.isEmpty {
                maskPass.setVertexBuffer(0, buffer: vertexBuffer, offset: 0)
                maskPass.draw(vertexCount: UInt32(vertices.count))
            }
            maskPass.end()
            horizontalInputView = resources.maskView
        } else {
            horizontalInputView = contentTexture.createView()
        }

        var blurUniforms = BlurUniforms(
            texelSize: SIMD2<Float>(1 / Float(width), 1 / Float(height)),
            blurRadius: max(0, Float(layer.shadowRadius)) * 0.5
        )
        device.queue.writeBuffer(
            resources.blurUniformBuffer,
            bufferOffset: 0,
            data: createFloat32Array(from: &blurUniforms)
        )
        let horizontalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(
                    resources.blurUniformBuffer,
                    offset: 0,
                    size: UInt64(MemoryLayout<BlurUniforms>.stride)
                )),
                GPUBindGroupEntry(binding: 1, resource: .textureView(horizontalInputView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))
        let verticalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(
                    resources.blurUniformBuffer,
                    offset: 0,
                    size: UInt64(MemoryLayout<BlurUniforms>.stride)
                )),
                GPUBindGroupEntry(binding: 1, resource: .textureView(resources.intermediateView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))
        let horizontalPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: resources.intermediateView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))
        horizontalPass.setPipeline(shadowBlurHorizontalPipeline)
        horizontalPass.setBindGroup(0, bindGroup: horizontalBindGroup)
        horizontalPass.draw(vertexCount: 6)
        horizontalPass.end()

        let verticalPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: resources.maskView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))
        verticalPass.setPipeline(shadowBlurVerticalPipeline)
        verticalPass.setBindGroup(0, bindGroup: verticalBindGroup)
        verticalPass.draw(vertexCount: 6)
        verticalPass.end()

        let colorComponents = layer.shadowColor?.components ?? []
        let color = SIMD4<Float>(
            colorComponents.count > 0 ? Float(colorComponents[0]) : 0,
            colorComponents.count > 1 ? Float(colorComponents[1]) : 0,
            colorComponents.count > 2 ? Float(colorComponents[2]) : 0,
            (colorComponents.count > 3 ? Float(colorComponents[3]) : 1) * layer.shadowOpacity
        )
        var compositeUniforms = RasterShadowCompositeUniforms(
            shadowColor: color,
            shadowOffsetUV: SIMD2<Float>(
                Float(layer.shadowOffset.width / captureBounds.width),
                Float(-layer.shadowOffset.height / captureBounds.height)
            )
        )
        device.queue.writeBuffer(
            resources.compositeUniformBuffer,
            bufferOffset: 0,
            data: createFloat32Array(from: &compositeUniforms)
        )

        let finalTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: preferredFormat,
            usage: [.renderAttachment, .textureBinding]
        ))
        let compositeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: rasterizedShadowCompositePipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(
                    resources.compositeUniformBuffer,
                    offset: 0,
                    size: UInt64(MemoryLayout<RasterShadowCompositeUniforms>.stride)
                )),
                GPUBindGroupEntry(binding: 1, resource: .textureView(contentTexture.createView())),
                GPUBindGroupEntry(binding: 2, resource: .textureView(resources.maskView)),
                GPUBindGroupEntry(binding: 3, resource: .sampler(blurSampler))
            ]
        ))
        let compositePass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: finalTexture.createView(),
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))
        compositePass.setPipeline(rasterizedShadowCompositePipeline)
        compositePass.setBindGroup(0, bindGroup: compositeBindGroup)
        compositePass.draw(vertexCount: 6)
        compositePass.end()
        return finalTexture
    }

    /// Hash of the inputs that determine the captured pixels. Used by
    /// `RasterizationDecisions.canReuseRasterizedTexture` to detect
    /// content invalidation independent of the dirty-bit pathway.
    ///
    /// Excludes `layer.transform` per PERFORMANCE_DESIGN.md §5.6: the
    /// layer's own transform is a uniform applied at composite, not part
    /// of the captured pixels — including it would force a recapture on
    /// every transform-only change (e.g. scrolling), defeating the
    /// cache. Internal layout is captured by `bounds` alone.
    private func rasterizationContentBoundsHash(
        for layer: CALayer,
        captureBounds: CGRect
    ) -> Int {
        let presentationLayer = renderPresentation(for: layer)
        var hasher = Hasher()
        hasher.combine(presentationLayer.bounds.origin.x)
        hasher.combine(presentationLayer.bounds.origin.y)
        hasher.combine(presentationLayer.bounds.size.width)
        hasher.combine(presentationLayer.bounds.size.height)
        hasher.combine(presentationLayer.rasterizationScale)
        hasher.combine(captureBounds.minX)
        hasher.combine(captureBounds.minY)
        hasher.combine(captureBounds.width)
        hasher.combine(captureBounds.height)
        if let mask = layer.mask {
            var visited: Set<ObjectIdentifier> = []
            combineDetachedContentRevision(
                of: mask,
                into: &hasher,
                visited: &visited
            )
        }
        if hasVisibleShadow(presentationLayer) {
            hasher.combine(presentationLayer.shadowOpacity)
            hasher.combine(presentationLayer.shadowOffset.width)
            hasher.combine(presentationLayer.shadowOffset.height)
            hasher.combine(presentationLayer.shadowRadius)
            if let shadowPath = presentationLayer.shadowPath {
                let pathBounds = shadowPath.boundingBox
                hasher.combine(pathBounds.origin.x)
                hasher.combine(pathBounds.origin.y)
                hasher.combine(pathBounds.size.width)
                hasher.combine(pathBounds.size.height)
            }
            for component in presentationLayer.shadowColor?.components ?? [] {
                hasher.combine(component)
            }
        }
        return hasher.finalize()
    }

    /// Adds every identity and model mutation in a detached dependency tree.
    /// A mask has no superlayer, so ordinary subtree dirty propagation cannot
    /// invalidate the masked layer's raster cache when the mask changes.
    private func combineDetachedContentRevision(
        of layer: CALayer,
        into hasher: inout Hasher,
        visited: inout Set<ObjectIdentifier>
    ) {
        let identifier = ObjectIdentifier(layer)
        guard visited.insert(identifier).inserted else {
            hasher.combine(identifier)
            return
        }
        hasher.combine(identifier)
        hasher.combine(layer._contentRevision)
        if let mask = layer.mask {
            combineDetachedContentRevision(of: mask, into: &hasher, visited: &visited)
        }
        for sublayer in layer.sublayers ?? [] {
            combineDetachedContentRevision(of: sublayer, into: &hasher, visited: &visited)
        }
    }

    /// Composites a captured rasterization texture as a quad placed at
    /// the layer's transform, sized to the layer's bounds, with the
    /// layer's *current* opacity (R3.3 composite path).
    ///
    /// Uses the premultiplied textured pipeline because the capture is a
    /// render-target texture whose RGB channels already contain alpha. The quad
    /// vertices are emitted in normalised layer-bounds coordinates
    /// `[0, 1]`; the MVP matrix scales them by `bounds.size` and
    /// applies the layer's `modelMatrix` to land at the right screen
    /// position. Texture V is flipped to bridge Y-up world / Y-down
    /// texture rows (mirrors `renderContentsLayer`).
    private func renderRasterizedLayerComposite(
        _ layer: CALayer,
        presentationLayer: CALayer,
        prerasterized: PrerasterizedTexture,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        let captureBounds = prerasterized.captureBounds
        guard captureBounds.width > 0, captureBounds.height > 0 else { return }

        // The opacity stack already contains parent opacity × this layer's current
        // opacity. Capture baked the root at 1.0, so applying the stack once here
        // restores the model value without squaring the layer opacity.
        let composite = currentEffectiveOpacity

        // Quad in normalised layer-bounds space. UVs V-flip to convert
        // between Y-up world coords and Y-down texture rows — same
        // convention used by `renderContentsLayer`.
        let white = currentReplicatorColor
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else {
            shadowRenderFailureCount += 1
            return
        }
        let (vertexOffset, layerIndex) = allocation

        // Scale the [0, 1] quad to bounds.size, then apply the layer's
        // own modelMatrix to position/rotate/scale it into the world.
        let originMatrix = Matrix4x4(translation: SIMD3<Float>(
            Float(captureBounds.minX),
            Float(captureBounds.minY),
            0
        ))
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(captureBounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(captureBounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * originMatrix * scaleMatrix

        // cornerRadius is already baked into the captured texture by
        // the capture pass — passing 0 here avoids double-applying it.
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: composite,
            cornerRadius: 0,
            layerSize: SIMD2<Float>(Float(captureBounds.width), Float(captureBounds.height)),
            cornerRadii: .zero
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Rasterized textures are owned by the layer's entry in
        // `prerasterizedTextures`/`rasterizationCache`, so key the bind
        // group cache by the layer's identity rather than the texture's.
        // The `.rasterizedLayer(...)` tag keeps this entry from aliasing a CGImage
        // entry whose `ObjectIdentifier` happens to match this layer's
        // (post-dealloc) heap address.
        let texturedBindGroup = cachedTexturedBindGroup(
            cacheKey: .rasterizedLayer(renderKey(for: layer), prerasterized.purpose),
            gpuTexture: prerasterized.texture,
            device: device,
            layout: texturedBindGroupLayout,
            sampler: textureSampler,
            uniformBuffer: uniformBuffer,
            uniformStride: UInt64(MemoryLayout<TexturedUniforms>.stride)
        )

        // Captured render-target textures are premultiplied. Using the regular
        // contents pipeline here would multiply RGB by alpha a second time.
        if let selected = selectPremultipliedTexturedPipeline() {
            renderPass.setPipeline(selected)
        }
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Restore the regular pipeline so subsequent per-layer draws in
        // the main pass continue with the right state.
        if let pipeline = pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    private func applyBlurFilter(
        inputTexture: GPUTexture,
        intermediateTexture: GPUTexture,
        outputTexture: GPUTexture,
        radius: CGFloat,
        uniformBuffer: GPUBuffer,
        resources: FilterLayerResources,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let device = device,
              let filterBlurHorizontalPipeline = filterBlurHorizontalPipeline,
              let filterBlurVerticalPipeline = filterBlurVerticalPipeline,
              let shadowBindGroupLayout = shadowBindGroupLayout,
              let blurSampler = blurSampler,
              let intermediateView = resources.view(for: intermediateTexture),
              let outputView = resources.view(for: outputTexture) else { return false }
        let inputView = inputTexture.createView()

        let inputBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(uniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(inputView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))
        let intermediateBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(uniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(intermediateView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        var blurUniforms = BlurUniforms(
            texelSize: SIMD2<Float>(
                1.0 / Float(resources.width),
                1.0 / Float(resources.height)
            ),
            blurRadius: Float(radius)
        )
        let blurUniformData = createFloat32Array(from: &blurUniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: 0, data: blurUniformData)

        let hBlurPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: intermediateView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        hBlurPass.setPipeline(filterBlurHorizontalPipeline)
        hBlurPass.setBindGroup(0, bindGroup: inputBindGroup)
        hBlurPass.draw(vertexCount: 6)
        hBlurPass.end()

        let vBlurPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: outputView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        vBlurPass.setPipeline(filterBlurVerticalPipeline)
        vBlurPass.setBindGroup(0, bindGroup: intermediateBindGroup)
        vBlurPass.draw(vertexCount: 6)
        vBlurPass.end()
        return true
    }

    private func applyColorFilter(
        _ operation: CAFilterOperation,
        inputTexture: GPUTexture,
        outputTexture: GPUTexture,
        uniformBuffer: GPUBuffer,
        resources: FilterLayerResources,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let uniforms = filterCompositeUniforms(for: operation),
              let outputView = resources.view(for: outputTexture) else { return false }
        return applyFilterOperation(
            uniforms: uniforms,
            inputTexture: inputTexture,
            outputView: outputView,
            uniformBuffer: uniformBuffer,
            encoder: encoder
        )
    }

    private func applyAlphaConversion(
        from sourceMode: FilterTextureAlphaMode,
        inputTexture: GPUTexture,
        outputTexture: GPUTexture,
        uniformBuffer: GPUBuffer,
        resources: FilterLayerResources,
        encoder: GPUCommandEncoder
    ) -> Bool {
        let filterType: Float = sourceMode == .premultiplied ? 5 : 6
        guard let outputView = resources.view(for: outputTexture) else { return false }
        return applyFilterOperation(
            uniforms: FilterCompositeUniforms(filterType: filterType),
            inputTexture: inputTexture,
            outputView: outputView,
            uniformBuffer: uniformBuffer,
            encoder: encoder
        )
    }

    private func applyAlphaConversion(
        from sourceMode: FilterTextureAlphaMode,
        inputTexture: GPUTexture,
        outputView: GPUTextureView,
        uniformBuffer: GPUBuffer,
        encoder: GPUCommandEncoder
    ) -> Bool {
        let filterType: Float = sourceMode == .premultiplied ? 5 : 6
        return applyFilterOperation(
            uniforms: FilterCompositeUniforms(filterType: filterType),
            inputTexture: inputTexture,
            outputView: outputView,
            uniformBuffer: uniformBuffer,
            encoder: encoder
        )
    }

    private func applyFilterOperation(
        uniforms initialUniforms: FilterCompositeUniforms,
        inputTexture: GPUTexture,
        outputView: GPUTextureView,
        uniformBuffer: GPUBuffer,
        encoder: GPUCommandEncoder
    ) -> Bool {
        guard let device = device,
              let filterOperationPipeline = filterOperationPipeline,
              let blurSampler = blurSampler else { return false }
        let inputView = inputTexture.createView()

        var uniforms = initialUniforms
        let filterUniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: 0, data: filterUniformData)

        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: filterOperationPipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(uniformBuffer, offset: 0, size: UInt64(MemoryLayout<FilterCompositeUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(inputView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: outputView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        renderPass.setPipeline(filterOperationPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup)
        renderPass.draw(vertexCount: 6)
        renderPass.end()
        return true
    }

    private func filterCompositeUniforms(for operation: CAFilterOperation) -> FilterCompositeUniforms? {
        switch operation {
        case let .brightness(amount):
            return FilterCompositeUniforms(opacity: 1.0, filterType: 1.0, parameter0: Float(amount))
        case let .contrast(amount):
            return FilterCompositeUniforms(opacity: 1.0, filterType: 2.0, parameter0: Float(amount))
        case let .saturation(amount):
            return FilterCompositeUniforms(opacity: 1.0, filterType: 3.0, parameter0: Float(amount))
        case .colorInvert:
            return FilterCompositeUniforms(opacity: 1.0, filterType: 4.0)
        case .gaussianBlur:
            return nil
        }
    }

    /// Collects visible filtered layers in descendant-first order.
    private func collectFilteredLayers(
        _ layer: CALayer,
        parentMatrix: Matrix4x4,
        visitedRenderKeys: inout Set<LayerRenderKey>,
        into result: inout [LayerPrepassTarget]
    ) {
        let presentationLayer = renderPresentation(for: layer)

        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else {
            return
        }

        let key = renderKey(for: layer)
        guard visitedRenderKeys.insert(key).inserted else { return }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)
        if let maskLayer = presentationLayer.mask {
            collectFilteredLayers(
                maskLayer,
                parentMatrix: modelMatrix,
                visitedRenderKeys: &visitedRenderKeys,
                into: &result
            )
        }
        forEachPrepassSublayer(
            of: layer,
            presentationLayer: presentationLayer,
            modelMatrix: modelMatrix
        ) { sublayer, sublayerParentMatrix in
            collectFilteredLayers(
                sublayer,
                parentMatrix: sublayerParentMatrix,
                visitedRenderKeys: &visitedRenderKeys,
                into: &result
            )
        }

        if !(layer is CATransformLayer),
           presentationLayer.filters?.isEmpty == false
            || requiresGroupOpacity(presentationLayer)
            || presentationLayer.compositingFilter != nil
            || presentationLayer.backgroundFilters?.isEmpty == false
            || presentationLayer.mask != nil {
            result.append(LayerPrepassTarget(
                layer: layer,
                presentationLayer: presentationLayer,
                parentMatrix: parentMatrix,
                renderKey: key,
                timeOffset: currentReplicatorTimeOffset
            ))
        }
    }

    /// Collects visible shadow-producing layers in main-pass render order.
    private func collectShadowLayers(
        _ layer: CALayer,
        parentMatrix: Matrix4x4,
        into result: inout [LayerPrepassTarget]
    ) {
        let presentationLayer = renderPresentation(for: layer)

        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else {
            return
        }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        if !(layer is CATransformLayer),
           presentationLayer.shadowOpacity > 0,
           presentationLayer.shadowColor != nil {
            result.append(LayerPrepassTarget(
                layer: layer,
                presentationLayer: presentationLayer,
                parentMatrix: parentMatrix,
                renderKey: renderKey(for: layer),
                timeOffset: currentReplicatorTimeOffset
            ))
        }

        forEachPrepassSublayer(
            of: layer,
            presentationLayer: presentationLayer,
            modelMatrix: modelMatrix
        ) { sublayer, sublayerParentMatrix in
            collectShadowLayers(
                sublayer,
                parentMatrix: sublayerParentMatrix,
                into: &result
            )
        }
    }

    /// Renders the shadow for a layer using the pre-blurred shadow texture.
    private func renderLayerShadow(
        _ modelLayer: CALayer,
        presentationLayer layer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder
    ) {
        guard let shadowColor = layer.shadowColor,
              let vertexBuffer = vertexBuffer,
              let prerendered = prerenderedShadows[renderKey(for: modelLayer)],
              let shadowCompositePipeline = shadowCompositePipeline,
              let blurSampler = blurSampler else {
            shadowRenderFailureCount += 1
            return
        }

        let shadowOpacity = layer.shadowOpacity
        let shadowOffset = layer.shadowOffset

        let effectiveOpacity = currentEffectiveOpacity
        let colorComponents: SIMD4<Float>
        if let components = shadowColor.components, components.count >= 3 {
            colorComponents = replicatedColor(SIMD4<Float>(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                (components.count > 3 ? Float(components[3]) * shadowOpacity : shadowOpacity) * effectiveOpacity
            ))
        } else {
            colorComponents = replicatedColor(
                SIMD4<Float>(0, 0, 0, shadowOpacity * effectiveOpacity)
            )
        }

        var shadowUniforms = ShadowUniforms(
            mvpMatrix: Matrix4x4.orthographic(
                left: 0,
                right: Float(size.width),
                bottom: 0,
                top: Float(size.height),
                near: -1000,
                far: 1000
            ),
            shadowColor: colorComponents,
            shadowOffset: SIMD2<Float>(Float(shadowOffset.width), Float(shadowOffset.height)),
            layerSize: SIMD2<Float>(Float(size.width), Float(size.height))
        )
        let shadowUniformData = createFloat32Array(from: &shadowUniforms)
        device.queue.writeBuffer(
            prerendered.resources.compositeUniformBuffer,
            bufferOffset: 0,
            data: shadowUniformData
        )

        let compositeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowCompositePipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(
                    prerendered.resources.compositeUniformBuffer,
                    offset: 0,
                    size: UInt64(MemoryLayout<ShadowUniforms>.stride)
                )),
                GPUBindGroupEntry(binding: 1, resource: .textureView(prerendered.resources.maskView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: colorComponents),
            CARendererVertex(position: SIMD2(Float(size.width), 0), texCoord: SIMD2(1, 1), color: colorComponents),
            CARendererVertex(position: SIMD2(0, Float(size.height)), texCoord: SIMD2(0, 0), color: colorComponents),
            CARendererVertex(position: SIMD2(Float(size.width), 0), texCoord: SIMD2(1, 1), color: colorComponents),
            CARendererVertex(position: SIMD2(Float(size.width), Float(size.height)), texCoord: SIMD2(1, 0), color: colorComponents),
            CARendererVertex(position: SIMD2(0, Float(size.height)), texCoord: SIMD2(0, 0), color: colorComponents),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else { return }
        let vertexOffset = allocation.vertexOffset

        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

        renderPass.setPipeline(
            maskNestingDepth > 0
                ? (shadowCompositeStencilPipeline ?? shadowCompositePipeline)
                : shadowCompositePipeline
        )
        renderPass.setBindGroup(0, bindGroup: compositeBindGroup)
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    /// Renders a filtered layer by compositing the pre-rendered filter texture.
    ///
    /// This method draws a full-screen quad textured with the filtered layer content
    /// from the filter pre-rendering pass.
    private func renderFilteredLayerComposite(
        _ layer: CALayer,
        prerendered: PrerenderedFilter,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        renderPremultipliedFullScreenTexture(
            prerendered.outputView,
            uniformBuffer: prerendered.resources.compositeUniformBuffer,
            uniforms: FilterCompositeUniforms(
                opacity: currentEffectiveOpacity,
                colorMultiplier: currentReplicatorColor
            ),
            device: device,
            renderPass: renderPass
        )
    }

    private func renderPreparedComposition(
        _ composition: PrerenderedComposition,
        presentationLayer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        if transformDepthNesting > 0 || rasterizePrerenderRootLayer != nil {
            renderPreparedCompositionPlane(
                composition,
                presentationLayer: presentationLayer,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix
            )
            return
        }
        guard let filterReplacementPipeline,
              let blurSampler else { return }

        var uniforms = FilterCompositeUniforms()
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            composition.resources.displayUniformBuffer,
            bufferOffset: 0,
            data: uniformData
        )
        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: filterReplacementPipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(
                    composition.resources.displayUniformBuffer,
                    offset: 0,
                    size: UInt64(MemoryLayout<FilterCompositeUniforms>.stride)
                )),
                GPUBindGroupEntry(binding: 1, resource: .textureView(composition.outputView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler)),
            ]
        ))

        renderPass.setPipeline(filterReplacementPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup)
        renderPass.draw(vertexCount: 6)
    }

    private func renderPreparedCompositionPlane(
        _ composition: PrerenderedComposition,
        presentationLayer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        let selectedPipeline: GPURenderPipeline?
        if transformDepthNesting > 0 {
            selectedPipeline = maskNestingDepth > 0
                ? transformedCompositionStencilPipeline
                : transformedCompositionPipeline
        } else {
            selectedPipeline = maskNestingDepth > 0
                ? capturedCompositionStencilPipeline
                : capturedCompositionPipeline
        }
        guard let selectedPipeline,
              let blurSampler,
              let vertexBuffer else { return }
        let bounds = presentationLayer.bounds
        guard bounds.width > 0, bounds.height > 0 else { return }

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        var uniforms = TexturedUniforms(
            mvpMatrix: modelMatrix * scaleMatrix,
            layerSize: SIMD2<Float>(Float(size.width), Float(size.height))
        )
        device.queue.writeBuffer(
            composition.resources.transformedDisplayUniformBuffer,
            bufferOffset: 0,
            data: createFloat32Array(from: &uniforms)
        )

        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: selectedPipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(
                    composition.resources.transformedDisplayUniformBuffer,
                    offset: 0,
                    size: UInt64(MemoryLayout<TexturedUniforms>.stride)
                )),
                GPUBindGroupEntry(binding: 1, resource: .textureView(composition.outputView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler)),
            ]
        ))
        var vertices: [CARendererVertex]
        if rasterizePrerenderRootLayer != nil {
            guard let capturedVertices = capturedCompositionPlaneVertices(
                samplingMatrix: composition.samplingModelMatrix,
                bounds: bounds
            ) else { return }
            vertices = capturedVertices
        } else {
            let white = SIMD4<Float>(repeating: 1)
            vertices = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: .zero, color: white),
                CARendererVertex(position: SIMD2(1, 0), texCoord: .zero, color: white),
                CARendererVertex(position: SIMD2(0, 1), texCoord: .zero, color: white),
                CARendererVertex(position: SIMD2(1, 0), texCoord: .zero, color: white),
                CARendererVertex(position: SIMD2(1, 1), texCoord: .zero, color: white),
                CARendererVertex(position: SIMD2(0, 1), texCoord: .zero, color: white),
            ]
        }
        guard let allocation = allocateVertices(count: vertices.count) else { return }
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: allocation.vertexOffset,
            data: createFloat32Array(from: &vertices)
        )

        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup)
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: allocation.vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    private func capturedCompositionPlaneVertices(
        samplingMatrix: Matrix4x4,
        bounds: CGRect
    ) -> [CARendererVertex]? {
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let samplingMVP = samplingMatrix * scaleMatrix

        func vertex(at position: SIMD2<Float>) -> CARendererVertex? {
            let clip = samplingMVP * SIMD4<Float>(position.x, position.y, 0, 1)
            guard clip.x.isFinite,
                  clip.y.isFinite,
                  clip.w.isFinite,
                  abs(clip.w) > 0.000001 else {
                return nil
            }
            let viewportNumerator = SIMD2<Float>(
                (clip.x + clip.w) * 0.5,
                (clip.w - clip.y) * 0.5
            )
            return CARendererVertex(
                position: position,
                texCoord: viewportNumerator,
                color: SIMD4<Float>(clip.w, 0, 0, 0)
            )
        }

        guard let minMin = vertex(at: SIMD2(0, 0)),
              let maxMin = vertex(at: SIMD2(1, 0)),
              let minMax = vertex(at: SIMD2(0, 1)),
              let maxMax = vertex(at: SIMD2(1, 1)) else {
            return nil
        }
        return [minMin, maxMin, minMax, maxMin, maxMax, minMax]
    }

    private func renderPremultipliedFullScreenTexture(
        _ textureView: GPUTextureView,
        uniformBuffer: GPUBuffer,
        uniforms initialUniforms: FilterCompositeUniforms,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder
    ) {
        guard let filterCompositePipeline = filterCompositePipeline,
              let blurSampler = blurSampler else { return }

        var filterUniforms = initialUniforms
        let filterUniformData = createFloat32Array(from: &filterUniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: 0,
            data: filterUniformData
        )

        // Create composite bind group with the filtered texture.
        let compositeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: filterCompositePipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(uniformBuffer, offset: 0, size: UInt64(MemoryLayout<FilterCompositeUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(textureView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        renderPass.setPipeline(maskNestingDepth > 0 ? (filterCompositeStencilPipeline ?? filterCompositePipeline) : filterCompositePipeline)
        renderPass.setBindGroup(0, bindGroup: compositeBindGroup)
        renderPass.draw(vertexCount: 6)

        // Switch back to regular pipeline
        if let pipeline = pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    // MARK: - CATransformLayer Rendering

    /// Renders a CATransformLayer's sublayers without rendering the layer's own properties.
    ///
    /// CATransformLayer is different from regular layers in that:
    /// 1. It does not render its own content (backgroundColor, contents, etc.)
    /// 2. It does NOT flatten sublayers - they maintain their full 3D positions
    /// 3. Depth buffer z-testing is used for correct occlusion of overlapping layers
    ///
    /// Unlike regular layers where sublayers are composited as flat 2D images,
    /// CATransformLayer sublayers participate in true 3D depth testing.
    /// The depth buffer (z-buffer) handles occlusion correctly based on the
    /// transformed z-coordinates of each sublayer.
    private func renderTransformLayerSublayers(
        _ layer: CALayer,
        presentationLayer: CALayer,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard layer.sublayers != nil else { return }

        let beginsIndependentDepthGroup = transformDepthNesting == 0
        if beginsIndependentDepthGroup {
            guard let depthClearPipeline else { return }
            renderPass.setPipeline(depthClearPipeline)
            renderPass.draw(vertexCount: 3)
        }
        transformDepthNesting += 1
        defer { transformDepthNesting -= 1 }

        // `renderLayer` already applied this transform layer's presentation transform.
        // Only the sublayer transform and bounds origin remain before traversing children.
        let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)

        for sublayer in depthOrderedSublayers(for: layer, parentMatrix: sublayerMatrix) {
            self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
        }
    }

    // MARK: - CAEmitterLayer Rendering

    /// Rotates a 2D point around the center (0.5, 0.5) by the given angle in radians.
    private func rotatePoint(_ point: SIMD2<Float>, angle: Float) -> SIMD2<Float> {
        let center = SIMD2<Float>(0.5, 0.5)
        let p = point - center
        let cosA = cos(angle)
        let sinA = sin(angle)
        let rotated = SIMD2<Float>(
            p.x * cosA - p.y * sinA,
            p.x * sinA + p.y * cosA
        )
        return rotated + center
    }

    /// Renders a CAEmitterLayer with its particle system.
    private func renderEmitterLayer(
        _ modelLayer: CAEmitterLayer,
        presentation emitterLayer: CAEmitterLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }
        let emitterCells = emitterLayer.emitterCells ?? []

        let layerID = ObjectIdentifier(modelLayer)
        activeEmitterLayerIDs.insert(layerID)
        let state: EmitterLayerState
        if let existing = emitterLayerStates[layerID], existing.owner === modelLayer {
            state = existing
        } else {
            state = EmitterLayerState(owner: modelLayer, seed: emitterLayer.seed)
            emitterLayerStates[layerID] = state
        }
        if state.configuredSeed != emitterLayer.seed {
            state.randomSource.reset(seed: emitterLayer.seed)
            state.configuredSeed = emitterLayer.seed
        }
        let currentCellIDs = Set(emitterCells.map(ObjectIdentifier.init))
        state.birthRemainders = state.birthRemainders.filter { key, _ in
            switch key {
            case .root(let cellID): return currentCellIDs.contains(cellID)
            case .child: return true
            }
        }

        let frameToken = CALayer._currentFrameToken
        if state.lastUpdatedFrame != frameToken {
            let currentTime = CACurrentMediaTime()
            let deltaTime = state.lastUpdateTime > 0
                ? min(max(Float(currentTime - state.lastUpdateTime), 0), 0.1)
                : 1.0 / 60.0
            state.lastUpdateTime = currentTime
            state.lastUpdatedFrame = frameToken
            let previousSimulationTime = state.simulationTime
            state.simulationTime += CFTimeInterval(deltaTime)
            let existingParticleCount = state.particles.count

            for index in state.particles.indices {
                state.particles[index].update(deltaTime: deltaTime)
            }
            var projectedLiveParticleCount = state.particles.reduce(into: 0) { count, particle in
                if particle.isAlive { count += 1 }
            }

            for cell in emitterCells where cell.isEnabled {
                let cellID = ObjectIdentifier(cell)
                let activeDelta: Float
                do {
                    activeDelta = try EmitterCellSimulation.activeEmissionDelta(
                        for: cell,
                        from: previousSimulationTime,
                        to: state.simulationTime
                    )
                } catch {
                    emitterSpawnFailureCount += 1
                    continue
                }
                guard let particlesToSpawn = emitterParticleBirthCount(
                    cell: cell,
                    activeDelta: activeDelta,
                    rateMultiplier: emitterLayer.birthRate,
                    remainderKey: .root(cellID),
                    state: state
                ) else {
                    emitterSpawnFailureCount += 1
                    continue
                }

                for _ in 0..<particlesToSpawn {
                    guard projectedLiveParticleCount < Self.maxParticles else { break }
                    guard let position = EmitterGeometry.position(
                        shape: emitterLayer.emitterShape,
                        mode: emitterLayer.emitterMode,
                        position: emitterLayer.emitterPosition,
                        zPosition: emitterLayer.emitterZPosition,
                        size: emitterLayer.emitterSize,
                        depth: emitterLayer.emitterDepth,
                        random: &state.randomSource
                    ) else {
                        emitterSpawnFailureCount += 1
                        continue
                    }
                    guard var particle = makeEmitterParticle(
                        cell: cell,
                        position: position,
                        parentDirection: nil,
                        inheritedColor: SIMD4(1, 1, 1, 1),
                        inheritedScale: 1,
                        generation: 0,
                        emitterLayer: emitterLayer,
                        state: state
                    ) else {
                        emitterSpawnFailureCount += 1
                        continue
                    }
                    guard particle.isAlive else { continue }
                    assignBirthSequence(to: &particle, state: state)
                    state.particles.append(particle)
                    projectedLiveParticleCount += 1
                }
            }

            var childParticles: [EmitterParticle] = []
            for parentIndex in 0..<existingParticleCount {
                let parent = state.particles[parentIndex]
                let parentStartTime = CFTimeInterval(
                    max(0, parent.maxLifetime - parent.previousLifetime)
                )
                let parentEndTime = CFTimeInterval(
                    max(0, parent.maxLifetime - parent.lifetime)
                )
                guard parentEndTime > parentStartTime else { continue }

                for childCell in parent.emitterCells where childCell.isEnabled {
                    let activeDelta: Float
                    do {
                        activeDelta = try EmitterCellSimulation.activeEmissionDelta(
                            for: childCell,
                            from: parentStartTime,
                            to: parentEndTime
                        )
                    } catch {
                        emitterSpawnFailureCount += 1
                        continue
                    }
                    guard let particlesToSpawn = emitterParticleBirthCount(
                        cell: childCell,
                        activeDelta: activeDelta,
                        rateMultiplier: emitterLayer.birthRate,
                        remainderKey: .child(
                            parentBirthSequence: parent.birthSequence,
                            cell: ObjectIdentifier(childCell)
                        ),
                        state: state
                    ) else {
                        emitterSpawnFailureCount += 1
                        continue
                    }

                    for childIndex in 0..<particlesToSpawn {
                        guard projectedLiveParticleCount < Self.maxParticles else {
                            break
                        }
                        let fraction = Float(childIndex + 1) / Float(particlesToSpawn + 1)
                        let position = parent.previousPosition
                            + (parent.position - parent.previousPosition) * fraction
                        let inheritedColor = parent.previousColor
                            + (parent.color - parent.previousColor) * fraction
                        let inheritedScale = parent.previousScale
                            + (parent.scale - parent.previousScale) * fraction
                        guard var child = makeEmitterParticle(
                            cell: childCell,
                            position: position,
                            parentDirection: parent.emissionDirection,
                            inheritedColor: inheritedColor,
                            inheritedScale: inheritedScale,
                            generation: parent.generation + 1,
                            emitterLayer: emitterLayer,
                            state: state
                        ) else {
                            emitterSpawnFailureCount += 1
                            continue
                        }
                        guard child.isAlive else { continue }
                        assignBirthSequence(to: &child, state: state)
                        childParticles.append(child)
                        projectedLiveParticleCount += 1
                    }
                }
            }
            state.particles.removeAll { !$0.isAlive }
            state.particles.append(contentsOf: childParticles)
            let liveBirthSequences = Set(state.particles.map(\.birthSequence))
            state.birthRemainders = state.birthRemainders.filter { key, _ in
                switch key {
                case .root(let cellID): return currentCellIDs.contains(cellID)
                case .child(let parentBirthSequence, _):
                    return liveBirthSequences.contains(parentBirthSequence)
                }
            }
        }
        state.lastRenderedBirthSequences.removeAll(keepingCapacity: true)
        state.lastRenderUsedAdditiveBlending = false

        func draw(_ particle: EmitterParticle, additive: Bool = false) {
            guard particle.isAlive else { return }
            if renderEmitterParticle(
                particle,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix,
                vertexBuffer: vertexBuffer,
                uniformBuffer: uniformBuffer,
                additive: additive
            ) {
                state.lastRenderedBirthSequences.append(particle.birthSequence)
            }
        }

        switch emitterLayer.renderMode {
        case .unordered, .oldestFirst:
            for particle in state.particles {
                draw(particle)
            }
        case .oldestLast:
            for particle in state.particles.reversed() {
                draw(particle)
            }
        case .backToFront:
            let indices = state.particles.indices.sorted { lhs, rhs in
                let lhsParticle = state.particles[lhs]
                let rhsParticle = state.particles[rhs]
                if lhsParticle.position.z == rhsParticle.position.z {
                    return lhsParticle.birthSequence < rhsParticle.birthSequence
                }
                return lhsParticle.position.z < rhsParticle.position.z
            }
            for index in indices {
                draw(state.particles[index])
            }
        case .additive:
            let additivePipeline: GPURenderPipeline?
            if transformDepthNesting > 0 {
                additivePipeline = maskNestingDepth > 0
                    ? emitterTexturedAdditiveDepthStencilPipeline
                    : emitterTexturedAdditiveDepthPipeline
            } else {
                additivePipeline = maskNestingDepth > 0
                    ? emitterTexturedAdditiveStencilPipeline
                    : emitterTexturedAdditivePipeline
            }
            guard additivePipeline != nil else {
                emitterRenderFailureCount += 1
                return
            }
            state.lastRenderUsedAdditiveBlending = true
            for particle in state.particles {
                draw(particle, additive: true)
            }
        default:
            emitterRenderFailureCount += 1
        }
    }

    private func emitterParticleBirthCount(
        cell: CAEmitterCell,
        activeDelta: Float,
        rateMultiplier: Float,
        remainderKey: EmitterBirthRemainderKey,
        state: EmitterLayerState
    ) -> Int? {
        let configuredBirthRate = cell.birthRate * rateMultiplier
        guard activeDelta.isFinite,
              activeDelta >= 0,
              configuredBirthRate.isFinite else {
            return nil
        }
        let birthRate = max(0, configuredBirthRate)
        let accumulated = birthRate * activeDelta
            + state.birthRemainders[remainderKey, default: 0]
        guard accumulated.isFinite,
              accumulated >= 0,
              accumulated <= Float(Int.max) else {
            return nil
        }
        let particlesToSpawn = Int(accumulated.rounded(.down))
        state.birthRemainders[remainderKey] = accumulated - Float(particlesToSpawn)
        return particlesToSpawn
    }

    private func assignBirthSequence(
        to particle: inout EmitterParticle,
        state: EmitterLayerState
    ) {
        particle.birthSequence = state.nextBirthSequence
        state.nextBirthSequence &+= 1
    }

    private func makeEmitterParticle(
        cell: CAEmitterCell,
        position: SIMD3<Float>,
        parentDirection: SIMD3<Float>?,
        inheritedColor: SIMD4<Float>,
        inheritedScale: Float,
        generation: Int,
        emitterLayer: CAEmitterLayer,
        state: EmitterLayerState
    ) -> EmitterParticle? {
        var particle = EmitterParticle()
        if let configuredContents = cell.contents {
            guard let image = configuredContents as? CGImage,
                  image.width > 0,
                  image.height > 0,
                  cell.contentsScale.isFinite,
                  cell.contentsScale > 0,
                  cell.contentsRect.origin.x.isFinite,
                  cell.contentsRect.origin.y.isFinite,
                  cell.contentsRect.width.isFinite,
                  cell.contentsRect.height.isFinite,
                  cell.contentsRect.width > 0,
                  cell.contentsRect.height > 0,
                  cell.minificationFilterBias.isFinite,
                  EmitterTextureSampling(
                    magnificationFilter: cell.magnificationFilter,
                    minificationFilter: cell.minificationFilter
                  ) != nil else {
                return nil
            }
            particle.contents = image
            particle.contentsRect = cell.contentsRect
            particle.contentsScale = Float(cell.contentsScale)
            particle.magnificationFilter = cell.magnificationFilter
            particle.minificationFilter = cell.minificationFilter
            particle.minificationFilterBias = cell.minificationFilterBias
        }

        guard let localDirection = EmitterGeometry.direction(
            longitude: cell.emissionLongitude,
            latitude: cell.emissionLatitude,
            range: cell.emissionRange,
            random: &state.randomSource
        ) else {
            return nil
        }
        let direction: SIMD3<Float>
        if let parentDirection {
            do {
                direction = try EmitterCellSimulation.childDirection(
                    localDirection: localDirection,
                    parentDirection: parentDirection
                )
            } catch {
                return nil
            }
        } else {
            direction = localDirection
        }
        let velocityVariation = CGFloat(state.randomSource.signedFloat()) * cell.velocityRange
        let velocity = Float(cell.velocity + velocityVariation) * emitterLayer.velocity
        guard velocity.isFinite else { return nil }

        particle.generation = generation
        particle.emitterCells = cell.emitterCells ?? []
        particle.position = position
        particle.previousPosition = position
        particle.velocity = direction * velocity
        particle.emissionDirection = direction
        particle.acceleration = SIMD3(
            Float(cell.xAcceleration),
            Float(cell.yAcceleration),
            Float(cell.zAcceleration)
        )

        particle.lifetime = (
            cell.lifetime + state.randomSource.signedFloat() * cell.lifetimeRange
        ) * emitterLayer.lifetime
        particle.previousLifetime = particle.lifetime
        particle.maxLifetime = particle.lifetime
        particle.scale = Float(
            cell.scale + CGFloat(state.randomSource.signedFloat()) * cell.scaleRange
        ) * emitterLayer.scale * inheritedScale
        particle.previousScale = particle.scale
        particle.scaleSpeed = Float(cell.scaleSpeed) * emitterLayer.scale * inheritedScale
        particle.rotationSpeed = Float(
            cell.spin + CGFloat(state.randomSource.signedFloat()) * cell.spinRange
        ) * emitterLayer.spin

        let baseColor = emitterCellColor(cell, random: &state.randomSource)
        particle.color = baseColor * inheritedColor
        particle.previousColor = particle.color
        particle.colorSpeed = SIMD4(
            cell.redSpeed,
            cell.greenSpeed,
            cell.blueSpeed,
            cell.alphaSpeed
        ) * inheritedColor
        guard particleStateIsFinite(particle) else { return nil }
        particle.isAlive = particle.lifetime > 0
        return particle
    }

    private func emitterCellColor(
        _ cell: CAEmitterCell,
        random: inout EmitterRandomSource
    ) -> SIMD4<Float> {
        let components = cell.color?.components ?? []
        let base: SIMD4<Float>
        switch components.count {
        case 4...:
            base = SIMD4(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
        case 3:
            base = SIMD4(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                1
            )
        case 2:
            let gray = Float(components[0])
            base = SIMD4(gray, gray, gray, Float(components[1]))
        case 1:
            let gray = Float(components[0])
            base = SIMD4(gray, gray, gray, 1)
        default:
            base = SIMD4(1, 1, 1, 1)
        }
        return SIMD4(
            base.x + random.signedFloat() * cell.redRange,
            base.y + random.signedFloat() * cell.greenRange,
            base.z + random.signedFloat() * cell.blueRange,
            base.w + random.signedFloat() * cell.alphaRange
        )
    }

    private func renderEmitterParticle(
        _ particle: EmitterParticle,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        vertexBuffer: GPUBuffer,
        uniformBuffer: GPUBuffer,
        additive: Bool
    ) -> Bool {
        guard let image = particle.contents,
              let sampling = EmitterTextureSampling(
                magnificationFilter: particle.magnificationFilter,
                minificationFilter: particle.minificationFilter
              ),
              let sampler = emitterTextureSamplers[sampling],
              let texturedBindGroupLayout,
              let selectedPipeline = selectedEmitterPipeline(additive: additive),
              let texture = textureManager?.getOrCreateTexture(
                for: image,
                width: image.width,
                height: image.height,
                memorySizeBytes: mipmappedRGBAByteCount(
                    width: image.width,
                    height: image.height
                ),
                factory: { [weak self] in
                    self?.createGPUTexture(from: image, device: device)
                }
              ) else {
            return false
        }

        let width = Float(image.width) / particle.contentsScale * particle.scale
        let height = Float(image.height) / particle.contentsScale * particle.scale
        let rotation = particle.rotation
        let center = SIMD2<Float>(0.5, 0.5)
        let p0 = rotatePoint(SIMD2(0, 0), angle: rotation) - center
        let p1 = rotatePoint(SIMD2(1, 0), angle: rotation) - center
        let p2 = rotatePoint(SIMD2(0, 1), angle: rotation) - center
        let p3 = rotatePoint(SIMD2(1, 1), angle: rotation) - center
        let contentsRect = particle.contentsRect
        let uvMinX = Float(contentsRect.minX)
        let uvMinY = Float(contentsRect.minY)
        let uvMaxX = Float(contentsRect.maxX)
        let uvMaxY = Float(contentsRect.maxY)

        let replicatedParticleColor = replicatedColor(particle.color)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: p0, texCoord: SIMD2(uvMinX, uvMaxY), color: replicatedParticleColor),
            CARendererVertex(position: p1, texCoord: SIMD2(uvMaxX, uvMaxY), color: replicatedParticleColor),
            CARendererVertex(position: p2, texCoord: SIMD2(uvMinX, uvMinY), color: replicatedParticleColor),
            CARendererVertex(position: p1, texCoord: SIMD2(uvMaxX, uvMaxY), color: replicatedParticleColor),
            CARendererVertex(position: p3, texCoord: SIMD2(uvMaxX, uvMinY), color: replicatedParticleColor),
            CARendererVertex(position: p2, texCoord: SIMD2(uvMinX, uvMinY), color: replicatedParticleColor),
        ]
        guard let (vertexOffset, layerIndex) = allocateVertices(count: vertices.count) else {
            return false
        }

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(width, 0, 0, 0),
            SIMD4<Float>(0, height, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let translateMatrix = Matrix4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(particle.position.x, particle.position.y, particle.position.z, 1)
        ))
        var uniforms = TexturedUniforms(
            mvpMatrix: modelMatrix * translateMatrix * scaleMatrix,
            opacity: currentEffectiveOpacity,
            layerSize: SIMD2<Float>(width, height),
            samplingBias: min(max(particle.minificationFilterBias, -16), 15.99)
        )
        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: createFloat32Array(from: &uniforms)
        )
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: createFloat32Array(from: &vertices)
        )
        let texturedBindGroup = cachedTexturedBindGroup(
            cacheKey: .emitterImage(ObjectIdentifier(image), sampling),
            gpuTexture: texture,
            device: device,
            layout: texturedBindGroupLayout,
            sampler: sampler,
            uniformBuffer: uniformBuffer,
            uniformStride: UInt64(MemoryLayout<TexturedUniforms>.stride)
        )
        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(
            0,
            bindGroup: texturedBindGroup,
            dynamicOffsets: [UInt32(uniformOffset)]
        )
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
        return true
    }

    private func selectedEmitterPipeline(additive: Bool) -> GPURenderPipeline? {
        if additive {
            if transformDepthNesting > 0 {
                return maskNestingDepth > 0
                    ? emitterTexturedAdditiveDepthStencilPipeline
                    : emitterTexturedAdditiveDepthPipeline
            }
            return maskNestingDepth > 0
                ? emitterTexturedAdditiveStencilPipeline
                : emitterTexturedAdditivePipeline
        }
        if transformDepthNesting > 0 {
            return maskNestingDepth > 0
                ? texturedDepthStencilPipeline
                : texturedDepthPipeline
        }
        return maskNestingDepth > 0 ? texturedStencilPipeline : texturedPipeline
    }

    private func particleStateIsFinite(_ particle: EmitterParticle) -> Bool {
        particle.position.x.isFinite
            && particle.position.y.isFinite
            && particle.position.z.isFinite
            && particle.previousPosition.x.isFinite
            && particle.previousPosition.y.isFinite
            && particle.previousPosition.z.isFinite
            && particle.velocity.x.isFinite
            && particle.velocity.y.isFinite
            && particle.velocity.z.isFinite
            && particle.emissionDirection.x.isFinite
            && particle.emissionDirection.y.isFinite
            && particle.emissionDirection.z.isFinite
            && particle.acceleration.x.isFinite
            && particle.acceleration.y.isFinite
            && particle.acceleration.z.isFinite
            && particle.color.x.isFinite
            && particle.color.y.isFinite
            && particle.color.z.isFinite
            && particle.color.w.isFinite
            && particle.previousColor.x.isFinite
            && particle.previousColor.y.isFinite
            && particle.previousColor.z.isFinite
            && particle.previousColor.w.isFinite
            && particle.colorSpeed.x.isFinite
            && particle.colorSpeed.y.isFinite
            && particle.colorSpeed.z.isFinite
            && particle.colorSpeed.w.isFinite
            && particle.scale.isFinite
            && particle.previousScale.isFinite
            && particle.scaleSpeed.isFinite
            && particle.rotation.isFinite
            && particle.rotationSpeed.isFinite
            && particle.lifetime.isFinite
            && particle.previousLifetime.isFinite
            && particle.maxLifetime.isFinite
            && particle.contentsScale.isFinite
            && particle.contentsScale > 0
            && particle.contentsRect.origin.x.isFinite
            && particle.contentsRect.origin.y.isFinite
            && particle.contentsRect.width.isFinite
            && particle.contentsRect.height.isFinite
            && particle.minificationFilterBias.isFinite
    }

    // MARK: - CATiledLayer Rendering

    /// Calculates the current LOD level based on the layer's transform.
    /// Returns the LOD level (0 = highest detail, higher = lower detail).
    private func calculateLODLevel(
        tiledLayer: CATiledLayer,
        modelMatrix: Matrix4x4
    ) -> Int {
        // The model matrix already contains the orthographic projection. Convert its
        // normalized-device-coordinate scale back to pixels before selecting an LOD.
        let renderSize = currentRenderTargetSize
        let viewportScaleX = Float(renderSize.width) * 0.5
        let viewportScaleY = Float(renderSize.height) * 0.5
        let xInPixels = SIMD2<Float>(
            modelMatrix.columns.0.x * viewportScaleX,
            modelMatrix.columns.0.y * viewportScaleY
        )
        let yInPixels = SIMD2<Float>(
            modelMatrix.columns.1.x * viewportScaleX,
            modelMatrix.columns.1.y * viewportScaleY
        )
        let scaleX = sqrt(xInPixels.x * xInPixels.x + xInPixels.y * xInPixels.y)
        let scaleY = sqrt(yInPixels.x * yInPixels.x + yInPixels.y * yInPixels.y)
        let screenScale = CGFloat(max(scaleX, scaleY))
        return tiledLayer.lodLevel(forScreenScale: screenScale)
    }

    /// Renders a CATiledLayer with tile-based rendering and LOD support.
    private func renderTiledLayer(
        _ tiledLayer: CATiledLayer,
        presentation: CATiledLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        defer {
            if let sublayers = tiledLayer.sublayers, !sublayers.isEmpty {
                let sublayerMatrix = presentation.sublayerMatrix(modelMatrix: modelMatrix)
                for sublayer in orderedSublayers(for: tiledLayer) {
                    self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
                }
            }
        }

        // Calculate current LOD level
        let lodLevel = calculateLODLevel(tiledLayer: presentation, modelMatrix: modelMatrix)

        // Adjust tile size based on LOD level
        // Higher LOD = larger tiles (covering more area with less detail)
        let lodScale = pow(2.0, CGFloat(lodLevel))
        let pixelScale = presentation.contentsScale / lodScale
        guard pixelScale.isFinite, pixelScale > 0 else { return }
        let maximumTextureDimension = max(1, Int(device.limits.maxTextureDimension2D))
        let maximumLogicalTileDimension = CGFloat(maximumTextureDimension) / pixelScale
        let adjustedTileSize = CGSize(
            width: min(presentation.tileSize.width * lodScale, maximumLogicalTileDimension),
            height: min(presentation.tileSize.height * lodScale, maximumLogicalTileDimension)
        )

        let bounds = presentation.bounds
        guard bounds.width > 0,
              bounds.height > 0,
              bounds.width.isFinite,
              bounds.height.isFinite,
              adjustedTileSize.width.isFinite,
              adjustedTileSize.height.isFinite,
              adjustedTileSize.width > 0,
              adjustedTileSize.height > 0 else {
            return
        }
        let tileCountX = ceil(bounds.width / adjustedTileSize.width)
        let tileCountY = ceil(bounds.height / adjustedTileSize.height)
        guard tileCountX.isFinite,
              tileCountY.isFinite,
              tileCountX <= CGFloat(Int.max),
              tileCountY <= CGFloat(Int.max) else {
            return
        }
        let tilesX = Int(tileCountX)
        let tilesY = Int(tileCountY)
        let tileMediaTime = CACurrentMediaTime()

        // Render tiles
        for ty in 0..<tilesY {
            for tx in 0..<tilesX {
                let tileKey = CATiledLayer.TileKey(column: tx, row: ty, lodLevel: lodLevel)
                let localTileX = CGFloat(tx) * adjustedTileSize.width
                let localTileY = CGFloat(ty) * adjustedTileSize.height
                let tileW = min(adjustedTileSize.width, bounds.width - localTileX)
                let tileH = min(adjustedTileSize.height, bounds.height - localTileY)

                let tileTranslate = Matrix4x4(columns: (
                    SIMD4<Float>(1, 0, 0, 0),
                    SIMD4<Float>(0, 1, 0, 0),
                    SIMD4<Float>(0, 0, 1, 0),
                    SIMD4<Float>(Float(localTileX), Float(localTileY), 0, 1)
                ))

                let tileScale = Matrix4x4(columns: (
                    SIMD4<Float>(Float(tileW), 0, 0, 0),
                    SIMD4<Float>(0, Float(tileH), 0, 0),
                    SIMD4<Float>(0, 0, 1, 0),
                    SIMD4<Float>(0, 0, 0, 1)
                ))

                let tileMatrix = modelMatrix * tileTranslate * tileScale

                // Check if tile has cached content
                if let cachedImage = tiledLayer.cachedImage(for: tileKey) {
                    let fadeOpacity = tiledLayer.tileOpacity(for: tileKey, at: tileMediaTime)
                    if fadeOpacity < 1 {
                        tiledLayer.markDirty(.contents)
                    }
                    var outerEdges: CAEdgeAntialiasingMask = []
                    if tx == 0 { outerEdges.insert(.layerLeftEdge) }
                    if tx == tilesX - 1 { outerEdges.insert(.layerRightEdge) }
                    if ty == 0 { outerEdges.insert(.layerBottomEdge) }
                    if ty == tilesY - 1 { outerEdges.insert(.layerTopEdge) }
                    let tileEdgeAntialiasingMask = presentation.allowsEdgeAntialiasing
                        ? Float(presentation.edgeAntialiasingMask.intersection(outerEdges).rawValue)
                        : 0
                    // Render cached tile as texture
                    renderTileWithImage(
                        cachedImage,
                        layer: presentation,
                        device: device,
                        renderPass: renderPass,
                        tileMatrix: tileMatrix,
                        tileSize: CGSize(width: tileW, height: tileH),
                        opacity: currentEffectiveOpacity * fadeOpacity,
                        edgeAntialiasingMask: tileEdgeAntialiasingMask
                    )
                } else {
                    // Request tile from delegate if not already loading
                    if !tiledLayer.loadingTiles.contains(tileKey) {
                        requestTileFromDelegate(
                            tiledLayer: tiledLayer,
                            tileKey: tileKey,
                            tileRect: CGRect(
                                x: bounds.minX + localTileX,
                                y: bounds.minY + localTileY,
                                width: tileW,
                                height: tileH
                            ),
                            scale: pixelScale,
                            maximumTextureDimension: maximumTextureDimension
                        )
                    }
                }
            }
        }

    }

    /// Renders a tile with a cached image texture.
    private func renderTileWithImage(
        _ image: CGImage,
        layer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        tileMatrix: Matrix4x4,
        tileSize: CGSize,
        opacity: Float,
        edgeAntialiasingMask: Float
    ) {
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let textureSampler = textureSampler else { return }

        // Get or create texture for image using the texture manager.
        // The manager retains `image` for the cached lifetime; passing
        // the CGImage directly (rather than `image as AnyObject`) keeps
        // the cache key on the same identity that downstream caches use.
        let imageWidth = image.width
        let imageHeight = image.height
        guard let texture = textureManager?.getOrCreateTexture(
            for: image,
            width: imageWidth,
            height: imageHeight,
            memorySizeBytes: mipmappedRGBAByteCount(width: imageWidth, height: imageHeight),
            factory: { [weak self] in
                self?.createGPUTexture(from: image, device: device)
            }
        ) else { return }

        // Create vertices with white color (texture provides color)
        // V-flip: in Y-up system, bottom vertices (y=0) get V=1, top vertices (y=1) get V=0
        let white = currentReplicatorColor
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else { return }
        let (vertexOffset, layerIndex) = allocation

        // Create uniforms
        var uniforms = TexturedUniforms(
            mvpMatrix: tileMatrix,
            opacity: opacity,
            cornerRadius: 0,
            layerSize: SIMD2<Float>(Float(tileSize.width), Float(tileSize.height)),
            edgeAntialiasingMask: edgeAntialiasingMask
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: uniformOffset, data: uniformData)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

        // Create bind group for textured rendering
        guard let texturedBindGroupLayout = texturedBindGroupLayout else { return }
        let texturedBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: texturedBindGroupLayout,
            entries: [
                GPUBindGroupEntry(
                    binding: 0,
                    resource: .bufferBinding(GPUBufferBinding(
                        buffer: uniformBuffer,
                        size: UInt64(MemoryLayout<TexturedUniforms>.stride)
                    ))
                ),
                GPUBindGroupEntry(binding: 1, resource: .sampler(textureSampler)),
                GPUBindGroupEntry(binding: 2, resource: .textureView(texture.createView()))
            ]
        ))

        if let selected = selectTexturedPipeline(for: layer) {
            renderPass.setPipeline(selected)
        }
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    /// Requests a tile from the layer's delegate.
    ///
    /// This method calls the delegate's draw(_:in:) method with a CGContext
    /// configured for the specific tile's bounds and scale.
    private func requestTileFromDelegate(
        tiledLayer: CATiledLayer,
        tileKey: CATiledLayer.TileKey,
        tileRect: CGRect,
        scale: CGFloat,
        maximumTextureDimension: Int
    ) {
        guard let delegate = tiledLayer.delegate else { return }

        let pixelWidth = min(maximumTextureDimension, max(1, Int(ceil(tileRect.width * scale))))
        let pixelHeight = min(maximumTextureDimension, max(1, Int(ceil(tileRect.height * scale))))
        tiledLayer.loadingTiles.insert(tileKey)

        pendingTileDraws.append(PendingTileDraw(
            tiledLayer: tiledLayer,
            delegate: delegate,
            tileKey: tileKey,
            tileRect: tileRect,
            scale: scale,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight
        ))
    }

    private func processPendingTileDraws() {
        let requests = pendingTileDraws
        pendingTileDraws.removeAll(keepingCapacity: true)
        for request in requests {
            Self.beginTileDraw(
                tiledLayer: request.tiledLayer,
                delegate: request.delegate,
                tileKey: request.tileKey,
                tileRect: request.tileRect,
                scale: request.scale,
                pixelWidth: request.pixelWidth,
                pixelHeight: request.pixelHeight
            )
        }
    }

    private static func beginTileDraw(
        tiledLayer: CATiledLayer,
        delegate: any CALayerDelegate,
        tileKey: CATiledLayer.TileKey,
        tileRect: CGRect,
        scale: CGFloat,
        pixelWidth: Int,
        pixelHeight: Int
    ) {
        guard let context = CGContext(
                softwareData: nil,
                width: pixelWidth,
                height: pixelHeight,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: .deviceRGB,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
              ) else {
            tiledLayer.loadingTiles.remove(tileKey)
            return
        }

        context.scaleBy(x: scale, y: scale)
        context.translateBy(x: -tileRect.minX, y: -tileRect.minY)
        delegate.draw(tiledLayer, in: context)

        Task { @MainActor in
            guard let image = await context.makeImageAsync() else {
                tiledLayer.loadingTiles.remove(tileKey)
                return
            }
            tiledLayer.cacheImage(image, for: tileKey)
            tiledLayer.markDirty(.contents)
        }
    }
}

#endif
