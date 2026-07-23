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

/// A dynamic-range request that cannot be represented by the active WebGPU output.
@_spi(RendererDiagnostics)
public enum CADynamicRangeRenderFailure: Error, Equatable, Sendable {
    case invalidToneMapMode(String)
    case invalidPreferredDynamicRange(String)
    case invalidContentsHeadroom(CGFloat)
    case canvasConfigurationFailed
    case extendedCanvasUnavailable
}

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
    let cacheGeneration: UInt64
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

private struct TransitionCompositeRequest {
    let participant: TransitionParticipantCapture
    let cacheKey: TexturedCacheKey
    let offset: CGPoint
    let opacityMultiplier: Float
}

private struct TransitionCompositeResources {
    let device: GPUDevice
    let bindGroupLayout: GPUBindGroupLayout
    let sampler: GPUSampler
    let vertexBuffer: GPUBuffer
    let uniformBuffer: GPUBuffer
    let selectedPipeline: GPURenderPipeline
    let basePipeline: GPURenderPipeline
}

private struct PreparedTransitionComposite {
    let request: TransitionCompositeRequest
    let configuration: CATransitionCompositeConfiguration
    let finalMatrix: Matrix4x4
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
    let cornerCurveExponent: Float
    let blurRadius: Float
    let detachedMaskRevisionHash: Int

    init(
        matrix: Matrix4x4,
        layerSize: SIMD2<Float>,
        cornerRadius: Float,
        cornerRadii: SIMD4<Float>,
        cornerCurveExponent: Float,
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
        self.cornerCurveExponent = cornerCurveExponent
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
/// It conforms to the internal renderer-backend contract used by the animation engine.
public final class CAWebGPURenderer: CARendererDelegate {

    // MARK: - Constants

    /// Maximum number of layers that can be rendered per frame.
    private static let maxLayers = 1024

    /// Size of aligned uniform data per layer.
    private static var alignedUniformSize: UInt64 {
        caRendererAlignedUniformSize
    }

    /// WGSL `GradientStop` stores one vec4 color and one padded location vector.
    private static let gradientStopStride: UInt64 = 32

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

    /// Placeholder required by solid draws whose shader layout also exposes gradient storage.
    private var dummyGradientStopBuffer: GPUBuffer?

    /// Dynamically sized, triple-buffered storage for the current frame's gradient stops.
    private var gradientStopBufferPool: GradientStopBufferPool?

    /// Next byte offset in the active gradient-stop buffer.
    private var gradientStopByteOffset: UInt64 = 0

    /// Storage offsets already uploaded for presentation-layer objects in this frame.
    private var gradientStopOffsets: [ObjectIdentifier: UInt32] = [:]

    /// Bind group pairing the current uniform buffer with the active gradient-stop buffer.
    private var gradientBindGroup: GPUBindGroup?

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

    /// The most recent typed transition capture or filter failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastTransitionRenderFailure: CATransitionRenderFailure?

    /// Number of requested layer filters rejected before GPU dispatch.
    @_spi(RendererDiagnostics)
    public private(set) var layerFilterFailureCount: Int = 0

    /// The most recent typed layer-filter failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastLayerFilterFailure: CALayerFilterRenderFailure?

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

    /// The most recent typed emitter simulation or spawn failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastEmitterSpawnFailure: CAEmitterFailure?

    /// Number of emitter draws rejected by configuration or renderer resources.
    @_spi(RendererDiagnostics)
    public private(set) var emitterRenderFailureCount: Int = 0

    /// The most recent typed emitter rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastEmitterRenderFailure: CAEmitterFailure?

    /// Number of replicator draws rejected by validation or renderer resources.
    @_spi(RendererDiagnostics)
    public private(set) var replicatorRenderFailureCount: Int = 0

    /// The most recent typed replicator rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastReplicatorRenderFailure: CAReplicatorRenderFailure?

    /// Monotonic within a renderer lifetime so capture scopes observe every failure,
    /// even when the public diagnostic count is deduplicated by render key.
    private var replicatorRenderFailureGeneration: UInt64 = 0
    private var reportedReplicatorRenderFailureKeys: Set<LayerRenderKey> = []

    /// Number of transform-layer depth groups rejected before rendering.
    @_spi(RendererDiagnostics)
    public private(set) var transformDepthRenderFailureCount: Int = 0

    /// The most recent typed transform-layer depth failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastTransformDepthRenderFailure: CATransformDepthRenderFailure?

    /// Number of visible shadows that could not complete the GPU render path.
    @_spi(RendererDiagnostics)
    public private(set) var shadowRenderFailureCount: Int = 0

    /// The most recent typed shadow-rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastShadowRenderFailure: CAShadowRenderFailure?

    /// Number of shape fills rejected because their path or fill rule was invalid.
    @_spi(RendererDiagnostics)
    public private(set) var shapeRenderFailureCount: Int = 0

    /// The most recent typed shape-rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastShapeRenderFailure: CAShapeRenderFailure?

    /// Number of gradient draws rejected because their configuration was invalid.
    @_spi(RendererDiagnostics)
    public private(set) var gradientRenderFailureCount: Int = 0

    /// The most recent typed gradient-rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastGradientRenderFailure: CAGradientRenderFailure?

    /// Number of layers rejected because their requested corner curve is unsupported.
    @_spi(RendererDiagnostics)
    public private(set) var cornerCurveRenderFailureCount: Int = 0

    /// The most recent typed corner-curve rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastCornerCurveRenderFailure: CACornerCurveRenderFailure?

    /// Number of stencil masks or rounded clips rejected before safe state mutation.
    @_spi(RendererDiagnostics)
    public private(set) var maskRenderFailureCount: Int = 0

    /// The most recent typed mask or stencil-state failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastMaskRenderFailure: CAMaskRenderFailure?

    /// Number of background or border solid quads rejected before GPU submission.
    @_spi(RendererDiagnostics)
    public private(set) var solidRenderFailureCount: Int = 0

    /// The most recent typed background or border failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastSolidRenderFailure: CASolidRenderFailure?

    /// Number of image-content draws rejected because their geometry, resources, or pixels are invalid.
    @_spi(RendererDiagnostics)
    public private(set) var contentsRenderFailureCount: Int = 0

    /// The most recent typed contents-rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastContentsRenderFailure: CAContentsRenderFailure?

    /// Number of text draws rejected because their input, browser rasterizer, or GPU resources failed.
    @_spi(RendererDiagnostics)
    public private(set) var textRenderFailureCount: Int = 0

    /// The most recent typed text-rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastTextRenderFailure: CATextRenderFailure?

    /// Number of tiled-layer draws rejected by validation or renderer resources.
    @_spi(RendererDiagnostics)
    public private(set) var tiledLayerRenderFailureCount: Int = 0

    /// The most recent typed tiled-layer rendering failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastTiledLayerRenderFailure: CATiledLayerRenderFailure?

    /// The most recent typed image-conversion failure in the current frame.
    @_spi(RendererDiagnostics)
    public private(set) var lastContentsConversionError: CAImageContentsConversionError?

    /// Number of frames rejected because their dynamic-range contract could not be honored.
    @_spi(RendererDiagnostics)
    public private(set) var dynamicRangeRenderFailureCount: Int = 0

    /// The most recent dynamic-range failure, retained for typed diagnostics.
    @_spi(RendererDiagnostics)
    public private(set) var lastDynamicRangeRenderFailure: CADynamicRangeRenderFailure?

    /// Number of frames rejected before command encoding could begin.
    @_spi(RendererDiagnostics)
    public private(set) var frameRenderFailureCount: Int = 0

    /// The most recent typed frame-start failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastFrameRenderFailure: CAWebGPUFrameRenderFailure?

    /// Whether the browser reports configurable canvas tone mapping.
    @_spi(RendererDiagnostics)
    public private(set) var supportsExtendedDynamicRangeOutput: Bool = false

    /// Whether the canvas is currently configured for extended-range presentation.
    @_spi(RendererDiagnostics)
    public private(set) var isExtendedDynamicRangeActive: Bool = false

    /// Number of shape-fill draw calls encoded in the latest frame.
    @_spi(RendererDiagnostics)
    public private(set) var shapeFillDrawCount: Int = 0

    /// Number of tessellated shape-fill vertices submitted in the latest frame.
    @_spi(RendererDiagnostics)
    public private(set) var shapeFillVertexCount: Int = 0

    /// Number of rasterization captures rejected because their extent or scale was invalid.
    @_spi(RendererDiagnostics)
    public private(set) var rasterizationFailureCount: Int = 0

    /// The most recent typed rasterization capture or composite failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastRasterizationRenderFailure: CARasterizationRenderFailure?

    /// Number of requested delegate backing-store draws that could not produce an image.
    @_spi(RendererDiagnostics)
    public private(set) var delegateDrawFailureCount: Int = 0

    /// The most recent typed delegate backing-store failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastDelegateBackingStoreError: CADelegateBackingStoreError?

    /// The concrete storage format used for the most recently completed delegate draw.
    @_spi(RendererDiagnostics)
    public private(set) var lastDelegateBackingStoreFormat: CALayerContentsFormat?

    /// Number of live delegate-generated backing stores retained by the renderer.
    @_spi(RendererDiagnostics)
    public var activeDelegateBackingStoreCount: Int {
        delegateBackingStores.count
    }

    /// The preferred texture format.
    private var preferredFormat: GPUTextureFormat = .rgba16float

    /// Tone-mapping mode confirmed by `GPUCanvasContext.getConfiguration()`.
    private var canvasToneMappingMode: GPUCanvasToneMappingMode = .standard

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
        guard let requiredSize = availableVertexAllocationSize(count: count) else {
            droppedLayerCount += 1
            return nil
        }

        let result = (vertexOffset: currentVertexOffset, uniformIndex: currentLayerIndex)

        // Advance pointers
        currentVertexOffset += requiredSize
        currentLayerIndex += 1

        return result
    }

    /// Returns the byte count when a vertex allocation can be committed without
    /// overflowing either the vertex buffer or the per-frame uniform table.
    private func availableVertexAllocationSize(count: Int) -> UInt64? {
        guard count >= 0 else { return nil }
        let byteCount = count.multipliedReportingOverflow(
            by: MemoryLayout<CARendererVertex>.stride
        )
        guard !byteCount.overflow,
              let requiredSize = UInt64(exactly: byteCount.partialValue),
              currentVertexOffset <= Self.maxVertexBufferSize,
              requiredSize <= Self.maxVertexBufferSize - currentVertexOffset,
              currentLayerIndex < Self.maxLayers else {
            return nil
        }
        return requiredSize
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
        // An emitter normally flattens its particles into the layer plane. A
        // depth-preserving emitter is the explicit exception: its particles
        // remain in the containing 3D space and must reach the depth pipelines.
        if let emitterLayer = presentationLayer as? CAEmitterLayer {
            return !emitterLayer.preservesDepth
        }
        return modelLayer.sublayers?.isEmpty == false
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
    ) throws(CATransformDepthRenderFailure) -> [CALayer] {
        var projectedSublayers: [(index: Int, layer: CALayer, depth: Float)] = []
        for (index, sublayer) in (layer.sublayers ?? []).enumerated() {
            let depth: Float
            do {
                depth = try projectedCenterDepth(
                    of: sublayer,
                    parentMatrix: parentMatrix
                )
            } catch {
                throw .invalidProjectedDepth(sublayerIndex: index, reason: error)
            }
            projectedSublayers.append((index, sublayer, depth))
        }
        projectedSublayers.sort { lhs, rhs in
            lhs.depth == rhs.depth ? lhs.index < rhs.index : lhs.depth < rhs.depth
        }
        return projectedSublayers.map(\.layer)
    }

    private func projectedCenterDepth(
        of layer: CALayer,
        parentMatrix: Matrix4x4
    ) throws(CAProjectedDepthError) -> Float {
        try projectedCenterDepth(
            of: layer,
            parentMatrix: parentMatrix,
            timeOffset: currentReplicatorTimeOffset
        )
    }

    private func projectedCenterDepth(
        of layer: CALayer,
        parentMatrix: Matrix4x4,
        timeOffset: CFTimeInterval
    ) throws(CAProjectedDepthError) -> Float {
        let presentation = timeOffset == 0
            ? layer._renderTimePresentation()
            : layer.presentationAtTimeOffset(timeOffset)
        let matrix = presentation.modelMatrix(parentMatrix: parentMatrix)
        let center = SIMD4<Float>(
            Float(presentation.bounds.width * 0.5),
            Float(presentation.bounds.height * 0.5),
            0,
            1
        )
        let projected = matrix * center
        return try CAProjectedDepth.resolve(z: projected.z, w: projected.w)
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

        let configuration: CAReplicatorRenderConfiguration
        do {
            configuration = try CAReplicatorRenderConfiguration(
                layer: replicatorPresentation,
                maximumInstanceCount: Self.maxLayers
            )
        } catch {
            recordReplicatorRenderFailure(error, for: replicatorModel)
            return
        }
        var cumulativeTransform = CATransform3DIdentity
        for instanceIndex in 0..<configuration.instanceCount {
            guard CAReplicatorRenderConfiguration.isFinite(cumulativeTransform) else {
                recordReplicatorRenderFailure(
                    .cumulativeTransformOverflow(instanceIndex: instanceIndex),
                    for: replicatorModel
                )
                return
            }
            let inheritedTimeOffset = currentReplicatorTimeOffset
            let inheritedColor = currentReplicatorColor
            let instanceColor: SIMD4<Float>
            let instanceTimeOffset: CFTimeInterval
            do {
                instanceColor = try configuration.color(at: instanceIndex)
                instanceTimeOffset = try configuration.timeOffset(at: instanceIndex)
            } catch {
                recordReplicatorRenderFailure(error, for: replicatorModel)
                return
            }
            let combinedTimeOffset = inheritedTimeOffset + instanceTimeOffset
            guard combinedTimeOffset.isFinite else {
                recordReplicatorRenderFailure(
                    .instanceTimeOffsetOverflow(instanceIndex: instanceIndex),
                    for: replicatorModel
                )
                return
            }
            replicatorColorStack.append(inheritedColor * instanceColor)
            replicatorTimeOffsetStack.append(combinedTimeOffset)
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
            if instanceIndex + 1 < configuration.instanceCount {
                do {
                    cumulativeTransform = try configuration.nextTransform(
                        after: cumulativeTransform,
                        nextInstanceIndex: instanceIndex + 1
                    )
                } catch {
                    recordReplicatorRenderFailure(error, for: replicatorModel)
                    return
                }
            }
        }
    }

    private func withPrepassContext<T>(
        _ target: LayerPrepassTarget,
        _ body: () throws -> T
    ) rethrows -> T {
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
        return try body()
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

    /// Shadow paths rejected during the current pre-render pass.
    private var failedShadowRenderKeys: Set<LayerRenderKey> = []

    /// Active pre-render failures already counted for diagnostics.
    private var reportedShadowRenderFailureKeys: Set<LayerRenderKey> = []

    /// Pre-rendered shadows whose final display composite is currently unavailable.
    private var failedShadowDisplayKeys: Set<LayerRenderKey> = []

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

    /// Prepared layer-filter textures whose final display composite is currently unavailable.
    private var failedLayerFilterDisplayKeys: Set<LayerRenderKey> = []

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
    private var failedCompositionDisplayKeys: Set<LayerRenderKey> = []

    /// Number of requested backdrop compositions rejected before GPU dispatch.
    @_spi(RendererDiagnostics)
    public private(set) var compositionFilterFailureCount: Int = 0

    /// The most recent typed backdrop-composition failure.
    @_spi(RendererDiagnostics)
    public private(set) var lastCompositionFilterFailure: CACompositionFilterRenderFailure?

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

    /// Transition resources removed while a command encoder may still reference them.
    /// They are destroyed at the next frame boundary, after the current submission.
    private var retiredTransitionCaptures: [TransitionCapturePair] = []
    private var retiredTransitionTextures: [GPUTexture] = []

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

    /// Per-frame capture failures that must not fall through to live subtree rendering.
    private var failedRasterizationRenderKeys: Set<LayerRenderKey> = []

    @_spi(RendererDiagnostics)
    public private(set) var transformFlatteningCaptureCount: Int = 0

    /// Number of explicit `shouldRasterize` subtree captures encoded in the latest frame.
    @_spi(RendererDiagnostics)
    public private(set) var explicitRasterizationCaptureCount: Int = 0

    /// Pixel sizes of explicit `shouldRasterize` captures encoded in the latest frame.
    @_spi(RendererDiagnostics)
    public private(set) var explicitRasterizationCapturePixelSizes: [CGSize] = []

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

        // Float16 storage is required to preserve values above SDR white until
        // browser presentation. An 8-bit normalized canvas would clamp them
        // before its tone-mapping stage.
        preferredFormat = .rgba16float

        // Configure canvas context
        let ctx = canvas.getContext!("webgpu")
        guard let ctxObject = ctx.object else {
            throw CARendererError.canvasNotConfigured
        }
        context = GPUCanvasContext(jsObject: ctxObject)

        guard configureCanvas(toneMappingMode: .standard) else {
            throw CARendererError.canvasNotConfigured
        }

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
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: [.fragment],
                    buffer: GPUBufferBindingLayout(
                        type: .readOnlyStorage,
                        minBindingSize: Self.gradientStopStride
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

        guard device.limits.maxStorageBufferBindingSize >= Self.gradientStopStride else {
            throw CARendererError.pipelineCreationFailed
        }
        let dummyGradientStopBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: Self.gradientStopStride,
            usage: [.storage]
        ))
        self.dummyGradientStopBuffer = dummyGradientStopBuffer
        gradientStopBufferPool = GradientStopBufferPool(
            device: device,
            initialCapacity: Self.gradientStopStride,
            maximumCapacity: device.limits.maxStorageBufferBindingSize
        )

        // Create triple-buffered uniform buffer pool with bind groups.
        // Solid draws bind the placeholder at binding 1; gradient draws replace
        // it with the dynamically sized stop buffer for the active frame.
        let uniformBufferSize = Self.alignedUniformSize * UInt64(Self.maxLayers)
        uniformBufferPool = UniformBufferPool(
            device: device,
            bufferSize: uniformBufferSize,
            bindGroupLayout: layout,
            bindingSize: UInt64(MemoryLayout<CARendererUniforms>.stride),
            additionalEntries: [
                GPUBindGroupEntry(
                    binding: 1,
                    resource: .bufferBinding(GPUBufferBinding(
                        buffer: dummyGradientStopBuffer,
                        size: Self.gradientStopStride
                    ))
                )
            ],
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
        let renderTarget: CARenderTargetConfiguration
        do {
            renderTarget = try CARenderTargetConfiguration(
                width: width,
                height: height,
                maximumTextureDimension: Int(device.limits.maxTextureDimension2D)
            )
        } catch {
            throw CARendererError.invalidRenderTarget(error)
        }
        size = CGSize(width: renderTarget.width, height: renderTarget.height)

        // Create depth texture
        let depthResources = createDepthResources(
            device: device,
            width: renderTarget.width,
            height: renderTarget.height
        )
        depthTexture = depthResources.texture
        depthTextureView = depthResources.view
        configureRasterizationCache(width: renderTarget.width, height: renderTarget.height)

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

    /// R3.5: picks the textured pipeline without weakening the active stencil
    /// or depth contract. An unavailable opaque variant may use the matching
    /// alpha-blended variant, but a stencil variant never falls back to an
    /// unmasked pipeline.
    private func selectTexturedPipeline(
        for layer: CALayer,
        forceBlending: Bool = false
    ) -> GPURenderPipeline? {
        let blendOff = !forceBlending && !RasterizationDecisions.blendEnabled(for: layer)
        if transformDepthNesting > 0 {
            if maskNestingDepth > 0 {
                if blendOff, let opaque = texturedDepthStencilOpaquePipeline { return opaque }
                return texturedDepthStencilPipeline
            }
            if blendOff, let opaque = texturedDepthOpaquePipeline { return opaque }
            return texturedDepthPipeline
        }
        if maskNestingDepth > 0 {
            if blendOff, let opaque = texturedStencilOpaquePipeline { return opaque }
            return texturedStencilPipeline
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
        guard let bindGroupLayout = bindGroupLayout else {
            throw CARendererError.pipelineCreationFailed
        }
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

    /// Creates a complete depth resource pair for validated target dimensions.
    private func createDepthResources(
        device: GPUDevice,
        width: Int,
        height: Int
    ) -> (texture: GPUTexture, view: GPUTextureView) {
        let texture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .depth24plusStencil8,
            usage: .renderAttachment
        ))
        return (texture, texture.createView())
    }

    /// Configures and then reads back the canvas contract. Browsers that do
    /// not implement configurable tone mapping omit the member from
    /// `getConfiguration()`; an extended request must not be treated as
    /// successful in that case.
    private func configureCanvas(toneMappingMode: GPUCanvasToneMappingMode) -> Bool {
        guard let device, let context else { return false }
        context.configure(GPUCanvasConfiguration(
            device: device,
            format: preferredFormat,
            usage: [.renderAttachment, .copySrc],
            toneMapping: GPUCanvasToneMapping(mode: toneMappingMode)
        ))

        let reportedMode = context.getConfiguration()?.toneMapping?.mode
        supportsExtendedDynamicRangeOutput = reportedMode != nil
        if toneMappingMode == .extended, reportedMode != .extended {
            return false
        }
        canvasToneMappingMode = reportedMode ?? .standard
        isExtendedDynamicRangeActive = canvasToneMappingMode == .extended
        return true
    }

    private struct DynamicRangeRequest {
        var requiresExtendedOutput = false
        var prefersExtendedOutput = false

        mutating func merge(_ other: DynamicRangeRequest) {
            requiresExtendedOutput = requiresExtendedOutput || other.requiresExtendedOutput
            prefersExtendedOutput = prefersExtendedOutput || other.prefersExtendedOutput
        }
    }

    /// Resolves the complete visible tree before acquiring a canvas texture.
    /// Explicit high-range requests fail the frame if the browser cannot
    /// represent them; automatic requests may select standard output.
    private func prepareDynamicRangeOutput(for rootLayer: CALayer) -> Bool {
        let request: DynamicRangeRequest
        do {
            request = try dynamicRangeRequest(for: rootLayer)
        } catch {
            let failure = error
            recordDynamicRangeFailure(failure)
            return false
        }

        let wantsExtended = request.requiresExtendedOutput
            || (request.prefersExtendedOutput && supportsExtendedDynamicRangeOutput)
        let requestedMode: GPUCanvasToneMappingMode = wantsExtended ? .extended : .standard
        if requestedMode != canvasToneMappingMode {
            guard configureCanvas(toneMappingMode: requestedMode) else {
                if requestedMode == .extended {
                    recordDynamicRangeFailure(.extendedCanvasUnavailable)
                } else {
                    recordDynamicRangeFailure(.canvasConfigurationFailed)
                }
                return false
            }
        }
        if request.requiresExtendedOutput && !isExtendedDynamicRangeActive {
            recordDynamicRangeFailure(.extendedCanvasUnavailable)
            return false
        }

        lastDynamicRangeRenderFailure = nil
        return true
    }

    private func dynamicRangeRequest(
        for layer: CALayer
    ) throws(CADynamicRangeRenderFailure) -> DynamicRangeRequest {
        guard !layer.isHidden, layer.opacity > 0 else {
            return DynamicRangeRequest()
        }
        guard layer.toneMapMode == .automatic
                || layer.toneMapMode == .never
                || layer.toneMapMode == .ifSupported else {
            throw CADynamicRangeRenderFailure.invalidToneMapMode(layer.toneMapMode.rawValue)
        }
        guard layer.preferredDynamicRange == .automatic
                || layer.preferredDynamicRange == .standard
                || layer.preferredDynamicRange == .constrainedHigh
                || layer.preferredDynamicRange == .high else {
            throw CADynamicRangeRenderFailure.invalidPreferredDynamicRange(
                layer.preferredDynamicRange.rawValue
            )
        }
        guard layer.contentsHeadroom.isFinite,
              layer.contentsHeadroom == 0 || layer.contentsHeadroom >= 1 else {
            throw CADynamicRangeRenderFailure.invalidContentsHeadroom(layer.contentsHeadroom)
        }

        let containsExtendedContent = layerContainsExtendedContent(layer)
        let explicitlyRequestsHighRange = layer.preferredDynamicRange == .high
            || layer.preferredDynamicRange == .constrainedHigh
        var request = DynamicRangeRequest(
            requiresExtendedOutput: explicitlyRequestsHighRange
                || (containsExtendedContent && layer.toneMapMode == .never),
            prefersExtendedOutput: containsExtendedContent
                && layer.preferredDynamicRange == .automatic
        )
        for sublayer in layer.sublayers ?? [] {
            request.merge(try dynamicRangeRequest(for: sublayer))
        }
        return request
    }

    private func layerContainsExtendedContent(_ layer: CALayer) -> Bool {
        if layer.contentsHeadroom > 1 { return true }
        if let image = layer.contents as? CGImage,
           image.contentHeadroom > 1 || image.colorSpace?.isHDR() == true {
            return true
        }
        if colorContainsExtendedComponents(layer.backgroundColor)
            || colorContainsExtendedComponents(layer.borderColor)
            || colorContainsExtendedComponents(layer.shadowColor) {
            return true
        }
        if let shape = layer as? CAShapeLayer,
           colorContainsExtendedComponents(shape.fillColor)
            || colorContainsExtendedComponents(shape.strokeColor) {
            return true
        }
        if let gradient = layer as? CAGradientLayer,
           gradient.colors?.contains(where: { value in
               colorContainsExtendedComponents(value as? CGColor)
           }) == true {
            return true
        }
        if let text = layer as? CATextLayer,
           colorContainsExtendedComponents(text.foregroundColor) {
            return true
        }
        return false
    }

    private func colorContainsExtendedComponents(_ color: CGColor?) -> Bool {
        guard let color else { return false }
        if color.colorSpace?.isHDR() == true { return true }
        guard let components = color.components else { return false }
        let colorComponentCount = max(0, components.count - 1)
        return components.prefix(colorComponentCount).contains { $0 < 0 || $0 > 1 }
    }

    private func recordDynamicRangeFailure(_ failure: CADynamicRangeRenderFailure) {
        dynamicRangeRenderFailureCount += 1
        lastDynamicRangeRenderFailure = failure
    }

    private func recordFrameRenderFailure(_ failure: CAWebGPUFrameRenderFailure) {
        frameRenderFailureCount += 1
        lastFrameRenderFailure = failure
    }

    public func resize(width: Int, height: Int) {
        let maximumTextureDimension = device.map {
            Int($0.limits.maxTextureDimension2D)
        } ?? Int.max
        let renderTarget: CARenderTargetConfiguration
        do {
            renderTarget = try CARenderTargetConfiguration(
                width: Double(width),
                height: Double(height),
                maximumTextureDimension: maximumTextureDimension
            )
        } catch {
            recordFrameRenderFailure(.invalidRenderTarget(error))
            return
        }

        // Configure first so a rejected context change cannot leave the public
        // size and canvas dimensions partially committed.
        if device != nil,
           !configureCanvas(toneMappingMode: canvasToneMappingMode) {
            recordDynamicRangeFailure(.canvasConfigurationFailed)
            recordFrameRenderFailure(.canvasConfigurationFailed)
            return
        }

        // Update canvas size
        size = CGSize(width: renderTarget.width, height: renderTarget.height)
        canvas.width = .number(Double(renderTarget.width))
        canvas.height = .number(Double(renderTarget.height))

        // Recreate depth texture
        if let device {
            let depthResources = createDepthResources(
                device: device,
                width: renderTarget.width,
                height: renderTarget.height
            )
            depthTexture?.destroy()
            depthTexture = depthResources.texture
            depthTextureView = depthResources.view
        }

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
        configureRasterizationCache(width: renderTarget.width, height: renderTarget.height)
        prerasterizedTextures.removeAll(keepingCapacity: true)
        failedRasterizationRenderKeys.removeAll(keepingCapacity: true)
        rasterizePrerenderRootLayer = nil
    }

    private func configureRasterizationCache(width: Int, height: Int) {
        let bytesPerPixel = preferredFormat == .rgba16float ? 8 : 4
        let budget = max(0, Int((
            Double(width) * Double(height) * Double(bytesPerPixel) * 2.5
        ).rounded()))
        rasterizationCache?.removeAll()
        let cache = RasterizationCache<GPUTexture>(
            maxBytes: budget,
            bytesPerPixel: bytesPerPixel
        )
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
        lastContentsConversionError = nil
        lastContentsRenderFailure = nil
        guard let device else {
            recordFrameRenderFailure(.deviceUnavailable)
            return
        }
        let renderTarget: CARenderTargetConfiguration
        do {
            renderTarget = try CARenderTargetConfiguration(
                width: Double(size.width),
                height: Double(size.height),
                maximumTextureDimension: Int(device.limits.maxTextureDimension2D)
            )
        } catch {
            recordFrameRenderFailure(.invalidRenderTarget(error))
            return
        }
        guard let context else {
            recordFrameRenderFailure(.contextUnavailable)
            return
        }
        guard let pipeline else {
            recordFrameRenderFailure(.basePipelineUnavailable)
            return
        }
        guard bindGroup != nil else {
            recordFrameRenderFailure(.baseBindGroupUnavailable)
            return
        }
        guard depthTexture != nil else {
            recordFrameRenderFailure(.depthTextureUnavailable)
            return
        }
        guard let depthTextureView else {
            recordFrameRenderFailure(.depthTextureViewUnavailable)
            return
        }
        guard let layerFilterProcessor else {
            recordFrameRenderFailure(.layerFilterProcessorUnavailable)
            return
        }
        guard let rasterizationCache else {
            recordFrameRenderFailure(.rasterizationCacheUnavailable)
            return
        }

        guard prepareDynamicRangeOutput(for: rootLayer) else { return }

        processPendingTileDraws()

        // Phase 1 (PERFORMANCE_DESIGN.md §3.6): bump the per-frame token
        // before any presentation cache lookup so this frame is distinct
        // from the previous one. The process-wide token is synchronized.
        CALayer.advanceFrameToken()

        // Reset per-frame state
        currentLayerIndex = 0
        currentVertexOffset = 0
        droppedLayerCount = 0
        shapeFillDrawCount = 0
        shapeFillVertexCount = 0
        gradientStopByteOffset = 0
        gradientStopOffsets.removeAll(keepingCapacity: true)
        gradientBindGroup = nil
        opacityStack.removeAll()
        replicatorColorStack.removeAll()
        replicatorTimeOffsetStack.removeAll()
        replicatorInstancePath.removeAll()
        reportedReplicatorRenderFailureKeys.removeAll(keepingCapacity: true)
        transitionSuppressedLayer = nil
        renderTargetSizeOverride = nil
        activeTransitionSourceIDs.removeAll(keepingCapacity: true)
        for capture in retiredTransitionCaptures {
            capture.filterExecution?.invalidate()
            capture.source.texture.destroy()
            capture.target.texture.destroy()
        }
        retiredTransitionCaptures.removeAll(keepingCapacity: true)
        for texture in retiredTransitionTextures {
            texture.destroy()
        }
        retiredTransitionTextures.removeAll(keepingCapacity: true)
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
        explicitRasterizationCapturePixelSizes.removeAll(keepingCapacity: true)
        transformFlatteningCompositeCount = 0
        isRenderingMainPass = false

        // Reset rasterization pre-rendering state (R3.2)
        prerasterizedTextures.removeAll(keepingCapacity: true)
        failedRasterizationRenderKeys.removeAll(keepingCapacity: true)
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
            right: renderTarget.viewportSize.x,
            bottom: 0,
            top: renderTarget.viewportSize.y,
            near: -1000,
            far: 1000
        )

        prerenderTransitions(
            rootLayer,
            device: device,
            pipeline: pipeline,
            encoder: encoder
        )

        // A shadow silhouette can depend on rendered mask-tree alpha. Prepare
        // those filtered/masked captures without recursively drawing shadows,
        // then run the normal shadow pass and rebuild final filter captures.
        if shadowPrepassRequiresContentMasks(rootLayer) {
            suppressShadowRendering = true
            prerenderFilteredLayers(
                rootLayer,
                device: device,
                pipeline: pipeline,
                depthTextureView: depthTextureView,
                encoder: encoder,
                projectionMatrix: projectionMatrix
            )
            suppressShadowRendering = false
        }

        // Pre-render shadows with 2-pass Gaussian blur.
        prerenderShadows(
            rootLayer,
            device: device,
            pipeline: pipeline,
            depthTextureView: depthTextureView,
            encoder: encoder,
            projectionMatrix: projectionMatrix
        )

        // Pre-render layers with blur filters
        prerenderFilteredLayers(
            rootLayer,
            device: device,
            pipeline: pipeline,
            depthTextureView: depthTextureView,
            encoder: encoder,
            projectionMatrix: projectionMatrix
        )

        prepareDeferredCompositionRasterizations(
            rootLayer,
            projectionMatrix: projectionMatrix
        )

        // Pre-render shouldRasterize subtrees (R3.2 / R3.3)
        prerenderRasterizedLayers(
            rootLayer,
            device: device,
            pipeline: pipeline,
            cache: rasterizationCache,
            encoder: encoder,
            projectionMatrix: projectionMatrix
        )

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
            device: device,
            pipeline: pipeline,
            depthTextureView: depthTextureView,
            processor: layerFilterProcessor,
            clearColor: clearColor,
            encoder: encoder,
            projectionMatrix: projectionMatrix
        )
        if !deferredCompositionRasterizationKeys.isEmpty {
            capturesOnlyDeferredCompositionRasterizations = true
            prerenderRasterizedLayers(
                rootLayer,
                device: device,
                pipeline: pipeline,
                cache: rasterizationCache,
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
        gradientStopBufferPool?.advanceFrame()
        textureManager?.advanceFrame()
        geometryCache?.advanceFrame()

        // Periodically evict stale resources (not used in last 300 frames = ~5 seconds at 60fps)
        textureManager?.evictStale(olderThan: 300)
        geometryCache?.evictStale(olderThan: 300)

        // R3.4: drop rasterization entries that have sat idle longer
        // than 6 frames (~100 ms @ 60 Hz) and any overflow above the
        // viewport-derived byte budget. Idle eviction first so the
        // budget pass operates on the trimmed live set.
        rasterizationCache.evictIdle(
            currentFrame: CALayer._currentFrameToken,
            olderThan: 6
        )
        rasterizationCache.evictToBudget()
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
        guard let displayInvalidation = layer.pendingDisplayInvalidation else { return }
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
            rejectDelegateBackingStore(identifier: identifier, error: .invalidGeometry)
            return
        }

        let pixelWidthValue = ceil(bounds.width * scale)
        let pixelHeightValue = ceil(bounds.height * scale)
        let maximumDimension = CGFloat(maximumTextureDimension)
        guard pixelWidthValue.isFinite,
              pixelHeightValue.isFinite,
              pixelWidthValue <= maximumDimension,
              pixelHeightValue <= maximumDimension else {
            let reportedWidth = pixelWidthValue.isFinite && pixelWidthValue <= CGFloat(Int.max)
                ? Int(pixelWidthValue)
                : Int.max
            let reportedHeight = pixelHeightValue.isFinite && pixelHeightValue <= CGFloat(Int.max)
                ? Int(pixelHeightValue)
                : Int.max
            rejectDelegateBackingStore(
                identifier: identifier,
                error: .dimensionsExceedTextureLimit(
                    width: reportedWidth,
                    height: reportedHeight,
                    maximum: maximumTextureDimension
                )
            )
            return
        }
        let pixelWidth = Int(pixelWidthValue)
        let pixelHeight = Int(pixelHeightValue)
        let backingStoreFormat: CADelegateBackingStoreFormat
        do {
            backingStoreFormat = try CADelegateBackingStoreFormat.resolve(
                contentsFormat: layer.contentsFormat,
                contentsHeadroom: layer.contentsHeadroom
            )
        } catch {
            rejectDelegateBackingStore(identifier: identifier, error: error)
            return
        }

        let colorSpace: CGColorSpace
        let bitmapInfo: CGBitmapInfo
        switch backingStoreFormat {
        case .rgba8Uint:
            colorSpace = .deviceRGB
            bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        case .rgba16Float:
            guard let extendedColorSpace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB) else {
                rejectDelegateBackingStore(identifier: identifier, error: .extendedColorSpaceUnavailable)
                return
            }
            colorSpace = extendedColorSpace
            bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
                .union(.floatComponents)
                .union(.byteOrder16Little)
        case .gray8Uint:
            colorSpace = .deviceGray
            bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        }
        guard let context = CGContext(
            softwareData: nil,
            width: pixelWidth,
            height: pixelHeight,
            bitsPerComponent: backingStoreFormat.bitsPerComponent,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            rejectDelegateBackingStore(identifier: identifier, error: .contextCreationFailed)
            return
        }
        if backingStoreFormat == .rgba16Float,
           layer.contentsHeadroom > 1,
           !context.setEDRTargetHeadroom(Float(layer.contentsHeadroom)) {
            rejectDelegateBackingStore(
                identifier: identifier,
                error: .extendedHeadroomRejected(layer.contentsHeadroom)
            )
            return
        }

        let invalidationRect: CGRect
        switch displayInvalidation {
        case .full:
            invalidationRect = bounds
        case .partial(let requestedRect):
            invalidationRect = requestedRect.intersection(bounds)
            if let previousImage = delegateBackingStores[identifier],
               previousImage.width == pixelWidth,
               previousImage.height == pixelHeight,
               previousImage.bitsPerComponent == backingStoreFormat.bitsPerComponent,
               previousImage.bitsPerPixel == backingStoreFormat.bitsPerPixel,
               previousImage.bytesPerRow == context.bytesPerRow,
               previousImage.colorSpace == colorSpace,
               previousImage.bitmapInfo == bitmapInfo,
               let previousData = previousImage.data,
               previousData.count >= context.bytesPerRow * pixelHeight,
               let destination = CGBitmapContextGetData(context) {
                // Partial redraw must preserve untouched pixels. The copy is
                // required because CGImage snapshots are immutable while the
                // new CGContext needs independent mutable storage.
                previousData.withUnsafeBytes { source in
                    if let sourceAddress = source.baseAddress {
                        destination.copyMemory(
                            from: sourceAddress,
                            byteCount: context.bytesPerRow * pixelHeight
                        )
                    }
                }
            }
        }

        let hasDrawableInvalidation = invalidationRect.origin.x.isFinite
            && invalidationRect.origin.y.isFinite
            && invalidationRect.width.isFinite
            && invalidationRect.height.isFinite
            && invalidationRect.width > 0
            && invalidationRect.height > 0
        if hasDrawableInvalidation {
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
            context.clip(to: invalidationRect)
            context.clear(invalidationRect)
            delegate.layerWillDraw(layer)
            layer.draw(in: context)
        }
        guard let image = context.makeImage() else {
            rejectDelegateBackingStore(identifier: identifier, error: .snapshotFailed)
            return
        }
        delegateBackingStores[identifier] = image
        lastDelegateBackingStoreError = nil
        lastDelegateBackingStoreFormat = backingStoreFormat.contentsFormat
    }

    private func rejectDelegateBackingStore(
        identifier: ObjectIdentifier,
        error: CADelegateBackingStoreError
    ) {
        delegateBackingStores.removeValue(forKey: identifier)
        delegateDrawFailureCount += 1
        lastDelegateBackingStoreError = error
        lastDelegateBackingStoreFormat = nil
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

    /// Reads one pixel without reducing extended-range components to UInt8.
    @_spi(RendererDiagnostics)
    @MainActor
    public func readbackFloatPixel(x: Int, y: Int) async throws -> [Float] {
        try await readbackFloatPixels(at: [CGPoint(x: x, y: y)])[0]
    }

    /// Reads pixels from the most recently submitted canvas texture in one copy.
    ///
    /// A browser canvas texture is presentation-scoped. Copying the complete
    /// texture once keeps multi-point diagnostics deterministic after present.
    @MainActor
    public func readbackPixels(at points: [CGPoint]) async throws -> [[UInt8]] {
        try await readbackFloatPixels(at: points).map { components in
            components.map(Self.normalizedUInt8)
        }
    }

    /// Reads raw normalized or extended-range floating-point components.
    @_spi(RendererDiagnostics)
    @MainActor
    public func readbackFloatPixels(at points: [CGPoint]) async throws -> [[Float]] {
        guard let device, let texture = lastRenderedTexture else {
            throw CARendererError.renderingFailed("No rendered texture is available for readback")
        }
        guard !points.isEmpty else { return [] }
        for point in points {
            guard point.x.isFinite,
                  point.y.isFinite,
                  point.x.rounded(.towardZero) == point.x,
                  point.y.rounded(.towardZero) == point.y,
                  point.x >= 0,
                  point.y >= 0,
                  point.x < CGFloat(texture.width),
                  point.y < CGFloat(texture.height) else {
                throw CARendererError.invalidReadbackCoordinate(
                    x: point.x,
                    y: point.y,
                    width: Int(texture.width),
                    height: Int(texture.height)
                )
            }
        }

        let bytesPerPixel: UInt32
        switch preferredFormat {
        case .rgba16float:
            bytesPerPixel = 8
        case .bgra8unorm, .rgba8unorm:
            bytesPerPixel = 4
        default:
            throw CARendererError.renderingFailed(
                "Unsupported canvas readback format: \(preferredFormat.rawValue)"
            )
        }
        let unalignedBytesPerRow = UInt32(texture.width) * bytesPerPixel
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
        defer {
            stagingBuffer.unmap()
            stagingBuffer.destroy()
        }
        let pixels = try points.map { point -> [Float] in
            let offset = Int(point.y) * Int(bytesPerRow)
                + Int(point.x) * Int(bytesPerPixel)
            if preferredFormat == .rgba16float {
                return try (0..<4).map { component in
                    let componentOffset = offset + component * 2
                    let low = UInt16(try Self.readbackByte(bytes, at: componentOffset))
                    let high = UInt16(try Self.readbackByte(bytes, at: componentOffset + 1))
                    return Float(Float16(bitPattern: low | (high << 8)))
                }
            }
            let first = Float(try Self.readbackByte(bytes, at: offset)) / 255
            let second = Float(try Self.readbackByte(bytes, at: offset + 1)) / 255
            let third = Float(try Self.readbackByte(bytes, at: offset + 2)) / 255
            let alpha = Float(try Self.readbackByte(bytes, at: offset + 3)) / 255
            return preferredFormat == .bgra8unorm
                ? [third, second, first, alpha]
                : [first, second, third, alpha]
        }
        return pixels
    }

    private static func readbackByte(
        _ bytes: JSObject,
        at index: Int
    ) throws(CARendererError) -> UInt8 {
        guard let number = bytes[index].number,
              number.isFinite,
              let integer = Int(exactly: number),
              let byte = UInt8(exactly: integer) else {
            throw .invalidReadbackByte(index: index)
        }
        return byte
    }

    private static func normalizedUInt8(_ value: Float) -> UInt8 {
        if value.isNaN { return 0 }
        if value == .infinity { return 255 }
        if value == -Float.infinity { return 0 }
        let scaled = (max(0, min(1, value)) * 255).rounded()
        return UInt8(scaled)
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
        dummyGradientStopBuffer?.destroy()
        dummyGradientStopBuffer = nil
        gradientStopBufferPool?.invalidate()
        gradientStopBufferPool = nil
        gradientStopByteOffset = 0
        gradientStopOffsets.removeAll(keepingCapacity: false)
        gradientBindGroup = nil
        depthTexture?.destroy()
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
        failedShadowRenderKeys.removeAll(keepingCapacity: false)
        reportedShadowRenderFailureKeys.removeAll(keepingCapacity: false)
        failedShadowDisplayKeys.removeAll(keepingCapacity: false)
        shadowRenderFailureCount = 0
        lastShadowRenderFailure = nil
        shapeRenderFailureCount = 0
        lastShapeRenderFailure = nil
        gradientRenderFailureCount = 0
        lastGradientRenderFailure = nil
        cornerCurveRenderFailureCount = 0
        lastCornerCurveRenderFailure = nil
        maskRenderFailureCount = 0
        lastMaskRenderFailure = nil
        solidRenderFailureCount = 0
        lastSolidRenderFailure = nil
        contentsRenderFailureCount = 0
        lastContentsRenderFailure = nil
        textRenderFailureCount = 0
        lastTextRenderFailure = nil
        tiledLayerRenderFailureCount = 0
        lastTiledLayerRenderFailure = nil
        lastContentsConversionError = nil
        dynamicRangeRenderFailureCount = 0
        lastDynamicRangeRenderFailure = nil
        frameRenderFailureCount = 0
        lastFrameRenderFailure = nil
        supportsExtendedDynamicRangeOutput = false
        isExtendedDynamicRangeActive = false
        canvasToneMappingMode = .standard
        shapeFillDrawCount = 0
        shapeFillVertexCount = 0
        rasterizationFailureCount = 0
        lastRasterizationRenderFailure = nil
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
        for texture in textTextureCache.values {
            texture.destroy()
        }
        textTextureCache.removeAll()
        textTextureAccessOrder.removeAll()
        textTextureCacheByteCount = 0

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
        failedLayerFilterDisplayKeys.removeAll(keepingCapacity: false)
        layerFilterFailureCount = 0
        lastLayerFilterFailure = nil
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
        failedCompositionDisplayKeys.removeAll(keepingCapacity: false)
        compositionFilterFailureCount = 0
        lastCompositionFilterFailure = nil

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
        for capture in retiredTransitionCaptures {
            capture.filterExecution?.invalidate()
            capture.source.texture.destroy()
            capture.target.texture.destroy()
        }
        retiredTransitionCaptures.removeAll(keepingCapacity: false)
        for texture in retiredTransitionTextures {
            texture.destroy()
        }
        retiredTransitionTextures.removeAll(keepingCapacity: false)
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
        lastTransitionRenderFailure = nil
        transitionFilterProcessor?.invalidate()
        transitionFilterProcessor = nil
        prerasterizedTextures.removeAll(keepingCapacity: false)
        failedRasterizationRenderKeys.removeAll(keepingCapacity: false)
        rasterizePrerenderRootLayer = nil
        shadowCaptureRootLayer = nil
        suppressShadowRendering = false
        contentMaskCaptureSuppressedRootLayer = nil
        for request in pendingTileDraws {
            if request.tiledLayer.loadingTileGenerations[request.tileKey]
                == request.cacheGeneration {
                request.tiledLayer.loadingTiles.remove(request.tileKey)
                request.tiledLayer.loadingTileGenerations.removeValue(forKey: request.tileKey)
            }
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
        lastEmitterSpawnFailure = nil
        emitterRenderFailureCount = 0
        lastEmitterRenderFailure = nil
        replicatorRenderFailureCount = 0
        lastReplicatorRenderFailure = nil
        replicatorRenderFailureGeneration = 0
        reportedReplicatorRenderFailureKeys.removeAll(keepingCapacity: false)
        transformDepthRenderFailureCount = 0
        lastTransformDepthRenderFailure = nil

        // Textured bind group / view caches (release JS-side handles)
        perFrameTexturedBindGroupCache.removeAll(keepingCapacity: false)
        texturedTextureViewCache.removeAll(keepingCapacity: false)

        // Persistent JS Float32Array staging pool (release JS handles)
        float32StagingPool.removeAll(keepingCapacity: false)

        context?.unconfigure()
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

    /// Writes scalar storage-buffer data into a pooled JavaScript typed array.
    private func createFloat32Array(from values: inout [Float]) -> JSObject {
        let array = stagingFloat32Array(floatCount: values.count)
        for index in values.indices {
            array[index] = .number(Double(values[index]))
        }
        return array
    }

    private func renderLayer(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        guard let device else {
            recordFrameRenderFailure(.deviceUnavailable)
            return
        }
        guard let bindGroup else {
            recordFrameRenderFailure(.baseBindGroupUnavailable)
            return
        }

        // Get the presentation layer for animated values, fall back to model layer
        // This is critical for animations to be visible - the presentation layer
        // reflects the current animated state of all properties
        let presentationLayer = renderPresentation(for: layer)

        // Skip hidden layers (using presentation layer values)
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }
        if presentationLayer.cornerRadius > 0,
           let error = presentationLayer.cornerCurveRenderError {
            recordCornerCurveRenderFailure(.layer(error))
            return
        }

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

        // With preservesDepth enabled, CAReplicatorLayer has the same rendering
        // restrictions as CATransformLayer: it contributes transforms and
        // replicated descendants, but does not create a flattened drawable plane.
        if let replicatorPresentation = presentationLayer as? CAReplicatorLayer,
           replicatorPresentation.preservesDepth,
           let replicatorModel = layer as? CAReplicatorLayer {
            let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)
            renderReplicatorSublayers(
                replicatorModelLayer: replicatorModel,
                replicatorLayer: replicatorPresentation,
                sublayers: layer.sublayers ?? [],
                renderPass: renderPass,
                parentMatrix: sublayerMatrix
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

        if rasterizePrerenderRootLayer !== layer,
           failedRasterizationRenderKeys.contains(currentRenderKey) {
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
        var didApplyContentMask = false
        if hasMask, let maskLayer = presentationLayer.mask {
            // Render mask to stencil buffer
            didApplyContentMask = renderMaskToStencil(
                maskLayer,
                renderPass: renderPass,
                parentMatrix: modelMatrix
            )
        }
        defer {
            if didApplyContentMask {
                clearStencilMask(renderPass: renderPass)
            }
        }
        if hasMask && !didApplyContentMask {
            // A requested mask is part of the layer's rendering contract.
            // Never fall through to an unmasked draw when stencil generation fails.
            return
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
                do {
                    try renderPreparedComposition(
                        composition,
                        presentationLayer: presentationLayer,
                        device: device,
                        renderPass: renderPass,
                        modelMatrix: modelMatrix
                    )
                    failedCompositionDisplayKeys.remove(currentRenderKey)
                } catch let failure {
                    lastCompositionFilterFailure = failure
                    if failedCompositionDisplayKeys.insert(currentRenderKey).inserted {
                        compositionFilterFailureCount += 1
                    }
                }
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
                do {
                    try renderFilteredLayerComposite(
                        prerendered,
                        device: device,
                        renderPass: renderPass
                    )
                    failedLayerFilterDisplayKeys.remove(currentRenderKey)
                } catch let failure {
                    lastLayerFilterFailure = failure
                    if failedLayerFilterDisplayKeys.insert(currentRenderKey).inserted {
                        layerFilterFailureCount += 1
                    }
                }
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
            // A non-preserving emitter below a depth container must have been
            // captured as one plane. Never silently render its particles as
            // independent 3D geometry if that capture was unavailable.
            if transformDepthNesting > 0,
               !emitterPresentation.preservesDepth,
               rasterizePrerenderRootLayer !== layer {
                recordEmitterRenderFailure(.flatteningCaptureUnavailable)
                return
            }
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
                               modelMatrix: modelMatrix)
            }
        } else if let shapeLayer = presentationLayer as? CAShapeLayer {
            if shapeLayer.path != nil {
                renderShapeLayer(shapeLayer, device: device, renderPass: renderPass,
                                modelMatrix: modelMatrix, bindGroup: bindGroup)
            }
        } else if let gradientLayer = presentationLayer as? CAGradientLayer,
                  let colors = gradientLayer.colors, !colors.isEmpty {
            renderGradientLayer(gradientLayer, device: device, renderPass: renderPass,
                              modelMatrix: modelMatrix)
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
            var didApplyRoundedClip = false
            if shouldClip {
                // Scissor rect for axis-aligned rectangular clipping (always applied as optimization)
                let layerClipRect = calculateClipRect(layer: presentationLayer, modelMatrix: modelMatrix)
                let newClipRect = currentClipRect.intersection(with: layerClipRect)
                clipRectStack.append(newClipRect)
                applyScissorRect(renderPass)

                // Rounded corner clipping via stencil buffer (only when cornerRadius > 0)
                if useStencilClip {
                    didApplyRoundedClip = renderRoundedRectToStencil(
                        presentationLayer,
                        renderPass: renderPass,
                        modelMatrix: modelMatrix,
                        device: device
                    )
                    if !didApplyRoundedClip {
                        _ = clipRectStack.popLast()
                        applyScissorRect(renderPass)
                        return
                    }
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
                if didApplyRoundedClip {
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

    private enum TransitionFailureCategory {
        case filter
        case render
    }

    private func recordTransitionFailure(
        _ failure: CATransitionRenderFailure,
        category: TransitionFailureCategory
    ) {
        switch category {
        case .filter:
            transitionFilterFailureCount += 1
        case .render:
            transitionRenderFailureCount += 1
        }
        lastTransitionRenderFailure = failure
    }

    private func builtInTransitionValidationFailure(
        type: CATransitionType,
        subtype: CATransitionSubtype?
    ) -> CATransitionRenderFailure? {
        switch type {
        case .fade:
            return nil
        case .moveIn, .push, .reveal:
            switch subtype {
            case .fromRight, .fromLeft, .fromTop, .fromBottom, nil:
                return nil
            default:
                return .unsupportedTransitionSubtype(subtype?.rawValue ?? "nil")
            }
        default:
            return .unsupportedTransitionType(type.rawValue)
        }
    }

    private func prerenderTransitions(
        _ rootLayer: CALayer,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        encoder: GPUCommandEncoder
    ) {
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
                    let capture: TransitionCapturePair
                    if let filterValue = state.filter {
                        guard let filter = filterValue as? CIFilter else {
                            failedTransitionSourceIDs.insert(sourceID)
                            recordTransitionFailure(
                                .unsupportedFilterValue(String(reflecting: type(of: filterValue))),
                                category: .filter
                            )
                            return
                        }
                        guard let processor = transitionFilterProcessor else {
                            failedTransitionSourceIDs.insert(sourceID)
                            recordTransitionFailure(.filterProcessorUnavailable, category: .filter)
                            return
                        }
                        guard processor.supports(filter) else {
                            failedTransitionSourceIDs.insert(sourceID)
                            recordTransitionFailure(
                                .unsupportedFilter(String(describing: filter)),
                                category: .filter
                            )
                            return
                        }
                        do {
                            capture = try createFilteredTransitionCapture(
                                sourceLayer: state.sourceLayer,
                                targetLayer: layer,
                                filter: filter,
                                processor: processor,
                                device: device,
                                pipeline: pipeline,
                                encoder: encoder
                            )
                        } catch let failure {
                            failedTransitionSourceIDs.insert(sourceID)
                            recordTransitionFailure(failure, category: .filter)
                            return
                        }
                    } else {
                        if let failure = builtInTransitionValidationFailure(
                            type: state.type,
                            subtype: state.subtype
                        ) {
                            failedTransitionSourceIDs.insert(sourceID)
                            recordTransitionFailure(failure, category: .render)
                            return
                        }
                        do {
                            capture = try createBuiltInTransitionCapture(
                                sourceLayer: state.sourceLayer,
                                targetLayer: layer,
                                device: device,
                                pipeline: pipeline,
                                encoder: encoder
                            )
                        } catch let failure {
                            failedTransitionSourceIDs.insert(sourceID)
                            recordTransitionFailure(failure, category: .render)
                            return
                        }
                    }
                    transitionCaptures[sourceID] = capture
                    transitionSourceCaptureCount += 1
                    transitionTargetCaptureCount += 1
                }

                if let execution = transitionCaptures[sourceID]?.filterExecution {
                    guard state.progress.isFinite else {
                        retireTransitionCapture(for: sourceID)
                        failedTransitionSourceIDs.insert(sourceID)
                        recordTransitionFailure(.invalidProgress(state.progress), category: .filter)
                        return
                    }
                    do {
                        try execution.encode(
                            progress: Float(state.progress),
                            commandEncoder: encoder
                        )
                        transitionFilterDispatchCount += 1
                    } catch {
                        retireTransitionCapture(for: sourceID)
                        failedTransitionSourceIDs.insert(sourceID)
                        recordTransitionFailure(
                            .filterDispatchFailed(String(describing: error)),
                            category: .filter
                        )
                    }
                }
            }
        }

        collect(rootLayer)

        let staleSourceIDs = transitionCaptures.keys.filter {
            !activeTransitionSourceIDs.contains($0)
        }
        for sourceID in staleSourceIDs {
            retireTransitionCapture(for: sourceID)
        }
        failedTransitionSourceIDs = failedTransitionSourceIDs.intersection(activeTransitionSourceIDs)
    }

    private func createBuiltInTransitionCapture(
        sourceLayer: CALayer,
        targetLayer: CALayer,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        encoder: GPUCommandEncoder
    ) throws(CATransitionRenderFailure) -> TransitionCapturePair {
        let source = try captureTransitionParticipant(
            sourceLayer,
            role: .source,
            device: device,
            pipeline: pipeline,
            encoder: encoder
        )
        let target: TransitionParticipantCapture
        do {
            target = try captureTransitionParticipant(
                targetLayer,
                role: .target,
                device: device,
                pipeline: pipeline,
                encoder: encoder
            )
        } catch let failure {
            retiredTransitionTextures.append(source.texture)
            throw failure
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
    ) throws(CATransitionRenderFailure) -> TransitionCapturePair {
        let target = try captureTransitionParticipant(
            targetLayer,
            role: .target,
            device: device,
            pipeline: pipeline,
            encoder: encoder
        )
        let sharedPixelSize = CGSize(width: target.pixelWidth, height: target.pixelHeight)
        let source: TransitionParticipantCapture
        do {
            source = try captureTransitionParticipant(
                sourceLayer,
                role: .source,
                pixelSizeOverride: sharedPixelSize,
                device: device,
                pipeline: pipeline,
                encoder: encoder
            )
        } catch let failure {
            retiredTransitionTextures.append(target.texture)
            throw failure
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
            retiredTransitionTextures.append(source.texture)
            retiredTransitionTextures.append(target.texture)
            throw .filterExecutionCreationFailed(String(describing: error))
        }
    }

    private func retireTransitionCapture(for sourceID: ObjectIdentifier) {
        if let capture = transitionCaptures.removeValue(forKey: sourceID) {
            retiredTransitionCaptures.append(capture)
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
        role: CATransitionParticipantRole,
        pixelSizeOverride: CGSize? = nil,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        encoder: GPUCommandEncoder
    ) throws(CATransitionRenderFailure) -> TransitionParticipantCapture {
        let presentation = layer._renderTimePresentation()
        let bounds = presentation.bounds
        let configuration = try CATransitionCaptureConfiguration(
            bounds: bounds,
            contentsScale: presentation.contentsScale,
            pixelSizeOverride: pixelSizeOverride,
            maximumTextureDimension: Int(device.limits.maxTextureDimension2D),
            role: role
        )
        let pixelWidth = configuration.pixelWidth
        let pixelHeight = configuration.pixelHeight
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
            left: configuration.projectionLeft,
            right: configuration.projectionRight,
            bottom: configuration.projectionBottom,
            top: configuration.projectionTop,
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

        let replicatorFailureGenerationBeforeCapture = replicatorRenderFailureGeneration
        renderLayer(layer, renderPass: capturePass, parentMatrix: captureProjection)

        rasterizePrerenderRootLayer = previousCaptureRoot
        transitionSuppressedLayer = previousSuppression
        renderTargetSizeOverride = previousRenderTargetSize
        clipRectStack = previousClipStack
        opacityStack = previousOpacityStack
        maskNestingDepth = previousMaskNestingDepth
        currentStencilValue = previousStencilValue
        capturePass.end()

        if replicatorRenderFailureGeneration != replicatorFailureGenerationBeforeCapture,
           let failure = lastReplicatorRenderFailure {
            retiredTransitionTextures.append(texture)
            throw .participantReplicatorFailed(role, failure)
        }

        let compositeLayer = type(of: presentation).init(layer: presentation)
        compositeLayer.recursivelyClearDirtyAfterCommit()
        return TransitionParticipantCapture(
            texture: texture,
            compositeLayer: compositeLayer,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight
        )
    }

    private func transitionCompositeResources() throws(CATransitionRenderFailure) -> TransitionCompositeResources {
        guard let device,
              let texturedBindGroupLayout,
              let textureSampler,
              let vertexBuffer,
              let uniformBuffer,
              let basePipeline = pipeline else {
            throw .compositeResourcesUnavailable
        }
        guard let selectedPipeline = selectPremultipliedTexturedPipeline() else {
            throw .compositePipelineUnavailable
        }
        return TransitionCompositeResources(
            device: device,
            bindGroupLayout: texturedBindGroupLayout,
            sampler: textureSampler,
            vertexBuffer: vertexBuffer,
            uniformBuffer: uniformBuffer,
            selectedPipeline: selectedPipeline,
            basePipeline: basePipeline
        )
    }

    private func transitionCompositeConfiguration(
        for request: TransitionCompositeRequest
    ) throws(CATransitionRenderFailure) -> CATransitionCompositeConfiguration {
        try CATransitionCompositeConfiguration(
            bounds: request.participant.compositeLayer.bounds,
            position: request.participant.compositeLayer.position,
            offset: request.offset,
            opacity: currentEffectiveOpacity
                * request.participant.compositeLayer.opacity
                * request.opacityMultiplier
        )
    }

    private func matrixIsFinite(_ matrix: Matrix4x4) -> Bool {
        func vectorIsFinite(_ vector: SIMD4<Float>) -> Bool {
            vector.x.isFinite
                && vector.y.isFinite
                && vector.z.isFinite
                && vector.w.isFinite
        }
        return vectorIsFinite(matrix.columns.0)
            && vectorIsFinite(matrix.columns.1)
            && vectorIsFinite(matrix.columns.2)
            && vectorIsFinite(matrix.columns.3)
    }

    private func prepareTransitionComposite(
        _ request: TransitionCompositeRequest,
        parentMatrix: Matrix4x4
    ) throws(CATransitionRenderFailure) -> PreparedTransitionComposite {
        let configuration = try transitionCompositeConfiguration(for: request)
        let presentation = request.participant.compositeLayer
        let originalPosition = presentation.position
        presentation.position = configuration.translatedPosition
        let modelMatrix = presentation.modelMatrix(parentMatrix: parentMatrix)
        presentation.position = originalPosition
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(configuration.size.x, 0, 0, 0),
            SIMD4<Float>(0, configuration.size.y, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix
        guard matrixIsFinite(finalMatrix) else {
            throw .invalidCompositeTransform
        }
        return PreparedTransitionComposite(
            request: request,
            configuration: configuration,
            finalMatrix: finalMatrix
        )
    }

    private func reserveTransitionCompositeAllocations(
        drawCount: Int
    ) throws(CATransitionRenderFailure) -> [(vertexOffset: UInt64, uniformIndex: Int)] {
        let verticesPerDraw = 6
        let bytesPerDraw = UInt64(verticesPerDraw * MemoryLayout<CARendererVertex>.stride)
        let requiredVertexBytes = UInt64(drawCount) * bytesPerDraw
        guard drawCount > 0,
              requiredVertexBytes <= Self.maxVertexBufferSize,
              currentVertexOffset <= Self.maxVertexBufferSize - requiredVertexBytes,
              drawCount <= Self.maxLayers,
              currentLayerIndex <= Self.maxLayers - drawCount else {
            droppedLayerCount += 1
            throw .compositeVertexCapacityExceeded(drawCount)
        }

        let allocations = (0..<drawCount).map { index in
            (
                vertexOffset: currentVertexOffset + UInt64(index) * bytesPerDraw,
                uniformIndex: currentLayerIndex + index
            )
        }
        currentVertexOffset += requiredVertexBytes
        currentLayerIndex += drawCount
        return allocations
    }

    /// Composites frozen source and target layer trees for a built-in or filtered transition.
    private func renderTransition(
        state: CATransitionRenderState,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let sourceID = ObjectIdentifier(state.sourceLayer)
        guard let capture = transitionCaptures[sourceID] else { return }
        let failureCategory: TransitionFailureCategory = capture.filterExecution == nil
            ? .render
            : .filter
        guard state.progress.isFinite else {
            recordTransitionFailure(.invalidProgress(state.progress), category: failureCategory)
            return
        }
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

        if let filterExecution = capture.filterExecution {
            do {
                try renderTransitionCompositeRequests(
                    [TransitionCompositeRequest(
                        participant: TransitionParticipantCapture(
                            texture: filterExecution.outputTexture,
                            compositeLayer: capture.target.compositeLayer,
                            pixelWidth: capture.target.pixelWidth,
                            pixelHeight: capture.target.pixelHeight
                        ),
                        cacheKey: .transitionFilter(sourceID),
                        offset: .zero,
                        opacityMultiplier: 1
                    )],
                    renderPass: renderPass,
                    parentMatrix: parentMatrix
                )
            } catch let failure {
                recordTransitionFailure(failure, category: .filter)
            }
            return
        }

        do {
            switch state.type {
            case .fade:
                try renderFadeTransition(
                    capture,
                    sourceID: sourceID,
                    progress: Float(progress),
                    renderPass: renderPass,
                    parentMatrix: parentMatrix
                )

            case .moveIn, .push, .reveal:
                guard let direction = transitionDirection(
                    subtype: state.subtype,
                    bounds: targetLayer.bounds
                ) else {
                    recordTransitionFailure(
                        .unsupportedTransitionSubtype(state.subtype?.rawValue ?? "nil"),
                        category: .render
                    )
                    return
                }
                let sourceOffset: CGPoint
                let targetOffset: CGPoint
                switch state.type {
                case .moveIn:
                    sourceOffset = .zero
                    targetOffset = CGPoint(
                        x: direction.x * (1 - progress),
                        y: direction.y * (1 - progress)
                    )
                case .push:
                    sourceOffset = CGPoint(x: -direction.x * progress, y: -direction.y * progress)
                    targetOffset = CGPoint(
                        x: direction.x * (1 - progress),
                        y: direction.y * (1 - progress)
                    )
                case .reveal:
                    sourceOffset = CGPoint(x: -direction.x * progress, y: -direction.y * progress)
                    targetOffset = .zero
                default:
                    sourceOffset = .zero
                    targetOffset = .zero
                }
                let requests: [TransitionCompositeRequest]
                if state.type == .reveal {
                    requests = [
                        TransitionCompositeRequest(participant: capture.target, cacheKey: .transitionTarget(sourceID), offset: targetOffset, opacityMultiplier: 1),
                        TransitionCompositeRequest(participant: capture.source, cacheKey: .transitionSource(sourceID), offset: sourceOffset, opacityMultiplier: 1),
                    ]
                } else {
                    requests = [
                        TransitionCompositeRequest(participant: capture.source, cacheKey: .transitionSource(sourceID), offset: sourceOffset, opacityMultiplier: 1),
                        TransitionCompositeRequest(participant: capture.target, cacheKey: .transitionTarget(sourceID), offset: targetOffset, opacityMultiplier: 1),
                    ]
                }
                try renderTransitionCompositeRequests(
                    requests,
                    renderPass: renderPass,
                    parentMatrix: parentMatrix
                )

            default:
                recordTransitionFailure(
                    .unsupportedTransitionType(state.type.rawValue),
                    category: .render
                )
                return
            }
        } catch let failure {
            recordTransitionFailure(failure, category: .render)
        }
    }

    private func renderFadeTransition(
        _ capture: TransitionCapturePair,
        sourceID: ObjectIdentifier,
        progress: Float,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) throws(CATransitionRenderFailure) {
        guard let device,
              let transitionFadeBindGroupLayout,
              let textureSampler,
              let vertexBuffer,
              let uniformBuffer,
              let basePipeline = pipeline else {
            throw .compositeResourcesUnavailable
        }
        guard let selectedPipeline = selectedTransitionFadePipeline() else {
            throw .compositePipelineUnavailable
        }

        let prepared = try prepareTransitionComposite(TransitionCompositeRequest(
            participant: capture.target,
            cacheKey: .transitionTarget(sourceID),
            offset: .zero,
            opacityMultiplier: 1
        ), parentMatrix: parentMatrix)

        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 0), color: SIMD4(repeating: 1)),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: SIMD4(repeating: 1)),
        ]
        let allocation = try reserveTransitionCompositeAllocations(drawCount: 1)[0]
        let vertexOffset = allocation.vertexOffset
        let uniformIndex = allocation.uniformIndex

        var uniforms = TransitionFadeUniforms(
            mvpMatrix: prepared.finalMatrix,
            colorMultiplier: currentReplicatorColor,
            opacity: prepared.configuration.opacity,
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
        renderPass.setPipeline(basePipeline)
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

    private func renderTransitionCompositeRequests(
        _ requests: [TransitionCompositeRequest],
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) throws(CATransitionRenderFailure) {
        let resources = try transitionCompositeResources()
        var preparedComposites: [PreparedTransitionComposite] = []
        preparedComposites.reserveCapacity(requests.count)
        for request in requests {
            preparedComposites.append(
                try prepareTransitionComposite(request, parentMatrix: parentMatrix)
            )
        }
        let allocations = try reserveTransitionCompositeAllocations(drawCount: requests.count)
        for index in requests.indices {
            renderPreparedTransitionComposite(
                preparedComposites[index],
                allocation: allocations[index],
                resources: resources,
                renderPass: renderPass
            )
        }
        renderPass.setPipeline(resources.basePipeline)
    }

    private func renderPreparedTransitionComposite(
        _ prepared: PreparedTransitionComposite,
        allocation: (vertexOffset: UInt64, uniformIndex: Int),
        resources: TransitionCompositeResources,
        renderPass: GPURenderPassEncoder
    ) {
        let request = prepared.request

        let white = currentReplicatorColor
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: white),
        ]
        var uniforms = TexturedUniforms(
            mvpMatrix: prepared.finalMatrix,
            opacity: prepared.configuration.opacity,
            cornerRadius: 0,
            layerSize: prepared.configuration.size,
            cornerRadii: .zero
        )
        let uniformOffset = UInt64(allocation.uniformIndex) * Self.alignedUniformSize
        resources.device.queue.writeBuffer(
            resources.uniformBuffer,
            bufferOffset: uniformOffset,
            data: createFloat32Array(from: &uniforms)
        )
        resources.device.queue.writeBuffer(
            resources.vertexBuffer,
            bufferOffset: allocation.vertexOffset,
            data: createFloat32Array(from: &vertices)
        )

        let bindGroup = cachedTexturedBindGroup(
            cacheKey: request.cacheKey,
            gpuTexture: request.participant.texture,
            device: resources.device,
            layout: resources.bindGroupLayout,
            sampler: resources.sampler,
            uniformBuffer: resources.uniformBuffer,
            uniformStride: UInt64(MemoryLayout<TexturedUniforms>.stride)
        )
        renderPass.setPipeline(resources.selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: resources.vertexBuffer, offset: allocation.vertexOffset)
        renderPass.draw(vertexCount: 6)
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

    private func solidRenderConfiguration(
        for layer: CALayer,
        color: SIMD4<Float>,
        borderWidth: CGFloat,
        context: CASolidRenderContext
    ) throws(CASolidRenderFailure) -> CASolidRenderConfiguration {
        try CASolidRenderConfiguration(
            bounds: layer.bounds,
            color: color,
            opacity: currentEffectiveOpacity,
            cornerRadius: layer.cornerRadius,
            cornerCurveExponent: layer.cornerCurveRenderExponent
                ?? Float(CornerCurveRenderConfiguration.circularExponent),
            cornerRadii: layer.cornerRadiiComponents,
            borderWidth: borderWidth,
            context: context
        )
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
              let uniformBuffer = uniformBuffer else {
            recordSolidRenderFailure(.resourcesUnavailable(.background))
            return
        }

        let color = replicatedColor(presentationLayer.backgroundColorComponents)
        let configuration: CASolidRenderConfiguration
        do {
            configuration = try solidRenderConfiguration(
                for: presentationLayer,
                color: color,
                borderWidth: 0,
                context: .background
            )
        } catch let failure {
            recordSolidRenderFailure(failure)
            return
        }
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(configuration.size.x, 0, 0, 0),
            SIMD4<Float>(0, configuration.size.y, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix
        guard matrixIsFinite(finalMatrix) else {
            recordSolidRenderFailure(.invalidTransform(.background))
            return
        }
        guard let selectedPipeline = stencilAwarePipeline() else {
            recordSolidRenderFailure(.pipelineUnavailable(.background))
            return
        }
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: configuration.color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: configuration.color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: configuration.color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: configuration.color),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: configuration.color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: configuration.color),
        ]

        guard let (vertexOffset, uniformIndex) = allocateVertices(count: vertices.count) else {
            recordSolidRenderFailure(.vertexCapacityExceeded(.background))
            return
        }

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: configuration.opacity,
            cornerRadius: configuration.cornerRadius,
            layerSize: configuration.size,
            edgeAntialiasingMask: presentationLayer.edgeAntialiasingMaskValue,
            cornerCurveExponent: configuration.cornerCurveExponent,
            cornerRadii: configuration.cornerRadii
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
        renderPass.setPipeline(selectedPipeline)
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
              let uniformBuffer = uniformBuffer else {
            recordSolidRenderFailure(.resourcesUnavailable(.border))
            return
        }

        let color = replicatedColor(presentationLayer.borderColorComponents)
        let configuration: CASolidRenderConfiguration
        do {
            configuration = try solidRenderConfiguration(
                for: presentationLayer,
                color: color,
                borderWidth: presentationLayer.borderWidth,
                context: .border
            )
        } catch let failure {
            recordSolidRenderFailure(failure)
            return
        }
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(configuration.size.x, 0, 0, 0),
            SIMD4<Float>(0, configuration.size.y, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix
        guard matrixIsFinite(finalMatrix) else {
            recordSolidRenderFailure(.invalidTransform(.border))
            return
        }
        guard let selectedPipeline = stencilAwarePipeline() else {
            recordSolidRenderFailure(.pipelineUnavailable(.border))
            return
        }
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: configuration.color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: configuration.color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: configuration.color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: configuration.color),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: configuration.color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: configuration.color),
        ]

        guard let (vertexOffset, uniformIndex) = allocateVertices(count: vertices.count) else {
            recordSolidRenderFailure(.vertexCapacityExceeded(.border))
            return
        }

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: configuration.opacity,
            cornerRadius: configuration.cornerRadius,
            layerSize: configuration.size,
            borderWidth: configuration.borderWidth,
            renderMode: 1.0,  // Border mode
            edgeAntialiasingMask: presentationLayer.edgeAntialiasingMaskValue,
            cornerCurveExponent: configuration.cornerCurveExponent,
            cornerRadii: configuration.cornerRadii
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
        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    /// Returns the stencil-testing variant when a mask/clip stencil is active,
    /// otherwise the default pipeline with stencil compare == .always.
    /// Used by background and border rendering so solid-colored quads respect CALayer.mask
    /// and rounded-corner masksToBounds clipping.
    private func stencilAwarePipeline() -> GPURenderPipeline? {
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
        return pipeline
    }

    // MARK: - Mask Rendering (Stencil)

    private func recordCornerCurveRenderFailure(
        _ failure: CACornerCurveRenderFailure
    ) {
        cornerCurveRenderFailureCount += 1
        lastCornerCurveRenderFailure = failure
    }

    private func recordMaskRenderFailure(_ failure: CAMaskRenderFailure) {
        maskRenderFailureCount += 1
        lastMaskRenderFailure = failure
    }

    private func recordSolidRenderFailure(_ failure: CASolidRenderFailure) {
        solidRenderFailureCount += 1
        lastSolidRenderFailure = failure
    }

    private var stencilStateIsValid: Bool {
        guard let depthReference = UInt32(exactly: maskNestingDepth) else {
            return false
        }
        return depthReference == currentStencilValue
    }

    /// Renders a mask layer to the stencil buffer.
    ///
    /// This writes to the stencil buffer where the mask layer has visible content.
    /// Subsequent layer rendering will only appear where the stencil value matches.
    private func renderMaskToStencil(
        _ maskLayer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) -> Bool {
        guard let device = device,
              let stencilWritePipeline = stencilWritePipeline,
              let stencilTestPipeline = stencilTestPipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup,
              pipeline != nil else {
            recordMaskRenderFailure(.resourcesUnavailable(.contentMask))
            return false
        }

        // Validate before mutating the stencil state. A failed mask must not clear
        // or decrement an enclosing mask when the caller unwinds.
        let maskPresentationLayer = renderPresentation(for: maskLayer)
        if maskPresentationLayer.cornerRadius > 0,
           let error = maskPresentationLayer.cornerCurveRenderError {
            recordCornerCurveRenderFailure(.mask(error))
            recordMaskRenderFailure(
                .unsupportedCornerCurve(.contentMask, maskPresentationLayer.cornerCurve.rawValue)
            )
            return false
        }

        let configuration: CAMaskRenderConfiguration
        do {
            configuration = try CAMaskRenderConfiguration(
                bounds: maskPresentationLayer.bounds,
                cornerRadius: maskPresentationLayer.cornerRadius,
                cornerCurveExponent: maskPresentationLayer.cornerCurveRenderExponent
                    ?? Float(CornerCurveRenderConfiguration.circularExponent),
                cornerRadii: maskPresentationLayer.cornerRadiiComponents,
                context: .contentMask
            )
        } catch let failure {
            recordMaskRenderFailure(failure)
            return false
        }

        // Render the mask layer (this writes to stencil buffer)
        let maskModelMatrix = maskPresentationLayer.modelMatrix(parentMatrix: parentMatrix)

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(configuration.size.x, 0, 0, 0),
            SIMD4<Float>(0, configuration.size.y, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = maskModelMatrix * scaleMatrix
        guard matrixIsFinite(finalMatrix) else {
            recordMaskRenderFailure(.invalidTransform(.contentMask))
            return false
        }
        guard stencilStateIsValid else {
            recordMaskRenderFailure(
                .invalidStencilState(
                    depth: maskNestingDepth,
                    reference: currentStencilValue
                )
            )
            return false
        }
        guard currentStencilValue < UInt32.max,
              maskNestingDepth < Int.max else {
            recordMaskRenderFailure(.stencilReferenceOverflow(.contentMask))
            return false
        }

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

        guard let allocation = allocateVertices(count: vertices.count) else {
            recordMaskRenderFailure(.vertexCapacityExceeded(.contentMask))
            return false
        }
        let (vertexOffset, layerIndex) = allocation

        // Mutate stencil state only after every CPU-side prerequisite succeeds.
        currentStencilValue += 1
        renderPass.setPipeline(stencilWritePipeline)
        renderPass.setStencilReference(currentStencilValue)

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: 1.0,  // Full opacity for stencil write
            cornerRadius: configuration.cornerRadius,
            layerSize: configuration.size,
            cornerCurveExponent: configuration.cornerCurveExponent,
            cornerRadii: configuration.cornerRadii
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
        renderPass.setPipeline(stencilTestPipeline)
        maskNestingDepth += 1
        return true
    }

    /// Clears the stencil mask after layer rendering is complete.
    /// Supports nesting: if still inside a parent mask, restores the stencil test
    /// pipeline with the parent's stencil reference value instead of fully disabling.
    private func clearStencilMask(renderPass: GPURenderPassEncoder) {
        guard stencilStateIsValid,
              maskNestingDepth > 0,
              currentStencilValue > 0 else {
            recordMaskRenderFailure(
                .invalidStencilState(
                    depth: maskNestingDepth,
                    reference: currentStencilValue
                )
            )
            return
        }

        let nextDepth = maskNestingDepth - 1
        let nextReference = currentStencilValue - 1
        let restorationPipeline: GPURenderPipeline?
        if nextDepth > 0 {
            restorationPipeline = stencilTestPipeline
        } else {
            restorationPipeline = pipeline
        }
        guard let restorationPipeline else {
            recordMaskRenderFailure(.pipelineUnavailable(.restoration))
            return
        }

        maskNestingDepth = nextDepth
        currentStencilValue = nextReference
        renderPass.setPipeline(restorationPipeline)
        renderPass.setStencilReference(nextDepth > 0 ? nextReference : 0)
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
    ) -> Bool {
        guard let stencilWriteRoundedPipeline = stencilWriteRoundedPipeline,
              let stencilTestPipeline = stencilTestPipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup,
              pipeline != nil else {
            recordMaskRenderFailure(.resourcesUnavailable(.roundedClip))
            return false
        }
        if let error = layer.cornerCurveRenderError {
            recordCornerCurveRenderFailure(.roundedClip(error))
            recordMaskRenderFailure(
                .unsupportedCornerCurve(.roundedClip, layer.cornerCurve.rawValue)
            )
            return false
        }

        let configuration: CAMaskRenderConfiguration
        do {
            configuration = try CAMaskRenderConfiguration(
                bounds: layer.bounds,
                cornerRadius: layer.cornerRadius,
                cornerCurveExponent: layer.cornerCurveRenderExponent
                    ?? Float(CornerCurveRenderConfiguration.circularExponent),
                cornerRadii: layer.cornerRadiiComponents,
                context: .roundedClip
            )
        } catch let failure {
            recordMaskRenderFailure(failure)
            return false
        }

        // Build transform that maps the unit quad to the layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(configuration.size.x, 0, 0, 0),
            SIMD4<Float>(0, configuration.size.y, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix
        guard matrixIsFinite(finalMatrix) else {
            recordMaskRenderFailure(.invalidTransform(.roundedClip))
            return false
        }
        guard stencilStateIsValid else {
            recordMaskRenderFailure(
                .invalidStencilState(
                    depth: maskNestingDepth,
                    reference: currentStencilValue
                )
            )
            return false
        }
        guard currentStencilValue < UInt32.max,
              maskNestingDepth < Int.max else {
            recordMaskRenderFailure(.stencilReferenceOverflow(.roundedClip))
            return false
        }

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

        guard let allocation = allocateVertices(count: vertices.count) else {
            recordMaskRenderFailure(.vertexCapacityExceeded(.roundedClip))
            return false
        }
        let (vertexOffset, layerIndex) = allocation

        // Mutate stencil state only after every CPU-side prerequisite succeeds.
        currentStencilValue += 1
        renderPass.setPipeline(stencilWriteRoundedPipeline)
        renderPass.setStencilReference(currentStencilValue)

        // Set uniforms with corner radius info for SDF calculation in the shader
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: 1.0,
            cornerRadius: configuration.cornerRadius,
            layerSize: configuration.size,
            cornerCurveExponent: configuration.cornerCurveExponent
        )

        uniforms.cornerRadii = configuration.cornerRadii

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
        return true
    }

    // MARK: - Replicator Layer Rendering

    private func recordReplicatorRenderFailure(
        _ failure: CAReplicatorRenderFailure,
        for replicatorModelLayer: CAReplicatorLayer
    ) {
        if replicatorRenderFailureGeneration == .max {
            replicatorRenderFailureGeneration = 0
        } else {
            replicatorRenderFailureGeneration += 1
        }
        lastReplicatorRenderFailure = failure
        let key = renderKey(for: replicatorModelLayer)
        if reportedReplicatorRenderFailureKeys.insert(key).inserted {
            replicatorRenderFailureCount += 1
        }
    }

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
        let configuration: CAReplicatorRenderConfiguration
        do {
            configuration = try CAReplicatorRenderConfiguration(
                layer: replicatorLayer,
                maximumInstanceCount: Self.maxLayers
            )
        } catch {
            recordReplicatorRenderFailure(error, for: replicatorModelLayer)
            return
        }
        guard configuration.instanceCount > 0 else { return }

        if configuration.preservesDepth {
            renderDepthPreservingReplicatorSublayers(
                replicatorModelLayer: replicatorModelLayer,
                configuration: configuration,
                sublayers: sublayers,
                renderPass: renderPass,
                parentMatrix: parentMatrix
            )
            return
        }

        // Render each flattened instance.
        var cumulativeTransform = CATransform3DIdentity
        for instanceIndex in 0..<configuration.instanceCount {
            guard CAReplicatorRenderConfiguration.isFinite(cumulativeTransform) else {
                recordReplicatorRenderFailure(
                    .cumulativeTransformOverflow(instanceIndex: instanceIndex),
                    for: replicatorModelLayer
                )
                return
            }
            let instanceColor: SIMD4<Float>
            let timeOffset: CFTimeInterval
            do {
                instanceColor = try configuration.color(at: instanceIndex)
                timeOffset = try configuration.timeOffset(at: instanceIndex)
            } catch {
                recordReplicatorRenderFailure(error, for: replicatorModelLayer)
                return
            }
            // Calculate instance matrix
            let instanceMatrix = parentMatrix * cumulativeTransform.matrix4x4

            // Calculate time offset for this instance
            // Positive instanceDelay means later instances start their animations later
            // So instance N's animations are evaluated at (currentTime - N * instanceDelay)
            let inheritedColor = currentReplicatorColor
            let inheritedTimeOffset = currentReplicatorTimeOffset
            let combinedTimeOffset = inheritedTimeOffset + timeOffset
            guard combinedTimeOffset.isFinite else {
                recordReplicatorRenderFailure(
                    .instanceTimeOffsetOverflow(instanceIndex: instanceIndex),
                    for: replicatorModelLayer
                )
                return
            }
            replicatorColorStack.append(inheritedColor * instanceColor)
            replicatorTimeOffsetStack.append(combinedTimeOffset)
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
            if instanceIndex + 1 < configuration.instanceCount {
                do {
                    cumulativeTransform = try configuration.nextTransform(
                        after: cumulativeTransform,
                        nextInstanceIndex: instanceIndex + 1
                    )
                } catch {
                    recordReplicatorRenderFailure(error, for: replicatorModelLayer)
                    return
                }
            }
        }
    }

    private func renderDepthPreservingReplicatorSublayers(
        replicatorModelLayer: CAReplicatorLayer,
        configuration: CAReplicatorRenderConfiguration,
        sublayers: [CALayer],
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        struct DrawItem {
            let layer: CALayer
            let parentMatrix: Matrix4x4
            let color: SIMD4<Float>
            let timeOffset: CFTimeInterval
            let instanceIndex: Int
            let depth: Float
            let insertionOrder: Int
        }

        let inheritedColor = currentReplicatorColor
        let inheritedTimeOffset = currentReplicatorTimeOffset
        var items: [DrawItem] = []
        var cumulativeTransform = CATransform3DIdentity
        var insertionOrder = 0

        for instanceIndex in 0..<configuration.instanceCount {
            guard CAReplicatorRenderConfiguration.isFinite(cumulativeTransform) else {
                recordReplicatorRenderFailure(
                    .cumulativeTransformOverflow(instanceIndex: instanceIndex),
                    for: replicatorModelLayer
                )
                return
            }
            let instanceColor: SIMD4<Float>
            let instanceTimeOffset: CFTimeInterval
            do {
                instanceColor = try configuration.color(at: instanceIndex)
                instanceTimeOffset = try configuration.timeOffset(at: instanceIndex)
            } catch {
                recordReplicatorRenderFailure(error, for: replicatorModelLayer)
                return
            }
            let instanceMatrix = parentMatrix * cumulativeTransform.matrix4x4
            let timeOffset = inheritedTimeOffset + instanceTimeOffset
            guard timeOffset.isFinite else {
                recordReplicatorRenderFailure(
                    .instanceTimeOffsetOverflow(instanceIndex: instanceIndex),
                    for: replicatorModelLayer
                )
                return
            }

            for (sublayerIndex, sublayer) in sublayers.enumerated() {
                let depth: Float
                do {
                    depth = try projectedCenterDepth(
                        of: sublayer,
                        parentMatrix: instanceMatrix,
                        timeOffset: timeOffset
                    )
                } catch {
                    recordReplicatorRenderFailure(.invalidProjectedDepth(
                        instanceIndex: instanceIndex,
                        sublayerIndex: sublayerIndex,
                        reason: error
                    ), for: replicatorModelLayer)
                    return
                }
                items.append(DrawItem(
                    layer: sublayer,
                    parentMatrix: instanceMatrix,
                    color: inheritedColor * instanceColor,
                    timeOffset: timeOffset,
                    instanceIndex: instanceIndex,
                    depth: depth,
                    insertionOrder: insertionOrder
                ))
                insertionOrder += 1
            }
            if instanceIndex + 1 < configuration.instanceCount {
                do {
                    cumulativeTransform = try configuration.nextTransform(
                        after: cumulativeTransform,
                        nextInstanceIndex: instanceIndex + 1
                    )
                } catch {
                    recordReplicatorRenderFailure(error, for: replicatorModelLayer)
                    return
                }
            }
        }

        // Resolve every CPU-side input before clearing or otherwise mutating
        // the active depth state.
        let depthConfiguration: CADepthGroupRenderConfiguration
        do {
            depthConfiguration = try CADepthGroupRenderConfiguration(
                currentNestingDepth: transformDepthNesting
            )
        } catch {
            recordReplicatorRenderFailure(
                .depthGroupStateFailure(error),
                for: replicatorModelLayer
            )
            return
        }
        if depthConfiguration.requiresDepthClear {
            guard let depthClearPipeline else {
                recordReplicatorRenderFailure(
                    .depthResourcesUnavailable,
                    for: replicatorModelLayer
                )
                return
            }
            renderPass.setPipeline(depthClearPipeline)
            renderPass.draw(vertexCount: 3)
        }
        let previousDepthNesting = transformDepthNesting
        transformDepthNesting = depthConfiguration.enteredNestingDepth
        defer { transformDepthNesting = previousDepthNesting }

        // Depth testing resolves opaque intersections. Far-to-near submission
        // preserves source-over blending for translucent replicated planes.
        items.sort {
            $0.depth == $1.depth
                ? $0.insertionOrder < $1.insertionOrder
                : $0.depth < $1.depth
        }

        for item in items {
            replicatorColorStack.append(item.color)
            replicatorTimeOffsetStack.append(item.timeOffset)
            replicatorInstancePath.append(ReplicatorInstancePathComponent(
                replicator: ObjectIdentifier(replicatorModelLayer),
                instanceIndex: item.instanceIndex
            ))
            renderLayer(
                item.layer,
                renderPass: renderPass,
                parentMatrix: item.parentMatrix
            )
            _ = replicatorInstancePath.popLast()
            _ = replicatorTimeOffsetStack.popLast()
            _ = replicatorColorStack.popLast()
        }
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

    private func recordContentsRenderFailure(_ failure: CAContentsRenderFailure) {
        contentsRenderFailureCount += 1
        lastContentsRenderFailure = failure
        if case .imageConversion(let error) = failure {
            lastContentsConversionError = error
        }
    }

    /// Checks if 9-patch (contentsCenter) scaling is needed.
    ///
    /// Returns true if contentsCenter is not the default (0, 0, 1, 1),
    /// indicating that 9-slice scaling should be applied.
    private func needs9PatchScaling(_ layer: CALayer) -> Bool {
        ContentsRenderConfiguration.usesNineSlice(
            gravity: layer.contentsGravity,
            contentsCenter: layer.contentsCenter
        )
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
        guard let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let selectedPipeline = selectTexturedPipeline(for: layer) else {
            recordContentsRenderFailure(.rendererResourcesUnavailable)
            return
        }

        let boundsWidth = layer.bounds.width
        let boundsHeight = layer.bounds.height
        let configuration: ContentsRenderConfiguration
        do {
            configuration = try ContentsRenderConfiguration(
                imageSize: CGSize(width: imageWidth, height: imageHeight),
                boundsSize: layer.bounds.size,
                contentsRect: layer.contentsRect,
                contentsCenter: layer.contentsCenter,
                contentsScale: layer.contentsScale,
                gravity: layer.contentsGravity
            )
        } catch {
            recordContentsRenderFailure(.nineSliceConfiguration(error))
            return
        }

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
        let effectiveCornerCurveExponent = layer.masksToBounds
            ? (layer.cornerCurveRenderExponent ?? Float(CornerCurveRenderConfiguration.circularExponent))
            : Float(CornerCurveRenderConfiguration.circularExponent)

        // Create uniforms (shared for all 9 patches)
        let uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: effectiveCornerRadius,
            layerSize: SIMD2<Float>(Float(boundsWidth), Float(boundsHeight)),
            cornerRadii: effectiveCornerRadii,
            edgeAntialiasingMask: layer.edgeAntialiasingMaskValue,
            cornerCurveExponent: effectiveCornerCurveExponent
        )

        let white = currentReplicatorColor
        let totalVertexCount = configuration.patches.count * 6
        guard totalVertexCount > 0 else { return }
        guard let (baseVertexOffset, layerIndex) = allocateVertices(
            count: totalVertexCount
        ) else {
            recordContentsRenderFailure(.nineSliceVertexCapacityExceeded)
            return
        }

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        var uploadedUniforms = uniforms
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: createFloat32Array(from: &uploadedUniforms)
        )
        let texturedBindGroup = cachedTexturedBindGroup(
            cacheKey: .image(ObjectIdentifier(contents)),
            gpuTexture: gpuTexture,
            device: device,
            layout: texturedBindGroupLayout,
            sampler: textureSampler,
            uniformBuffer: uniformBuffer,
            uniformStride: UInt64(MemoryLayout<TexturedUniforms>.stride)
        )
        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(
            0,
            bindGroup: texturedBindGroup,
            dynamicOffsets: [UInt32(uniformOffset)]
        )

        // Render each patch
        for (patchIndex, patch) in configuration.patches.enumerated() {
            let destination = patch.destinationRect
            let source = patch.sourceUnitRect
            let pMinX = Float(destination.minX / boundsWidth)
            let pMaxX = Float(destination.maxX / boundsWidth)
            let pMinY = Float(destination.minY / boundsHeight)
            let pMaxY = Float(destination.maxY / boundsHeight)
            let uMinX = Float(source.minX)
            let uMaxX = Float(source.maxX)
            let uMinY = Float(source.minY)
            let uMaxY = Float(source.maxY)

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

            let vertexOffset = baseVertexOffset + UInt64(
                patchIndex * 6 * MemoryLayout<CARendererVertex>.stride
            )
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            // Render this patch
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
        guard let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let selectedPipeline = selectTexturedPipeline(for: layer) else {
            recordContentsRenderFailure(.rendererResourcesUnavailable)
            return
        }

        // Get or create GPU texture from CGImage using the texture manager.
        // The manager retains `contents` for as long as the texture is
        // cached so `ObjectIdentifier(contents)` stays unique across the
        // dependent view / bind group caches keyed by the same identity.
        let imageWidth = contents.width
        let imageHeight = contents.height
        let textureFormat = CGImageTexturePixelFormat.recommended(for: contents)
        let memorySizeBytes: UInt64
        do {
            memorySizeBytes = try mipmappedRGBAByteCount(
                width: imageWidth,
                height: imageHeight,
                format: textureFormat,
                device: device
            )
        } catch {
            recordContentsConversionFailure(error)
            return
        }
        guard let textureManager else {
            recordContentsRenderFailure(.textureManagerUnavailable)
            return
        }
        var textureConversionError: CAImageContentsConversionError?
        let gpuTexture = textureManager.getOrCreateTexture(
            for: contents,
            width: imageWidth,
            height: imageHeight,
            memorySizeBytes: memorySizeBytes,
            factory: {
                do {
                    return try self.createGPUTexture(
                        from: contents,
                        format: textureFormat,
                        device: device
                    )
                } catch let error as CAImageContentsConversionError {
                    textureConversionError = error
                    return nil
                } catch {
                    textureConversionError = .conversionFailed
                    return nil
                }
            }
        )
        guard let gpuTexture else {
            if let textureConversionError {
                recordContentsConversionFailure(textureConversionError)
            } else {
                recordContentsRenderFailure(.textureCreationFailed)
            }
            return
        }

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

        // Standard single-quad rendering. The selected contentsRect controls
        // both the UV range and the logical size used by contentsGravity.
        let destRect: CGRect
        do {
            destRect = try ContentsRenderConfiguration.destinationRect(
                imageSize: CGSize(width: imageWidth, height: imageHeight),
                boundsSize: layer.bounds.size,
                contentsRect: layer.contentsRect,
                contentsScale: layer.contentsScale,
                gravity: layer.contentsGravity
            )
        } catch {
            recordContentsRenderFailure(.standardConfiguration(error))
            return
        }

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

        guard let allocation = allocateVertices(count: vertices.count) else {
            recordContentsRenderFailure(.standardVertexCapacityExceeded)
            return
        }
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
        let effectiveCornerCurveExponent = layer.masksToBounds
            ? (layer.cornerCurveRenderExponent ?? Float(CornerCurveRenderConfiguration.circularExponent))
            : Float(CornerCurveRenderConfiguration.circularExponent)

        // Create uniforms for textured rendering
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: effectiveCornerRadius,
            layerSize: SIMD2<Float>(Float(boundsWidth), Float(boundsHeight)),
            cornerRadii: effectiveCornerRadii,
            edgeAntialiasingMask: layer.edgeAntialiasingMaskValue,
            cornerCurveExponent: effectiveCornerCurveExponent
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
        renderPass.setPipeline(selectedPipeline)
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
    private func createGPUTexture(
        from cgImage: CGImage,
        format: CGImageTexturePixelFormat,
        device: GPUDevice
    ) throws(CAImageContentsConversionError) -> GPUTexture {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0 && height > 0 else {
            throw .invalidDimensions(width: width, height: height)
        }

        let baseLevel = try CGImageTextureStorageConverter.convert(cgImage, to: format)

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
        let gpuTextureFormat: GPUTextureFormat
        switch format {
        case .rgba8Unorm: gpuTextureFormat = .rgba8unorm
        case .rgba16Float: gpuTextureFormat = .rgba16float
        }
        let textureDescriptor = GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            mipLevelCount: mipLevelCount,
            format: gpuTextureFormat,
            usage: [.textureBinding, .copyDst, .renderAttachment]
        )

        let texture = device.createTexture(descriptor: textureDescriptor)

        var levelStorage = baseLevel
        for level in 0..<mipLevelCount {
            device.queue.writeTexture(
                destination: GPUImageCopyTexture(texture: texture, mipLevel: level),
                data: createUint8Array(from: levelStorage.data),
                dataLayout: GPUImageDataLayout(
                    offset: 0,
                    bytesPerRow: UInt32(levelStorage.bytesPerRow),
                    rowsPerImage: UInt32(levelStorage.height)
                ),
                size: GPUExtent3D(
                    width: UInt32(levelStorage.width),
                    height: UInt32(levelStorage.height)
                )
            )
            guard level + 1 < mipLevelCount else { continue }
            levelStorage = try CGImageTextureMipGenerator.nextLevel(from: levelStorage)
        }

        return texture
    }

    private func mipmappedRGBAByteCount(
        width: Int,
        height: Int,
        format: CGImageTexturePixelFormat,
        device: GPUDevice
    ) throws(CAImageContentsConversionError) -> UInt64 {
        guard width > 0, height > 0 else {
            throw .invalidDimensions(width: width, height: height)
        }
        let maximumDimension = max(1, Int(device.limits.maxTextureDimension2D))
        guard width <= maximumDimension, height <= maximumDimension else {
            throw .dimensionsExceedTextureLimit(
                width: width,
                height: height,
                maximum: maximumDimension
            )
        }

        return try CGImageTextureStorage.mipmappedByteCount(
            width: width,
            height: height,
            format: format
        )
    }

    private func recordContentsConversionFailure(_ error: CAImageContentsConversionError) {
        recordContentsRenderFailure(.imageConversion(error))
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
        for texture in textTextureCache.values {
            texture.destroy()
        }
        textTextureCache.removeAll()
        textTextureAccessOrder.removeAll()
        textTextureCacheByteCount = 0
        texturedTextureViewCache.removeAll(keepingCapacity: true)
        perFrameTexturedBindGroupCache.removeAll(keepingCapacity: true)
    }

    // MARK: - Text Layer Rendering

    private struct TextTextureCacheKey: Hashable {
        let text: String
        let pixelWidth: Int
        let pixelHeight: Int
        let fontSize: CGFloat
        let contentsScale: CGFloat
        let alignmentMode: CATextLayerAlignmentMode
        let truncationMode: CATextLayerTruncationMode
        let fontFamily: String
        let red: Float
        let green: Float
        let blue: Float
        let alpha: Float
        let isWrapped: Bool

        var byteCount: UInt64 {
            UInt64(pixelWidth) * UInt64(pixelHeight) * 4
        }
    }

    /// Cache for text textures to avoid recreating them every frame.
    private var textTextureCache: [TextTextureCacheKey: GPUTexture] = [:]

    /// LRU order for text textures. Dynamic labels can otherwise create one
    /// GPU texture per distinct string for the lifetime of the renderer.
    private var textTextureAccessOrder: [TextTextureCacheKey] = []

    /// Estimated GPU storage retained by the text texture cache.
    private var textTextureCacheByteCount: UInt64 = 0

    /// Set when Canvas2D returns a missing or invalid width inside a nonthrowing
    /// layout callback. The complete offscreen result is rejected before upload.
    private var textMeasurementFailureDetected = false

    private let maxTextTextureCacheEntries = 128
    private let maxTextTextureCacheBytes: UInt64 = 64 * 1024 * 1024

    private func touchTextTexture(_ key: TextTextureCacheKey) {
        textTextureAccessOrder.removeAll { $0 == key }
        textTextureAccessOrder.append(key)
    }

    private func cacheTextTexture(_ texture: GPUTexture, for key: TextTextureCacheKey) {
        if let replacedTexture = textTextureCache.updateValue(texture, forKey: key) {
            replacedTexture.destroy()
            textTextureCacheByteCount -= min(textTextureCacheByteCount, key.byteCount)
        }
        textTextureCacheByteCount += key.byteCount
        touchTextTexture(key)
        while (textTextureCache.count > maxTextTextureCacheEntries
               || textTextureCacheByteCount > maxTextTextureCacheBytes),
              let oldest = textTextureAccessOrder.first {
            textTextureAccessOrder.removeFirst()
            if let removedTexture = textTextureCache.removeValue(forKey: oldest) {
                removedTexture.destroy()
                textTextureCacheByteCount -= min(textTextureCacheByteCount, oldest.byteCount)
            }
        }
    }

    private func recordTextRenderFailure(_ failure: CATextRenderFailure) {
        textRenderFailureCount += 1
        lastTextRenderFailure = failure
    }

    /// Renders a CATextLayer using Canvas2D for text rasterization and texture-based rendering.
    ///
    /// This method uses an offscreen Canvas2D to render the text, creates a WebGPU texture
    /// from the canvas data, and displays it using the textured pipeline.
    private func renderTextLayer(
        _ textLayer: CATextLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        let configuration: CATextRenderConfiguration
        do {
            configuration = try CATextRenderConfiguration(layer: textLayer)
        } catch {
            recordTextRenderFailure(error)
            return
        }
        let text = configuration.text
        guard !text.isEmpty else {
            return
        }

        let logicalSize = configuration.bounds.size
        guard logicalSize.width > 0, logicalSize.height > 0 else {
            return
        }
        guard let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let pipeline = pipeline,
              let selectedPipeline = selectTexturedPipeline(for: textLayer) else {
            recordTextRenderFailure(.rendererResourcesUnavailable)
            return
        }
        let maximumTextureDimension = max(1, Int(device.limits.maxTextureDimension2D))
        let scaledWidth = logicalSize.width * configuration.contentsScale
        let scaledHeight = logicalSize.height * configuration.contentsScale
        guard scaledWidth.isFinite,
              scaledHeight.isFinite,
              scaledWidth > 0,
              scaledHeight > 0,
              scaledWidth <= CGFloat(maximumTextureDimension),
              scaledHeight <= CGFloat(maximumTextureDimension) else {
            recordTextRenderFailure(.textureDimensionsUnsupported)
            return
        }
        let pixelWidth = Int(ceil(scaledWidth))
        let pixelHeight = Int(ceil(scaledHeight))
        guard UInt64(pixelWidth) * UInt64(pixelHeight) * 4 <= maxTextTextureCacheBytes else {
            recordTextRenderFailure(.textureDimensionsUnsupported)
            return
        }

        let quadConfiguration: CATextQuadRenderConfiguration
        do {
            quadConfiguration = try CATextQuadRenderConfiguration(
                bounds: configuration.bounds,
                color: currentReplicatorColor,
                opacity: currentEffectiveOpacity,
                masksToBounds: textLayer.masksToBounds,
                cornerRadius: textLayer.cornerRadius,
                cornerCurveExponent: textLayer.cornerCurveRenderExponent
                    ?? Float(CornerCurveRenderConfiguration.circularExponent),
                cornerRadii: textLayer.cornerRadiiComponents
            )
        } catch let failure {
            recordTextRenderFailure(failure)
            return
        }

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(quadConfiguration.size.x, 0, 0, 0),
            SIMD4<Float>(0, quadConfiguration.size.y, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix
        guard matrixIsFinite(finalMatrix) else {
            recordTextRenderFailure(.invalidTransform)
            return
        }

        var vertices: [CARendererVertex] = [
            // Canvas text uses top-left texture coordinates while layer space is Y-up.
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 1), color: quadConfiguration.color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: quadConfiguration.color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: quadConfiguration.color),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 1), color: quadConfiguration.color),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 0), color: quadConfiguration.color),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 0), color: quadConfiguration.color),
        ]
        guard availableVertexAllocationSize(count: vertices.count) != nil else {
            droppedLayerCount += 1
            recordTextRenderFailure(.vertexCapacityExceeded)
            return
        }

        // Create cache key based on text content and properties.
        // Includes font fingerprint and foreground color so that two layers
        // with the same text but different font / color do not collide.
        let cacheKey = textCacheKey(
            text: text,
            width: pixelWidth,
            height: pixelHeight,
            configuration: configuration
        )

        // Check cache first
        let gpuTexture: GPUTexture
        let shouldCacheTexture: Bool
        if let cached = textTextureCache[cacheKey] {
            touchTextTexture(cacheKey)
            gpuTexture = cached
            shouldCacheTexture = false
        } else {
            // Create offscreen canvas for text rendering
            let document = JSObject.global.document
            let offscreenCanvas = document.createElement("canvas")
            offscreenCanvas.width = .number(Double(pixelWidth))
            offscreenCanvas.height = .number(Double(pixelHeight))

            guard let ctx = offscreenCanvas.getContext("2d").object else {
                recordTextRenderFailure(.canvas2DUnavailable)
                return
            }

            // The layer background is rendered by the normal background pass.
            // Keeping the text texture transparent prevents translucent backgrounds
            // from being composited twice.
            _ = ctx.clearRect!(0, 0, pixelWidth, pixelHeight)
            _ = ctx.scale!(
                Double(configuration.contentsScale),
                Double(configuration.contentsScale)
            )

            // Set font
            ctx.font = .string(
                "\(configuration.fontSize)px \(configuration.cssFontFamily)"
            )

            // Set text color
            let color = configuration.foregroundRGBA
            ctx.fillStyle = .string(
                "rgba(\(Int((color.x * 255).rounded())),"
                    + "\(Int((color.y * 255).rounded())),"
                    + "\(Int((color.z * 255).rounded())),\(color.w))"
            )

            // Set text alignment
            switch configuration.alignmentMode {
            case .left:
                ctx.textAlign = .string("left")
            case .right:
                ctx.textAlign = .string("right")
            case .center:
                ctx.textAlign = .string("center")
            case .justified, .natural:
                ctx.textAlign = .string("start")
            default:
                recordTextRenderFailure(
                    .unsupportedAlignmentMode(configuration.alignmentMode.rawValue)
                )
                return
            }

            ctx.textBaseline = .string("top")

            // Calculate text position based on alignment
            let x: Double
            switch configuration.alignmentMode {
            case .center:
                x = Double(logicalSize.width) / 2
            case .right:
                x = Double(logicalSize.width)
            default:
                x = 0
            }

            // Draw measured single-line or multiline text into the layer texture.
            textMeasurementFailureDetected = false
            if configuration.isWrapped || CATextLayoutEngine.containsParagraphBreak(text) {
                drawMultilineText(
                    ctx: ctx,
                    text: text,
                    x: x,
                    y: 0,
                    maxWidth: Double(logicalSize.width),
                    maxHeight: Double(logicalSize.height),
                    lineHeight: configuration.lineHeight,
                    truncationMode: configuration.truncationMode,
                    alignmentMode: configuration.alignmentMode,
                    wrapsToWidth: configuration.isWrapped
                )
            } else {
                let displayedText = CATextLayoutEngine.truncatedText(
                    text,
                    mode: configuration.truncationMode,
                    maximumWidth: logicalSize.width,
                    measure: { candidate in
                        CGFloat(self.measureWidth(ctx: ctx, candidate))
                    }
                )
                _ = ctx.fillText!(displayedText, x, Double(configuration.fontSize * 0.1))
            }
            if textMeasurementFailureDetected {
                recordTextRenderFailure(.textMeasurementUnavailable)
                return
            }

            // Get image data from canvas
            // Reset the transform because getImageData uses untransformed pixel coordinates.
            _ = ctx.resetTransform!()
            guard let imageData = ctx.getImageData!(0, 0, pixelWidth, pixelHeight).object else {
                recordTextRenderFailure(.imageDataUnavailable)
                return
            }
            guard let dataArray = imageData.data.object else {
                recordTextRenderFailure(.imageDataStorageUnavailable)
                return
            }

            // Create WebGPU texture
            let textureDescriptor = GPUTextureDescriptor(
                size: GPUExtent3D(width: UInt32(pixelWidth), height: UInt32(pixelHeight)),
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
                    bytesPerRow: UInt32(pixelWidth * 4),
                    rowsPerImage: UInt32(pixelHeight)
                ),
                size: GPUExtent3D(width: UInt32(pixelWidth), height: UInt32(pixelHeight))
            )

            gpuTexture = texture
            shouldCacheTexture = true
        }

        guard let allocation = allocateVertices(count: vertices.count) else {
            if shouldCacheTexture {
                gpuTexture.destroy()
            }
            recordTextRenderFailure(.vertexCapacityExceeded)
            return
        }
        let (vertexOffset, layerIndex) = allocation
        if shouldCacheTexture {
            cacheTextTexture(gpuTexture, for: cacheKey)
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

        // Update uniforms
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: quadConfiguration.opacity,
            cornerRadius: quadConfiguration.cornerRadius,
            layerSize: quadConfiguration.size,
            cornerRadii: quadConfiguration.cornerRadii,
            edgeAntialiasingMask: textLayer.edgeAntialiasingMaskValue,
            cornerCurveExponent: quadConfiguration.cornerCurveExponent
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
        renderPass.setPipeline(selectedPipeline)
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
        configuration: CATextRenderConfiguration
    ) -> TextTextureCacheKey {
        let color = configuration.foregroundRGBA
        return TextTextureCacheKey(
            text: text,
            pixelWidth: width,
            pixelHeight: height,
            fontSize: configuration.fontSize,
            contentsScale: configuration.contentsScale,
            alignmentMode: configuration.alignmentMode,
            truncationMode: configuration.truncationMode,
            fontFamily: configuration.fontFamily,
            red: color.x,
            green: color.y,
            blue: color.z,
            alpha: color.w,
            isWrapped: configuration.isWrapped
        )
    }

    /// Canvas2D text width for a string.
    private func measureWidth(ctx: JSObject, _ s: String) -> Double {
        let metrics = ctx.measureText!(s)
        guard let width = metrics.width.number, width.isFinite, width >= 0 else {
            textMeasurementFailureDetected = true
            return 0
        }
        return width
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
        guard lineHeight.isFinite, lineHeight > 0 else {
            textMeasurementFailureDetected = true
            return
        }
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

    // MARK: - Shape Layer Rendering

    private func recordShapeRenderFailure(_ failure: CAShapeRenderFailure) {
        shapeRenderFailureCount += 1
        lastShapeRenderFailure = failure
    }

    private func shapeColorComponents(
        _ color: CGColor,
        invalidFailure: CAShapeRenderFailure
    ) -> SIMD4<Float>? {
        guard let converted = color.converted(
            to: .deviceRGB,
            intent: .defaultIntent,
            options: nil
        ), let components = converted.components,
           components.count == 4,
           components.allSatisfy(\.isFinite) else {
            recordShapeRenderFailure(invalidFailure)
            return nil
        }
        let result = SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
        guard result.x.isFinite,
              result.y.isFinite,
              result.z.isFinite,
              result.w.isFinite else {
            recordShapeRenderFailure(invalidFailure)
            return nil
        }
        return replicatedColor(result)
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
            recordShapeRenderFailure(.rendererResourcesUnavailable)
            return
        }
        do {
            try ShapeFillTessellator.validate(path)
        } catch {
            recordShapeRenderFailure(.pathValidationFailed(error))
            return
        }

        // Renderers for textured, gradient, filter, and emitter layers change
        // the active pipeline. A shape must therefore select its own solid
        // pipeline instead of depending on whichever sibling rendered first.
        guard let selectedPipeline = stencilAwarePipeline() else {
            recordMaskRenderFailure(.pipelineUnavailable(.activeStencil))
            return
        }
        renderPass.setPipeline(selectedPipeline)

        // Render the complete fill in one tessellation so subpath winding and
        // even-odd holes are preserved across contour boundaries.
        if let fillColor = shapeLayer.fillColor {
            let fillPoints: [CGPoint]
            do {
                fillPoints = try ShapeFillTessellator.triangles(
                    for: path,
                    rule: shapeLayer.fillRule
                )
            } catch {
                recordShapeRenderFailure(.fillTessellationFailed(error))
                fillPoints = []
            }

            if !fillPoints.isEmpty,
               let colorComponents = shapeColorComponents(
                   fillColor,
                   invalidFailure: .invalidFillColor
               ) {
                let bounds = shapeLayer.bounds
                let hasValidBounds = bounds.width > 0 && bounds.height > 0
                var vertices = fillPoints.map { point in
                    let layerCoordinate = hasValidBounds
                        ? SIMD2(
                            Float((point.x - bounds.minX) / bounds.width),
                            Float((point.y - bounds.minY) / bounds.height)
                        )
                        : .zero
                    return CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: layerCoordinate,
                        color: colorComponents
                    )
                }
                if let (vertexOffset, uniformIndex) = allocateVertices(
                    count: vertices.count
                ) {
                    var uniforms = CARendererUniforms(
                        mvpMatrix: modelMatrix,
                        opacity: currentEffectiveOpacity,
                        cornerRadius: 0,
                        layerSize: SIMD2(Float(bounds.width), Float(bounds.height)),
                        edgeAntialiasingMask: hasValidBounds ? shapeLayer.edgeAntialiasingMaskValue : 0
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
                    renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
                    renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
                    renderPass.draw(vertexCount: UInt32(vertices.count))
                    shapeFillDrawCount += 1
                    shapeFillVertexCount += vertices.count
                } else {
                    recordShapeRenderFailure(.fillVertexCapacityExceeded)
                }
            }
        }

        // Render stroke if strokeColor is set
        if let strokeColor = shapeLayer.strokeColor {
            guard shapeLayer.lineWidth.isFinite else {
                recordShapeRenderFailure(.nonFiniteLineWidth)
                return
            }
            guard shapeLayer.lineWidth > 0 else { return }
            let strokePoints: [CGPoint]
            do {
                strokePoints = try ShapeStrokeTessellator.triangles(
                    for: path,
                    lineWidth: shapeLayer.lineWidth,
                    lineCap: shapeLayer.lineCap,
                    lineJoin: shapeLayer.lineJoin,
                    miterLimit: shapeLayer.miterLimit,
                    dashPattern: shapeLayer.lineDashPattern,
                    dashPhase: shapeLayer.lineDashPhase,
                    strokeStart: shapeLayer.strokeStart,
                    strokeEnd: shapeLayer.strokeEnd
                )
            } catch {
                recordShapeRenderFailure(.strokeTessellationFailed(error))
                strokePoints = []
            }

            if !strokePoints.isEmpty,
               let colorComponents = shapeColorComponents(
                   strokeColor,
                   invalidFailure: .invalidStrokeColor
               ) {
                let bounds = shapeLayer.bounds
                let hasValidBounds = bounds.width > 0 && bounds.height > 0
                var vertices = strokePoints.map { point in
                    CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: hasValidBounds
                            ? SIMD2(
                                Float((point.x - bounds.minX) / bounds.width),
                                Float((point.y - bounds.minY) / bounds.height)
                            )
                            : .zero,
                        color: colorComponents
                    )
                }
                if let (vertexOffset, uniformIndex) = allocateVertices(
                    count: vertices.count
                ) {
                    var uniforms = CARendererUniforms(
                        mvpMatrix: modelMatrix,
                        opacity: currentEffectiveOpacity,
                        cornerRadius: 0,
                        layerSize: SIMD2(Float(bounds.width), Float(bounds.height)),
                        edgeAntialiasingMask: hasValidBounds ? shapeLayer.edgeAntialiasingMaskValue : 0
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
                    renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
                    renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
                    renderPass.draw(vertexCount: UInt32(vertices.count))
                } else {
                    recordShapeRenderFailure(.strokeVertexCapacityExceeded)
                }
            }
        }
    }

    // MARK: - Gradient Layer Rendering

    private func recordGradientRenderFailure(
        _ failure: CAGradientRenderFailure
    ) {
        gradientRenderFailureCount += 1
        lastGradientRenderFailure = failure
    }

    private func gradientStopBinding(
        for gradientLayer: CAGradientLayer,
        configuration: GradientRenderConfiguration,
        device: GPUDevice
    ) throws(CAGradientRenderFailure) -> (
        stopOffset: UInt32,
        bindGroup: GPUBindGroup
    ) {
        guard let uniformBuffer,
              let bindGroupLayout,
              let gradientStopBufferPool else {
            throw .rendererResourcesUnavailable
        }

        let identifier = ObjectIdentifier(gradientLayer)
        if let stopOffset = gradientStopOffsets[identifier],
           let gradientBindGroup {
            return (stopOffset, gradientBindGroup)
        }

        let (byteCount, byteCountOverflow) = UInt64(configuration.colors.count)
            .multipliedReportingOverflow(by: Self.gradientStopStride)
        let (requiredCapacity, capacityOverflow) = gradientStopByteOffset
            .addingReportingOverflow(byteCount)
        guard !byteCountOverflow else {
            throw .stopByteCountOverflow(colorCount: configuration.colors.count)
        }
        guard !capacityOverflow else {
            throw .stopCapacityOverflow(
                byteOffset: gradientStopByteOffset,
                byteCount: byteCount
            )
        }

        do {
            if try gradientStopBufferPool.ensureCapacity(requiredCapacity) {
                // Draws already encoded in this pass retain the superseded buffer
                // through their bind groups. New draws restart at offset zero in
                // the replacement buffer.
                gradientStopByteOffset = 0
                gradientStopOffsets.removeAll(keepingCapacity: true)
                gradientBindGroup = nil
            }
        } catch {
            throw .stopBufferFailure(error)
        }

        let stopOffsetValue = gradientStopByteOffset / Self.gradientStopStride
        guard stopOffsetValue <= UInt64(UInt32.max) else {
            throw .stopOffsetOutOfRange(stopOffsetValue)
        }
        let stopOffset = UInt32(stopOffsetValue)

        // The storage-buffer ABI requires interleaved color/location records.
        // This is the single materialization at the external GPU upload boundary;
        // the JavaScript Float32Array itself is pooled and reused.
        var stopData: [Float] = []
        stopData.reserveCapacity(Int(byteCount / UInt64(MemoryLayout<Float>.stride)))
        for (components, location) in zip(
            configuration.colorComponents,
            configuration.locations
        ) {
            stopData.append(contentsOf: [
                components.x, components.y, components.z, components.w,
                location, 0, 0, 0,
            ])
        }
        let data = createFloat32Array(from: &stopData)
        device.queue.writeBuffer(
            gradientStopBufferPool.currentBuffer,
            bufferOffset: gradientStopByteOffset,
            data: data
        )
        gradientStopByteOffset += byteCount
        gradientStopOffsets[identifier] = stopOffset

        let activeBindGroup: GPUBindGroup
        if let gradientBindGroup {
            activeBindGroup = gradientBindGroup
        } else {
            activeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                layout: bindGroupLayout,
                entries: [
                    GPUBindGroupEntry(
                        binding: 0,
                        resource: .bufferBinding(GPUBufferBinding(
                            buffer: uniformBuffer,
                            size: UInt64(MemoryLayout<CARendererUniforms>.stride)
                        ))
                    ),
                    GPUBindGroupEntry(
                        binding: 1,
                        resource: .bufferBinding(GPUBufferBinding(
                            buffer: gradientStopBufferPool.currentBuffer,
                            size: gradientStopBufferPool.currentCapacity
                        ))
                    ),
                ]
            ))
            gradientBindGroup = activeBindGroup
        }
        return (stopOffset, activeBindGroup)
    }

    /// Renders a CAGradientLayer with its gradient colors.
    private func renderGradientLayer(
        _ gradientLayer: CAGradientLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let colors = gradientLayer.colors, !colors.isEmpty else { return }
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else {
            recordGradientRenderFailure(.rendererResourcesUnavailable)
            return
        }

        let configuration: GradientRenderConfiguration
        do {
            configuration = try GradientRenderConfiguration(
                type: gradientLayer.type,
                colors: colors,
                locations: gradientLayer.locations,
                startPoint: gradientLayer.startPoint,
                endPoint: gradientLayer.endPoint
            )
        } catch {
            recordGradientRenderFailure(.invalidConfiguration(error))
            return
        }

        let stopBinding: (stopOffset: UInt32, bindGroup: GPUBindGroup)
        do {
            stopBinding = try gradientStopBinding(
                for: gradientLayer,
                configuration: configuration,
                device: device
            )
        } catch {
            recordGradientRenderFailure(error)
            return
        }

        // Other foreground renderers select textured or additive pipelines.
        // Gradients always return to the solid pipeline before binding uniforms.
        guard let selectedPipeline = stencilAwarePipeline() else {
            recordMaskRenderFailure(.pipelineUnavailable(.activeStencil))
            return
        }
        renderPass.setPipeline(selectedPipeline)

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
            renderMode: configuration.renderMode,
            gradientStartPoint: SIMD2<Float>(Float(gradientLayer.startPoint.x), Float(gradientLayer.startPoint.y)),
            gradientEndPoint: SIMD2<Float>(Float(gradientLayer.endPoint.x), Float(gradientLayer.endPoint.y)),
            gradientColorCount: Float(configuration.colors.count),
            gradientColorMultiplier: currentReplicatorColor,
            gradientStopOffset: Float(stopBinding.stopOffset),
            edgeAntialiasingMask: gradientLayer.edgeAntialiasingMaskValue,
            cornerCurveExponent: gradientLayer.cornerCurveRenderExponent ?? Float(CornerCurveRenderConfiguration.circularExponent),
            cornerRadii: gradientLayer.cornerRadiiComponents
        )

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

        guard let allocation = allocateVertices(count: vertices.count) else {
            recordGradientRenderFailure(.vertexCapacityExceeded)
            return
        }
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
        renderPass.setBindGroup(
            0,
            bindGroup: stopBinding.bindGroup,
            dynamicOffsets: [UInt32(uniformOffset)]
        )
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
    private func recordShadowRenderFailure(
        _ failure: CAShadowRenderFailure,
        for renderKey: LayerRenderKey
    ) {
        lastShadowRenderFailure = failure
        prerenderedShadows.removeValue(forKey: renderKey)
        failedShadowRenderKeys.insert(renderKey)
        if reportedShadowRenderFailureKeys.insert(renderKey).inserted {
            shadowRenderFailureCount += 1
        }
    }

    private func recordShadowDisplayFailure(
        _ failure: CAShadowRenderFailure,
        for renderKey: LayerRenderKey
    ) {
        lastShadowRenderFailure = failure
        if failedShadowDisplayKeys.insert(renderKey).inserted {
            shadowRenderFailureCount += 1
        }
    }

    private func prerenderShadows(
        _ rootLayer: CALayer,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        depthTextureView: GPUTextureView,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        failedShadowRenderKeys.removeAll(keepingCapacity: true)
        var targets: [LayerPrepassTarget] = []
        collectShadowLayers(rootLayer, parentMatrix: projectionMatrix, into: &targets)

        guard !targets.isEmpty else {
            reportedShadowRenderFailureKeys.removeAll(keepingCapacity: true)
            failedShadowDisplayKeys.removeAll(keepingCapacity: true)
            evictShadowResources(except: [])
            return
        }

        let targetRenderKeys = Set(targets.lazy.map(\.renderKey))
        reportedShadowRenderFailureKeys.formIntersection(targetRenderKeys)
        failedShadowDisplayKeys.formIntersection(targetRenderKeys)

        guard let bindGroupLayout = bindGroupLayout,
              let dummyGradientStopBuffer = dummyGradientStopBuffer,
              let shadowMaskPipeline = shadowMaskPipeline,
              let shadowBlurHorizontalPipeline = shadowBlurHorizontalPipeline,
              let shadowBlurVerticalPipeline = shadowBlurVerticalPipeline,
              let shadowBindGroupLayout = shadowBindGroupLayout,
              let blurSampler = blurSampler else {
            for target in targets {
                recordShadowRenderFailure(
                    .rendererResourcesUnavailable,
                    for: target.renderKey
                )
            }
            return
        }

        var activeRenderKeys: Set<LayerRenderKey> = []
        activeRenderKeys.reserveCapacity(targets.count)

        for target in targets {
            let shadowLayer = target.layer
            let shadowRenderKey = target.renderKey
            let presentationLayer = target.presentationLayer
            let shadowConfiguration: CAShadowRenderConfiguration
            do {
                shadowConfiguration = try CAShadowRenderConfiguration(layer: presentationLayer)
            } catch {
                recordShadowRenderFailure(error, for: shadowRenderKey)
                continue
            }
            activeRenderKeys.insert(shadowRenderKey)
            guard presentationLayer.cornerRadius <= 0
                    || presentationLayer.cornerCurveRenderExponent != nil else {
                // The main traversal records the diagnostic exactly once and
                // rejects the layer. Do not populate a cache with a circular
                // fallback before that rejection occurs.
                continue
            }

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
            let cornerCurveExponent = presentationLayer.cornerCurveRenderExponent
                ?? Float(CornerCurveRenderConfiguration.circularExponent)
            let blurRadius = shadowConfiguration.radius * 0.5
            let captureState = ShadowCaptureState(
                matrix: finalMatrix,
                layerSize: layerSize,
                cornerRadius: cornerRadius,
                cornerRadii: cornerRadii,
                cornerCurveExponent: cornerCurveExponent,
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
                    cornerCurveExponent: cornerCurveExponent,
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
                        ),
                        GPUBindGroupEntry(
                            binding: 1,
                            resource: .buffer(
                                dummyGradientStopBuffer,
                                offset: 0,
                                size: Self.gradientStopStride
                            )
                        )
                    ]
                ))

                let whiteColor = SIMD4<Float>(1, 1, 1, 1)
                var maskVertices: [CARendererVertex]
                do {
                    maskVertices = try ShapeFillTessellator.triangles(
                        for: shadowPath,
                        rule: .nonZero
                    ).map { point in
                        CARendererVertex(
                            position: SIMD2(Float(point.x), Float(point.y)),
                            texCoord: .zero,
                            color: whiteColor
                        )
                    }
                } catch {
                    recordShadowRenderFailure(
                        .shadowPathTessellationFailed,
                        for: shadowRenderKey
                    )
                    continue
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
                do {
                    try withPrepassContext(target) {
                        try captureShadowContent(
                            shadowLayer,
                            parentMatrix: target.parentMatrix,
                            resources: resources,
                            pipeline: pipeline,
                            depthTextureView: depthTextureView,
                            encoder: encoder
                        )
                    }
                } catch let error as CAShadowRenderFailure {
                    recordShadowRenderFailure(error, for: shadowRenderKey)
                    continue
                } catch {
                    recordShadowRenderFailure(
                        .rendererResourcesUnavailable,
                        for: shadowRenderKey
                    )
                    continue
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
            failedShadowRenderKeys.remove(shadowRenderKey)
            reportedShadowRenderFailureKeys.remove(shadowRenderKey)
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
    ) throws(CAShadowRenderFailure) {
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

        let replicatorFailureGenerationBeforeCapture = replicatorRenderFailureGeneration
        renderLayer(layer, renderPass: contentRenderPass, parentMatrix: parentMatrix)

        shadowCaptureRootLayer = previousCaptureRoot
        renderTargetSizeOverride = previousRenderTargetSize
        clipRectStack = previousClipStack
        opacityStack = previousOpacityStack
        maskNestingDepth = previousMaskNestingDepth
        currentStencilValue = previousStencilValue
        contentRenderPass.end()
        if replicatorRenderFailureGeneration != replicatorFailureGenerationBeforeCapture,
           let failure = lastReplicatorRenderFailure {
            throw .subtreeReplicatorFailed(failure)
        }
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
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        depthTextureView: GPUTextureView,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
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
            let stages: [LayerFilterStage]
            do {
                stages = try layerFilterStages(from: requestedFilters)
            } catch {
                failedRenderKeys.insert(filteredRenderKey)
                recordLayerFilterFailure(error, for: filteredRenderKey)
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
            let replicatorFailureGenerationBeforeCapture = replicatorRenderFailureGeneration
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

            if replicatorRenderFailureGeneration != replicatorFailureGenerationBeforeCapture,
               let failure = lastReplicatorRenderFailure {
                failedRenderKeys.insert(filteredRenderKey)
                recordLayerFilterFailure(
                    .subtreeReplicatorFailed(failure),
                    for: filteredRenderKey
                )
                continue
            }

            var currentTexture = resources.sourceTexture
            var currentView = resources.sourceView
            var currentAlphaMode = FilterTextureAlphaMode.premultiplied
            var conversionIndex = 0
            var executionFailure: CALayerFilterRenderFailure?

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
                        executionFailure = .alphaConversionFailed
                        break
                    }
                    let outputTexture = currentTexture === resources.resultTexture
                        ? resources.sourceTexture
                        : resources.resultTexture
                    guard let outputView = resources.view(for: outputTexture) else {
                        executionFailure = .rendererResourcesUnavailable
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
                        executionFailure = .rendererOperationFailed
                        break
                    }
                    currentTexture = outputTexture
                    currentView = outputView

                case let .coreImage(filter):
                    guard convert(to: .straight) else {
                        executionFailure = .alphaConversionFailed
                        break
                    }
                    guard let processor = layerFilterProcessor else {
                        executionFailure = .coreImageProcessorUnavailable
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
                        executionFailure = .coreImageExecutionFailed
                    }
                }

                if executionFailure != nil {
                    break
                }
            }

            if let executionFailure {
                failedRenderKeys.insert(filteredRenderKey)
                recordLayerFilterFailure(executionFailure, for: filteredRenderKey)
                continue
            }

            guard convert(to: .premultiplied) else {
                failedRenderKeys.insert(filteredRenderKey)
                recordLayerFilterFailure(.alphaConversionFailed, for: filteredRenderKey)
                continue
            }

            if requiresContentMask {
                guard let maskLayer = target.presentationLayer.mask,
                      let compositionMaskApplyPipeline else {
                    failedRenderKeys.insert(filteredRenderKey)
                    recordLayerFilterFailure(.contentMaskUnavailable, for: filteredRenderKey)
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
                    recordLayerFilterFailure(.contentMaskCaptureFailed, for: filteredRenderKey)
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
                    recordLayerFilterFailure(.contentMaskCompositeFailed, for: filteredRenderKey)
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
        failedLayerFilterDisplayKeys.formIntersection(activeRenderKeys)
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

    private func recordLayerFilterFailure(
        _ failure: CALayerFilterRenderFailure,
        for renderKey: LayerRenderKey
    ) {
        lastLayerFilterFailure = failure
        if failedLayerFilterKeys.insert(renderKey).inserted {
            layerFilterFailureCount += 1
        }
    }

    private func layerFilterStages(
        from filters: [Any]
    ) throws(CALayerFilterRenderFailure) -> [LayerFilterStage] {
        var stages: [LayerFilterStage] = []
        stages.reserveCapacity(filters.count)

        for value in filters {
            if let filter = value as? CAFilter {
                let executionPlan: CAFilterExecutionPlan
                do {
                    executionPlan = try filter.executionPlan()
                } catch {
                    throw .invalidConfiguration(error)
                }
                switch executionPlan {
                case let .renderer(operation):
                    stages.append(.renderer(operation))
                case let .coreImage(name, parameters):
                    guard let coreImageFilter = CIFilter(name: name) else {
                        throw .unavailableCoreImageFilter(name)
                    }
                    for (key, parameter) in parameters {
                        coreImageFilter.setValue(parameter, forKey: key)
                    }
                    stages.append(.coreImage(coreImageFilter))
                }
            } else if let filter = value as? CIFilter {
                if filter.isEnabled {
                    stages.append(.coreImage(filter))
                }
            } else {
                throw .unsupportedFilterValue(String(reflecting: type(of: value)))
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
        device: GPUDevice,
        processor: CIWebGPUFilterProcessor?,
        encoder: GPUCommandEncoder
    ) throws(CALayerFilterRenderFailure) -> (texture: GPUTexture, view: GPUTextureView) {
        var currentTexture = inputTexture
        var currentView = inputView
        var currentAlphaMode = inputAlphaMode
        var conversionIndex = 0

        func convert(to targetMode: FilterTextureAlphaMode) throws(CALayerFilterRenderFailure) {
            guard currentAlphaMode != targetMode else { return }
            let outputTexture = currentTexture === resources.resultTexture
                ? resources.sourceTexture
                : resources.resultTexture
            guard let outputView = resources.view(for: outputTexture) else {
                throw .rendererResourcesUnavailable
            }
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
            ) else {
                throw .alphaConversionFailed
            }
            currentTexture = outputTexture
            currentView = outputView
            currentAlphaMode = targetMode
        }

        for (stageIndex, stage) in stages.enumerated() {
            switch stage {
            case let .renderer(operation):
                try convert(to: .premultiplied)
                let outputTexture = currentTexture === resources.resultTexture
                    ? resources.sourceTexture
                    : resources.resultTexture
                guard let outputView = resources.view(for: outputTexture) else {
                    throw .rendererResourcesUnavailable
                }
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
                guard applied else {
                    throw .rendererOperationFailed
                }
                currentTexture = outputTexture
                currentView = outputView

            case let .coreImage(filter):
                try convert(to: .straight)
                guard let processor else {
                    throw .coreImageProcessorUnavailable
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
                    activeCompositionExecutions.append(execution)
                    currentTexture = execution.outputTexture
                    currentView = execution.outputTexture.createView()
                    currentAlphaMode = .straight
                } catch {
                    throw .coreImageExecutionFailed
                }
            }
        }

        try convert(to: .straight)
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
              let dummyGradientStopBuffer,
              let shadowMaskPipeline else { return false }

        let presentation = target.presentationLayer
        let bounds = presentation.bounds
        guard bounds.width > 0, bounds.height > 0 else { return false }
        guard presentation.cornerRadius <= 0
                || presentation.cornerCurveRenderExponent != nil else {
            return false
        }

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
            cornerCurveExponent: presentation.cornerCurveRenderExponent
                ?? Float(CornerCurveRenderConfiguration.circularExponent),
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
                GPUBindGroupEntry(binding: 1, resource: .buffer(
                    dummyGradientStopBuffer,
                    offset: 0,
                    size: Self.gradientStopStride
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

    private func contentMaskCornerCurveFailure(
        in layer: CALayer
    ) -> (error: CornerCurveRenderConfigurationError, rawValue: String)? {
        var visited: Set<ObjectIdentifier> = []
        func visit(
            _ candidate: CALayer
        ) -> (error: CornerCurveRenderConfigurationError, rawValue: String)? {
            guard visited.insert(ObjectIdentifier(candidate)).inserted else {
                return nil
            }
            let presentation = renderPresentation(for: candidate)
            if presentation.cornerRadius > 0,
               let error = presentation.cornerCurveRenderError {
                return (error, presentation.cornerCurve.rawValue)
            }
            if let nestedMask = presentation.mask,
               let failure = visit(nestedMask) {
                return failure
            }
            for sublayer in candidate.sublayers ?? [] {
                if let failure = visit(sublayer) {
                    return failure
                }
            }
            return nil
        }
        return visit(layer)
    }

    private func renderRawCompositionContentMask(
        _ target: LayerPrepassTarget,
        outputView: GPUTextureView,
        suppressRootFilters: Bool,
        renderSize: CGSize? = nil,
        depthStencilView: GPUTextureView? = nil,
        encoder: GPUCommandEncoder
    ) -> Bool {
        if let failure = contentMaskCornerCurveFailure(in: target.layer) {
            recordCornerCurveRenderFailure(.mask(failure.error))
            recordMaskRenderFailure(
                .unsupportedCornerCurve(.contentMask, failure.rawValue)
            )
            return false
        }
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

        let replicatorFailureGenerationBeforeCapture = replicatorRenderFailureGeneration
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
        if replicatorRenderFailureGeneration != replicatorFailureGenerationBeforeCapture {
            return false
        }
        return true
    }

    private func renderCompositionContentMask(
        _ target: LayerPrepassTarget,
        outputView: GPUTextureView,
        resources: CompositionLayerResources,
        contentMaskIndex: Int,
        encoder: GPUCommandEncoder
    ) throws(CACompositionFilterRenderFailure) {
        guard let device else {
            throw .contentMaskFilterExecutionFailed(.rendererResourcesUnavailable)
        }
        let stages: [LayerFilterStage]
        do {
            stages = try layerFilterStages(from: target.presentationLayer.filters ?? [])
        } catch {
            throw .contentMaskFilterPlanningFailed(error)
        }
        guard !stages.isEmpty else {
            guard renderRawCompositionContentMask(
                target,
                outputView: outputView,
                suppressRootFilters: false,
                encoder: encoder
            ) else {
                throw .clipMaskFailed
            }
            return
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
        ) else {
            throw .clipMaskFailed
        }

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
            ) else {
                throw .contentMaskFilterExecutionFailed(.rendererOperationFailed)
            }
            inputTexture = filterResources.resultTexture
            inputView = filterResources.resultView
        }

        let filteredMask: (texture: GPUTexture, view: GPUTextureView)
        do {
            filteredMask = try executeBackdropFilterStages(
                stages,
                inputTexture: inputTexture,
                inputView: inputView,
                inputAlphaMode: .premultiplied,
                resources: filterResources,
                device: device,
                processor: layerFilterProcessor,
                encoder: encoder
            )
        } catch {
            throw .contentMaskFilterExecutionFailed(error)
        }
        let outputUniformBuffer = filterResources.uniformBuffer(
            forOperationAt: stages.count + 17,
            device: device
        )
        guard applyAlphaConversion(
            from: .straight,
            inputTexture: filteredMask.texture,
            outputView: outputView,
            uniformBuffer: outputUniformBuffer,
            encoder: encoder
        ) else {
            throw .contentMaskFilterExecutionFailed(.alphaConversionFailed)
        }
    }

    private func buildCompositionClipMask(
        clipTargets: [LayerPrepassTarget],
        contentMaskTargets: [LayerPrepassTarget],
        resources: CompositionLayerResources,
        encoder: GPUCommandEncoder
    ) throws(CACompositionFilterRenderFailure) -> GPUTextureView? {
        let maskTargets = clipTargets.map(CompositionMaskTarget.clipShape)
            + contentMaskTargets.map(CompositionMaskTarget.layerContent)
        guard !maskTargets.isEmpty else { return nil }
        guard let device,
              let compositionMaskIntersectPipeline else {
            throw .clipMaskFailed
        }

        var currentView = resources.clipCumulativeViewA
        for (index, maskTarget) in maskTargets.enumerated() {
            let outputView = index == 0
                ? resources.clipCumulativeViewA
                : resources.clipShapeView
            switch maskTarget {
            case let .clipShape(clipTarget):
                guard renderCompositionClipShape(
                    clipTarget,
                    outputView: outputView,
                    uniformBuffer: resources.clipUniformBuffer(at: index, device: device),
                    vertexBuffer: resources.backdropMaskVertexBuffer,
                    encoder: encoder
                ) else {
                    throw .clipMaskFailed
                }
            case let .layerContent(contentTarget):
                try renderCompositionContentMask(
                    contentTarget,
                    outputView: outputView,
                    resources: resources,
                    contentMaskIndex: index,
                    encoder: encoder
                )
            }

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
            ) else {
                throw .clipMaskFailed
            }
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
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        depthTextureView: GPUTextureView,
        processor: CIWebGPUFilterProcessor,
        clearColor: GPUColor,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
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
            let recordFailure: (
                LayerRenderKey,
                CACompositionFilterRenderFailure
            ) -> Void = { key, failure in
                failedKeys.insert(key)
                self.lastCompositionFilterFailure = failure
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
                        device: device,
                        pipeline: pipeline,
                        depthTextureView: depthTextureView,
                        encoder: encoder,
                        projectionMatrix: projectionMatrix
                    )
                }
                previousDepth = compositionTarget.depth
                let target = compositionTarget.prepass
                let key = target.renderKey
                let presentation = target.presentationLayer
                let requestedBackgroundFilters = presentation.backgroundFilters ?? []
                let backgroundStages: [LayerFilterStage]
                do {
                    backgroundStages = try layerFilterStages(from: requestedBackgroundFilters)
                } catch {
                    recordFailure(key, .backgroundFilterPlanningFailed(error))
                    continue
                }
                let compositionFilter: CIFilter
                if let requestedFilter = presentation.compositingFilter as? CIFilter {
                    compositionFilter = requestedFilter
                } else if let requestedFilter = presentation.compositingFilter {
                    recordFailure(
                        key,
                        .unsupportedCompositingFilterValue(
                            String(reflecting: type(of: requestedFilter))
                        )
                    )
                    continue
                } else if let sourceOver = CIFilter(name: "CISourceOverCompositing") {
                    compositionFilter = sourceOver
                } else {
                    recordFailure(key, .defaultCompositingFilterUnavailable)
                    continue
                }
                guard prefixIsValid else {
                    recordFailure(key, .invalidBackdropPrefix)
                    continue
                }
                guard processor.supports(
                    compositionFilter,
                    inputMode: .foregroundAndBackground
                ) else {
                    recordFailure(key, .unsupportedCompositingFilter(compositionFilter.name))
                    continue
                }
                guard let source = prerenderedFilters[key] else {
                    recordFailure(key, .sourceCaptureUnavailable)
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
                let replicatorFailureGenerationBeforeCapture =
                    replicatorRenderFailureGeneration

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

                if replicatorRenderFailureGeneration
                    != replicatorFailureGenerationBeforeCapture,
                   let failure = lastReplicatorRenderFailure {
                    recordFailure(key, .backdropReplicatorFailed(failure))
                    continue
                }

                let clipMaskView: GPUTextureView?
                do {
                    clipMaskView = try buildCompositionClipMask(
                        clipTargets: compositionTarget.clipAncestors,
                        contentMaskTargets: compositionTarget.contentMaskAncestors,
                        resources: resources,
                        encoder: encoder
                    )
                } catch {
                    recordFailure(key, error)
                    continue
                }
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
                        recordFailure(key, .clipMaskFailed)
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
                        recordFailure(key, .sourceAdjustmentFailed)
                        continue
                    }
                    premultipliedSourceTexture = resources.sourcePremultipliedTexture
                }

                guard didReachTarget else {
                    recordFailure(key, .backdropCaptureIncomplete)
                    continue
                }
                guard applyAlphaConversion(
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
                    recordFailure(key, .alphaConversionFailed)
                    continue
                }

                var compositionBackdropTexture = resources.backdropStraightTexture
                if !backgroundStages.isEmpty {
                    let filteredBackdrop: (texture: GPUTexture, view: GPUTextureView)
                    do {
                        filteredBackdrop = try executeBackdropFilterStages(
                            backgroundStages,
                            inputTexture: resources.backdropStraightTexture,
                            inputView: resources.backdropStraightView,
                            resources: resources.backgroundFilterResources,
                            device: device,
                            processor: processor,
                            encoder: encoder
                        )
                    } catch {
                        recordFailure(key, .backgroundFilterExecutionFailed(error))
                        continue
                    }
                    guard renderBackdropFilterMask(
                        for: compositionTarget,
                        resources: resources,
                        encoder: encoder
                    ) else {
                        recordFailure(key, .backgroundFilterMaskFailed)
                        continue
                    }
                    var backgroundMaskView = resources.backdropMaskView
                    let backgroundClipMaskView: GPUTextureView?
                    if let targetContentMask = compositionTarget.targetContentMask {
                        do {
                            backgroundClipMaskView = try buildCompositionClipMask(
                                clipTargets: compositionTarget.clipAncestors,
                                contentMaskTargets: compositionTarget.contentMaskAncestors
                                    + [targetContentMask],
                                resources: resources,
                                encoder: encoder
                            )
                        } catch {
                            recordFailure(key, error)
                            continue
                        }
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
                            recordFailure(key, .backgroundFilterMaskFailed)
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
                        recordFailure(key, .backgroundFilterMixFailed)
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
                        recordFailure(key, .alphaConversionFailed)
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
                    recordFailure(key, .compositionExecutionFailed)
                }
            }

            if previousDepth != nil {
                // Refresh every context so resources referenced by commands
                // encoded for sibling mask roots remain alive until submission.
                // The refreshed main tree also picks up depth-zero compositions
                // that became available inside detached content masks.
                prerenderFilteredLayers(
                    rootLayer,
                    device: device,
                    pipeline: pipeline,
                    depthTextureView: depthTextureView,
                    encoder: encoder,
                    projectionMatrix: projectionMatrix
                )
            }
        }

        failedCompositionKeys.formIntersection(failedKeys)
        failedCompositionDisplayKeys.formIntersection(activeKeys)
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
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        cache: RasterizationCache<GPUTexture>,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
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
            || (presentationLayer as? CAReplicatorLayer)?.preservesDepth == true
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
    private func recordRasterizationFailure(
        _ failure: CARasterizationRenderFailure
    ) {
        rasterizationFailureCount += 1
        lastRasterizationRenderFailure = failure
    }

    private func recordRasterizationCaptureFailure(
        _ failure: CARasterizationRenderFailure,
        for renderKey: LayerRenderKey
    ) {
        failedRasterizationRenderKeys.insert(renderKey)
        recordRasterizationFailure(failure)
    }

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
            recordRasterizationCaptureFailure(
                .invalidCaptureBounds,
                for: layerRenderKey
            )
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
            failedRasterizationRenderKeys.remove(layerRenderKey)
            prerasterizedTextures[layerRenderKey] = PrerasterizedTexture(
                texture: entry.texture,
                purpose: purpose,
                captureBounds: captureBounds
            )
            return
        }

        // Miss — allocate a visible-subtree texture (`captureBounds × scale`)
        // and render the subtree into it under a bounds-local projection.
        let captureConfiguration: CARasterizationCaptureConfiguration
        do {
            captureConfiguration = try CARasterizationCaptureConfiguration(
                captureBounds: captureBounds,
                rasterizationScale: presentationLayer.rasterizationScale,
                maximumTextureDimension: Int(device.limits.maxTextureDimension2D)
            )
        } catch let failure {
            recordRasterizationCaptureFailure(failure, for: layerRenderKey)
            return
        }
        let pixelWidth = captureConfiguration.pixelWidth
        let pixelHeight = captureConfiguration.pixelHeight

        let requestedFilters = presentationLayer.filters ?? []
        let filterStages: [LayerFilterStage]
        if requestedFilters.isEmpty {
            filterStages = []
        } else {
            do {
                filterStages = try layerFilterStages(from: requestedFilters)
            } catch {
                recordLayerFilterFailure(error, for: layerRenderKey)
                failedRasterizationRenderKeys.insert(layerRenderKey)
                return
            }
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
            left: captureConfiguration.projectionLeft,
            right: captureConfiguration.projectionRight,
            bottom: captureConfiguration.projectionBottom,
            top: captureConfiguration.projectionTop,
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
        let replicatorFailureGenerationBeforeCapture = replicatorRenderFailureGeneration
        renderLayer(layer, renderPass: capturePass, parentMatrix: captureProjection)

        rasterizePrerenderRootLayer = previousCaptureRoot
        renderTargetSizeOverride = previousRenderTargetSize
        clipRectStack = previousClipStack
        opacityStack = previousOpacityStack
        maskNestingDepth = previousMaskNestingDepth
        transformDepthNesting = previousTransformDepthNesting
        currentStencilValue = previousStencilValue

        capturePass.end()

        if replicatorRenderFailureGeneration != replicatorFailureGenerationBeforeCapture,
           let failure = lastReplicatorRenderFailure {
            transientRasterizationTextures.append(captureTexture)
            recordRasterizationCaptureFailure(
                .subtreeReplicatorFailed(failure),
                for: layerRenderKey
            )
            return
        }

        var cachedTexture: GPUTexture
        if filterStages.isEmpty {
            cachedTexture = captureTexture
        } else {
            transientRasterizationTextures.append(captureTexture)
            let filteredTexture: GPUTexture
            do {
                filteredTexture = try executeRasterizedFilterStages(
                    filterStages,
                    inputTexture: captureTexture,
                    width: pixelWidth,
                    height: pixelHeight,
                    device: device,
                    processor: layerFilterProcessor,
                    encoder: encoder
                )
            } catch {
                recordLayerFilterFailure(error, for: layerRenderKey)
                failedRasterizationRenderKeys.insert(layerRenderKey)
                return
            }
            failedLayerFilterKeys.remove(layerRenderKey)
            cachedTexture = filteredTexture
        }

        if let maskLayer = presentationLayer.mask {
            let maskedTexture: GPUTexture
            do {
                maskedTexture = try executeRasterizedContentMask(
                    maskLayer,
                    contentTexture: cachedTexture,
                    projectionMatrix: captureProjection,
                    renderSize: captureSize,
                    depthStencilView: captureDepthView,
                    width: pixelWidth,
                    height: pixelHeight,
                    device: device,
                    processor: layerFilterProcessor,
                    encoder: encoder
                )
            } catch {
                recordLayerFilterFailure(error, for: layerRenderKey)
                failedRasterizationRenderKeys.insert(layerRenderKey)
                return
            }
            failedLayerFilterKeys.remove(layerRenderKey)
            cachedTexture = maskedTexture
        }

        if hasVisibleShadow(presentationLayer) {
            transientRasterizationTextures.append(cachedTexture)
            do {
                cachedTexture = try executeRasterizedShadow(
                    layer: presentationLayer,
                    contentTexture: cachedTexture,
                    captureBounds: captureBounds,
                    width: pixelWidth,
                    height: pixelHeight,
                    encoder: encoder
                )
            } catch {
                recordShadowRenderFailure(error, for: layerRenderKey)
                failedRasterizationRenderKeys.insert(layerRenderKey)
                return
            }
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
        failedRasterizationRenderKeys.remove(layerRenderKey)
        if purpose == .transformFlattening {
            transformFlatteningCaptureCount += 1
        } else if purpose == .explicit {
            explicitRasterizationCaptureCount += 1
            explicitRasterizationCapturePixelSizes.append(
                CGSize(width: pixelWidth, height: pixelHeight)
            )
        }
    }

    private func canReuseRasterizedTexture(
        layer: CALayer,
        entry: RasterizedEntry<GPUTexture>,
        purpose: RasterizationCachePurpose,
        contentBoundsHash: Int
    ) -> Bool {
        // Particle simulation advances independently of CALayer dirty flags.
        // Reusing a cached ancestor texture would freeze live particles even
        // though the layer tree itself has not been mutated.
        if subtreeContainsEmitter(layer) {
            return false
        }
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

    private func subtreeContainsEmitter(_ layer: CALayer) -> Bool {
        var pending: [CALayer] = [layer]
        var visited: Set<ObjectIdentifier> = []
        while let candidate = pending.popLast() {
            guard visited.insert(ObjectIdentifier(candidate)).inserted else {
                continue
            }
            if candidate is CAEmitterLayer {
                return true
            }
            if let mask = candidate.mask {
                pending.append(mask)
            }
            pending.append(contentsOf: candidate.sublayers ?? [])
        }
        return false
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
        device: GPUDevice,
        processor: CIWebGPUFilterProcessor?,
        encoder: GPUCommandEncoder
    ) throws(CALayerFilterRenderFailure) -> GPUTexture {
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

        func convert(to targetMode: FilterTextureAlphaMode) throws(CALayerFilterRenderFailure) {
            guard currentAlphaMode != targetMode else { return }
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
            ) else {
                throw .alphaConversionFailed
            }
            currentTexture = outputTexture
            currentAlphaMode = targetMode
        }

        for (stageIndex, stage) in stages.enumerated() {
            let uniformBuffer = resources.uniformBuffer(
                forOperationAt: stageIndex,
                device: device
            )

            switch stage {
            case let .renderer(operation):
                try convert(to: .premultiplied)
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
                guard applied else {
                    throw .rendererOperationFailed
                }
                currentTexture = outputTexture

            case let .coreImage(filter):
                try convert(to: .straight)
                guard let processor else {
                    throw .coreImageProcessorUnavailable
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
                    throw .coreImageExecutionFailed
                }
            }
        }

        try convert(to: .premultiplied)

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
            throw .rendererOperationFailed
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
        device: GPUDevice,
        processor: CIWebGPUFilterProcessor?,
        encoder: GPUCommandEncoder
    ) throws(CALayerFilterRenderFailure) -> GPUTexture {
        guard let compositionMaskApplyPipeline else {
            throw .contentMaskUnavailable
        }
        let maskStages = try layerFilterStages(
            from: renderPresentation(for: maskLayer).filters ?? []
        )

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
        ) else {
            throw .contentMaskCaptureFailed
        }

        let maskTexture: GPUTexture
        if !executesRootMaskStages {
            maskTexture = resources.sourceTexture
        } else {
            let filteredMask = try executeRasterizedFilterStages(
                maskStages,
                inputTexture: resources.sourceTexture,
                width: width,
                height: height,
                device: device,
                processor: processor,
                encoder: encoder
            )
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
            throw .contentMaskCompositeFailed
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
    ) throws(CAShadowRenderFailure) -> GPUTexture {
        guard let device,
              let bindGroupLayout,
              let dummyGradientStopBuffer,
              let shadowMaskPipeline,
              let shadowBlurHorizontalPipeline,
              let shadowBlurVerticalPipeline,
              let shadowBindGroupLayout,
              let rasterizedShadowCompositePipeline,
              let blurSampler else {
            throw .rasterizedShadowResourcesUnavailable
        }
        let configuration = try CAShadowRenderConfiguration(layer: layer)

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
                    ),
                    GPUBindGroupEntry(
                        binding: 1,
                        resource: .buffer(
                            dummyGradientStopBuffer,
                            offset: 0,
                            size: Self.gradientStopStride
                        )
                    )
                ]
            ))
            let white = SIMD4<Float>(repeating: 1)
            var vertices: [CARendererVertex]
            do {
                vertices = try ShapeFillTessellator.triangles(
                    for: shadowPath,
                    rule: .nonZero
                ).map { point in
                    CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: .zero,
                        color: white
                    )
                }
            } catch {
                throw .shadowPathTessellationFailed
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
            blurRadius: configuration.radius * 0.5
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

        let color = SIMD4<Float>(
            configuration.color.x,
            configuration.color.y,
            configuration.color.z,
            configuration.color.w * configuration.opacity
        )
        var compositeUniforms = RasterShadowCompositeUniforms(
            shadowColor: color,
            shadowOffsetUV: SIMD2<Float>(
                Float(configuration.offset.width / captureBounds.width),
                Float(-configuration.offset.height / captureBounds.height)
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
              let uniformBuffer = uniformBuffer,
              let basePipeline = pipeline else {
            recordRasterizationFailure(.compositeResourcesUnavailable)
            return
        }
        guard let selectedPipeline = selectPremultipliedTexturedPipeline() else {
            recordRasterizationFailure(.compositePipelineUnavailable)
            return
        }

        let captureBounds = prerasterized.captureBounds
        let captureMinX = Float(captureBounds.minX)
        let captureMinY = Float(captureBounds.minY)
        let captureWidth = Float(captureBounds.width)
        let captureHeight = Float(captureBounds.height)
        guard captureMinX.isFinite,
              captureMinY.isFinite,
              captureWidth.isFinite,
              captureHeight.isFinite,
              captureWidth > 0,
              captureHeight > 0 else {
            recordRasterizationFailure(.invalidCompositeBounds(captureBounds))
            return
        }

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
            recordRasterizationFailure(.compositeVertexCapacityExceeded)
            return
        }
        let (vertexOffset, layerIndex) = allocation

        // Scale the [0, 1] quad to bounds.size, then apply the layer's
        // own modelMatrix to position/rotate/scale it into the world.
        let originMatrix = Matrix4x4(translation: SIMD3<Float>(
            captureMinX,
            captureMinY,
            0
        ))
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(captureWidth, 0, 0, 0),
            SIMD4<Float>(0, captureHeight, 0, 0),
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
            layerSize: SIMD2<Float>(captureWidth, captureHeight),
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
        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Restore the regular pipeline so subsequent per-layer draws in
        // the main pass continue with the right state.
        renderPass.setPipeline(basePipeline)
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
        let shadowRenderKey = renderKey(for: modelLayer)
        guard !failedShadowRenderKeys.contains(shadowRenderKey) else { return }
        let configuration: CAShadowRenderConfiguration
        do {
            configuration = try CAShadowRenderConfiguration(layer: layer)
        } catch {
            recordShadowDisplayFailure(error, for: shadowRenderKey)
            return
        }
        let compositeConfiguration: CAShadowCompositeConfiguration
        do {
            compositeConfiguration = try CAShadowCompositeConfiguration(
                shadow: configuration,
                effectiveOpacity: currentEffectiveOpacity,
                replicatorColor: currentReplicatorColor,
                viewportSize: size
            )
        } catch {
            recordShadowDisplayFailure(error, for: shadowRenderKey)
            return
        }
        guard let prerendered = prerenderedShadows[shadowRenderKey] else {
            recordShadowDisplayFailure(.prerenderedShadowUnavailable, for: shadowRenderKey)
            return
        }
        guard let vertexBuffer,
              let blurSampler else {
            recordShadowDisplayFailure(.rendererResourcesUnavailable, for: shadowRenderKey)
            return
        }
        let selectedPipeline: GPURenderPipeline
        if maskNestingDepth > 0 {
            guard let shadowCompositeStencilPipeline else {
                recordShadowDisplayFailure(
                    .compositeStencilPipelineUnavailable,
                    for: shadowRenderKey
                )
                return
            }
            selectedPipeline = shadowCompositeStencilPipeline
        } else {
            guard let shadowCompositePipeline else {
                recordShadowDisplayFailure(.rendererResourcesUnavailable, for: shadowRenderKey)
                return
            }
            selectedPipeline = shadowCompositePipeline
        }
        guard let restorationPipeline = pipeline else {
            recordShadowDisplayFailure(
                .compositeRestorationPipelineUnavailable,
                for: shadowRenderKey
            )
            return
        }

        var vertices: [CARendererVertex] = [
            CARendererVertex(
                position: SIMD2(0, 0),
                texCoord: SIMD2(0, 1),
                color: compositeConfiguration.color
            ),
            CARendererVertex(
                position: SIMD2(compositeConfiguration.viewportSize.x, 0),
                texCoord: SIMD2(1, 1),
                color: compositeConfiguration.color
            ),
            CARendererVertex(
                position: SIMD2(0, compositeConfiguration.viewportSize.y),
                texCoord: SIMD2(0, 0),
                color: compositeConfiguration.color
            ),
            CARendererVertex(
                position: SIMD2(compositeConfiguration.viewportSize.x, 0),
                texCoord: SIMD2(1, 1),
                color: compositeConfiguration.color
            ),
            CARendererVertex(
                position: compositeConfiguration.viewportSize,
                texCoord: SIMD2(1, 0),
                color: compositeConfiguration.color
            ),
            CARendererVertex(
                position: SIMD2(0, compositeConfiguration.viewportSize.y),
                texCoord: SIMD2(0, 0),
                color: compositeConfiguration.color
            ),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else {
            recordShadowDisplayFailure(.vertexCapacityExceeded, for: shadowRenderKey)
            return
        }
        let vertexOffset = allocation.vertexOffset

        var shadowUniforms = ShadowUniforms(
            mvpMatrix: Matrix4x4.orthographic(
                left: 0,
                right: compositeConfiguration.viewportSize.x,
                bottom: 0,
                top: compositeConfiguration.viewportSize.y,
                near: -1000,
                far: 1000
            ),
            shadowColor: compositeConfiguration.color,
            shadowOffset: compositeConfiguration.offset,
            layerSize: compositeConfiguration.viewportSize
        )
        let compositeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: selectedPipeline.getBindGroupLayout(index: 0),
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
        device.queue.writeBuffer(
            prerendered.resources.compositeUniformBuffer,
            bufferOffset: 0,
            data: createFloat32Array(from: &shadowUniforms)
        )
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: compositeBindGroup)
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
        renderPass.setPipeline(restorationPipeline)
        failedShadowDisplayKeys.remove(shadowRenderKey)
    }

    /// Renders a filtered layer by compositing the pre-rendered filter texture.
    ///
    /// This method draws a full-screen quad textured with the filtered layer content
    /// from the filter pre-rendering pass.
    private func renderFilteredLayerComposite(
        _ prerendered: PrerenderedFilter,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder
    ) throws(CALayerFilterRenderFailure) {
        let configuration = try CALayerFilterCompositeConfiguration(
            opacity: currentEffectiveOpacity,
            colorMultiplier: currentReplicatorColor
        )
        try renderPremultipliedFullScreenTexture(
            prerendered.outputView,
            uniformBuffer: prerendered.resources.compositeUniformBuffer,
            configuration: configuration,
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
    ) throws(CACompositionFilterRenderFailure) {
        if transformDepthNesting > 0 || rasterizePrerenderRootLayer != nil {
            try renderPreparedCompositionPlane(
                composition,
                presentationLayer: presentationLayer,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix
            )
            return
        }
        guard let filterReplacementPipeline,
              let blurSampler else {
            throw .displayResourcesUnavailable
        }

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
    ) throws(CACompositionFilterRenderFailure) {
        let configuration = try CACompositionPlaneRenderConfiguration(
            bounds: presentationLayer.bounds,
            viewportSize: size
        )
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(configuration.size.x, 0, 0, 0),
            SIMD4<Float>(0, configuration.size.y, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix
        try configuration.validateDisplayTransform(columns: finalMatrix.columns)

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
              let vertexBuffer else {
            throw .displayResourcesUnavailable
        }

        let planeVertices: [CACompositionPlaneVertex]
        if rasterizePrerenderRootLayer != nil {
            planeVertices = try configuration.capturedVertices(
                samplingColumns: composition.samplingModelMatrix.columns
            )
        } else {
            planeVertices = configuration.standardVertices()
        }
        var vertices = planeVertices.map {
            CARendererVertex(position: $0.position, texCoord: $0.texCoord, color: $0.color)
        }
        guard let allocation = allocateVertices(count: vertices.count) else {
            throw .displayVertexCapacityExceeded
        }

        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            layerSize: configuration.viewportSize
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
        device.queue.writeBuffer(
            composition.resources.transformedDisplayUniformBuffer,
            bufferOffset: 0,
            data: createFloat32Array(from: &uniforms)
        )
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

    private func renderPremultipliedFullScreenTexture(
        _ textureView: GPUTextureView,
        uniformBuffer: GPUBuffer,
        configuration: CALayerFilterCompositeConfiguration,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder
    ) throws(CALayerFilterRenderFailure) {
        guard let blurSampler else {
            throw .compositeResourcesUnavailable
        }
        let selectedPipeline: GPURenderPipeline
        if maskNestingDepth > 0 {
            guard let filterCompositeStencilPipeline else {
                throw .compositeStencilPipelineUnavailable
            }
            selectedPipeline = filterCompositeStencilPipeline
        } else {
            guard let filterCompositePipeline else {
                throw .compositeResourcesUnavailable
            }
            selectedPipeline = filterCompositePipeline
        }
        guard let restorationPipeline = pipeline else {
            throw .compositeRestorationPipelineUnavailable
        }

        var filterUniforms = FilterCompositeUniforms(
            opacity: configuration.opacity,
            colorMultiplier: configuration.colorMultiplier
        )
        let filterUniformData = createFloat32Array(from: &filterUniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: 0,
            data: filterUniformData
        )

        // Create composite bind group with the filtered texture.
        let compositeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: selectedPipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(uniformBuffer, offset: 0, size: UInt64(MemoryLayout<FilterCompositeUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(textureView)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: compositeBindGroup)
        renderPass.draw(vertexCount: 6)

        // Switch back to regular pipeline
        renderPass.setPipeline(restorationPipeline)
    }

    // MARK: - CATransformLayer Rendering

    private func recordTransformDepthRenderFailure(
        _ failure: CATransformDepthRenderFailure
    ) {
        transformDepthRenderFailureCount += 1
        lastTransformDepthRenderFailure = failure
    }

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
        guard layer.sublayers?.isEmpty == false else { return }

        // Resolve ordering before depth clear so invalid homogeneous geometry
        // cannot partially mutate the active render pass.
        let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)
        let orderedSublayers: [CALayer]
        do {
            orderedSublayers = try depthOrderedSublayers(
                for: layer,
                parentMatrix: sublayerMatrix
            )
        } catch {
            recordTransformDepthRenderFailure(error)
            return
        }

        let configuration: CADepthGroupRenderConfiguration
        do {
            configuration = try CADepthGroupRenderConfiguration(
                currentNestingDepth: transformDepthNesting
            )
        } catch {
            recordTransformDepthRenderFailure(.depthGroupStateFailure(error))
            return
        }
        if configuration.requiresDepthClear {
            guard let depthClearPipeline else {
                recordTransformDepthRenderFailure(.depthClearPipelineUnavailable)
                return
            }
            renderPass.setPipeline(depthClearPipeline)
            renderPass.draw(vertexCount: 3)
        }
        let previousNestingDepth = transformDepthNesting
        transformDepthNesting = configuration.enteredNestingDepth
        defer { transformDepthNesting = previousNestingDepth }

        // `renderLayer` already applied this transform layer's presentation transform.
        // Only the sublayer transform and bounds origin remain before traversing children.
        for sublayer in orderedSublayers {
            self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
        }
    }

    // MARK: - CAEmitterLayer Rendering

    private func recordEmitterSpawnFailure(_ failure: CAEmitterFailure) {
        emitterSpawnFailureCount += 1
        lastEmitterSpawnFailure = failure
    }

    private func recordEmitterRenderFailure(_ failure: CAEmitterFailure) {
        emitterRenderFailureCount += 1
        lastEmitterRenderFailure = failure
    }

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
        let layerID = ObjectIdentifier(modelLayer)
        if let existing = emitterLayerStates[layerID], existing.owner === modelLayer {
            existing.lastRenderedBirthSequences.removeAll(keepingCapacity: true)
            existing.lastRenderUsedAdditiveBlending = false
        }
        let configuration: CAEmitterRenderConfiguration
        do {
            configuration = try CAEmitterRenderConfiguration(layer: emitterLayer)
        } catch {
            if case .unsupportedRenderMode(_) = error {
                recordEmitterRenderFailure(error)
            } else {
                recordEmitterSpawnFailure(error)
            }
            return
        }
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else {
            recordEmitterRenderFailure(.rendererResourcesUnavailable)
            return
        }
        let entersDepthSpace = configuration.preservesDepth
        let depthConfiguration: CADepthGroupRenderConfiguration?
        if entersDepthSpace {
            do {
                depthConfiguration = try CADepthGroupRenderConfiguration(
                    currentNestingDepth: transformDepthNesting
                )
            } catch {
                recordEmitterRenderFailure(.depthGroupStateFailure(error))
                return
            }
        } else {
            depthConfiguration = nil
        }
        if depthConfiguration?.requiresDepthClear == true {
            guard let depthClearPipeline else {
                recordEmitterRenderFailure(.depthResourcesUnavailable)
                return
            }
            renderPass.setPipeline(depthClearPipeline)
            renderPass.draw(vertexCount: 3)
        }
        let previousDepthNesting = transformDepthNesting
        if let depthConfiguration {
            transformDepthNesting = depthConfiguration.enteredNestingDepth
        }
        defer {
            transformDepthNesting = previousDepthNesting
        }
        let emitterCells = configuration.emitterCells

        activeEmitterLayerIDs.insert(layerID)
        let state: EmitterLayerState
        if let existing = emitterLayerStates[layerID], existing.owner === modelLayer {
            state = existing
        } else {
            state = EmitterLayerState(owner: modelLayer, seed: configuration.seed)
            emitterLayerStates[layerID] = state
        }
        if state.configuredSeed != configuration.seed {
            state.randomSource.reset(seed: configuration.seed)
            state.configuredSeed = configuration.seed
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
            let currentTime = CARenderTimeContext.currentMediaTime
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
                if !particleStateIsFinite(state.particles[index]) {
                    state.particles[index].isAlive = false
                    recordEmitterSpawnFailure(.nonFiniteParticleState)
                }
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
                    recordEmitterSpawnFailure(.invalidCellTiming)
                    continue
                }
                let particlesToSpawn: Int
                do {
                    particlesToSpawn = try emitterParticleBirthCount(
                        cell: cell,
                        activeDelta: activeDelta,
                        rateMultiplier: configuration.birthRate,
                        remainderKey: .root(cellID),
                        state: state
                    )
                } catch {
                    recordEmitterSpawnFailure(error)
                    continue
                }

                for _ in 0..<particlesToSpawn {
                    guard projectedLiveParticleCount < Self.maxParticles else {
                        recordEmitterSpawnFailure(
                            .particleCapacityExceeded(maximum: Self.maxParticles)
                        )
                        break
                    }
                    guard let position = EmitterGeometry.position(
                        shape: configuration.emitterShape,
                        mode: configuration.emitterMode,
                        position: configuration.emitterPosition,
                        zPosition: configuration.emitterZPosition,
                        size: configuration.emitterSize,
                        depth: configuration.emitterDepth,
                        random: &state.randomSource
                    ) else {
                        recordEmitterSpawnFailure(.nonFiniteLayerGeometry)
                        continue
                    }
                    var particle: EmitterParticle
                    do {
                        particle = try makeEmitterParticle(
                            cell: cell,
                            position: position,
                            parentDirection: nil,
                            inheritedColor: SIMD4(1, 1, 1, 1),
                            inheritedScale: 1,
                            generation: 0,
                            configuration: configuration,
                            state: state
                        )
                    } catch {
                        recordEmitterSpawnFailure(error)
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
                        recordEmitterSpawnFailure(.invalidCellTiming)
                        continue
                    }
                    let particlesToSpawn: Int
                    do {
                        particlesToSpawn = try emitterParticleBirthCount(
                            cell: childCell,
                            activeDelta: activeDelta,
                            rateMultiplier: configuration.birthRate,
                            remainderKey: .child(
                                parentBirthSequence: parent.birthSequence,
                                cell: ObjectIdentifier(childCell)
                            ),
                            state: state
                        )
                    } catch {
                        recordEmitterSpawnFailure(error)
                        continue
                    }

                    for childIndex in 0..<particlesToSpawn {
                        guard projectedLiveParticleCount < Self.maxParticles else {
                            recordEmitterSpawnFailure(
                                .particleCapacityExceeded(maximum: Self.maxParticles)
                            )
                            break
                        }
                        let fraction = Float(childIndex + 1) / Float(particlesToSpawn + 1)
                        let position = parent.previousPosition
                            + (parent.position - parent.previousPosition) * fraction
                        let inheritedColor = parent.previousColor
                            + (parent.color - parent.previousColor) * fraction
                        let inheritedScale = parent.previousScale
                            + (parent.scale - parent.previousScale) * fraction
                        var child: EmitterParticle
                        do {
                            child = try makeEmitterParticle(
                                cell: childCell,
                                position: position,
                                parentDirection: parent.emissionDirection,
                                inheritedColor: inheritedColor,
                                inheritedScale: inheritedScale,
                                generation: parent.generation + 1,
                                configuration: configuration,
                                state: state
                            )
                        } catch {
                            recordEmitterSpawnFailure(error)
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
        func draw(_ particle: EmitterParticle, additive: Bool = false) {
            guard particle.isAlive else { return }
            do {
                if try renderEmitterParticle(
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
            } catch {
                recordEmitterRenderFailure(error)
            }
        }

        switch configuration.renderMode {
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
                recordEmitterRenderFailure(.additivePipelineUnavailable)
                return
            }
            state.lastRenderUsedAdditiveBlending = true
            for particle in state.particles {
                draw(particle, additive: true)
            }
        default:
            recordEmitterRenderFailure(
                .unsupportedRenderMode(configuration.renderMode.rawValue)
            )
        }
    }

    private func emitterParticleBirthCount(
        cell: CAEmitterCell,
        activeDelta: Float,
        rateMultiplier: Float,
        remainderKey: EmitterBirthRemainderKey,
        state: EmitterLayerState
    ) throws(CAEmitterFailure) -> Int {
        let configuredBirthRate = cell.birthRate * rateMultiplier
        guard activeDelta.isFinite,
              activeDelta >= 0,
              configuredBirthRate.isFinite else {
            throw .invalidCellBirthRate
        }
        let birthRate = max(0, configuredBirthRate)
        let accumulated = birthRate * activeDelta
            + state.birthRemainders[remainderKey, default: 0]
        guard accumulated.isFinite,
              accumulated >= 0,
              accumulated <= Float(Int.max) else {
            throw .invalidCellBirthRate
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
        configuration: CAEmitterRenderConfiguration,
        state: EmitterLayerState
    ) throws(CAEmitterFailure) -> EmitterParticle {
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
                throw .invalidCellContents
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
            throw .nonFiniteEmissionDirection
        }
        let direction: SIMD3<Float>
        if let parentDirection {
            do {
                direction = try EmitterCellSimulation.childDirection(
                    localDirection: localDirection,
                    parentDirection: parentDirection
                )
            } catch {
                throw .invalidChildDirection
            }
        } else {
            direction = localDirection
        }
        let velocityVariation = CGFloat(state.randomSource.signedFloat()) * cell.velocityRange
        let velocity = Float(cell.velocity + velocityVariation) * configuration.velocity
        guard velocity.isFinite else { throw .nonFiniteParticleState }

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
        ) * configuration.lifetime
        particle.previousLifetime = particle.lifetime
        particle.maxLifetime = particle.lifetime
        particle.scale = Float(
            cell.scale + CGFloat(state.randomSource.signedFloat()) * cell.scaleRange
        ) * configuration.scale * inheritedScale
        particle.previousScale = particle.scale
        particle.scaleSpeed = Float(cell.scaleSpeed) * configuration.scale * inheritedScale
        particle.rotationSpeed = Float(
            cell.spin + CGFloat(state.randomSource.signedFloat()) * cell.spinRange
        ) * configuration.spin

        let baseColor = try emitterCellColor(cell, random: &state.randomSource)
        particle.color = baseColor * inheritedColor
        particle.previousColor = particle.color
        particle.colorSpeed = SIMD4(
            cell.redSpeed,
            cell.greenSpeed,
            cell.blueSpeed,
            cell.alphaSpeed
        ) * inheritedColor
        guard particleStateIsFinite(particle) else { throw .nonFiniteParticleState }
        particle.isAlive = particle.lifetime > 0
        return particle
    }

    private func emitterCellColor(
        _ cell: CAEmitterCell,
        random: inout EmitterRandomSource
    ) throws(CAEmitterFailure) -> SIMD4<Float> {
        let base: SIMD4<Float>
        if let color = cell.color {
            guard let converted = color.converted(
                to: .deviceRGB,
                intent: .defaultIntent,
                options: nil
            ), let components = converted.components,
               components.count == 4,
               components.allSatisfy(\.isFinite) else {
                throw .invalidCellColor
            }
            base = SIMD4(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
        } else {
            base = SIMD4(1, 1, 1, 1)
        }
        let color = SIMD4(
            base.x + random.signedFloat() * cell.redRange,
            base.y + random.signedFloat() * cell.greenRange,
            base.z + random.signedFloat() * cell.blueRange,
            base.w + random.signedFloat() * cell.alphaRange
        )
        guard color.x.isFinite,
              color.y.isFinite,
              color.z.isFinite,
              color.w.isFinite else {
            throw .invalidCellColor
        }
        return color
    }

    private func renderEmitterParticle(
        _ particle: EmitterParticle,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        vertexBuffer: GPUBuffer,
        uniformBuffer: GPUBuffer,
        additive: Bool
    ) throws(CAEmitterFailure) -> Bool {
        guard let image = particle.contents else { return false }
        let textureFormat = CGImageTexturePixelFormat.recommended(for: image)
        let memorySizeBytes: UInt64
        do {
            memorySizeBytes = try mipmappedRGBAByteCount(
                width: image.width,
                height: image.height,
                format: textureFormat,
                device: device
            )
        } catch {
            recordContentsConversionFailure(error)
            throw .imageConversionFailed(error)
        }
        guard let sampling = EmitterTextureSampling(
            magnificationFilter: particle.magnificationFilter,
            minificationFilter: particle.minificationFilter
        ) else {
            throw .invalidCellContents
        }
        guard let sampler = emitterTextureSamplers[sampling],
              let texturedBindGroupLayout,
              let selectedPipeline = selectedEmitterPipeline(additive: additive) else {
            throw .rendererResourcesUnavailable
        }
        guard let textureManager else {
            throw .textureResourcesUnavailable
        }
        var textureConversionError: CAImageContentsConversionError?
        let texture = textureManager.getOrCreateTexture(
            for: image,
            width: image.width,
            height: image.height,
            memorySizeBytes: memorySizeBytes,
            factory: {
                do {
                    return try self.createGPUTexture(
                        from: image,
                        format: textureFormat,
                        device: device
                    )
                } catch let error as CAImageContentsConversionError {
                    textureConversionError = error
                    return nil
                } catch {
                    textureConversionError = .conversionFailed
                    return nil
                }
            }
        )
        guard let texture else {
            if let textureConversionError {
                recordContentsConversionFailure(textureConversionError)
                throw .imageConversionFailed(textureConversionError)
            }
            throw .textureResourcesUnavailable
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
            throw .vertexCapacityExceeded
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
    private func recordTiledLayerRenderFailure(_ failure: CATiledLayerRenderFailure) {
        tiledLayerRenderFailureCount += 1
        lastTiledLayerRenderFailure = failure
    }

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

        let configuration: CATiledLayerRenderConfiguration
        do {
            configuration = try CATiledLayerRenderConfiguration(layer: presentation)
        } catch {
            recordTiledLayerRenderFailure(error)
            return
        }

        let bounds = configuration.bounds
        guard bounds.width > 0, bounds.height > 0 else { return }

        // Calculate current LOD level
        let lodLevel = calculateLODLevel(tiledLayer: presentation, modelMatrix: modelMatrix)

        // Adjust tile size based on LOD level
        // Higher LOD = larger tiles (covering more area with less detail)
        let lodScale = pow(2.0, CGFloat(lodLevel))
        let pixelScale = configuration.contentsScale / lodScale
        guard pixelScale.isFinite, pixelScale > 0 else {
            recordTiledLayerRenderFailure(.invalidContentsScale(configuration.contentsScale))
            return
        }
        let maximumTextureDimension = max(1, Int(device.limits.maxTextureDimension2D))
        let maximumLogicalTileDimension = CGFloat(maximumTextureDimension) / pixelScale
        let adjustedTileSize = CGSize(
            width: min(configuration.tileSize.width * lodScale, maximumLogicalTileDimension),
            height: min(configuration.tileSize.height * lodScale, maximumLogicalTileDimension)
        )

        guard adjustedTileSize.width.isFinite,
              adjustedTileSize.height.isFinite,
              adjustedTileSize.width > 0,
              adjustedTileSize.height > 0 else {
            recordTiledLayerRenderFailure(.invalidTileSize(configuration.tileSize))
            return
        }
        let tileCountX = ceil(bounds.width / adjustedTileSize.width)
        let tileCountY = ceil(bounds.height / adjustedTileSize.height)
        guard tileCountX.isFinite,
              tileCountY.isFinite,
              tileCountX > 0,
              tileCountY > 0,
              tileCountX <= CGFloat(Self.maxLayers),
              tileCountY <= CGFloat(Self.maxLayers) else {
            recordTiledLayerRenderFailure(.tileCountExceedsRendererCapacity(Int.max))
            return
        }
        let tilesX = Int(tileCountX)
        let tilesY = Int(tileCountY)
        guard tilesY == 0 || tilesX <= Self.maxLayers / tilesY else {
            let reportedCount = tilesX > Int.max / max(tilesY, 1)
                ? Int.max
                : tilesX * tilesY
            recordTiledLayerRenderFailure(.tileCountExceedsRendererCapacity(reportedCount))
            return
        }
        let tileMediaTime = CARenderTimeContext.currentMediaTime

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
              let textureSampler = textureSampler else {
            recordTiledLayerRenderFailure(.rendererResourcesUnavailable)
            return
        }

        // Get or create texture for image using the texture manager.
        // The manager retains `image` for the cached lifetime; passing
        // the CGImage directly (rather than `image as AnyObject`) keeps
        // the cache key on the same identity that downstream caches use.
        let imageWidth = image.width
        let imageHeight = image.height
        let textureFormat = CGImageTexturePixelFormat.recommended(for: image)
        let memorySizeBytes: UInt64
        do {
            memorySizeBytes = try mipmappedRGBAByteCount(
                width: imageWidth,
                height: imageHeight,
                format: textureFormat,
                device: device
            )
        } catch {
            recordContentsConversionFailure(error)
            return
        }
        guard let textureManager else {
            recordTiledLayerRenderFailure(.rendererResourcesUnavailable)
            return
        }
        var textureConversionError: CAImageContentsConversionError?
        let texture = textureManager.getOrCreateTexture(
            for: image,
            width: imageWidth,
            height: imageHeight,
            memorySizeBytes: memorySizeBytes,
            factory: {
                do {
                    return try self.createGPUTexture(
                        from: image,
                        format: textureFormat,
                        device: device
                    )
                } catch let error as CAImageContentsConversionError {
                    textureConversionError = error
                    return nil
                } catch {
                    textureConversionError = .conversionFailed
                    return nil
                }
            }
        )
        guard let texture else {
            if let textureConversionError {
                recordContentsConversionFailure(textureConversionError)
                recordTiledLayerRenderFailure(.imageConversionFailed(textureConversionError))
            } else {
                recordTiledLayerRenderFailure(.rendererResourcesUnavailable)
            }
            return
        }

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

        guard let allocation = allocateVertices(count: vertices.count) else {
            recordTiledLayerRenderFailure(.rendererResourcesUnavailable)
            return
        }
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
        guard let texturedBindGroupLayout = texturedBindGroupLayout,
              let selectedPipeline = selectTexturedPipeline(for: layer) else {
            recordTiledLayerRenderFailure(.rendererResourcesUnavailable)
            return
        }
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

        renderPass.setPipeline(selectedPipeline)
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
        let cacheGeneration = tiledLayer.tileCacheGeneration
        tiledLayer.loadingTiles.insert(tileKey)
        tiledLayer.loadingTileGenerations[tileKey] = cacheGeneration

        pendingTileDraws.append(PendingTileDraw(
            tiledLayer: tiledLayer,
            delegate: delegate,
            tileKey: tileKey,
            tileRect: tileRect,
            scale: scale,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            cacheGeneration: cacheGeneration
        ))
    }

    private func processPendingTileDraws() {
        let requests = pendingTileDraws
        pendingTileDraws.removeAll(keepingCapacity: true)
        for request in requests {
            beginTileDraw(
                tiledLayer: request.tiledLayer,
                delegate: request.delegate,
                tileKey: request.tileKey,
                tileRect: request.tileRect,
                scale: request.scale,
                pixelWidth: request.pixelWidth,
                pixelHeight: request.pixelHeight,
                cacheGeneration: request.cacheGeneration
            )
        }
    }

    private func beginTileDraw(
        tiledLayer: CATiledLayer,
        delegate: any CALayerDelegate,
        tileKey: CATiledLayer.TileKey,
        tileRect: CGRect,
        scale: CGFloat,
        pixelWidth: Int,
        pixelHeight: Int,
        cacheGeneration: UInt64
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
            recordTiledLayerRenderFailure(.drawingContextCreationFailed)
            if tiledLayer.loadingTileGenerations[tileKey] == cacheGeneration {
                tiledLayer.loadingTiles.remove(tileKey)
                tiledLayer.loadingTileGenerations.removeValue(forKey: tileKey)
            }
            return
        }

        context.scaleBy(x: scale, y: scale)
        context.translateBy(x: -tileRect.minX, y: -tileRect.minY)
        delegate.draw(tiledLayer, in: context)

        Task { @MainActor [weak self, weak tiledLayer] in
            guard let tiledLayer else { return }
            guard let image = await context.makeImageAsync() else {
                self?.recordTiledLayerRenderFailure(.imageCreationFailed)
                if tiledLayer.loadingTileGenerations[tileKey] == cacheGeneration {
                    tiledLayer.loadingTiles.remove(tileKey)
                    tiledLayer.loadingTileGenerations.removeValue(forKey: tileKey)
                }
                return
            }
            if tiledLayer.cacheImage(
                image,
                for: tileKey,
                requestGeneration: cacheGeneration
            ) {
                tiledLayer.markDirty(.contents)
            }
        }
    }
}

#endif
