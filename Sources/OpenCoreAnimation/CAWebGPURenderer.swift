#if arch(wasm32)
@_spi(SoftwareBitmapContext) import OpenCoreGraphics
import Foundation
import JavaScriptKit
import SwiftWebGPU

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
fileprivate enum TexturedCacheKey: Hashable {
    case image(ObjectIdentifier)
    case layer(ObjectIdentifier)
    case transitionSource(ObjectIdentifier)
    case transitionTarget(ObjectIdentifier)
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

private struct TransitionParticipantCapture {
    let texture: GPUTexture
    let compositeLayer: CALayer
}

private struct TransitionCapturePair {
    let source: TransitionParticipantCapture
    let target: TransitionParticipantCapture
}

/// A renderer that uses WebGPU to render layer trees in WASM/Web environments.
///
/// This is the primary renderer for OpenCoreAnimation in production.
/// It conforms to both `CARenderer` (public API) and `CARendererDelegate` (internal).
public final class CAWebGPURenderer: CARenderer, CARendererDelegate {

    // MARK: - Constants

    /// Maximum number of layers that can be rendered per frame.
    private static let maxLayers = 1024

    /// Uniform buffer alignment requirement (WebGPU minimum is 256 bytes).
    private static let uniformAlignment: UInt64 = 256

    /// Size of aligned uniform data per layer.
    private static var alignedUniformSize: UInt64 {
        let baseSize = UInt64(MemoryLayout<CARendererUniforms>.stride)
        return ((baseSize + uniformAlignment - 1) / uniformAlignment) * uniformAlignment
    }

    // MARK: - Properties

    /// The WebGPU device.
    private var device: GPUDevice?

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

    /// Number of frozen source and target textures currently retained by the renderer.
    @_spi(RendererDiagnostics)
    public var activeTransitionTextureCount: Int {
        transitionCaptures.count * 2
    }

    /// The preferred texture format.
    private var preferredFormat: GPUTextureFormat = .bgra8unorm

    /// The swap-chain texture most recently submitted for presentation.
    private var lastRenderedTexture: GPUTexture?

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

    /// Composites premultiplied-alpha offscreen captures.
    private var premultipliedTexturedPipeline: GPURenderPipeline?

    /// Stencil-aware variant for premultiplied-alpha captures.
    private var premultipliedTexturedStencilPipeline: GPURenderPipeline?

    /// The bind group layout for textured rendering.
    private var texturedBindGroupLayout: GPUBindGroupLayout?

    /// The texture sampler.
    private var textureSampler: GPUSampler?

    /// Texture manager with LRU cache for efficient texture memory management.
    private var textureManager: GPUTextureManager?

    /// Geometry cache for tessellated path data.
    private var geometryCache: GeometryCache?

    // MARK: - Shadow Rendering

    /// Shadow mask texture for blur operations.
    private var shadowMaskTexture: GPUTexture?

    /// Cached view for `shadowMaskTexture` (reset when the texture is recreated).
    private var shadowMaskTextureView: GPUTextureView?

    /// Intermediate texture for blur ping-pong.
    private var shadowBlurTexture: GPUTexture?

    /// Cached view for `shadowBlurTexture` (reset when the texture is recreated).
    private var shadowBlurTextureView: GPUTextureView?

    /// Shadow blur pipeline (horizontal pass).
    private var shadowBlurHorizontalPipeline: GPURenderPipeline?

    /// Shadow blur pipeline (vertical pass).
    private var shadowBlurVerticalPipeline: GPURenderPipeline?

    /// Shadow composite pipeline.
    private var shadowCompositePipeline: GPURenderPipeline?

    /// Shadow mask pipeline.
    private var shadowMaskPipeline: GPURenderPipeline?

    /// Shadow bind group layout.
    private var shadowBindGroupLayout: GPUBindGroupLayout?

    /// Full-screen quad sampler for blur.
    private var blurSampler: GPUSampler?

    /// Blur uniform buffer.
    private var blurUniformBuffer: GPUBuffer?

    /// Bind group for horizontal blur pass (samples from shadowMaskTexture).
    private var blurHorizontalBindGroup: GPUBindGroup?

    /// Bind group for vertical blur pass (samples from shadowBlurTexture).
    private var blurVerticalBindGroup: GPUBindGroup?

    /// Cache of pre-blurred shadow textures keyed by layer identity.
    private var blurredShadowCache: [ObjectIdentifier: GPUTexture] = [:]

    // MARK: - Filter Rendering (CAFilter)

    /// Filter source texture - layer content is rendered here before blur.
    private var filterSourceTexture: GPUTexture?

    /// Cached view for `filterSourceTexture`.
    private var filterSourceTextureView: GPUTextureView?

    /// Filter blur intermediate texture for ping-pong blurring.
    private var filterBlurTexture: GPUTexture?

    /// Cached view for `filterBlurTexture`.
    private var filterBlurTextureView: GPUTextureView?

    /// Filter result texture - stores the final filter pass result.
    private var filterResultTexture: GPUTexture?

    /// Cached view for `filterResultTexture`.
    private var filterResultTextureView: GPUTextureView?

    /// Bind group for filter horizontal blur (samples from filterSourceTexture).
    private var filterBlurHorizontalBindGroup: GPUBindGroup?

    /// Bind group for filter vertical blur (samples from filterBlurTexture).
    private var filterBlurVerticalBindGroup: GPUBindGroup?

    /// applyBlurFilter horizontal-pass bind group reading `filterSourceTexture`.
    ///
    /// applyBlurFilter is called once per CIFilter operation per filtered layer.
    /// Each call previously did two `device.createBindGroup` round-trips to JS
    /// even though the bind groups only depend on which persistent texture
    /// (filterSourceTexture / filterBlurTexture / filterResultTexture) they
    /// sample from. Cache them keyed by texture identity; recreated alongside
    /// the textures in `createFilterTextures`.
    private var applyBlurFromSourceBindGroup: GPUBindGroup?

    /// applyBlurFilter horizontal-pass bind group reading `filterResultTexture`.
    private var applyBlurFromResultBindGroup: GPUBindGroup?

    /// applyBlurFilter vertical-pass bind group reading `filterBlurTexture`.
    private var applyBlurFromBlurBindGroup: GPUBindGroup?

    // MARK: - Pre-Render Dedicated Buffers

    /// Dedicated vertex buffer for pre-render passes (shadow mask, filter source).
    /// Prevents overwriting main render vertex data at offset 0.
    private var preRenderVertexBuffer: GPUBuffer?

    /// Dedicated uniform buffer for pre-render passes.
    private var preRenderUniformBuffer: GPUBuffer?

    /// Dedicated bind group for the pre-render uniform buffer.
    private var preRenderBindGroup: GPUBindGroup?

    // MARK: - Composite Buffers

    /// Pre-allocated vertex buffer for shadow composite rendering.
    private var shadowCompositeVertexBuffer: GPUBuffer?

    /// Pre-allocated uniform buffer for shadow composite rendering.
    private var shadowCompositeUniformBuffer: GPUBuffer?

    /// Pre-allocated vertex buffer for filter composite rendering.
    private var filterCompositeVertexBuffer: GPUBuffer?

    /// Pre-allocated uniform buffer for filter composite rendering.
    private var filterCompositeUniformBuffer: GPUBuffer?

    /// Filter composite pipeline for blending filtered result with optional tint.
    private var filterCompositePipeline: GPURenderPipeline?

    /// Stencil-aware filter composite pipeline.
    private var filterCompositeStencilPipeline: GPURenderPipeline?

    /// Indicates whether a filtered layer has been pre-rendered.
    private var hasPrerenderredFilter: Bool = false

    /// The layer that has been pre-rendered with filters.
    private var prerenderredFilterLayer: CALayer?

    /// The texture that holds the final pre-rendered filter output for this frame.
    private var prerenderredFilterTexture: GPUTexture?

    /// The blur radius used for the pre-rendered filter.
    private var prerenderredFilterBlurRadius: CGFloat = 0

    /// The root layer currently being rendered into the offscreen filter source texture.
    /// While set, filter compositing is suppressed so the subtree can be captured raw.
    private weak var filterPrerenderRootLayer: CALayer?

    // MARK: - Rasterization Cache (R3.2 / R3.4)

    /// LRU + byte-budget cache of captured `shouldRasterize` subtrees.
    /// Allocated on first `resize` once the viewport size is known so the
    /// budget can be sized to `viewport × 4 × 2.5` per WWDC 2014 #419.
    private var rasterizationCache: RasterizationCache<GPUTexture>?

    /// Frozen source/target pairs keyed by the immutable transition source snapshot.
    private var transitionCaptures: [ObjectIdentifier: TransitionCapturePair] = [:]

    /// Source snapshots referenced by an active transition during the current frame.
    private var activeTransitionSourceIDs: Set<ObjectIdentifier> = []

    /// Capture-only depth textures retained until their command buffer has been submitted.
    private var transientCaptureDepthTextures: [GPUTexture] = []

    /// Per-frame scratch storage: layers whose subtree has been captured
    /// (or had a fresh cache hit) this frame, mapped to the texture used
    /// for compositing. Populated by `prerenderRasterizedLayers`,
    /// consumed by `renderLayer`, cleared after submit.
    private var prerasterizedTextures: [ObjectIdentifier: GPUTexture] = [:]

    /// Persistent cache of `GPUTextureView`s keyed by `TexturedCacheKey`.
    ///
    /// `gpuTexture.createView()` is a JS round-trip to allocate a fresh
    /// `GPUTextureView`. The view is invariant for a given texture, so we
    /// keep it across frames.
    ///
    /// **Identity contract**: keyed by either `.image(OID(cgImage))` for
    /// regular content layers (eviction wired to `GPUTextureManager.onEvict`)
    /// or `.layer(OID(layer))` for rasterized composites of
    /// `shouldRasterize` subtrees. Tagging the kind keeps the two
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

    // MARK: - Mask Rendering (Stencil-based)

    /// Mask texture for layer masking (texture-based approach).
    private var maskTexture: GPUTexture?

    /// Masked rendering pipeline (texture-based).
    private var maskedPipeline: GPURenderPipeline?

    /// Mask bind group layout.
    private var maskBindGroupLayout: GPUBindGroupLayout?

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

    /// Current stencil reference value for nested masks.
    private var currentStencilValue: UInt32 = 0

    /// Tracks mask nesting depth for stencil-based masking.
    /// When > 0, textured and composite pipelines must use stencil-aware variants.
    /// Supports nested masks (e.g., child B with mask inside parent A with mask).
    private var maskNestingDepth: Int = 0

    /// Stencil-aware textured pipeline (tests stencil buffer for mask).
    private var texturedStencilPipeline: GPURenderPipeline?

    /// Opaque textured pipeline (no alpha blend). R3.5 — selected when
    /// `RasterizationDecisions.blendEnabled(for:)` returns false.
    private var texturedOpaquePipeline: GPURenderPipeline?

    /// Opaque stencil-aware textured pipeline (R3.5 + mask).
    private var texturedStencilOpaquePipeline: GPURenderPipeline?

    /// Stencil-aware shadow composite pipeline (tests stencil buffer for mask).
    private var shadowCompositeStencilPipeline: GPURenderPipeline?

    // MARK: - Particle System (CAEmitterLayer)

    /// Particle instance buffer.
    private var particleBuffer: GPUBuffer?

    /// Particle pipeline.
    private var particlePipeline: GPURenderPipeline?

    /// Maximum number of particles.
    private static let maxParticles = 10000

    /// Active particle data.
    private var activeParticles: [EmitterParticle] = []

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
            let key = TexturedCacheKey.image(ObjectIdentifier(cgImage))
            self.texturedTextureViewCache.removeValue(forKey: key)
            self.perFrameTexturedBindGroupCache.removeValue(forKey: key)
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

        // Create textured pipeline
        try createTexturedPipeline(device: device)

        // Create shadow/blur pipelines
        try createShadowPipelines(device: device)

        // Create shadow textures
        createShadowTextures(width: Int(width), height: Int(height))

        // Create stencil pipelines for CALayer.mask
        try createStencilPipelines(device: device)

        // Create dedicated pre-render buffers (6 vertices max for a quad)
        let preRenderVertexSize = UInt64(6 * MemoryLayout<CARendererVertex>.stride)
        preRenderVertexBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: preRenderVertexSize,
            usage: [.vertex, .copyDst]
        ))

        let preRenderUniformSize = Self.alignedUniformSize
        preRenderUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: preRenderUniformSize,
            usage: [.uniform, .copyDst]
        ))

        // Create bind group for pre-render uniform buffer
        if let bindGroupLayout = bindGroupLayout, let preRenderUniformBuffer = preRenderUniformBuffer {
            preRenderBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                layout: bindGroupLayout,
                entries: [
                    GPUBindGroupEntry(
                        binding: 0,
                        resource: .bufferBinding(GPUBufferBinding(
                            buffer: preRenderUniformBuffer,
                            size: UInt64(MemoryLayout<CARendererUniforms>.stride)
                        ))
                    )
                ]
            ))
        }

        // Create composite buffers
        shadowCompositeVertexBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: preRenderVertexSize,
            usage: [.vertex, .copyDst]
        ))
        shadowCompositeUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: preRenderUniformSize,
            usage: [.uniform, .copyDst]
        ))
        filterCompositeVertexBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: preRenderVertexSize,
            usage: [.vertex, .copyDst]
        ))
        filterCompositeUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: preRenderUniformSize,
            usage: [.uniform, .copyDst]
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
            mipmapFilter: .linear
        ))

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

        createPremultipliedTexturedPipelines(
            device: device,
            bindGroupLayout: texturedBindGroupLayout
        )
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

        func makePipeline(stencilCompare: GPUCompareFunction) -> GPURenderPipeline {
            device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
                vertex: vertexState,
                depthStencil: GPUDepthStencilState(
                    format: .depth24plusStencil8,
                    depthWriteEnabled: false,
                    depthCompare: .always,
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

        premultipliedTexturedPipeline = makePipeline(stencilCompare: .always)
        premultipliedTexturedStencilPipeline = makePipeline(stencilCompare: .equal)
    }

    /// R3.5: pick the textured pipeline based on stencil state and the
    /// `blendEnabled` decision for the layer. Falls back to the alpha-blended
    /// variant when an opaque pipeline isn't created (test fallback).
    private func selectTexturedPipeline(for layer: CALayer) -> GPURenderPipeline? {
        let blendOff = !RasterizationDecisions.blendEnabled(for: layer)
        if maskNestingDepth > 0 {
            if blendOff, let opaque = texturedStencilOpaquePipeline { return opaque }
            return texturedStencilPipeline ?? texturedPipeline
        }
        if blendOff, let opaque = texturedOpaquePipeline { return opaque }
        return texturedPipeline
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
        let blurHShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: CAWebGPUShaders.blurHorizontal
        ))

        shadowBlurHorizontalPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: blurHShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: blurHShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(format: .rgba8unorm)
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
                    GPUColorTargetState(format: .rgba8unorm)
                ]
            ),
            layout: .layout(blurPipelineLayout)
        ))

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
                    GPUColorTargetState(format: .rgba8unorm)
                ]
            ),
            layout: .layout(maskPipelineLayout)
        ))
    }

    /// Creates or recreates shadow textures for the given size.
    private func createShadowTextures(width: Int, height: Int) {
        guard let device = device,
              let shadowBindGroupLayout = shadowBindGroupLayout,
              let blurSampler = blurSampler,
              width > 0, height > 0 else { return }

        // R3.7: the cache references the previous `shadowMaskTexture` handle;
        // recreating it invalidates every entry.
        blurredShadowCache.removeAll(keepingCapacity: true)

        // Shadow mask texture (stores layer shape)
        shadowMaskTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.renderAttachment, .textureBinding]
        ))
        shadowMaskTextureView = shadowMaskTexture?.createView()

        // Shadow blur texture (for ping-pong blur)
        shadowBlurTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.renderAttachment, .textureBinding]
        ))
        shadowBlurTextureView = shadowBlurTexture?.createView()

        // Create blur uniform buffer
        blurUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(MemoryLayout<BlurUniforms>.stride),
            usage: [.uniform, .copyDst]
        ))

        // Create bind groups for blur passes
        guard let shadowMaskTexture = shadowMaskTexture,
              let shadowBlurTexture = shadowBlurTexture,
              let blurUniformBuffer = blurUniformBuffer else { return }

        // Horizontal blur: samples from shadowMaskTexture, outputs to shadowBlurTexture
        blurHorizontalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(shadowMaskTexture.createView())),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        // Vertical blur: samples from shadowBlurTexture, outputs to shadowMaskTexture
        blurVerticalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(shadowBlurTexture.createView())),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))
    }

    /// Creates or recreates filter textures for the given size.
    private func createFilterTextures(width: Int, height: Int) {
        guard let device = device,
              let shadowBindGroupLayout = shadowBindGroupLayout,
              let blurSampler = blurSampler,
              let blurUniformBuffer = blurUniformBuffer,
              width > 0, height > 0 else { return }

        // Filter source texture (stores layer content before blur)
        filterSourceTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.renderAttachment, .textureBinding]
        ))
        filterSourceTextureView = filterSourceTexture?.createView()

        // Filter blur texture (for ping-pong blur)
        filterBlurTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.renderAttachment, .textureBinding]
        ))
        filterBlurTextureView = filterBlurTexture?.createView()

        // Filter result texture (stores final blurred result)
        filterResultTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.renderAttachment, .textureBinding]
        ))
        filterResultTextureView = filterResultTexture?.createView()

        // Create bind groups for filter blur passes
        guard let filterSourceTexture = filterSourceTexture,
              let filterBlurTexture = filterBlurTexture else { return }

        // Horizontal blur: samples from filterSourceTexture, outputs to filterBlurTexture
        filterBlurHorizontalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(filterSourceTexture.createView())),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        // Vertical blur: samples from filterBlurTexture, outputs to filterResultTexture
        filterBlurVerticalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(filterBlurTexture.createView())),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        // Pre-build the per-input bind groups consumed by `applyBlurFilter`.
        // The chain in prerenderFilteredLayers ping-pongs `currentTexture`
        // between filterSourceTexture and filterResultTexture as the horizontal
        // input, with filterBlurTexture always serving as the vertical input.
        // Caching these three avoids creating two bind groups per CIFilter
        // operation per frame.
        guard let filterResultTexture = filterResultTexture else { return }
        applyBlurFromSourceBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(filterSourceTextureView!)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))
        applyBlurFromResultBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(filterResultTexture.createView())),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))
        applyBlurFromBlurBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(filterBlurTextureView!)),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
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

        // Recreate shadow textures
        createShadowTextures(width: width, height: height)

        // Recreate filter textures
        createFilterTextures(width: width, height: height)

        // R3.2/R3.4: size the rasterization cache to viewport × 4 × 2.5
        // per PERFORMANCE_DESIGN.md §5.2 (WWDC 2014 #419 budget).
        // Recreating the cache on resize drops every entry — the
        // captured textures are canvas-sized and would render at the
        // wrong dimensions otherwise.
        let budget = max(0, Int((Double(width) * Double(height) * 4.0 * 2.5).rounded()))
        rasterizationCache = RasterizationCache<GPUTexture>(maxBytes: budget)
        prerasterizedTextures.removeAll(keepingCapacity: true)
        rasterizePrerenderRootLayer = nil
    }

    /// Tracks whether shadow pre-rendering has been done for this frame.
    private var shadowsPrerendered: Bool = false

    /// Stores the pre-blurred shadow alpha value for the current frame.
    /// After blur, this is stored in shadowMaskTexture.
    private var hasPrerenderredShadow: Bool = false

    public func render(layer rootLayer: CALayer) {
        guard let device = device,
              let context = context,
              let pipeline = pipeline,
              bindGroup != nil,
              let depthTexture = depthTexture else { return }

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
        transitionSuppressedLayer = nil
        renderTargetSizeOverride = nil
        activeTransitionSourceIDs.removeAll(keepingCapacity: true)
        for texture in transientCaptureDepthTextures {
            texture.destroy()
        }
        transientCaptureDepthTextures.removeAll(keepingCapacity: true)
        currentRootLayer = nil
        filterPrerenderRootLayer = nil

        // Reset clip rect stack for this frame
        clipRectStack.removeAll()

        // Reset shadow pre-rendering state
        shadowsPrerendered = false
        hasPrerenderredShadow = false

        // Reset filter pre-rendering state
        hasPrerenderredFilter = false
        prerenderredFilterLayer = nil
        prerenderredFilterTexture = nil
        prerenderredFilterBlurRadius = 0

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

        // Pre-render shadows with 2-pass Gaussian blur
        prerenderShadows(rootLayer, encoder: encoder, projectionMatrix: projectionMatrix)

        // Pre-render layers with blur filters
        prerenderFilteredLayers(rootLayer, encoder: encoder, projectionMatrix: projectionMatrix)

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
        } else if hasPrerenderredFilter, prerenderredFilterLayer === rootLayer {
            // The filtered root layer is composited from the offscreen texture, so drawing
            // its background here would double-apply the root background color.
            clearColor = GPUColor(r: 0, g: 0, b: 0, a: 0)
        } else if let bgColor = rootLayer.backgroundColor,
           let components = bgColor.components,
           components.count >= 4 {
            clearColor = GPUColor(
                r: components[0],
                g: components[1],
                b: components[2],
                a: components[3]
            )
        } else {
            clearColor = GPUColor(r: 0, g: 0, b: 0, a: 1)
        }

        // Store root layer to skip its backgroundColor rendering in renderLayer()
        currentRootLayer = rootLayer

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
                depthClearValue: 1.0,
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
        renderLayer(rootLayer, renderPass: renderPass, parentMatrix: projectionMatrix)

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
        texturedPipeline = nil
        texturedOpaquePipeline = nil
        premultipliedTexturedPipeline = nil
        premultipliedTexturedStencilPipeline = nil
        texturedBindGroupLayout = nil
        textureSampler = nil

        // Shadow resources
        shadowMaskTexture = nil
        shadowMaskTextureView = nil
        shadowBlurTexture = nil
        shadowBlurTextureView = nil
        depthTextureView = nil
        lastRenderedTexture = nil
        shadowBlurHorizontalPipeline = nil
        shadowBlurVerticalPipeline = nil
        shadowCompositePipeline = nil
        shadowMaskPipeline = nil
        shadowBindGroupLayout = nil
        blurSampler = nil
        blurUniformBuffer = nil
        blurHorizontalBindGroup = nil
        blurVerticalBindGroup = nil
        blurredShadowCache.removeAll()
        textTextureCache.removeAll()
        textTextureAccessOrder.removeAll()

        // Pre-render buffers
        preRenderVertexBuffer = nil
        preRenderUniformBuffer = nil
        preRenderBindGroup = nil
        shadowCompositeVertexBuffer = nil
        shadowCompositeUniformBuffer = nil
        filterCompositeVertexBuffer = nil
        filterCompositeUniformBuffer = nil

        // Filter resources
        filterSourceTexture = nil
        filterSourceTextureView = nil
        filterBlurTexture = nil
        filterBlurTextureView = nil
        filterResultTexture = nil
        filterResultTextureView = nil
        filterBlurHorizontalBindGroup = nil
        filterBlurVerticalBindGroup = nil
        applyBlurFromSourceBindGroup = nil
        applyBlurFromResultBindGroup = nil
        applyBlurFromBlurBindGroup = nil
        filterCompositePipeline = nil
        filterCompositeStencilPipeline = nil
        hasPrerenderredFilter = false
        prerenderredFilterLayer = nil
        prerenderredFilterTexture = nil
        prerenderredFilterBlurRadius = 0

        // Rasterization cache (R3.2 / R3.4)
        rasterizationCache?.removeAll()
        rasterizationCache = nil
        for capture in transitionCaptures.values {
            capture.source.texture.destroy()
            capture.target.texture.destroy()
        }
        transitionCaptures.removeAll(keepingCapacity: false)
        activeTransitionSourceIDs.removeAll(keepingCapacity: false)
        for texture in transientCaptureDepthTextures {
            texture.destroy()
        }
        transientCaptureDepthTextures.removeAll(keepingCapacity: false)
        transitionSourceCaptureCount = 0
        transitionTargetCaptureCount = 0
        prerasterizedTextures.removeAll(keepingCapacity: false)
        rasterizePrerenderRootLayer = nil
        for request in pendingTileDraws {
            request.tiledLayer.loadingTiles.remove(request.tileKey)
        }
        pendingTileDraws.removeAll(keepingCapacity: false)

        // Mask resources
        maskTexture = nil
        maskedPipeline = nil
        maskBindGroupLayout = nil

        // Stencil resources
        stencilWritePipeline = nil
        stencilWriteRoundedPipeline = nil
        stencilTestPipeline = nil
        texturedStencilPipeline = nil
        texturedStencilOpaquePipeline = nil
        shadowCompositeStencilPipeline = nil
        maskNestingDepth = 0
        currentStencilValue = 0

        // Particle resources
        particleBuffer = nil
        particlePipeline = nil
        activeParticles.removeAll()

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
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        // Get the presentation layer for animated values, fall back to model layer
        // This is critical for animations to be visible - the presentation layer
        // reflects the current animated state of all properties
        let presentationLayer = layer._renderTimePresentation()

        // Skip hidden layers (using presentation layer values)
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }

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
        let hasMask = presentationLayer.mask != nil
        if hasMask, let maskLayer = presentationLayer.mask {
            // Render mask to stencil buffer
            renderMaskToStencil(maskLayer, renderPass: renderPass, parentMatrix: modelMatrix)
        }

        // Render shadow before layer content. Filters apply to the layer subtree capture, but
        // the shadow itself is composited separately in the main pass.
        if filterPrerenderRootLayer == nil,
           presentationLayer.shadowOpacity > 0 && presentationLayer.shadowColor != nil {
            renderLayerShadow(presentationLayer, device: device, renderPass: renderPass,
                            modelMatrix: modelMatrix, parentMatrix: parentMatrix)
        }

        // Handle special layer types first

        // Check for layer filters.
        // If this layer has supported filters and was pre-rendered, composite the filtered result.
        let supportedFilterOperations = presentationLayer.supportedFilterOperations
        if filterPrerenderRootLayer == nil, !supportedFilterOperations.isEmpty {
            // Check if this specific layer was pre-rendered
            if hasPrerenderredFilter && prerenderredFilterLayer === layer {
                // Composite the pre-rendered filtered texture.
                renderFilteredLayerComposite(
                    layer,
                    device: device,
                    renderPass: renderPass,
                    modelMatrix: modelMatrix
                )
                // Mark filter as consumed
                hasPrerenderredFilter = false
                prerenderredFilterLayer = nil
                return
            }
        }

        // R3.2 cache-hit composite: when this layer was pre-rendered
        // into a rasterization texture this frame and the renderer is
        // now walking the tree for the main pass (i.e. we are NOT
        // inside the capture pass for this same layer), draw the cached
        // pixelSize texture as a quad placed at the layer's transform
        // with the layer's current opacity. The capture pass itself
        // reaches this layer with `rasterizePrerenderRootLayer === layer`,
        // so we must let it through to render the actual content into
        // the texture.
        if rasterizePrerenderRootLayer !== layer,
           let cachedTexture = prerasterizedTextures[ObjectIdentifier(layer)] {
            renderRasterizedLayerComposite(
                layer,
                presentationLayer: presentationLayer,
                texture: cachedTexture,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix
            )
            return
        }

        // CATransformLayer: Only render sublayers, skip own properties
        if presentationLayer is CATransformLayer {
            renderTransformLayerSublayers(layer, renderPass: renderPass, parentMatrix: modelMatrix)
            return
        }

        // CAEmitterLayer: Render particle system
        if let emitterLayer = presentationLayer as? CAEmitterLayer {
            renderEmitterLayer(emitterLayer, device: device, renderPass: renderPass,
                             modelMatrix: modelMatrix, bindGroup: bindGroup)
            // Render sublayers after particles
            if let sublayers = layer.sublayers, !sublayers.isEmpty {
                // Use sublayerMatrix helper to apply sublayerTransform and bounds.origin offset
                let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)
                for sublayer in layer.sortedSublayers() {
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
            && (layer !== currentRootLayer || presentationLayer.cornerRadius > 0) {
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
        } else if let contents = presentationLayer.contents as? CGImage {
            renderContentsLayer(presentationLayer, contents: contents, device: device,
                               renderPass: renderPass, modelMatrix: modelMatrix)
        }

        // Render sublayers (use model layer hierarchy, but presentation layer's sublayerTransform)
        if let sublayers = layer.sublayers {
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
            let sortedSubs = layer.sortedSublayers()
            if let replicatorLayer = presentationLayer as? CAReplicatorLayer {
                renderReplicatorSublayers(
                    replicatorLayer: replicatorLayer,
                    sublayers: sortedSubs,
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix
                )
            } else {
                for sublayer in sortedSubs {
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
        if presentationLayer.borderWidth > 0 && presentationLayer.borderColor != nil {
            renderLayerBorder(
                presentationLayer,
                device: device,
                renderPass: renderPass,
                modelMatrix: modelMatrix,
                bindGroup: bindGroup
            )
        }

        // Clear stencil mask if we used one
        if hasMask {
            clearStencilMask(renderPass: renderPass)
        }
    }

    private func prerenderTransitions(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder
    ) {
        guard let device, let pipeline else { return }

        func collect(_ layer: CALayer) {
            // Capture descendants first so a nested active transition is available
            // when its parent subtree is frozen into the target texture.
            for sublayer in layer.sublayers ?? [] {
                collect(sublayer)
            }

            let presentation = layer._renderTimePresentation()
            if let state = presentation._transitionRenderState {
                let sourceID = ObjectIdentifier(state.sourceLayer)
                activeTransitionSourceIDs.insert(sourceID)
                if transitionCaptures[sourceID] == nil {
                    guard let source = captureTransitionParticipant(
                        state.sourceLayer,
                        device: device,
                        pipeline: pipeline,
                        encoder: encoder
                    ) else {
                        return
                    }
                    guard let target = captureTransitionParticipant(
                        layer,
                        device: device,
                        pipeline: pipeline,
                        encoder: encoder
                    ) else {
                        source.texture.destroy()
                        return
                    }
                    transitionCaptures[sourceID] = TransitionCapturePair(
                        source: source,
                        target: target
                    )
                    transitionSourceCaptureCount += 1
                    transitionTargetCaptureCount += 1
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
    }

    private func destroyTransitionCapture(for sourceID: ObjectIdentifier) {
        if let capture = transitionCaptures.removeValue(forKey: sourceID) {
            capture.source.texture.destroy()
            capture.target.texture.destroy()
        }
        texturedTextureViewCache.removeValue(forKey: .transitionSource(sourceID))
        texturedTextureViewCache.removeValue(forKey: .transitionTarget(sourceID))
        perFrameTexturedBindGroupCache.removeValue(forKey: .transitionSource(sourceID))
        perFrameTexturedBindGroupCache.removeValue(forKey: .transitionTarget(sourceID))
    }

    private func captureTransitionParticipant(
        _ layer: CALayer,
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
        let pixelWidth = max(1, Int(ceil(requestedWidth * fittingScale)))
        let pixelHeight = max(1, Int(ceil(requestedHeight * fittingScale)))
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
        return TransitionParticipantCapture(texture: texture, compositeLayer: compositeLayer)
    }

    /// Composites frozen source and target layer trees for a built-in transition.
    private func renderTransition(
        state: CATransitionRenderState,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let sourceID = ObjectIdentifier(state.sourceLayer)
        guard let capture = transitionCaptures[sourceID] else { return }
        let targetLayer = capture.target.compositeLayer
        let progress = CGFloat(max(0, min(1, state.progress)))
        let direction = transitionDirection(
            subtype: state.subtype,
            bounds: targetLayer.bounds
        )
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

        switch state.type {
        case .fade:
            renderParticipant(capture.source, cacheKey: .transitionSource(sourceID), offset: .zero, opacityMultiplier: 1)
            renderParticipant(capture.target, cacheKey: .transitionTarget(sourceID), offset: .zero, opacityMultiplier: Float(progress))

        case .moveIn:
            renderParticipant(capture.source, cacheKey: .transitionSource(sourceID), offset: .zero, opacityMultiplier: 1)
            renderParticipant(
                capture.target,
                cacheKey: .transitionTarget(sourceID),
                offset: CGPoint(x: direction.x * (1 - progress), y: direction.y * (1 - progress)),
                opacityMultiplier: 1
            )

        case .push:
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
            renderParticipant(capture.target, cacheKey: .transitionTarget(sourceID), offset: .zero, opacityMultiplier: 1)
            renderParticipant(
                capture.source,
                cacheKey: .transitionSource(sourceID),
                offset: CGPoint(x: -direction.x * progress, y: -direction.y * progress),
                opacityMultiplier: 1
            )

        default:
            renderParticipant(capture.target, cacheKey: .transitionTarget(sourceID), offset: .zero, opacityMultiplier: 1)
        }
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

        let white = SIMD4<Float>(1, 1, 1, 1)
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
        let selectedPipeline = maskNestingDepth > 0
            ? premultipliedTexturedStencilPipeline
            : premultipliedTexturedPipeline
        guard let selectedPipeline else { return }
        renderPass.setPipeline(selectedPipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        if let pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    private func transitionDirection(subtype: CATransitionSubtype?, bounds: CGRect) -> CGPoint {
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
            return CGPoint(x: -bounds.width, y: 0)
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

        let color = presentationLayer.backgroundColorComponents
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

        let color = presentationLayer.borderColorComponents
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
        let maskPresentationLayer = maskLayer._renderTimePresentation()
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
        replicatorLayer: CAReplicatorLayer,
        sublayers: [CALayer],
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let instanceCount = max(1, replicatorLayer.instanceCount)
        let instanceTransform = replicatorLayer.instanceTransform
        let instanceDelay = replicatorLayer.instanceDelay

        // Get base instance color (defaults to white if not set)
        let baseColor: SIMD4<Float>
        if let color = replicatorLayer.instanceColor,
           let components = color.components, components.count >= 4 {
            baseColor = SIMD4<Float>(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
        } else {
            baseColor = SIMD4<Float>(1, 1, 1, 1)
        }

        // Color offsets per instance
        let redOffset = replicatorLayer.instanceRedOffset
        let greenOffset = replicatorLayer.instanceGreenOffset
        let blueOffset = replicatorLayer.instanceBlueOffset
        let alphaOffset = replicatorLayer.instanceAlphaOffset

        // Render each instance
        var cumulativeTransform = CATransform3DIdentity
        for instanceIndex in 0..<instanceCount {
            // Calculate color multiplier for this instance
            let colorMultiplier = SIMD4<Float>(
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

            // Render all sublayers with this instance's transform, color, and time offset
            // (caller is responsible for passing `sublayers` already sorted by zPosition).
            for sublayer in sublayers {
                renderLayerWithColorMultiplierAndTimeOffset(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: instanceMatrix,
                    colorMultiplier: colorMultiplier,
                    timeOffset: timeOffset
                )
            }

            // Apply instance transform for next iteration
            cumulativeTransform = CATransform3DConcat(cumulativeTransform, instanceTransform)
        }
    }

    /// Renders a layer with a color multiplier applied (for replicator instances).
    private func renderLayerWithColorMultiplier(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4,
        colorMultiplier: SIMD4<Float>
    ) {
        guard let device = device,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        let presentationLayer = layer._renderTimePresentation()
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }

        // Push effective opacity for this replicator instance subtree
        let effectiveOpacity = currentEffectiveOpacity * presentationLayer.opacity
        opacityStack.append(effectiveOpacity)
        defer { _ = opacityStack.popLast() }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // For now, render background color with the color multiplier applied
        if presentationLayer.backgroundColor != nil {
            // Apply color multiplier to the layer's background color
            var baseColor = presentationLayer.backgroundColorComponents
            baseColor.x *= colorMultiplier.x
            baseColor.y *= colorMultiplier.y
            baseColor.z *= colorMultiplier.z
            baseColor.w *= colorMultiplier.w

            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
            ]

            guard let allocation = allocateVertices(count: vertices.count) else { return }
            let (vertexOffset, layerIndex) = allocation

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
                cornerRadii: presentationLayer.cornerRadiiComponents
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

            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Recursively render sublayers
        if let sublayers = layer.sublayers {
            // Use sublayerMatrix helper to apply sublayerTransform and bounds.origin offset
            let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)

            for sublayer in layer.sortedSublayers() {
                renderLayerWithColorMultiplier(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix,
                    colorMultiplier: colorMultiplier
                )
            }
        }
    }

    /// Renders a layer with a color multiplier and time offset applied (for replicator instances with instanceDelay).
    ///
    /// The time offset is used to evaluate animations at a different point in time,
    /// creating staggered animation effects across replicator instances.
    private func renderLayerWithColorMultiplierAndTimeOffset(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4,
        colorMultiplier: SIMD4<Float>,
        timeOffset: CFTimeInterval
    ) {
        guard let device = device,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        // Get presentation layer at the specified time offset
        let presentationLayer: CALayer
        if timeOffset != 0 {
            presentationLayer = layer.presentationAtTimeOffset(timeOffset)
        } else {
            presentationLayer = layer._renderTimePresentation()
        }

        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }

        // Push effective opacity for this replicator instance subtree
        let effectiveOpacityForInstance = currentEffectiveOpacity * presentationLayer.opacity
        opacityStack.append(effectiveOpacityForInstance)
        defer { _ = opacityStack.popLast() }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Render background color with the color multiplier applied
        if presentationLayer.backgroundColor != nil {
            // Apply color multiplier to the layer's background color
            var baseColor = presentationLayer.backgroundColorComponents
            baseColor.x *= colorMultiplier.x
            baseColor.y *= colorMultiplier.y
            baseColor.z *= colorMultiplier.z
            baseColor.w *= colorMultiplier.w

            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
            ]

            guard let allocation = allocateVertices(count: vertices.count) else { return }
            let (vertexOffset, layerIndex) = allocation

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
                cornerRadii: presentationLayer.cornerRadiiComponents
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

            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Recursively render sublayers with the same time offset
        if let sublayers = layer.sublayers {
            // Use sublayerMatrix helper to apply sublayerTransform and bounds.origin offset
            let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)

            for sublayer in layer.sortedSublayers() {
                renderLayerWithColorMultiplierAndTimeOffset(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix,
                    colorMultiplier: colorMultiplier,
                    timeOffset: timeOffset
                )
            }
        }
    }

    /// Clamps a float value between min and max.
    private func clamp(_ value: Float, _ minVal: Float, _ maxVal: Float) -> Float {
        return min(max(value, minVal), maxVal)
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
            cornerRadii: effectiveCornerRadii
        )

        let white = SIMD4<Float>(1, 1, 1, 1)

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
        let white = SIMD4<Float>(1, 1, 1, 1)
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
            cornerRadii: effectiveCornerRadii
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
    ///   notifies eviction via `onEvict`), or `.layer(OID(layer))` for
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

        // Create texture
        let textureDescriptor = GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.textureBinding, .copyDst, .renderAttachment]
        )

        let texture = device.createTexture(descriptor: textureDescriptor)

        // Create JS Uint8Array from data
        let jsArray = createUint8Array(from: rgbaData)

        // Upload data
        device.queue.writeTexture(
            destination: GPUImageCopyTexture(texture: texture),
            data: jsArray,
            dataLayout: GPUImageDataLayout(
                offset: 0,
                bytesPerRow: UInt32(width * 4),
                rowsPerImage: UInt32(height)
            ),
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height))
        )

        return texture
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
        blurredShadowCache.removeAll()
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

            // Draw text (simple single-line for now)
            // For wrapped text, we would need to implement line breaking
            if textLayer.isWrapped {
                // Simple word wrapping
                drawWrappedText(ctx: ctx, text: text, x: x, y: 0,
                               maxWidth: Double(width), lineHeight: textLayer.fontSize * 1.2)
            } else {
                _ = ctx.fillText!(text, x, Double(textLayer.fontSize * 0.1))
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
                    GPUBindGroupEntry(binding: 0, resource: .buffer(uniformBuffer, offset: 0, size: UInt64(MemoryLayout<CARendererUniforms>.stride))),
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
        let white = SIMD4<Float>(1, 1, 1, 1)
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
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: currentEffectiveOpacity,
            cornerRadius: effectiveCornerRadius,
            layerSize: SIMD2<Float>(Float(width), Float(height)),
            cornerRadii: effectiveCornerRadii
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

        return "\(text)_\(width)x\(height)_\(layer.fontSize)_\(layer.alignmentMode.rawValue)_\(fontFingerprint)_\(colorHex)_\(layer.isWrapped ? "w" : "n")"
    }

    /// Returns `true` if the given Unicode scalar is in a CJK script range where
    /// a line break is permitted between any two adjacent characters.
    ///
    /// This is a deliberately narrow approximation of UAX #14 — only the
    /// ranges that cover the vast majority of the text that OpenSpriteKit /
    /// megaman will encounter (Japanese, Simplified / Traditional Chinese,
    /// Korean). It does not attempt prohibited-break rules (line-start
    /// brackets, line-end punctuation), so the result is "break-friendly".
    private static func isCJKLineBreakable(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        // CJK Unified Ideographs
        if v >= 0x4E00 && v <= 0x9FFF { return true }
        // CJK Unified Ideographs Extension A
        if v >= 0x3400 && v <= 0x4DBF { return true }
        // Hiragana
        if v >= 0x3040 && v <= 0x309F { return true }
        // Katakana
        if v >= 0x30A0 && v <= 0x30FF { return true }
        // Hangul Syllables
        if v >= 0xAC00 && v <= 0xD7AF { return true }
        return false
    }

    /// Splits a string into "line break tokens" — atomic substrings between
    /// break opportunities. ASCII whitespace separators are dropped; soft
    /// hyphens and CJK characters each produce a one-character token.
    ///
    /// Example:
    ///   "hello world"       → ["hello", "world"]
    ///   "これはテスト"       → ["こ","れ","は","テ","ス","ト"]
    ///   "hello 世界 text"   → ["hello","世","界","text"]
    private static func lineBreakTokens(in text: String) -> [String] {
        var tokens: [String] = []
        var current = ""

        func flush() {
            if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        }

        for scalar in text.unicodeScalars {
            // ASCII whitespace → break before next token
            if scalar == " " || scalar == "\t" || scalar == "\n" || scalar == "\r" {
                flush()
                continue
            }
            // Soft hyphen → break after
            if scalar.value == 0x00AD {
                current.unicodeScalars.append(scalar)
                flush()
                continue
            }
            // CJK character → break before AND after (each is its own token)
            if isCJKLineBreakable(scalar) {
                flush()
                current.unicodeScalars.append(scalar)
                flush()
                continue
            }
            // Otherwise accumulate
            current.unicodeScalars.append(scalar)
        }
        flush()

        return tokens
    }

    /// Canvas2D text width for a string.
    private func measureWidth(ctx: JSObject, _ s: String) -> Double {
        let metrics = ctx.measureText!(s)
        return metrics.width.number ?? 0
    }

    /// Breaks an overlong token into per-character chunks that each fit in
    /// `maxWidth`. Used as a fallback when a single token is wider than the
    /// wrap limit (e.g. a very long URL).
    private func breakOversizedToken(
        ctx: JSObject,
        token: String,
        maxWidth: Double
    ) -> [String] {
        var out: [String] = []
        var buffer = ""
        for char in token {
            let candidate = buffer + String(char)
            if !buffer.isEmpty && measureWidth(ctx: ctx, candidate) > maxWidth {
                out.append(buffer)
                buffer = String(char)
            } else {
                buffer = candidate
            }
        }
        if !buffer.isEmpty { out.append(buffer) }
        return out
    }

    /// Draws wrapped text on a Canvas2D context using script-aware line breaks.
    private func drawWrappedText(ctx: JSObject, text: String, x: Double, y: Double,
                                 maxWidth: Double, lineHeight: CGFloat) {
        let tokens = Self.lineBreakTokens(in: text)
        var line = ""
        var currentY = y

        for token in tokens {
            let candidate = line.isEmpty ? token : line + " " + token
            let candidateWidth = measureWidth(ctx: ctx, candidate)

            if candidateWidth > maxWidth && !line.isEmpty {
                // Commit the current line and try to place this token on a new one.
                _ = ctx.fillText!(line, x, currentY)
                currentY += Double(lineHeight)

                // Does the token itself fit on an empty line?
                if measureWidth(ctx: ctx, token) > maxWidth {
                    let chunks = breakOversizedToken(ctx: ctx, token: token, maxWidth: maxWidth)
                    // All but the last chunk each become their own line.
                    for chunk in chunks.dropLast() {
                        _ = ctx.fillText!(chunk, x, currentY)
                        currentY += Double(lineHeight)
                    }
                    line = chunks.last ?? ""
                } else {
                    line = token
                }
            } else if line.isEmpty && candidateWidth > maxWidth {
                // First token on the line is already too wide — break per-character.
                let chunks = breakOversizedToken(ctx: ctx, token: token, maxWidth: maxWidth)
                for chunk in chunks.dropLast() {
                    _ = ctx.fillText!(chunk, x, currentY)
                    currentY += Double(lineHeight)
                }
                line = chunks.last ?? ""
            } else {
                line = candidate
            }
        }

        if !line.isEmpty {
            _ = ctx.fillText!(line, x, currentY)
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

        if isWrapped, let maxWidth = maxWidth, CGFloat(measuredWidth) > maxWidth {
            // Calculate wrapped text height using the same tokenization + oversized-token
            // fallback as drawWrappedText. Using identical logic ensures that the
            // measured size matches the rendered output line-for-line (otherwise the
            // caller's texture would clip or leave blank rows).
            let tokens = Self.lineBreakTokens(in: text)
            var lineCount = 1
            var line = ""

            for token in tokens {
                let candidate = line.isEmpty ? token : line + " " + token
                let candidateWidth = measureWidth(ctx: ctx, candidate)

                if candidateWidth > Double(maxWidth) && !line.isEmpty {
                    // Commit the current line.
                    lineCount += 1

                    // Does the token itself fit on an empty line?
                    if measureWidth(ctx: ctx, token) > Double(maxWidth) {
                        let chunks = breakOversizedToken(ctx: ctx, token: token, maxWidth: Double(maxWidth))
                        // All but the last chunk each become their own line.
                        lineCount += max(0, chunks.count - 1)
                        line = chunks.last ?? ""
                    } else {
                        line = token
                    }
                } else if line.isEmpty && candidateWidth > Double(maxWidth) {
                    // First token on the line is already too wide — break per-character.
                    let chunks = breakOversizedToken(ctx: ctx, token: token, maxWidth: Double(maxWidth))
                    // lineCount already starts at 1 for the first chunk; add the rest.
                    lineCount += max(0, chunks.count - 1)
                    line = chunks.last ?? ""
                } else {
                    line = candidate
                }
            }

            return CGSize(
                width: maxWidth,
                height: CGFloat(lineHeight) * CGFloat(lineCount) * 1.2
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

                for idx in indices {
                    let point = polyline[idx]
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: SIMD2(0, 0),  // Not used for solid color
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
                    layerSize: .zero  // No SDF-based corner radius for shapes
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

                for point in strokeVertices {
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: SIMD2(0, 0),
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
                    layerSize: .zero
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
        guard let components = color.components, components.count >= 4 else {
            return SIMD4<Float>(0, 0, 0, 1)
        }
        return SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
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
            cornerRadii: gradientLayer.cornerRadiiComponents
        )

        // Extract gradient colors
        var gradientColors: [SIMD4<Float>] = []
        for colorAny in colors.prefix(kMaxGradientStops) {
            if let cgColor = colorAny as? CGColor,
               let components = cgColor.components,
               components.count >= 4 {
                gradientColors.append(SIMD4<Float>(
                    Float(components[0]),
                    Float(components[1]),
                    Float(components[2]),
                    Float(components[3])
                ))
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

    /// Pre-renders shadows with 2-pass Gaussian blur before the main render pass.
    ///
    /// This method finds the first layer needing a shadow and renders it with blur.
    /// Currently supports one shadow per frame due to texture sharing constraints.
    /// For multiple shadows, a texture pool would be needed.
    private func prerenderShadows(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        // Fast-path: skip the recursive `findFirstShadowLayer` walk when no
        // descendant carries a shadow contribution. The counter mirrors model
        // state (see CALayer's subtree counters), so animations that animate
        // shadowOpacity from a model value of 0 are not detected here.
        guard rootLayer._subtreeShadowCount > 0 else { return }

        guard let device = device,
              let pipeline = pipeline,
              let preRenderVertexBuffer = preRenderVertexBuffer,
              let preRenderUniformBuffer = preRenderUniformBuffer,
              let preRenderBindGroup = preRenderBindGroup,
              let shadowMaskTexture = shadowMaskTexture,
              let shadowBlurTexture = shadowBlurTexture,
              let shadowBlurHorizontalPipeline = shadowBlurHorizontalPipeline,
              let shadowBlurVerticalPipeline = shadowBlurVerticalPipeline,
              let blurHorizontalBindGroup = blurHorizontalBindGroup,
              let blurVerticalBindGroup = blurVerticalBindGroup,
              let blurUniformBuffer = blurUniformBuffer else { return }

        // Find first layer with shadow to pre-render
        guard let (shadowLayer, _, shadowParentMatrix, shadowEffectiveOpacity) = findFirstShadowLayer(rootLayer, parentMatrix: projectionMatrix) else {
            return
        }

        // R3.7 (PERFORMANCE_DESIGN.md §5.5): when the contributor's subtree
        // is clean and we already have a blurred texture cached, skip the
        // mask-extraction + 2-pass blur. `shadowMaskTexture` still holds
        // last frame's blurred result because nothing else writes to it.
        let shadowLayerID = ObjectIdentifier(shadowLayer)
        if RasterizationDecisions.canReusePrerenderCache(
            contributorLayer: shadowLayer,
            hasCachedTexture: blurredShadowCache[shadowLayerID] != nil
        ) {
            hasPrerenderredShadow = true
            shadowsPrerendered = true
            return
        }

        let presentationLayer = shadowLayer._renderTimePresentation()
        let shadowRadius = presentationLayer.shadowRadius
        let shadowOffset = presentationLayer.shadowOffset

        // Calculate expanded bounds for shadow (includes blur radius)
        let expandedWidth = presentationLayer.bounds.width + shadowRadius * 4
        let expandedHeight = presentationLayer.bounds.height + shadowRadius * 4

        // Step 1: Render layer shape to shadow mask texture
        guard let shadowMaskTextureView = shadowMaskTextureView,
              let shadowBlurTextureView = shadowBlurTextureView else { return }
        let maskRenderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: shadowMaskTextureView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        // Apply shadow offset in parent coordinate space (not layer's local space).
        // In CoreAnimation, shadow offset direction is constant regardless of layer rotation.
        let shadowOffsetParentMatrix = shadowParentMatrix * Matrix4x4(translation: SIMD3<Float>(
            Float(shadowOffset.width), Float(shadowOffset.height), 0
        ))
        let finalMatrix = presentationLayer.modelMatrix(parentMatrix: shadowOffsetParentMatrix)

        var maskUniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: shadowEffectiveOpacity,
            cornerRadius: Float(presentationLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height)),
            cornerRadii: presentationLayer.cornerRadiiComponents
        )

        // Write mask uniforms to dedicated pre-render buffer
        let maskUniformData = createFloat32Array(from: &maskUniforms)
        device.queue.writeBuffer(preRenderUniformBuffer, bufferOffset: 0, data: maskUniformData)

        // R3.6 (PERFORMANCE_DESIGN.md §5.4): when `shadowPath` is set, replace
        // the bounds-rect silhouette with a tessellated path. The mvp matrix
        // is unchanged because both rect and path coordinates live in
        // layer-local space.
        let whiteColor = SIMD4<Float>(1, 1, 1, 1)
        var maskVertices: [CARendererVertex] = []
        if RasterizationDecisions.useShadowPathFastPath(for: presentationLayer),
           let shadowPath = presentationLayer.shadowPath {
            let polylines = flattenPath(shadowPath)
            for polyline in polylines {
                guard polyline.count >= 3 else { continue }
                let indices = triangulatePolygon(polyline)
                for idx in indices {
                    let point = polyline[idx]
                    maskVertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: SIMD2(0, 0),
                        color: whiteColor
                    ))
                }
            }
        }

        if maskVertices.isEmpty {
            // Bounds-rect silhouette fallback (R3.6 disabled or empty path).
            maskVertices = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: whiteColor),
                CARendererVertex(position: SIMD2(Float(presentationLayer.bounds.width), 0), texCoord: SIMD2(1, 0), color: whiteColor),
                CARendererVertex(position: SIMD2(0, Float(presentationLayer.bounds.height)), texCoord: SIMD2(0, 1), color: whiteColor),
                CARendererVertex(position: SIMD2(Float(presentationLayer.bounds.width), 0), texCoord: SIMD2(1, 0), color: whiteColor),
                CARendererVertex(position: SIMD2(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height)), texCoord: SIMD2(1, 1), color: whiteColor),
                CARendererVertex(position: SIMD2(0, Float(presentationLayer.bounds.height)), texCoord: SIMD2(0, 1), color: whiteColor),
            ]
        }

        let maskVertexCount = UInt32(maskVertices.count)
        let maskVertexData = createFloat32Array(from: &maskVertices)
        device.queue.writeBuffer(preRenderVertexBuffer, bufferOffset: 0, data: maskVertexData)

        // Use shadowMaskPipeline (no depth/stencil) since the mask pass has no depth attachment
        maskRenderPass.setPipeline(shadowMaskPipeline ?? pipeline)
        maskRenderPass.setBindGroup(0, bindGroup: preRenderBindGroup, dynamicOffsets: [0])
        maskRenderPass.setVertexBuffer(0, buffer: preRenderVertexBuffer, offset: 0)
        maskRenderPass.draw(vertexCount: maskVertexCount)
        maskRenderPass.end()

        // Step 2: Horizontal blur (shadowMaskTexture → shadowBlurTexture)
        var blurUniforms = BlurUniforms(
            texelSize: SIMD2<Float>(1.0 / Float(size.width), 1.0 / Float(size.height)),
            blurRadius: Float(shadowRadius) * 0.5
        )
        let blurUniformData = createFloat32Array(from: &blurUniforms)
        device.queue.writeBuffer(blurUniformBuffer, bufferOffset: 0, data: blurUniformData)

        let hBlurPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: shadowBlurTextureView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        hBlurPass.setPipeline(shadowBlurHorizontalPipeline)
        hBlurPass.setBindGroup(0, bindGroup: blurHorizontalBindGroup)
        hBlurPass.draw(vertexCount: 6)
        hBlurPass.end()

        // Step 3: Vertical blur (shadowBlurTexture → shadowMaskTexture)
        let vBlurPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: shadowMaskTextureView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        vBlurPass.setPipeline(shadowBlurVerticalPipeline)
        vBlurPass.setBindGroup(0, bindGroup: blurVerticalBindGroup)
        vBlurPass.draw(vertexCount: 6)
        vBlurPass.end()

        // Mark that shadow was pre-rendered
        hasPrerenderredShadow = true
        shadowsPrerendered = true

        // R3.7: remember which contributor produced the pixels currently
        // sitting in `shadowMaskTexture`. Next frame's reuse check looks
        // for this entry. The texture handle is shared/recreated on
        // `createShadowTextures` (resize); cache is cleared there.
        blurredShadowCache.removeAll(keepingCapacity: true)
        blurredShadowCache[shadowLayerID] = shadowMaskTexture
    }

    /// Pre-renders layers with supported filters into an offscreen texture.
    ///
    /// Currently supports one filtered layer per frame due to texture sharing constraints.
    private func prerenderFilteredLayers(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        // Fast-path: skip the recursive `findFirstFilteredLayer` walk when no
        // descendant has a non-empty `filters` array. The counter checks for
        // non-empty content; unsupported filter types are still filtered later
        // by `supportedFilterOperations`.
        guard rootLayer._subtreeFilterCount > 0 else { return }

        guard let pipeline = pipeline,
              let depthTexture = depthTexture,
              let filterSourceTexture = filterSourceTexture,
              let filterBlurTexture = filterBlurTexture,
              let filterResultTexture = filterResultTexture else { return }

        // Find first layer with supported filters to pre-render.
        guard let (filteredLayer, parentMatrix) = findFirstFilteredLayer(rootLayer, parentMatrix: projectionMatrix) else {
            return
        }

        let presentationLayer = filteredLayer._renderTimePresentation()
        let operations = presentationLayer.supportedFilterOperations
        guard !operations.isEmpty else { return }
        guard let depthTextureView = depthTextureView,
              let filterSourceTextureView = filterSourceTextureView else { return }

        // Step 1: Render the filtered layer subtree into the source texture without applying
        // the layer's own filter. The composite pass will apply the root opacity later.
        let contentRenderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: filterSourceTextureView,
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

        filterPrerenderRootLayer = filteredLayer
        renderLayer(filteredLayer, renderPass: contentRenderPass, parentMatrix: parentMatrix)
        filterPrerenderRootLayer = nil

        contentRenderPass.end()

        var currentTexture = filterSourceTexture
        var useResultTextureAsOutput = true

        for operation in operations {
            let outputTexture = useResultTextureAsOutput ? filterResultTexture : filterSourceTexture

            switch operation {
            case let .gaussianBlur(radius):
                guard radius > 0 else { continue }
                applyBlurFilter(
                    inputTexture: currentTexture,
                    intermediateTexture: filterBlurTexture,
                    outputTexture: outputTexture,
                    radius: radius,
                    encoder: encoder
                )
            case .brightness, .contrast, .saturation, .colorInvert:
                applyColorFilter(
                    operation,
                    inputTexture: currentTexture,
                    outputTexture: outputTexture,
                    encoder: encoder
                )
            }

            currentTexture = outputTexture
            useResultTextureAsOutput.toggle()
        }

        // Mark that filter was pre-rendered
        hasPrerenderredFilter = true
        prerenderredFilterLayer = filteredLayer
        prerenderredFilterTexture = currentTexture
        prerenderredFilterBlurRadius = presentationLayer.totalBlurRadius
    }

    // MARK: - Rasterized Layer Pre-render (R3.2 / R3.3)

    /// Walks the tree and captures every `shouldRasterize` subtree into
    /// its own offscreen texture, populating `prerasterizedTextures` so
    /// the main pass can composite the captured pixels as a quad.
    ///
    /// On a cache hit the existing texture is reused — no GPU work — and
    /// `lookup` updates the entry's `lastUsedFrame` to keep the idle
    /// eviction pass honest. On a miss the renderer allocates a fresh
    /// canvas-sized texture, redirects `renderLayer` into it with
    /// `rasterizePrerenderRootLayer` set so this same layer's composite
    /// branch in `renderLayer` is suppressed during the capture pass,
    /// and inserts the new entry. Capture clears with α = 1.0 (R3.3).
    private func prerenderRasterizedLayers(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        guard let device = device,
              let pipeline = pipeline,
              let depthTextureView = depthTextureView,
              let cache = rasterizationCache else { return }

        // Frame token already bumped at the top of `render`; use it for
        // both lookup `lastUsedFrame` updates and insert tagging.
        let frameToken = CALayer._currentFrameToken

        collectAndCaptureRasterizedLayers(
            rootLayer,
            parentMatrix: projectionMatrix,
            device: device,
            pipeline: pipeline,
            depthTextureView: depthTextureView,
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
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        depthTextureView: GPUTextureView,
        cache: RasterizationCache<GPUTexture>,
        encoder: GPUCommandEncoder,
        frameToken: UInt64
    ) {
        let presentationLayer = layer._renderTimePresentation()
        guard !presentationLayer.isHidden, presentationLayer.opacity > 0 else {
            return
        }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        if presentationLayer.shouldRasterize {
            captureRasterizedLayer(
                layer,
                device: device,
                pipeline: pipeline,
                depthTextureView: depthTextureView,
                cache: cache,
                encoder: encoder,
                frameToken: frameToken
            )
        }

        if let sublayers = layer.sublayers, !sublayers.isEmpty {
            let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)
            for sublayer in sublayers {
                collectAndCaptureRasterizedLayers(
                    sublayer,
                    parentMatrix: sublayerMatrix,
                    device: device,
                    pipeline: pipeline,
                    depthTextureView: depthTextureView,
                    cache: cache,
                    encoder: encoder,
                    frameToken: frameToken
                )
            }
        }
    }

    /// Captures or reuses the rasterized texture for one
    /// `shouldRasterize` layer.
    ///
    /// Per PERFORMANCE_DESIGN.md §5.2 the offscreen texture is sized to
    /// `bounds.size × rasterizationScale` (the layer's own pixel grid),
    /// not the canvas — so the composite path can place it as a partial
    /// quad at the layer's transform without burning canvas-sized memory
    /// per cached entry. The capture pass uses a bounds-local
    /// orthographic projection so the layer's own `position`, `transform`
    /// and `anchorPoint` are *excluded* from the bake — those land at
    /// composite time. `renderLayer` recognises the capture root via
    /// `rasterizePrerenderRootLayer` and skips its `modelMatrix`
    /// computation accordingly.
    private func captureRasterizedLayer(
        _ layer: CALayer,
        device: GPUDevice,
        pipeline: GPURenderPipeline,
        depthTextureView: GPUTextureView,
        cache: RasterizationCache<GPUTexture>,
        encoder: GPUCommandEncoder,
        frameToken: UInt64
    ) {
        let key = RasterizationCacheKey(ObjectIdentifier(layer))
        let contentBoundsHash = rasterizationContentBoundsHash(for: layer)

        // Cache lookup updates `lastUsedFrame` on a hit so the idle
        // eviction pass treats it as live.
        if let entry = cache.lookup(key, atFrame: frameToken),
           RasterizationDecisions.canReuseRasterizedTexture(
               layer: layer,
               cached: entry,
               currentContentBoundsHash: contentBoundsHash) {
            prerasterizedTextures[ObjectIdentifier(layer)] = entry.texture
            return
        }

        // Miss — allocate a layer-sized texture (`bounds × scale`) and
        // render the subtree into it under a bounds-local projection.
        let presentationLayer = layer._renderTimePresentation()
        let bounds = presentationLayer.bounds
        let scale = max(1.0, CGFloat(presentationLayer.rasterizationScale))
        let pixelWidth = max(1, Int((bounds.width * scale).rounded(.up)))
        let pixelHeight = max(1, Int((bounds.height * scale).rounded(.up)))
        guard bounds.width > 0, bounds.height > 0 else { return }

        let captureTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(pixelWidth), height: UInt32(pixelHeight)),
            format: preferredFormat,
            usage: [.renderAttachment, .textureBinding]
        ))
        let captureView = captureTexture.createView()

        // R3.3 / R3.4a: clear α is 1.0 regardless of `layer.opacity`; the
        // composite pass applies `layer.opacity` at draw time. With the
        // texture now matching the layer's own bounds, the layer's
        // backgroundColor / sublayers fill the texture and this clear
        // value only shows through where the layer doesn't draw at all
        // (a flat-rasterized `shouldRasterize` layer is treated as an
        // opaque image, mirroring CoreAnimation's WWDC 2014 #419 model).
        let clearAlpha = RasterizationDecisions.captureClearAlpha()
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
                view: depthTextureView,
                depthClearValue: 1.0,
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
            left: Float(bounds.minX),
            right: Float(bounds.maxX),
            bottom: Float(bounds.minY),
            top: Float(bounds.maxY),
            near: -1000,
            far: 1000
        )

        rasterizePrerenderRootLayer = layer
        renderLayer(layer, renderPass: capturePass, parentMatrix: captureProjection)
        rasterizePrerenderRootLayer = nil

        capturePass.end()

        // Insert the entry. The cache enforces the byte budget only
        // when `evictToBudget` is called (post-submit), so an oversized
        // single insert is allowed to land here.
        let pixelSize = CGSize(width: CGFloat(pixelWidth), height: CGFloat(pixelHeight))
        cache.insert(
            key,
            texture: captureTexture,
            pixelSize: pixelSize,
            contentBoundsHash: contentBoundsHash,
            atFrame: frameToken
        )
        prerasterizedTextures[ObjectIdentifier(layer)] = captureTexture
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
    private func rasterizationContentBoundsHash(for layer: CALayer) -> Int {
        var hasher = Hasher()
        hasher.combine(layer.bounds.origin.x)
        hasher.combine(layer.bounds.origin.y)
        hasher.combine(layer.bounds.size.width)
        hasher.combine(layer.bounds.size.height)
        hasher.combine(layer.rasterizationScale)
        return hasher.finalize()
    }

    /// Composites a captured rasterization texture as a quad placed at
    /// the layer's transform, sized to the layer's bounds, with the
    /// layer's *current* opacity (R3.3 composite path).
    ///
    /// Reuses the `texturedPipeline` (same shader path as `contents`
    /// rendering) so that R3.5 `isOpaque` / blending and stencil-mask
    /// state are honoured by `selectTexturedPipeline`. The quad
    /// vertices are emitted in normalised layer-bounds coordinates
    /// `[0, 1]`; the MVP matrix scales them by `bounds.size` and
    /// applies the layer's `modelMatrix` to land at the right screen
    /// position. Texture V is flipped to bridge Y-up world / Y-down
    /// texture rows (mirrors `renderContentsLayer`).
    private func renderRasterizedLayerComposite(
        _ layer: CALayer,
        presentationLayer: CALayer,
        texture: GPUTexture,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        let boundsWidth = presentationLayer.bounds.width
        let boundsHeight = presentationLayer.bounds.height
        guard boundsWidth > 0, boundsHeight > 0 else { return }

        // Composite opacity = parent-stack effective × this layer's
        // current opacity. Capture deliberately baked the layer at 1.0
        // (see `renderLayer`'s `isCaptureRoot` branch); reapply here.
        let composite = currentEffectiveOpacity * RasterizationDecisions.compositeOpacity(for: presentationLayer)

        // Quad in normalised layer-bounds space. UVs V-flip to convert
        // between Y-up world coords and Y-down texture rows — same
        // convention used by `renderContentsLayer`.
        let white = SIMD4<Float>(1, 1, 1, 1)
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

        // Scale the [0, 1] quad to bounds.size, then apply the layer's
        // own modelMatrix to position/rotate/scale it into the world.
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(boundsWidth), 0, 0, 0),
            SIMD4<Float>(0, Float(boundsHeight), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let finalMatrix = modelMatrix * scaleMatrix

        // cornerRadius is already baked into the captured texture by
        // the capture pass — passing 0 here avoids double-applying it.
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: composite,
            cornerRadius: 0,
            layerSize: SIMD2<Float>(Float(boundsWidth), Float(boundsHeight)),
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
        // The `.layer(...)` tag keeps this entry from aliasing a CGImage
        // entry whose `ObjectIdentifier` happens to match this layer's
        // (post-dealloc) heap address.
        let texturedBindGroup = cachedTexturedBindGroup(
            cacheKey: .layer(ObjectIdentifier(layer)),
            gpuTexture: texture,
            device: device,
            layout: texturedBindGroupLayout,
            sampler: textureSampler,
            uniformBuffer: uniformBuffer,
            uniformStride: UInt64(MemoryLayout<TexturedUniforms>.stride)
        )

        // R3.5 / mask-stencil aware pipeline selection.
        if let selected = selectTexturedPipeline(for: presentationLayer) {
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
        encoder: GPUCommandEncoder
    ) {
        guard let device = device,
              let shadowBlurHorizontalPipeline = shadowBlurHorizontalPipeline,
              let shadowBlurVerticalPipeline = shadowBlurVerticalPipeline,
              let blurUniformBuffer = blurUniformBuffer,
              let depthTextureView = depthTextureView else { return }

        // Resolve cached views and bind groups by texture identity. The
        // ping-pong loop in `prerenderFilteredLayers` only ever passes the
        // three persistent filter textures here, so no fallback path is
        // needed — if the lookup misses we drop the operation rather than
        // silently allocating a per-frame view (which would defeat the cache
        // and hide a real wiring bug behind a slow path).
        guard let inputBindGroup = applyBlurBindGroup(for: inputTexture),
              let intermediateView = filterTextureView(for: intermediateTexture),
              let intermediateBindGroup = applyBlurBindGroup(for: intermediateTexture),
              let outputView = filterTextureView(for: outputTexture) else { return }

        var blurUniforms = BlurUniforms(
            texelSize: SIMD2<Float>(1.0 / Float(size.width), 1.0 / Float(size.height)),
            blurRadius: Float(radius)
        )
        let blurUniformData = createFloat32Array(from: &blurUniforms)
        device.queue.writeBuffer(blurUniformBuffer, bufferOffset: 0, data: blurUniformData)

        let hBlurPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: intermediateView,
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

        hBlurPass.setPipeline(shadowBlurHorizontalPipeline)
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

        vBlurPass.setPipeline(shadowBlurVerticalPipeline)
        vBlurPass.setBindGroup(0, bindGroup: intermediateBindGroup)
        vBlurPass.draw(vertexCount: 6)
        vBlurPass.end()
    }

    /// Returns the cached view for one of the persistent filter textures.
    /// Reference equality is used because `GPUTexture` is a class and the
    /// caller always passes one of the three textures created in
    /// `createFilterTextures`.
    private func filterTextureView(for texture: GPUTexture) -> GPUTextureView? {
        if texture === filterSourceTexture { return filterSourceTextureView }
        if texture === filterBlurTexture { return filterBlurTextureView }
        if texture === filterResultTexture { return filterResultTextureView }
        return nil
    }

    /// Returns the cached blur-pass bind group for one of the persistent
    /// filter textures. See `filterTextureView(for:)`.
    private func applyBlurBindGroup(for texture: GPUTexture) -> GPUBindGroup? {
        if texture === filterSourceTexture { return applyBlurFromSourceBindGroup }
        if texture === filterBlurTexture { return applyBlurFromBlurBindGroup }
        if texture === filterResultTexture { return applyBlurFromResultBindGroup }
        return nil
    }

    private func applyColorFilter(
        _ operation: CAFilterOperation,
        inputTexture: GPUTexture,
        outputTexture: GPUTexture,
        encoder: GPUCommandEncoder
    ) {
        guard let device = device,
              let depthTextureView = depthTextureView,
              let filterCompositePipeline = filterCompositePipeline,
              let filterCompositeUniformBuffer = filterCompositeUniformBuffer,
              let blurSampler = blurSampler,
              var uniforms = filterCompositeUniforms(for: operation) else { return }
        // inputTexture / outputTexture are always one of the persistent
        // filter textures (see prerenderFilteredLayers); use the cached views.
        guard let inputView = filterTextureView(for: inputTexture),
              let outputView = filterTextureView(for: outputTexture) else { return }

        let filterUniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(filterCompositeUniformBuffer, bufferOffset: 0, data: filterUniformData)

        let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: filterCompositePipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(filterCompositeUniformBuffer, offset: 0, size: UInt64(MemoryLayout<FilterCompositeUniforms>.stride))),
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

        renderPass.setPipeline(filterCompositePipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup)
        renderPass.draw(vertexCount: 6)
        renderPass.end()
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

    /// Finds the first layer in the hierarchy that has supported filters.
    private func findFirstFilteredLayer(
        _ layer: CALayer,
        parentMatrix: Matrix4x4
    ) -> (layer: CALayer, parentMatrix: Matrix4x4)? {
        let presentationLayer = layer._renderTimePresentation()

        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else {
            return nil
        }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        if !presentationLayer.supportedFilterOperations.isEmpty {
            return (layer, parentMatrix)
        }

        // Recursively check sublayers in render order so the chosen pre-render target
        // matches the first filtered layer that will actually be composited.
        if let sublayers = layer.sublayers {
            let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)

            for sublayer in layer.sortedSublayers() {
                if let result = findFirstFilteredLayer(sublayer, parentMatrix: sublayerMatrix) {
                    return result
                }
            }
        }

        return nil
    }

    /// Finds the first layer in the hierarchy that has a visible shadow.
    /// Accumulates effective opacity through the hierarchy for correct shadow rendering.
    private func findFirstShadowLayer(
        _ layer: CALayer,
        parentMatrix: Matrix4x4,
        parentOpacity: Float = 1.0
    ) -> (layer: CALayer, matrix: Matrix4x4, parentMatrix: Matrix4x4, effectiveOpacity: Float)? {
        let presentationLayer = layer._renderTimePresentation()

        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else {
            return nil
        }

        let effectiveOpacity = parentOpacity * presentationLayer.opacity
        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Check if this layer has a shadow
        if presentationLayer.shadowOpacity > 0 && presentationLayer.shadowColor != nil {
            return (layer, modelMatrix, parentMatrix, effectiveOpacity)
        }

        // Recursively check sublayers in render order (sorted by zPosition)
        // to match the order in which shadows are actually drawn.
        if let sublayers = layer.sublayers {
            let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)

            for sublayer in layer.sortedSublayers() {
                if let result = findFirstShadowLayer(sublayer, parentMatrix: sublayerMatrix, parentOpacity: effectiveOpacity) {
                    return result
                }
            }
        }

        return nil
    }

    /// Renders the shadow for a layer using the pre-blurred shadow texture.
    ///
    /// If the shadow was pre-rendered with 2-pass Gaussian blur, this composites
    /// the blurred texture. Otherwise, falls back to a simplified shadow approximation.
    private func renderLayerShadow(
        _ layer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        parentMatrix: Matrix4x4
    ) {
        guard let shadowColor = layer.shadowColor,
              let pipeline = pipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        // Get shadow properties
        let shadowOpacity = layer.shadowOpacity
        let shadowOffset = layer.shadowOffset
        let shadowRadius = layer.shadowRadius

        // Get shadow color components
        // Multiply shadow alpha by effective opacity so shadows respect parent opacity
        let effectiveOpacity = currentEffectiveOpacity
        let colorComponents: SIMD4<Float>
        if let components = shadowColor.components, components.count >= 3 {
            colorComponents = SIMD4<Float>(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                (components.count > 3 ? Float(components[3]) * shadowOpacity : shadowOpacity) * effectiveOpacity
            )
        } else {
            colorComponents = SIMD4<Float>(0, 0, 0, shadowOpacity * effectiveOpacity)
        }

        // If shadow was pre-rendered with blur, use the textured composite pipeline
        if hasPrerenderredShadow,
           let shadowMaskTextureView = shadowMaskTextureView,
           let shadowCompositePipeline = shadowCompositePipeline,
           let blurSampler = blurSampler {

            // Create shadow uniforms for the composite shader
            // SpriteKit/CoreAnimation coordinate system (Y+ up)
            var shadowUniforms = ShadowUniforms(
                mvpMatrix: Matrix4x4.orthographic(
                    left: 0, right: Float(size.width),
                    bottom: 0, top: Float(size.height),
                    near: -1000, far: 1000
                ),
                shadowColor: colorComponents,
                shadowOffset: SIMD2<Float>(Float(shadowOffset.width), Float(shadowOffset.height)),
                layerSize: SIMD2<Float>(Float(size.width), Float(size.height))
            )

            // Use pre-allocated shadow composite uniform buffer
            guard let shadowCompositeUniformBuffer = shadowCompositeUniformBuffer else { return }

            let shadowUniformData = createFloat32Array(from: &shadowUniforms)
            device.queue.writeBuffer(shadowCompositeUniformBuffer, bufferOffset: 0, data: shadowUniformData)

            // Create shadow composite bind group with the blurred texture
            let compositeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                layout: shadowCompositePipeline.getBindGroupLayout(index: 0),
                entries: [
                    GPUBindGroupEntry(binding: 0, resource: .buffer(shadowCompositeUniformBuffer, offset: 0, size: UInt64(MemoryLayout<ShadowUniforms>.stride))),
                    GPUBindGroupEntry(binding: 1, resource: .textureView(shadowMaskTextureView)),
                    GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
                ]
            ))

            // Full-screen quad vertices for compositing the shadow texture
            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: colorComponents),
                CARendererVertex(position: SIMD2(Float(size.width), 0), texCoord: SIMD2(1, 0), color: colorComponents),
                CARendererVertex(position: SIMD2(0, Float(size.height)), texCoord: SIMD2(0, 1), color: colorComponents),
                CARendererVertex(position: SIMD2(Float(size.width), 0), texCoord: SIMD2(1, 0), color: colorComponents),
                CARendererVertex(position: SIMD2(Float(size.width), Float(size.height)), texCoord: SIMD2(1, 1), color: colorComponents),
                CARendererVertex(position: SIMD2(0, Float(size.height)), texCoord: SIMD2(0, 1), color: colorComponents),
            ]

            guard let allocation = allocateVertices(count: vertices.count) else { return }
            let (vertexOffset, _) = allocation

            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

            renderPass.setPipeline(maskNestingDepth > 0 ? (shadowCompositeStencilPipeline ?? shadowCompositePipeline) : shadowCompositePipeline)
            renderPass.setBindGroup(0, bindGroup: compositeBindGroup)
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)

            // Mark that we've consumed the pre-rendered shadow
            hasPrerenderredShadow = false
            return
        }

        // Fallback: Simplified shadow without blur (for subsequent shadows or when blur is unavailable)
        // Calculate shadow size (larger than layer for blur effect)
        let expandedWidth = layer.bounds.width + shadowRadius * 2
        let expandedHeight = layer.bounds.height + shadowRadius * 2

        // Apply shadow offset in parent coordinate space (not layer's local space).
        // In CoreAnimation, shadow offset direction is constant regardless of layer rotation.
        let shadowOffsetParentMatrix = parentMatrix * Matrix4x4(translation: SIMD3<Float>(
            Float(shadowOffset.width - shadowRadius),
            Float(shadowOffset.height - shadowRadius),
            0
        ))
        let shadowModelMatrix = layer.modelMatrix(parentMatrix: shadowOffsetParentMatrix)

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(expandedWidth), 0, 0, 0),
            SIMD4<Float>(0, Float(expandedHeight), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = shadowModelMatrix * scaleMatrix

        // Use larger corner radius to simulate blur
        let effectiveCornerRadius = Float(layer.cornerRadius + shadowRadius * 0.5)

        // Create shadow vertices
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: colorComponents),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: colorComponents),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: colorComponents),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: colorComponents),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: colorComponents),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: colorComponents),
        ]

        guard let allocation = allocateVertices(count: vertices.count) else { return }
        let (vertexOffset, layerIndex) = allocation

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: effectiveOpacity,
            cornerRadius: effectiveCornerRadius,
            layerSize: SIMD2<Float>(Float(expandedWidth), Float(expandedHeight))
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

        renderPass.setPipeline(pipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    /// Renders a filtered layer by compositing the pre-rendered filter texture.
    ///
    /// This method draws a full-screen quad textured with the filtered layer content
    /// from the filter pre-rendering pass.
    private func renderFilteredLayerComposite(
        _ layer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let filteredTexture = prerenderredFilterTexture ?? filterResultTexture,
              let filterCompositePipeline = filterCompositePipeline,
              let blurSampler = blurSampler,
              let filterCompositeUniformBuffer = filterCompositeUniformBuffer else { return }

        var filterUniforms = FilterCompositeUniforms(
            opacity: currentEffectiveOpacity,
            filterType: 0,
            parameter0: 0,
            parameter1: 0
        )
        let filterUniformData = createFloat32Array(from: &filterUniforms)
        device.queue.writeBuffer(filterCompositeUniformBuffer, bufferOffset: 0, data: filterUniformData)

        // Create composite bind group with the filtered texture.
        let compositeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: filterCompositePipeline.getBindGroupLayout(index: 0),
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(filterCompositeUniformBuffer, offset: 0, size: UInt64(MemoryLayout<FilterCompositeUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(filteredTexture.createView())),
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
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let presentationLayer = layer._renderTimePresentation()

        guard let sublayers = layer.sublayers else { return }

        // Apply the CATransformLayer's own transform (but not its content)
        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Use sublayerMatrix helper to apply sublayerTransform and bounds.origin offset
        let sublayerMatrix = presentationLayer.sublayerMatrix(modelMatrix: modelMatrix)

        // Render sublayers in array order.
        // The depth buffer handles z-ordering correctly - no pre-sorting needed.
        // Sublayers sorted by zPosition (painter's algorithm, back-to-front).
        for sublayer in layer.sortedSublayers() {
            self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
        }
    }

    // MARK: - CAEmitterLayer Rendering

    /// Last update time for particle simulation.
    private var lastParticleUpdateTime: CFTimeInterval = 0

    /// Random number generator state for particles.
    private var particleRandomSeed: UInt32 = 12345

    /// Tracks the last seed used to detect seed changes.
    private var lastEmitterSeed: UInt32 = 0

    /// Whether the random seed has been initialized.
    private var randomSeedInitialized: Bool = false

    /// Generates a random float in [0, 1).
    private func randomFloat() -> Float {
        particleRandomSeed = particleRandomSeed &* 1103515245 &+ 12345
        return Float(particleRandomSeed % 65536) / 65536.0
    }

    /// Generates a random float in [-1, 1).
    private func randomSignedFloat() -> Float {
        return randomFloat() * 2.0 - 1.0
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
        _ emitterLayer: CAEmitterLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let emitterCells = emitterLayer.emitterCells, !emitterCells.isEmpty,
              let pipeline = pipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        // Initialize or update random seed only when seed changes
        if !randomSeedInitialized || emitterLayer.seed != lastEmitterSeed {
            particleRandomSeed = emitterLayer.seed
            lastEmitterSeed = emitterLayer.seed
            randomSeedInitialized = true
        }

        // Calculate delta time
        let currentTime = CACurrentMediaTime()
        let deltaTime: Float
        if lastParticleUpdateTime > 0 {
            deltaTime = min(Float(currentTime - lastParticleUpdateTime), 0.1)
        } else {
            deltaTime = 1.0 / 60.0
        }
        lastParticleUpdateTime = currentTime

        // Update existing particles
        for i in 0..<activeParticles.count {
            activeParticles[i].update(deltaTime: deltaTime)
        }

        // Remove dead particles
        activeParticles.removeAll { !$0.isAlive }

        // Spawn new particles
        for cell in emitterCells where cell.isEnabled {
            let birthRate = cell.birthRate * emitterLayer.birthRate
            let particlesToSpawn = Int(birthRate * deltaTime + 0.5)

            for _ in 0..<particlesToSpawn {
                guard activeParticles.count < Self.maxParticles else { break }

                var particle = EmitterParticle()
                particle.position = calculateEmissionPosition(
                    shape: emitterLayer.emitterShape,
                    position: emitterLayer.emitterPosition,
                    zPosition: emitterLayer.emitterZPosition,
                    size: emitterLayer.emitterSize,
                    depth: emitterLayer.emitterDepth
                )

                // Calculate velocity
                let angle = Float(cell.emissionLongitude) + randomSignedFloat() * Float(cell.emissionRange)
                let velocity = Float(cell.velocity + CGFloat(randomSignedFloat()) * CGFloat(cell.velocityRange)) * emitterLayer.velocity
                particle.velocity = SIMD3(velocity * cos(angle), velocity * sin(angle), 0)
                particle.acceleration = SIMD3(Float(cell.xAcceleration), Float(cell.yAcceleration), Float(cell.zAcceleration))

                // Set lifetime
                let lifetime = cell.lifetime + randomSignedFloat() * cell.lifetimeRange
                particle.lifetime = lifetime * emitterLayer.lifetime
                particle.maxLifetime = particle.lifetime

                // Set scale
                particle.scale = Float(cell.scale + CGFloat(randomSignedFloat()) * CGFloat(cell.scaleRange)) * emitterLayer.scale
                particle.scaleSpeed = Float(cell.scaleSpeed)

                // Set rotation
                particle.rotationSpeed = Float(cell.spin + CGFloat(randomSignedFloat()) * CGFloat(cell.spinRange)) * emitterLayer.spin

                // Set color
                if let cellColor = cell.color, let components = cellColor.components, components.count >= 3 {
                    particle.color = SIMD4(
                        Float(components[0]) + randomSignedFloat() * cell.redRange,
                        Float(components[1]) + randomSignedFloat() * cell.greenRange,
                        Float(components[2]) + randomSignedFloat() * cell.blueRange,
                        (components.count > 3 ? Float(components[3]) : 1.0) + randomSignedFloat() * cell.alphaRange
                    )
                } else {
                    particle.color = SIMD4(1, 1, 1, 1)
                }

                particle.colorSpeed = SIMD4(cell.redSpeed, cell.greenSpeed, cell.blueSpeed, cell.alphaSpeed)
                particle.isAlive = true
                activeParticles.append(particle)
            }
        }

        // Render particles
        for particle in activeParticles where particle.isAlive {
            let scale = particle.scale * 20

            // Apply rotation to vertices
            let rotation = particle.rotation
            let p0 = rotatePoint(SIMD2(0, 0), angle: rotation)
            let p1 = rotatePoint(SIMD2(1, 0), angle: rotation)
            let p2 = rotatePoint(SIMD2(0, 1), angle: rotation)
            let p3 = rotatePoint(SIMD2(1, 1), angle: rotation)

            var vertices: [CARendererVertex] = [
                CARendererVertex(position: p0, texCoord: SIMD2(0, 0), color: particle.color),
                CARendererVertex(position: p1, texCoord: SIMD2(1, 0), color: particle.color),
                CARendererVertex(position: p2, texCoord: SIMD2(0, 1), color: particle.color),
                CARendererVertex(position: p1, texCoord: SIMD2(1, 0), color: particle.color),
                CARendererVertex(position: p3, texCoord: SIMD2(1, 1), color: particle.color),
                CARendererVertex(position: p2, texCoord: SIMD2(0, 1), color: particle.color),
            ]

            guard let allocation = allocateVertices(count: vertices.count) else { break }
            let (vertexOffset, layerIndex) = allocation

            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(scale, 0, 0, 0),
                SIMD4<Float>(0, scale, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let translateMatrix = Matrix4x4(columns: (
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(particle.position.x, particle.position.y, particle.position.z, 1)
            ))

            let particleMatrix = modelMatrix * translateMatrix * scaleMatrix

            var uniforms = CARendererUniforms(
                mvpMatrix: particleMatrix,
                opacity: 1.0,
                cornerRadius: scale * 0.5,
                layerSize: SIMD2<Float>(scale, scale)
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(uniformBuffer, bufferOffset: uniformOffset, data: uniformData)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

            renderPass.setPipeline(pipeline)
            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }
    }

    /// Calculates an emission position based on the emitter shape.
    private func calculateEmissionPosition(
        shape: CAEmitterLayerEmitterShape,
        position: CGPoint,
        zPosition: CGFloat,
        size: CGSize,
        depth: CGFloat
    ) -> SIMD3<Float> {
        let x = Float(position.x)
        let y = Float(position.y)
        let z = Float(zPosition)
        let w = Float(size.width)
        let h = Float(size.height)

        switch shape {
        case .point:
            return SIMD3(x, y, z)
        case .line:
            return SIMD3(x - w/2 + w * randomFloat(), y, z)
        case .rectangle:
            return SIMD3(x - w/2 + w * randomFloat(), y - h/2 + h * randomFloat(), z)
        case .circle:
            let angle = randomFloat() * 2 * .pi
            let radius = sqrt(randomFloat()) * min(w, h) / 2
            return SIMD3(x + radius * cos(angle), y + radius * sin(angle), z)
        case .sphere:
            let theta = randomFloat() * 2 * .pi
            let phi = acos(2 * randomFloat() - 1)
            let r = cbrt(randomFloat()) * min(w, h, Float(depth)) / 2
            return SIMD3(x + r * sin(phi) * cos(theta), y + r * sin(phi) * sin(theta), z + r * cos(phi))
        default:
            return SIMD3(x, y, z)
        }
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
                for sublayer in tiledLayer.sortedSublayers() {
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
                    // Render cached tile as texture
                    renderTileWithImage(
                        cachedImage,
                        layer: presentation,
                        device: device,
                        renderPass: renderPass,
                        tileMatrix: tileMatrix,
                        tileSize: CGSize(width: tileW, height: tileH),
                        opacity: currentEffectiveOpacity * fadeOpacity
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
        opacity: Float
    ) {
        guard let texturedPipeline = texturedPipeline,
              let vertexBuffer = vertexBuffer,
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
            factory: { [weak self] in
                self?.createGPUTexture(from: image, device: device)
            }
        ) else { return }

        // Create vertices with white color (texture provides color)
        // V-flip: in Y-up system, bottom vertices (y=0) get V=1, top vertices (y=1) get V=0
        let white = SIMD4<Float>(1, 1, 1, 1)
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
        var uniforms = CARendererUniforms(
            mvpMatrix: tileMatrix,
            opacity: opacity,
            cornerRadius: 0,
            layerSize: SIMD2<Float>(Float(tileSize.width), Float(tileSize.height))
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
                        size: UInt64(MemoryLayout<CARendererUniforms>.stride)
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
