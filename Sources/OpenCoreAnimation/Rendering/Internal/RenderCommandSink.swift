// Phase 3 — A renderer-agnostic command sink that the actual GPU
// renderer (CAWebGPURenderer) and the test mock (MockCARenderer) both
// write to. The sink records the *decisions* the renderer would make
// (pipeline state, uniform writes, draws), not the GPU bytes — that
// way Phase 3 tests assert against the same surface that production
// code drives, without having to spin up a real GPU.
//
// See PERFORMANCE_DESIGN.md §10 (test infra) and §5.7 tests 3.4a/3.4b/
// 3.7/3.8 for the assertions the snapshot types need to support.

import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif

// MARK: - Snapshots

/// The subset of pipeline state Phase 3 tests assert on. We deliberately
/// keep this narrower than a full WebGPU `RenderPipelineDescriptor` —
/// only the fields whose values are *decided* by Phase 3 logic (alpha
/// blend toggle, capture-pass clear alpha, debug label) appear here.
internal struct RenderPipelineSnapshot: Equatable, Sendable {
    /// `true` iff the pipeline performs source-over alpha blending. R3.5
    /// sets this to `false` when the layer is `isOpaque`.
    internal var blendEnabled: Bool
    /// The clear-alpha used by the render pass that this pipeline writes
    /// into. R3.3 mandates that *capture* passes clear with α = 1.0
    /// regardless of the layer's opacity.
    internal var clearAlpha: Float
    /// Free-form label used to disambiguate pipelines in test logs.
    internal var label: String?

    internal init(blendEnabled: Bool, clearAlpha: Float, label: String? = nil) {
        self.blendEnabled = blendEnabled
        self.clearAlpha = clearAlpha
        self.label = label
    }
}

/// The subset of per-quad uniform data Phase 3 tests assert on.
internal struct UniformPayload: Equatable, Sendable {
    /// The opacity multiplier applied at composite time. R3.3 mandates
    /// this equals the layer's *current* opacity even when the layer's
    /// pixels were captured opaque.
    internal var opacity: Float
    /// `true` iff this draw samples a texture (vs. a flat-colour quad).
    internal var hasTexture: Bool

    internal init(opacity: Float, hasTexture: Bool) {
        self.opacity = opacity
        self.hasTexture = hasTexture
    }
}

// MARK: - Sink

/// What the renderer calls into instead of swift-webgpu directly. The
/// production renderer wires this to a real GPU adapter; the test
/// renderer wires it to a recorder.
internal protocol RenderCommandSink: AnyObject {
    /// A texture upload (corresponds to `device.queue.writeTexture`).
    /// `key` identifies the source so tests can assert "no second upload
    /// for this key". `byteCount` is the upload size in bytes.
    func writeTexture(key: ObjectIdentifier, byteCount: Int)

    /// The pipeline the upcoming draws will use.
    func setPipeline(_ snapshot: RenderPipelineSnapshot)

    /// A uniform-buffer write that precedes a draw.
    func writeUniform(_ payload: UniformPayload)

    /// A single draw of one layer's quad. `layerID` lets tests reason
    /// about which layer was drawn.
    func dispatchQuad(layerID: ObjectIdentifier)

    /// Called once at the end of a frame, mirroring `queue.submit`.
    func submit()

    /// Called by the renderer after submit, when it cycles
    /// frame-token-keyed bookkeeping (LRU cache, etc.).
    func frameDidEnd(frameToken: UInt64)
}
