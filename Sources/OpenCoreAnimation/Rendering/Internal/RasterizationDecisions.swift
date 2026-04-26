// Phase 3 — Pure-function decision tree shared by the production
// renderer and the test mock. Anything Phase 3 tests assert against
// (capture-vs-reuse, blend-on/off, shadowPath fast path, opacity
// gating) is decided here so both renderers reach the same answer.
//
// See PERFORMANCE_DESIGN.md §5.1–§5.5.

import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif

internal enum RasterizationDecisions {

    // MARK: - Backing store (R3.1)

    /// Whether the renderer can reuse the existing GPU texture for
    /// `layer.contents` instead of re-uploading it. Reuse is safe when
    /// `.contentsRedraw` is clean — `contents`-affecting setters set
    /// the bit, so a clean bit means the source bytes are unchanged.
    internal static func canReuseContentsTexture(layer: CALayer) -> Bool {
        !layer._dirtyMask.contains(.contentsRedraw)
    }

    // MARK: - Rasterization (R3.2 / R3.3)

    /// Whether the renderer should reuse a previously-captured rasterized
    /// texture for `layer`. Reuse is correct when:
    /// (a) `shouldRasterize` is on,
    /// (b) the layer's `.rasterization` bit is clean (rasterizationScale
    ///     and shouldRasterize itself didn't change), and
    /// (c) no descendant of the layer is dirty (`_subtreeDirtyCount == 0`),
    ///     and
    /// (d) the cached entry's `contentBoundsHash` matches the current
    ///     bounds + transform inputs.
    internal static func canReuseRasterizedTexture<T>(
        layer: CALayer,
        cached: RasterizedEntry<T>?,
        currentContentBoundsHash: Int
    ) -> Bool {
        guard layer.shouldRasterize, let cached else { return false }
        if layer._dirtyMask.contains(.rasterization) { return false }
        if layer._subtreeDirtyCount > 0 { return false }
        return cached.contentBoundsHash == currentContentBoundsHash
    }

    /// The clear-alpha used by an offscreen capture pass. Always 1.0 —
    /// R3.3 separates per-pixel alpha (captured fully opaque) from the
    /// composite-time opacity multiplier (applied at draw).
    internal static func captureClearAlpha() -> Float { 1.0 }

    /// The opacity uniform written when compositing a cached rasterized
    /// quad. Comes from the *current* layer.opacity, not the captured
    /// state — that lets opacity changes skip re-capture.
    internal static func compositeOpacity(for layer: CALayer) -> Float {
        Float(layer.opacity)
    }

    // MARK: - isOpaque (R3.5)

    /// Whether the per-quad pipeline should enable source-over alpha
    /// blending. False when `isOpaque` (R3.5) — but still true when
    /// `opacity < 1`, because the *composite* uniform multiplier
    /// requires blending to land correctly even for an "opaque" layer
    /// drawn at half alpha.
    internal static func blendEnabled(for layer: CALayer) -> Bool {
        if layer.isOpaque && layer.opacity >= 1.0 { return false }
        return true
    }

    // MARK: - shadowPath (R3.6)

    /// Whether the renderer should skip the silhouette-extraction
    /// pipeline (which derives a mask from `contents`) and instead
    /// tessellate `shadowPath` directly. True iff `shadowPath` is set
    /// and the layer projects a shadow at all.
    internal static func useShadowPathFastPath(for layer: CALayer) -> Bool {
        guard layer.shadowOpacity > 0 else { return false }
        return layer.shadowPath != nil
    }

    // MARK: - Prerender skip (R3.7)

    /// Whether `prerenderShadows` / `prerenderFilteredLayers` can reuse a
    /// cached blurred texture instead of re-running the blur passes.
    /// Reuse requires the contributor's subtree to be clean and a cache
    /// entry for it to exist.
    internal static func canReusePrerenderCache(
        contributorLayer: CALayer,
        hasCachedTexture: Bool
    ) -> Bool {
        guard hasCachedTexture else { return false }
        if contributorLayer._subtreeDirtyCount > 0 { return false }
        return contributorLayer._dirtyMask.isDisjoint(
            with: [.shadow, .filters, .contentsRedraw, .geometry])
    }
}
