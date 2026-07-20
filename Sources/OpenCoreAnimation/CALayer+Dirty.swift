// Phase 1 of the OpenCoreAnimation performance plan.
// See PERFORMANCE_DESIGN.md §3 for the full design.
//
// This extension adds the dirty-bit infrastructure used by the renderer
// (and Phase 2/3 caches) to skip work for sub-trees that haven't changed
// since the last commit. The pattern mirrors the existing
// `_subtreeShadowCount` / `propagateShadowDelta` machinery in CALayer.swift
// — a single propagation primitive plus per-property setter wrappers that
// call into it.

import Foundation
import Synchronization

#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif

extension CALayer {

    // MARK: - DirtyFlags

    /// Bit-flag set tracking which categories of state are stale on the
    /// model layer since the last `recursivelyClearDirtyAfterCommit()`.
    ///
    /// The renderer (and Phase 2/3 caches) reads this to skip work for
    /// untouched sub-trees. `_subtreeDirtyCount` tracks how many descendants
    /// (incl. self) carry a non-empty mask so the renderer can early-return
    /// at any level.
    internal struct DirtyFlags: OptionSet, Sendable {
        let rawValue: UInt32
        init(rawValue: UInt32) { self.rawValue = rawValue }

        static let geometry          = DirtyFlags(rawValue: 1 << 0)
        static let appearance        = DirtyFlags(rawValue: 1 << 1)
        static let contents          = DirtyFlags(rawValue: 1 << 2)
        static let contentsRedraw    = DirtyFlags(rawValue: 1 << 3)
        static let shadow            = DirtyFlags(rawValue: 1 << 4)
        static let filters           = DirtyFlags(rawValue: 1 << 5)
        static let mask              = DirtyFlags(rawValue: 1 << 6)
        static let sublayerHierarchy = DirtyFlags(rawValue: 1 << 7)
        static let sublayerOrdering  = DirtyFlags(rawValue: 1 << 8)
        static let rasterization     = DirtyFlags(rawValue: 1 << 9)
        static let animations        = DirtyFlags(rawValue: 1 << 10)
        static let timing            = DirtyFlags(rawValue: 1 << 11)

        static let all: DirtyFlags = [
            .geometry, .appearance, .contents, .contentsRedraw,
            .shadow, .filters, .mask, .sublayerHierarchy,
            .sublayerOrdering, .rasterization, .animations, .timing,
        ]

        /// Bits that affect the *self* render output (not children selection
        /// nor timing). Used by the R2.2 fast path in `_renderTimePresentation`.
        static let presentationAffecting: DirtyFlags = [
            .geometry, .appearance, .contents, .shadow,
            .filters, .mask, .rasterization, .animations,
        ]
    }

    /// Initial mask given to a freshly-allocated CALayer.
    ///
    /// `.contentsRedraw` is intentionally NOT included (B7): it is the
    /// explicit `setNeedsDisplay()` axis and must not auto-trigger
    /// `display()` on the first frame for every newly-created layer.
    internal static var _initialDirtyMask: DirtyFlags {
        DirtyFlags.all.subtracting(.contentsRedraw)
    }

    // MARK: - Frame token

    /// Monotonic per-render-frame counter. Bumped by the renderer at the
    /// top of each `render(layer:)`. Native renderers may be driven from
    /// different executors, so the process-wide counter is synchronized.
    private static let frameTokenStorage = Mutex<UInt64>(0)

    internal static var _currentFrameToken: UInt64 {
        get { frameTokenStorage.withLock { $0 } }
        set { frameTokenStorage.withLock { $0 = newValue } }
    }

    @discardableResult
    internal static func advanceFrameToken() -> UInt64 {
        frameTokenStorage.withLock {
            $0 &+= 1
            return $0
        }
    }

    // MARK: - Mark dirty

    /// Mark a category of state as dirty on this layer. Propagates a +1
    /// delta to every ancestor's `_subtreeDirtyCount` only on the
    /// clean→dirty transition (idempotent).
    ///
    /// No-op on presentation layers — they are read-only consumers
    /// (PERFORMANCE_DESIGN.md §3.5).
    internal func markDirty(_ flags: DirtyFlags) {
        if _isPresentationLayer { return }
        let wasClean = _dirtyMask.isEmpty
        _dirtyMask.formUnion(flags)
        if wasClean && !_dirtyMask.isEmpty {
            CALayer.propagateDirtyDeltaPublic(+1, startingAt: self)
        }
        _presentationCacheIsValid = false
    }

    // MARK: - Subtree counter propagation

    /// Mirrors `propagateShadowDelta` exactly so the two counters stay
    /// behaviorally aligned. Internal because the sublayer mutators in
    /// CALayer.swift's body need to call it across extensions.
    internal static func propagateDirtyDeltaPublic(_ delta: Int, startingAt layer: CALayer?) {
        guard delta != 0 else { return }
        var node = layer
        while let n = node {
            n._subtreeDirtyCount += delta
            node = n._superlayerForDirty
        }
    }

    // MARK: - Clear at end of frame

    /// Walk every dirty descendant and clear its bits + decrement the
    /// per-ancestor counter. `_needsDisplay` is intentionally untouched
    /// (B7) — it has its own lifecycle managed by `displayIfNeeded()`.
    ///
    /// Called by `CAWebGPURenderer.render(layer:)` AFTER
    /// `device.queue.submit(...)` and BEFORE any user-visible
    /// completion blocks fire (PERFORMANCE_DESIGN.md §3.8 / §6.5).
    internal func recursivelyClearDirtyAfterCommit() {
        guard _subtreeDirtyCount > 0 else { return }
        if !_dirtyMask.isEmpty {
            _dirtyMask = []
            CALayer.propagateDirtyDeltaPublic(-1, startingAt: self)
        }
        _sublayersForDirty?.forEach { $0.recursivelyClearDirtyAfterCommit() }
    }
}
