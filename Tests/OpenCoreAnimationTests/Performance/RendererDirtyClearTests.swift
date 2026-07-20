// Phase 1 — Renderer-end commit hook (PERFORMANCE_DESIGN.md §3.8 / §6.5).
// Asserts that `CARenderer.render(layer:)` bumps the per-frame token at the
// top of the call AND clears the entire dirty subtree after `submit`.

import Testing
import Foundation
@testable import OpenCoreAnimation

#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif

#if canImport(Metal)

extension PerformanceTests {
@Suite
@MainActor
struct RendererDirtyClearTests {

    init() { resetPerformanceTestState() }

    /// 1.15 — A single render() call clears all dirty bits and counters
    /// on the root and every dirty descendant. Mirrors PERFORMANCE_DESIGN
    /// §3.8: `submit → clear → completionBlocks`.
    @Test func renderClearsDirtySubtree() async throws {
        let renderer = CAMetalRenderer()
        try await renderer.initialize()
        renderer.resize(width: 10, height: 10)

        let root = CALayer()
        let mid  = CALayer()
        let leaf = CALayer()
        root.addSublayer(mid)
        mid.addSublayer(leaf)
        root.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        leaf.opacity = 0.5

        #expect(root._subtreeDirtyCount > 0)
        #expect(leaf._dirtyMask.isEmpty == false)

        renderer.render(layer: root)

        #expect(root._dirtyMask.isEmpty)
        #expect(mid._dirtyMask.isEmpty)
        #expect(leaf._dirtyMask.isEmpty)
        #expect(root._subtreeDirtyCount == 0)
        #expect(mid._subtreeDirtyCount == 0)
        #expect(leaf._subtreeDirtyCount == 0)
    }

    /// 1.16 — render() bumps `CALayer._currentFrameToken` exactly once,
    /// regardless of how many layers are in the tree. Single counter is
    /// the cache key for Phase 2 presentation/order caches.
    @Test func renderBumpsFrameTokenOnce() async throws {
        let renderer = CAMetalRenderer()
        try await renderer.initialize()
        renderer.resize(width: 10, height: 10)

        let root = CALayer()
        for _ in 0..<5 { root.addSublayer(CALayer()) }

        let before = CALayer._currentFrameToken
        renderer.render(layer: root)
        #expect(CALayer._currentFrameToken == before &+ 1)

        renderer.render(layer: root)
        #expect(CALayer._currentFrameToken == before &+ 2)
    }

    /// 1.17 — A second render() with no intervening mutations is a no-op
    /// for the dirty machinery: counter stays 0, masks stay empty.
    @Test func secondRenderWithoutMutationsStaysClean() async throws {
        let renderer = CAMetalRenderer()
        try await renderer.initialize()
        renderer.resize(width: 10, height: 10)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        renderer.render(layer: layer)
        #expect(layer._subtreeDirtyCount == 0)

        renderer.render(layer: layer)
        #expect(layer._dirtyMask.isEmpty)
        #expect(layer._subtreeDirtyCount == 0)
    }

    /// 1.18 — `_needsDisplay` is the orthogonal axis (B7) and must NOT
    /// be cleared by the render-end hook. Only `displayIfNeeded()` does.
    @Test func renderLeavesNeedsDisplayAxisUntouched() async throws {
        let renderer = CAMetalRenderer()
        try await renderer.initialize()
        renderer.resize(width: 10, height: 10)

        let layer = CALayer()
        layer.setNeedsDisplay()
        #expect(layer._needsDisplayForTest == true)

        renderer.render(layer: layer)

        #expect(layer._dirtyMask.isEmpty)
        #expect(layer._needsDisplayForTest == true)
    }
}
}

#endif
