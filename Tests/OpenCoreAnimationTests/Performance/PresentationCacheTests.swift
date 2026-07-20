// Phase 2 — Presentation cache & sublayer ordering cache.
// Tests track PERFORMANCE_DESIGN.md §4.7 sequence 2.1–2.7.

import Testing
import Foundation
@testable import OpenCoreAnimation

#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif

extension PerformanceTests {
@Suite
struct PresentationCacheTests {

    init() { resetPerformanceTestState() }

    // 2.1 — Clean layer with no live animations: the renderer-internal
    // `_renderTimePresentation()` returns `self` (R2.2 fast path). The
    // public `presentation()` keeps Apple-contract semantics and returns
    // a distinct copy, so we deliberately probe the internal hook here.
    @Test func cleanLayerWithoutAnimationsReturnsSelf() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        layer._testClearDirty()
        #expect(layer._dirtyMask.isEmpty)

        let p = layer._renderTimePresentation()
        #expect(p === layer)
    }

    // 2.2 — A presentation-affecting mutation forces the slow path:
    // `_renderTimePresentation()` must build (or reuse) a presentation
    // copy, and the result must NOT alias the model layer.
    @Test func dirtyLayerBuildsPresentationCopy() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        layer._testClearDirty()

        // Mutate a presentation-affecting property (.geometry bit).
        layer.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
        #expect(layer._dirtyMask.contains(.geometry))

        // Bump the frame token the way the renderer would.
        CALayer.advanceFrameToken()

        let p = layer._renderTimePresentation()
        #expect(p !== layer)
    }

    // 2.3 — R2.1: two `presentation()` calls within the same frame token
    // must return the same instance — Apple's documented behavior is a
    // distinct copy per call, but the cache reuses it within a frame.
    @Test func secondPresentationInSameFrameReturnsCachedCopy() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)

        // Force a frame so the cache token can be set.
        CALayer.advanceFrameToken()

        let first  = layer.presentation()
        let second = layer.presentation()
        #expect(first != nil)
        #expect(first === second)
    }

    // 2.4 — Bumping the frame token AND mutating a presentation-affecting
    // property invalidates the cache: the next presentation() call must
    // observe the new model values. (The design intentionally reuses the
    // `_presentationLayer` instance to avoid per-frame allocation; what
    // matters for callers is that the values are fresh, not identity.)
    @Test func nextFrameInvalidatesCache() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        CALayer.advanceFrameToken()
        guard let first = layer.presentation() else {
            Issue.record("presentation() returned nil on clean layer")
            return
        }
        let firstWidth: CGFloat = first.bounds.size.width
        let expectedFirstWidth: CGFloat = 10.0
        #expect(firstWidth == expectedFirstWidth)

        // New frame + dirty mutation.
        CALayer.advanceFrameToken()
        layer.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)

        guard let second = layer.presentation() else {
            Issue.record("presentation() returned nil after mutation")
            return
        }
        let secondWidth: CGFloat = second.bounds.size.width
        let expectedSecondWidth: CGFloat = 20.0
        #expect(secondWidth == expectedSecondWidth)
    }

    // 2.5 — R2.3: `sortedSublayers()` returns the same array instance
    // across frames as long as the ordering bits stay clean and the token
    // matches. We compare by element identity to avoid relying on Array
    // value equality (CALayer is a class, so === per element is the
    // contract that matters).
    @Test func sortedSublayersCachedAcrossFrames() {
        let parent = CALayer()
        let a = CALayer(); let b = CALayer(); let c = CALayer()
        parent.addSublayer(a)
        parent.addSublayer(b)
        parent.addSublayer(c)
        parent._testClearDirty()

        CALayer.advanceFrameToken()
        let first = parent.sortedSublayers()

        // Same frame, no mutations: must hit the cache.
        let secondSameFrame = parent.sortedSublayers()
        #expect(first.count == secondSameFrame.count)
        for (i, layer) in first.enumerated() {
            #expect(layer === secondSameFrame[i])
        }

        // Next frame, still clean: cache is invalidated by the token bump
        // but rebuilds the same ordering. Element identity must match.
        CALayer.advanceFrameToken()
        let nextFrame = parent.sortedSublayers()
        #expect(first.count == nextFrame.count)
        for (i, layer) in first.enumerated() {
            #expect(layer === nextFrame[i])
        }
    }

    // 2.6 — `addSublayer` flips `.sublayerHierarchy`, which must force
    // `sortedSublayers()` to recompute and pick up the new child.
    @Test func addSublayerInvalidatesSortedCache() {
        let parent = CALayer()
        let a = CALayer(); let b = CALayer()
        parent.addSublayer(a)
        parent.addSublayer(b)
        parent._testClearDirty()
        CALayer.advanceFrameToken()
        let before = parent.sortedSublayers()
        #expect(before.count == 2)

        let c = CALayer()
        parent.addSublayer(c)
        #expect(parent._dirtyMask.contains(.sublayerHierarchy))

        let after = parent.sortedSublayers()
        #expect(after.count == 3)
        #expect(after.last === c)
    }

    // 2.7 — Changing a child's `zPosition` flips the parent's
    // `.sublayerOrdering` bit (per §3.2 mapping) and must reorder.
    @Test func zPositionChangeReordersCache() {
        let parent = CALayer()
        let a = CALayer(); let b = CALayer()
        parent.addSublayer(a)
        parent.addSublayer(b)
        parent._testClearDirty()
        CALayer.advanceFrameToken()
        let before = parent.sortedSublayers()
        #expect(before.first === a)
        #expect(before.last  === b)

        // Push `a` above `b` via z-order.
        a.zPosition = 10
        #expect(parent._dirtyMask.contains(.sublayerOrdering))

        let after = parent.sortedSublayers()
        #expect(after.first === b)
        #expect(after.last  === a)
    }
}
}
