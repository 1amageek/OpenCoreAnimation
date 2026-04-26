// Phase 1 — Dirty propagation infrastructure.
// Tests track PERFORMANCE_DESIGN.md §3.10 sequence 1.1–1.14.

import Testing
import Foundation
@testable import OpenCoreAnimation

#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif

@Suite(.serialized)
struct DirtyPropagationTests {

    init() { resetPerformanceTestState() }

    // 1.1 — A fresh layer is dirty for every render-affecting category
    // EXCEPT `.contentsRedraw` (B7), and `_needsDisplay` is independent.
    @Test func freshLayerIsAllDirtyExceptContentsRedraw() {
        let layer = CALayer()
        let expected = CALayer.DirtyFlags.all.subtracting(.contentsRedraw)
        #expect(layer._dirtyMask == expected)
        #expect(layer._subtreeDirtyCount == 1)
        // _needsDisplay starts false — it's the OTHER axis (B7).
        #expect(layer._needsDisplayForTest == false)
    }

    // 1.2 — A geometry setter sets the .geometry bit.
    @Test func geometrySetterMarksGeometryBit() {
        let layer = CALayer()
        layer._testClearDirty()
        #expect(layer._dirtyMask.isEmpty)

        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        #expect(layer._dirtyMask.contains(.geometry))
    }

    // 1.3 — Setting a property to its current value does NOT propagate.
    @Test func idempotentSetterDoesNotPropagate() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 50, height: 50)
        layer._testClearDirty()
        #expect(layer._subtreeDirtyCount == 0)

        // Same value → no-op
        layer.bounds = CGRect(x: 0, y: 0, width: 50, height: 50)
        #expect(layer._dirtyMask.isEmpty)
        #expect(layer._subtreeDirtyCount == 0)
    }

    // 1.4 — Each property in §3.2 sets its bit.
    @Test func appearancePropertiesMarkAppearanceBit() {
        let layer = CALayer()
        layer._testClearDirty()
        layer.opacity = 0.5
        #expect(layer._dirtyMask.contains(.appearance))

        layer._testClearDirty()
        layer.isHidden = true
        #expect(layer._dirtyMask.contains(.appearance))

        layer._testClearDirty()
        layer.cornerRadius = 4
        #expect(layer._dirtyMask.contains(.appearance))
    }

    @Test func contentsPropertiesMarkContentsBit() {
        let layer = CALayer()
        layer._testClearDirty()
        layer.contentsScale = 2.0
        #expect(layer._dirtyMask.contains(.contents))

        layer._testClearDirty()
        layer.contentsRect = CGRect(x: 0, y: 0, width: 0.5, height: 0.5)
        #expect(layer._dirtyMask.contains(.contents))
    }

    @Test func shadowPropertiesMarkShadowBit() {
        let layer = CALayer()
        layer._testClearDirty()
        layer.shadowOpacity = 0.8
        #expect(layer._dirtyMask.contains(.shadow))

        layer._testClearDirty()
        layer.shadowRadius = 4
        #expect(layer._dirtyMask.contains(.shadow))
    }

    @Test func rasterizationPropertiesMarkRasterizationBit() {
        let layer = CALayer()
        layer._testClearDirty()
        layer.shouldRasterize = true
        #expect(layer._dirtyMask.contains(.rasterization))

        layer._testClearDirty()
        layer.isOpaque = true
        #expect(layer._dirtyMask.contains(.rasterization))
    }

    // 1.5 — addSublayer marks parent's .sublayerHierarchy, not child's.
    @Test func addSublayerMarksParentHierarchyBit() {
        let parent = CALayer()
        let child = CALayer()
        parent._testClearDirty()
        child._testClearDirty()

        parent.addSublayer(child)

        #expect(parent._dirtyMask.contains(.sublayerHierarchy))
        #expect(child._dirtyMask.contains(.sublayerHierarchy) == false)
    }

    // 1.6 — Setting a child's zPosition marks parent's .sublayerOrdering.
    @Test func zPositionMarksParentOrderingBit() {
        let parent = CALayer()
        let child = CALayer()
        parent.addSublayer(child)
        parent._testClearDirty()
        child._testClearDirty()

        child.zPosition = 5

        #expect(child._dirtyMask.contains(.geometry))
        #expect(parent._dirtyMask.contains(.sublayerOrdering))
    }

    // 1.7 — Dirtying a leaf bumps every ancestor by 1.
    @Test func subtreeCounterIncrementsOnDirty() {
        let root = CALayer()
        let mid  = CALayer()
        let leaf = CALayer()
        root.addSublayer(mid)
        mid.addSublayer(leaf)

        // Start clean
        root._testClearDirty()
        #expect(root._subtreeDirtyCount == 0)
        #expect(mid._subtreeDirtyCount == 0)
        #expect(leaf._subtreeDirtyCount == 0)

        leaf.opacity = 0.5

        #expect(leaf._subtreeDirtyCount == 1)
        #expect(mid._subtreeDirtyCount == 1)
        #expect(root._subtreeDirtyCount == 1)
    }

    // 1.8 — Dirtying an already-dirty leaf does not double-count.
    @Test func subtreeCounterIdempotent() {
        let root = CALayer()
        let leaf = CALayer()
        root.addSublayer(leaf)
        root._testClearDirty()

        leaf.opacity = 0.5
        leaf.opacity = 0.4   // different value, but bit already set
        leaf.opacity = 0.3

        #expect(leaf._subtreeDirtyCount == 1)
        #expect(root._subtreeDirtyCount == 1)
    }

    // 1.9 — Re-parenting moves the dirty count from old to new ancestors.
    @Test func reparentingPreservesCounter() {
        let parentA = CALayer()
        let parentB = CALayer()
        let leaf = CALayer()
        parentA.addSublayer(leaf)
        parentA._testClearDirty()
        parentB._testClearDirty()
        #expect(parentA._subtreeDirtyCount == 0)

        leaf.opacity = 0.5
        #expect(parentA._subtreeDirtyCount == 1)
        #expect(parentB._subtreeDirtyCount == 0)

        parentB.addSublayer(leaf)   // implicit removeFromSuperlayer + reattach

        // removeFromSuperlayer drops leaf's +1 from parentA, then sets
        // parentA's `.sublayerHierarchy` bit (its child set changed) which
        // contributes a fresh +1. Net: parentA stays at 1.
        // parentB receives +leaf._subtreeDirtyCount (=1) AND its own
        // `.sublayerHierarchy` mark which contributes another +1 → 2.
        #expect(parentA._subtreeDirtyCount == 1)
        #expect(parentA._dirtyMask.contains(.sublayerHierarchy))
        #expect(parentB._subtreeDirtyCount >= 1)   // at least leaf's
        #expect(parentB._dirtyMask.contains(.sublayerHierarchy))
        #expect(leaf._subtreeDirtyCount == 1)
        #expect(leaf._superlayerForDirty === parentB)
    }

    // 1.10 — markDirty is a no-op on a presentation layer.
    @Test func presentationLayerCannotMarkDirty() {
        let model = CALayer()
        model.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        model._testClearDirty()
        // Materialize a presentation copy.
        let presentation = model.presentation()
        guard let presentation else {
            Issue.record("presentation() returned nil")
            return
        }
        let countBefore = presentation._subtreeDirtyCount
        presentation.bounds = CGRect(x: 0, y: 0, width: 99, height: 99)
        #expect(presentation._dirtyMask.isEmpty)
        #expect(presentation._subtreeDirtyCount == countBefore)
    }

    // 1.11 — recursivelyClearDirtyAfterCommit zeros bits AND counter.
    @Test func clearDirtyAfterCommitZeroesCounter() {
        let root = CALayer()
        let leaf = CALayer()
        root.addSublayer(leaf)
        leaf.opacity = 0.5
        root.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)

        root.recursivelyClearDirtyAfterCommit()

        #expect(root._dirtyMask.isEmpty)
        #expect(leaf._dirtyMask.isEmpty)
        #expect(root._subtreeDirtyCount == 0)
        #expect(leaf._subtreeDirtyCount == 0)
    }

    // 1.12 — setNeedsDisplay() sets BOTH axes; bounds= touches NEITHER.
    @Test func setNeedsDisplaySetsBothAxes() {
        let layer = CALayer()
        layer._testClearDirty()

        layer.setNeedsDisplay()
        #expect(layer._dirtyMask.contains(.contentsRedraw))
        #expect(layer._needsDisplayForTest == true)

        layer._testClearDirty()
        // After clear, _needsDisplay is intentionally untouched (B7);
        // only displayIfNeeded() clears it.
        #expect(layer._dirtyMask.isEmpty)
        #expect(layer._needsDisplayForTest == true)

        layer.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        #expect(layer._dirtyMask.contains(.contentsRedraw) == false)
    }

    // 1.13 — Subclass override of needsDisplay(forKey:) wires through.
    @Test func needsDisplayForKeyTriggersRedrawForOverride() {
        final class RedrawOnAppearanceLayer: CALayer {
            override class func needsDisplay(forKey key: String) -> Bool {
                if key == "opacity" { return true }
                return super.needsDisplay(forKey: key)
            }
        }
        let layer = RedrawOnAppearanceLayer()
        layer._testClearDirty()
        layer.opacity = 0.5

        #expect(layer._dirtyMask.contains(.appearance))
        #expect(layer._dirtyMask.contains(.contentsRedraw))
    }

    // 1.14 — clearDirty leaves _needsDisplay untouched (B7).
    @Test func clearDirtyAfterCommitLeavesNeedsDisplayUntouched() {
        let layer = CALayer()
        layer.setNeedsDisplay()
        #expect(layer._needsDisplayForTest == true)

        layer.recursivelyClearDirtyAfterCommit()

        #expect(layer._dirtyMask.isEmpty)
        #expect(layer._needsDisplayForTest == true)   // independent axis
    }
}
