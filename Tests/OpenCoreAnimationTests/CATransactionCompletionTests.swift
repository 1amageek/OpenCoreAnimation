import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CATransaction", .serialized)
struct CATransactionTestSuites {}

extension CATransactionTestSuites {
@Suite("Completion behavior")
struct CompletionBehavior {
    @Test("Completion is immediate when the transaction adds no animations")
    func noAnimationsCompletesAtCommit() {
        CATransaction.flush()
        var completionCount = 0

        CATransaction.begin()
        CATransaction.setCompletionBlock {
            completionCount += 1
        }
        CATransaction.commit()

        #expect(completionCount == 1)
    }

    @Test("Completion waits for an explicitly added animation")
    func explicitAnimationDelaysCompletion() {
        CATransaction.flush()
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.duration = 1
        var completionCount = 0

        CATransaction.begin()
        CATransaction.setCompletionBlock {
            completionCount += 1
        }
        layer.add(animation, forKey: "opacity")
        CATransaction.commit()

        #expect(completionCount == 0)
        submitCommittedFrame(layer)
        #expect(completionCount == 0)
        setStoredAnimationBeginTime(CACurrentMediaTime() - 2, on: layer, forKey: "opacity")
        layer.processAnimationCompletions()
        #expect(completionCount == 1)
    }

    @Test("Removing an animation releases its transaction completion")
    func removalCompletesTransaction() {
        CATransaction.flush()
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.duration = 10
        var completionCount = 0

        CATransaction.begin()
        CATransaction.setCompletionBlock {
            completionCount += 1
        }
        layer.add(animation, forKey: "opacity")
        CATransaction.commit()

        #expect(completionCount == 0)
        submitCommittedFrame(layer)
        #expect(completionCount == 0)
        layer.removeAnimation(forKey: "opacity")
        #expect(completionCount == 1)
    }

    @Test("Completion tracks an animation group as one attached animation")
    func groupCompletesTransactionOnce() {
        CATransaction.flush()
        let layer = CALayer()
        let child = CABasicAnimation(keyPath: "opacity")
        child.duration = 1
        let group = CAAnimationGroup()
        group.animations = [child]
        group.duration = 1
        var completionCount = 0

        CATransaction.begin()
        CATransaction.setCompletionBlock {
            completionCount += 1
        }
        layer.add(group, forKey: "group")
        CATransaction.commit()

        #expect(completionCount == 0)
        submitCommittedFrame(layer)
        #expect(completionCount == 0)
        setStoredAnimationBeginTime(CACurrentMediaTime() - 2, on: layer, forKey: "group")
        layer.processAnimationCompletions()
        layer.processAnimationCompletions()
        #expect(completionCount == 1)
    }

    @Test("Completion tracks an implicit animation created by a layer action")
    func implicitActionAnimationDelaysCompletion() {
        CATransaction.flush()
        let layer = CALayer()
        layer.actions = ["opacity": CABasicAnimation(keyPath: "opacity")]
        var completionCount = 0

        CATransaction.begin()
        CATransaction.setAnimationDuration(1)
        CATransaction.setCompletionBlock {
            completionCount += 1
        }
        layer.opacity = 0
        CATransaction.commit()

        #expect(layer.animation(forKey: "opacity") != nil)
        #expect(completionCount == 0)
        submitCommittedFrame(layer)
        #expect(completionCount == 0)
        setStoredAnimationBeginTime(CACurrentMediaTime() - 2, on: layer, forKey: "opacity")
        layer.processAnimationCompletions()
        #expect(completionCount == 1)
    }

    @MainActor
    @Test("Non-animated mutations complete after renderer submission and dirty clearing")
    func nonAnimatedMutationCompletesAfterRendererSubmission() {
        CATransaction.flush()
        var events: [String] = []
        let backend = TransactionSubmissionBackend {
            events.append("submit")
        }
        let renderer = CARenderer(backend: backend)
        let root = CALayer()
        renderer.layer = root
        renderer.bounds = CGRect(x: 0, y: 0, width: 32, height: 32)
        var callbackObservedCleanTree = false

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        CATransaction.setCompletionBlock {
            events.append("completion")
            callbackObservedCleanTree = root._dirtyMask.isEmpty
        }
        root.opacity = 0.5
        CATransaction.commit()

        #expect(events.isEmpty)
        renderer.beginFrame(atTime: CACurrentMediaTime(), timeStamp: nil)
        renderer.addUpdate(renderer.bounds)
        renderer.render()
        renderer.endFrame()

        #expect(events == ["submit", "completion"])
        #expect(callbackObservedCleanTree)
    }

    @Test("Hierarchy and detached mask mutations share the submitted frame")
    func hierarchyAndMaskMutationsCompleteAfterOneTreeSubmission() {
        CATransaction.flush()
        let root = CALayer()
        let child = CALayer()
        let mask = CALayer()
        var completionCount = 0

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        CATransaction.setCompletionBlock {
            completionCount += 1
        }
        root.addSublayer(child)
        root.mask = mask
        mask.opacity = 0.5
        CATransaction.commit()

        #expect(completionCount == 0)
        submitCommittedFrame(root)
        #expect(completionCount == 1)
    }

    @Test("A pending failed root does not absorb a later transaction")
    func pendingRootDoesNotAbsorbLaterTransaction() {
        CATransaction.flush()
        let failedRoot = CALayer()
        var failedCompletionCount = 0

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        CATransaction.setCompletionBlock {
            failedCompletionCount += 1
        }
        failedRoot.opacity = 0.5
        CATransaction.commit()

        #expect(failedRoot.pendingCommittedRenderState != nil)
        #expect(failedCompletionCount == 0)

        let succeedingRoot = CALayer()
        var succeedingCompletionCount = 0
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        CATransaction.setCompletionBlock {
            succeedingCompletionCount += 1
        }
        succeedingRoot.opacity = 0.25
        CATransaction.commit()

        #expect(succeedingRoot.pendingCommittedRenderState != nil)
        submitCommittedFrame(succeedingRoot)
        #expect(succeedingCompletionCount == 1)
        #expect(failedCompletionCount == 0)
        #expect(failedRoot.pendingCommittedRenderState != nil)
    }

    @Test("Completion mutations remain dirty for the following commit")
    func completionMutationSurvivesCommittedDirtyClear() {
        CATransaction.flush()
        let root = CALayer()

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        CATransaction.setCompletionBlock {
            root.position = CGPoint(x: 20, y: 30)
        }
        root.opacity = 0.5
        CATransaction.commit()

        submitCommittedFrame(root)

        #expect(root.position == CGPoint(x: 20, y: 30))
        #expect(root._dirtyMask.contains(.geometry))
        #expect(root._subtreeDirtyCount == 1)
        CATransaction.flush()
    }
}
}

private func submitCommittedFrame(_ rootLayer: CALayer) {
    rootLayer.recursivelyClearDirtyAfterCommit()
    rootLayer.completeTransactionsAfterRenderRecursively()
}

@MainActor
private final class TransactionSubmissionBackend: CARendererDelegate {
    var size = CGSize(width: 32, height: 32)
    private let onSubmit: () -> Void

    init(onSubmit: @escaping () -> Void) {
        self.onSubmit = onSubmit
    }

    func initialize() async throws {}

    func invalidate() {}

    func resize(width: Int, height: Int) {
        size = CGSize(width: width, height: height)
    }

    func render(layer rootLayer: CALayer) {
        rootLayer.recursivelyClearDirtyAfterCommit()
        onSubmit()
        rootLayer.completeTransactionsAfterRenderRecursively()
    }
}
