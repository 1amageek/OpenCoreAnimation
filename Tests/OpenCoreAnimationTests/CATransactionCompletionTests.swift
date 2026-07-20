import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CATransaction completion behavior", .serialized)
struct CATransactionCompletionTests {
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
        layer.removeAnimation(forKey: "opacity")
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
        setStoredAnimationBeginTime(CACurrentMediaTime() - 2, on: layer, forKey: "opacity")
        layer.processAnimationCompletions()
        #expect(completionCount == 1)
    }
}
