import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("Hierarchical CAMediaTiming evaluation")
struct CAMediaTimingEvaluatorTests {
    private final class AnimationDelegate: CAAnimationDelegate {
        var startCount = 0
        var stopEvents: [Bool] = []

        func animationDidStart(_ anim: CAAnimation) {
            startCount += 1
        }

        func animationDidStop(_ anim: CAAnimation, finished flag: Bool) {
            stopEvents.append(flag)
        }
    }

    @Test("repeatDuration is the complete active duration when autoreversing")
    func repeatDurationIsNotDoubledByAutoreverse() {
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.beginTime = 10
        animation.duration = 1
        animation.repeatDuration = 2.5
        animation.autoreverses = true

        let result = CAMediaTimingEvaluator.evaluate(animation, parentTime: 12.75, duration: 1)

        #expect(result.phase == .after)
        #expect(abs(result.progress - 0.5) < 0.000_001)
        #expect(abs(animation.totalDuration - 2.5) < 0.000_001)
    }

    @Test("repeatCount counts complete forward-reverse cycles")
    func repeatCountIncludesBothAutoreverseLegs() {
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.beginTime = 10
        animation.duration = 1
        animation.repeatCount = 2
        animation.autoreverses = true

        let forwardAgain = CAMediaTimingEvaluator.evaluate(animation, parentTime: 12.25, duration: 1)
        let completed = CAMediaTimingEvaluator.evaluate(animation, parentTime: 14, duration: 1)

        #expect(forwardAgain.phase == .active)
        #expect(abs(forwardAgain.progress - 0.25) < 0.000_001)
        #expect(completed.phase == .after)
        #expect(completed.progress == 0)
    }

    @Test("Interior repeat boundaries begin the next cycle")
    func repeatBoundaryUsesNextCycleStart() {
        let animation = CABasicAnimation(keyPath: "position.x")
        animation.beginTime = 10
        animation.duration = 1
        animation.repeatCount = 3

        let firstBoundary = CAMediaTimingEvaluator.evaluate(animation, parentTime: 11, duration: 1)
        let finalBoundary = CAMediaTimingEvaluator.evaluate(animation, parentTime: 13, duration: 1)

        #expect(firstBoundary.phase == .active)
        #expect(firstBoundary.progress == 0)
        #expect(firstBoundary.completedCycles == 1)
        #expect(finalBoundary.phase == .after)
        #expect(finalBoundary.progress == 1)
        #expect(finalBoundary.completedCycles == 2)
    }

    @Test("Cumulative basic animations add the previous cycle endpoint")
    func cumulativeBasicAnimationAccumulatesEndpoint() {
        let layer = CALayer()
        layer.position = .zero
        let animation = CABasicAnimation(keyPath: "position.x")
        animation.fromValue = CGFloat(10)
        animation.toValue = CGFloat(20)
        animation.duration = 1
        animation.repeatCount = 3
        animation.isCumulative = true
        layer.add(animation, forKey: "position.x")
        setStoredAnimationBeginTime(CACurrentMediaTime() - 1.5, on: layer, forKey: "position.x")

        let presentation = layer.presentation()
        #expect(abs((presentation?.position.x ?? 0) - 35) < 0.02)
    }

    @Test("Autoreversing cumulative animations use the cycle's reversed endpoint")
    func autoreversingCumulativeAnimationAccumulatesStartValue() {
        let layer = CALayer()
        layer.position = .zero
        let animation = CABasicAnimation(keyPath: "position.x")
        animation.fromValue = CGFloat(10)
        animation.toValue = CGFloat(20)
        animation.duration = 1
        animation.repeatCount = 2
        animation.autoreverses = true
        animation.isCumulative = true
        layer.add(animation, forKey: "position.x")
        setStoredAnimationBeginTime(CACurrentMediaTime() - 2.5, on: layer, forKey: "position.x")

        let presentation = layer.presentation()
        #expect(abs((presentation?.position.x ?? 0) - 25) < 0.02)
    }

    @Test("negative speed runs from timeOffset toward the start")
    func negativeSpeedMapsBackward() {
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.beginTime = 20
        animation.duration = 2
        animation.speed = -1
        animation.timeOffset = 2

        let midpoint = CAMediaTimingEvaluator.evaluate(animation, parentTime: 21, duration: 2)
        let completed = CAMediaTimingEvaluator.evaluate(animation, parentTime: 22, duration: 2)

        #expect(midpoint.phase == .active)
        #expect(abs(midpoint.progress - 0.5) < 0.000_001)
        #expect(completed.phase == .after)
        #expect(completed.progress == 0)
    }

    @Test("zero begin time and duration resolve when added")
    func addResolvesTransactionDefaults() {
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")

        CATransaction.begin()
        CATransaction.setAnimationDuration(0.75)
        layer.add(animation, forKey: "opacity")
        CATransaction.commit()

        guard let stored = layer.animation(forKey: "opacity") else {
            Issue.record("Expected stored animation")
            return
        }
        #expect(stored.beginTime > 0)
        #expect(stored.duration == 0.75)
        #expect(animation.beginTime == 0)
        #expect(animation.duration == 0)
    }

    @Test("layer timing controls presentation animation time")
    func layerLocalTimeDrivesAnimation() {
        let layer = CALayer()
        layer.opacity = 0
        layer.speed = 2

        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 4
        layer.add(animation, forKey: "opacity")

        let localNow = layer.convertTime(CACurrentMediaTime(), from: nil)
        setStoredAnimationBeginTime(localNow - 1, on: layer, forKey: "opacity")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }
        #expect(abs(presentation.opacity - 0.25) < 0.01)
    }

    @Test("paused layer does not complete its animations")
    func pausedLayerKeepsAnimationActive() {
        let layer = CALayer()
        layer.speed = 0
        layer.timeOffset = 0.5

        let animation = CABasicAnimation(keyPath: "opacity")
        animation.duration = 1
        layer.add(animation, forKey: "opacity")
        layer.processAnimationCompletions()

        #expect(layer.animation(forKey: "opacity") != nil)
    }

    @Test("Animation delegate starts at the active interval, not insertion")
    func delegateStartTracksActiveInterval() {
        let layer = CALayer()
        let delegate = AnimationDelegate()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 1
        animation.beginTime = CACurrentMediaTime() + 10
        animation.delegate = delegate

        layer.add(animation, forKey: "opacity")
        _ = layer.presentation()
        #expect(delegate.startCount == 0)

        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "opacity")
        layer.markDirty(.animations)
        _ = layer.presentation()
        #expect(delegate.startCount == 1)

        _ = layer.presentation()
        #expect(delegate.startCount == 1)
    }

    @Test("animation groups evaluate children in group basic time")
    func groupSpeedMapsChildTime() {
        let layer = CALayer()
        layer.opacity = 0

        let child = CABasicAnimation(keyPath: "opacity")
        child.fromValue = Float(0)
        child.toValue = Float(1)
        child.duration = 2
        child.fillMode = .both

        let group = CAAnimationGroup()
        group.animations = [child]
        group.duration = 2
        group.speed = 2
        group.fillMode = .both
        layer.add(group, forKey: "group")
        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.25, on: layer, forKey: "group")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }
        #expect(abs(presentation.opacity - 0.25) < 0.01)
    }
}
