import Testing
@testable import OpenCoreAnimation

@MainActor
@Suite("CAMediaTiming validation")
struct CAMediaTimingValidationTests {
    private final class Delegate: CAAnimationDelegate {
        var startCount = 0
        var stopEvents: [Bool] = []

        func animationDidStart(_ anim: CAAnimation) {
            startCount += 1
        }

        func animationDidStop(_ anim: CAAnimation, finished flag: Bool) {
            stopEvents.append(flag)
        }
    }

    @Test("Every non-finite timing input produces an invalid result")
    func evaluatorRejectsNonFiniteInputs() {
        for field in 0..<7 {
            let animation = CABasicAnimation(keyPath: "zPosition")
            animation.beginTime = 0
            animation.duration = 1
            var parentTime: CFTimeInterval = 0.5
            var evaluatedDuration: CFTimeInterval = 1

            switch field {
            case 0:
                animation.beginTime = .nan
            case 1:
                animation.timeOffset = .infinity
            case 2:
                animation.speed = .nan
            case 3:
                animation.repeatCount = -.infinity
            case 4:
                animation.repeatDuration = .nan
            case 5:
                parentTime = .nan
            default:
                evaluatedDuration = .nan
            }

            let result = CAMediaTimingEvaluator.evaluate(
                animation,
                parentTime: parentTime,
                duration: evaluatedDuration
            )
            #expect(!result.isValid)
            #expect(result.progress.isNaN)
        }

        let invalidDuration = CABasicAnimation()
        invalidDuration.duration = .nan
        invalidDuration.speed = 0
        #expect(invalidDuration.totalDuration.isNaN)

        let invalidRepeat = CABasicAnimation()
        invalidRepeat.duration = 1
        invalidRepeat.repeatCount = -.infinity
        #expect(invalidRepeat.totalDuration.isNaN)

        let indefinite = CABasicAnimation()
        indefinite.duration = 1
        indefinite.repeatCount = .infinity
        let indefiniteResult = CAMediaTimingEvaluator.evaluate(
            indefinite,
            parentTime: 100,
            duration: 1
        )
        #expect(indefiniteResult.isValid)
        #expect(indefiniteResult.phase == .active)
        #expect(indefinite.totalDuration == .infinity)
    }

    @Test("Invalid timing leaves presentation values unchanged")
    func presentationFailsAtomically() throws {
        for invalidAnimation in invalidPropertyAnimations() {
            let layer = CALayer()
            layer.zPosition = 7
            layer.add(invalidAnimation, forKey: "invalidTiming")
            #expect(try #require(layer.presentation()).zPosition == 7)
        }
    }

    @Test("Invalid timing reports unsuccessful completion and is removed")
    func completionFailsExplicitly() {
        let layer = CALayer()
        let delegate = Delegate()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.duration = 1
        animation.speed = .nan
        animation.delegate = delegate
        layer.add(animation, forKey: "invalidCompletion")

        layer.processAnimationCompletions()

        #expect(delegate.startCount == 0)
        #expect(delegate.stopEvents == [false])
        #expect(layer.animation(forKey: "invalidCompletion") == nil)
    }

    @Test("Invalid animations do not affect frame-rate arbitration")
    func schedulingIgnoresInvalidTiming() {
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.duration = 1
        animation.speed = .nan
        animation.preferredFrameRateRange = CAFrameRateRange(
            minimum: 80,
            maximum: 120,
            preferred: 120
        )
        layer.add(animation, forKey: "invalidScheduling")

        let engine = CAAnimationEngine()
        engine.preferredFrameRate = 30
        engine.rootLayer = layer

        #expect(engine.resolvedFrameRateRange(at: CACurrentMediaTime()) == CAFrameRateRange(
            minimum: 30,
            maximum: 30,
            preferred: 30
        ))
    }

    private func invalidPropertyAnimations() -> [CABasicAnimation] {
        let invalidSpeed = animation()
        invalidSpeed.speed = .nan

        let invalidDuration = animation()
        invalidDuration.duration = .nan

        let invalidOffset = animation()
        invalidOffset.timeOffset = .infinity

        return [invalidSpeed, invalidDuration, invalidOffset]
    }

    private func animation() -> CABasicAnimation {
        let animation = CABasicAnimation(keyPath: "zPosition")
        animation.fromValue = CGFloat(0)
        animation.toValue = CGFloat(100)
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
        return animation
    }
}
