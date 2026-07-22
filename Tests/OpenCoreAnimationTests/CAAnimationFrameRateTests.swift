import Foundation
import Testing
@testable import OpenCoreAnimation

@MainActor
@Suite("CAAnimation frame-rate hints")
struct CAAnimationFrameRateTests {
    @Test("Animation defaults and copies preserve the QuartzCore range contract")
    func defaultAndCopy() {
        let animation = CABasicAnimation(keyPath: "opacity")
        #expect(animation.preferredFrameRateRange == .default)
        #expect(CAAnimation.defaultValue(forKey: "preferredFrameRateRange") == nil)

        animation.preferredFrameRateRange = CAFrameRateRange(
            minimum: 30,
            maximum: 60,
            preferred: 48
        )
        let copy = animation.copy()
        #expect(copy.preferredFrameRateRange == animation.preferredFrameRateRange)
    }

    @Test("The engine uses its baseline when no active animation requests a range")
    func baselineRange() {
        let engine = CAAnimationEngine()
        engine.preferredFrameRate = 45
        engine.rootLayer = CALayer()

        #expect(engine.resolvedFrameRateRange(at: CACurrentMediaTime()) == CAFrameRateRange(
            minimum: 45,
            maximum: 45,
            preferred: 45
        ))
    }

    @Test("Active animation ranges are arbitrated across the complete layer tree")
    func activeTreeArbitration() {
        let root = CALayer()
        let child = CALayer()
        root.addSublayer(child)

        let economical = CABasicAnimation(keyPath: "opacity")
        economical.duration = 100
        economical.preferredFrameRateRange = CAFrameRateRange(
            minimum: 15,
            maximum: 30,
            preferred: 24
        )
        root.add(economical, forKey: "economical")

        let fluid = CABasicAnimation(keyPath: "position.x")
        fluid.duration = 100
        fluid.preferredFrameRateRange = CAFrameRateRange(
            minimum: 60,
            maximum: 120,
            preferred: 120
        )
        child.add(fluid, forKey: "fluid")

        let engine = CAAnimationEngine()
        engine.rootLayer = root
        let range = engine.resolvedFrameRateRange(at: CACurrentMediaTime())

        #expect(range.minimum == 60)
        #expect(range.maximum == 120)
        #expect(range.preferred == 120)
    }

    @Test("Future animations do not change the current callback range")
    func futureAnimationIsInactive() {
        let layer = CALayer()
        let now = CACurrentMediaTime()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.beginTime = now + 10
        animation.duration = 1
        animation.preferredFrameRateRange = CAFrameRateRange(
            minimum: 80,
            maximum: 120,
            preferred: 120
        )
        layer.add(animation, forKey: "future")

        let engine = CAAnimationEngine()
        engine.preferredFrameRate = 30
        engine.rootLayer = layer

        #expect(engine.resolvedFrameRateRange(at: now) == CAFrameRateRange(
            minimum: 30,
            maximum: 30,
            preferred: 30
        ))
    }

    @Test("Animation-group child hints participate in arbitration")
    func groupChildRange() {
        let child = CABasicAnimation(keyPath: "opacity")
        child.preferredFrameRateRange = CAFrameRateRange(
            minimum: 80,
            maximum: 120,
            preferred: 120
        )

        let group = CAAnimationGroup()
        group.duration = 100
        group.animations = [child]

        let layer = CALayer()
        layer.add(group, forKey: "group")

        let engine = CAAnimationEngine()
        engine.rootLayer = layer
        #expect(engine.resolvedFrameRateRange(at: CACurrentMediaTime()).preferred == 120)
    }

    @Test("Inactive group children do not participate in arbitration")
    func futureGroupChildRange() {
        let child = CABasicAnimation(keyPath: "opacity")
        child.beginTime = 50
        child.duration = 1
        child.preferredFrameRateRange = CAFrameRateRange(
            minimum: 80,
            maximum: 120,
            preferred: 120
        )

        let group = CAAnimationGroup()
        group.duration = 100
        group.animations = [child]
        group.preferredFrameRateRange = CAFrameRateRange(
            minimum: 15,
            maximum: 30,
            preferred: 30
        )

        let layer = CALayer()
        layer.add(group, forKey: "group")

        let engine = CAAnimationEngine()
        engine.rootLayer = layer
        #expect(engine.resolvedFrameRateRange(at: CACurrentMediaTime()).preferred == 30)
    }

    @Test("Starting the engine submits the active animation range to its display link")
    func startSubmitsResolvedRange() {
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.duration = 100
        animation.preferredFrameRateRange = CAFrameRateRange(
            minimum: 30,
            maximum: 60,
            preferred: 60
        )

        let layer = CALayer()
        layer.add(animation, forKey: "animation")

        let engine = CAAnimationEngine()
        engine.rootLayer = layer
        engine.start()
        defer { engine.stop() }

        #expect(engine.displayLinkFrameRateRange == animation.preferredFrameRateRange)
    }
}
