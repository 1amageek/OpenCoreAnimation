import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CASpringAnimation perceptual API")
struct CASpringAnimationPerceptualTests {
    @Test("Default perceptual properties match QuartzCore")
    func defaultPerceptualProperties() {
        let spring = CASpringAnimation()

        #expect(spring.allowsOverdamping == false)
        #expect(abs(spring.perceptualDuration - (2 * .pi / 10)) < 1e-12)
        #expect(abs(spring.bounce - 0.5) < 1e-12)
        #expect(CASpringAnimation.defaultValue(forKey: "allowsOverdamping") as? Bool == false)
    }

    @Test("Perceptual initializer derives physical parameters and duration")
    func perceptualInitializer() {
        let spring = CASpringAnimation(perceptualDuration: 1, bounce: 0.5)

        #expect(spring.allowsOverdamping)
        #expect(abs(spring.mass - 1) < 1e-12)
        #expect(abs(spring.stiffness - (4 * .pi * .pi)) < 1e-12)
        #expect(abs(spring.damping - (2 * .pi)) < 1e-12)
        #expect(abs(spring.perceptualDuration - 1) < 1e-12)
        #expect(abs(spring.bounce - 0.5) < 1e-12)
        #expect(abs(spring.duration - spring.settlingDuration) < 1e-12)
        #expect(abs(spring.settlingDuration - 2.3438753795710707) < 0.02)
    }

    @Test("Negative bounce maps to an overdamped physical coefficient")
    func negativeBounceMapping() {
        let spring = CASpringAnimation(perceptualDuration: 1, bounce: -0.5)

        #expect(spring.allowsOverdamping)
        #expect(abs(spring.damping - (8 * .pi)) < 1e-12)
        #expect(abs(spring.bounce + 0.5) < 1e-12)
        #expect(abs(spring.duration - spring.settlingDuration) < 1e-12)
        #expect(abs(spring.settlingDuration - 2.6) < 0.2)
    }

    @Test("A bounce of negative one produces the QuartzCore infinite damping boundary")
    func infiniteDampingBoundary() {
        let spring = CASpringAnimation(perceptualDuration: 1, bounce: -1)

        #expect(spring.damping == .infinity)
        #expect(spring.bounce == -1)
        #expect(spring.settlingDuration == 0)
        #expect(spring.duration == 0)
        #expect(spring.springValue(at: 0) == 0)
        #expect(spring.springValue(at: 0.1) == 1)
    }

    @Test("Overdamping policy changes response without rewriting stored damping")
    func overdampingPolicy() {
        let clamped = CASpringAnimation()
        clamped.damping = 40

        let explicit = CASpringAnimation()
        explicit.damping = 40
        explicit.allowsOverdamping = true

        #expect(clamped.damping == 40)
        #expect(explicit.damping == 40)
        #expect(abs(clamped.bounce + 0.5) < 1e-12)
        #expect(abs(explicit.bounce + 0.5) < 1e-12)
        #expect(abs(clamped.settlingDuration - 1) < 1e-12)
        #expect(explicit.settlingDuration > clamped.settlingDuration)
        #expect(explicit.springValue(at: 0.5) < clamped.springValue(at: 0.5))
    }

    @Test("Spring copies retain perceptual response policy")
    func copyRetainsPerceptualState() {
        let source = CASpringAnimation(perceptualDuration: 0.75, bounce: -0.25)
        source.initialVelocity = 3

        let copy = source.copy()

        #expect(copy !== source)
        #expect(copy.allowsOverdamping == source.allowsOverdamping)
        #expect(copy.mass == source.mass)
        #expect(copy.stiffness == source.stiffness)
        #expect(copy.damping == source.damping)
        #expect(copy.initialVelocity == source.initialVelocity)
        #expect(copy.perceptualDuration == source.perceptualDuration)
        #expect(copy.bounce == source.bounce)
    }

    @Test("Settling estimates match measured QuartzCore regimes")
    func settlingDurationConformance() {
        let underdamped = CASpringAnimation()
        #expect(abs(underdamped.settlingDuration - 1.4727003346780927) < 0.02)

        let lightlyDamped = CASpringAnimation()
        lightlyDamped.damping = 1
        #expect(abs(lightlyDamped.settlingDuration - 13.91321015404625) < 0.02)

        let critical = CASpringAnimation()
        critical.damping = 20
        #expect(abs(critical.settlingDuration - 1) < 1e-12)

        let highStoredDamping = CASpringAnimation()
        highStoredDamping.damping = 100
        #expect(abs(highStoredDamping.settlingDuration - 1) < 1e-12)
    }
}
