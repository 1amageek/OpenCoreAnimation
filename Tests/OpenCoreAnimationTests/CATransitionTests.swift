import Foundation
import Testing
#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif
@testable import OpenCoreAnimation

@Suite("CATransition Tests")
struct CATransitionTests {
    private let epsilon: CGFloat = 0.01

    private func assertApproximatelyEqual(_ lhs: CGFloat, _ rhs: CGFloat, tolerance: CGFloat = 0.0001) -> Bool {
        abs(lhs - rhs) < tolerance
    }

    @Test("Fade transition interpolates opacity")
    func fadeTransitionInterpolatesOpacity() {
        let layer = CALayer()

        let transition = CATransition()
        transition.type = .fade
        transition.duration = 1
        layer.add(transition, forKey: "fade")
        setStoredAnimationAddedTime(CACurrentMediaTime() - 0.25, on: layer, forKey: "fade")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(assertApproximatelyEqual(CGFloat(presentation.opacity), 0.25, tolerance: epsilon))
    }

    @Test("Push transition applies directional offset")
    func pushTransitionUsesSubtypeOffset() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 40)
        layer.position = CGPoint(x: 10, y: 20)

        let transition = CATransition()
        transition.type = .push
        transition.subtype = .fromRight
        transition.duration = 1
        layer.add(transition, forKey: "push")
        setStoredAnimationAddedTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "push")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(assertApproximatelyEqual(presentation.position.x, 60, tolerance: epsilon))
        #expect(assertApproximatelyEqual(presentation.position.y, 20, tolerance: epsilon))
    }

    @Test("MoveIn transition fades and moves the presentation layer")
    func moveInTransitionFadesAndMoves() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 50, height: 20)
        layer.position = CGPoint(x: 10, y: 20)

        let transition = CATransition()
        transition.type = .moveIn
        transition.subtype = .fromLeft
        transition.duration = 1
        layer.add(transition, forKey: "moveIn")
        setStoredAnimationAddedTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "moveIn")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(assertApproximatelyEqual(presentation.position.x, -15, tolerance: epsilon))
        #expect(assertApproximatelyEqual(CGFloat(presentation.opacity), 0.5, tolerance: epsilon))
    }

    @Test("Reveal transition respects start and end progress range")
    func revealTransitionUsesProgressRange() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 80, height: 40)
        layer.position = CGPoint(x: 30, y: 20)

        let transition = CATransition()
        transition.type = .reveal
        transition.subtype = .fromTop
        transition.startProgress = 0.25
        transition.endProgress = 0.75
        transition.duration = 1
        layer.add(transition, forKey: "reveal")
        setStoredAnimationAddedTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "reveal")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(assertApproximatelyEqual(CGFloat(presentation.opacity), 0.5, tolerance: epsilon))
        #expect(assertApproximatelyEqual(presentation.position.x, 30, tolerance: epsilon))
        #expect(assertApproximatelyEqual(presentation.position.y, 26, tolerance: epsilon))
    }

    @Test("Backwards fill mode applies the initial transition state before beginTime")
    func transitionBackwardsFillModeUsesInitialState() {
        let layer = CALayer()

        let transition = CATransition()
        transition.type = .fade
        transition.duration = 1
        transition.beginTime = 0.5
        transition.fillMode = .backwards
        layer.add(transition, forKey: "fade")
        setStoredAnimationAddedTime(CACurrentMediaTime(), on: layer, forKey: "fade")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(assertApproximatelyEqual(CGFloat(presentation.opacity), 0, tolerance: epsilon))
    }

    @Test("Removed fill mode leaves the model state untouched before beginTime")
    func transitionRemovedFillModeSkipsPreStartEffect() {
        let layer = CALayer()
        layer.opacity = 0.8

        let transition = CATransition()
        transition.type = .fade
        transition.duration = 1
        transition.beginTime = 0.5
        transition.fillMode = .removed
        layer.add(transition, forKey: "fade")
        setStoredAnimationAddedTime(CACurrentMediaTime(), on: layer, forKey: "fade")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(assertApproximatelyEqual(CGFloat(presentation.opacity), 0.8, tolerance: epsilon))
    }
}
