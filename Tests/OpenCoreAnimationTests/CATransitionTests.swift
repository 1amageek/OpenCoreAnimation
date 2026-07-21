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

    @Test("Fade transition produces a two-state render descriptor")
    func fadeTransitionProducesRenderDescriptor() throws {
        let layer = CALayer()
        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)

        let transition = CATransition()
        transition.type = .fade
        transition.duration = 1
        layer.add(transition, forKey: "fade")
        layer.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.25, on: layer, forKey: "fade")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        let state = try #require(presentation._transitionRenderState)
        #expect(assertApproximatelyEqual(CGFloat(state.progress), 0.25, tolerance: epsilon))
        #expect(state.type == .fade)
        #expect(state.sourceLayer.backgroundColor?.components == [1, 0, 0, 1])
        #expect(presentation.backgroundColor?.components == [0, 0, 1, 1])
        #expect(presentation.opacity == 1)
    }

    @Test("Push transition preserves model geometry and records its direction")
    func pushTransitionUsesSubtypeDirection() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 40)
        layer.position = CGPoint(x: 10, y: 20)

        let transition = CATransition()
        transition.type = .push
        transition.subtype = .fromRight
        transition.duration = 1
        layer.add(transition, forKey: "push")
        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "push")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        let state = try #require(presentation._transitionRenderState)
        #expect(assertApproximatelyEqual(CGFloat(state.progress), 0.5, tolerance: epsilon))
        #expect(state.type == .push)
        #expect(state.subtype == .fromRight)
        #expect(presentation.position.x == 10)
        #expect(presentation.position.y == 20)
    }

    @Test("MoveIn transition records composition without mutating presentation state")
    func moveInTransitionRecordsComposition() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 50, height: 20)
        layer.position = CGPoint(x: 10, y: 20)

        let transition = CATransition()
        transition.type = .moveIn
        transition.subtype = .fromLeft
        transition.duration = 1
        layer.add(transition, forKey: "moveIn")
        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "moveIn")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        let state = try #require(presentation._transitionRenderState)
        #expect(state.type == .moveIn)
        #expect(state.subtype == .fromLeft)
        #expect(assertApproximatelyEqual(CGFloat(state.progress), 0.5, tolerance: epsilon))
        #expect(presentation.position.x == 10)
        #expect(presentation.position.y == 20)
        #expect(presentation.opacity == 1)
    }

    @Test("Reveal transition respects start and end progress range")
    func revealTransitionUsesProgressRange() throws {
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
        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "reveal")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        let state = try #require(presentation._transitionRenderState)
        #expect(state.type == .reveal)
        #expect(state.subtype == .fromTop)
        #expect(assertApproximatelyEqual(CGFloat(state.progress), 0.5, tolerance: epsilon))
        #expect(presentation.position.x == 30)
        #expect(presentation.position.y == 20)
        #expect(presentation.opacity == 1)
    }

    @Test("Backwards fill mode applies the initial transition state before beginTime")
    func transitionBackwardsFillModeUsesInitialState() {
        let layer = CALayer()

        let transition = CATransition()
        transition.type = .fade
        transition.duration = 1
        transition.beginTime = CACurrentMediaTime() + 0.5
        transition.fillMode = .backwards
        layer.add(transition, forKey: "fade")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(assertApproximatelyEqual(
            CGFloat(presentation._transitionRenderState?.progress ?? -1),
            0,
            tolerance: epsilon
        ))
    }

    @Test("Removed fill mode leaves the model state untouched before beginTime")
    func transitionRemovedFillModeSkipsPreStartEffect() {
        let layer = CALayer()
        layer.opacity = 0.8

        let transition = CATransition()
        transition.type = .fade
        transition.duration = 1
        transition.beginTime = CACurrentMediaTime() + 0.5
        transition.fillMode = .removed
        layer.add(transition, forKey: "fade")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(assertApproximatelyEqual(CGFloat(presentation.opacity), 0.8, tolerance: epsilon))
        #expect(presentation._transitionRenderState == nil)
    }

    @Test("Transition snapshot recursively preserves the prior sublayer tree")
    func transitionCapturesPriorSublayerTree() throws {
        let layer = CALayer()
        let priorChild = CALayer()
        priorChild.name = "prior"
        layer.addSublayer(priorChild)

        let transition = CATransition()
        transition.duration = 1
        layer.add(transition, forKey: "transition")

        priorChild.removeFromSuperlayer()
        let currentChild = CALayer()
        currentChild.name = "current"
        layer.addSublayer(currentChild)

        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "transition")
        let state = try #require(layer.presentation()?._transitionRenderState)
        #expect(state.sourceLayer.sublayers?.map(\.name) == ["prior"])
        #expect(layer.sublayers?.map(\.name) == ["current"])
    }

    @Test("Filter transition is not reported as a completed built-in composition")
    func filterTransitionDoesNotUseBuiltInCompositor() throws {
        let layer = CALayer()
        let transition = CATransition()
        transition.filter = "typed-filter-bridge-required"
        transition.duration = 1
        layer.add(transition, forKey: "filteredTransition")
        setStoredAnimationBeginTime(
            CACurrentMediaTime() - 0.5,
            on: layer,
            forKey: "filteredTransition"
        )

        let presentation = try #require(layer.presentation())
        #expect(presentation._transitionRenderState == nil)
    }
}
