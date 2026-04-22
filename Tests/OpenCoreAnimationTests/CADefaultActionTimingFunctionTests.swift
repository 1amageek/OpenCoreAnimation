//
//  CADefaultActionTimingFunctionTests.swift
//  OpenCoreAnimationTests
//
//  Tests that CALayer.defaultAction(forKey:) (and the CAShapeLayer override)
//  picks up the current CATransaction's animation timing function. Covers the
//  recent fix: when the transaction has set a timing function, the implicit
//  default action must use that function rather than a hard-coded `.default`.
//
//  No CoreGraphics import — uses only OpenCoreAnimation's public surface.
//

import Testing
import OpenCoreAnimation

@Suite("CALayer.defaultAction picks up CATransaction timing function")
struct CALayerDefaultActionTimingFunctionTests {

    @Test("CALayer.defaultAction uses the transaction's timing function when one is set")
    func caLayerDefaultActionUsesTransactionTiming() throws {
        let expected = CAMediaTimingFunction(name: .easeInEaseOut)

        CATransaction.begin()
        CATransaction.setAnimationTimingFunction(expected)
        defer { CATransaction.commit() }

        let action = CALayer.defaultAction(forKey: "opacity")
        let animation = try #require(action as? CABasicAnimation)
        let timingFunction = try #require(animation.timingFunction)
        #expect(timingFunction == expected)
    }

    @Test("CALayer.defaultAction falls back to .default timing when transaction has no timing function")
    func caLayerDefaultActionFallsBackToDefault() throws {
        // Start a fresh transaction with no explicit timing function. The
        // implementation should produce a CAMediaTimingFunction(name: .default).
        CATransaction.begin()
        defer { CATransaction.commit() }

        let action = CALayer.defaultAction(forKey: "opacity")
        let animation = try #require(action as? CABasicAnimation)
        let timingFunction = try #require(animation.timingFunction)
        #expect(timingFunction == CAMediaTimingFunction(name: .default))
    }

    @Test("CALayer.defaultAction returns nil for non-animatable keys")
    func caLayerDefaultActionNilForUnknownKey() {
        let action = CALayer.defaultAction(forKey: "thisKeyIsNotAnimatable")
        #expect(action == nil)
    }

    @Test("CAShapeLayer.defaultAction uses transaction timing for shape-specific keys")
    func caShapeLayerDefaultActionUsesTransactionTiming() throws {
        let expected = CAMediaTimingFunction(name: .easeIn)

        CATransaction.begin()
        CATransaction.setAnimationTimingFunction(expected)
        defer { CATransaction.commit() }

        let action = CAShapeLayer.defaultAction(forKey: "strokeEnd")
        let animation = try #require(action as? CABasicAnimation)
        let timingFunction = try #require(animation.timingFunction)
        #expect(timingFunction == expected)
    }

    @Test("CAShapeLayer.defaultAction delegates to CALayer for inherited keys and still uses transaction timing")
    func caShapeLayerInheritedKeyUsesTransactionTiming() throws {
        let expected = CAMediaTimingFunction(name: .easeOut)

        CATransaction.begin()
        CATransaction.setAnimationTimingFunction(expected)
        defer { CATransaction.commit() }

        // "opacity" is not a shape-layer-specific key — it comes from the
        // parent CALayer's animatable set.
        let action = CAShapeLayer.defaultAction(forKey: "opacity")
        let animation = try #require(action as? CABasicAnimation)
        let timingFunction = try #require(animation.timingFunction)
        #expect(timingFunction == expected)
    }
}
