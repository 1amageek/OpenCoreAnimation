//
//  CAShapeLayerStrokeClampingTests.swift
//  OpenCoreAnimationTests
//
//  Tests that CAShapeLayer.strokeStart / strokeEnd no longer clamp values at
//  the setter. Pre-fix they were clamped into [0, 1] which prevented spring
//  animations from overshooting the range. The contract is now: the setter
//  stores the raw value, and it is the tessellation stage's responsibility
//  to clamp at use-site.
//
//  No CoreGraphics import — uses only OpenCoreAnimation's public surface.
//

import Testing
import OpenCoreAnimation

@Suite("CAShapeLayer strokeStart / strokeEnd unclamped")
struct CAShapeLayerStrokeClampingTests {

    @Test("strokeStart accepts negative values without clamping")
    func strokeStartAcceptsNegative() {
        let layer = CAShapeLayer()
        layer.strokeStart = -0.5
        #expect(layer.strokeStart == -0.5)
    }

    @Test("strokeStart accepts values greater than 1")
    func strokeStartAcceptsOvershoot() {
        let layer = CAShapeLayer()
        layer.strokeStart = 1.7
        #expect(layer.strokeStart == 1.7)
    }

    @Test("strokeEnd accepts values greater than 1 without clamping")
    func strokeEndAcceptsOvershoot() {
        let layer = CAShapeLayer()
        layer.strokeEnd = 1.5
        #expect(layer.strokeEnd == 1.5)
    }

    @Test("strokeEnd accepts negative values without clamping")
    func strokeEndAcceptsNegative() {
        let layer = CAShapeLayer()
        layer.strokeEnd = -0.25
        #expect(layer.strokeEnd == -0.25)
    }

    @Test("Default strokeStart is 0 and default strokeEnd is 1")
    func defaultStrokeValues() {
        let layer = CAShapeLayer()
        #expect(layer.strokeStart == 0)
        #expect(layer.strokeEnd == 1)
    }

    @Test("Setting back into range after overshoot restores the in-range value")
    func strokeStartRoundTrip() {
        let layer = CAShapeLayer()
        layer.strokeStart = 2.0
        layer.strokeStart = 0.25
        #expect(layer.strokeStart == 0.25)
    }
}
