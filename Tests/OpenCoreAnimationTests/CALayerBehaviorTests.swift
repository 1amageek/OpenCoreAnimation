import Testing
#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif
@testable import OpenCoreAnimation

@Suite("CALayer Bounds Behavior Tests")
struct CALayerBoundsBehaviorTests {
    @Test("Changing bounds marks layer dirty when needsDisplayOnBoundsChange is enabled")
    func boundsChangeMarksNeedsDisplay() {
        let layer = CALayer()
        layer.needsDisplayOnBoundsChange = true

        #expect(layer.needsDisplay() == false)
        layer.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)
        #expect(layer.needsDisplay() == true)
    }

    @Test("Changing bounds does not mark layer dirty when needsDisplayOnBoundsChange is disabled")
    func boundsChangeDoesNotMarkNeedsDisplayWithoutFlag() {
        let layer = CALayer()

        #expect(layer.needsDisplay() == false)
        layer.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)
        #expect(layer.needsDisplay() == false)
    }
}

@Suite("CALayer Time Conversion Tests")
struct CALayerTimeConversionTests {
    @Test("Converting time from parent to child matches Core Animation timing model")
    func convertTimeFromParentToChild() {
        let root = CALayer()
        root.speed = 2
        root.timeOffset = 3

        let child = CALayer()
        child.speed = 4
        child.timeOffset = 5
        root.addSublayer(child)

        // Cross-checked against QuartzCore on macOS.
        #expect(child.convertTime(10, from: root) == 45)
        #expect(root.convertTime(10, from: child) == 1.25)
    }

    @Test("Converting time to and from nil uses global media time")
    func convertTimeToAndFromGlobalTime() {
        let root = CALayer()
        root.speed = 2
        root.timeOffset = 3
        root.beginTime = 1

        let child = CALayer()
        child.speed = 4
        child.timeOffset = 5
        child.beginTime = 6
        root.addSublayer(child)

        // Cross-checked against QuartzCore on macOS.
        #expect(root.convertTime(10, from: nil) == 21)
        #expect(root.convertTime(10, to: nil) == 4.5)
        #expect(child.convertTime(10, from: nil) == 65)
        #expect(child.convertTime(10, to: nil) == 3.125)
    }

    @Test("Converting time across siblings uses the common ancestor correctly")
    func convertTimeAcrossSiblings() {
        let root = CALayer()
        root.speed = 2
        root.timeOffset = 3
        root.beginTime = 1

        let first = CALayer()
        first.speed = 4
        first.timeOffset = 5
        first.beginTime = 6

        let second = CALayer()
        second.speed = 0.5
        second.timeOffset = 7
        second.beginTime = 8

        root.addSublayer(first)
        root.addSublayer(second)

        // Cross-checked against QuartzCore on macOS.
        #expect(first.convertTime(10, from: root) == 21)
        #expect(root.convertTime(10, from: first) == 7.25)
        #expect(second.convertTime(10, from: first) == 0)
        #expect(first.convertTime(10, from: second) == 37)
    }

    @Test("Converting from parent clamps to zero before beginTime")
    func convertTimeClampsBeforeBeginTime() {
        let root = CALayer()

        let child = CALayer()
        child.speed = 0.5
        child.timeOffset = 7
        child.beginTime = 8
        root.addSublayer(child)

        // Cross-checked against QuartzCore on macOS.
        #expect(child.convertTime(7, from: root) == 0)
        #expect(child.convertTime(8, from: root) == 7)
        #expect(child.convertTime(10, from: root) == 8)
    }

    @Test("Converting from a paused layer collapses its upstream time to zero")
    func convertTimeFromPausedLayer() {
        let root = CALayer()
        root.speed = 2
        root.timeOffset = 3
        root.beginTime = 1

        let child = CALayer()
        child.speed = 0
        child.timeOffset = 7
        child.beginTime = 8
        root.addSublayer(child)

        // Cross-checked against QuartzCore on macOS.
        #expect(child.convertTime(10, from: root) == 7)
        #expect(root.convertTime(10, from: child) == 0)
        #expect(child.convertTime(10, to: nil) == -0.5)
    }
}

@Suite("CAValueFunction Animation Tests")
struct CAValueFunctionAnimationTests {
    private let epsilon: CGFloat = 0.0001

    private func isApproximatelyEqual(_ lhs: CATransform3D, _ rhs: CATransform3D, tolerance: CGFloat = 0.0001) -> Bool {
        abs(lhs.m11 - rhs.m11) < tolerance &&
        abs(lhs.m12 - rhs.m12) < tolerance &&
        abs(lhs.m13 - rhs.m13) < tolerance &&
        abs(lhs.m14 - rhs.m14) < tolerance &&
        abs(lhs.m21 - rhs.m21) < tolerance &&
        abs(lhs.m22 - rhs.m22) < tolerance &&
        abs(lhs.m23 - rhs.m23) < tolerance &&
        abs(lhs.m24 - rhs.m24) < tolerance &&
        abs(lhs.m31 - rhs.m31) < tolerance &&
        abs(lhs.m32 - rhs.m32) < tolerance &&
        abs(lhs.m33 - rhs.m33) < tolerance &&
        abs(lhs.m34 - rhs.m34) < tolerance &&
        abs(lhs.m41 - rhs.m41) < tolerance &&
        abs(lhs.m42 - rhs.m42) < tolerance &&
        abs(lhs.m43 - rhs.m43) < tolerance &&
        abs(lhs.m44 - rhs.m44) < tolerance
    }

    @Test("RotateY value function animates transform from scalar input")
    func rotateYValueFunctionAnimation() {
        let layer = CALayer()

        let animation = CABasicAnimation(keyPath: "transform")
        animation.valueFunction = CAValueFunction(name: .rotateY)
        animation.fromValue = CGFloat(0)
        animation.toValue = CGFloat.pi
        animation.duration = 1
        layer.add(animation, forKey: "rotateY")
        animation.addedTime = CACurrentMediaTime() - 0.5

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        let expected = CATransform3DMakeRotation(.pi / 2, 0, 1, 0)
        #expect(isApproximatelyEqual(presentation.transform, expected, tolerance: epsilon))
    }

    @Test("Scale value function uses 1 as the implicit starting value")
    func scaleValueFunctionImplicitStart() {
        let layer = CALayer()

        let animation = CABasicAnimation(keyPath: "transform")
        animation.valueFunction = CAValueFunction(name: .scale)
        animation.toValue = CGFloat(3)
        animation.duration = 1
        layer.add(animation, forKey: "scale")
        animation.addedTime = CACurrentMediaTime() - 0.5

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }

        #expect(abs(presentation.transform.m11 - 2) < epsilon)
        #expect(abs(presentation.transform.m22 - 2) < epsilon)
        #expect(abs(presentation.transform.m33 - 2) < epsilon)
    }
}

@Suite("CAFilter Tests")
struct CAFilterTests {
    @Test("Factory helpers expose typed parameters")
    func filterFactoriesExposeParameters() {
        let blur = CAFilter.blur(radius: 6)
        let brightness = CAFilter.brightness(0.25)
        let contrast = CAFilter.contrast(1.5)
        let saturation = CAFilter.saturation(0.75)

        #expect(blur.name == "CIGaussianBlur")
        #expect(blur.blurRadius == 6)
        #expect(brightness.brightnessAmount == 0.25)
        #expect(contrast.contrastAmount == 1.5)
        #expect(saturation.saturationAmount == 0.75)
    }

    @Test("Layer filter helpers aggregate CAFilter entries only")
    func layerFilterHelpersAggregateCAFilters() {
        let layer = CALayer()
        layer.filters = [
            CAFilter.blur(radius: 3),
            "ignored",
            CAFilter.blur(radius: 4),
            CAFilter.brightness(0.2)
        ]

        #expect(layer.activeFilters.count == 3)
        #expect(layer.hasBlurFilter == true)
        #expect(layer.totalBlurRadius == 7)
    }

    @Test("Supported filter operations preserve order and skip unsupported entries")
    func supportedFilterOperationsPreserveOrder() {
        let layer = CALayer()
        layer.filters = [
            CAFilter.blur(radius: 3),
            "ignored",
            CAFilter.brightness(0.2),
            CAFilter.colorInvert(),
            CAFilter(type: .sepiaTone, parameters: ["inputIntensity": 0.8]),
            CAFilter.contrast(1.5)
        ]

        #expect(layer.supportedFilterOperations == [
            .gaussianBlur(radius: 3),
            .brightness(amount: 0.2),
            .colorInvert,
            .contrast(amount: 1.5),
        ])
    }

    @Test("Hashable and equality include filter parameters")
    func filterEqualityIncludesParameters() {
        let blur3 = CAFilter.blur(radius: 3)
        let blur4 = CAFilter.blur(radius: 4)

        #expect(blur3 != blur4)
        #expect(Set([blur3, blur4]).count == 2)
    }
}
