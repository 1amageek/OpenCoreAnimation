import Testing
@testable import OpenCoreAnimation

@Suite("CALayer transform component animation evaluation")
struct CATransformComponentAnimationTests {
    private let epsilon: CGFloat = 0.0001

    private func presentation(
        for animation: CAPropertyAnimation,
        modelTransform: CATransform3D,
        elapsed: CFTimeInterval
    ) throws -> CALayer {
        let layer = CALayer()
        layer.transform = modelTransform
        animation.duration = 1
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        animation.speed = 0
        animation.timeOffset = elapsed
        layer.add(animation, forKey: "component")
        return try #require(layer.presentation())
    }

    private func rotation(_ x: CGFloat, _ y: CGFloat, _ z: CGFloat) -> CATransform3D {
        CATransform3DConcat(
            CATransform3DConcat(
                CATransform3DMakeRotation(x, 1, 0, 0),
                CATransform3DMakeRotation(y, 0, 1, 0)
            ),
            CATransform3DMakeRotation(z, 0, 0, 1)
        )
    }

    private func expectEqual(_ actual: CATransform3D, _ expected: CATransform3D) {
        let actualValues = [
            actual.m11, actual.m12, actual.m13, actual.m14,
            actual.m21, actual.m22, actual.m23, actual.m24,
            actual.m31, actual.m32, actual.m33, actual.m34,
            actual.m41, actual.m42, actual.m43, actual.m44,
        ]
        let expectedValues = [
            expected.m11, expected.m12, expected.m13, expected.m14,
            expected.m21, expected.m22, expected.m23, expected.m24,
            expected.m31, expected.m32, expected.m33, expected.m34,
            expected.m41, expected.m42, expected.m43, expected.m44,
        ]
        for (actualValue, expectedValue) in zip(actualValues, expectedValues) {
            #expect(abs(actualValue - expectedValue) < epsilon)
        }
    }

    @Test("Basic scale components replace decomposed scale instead of multiplying it")
    func basicScaleReplacesComponent() throws {
        let animation = CABasicAnimation(keyPath: "transform.scale.x")
        animation.toValue = CGFloat(4)

        let presentation = try presentation(
            for: animation,
            modelTransform: CATransform3DMakeScale(2, 3, 4),
            elapsed: 0.5
        )

        expectEqual(presentation.transform, CATransform3DMakeScale(3, 3, 4))
    }

    @Test("Basic translation byValue uses the decomposed model translation")
    func basicTranslationUsesModelComponent() throws {
        let animation = CABasicAnimation(keyPath: "transform.translation.x")
        animation.byValue = CGFloat(10)
        var model = CATransform3DMakeScale(2, 3, 4)
        model.m41 = 6
        model.m42 = 7
        model.m43 = 8

        let presentation = try presentation(for: animation, modelTransform: model, elapsed: 0.5)

        var expected = model
        expected.m41 = 11
        expectEqual(presentation.transform, expected)
    }

    @Test("Keyframe rotation replaces one Euler component and preserves the others")
    func keyframeRotationPreservesOtherComponents() throws {
        let animation = CAKeyframeAnimation(keyPath: "transform.rotation.x")
        animation.values = [CGFloat(0.2), CGFloat(0.6)]

        let presentation = try presentation(
            for: animation,
            modelTransform: rotation(0.2, 0.3, 0.4),
            elapsed: 0.5
        )

        expectEqual(presentation.transform, rotation(0.4, 0.3, 0.4))
    }

    @Test("Keyframe scale and translation components preserve unrelated components")
    func keyframeScaleAndTranslationPreserveComponents() throws {
        let scaleAnimation = CAKeyframeAnimation(keyPath: "transform.scale.y")
        scaleAnimation.values = [CGFloat(3), CGFloat(7)]
        let scaled = try presentation(
            for: scaleAnimation,
            modelTransform: CATransform3DMakeScale(2, 3, 4),
            elapsed: 0.5
        )
        expectEqual(scaled.transform, CATransform3DMakeScale(2, 5, 4))

        let translationAnimation = CAKeyframeAnimation(keyPath: "transform.translation")
        translationAnimation.values = [CGSize.zero, CGSize(width: 20, height: 30)]
        let translated = try presentation(
            for: translationAnimation,
            modelTransform: CATransform3DMakeScale(2, 3, 4),
            elapsed: 0.5
        )
        var expected = CATransform3DMakeScale(2, 3, 4)
        expected.m41 = 10
        expected.m42 = 15
        expectEqual(translated.transform, expected)
    }

    @Test("Discrete and cubic keyframes apply transform component values")
    func discreteAndCubicModesApplyComponents() throws {
        let discrete = CAKeyframeAnimation(keyPath: "transform.translation.z")
        discrete.values = [CGFloat(4), CGFloat(12)]
        discrete.calculationMode = .discrete
        let discretePresentation = try presentation(
            for: discrete,
            modelTransform: CATransform3DIdentity,
            elapsed: 0.75
        )
        #expect(abs(discretePresentation.transform.m43 - 4) < epsilon)

        let cubic = CAKeyframeAnimation(keyPath: "transform.scale.z")
        cubic.values = [CGFloat(1), CGFloat(3)]
        cubic.calculationMode = .cubic
        let cubicPresentation = try presentation(
            for: cubic,
            modelTransform: CATransform3DIdentity,
            elapsed: 0.5
        )
        #expect(abs(cubicPresentation.transform.m33 - 2) < epsilon)
    }

    @Test("Additive keyframes add component values to the presentation transform")
    func additiveKeyframesAddComponents() throws {
        let animation = CAKeyframeAnimation(keyPath: "transform.scale.x")
        animation.values = [CGFloat(0), CGFloat(1)]
        animation.isAdditive = true

        let presentation = try presentation(
            for: animation,
            modelTransform: CATransform3DMakeScale(2, 3, 4),
            elapsed: 0.5
        )

        expectEqual(presentation.transform, CATransform3DMakeScale(2.5, 3, 4))
    }

    @Test("Cumulative keyframes carry terminal component values into repeat cycles")
    func cumulativeKeyframesCarryTerminalValue() throws {
        let animation = CAKeyframeAnimation(keyPath: "transform.translation.x")
        animation.values = [CGFloat(0), CGFloat(10)]
        animation.isCumulative = true
        animation.repeatCount = 3

        let presentation = try presentation(
            for: animation,
            modelTransform: CATransform3DIdentity,
            elapsed: 1.5
        )

        #expect(abs(presentation.transform.m41 - 15) < epsilon)
    }

    @Test("Every scalar transform component alias reaches presentation state")
    func scalarComponentAliasesApply() throws {
        let angle: CGFloat = 0.5
        let cases: [(keyPath: String, values: [Any], expected: CATransform3D)] = [
            ("transform.rotation", [CGFloat(0), angle * 2], CATransform3DMakeRotation(angle, 0, 0, 1)),
            ("transform.rotation.x", [CGFloat(0), angle * 2], CATransform3DMakeRotation(angle, 1, 0, 0)),
            ("transform.rotation.y", [CGFloat(0), angle * 2], CATransform3DMakeRotation(angle, 0, 1, 0)),
            ("transform.rotation.z", [CGFloat(0), angle * 2], CATransform3DMakeRotation(angle, 0, 0, 1)),
            ("transform.scale", [CGFloat(1), CGFloat(3)], CATransform3DMakeScale(2, 2, 2)),
            ("transform.scale.x", [CGFloat(1), CGFloat(3)], CATransform3DMakeScale(2, 1, 1)),
            ("transform.scale.y", [CGFloat(1), CGFloat(3)], CATransform3DMakeScale(1, 2, 1)),
            ("transform.scale.z", [CGFloat(1), CGFloat(3)], CATransform3DMakeScale(1, 1, 2)),
            ("transform.translation.x", [CGFloat(0), CGFloat(16)], CATransform3DMakeTranslation(8, 0, 0)),
            ("transform.translation.y", [CGFloat(0), CGFloat(16)], CATransform3DMakeTranslation(0, 8, 0)),
            ("transform.translation.z", [CGFloat(0), CGFloat(16)], CATransform3DMakeTranslation(0, 0, 8)),
        ]

        for testCase in cases {
            let animation = CAKeyframeAnimation(keyPath: testCase.keyPath)
            animation.values = testCase.values
            let result = try presentation(
                for: animation,
                modelTransform: CATransform3DIdentity,
                elapsed: 0.5
            )
            expectEqual(result.transform, testCase.expected)
        }
    }

    @Test("Additive full-transform keyframes concatenate with the model transform")
    func additiveFullTransformKeyframesConcatenate() throws {
        let animation = CAKeyframeAnimation(keyPath: "transform")
        animation.values = [
            CATransform3DIdentity,
            CATransform3DMakeTranslation(10, 0, 0),
        ]
        animation.isAdditive = true

        let presentation = try presentation(
            for: animation,
            modelTransform: CATransform3DMakeScale(2, 2, 1),
            elapsed: 0.5
        )

        var expected = CATransform3DMakeScale(2, 2, 1)
        expected.m41 = 10
        expectEqual(presentation.transform, expected)
    }
}
