import Testing
@testable import OpenCoreAnimation

@Suite("CAValueFunction property animation evaluation")
struct CAValueFunctionKeyframeTests {
    private let epsilon: CGFloat = 0.0001

    private func presentation(
        for animation: CAPropertyAnimation,
        modelTransform: CATransform3D = CATransform3DIdentity,
        elapsed: CFTimeInterval = 0.5
    ) throws -> CATransform3D {
        let layer = CALayer()
        layer.transform = modelTransform
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = elapsed
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        layer.add(animation, forKey: "valueFunction")
        return try #require(layer.presentation()).transform
    }

    @Test("Initializers reject unknown names and aggregate functions enforce arity")
    func namesAndArityAreValidated() throws {
        #expect(CAValueFunction(name: CAValueFunctionName(rawValue: "unknown")) == nil)
        let translate = try #require(CAValueFunction(name: .translate))
        let scale = try #require(CAValueFunction(name: .scale))

        #expect(translate.transform(for: CGFloat(3)) == nil)
        #expect(scale.transform(for: [CGFloat(2), CGFloat(3)]) == nil)

        let translated = try #require(translate.transform(for: [CGFloat(1), CGFloat(2), CGFloat(3)]))
        #expect(abs(translated.m41 - 1) < epsilon)
        #expect(abs(translated.m42 - 2) < epsilon)
        #expect(abs(translated.m43 - 3) < epsilon)
    }

    @Test("Non-additive basic value functions replace the model transform")
    func basicValueFunctionReplacesModelTransform() throws {
        let animation = CABasicAnimation(keyPath: "transform")
        animation.valueFunction = CAValueFunction(name: .translateX)
        animation.fromValue = 0
        animation.toValue = 10

        let transform = try presentation(
            for: animation,
            modelTransform: CATransform3DMakeScale(2, 3, 1)
        )

        #expect(abs(transform.m11 - 1) < epsilon)
        #expect(abs(transform.m22 - 1) < epsilon)
        #expect(abs(transform.m41 - 5) < epsilon)
    }

    @Test("Additive basic value functions concatenate with the model transform")
    func additiveBasicValueFunctionConcatenates() throws {
        let animation = CABasicAnimation(keyPath: "transform")
        animation.valueFunction = CAValueFunction(name: .translateX)
        animation.fromValue = CGFloat(0)
        animation.toValue = CGFloat(10)
        animation.isAdditive = true

        let transform = try presentation(
            for: animation,
            modelTransform: CATransform3DMakeScale(2, 3, 1)
        )

        #expect(abs(transform.m11 - 2) < epsilon)
        #expect(abs(transform.m22 - 3) < epsilon)
        #expect(abs(transform.m41 - 10) < epsilon)
    }

    @Test("Linear and cubic keyframes apply value functions after interpolation")
    func interpolatedKeyframesApplyValueFunction() throws {
        let linear = CAKeyframeAnimation(keyPath: "transform")
        linear.valueFunction = CAValueFunction(name: .translateX)
        linear.values = [0, 10]
        let translated = try presentation(for: linear)
        #expect(abs(translated.m41 - 5) < epsilon)

        let cubic = CAKeyframeAnimation(keyPath: "transform")
        cubic.valueFunction = CAValueFunction(name: .rotateZ)
        cubic.values = [CGFloat(0), CGFloat.pi]
        cubic.calculationMode = .cubic
        let rotated = try presentation(for: cubic)
        #expect(abs(rotated.m11) < epsilon)
        #expect(abs(rotated.m12 - 1) < epsilon)
        #expect(abs(rotated.m21 + 1) < epsilon)
        #expect(abs(rotated.m22) < epsilon)
    }

    @Test("Single and discrete keyframes apply value functions directly")
    func directKeyframesApplyValueFunction() throws {
        let single = CAKeyframeAnimation(keyPath: "transform")
        single.valueFunction = CAValueFunction(name: .translateY)
        single.values = [12]
        let translated = try presentation(for: single)
        #expect(abs(translated.m42 - 12) < epsilon)

        let discrete = CAKeyframeAnimation(keyPath: "transform")
        discrete.valueFunction = CAValueFunction(name: .scaleX)
        discrete.values = [CGFloat(1), CGFloat(3)]
        discrete.keyTimes = [0, 1]
        discrete.calculationMode = .discrete
        let scaled = try presentation(for: discrete, elapsed: 0.75)
        #expect(abs(scaled.m11 - 1) < epsilon)
    }

    @Test("Additive and cumulative keyframes preserve value-function semantics")
    func additiveAndCumulativeKeyframesApplyValueFunction() throws {
        let additive = CAKeyframeAnimation(keyPath: "transform")
        additive.valueFunction = CAValueFunction(name: .translateX)
        additive.values = [CGFloat(0), CGFloat(10)]
        additive.isAdditive = true
        let additiveTransform = try presentation(
            for: additive,
            modelTransform: CATransform3DMakeScale(2, 3, 1)
        )
        #expect(abs(additiveTransform.m41 - 10) < epsilon)

        let cumulative = CAKeyframeAnimation(keyPath: "transform")
        cumulative.valueFunction = CAValueFunction(name: .translateX)
        cumulative.values = [CGFloat(0), CGFloat(10)]
        cumulative.isCumulative = true
        cumulative.repeatCount = 3
        let cumulativeTransform = try presentation(for: cumulative, elapsed: 1.5)
        #expect(abs(cumulativeTransform.m41 - 15) < epsilon)
    }

    @Test("Aggregate basic functions interpolate independent axis values")
    func aggregateBasicFunctionsInterpolateComponents() throws {
        let scale = CABasicAnimation(keyPath: "transform")
        scale.valueFunction = CAValueFunction(name: .scale)
        scale.toValue = [CGFloat(3), CGFloat(4), CGFloat(5)]
        let scaled = try presentation(for: scale)
        #expect(abs(scaled.m11 - 2) < epsilon)
        #expect(abs(scaled.m22 - 2.5) < epsilon)
        #expect(abs(scaled.m33 - 3) < epsilon)

        let translate = CABasicAnimation(keyPath: "transform")
        translate.valueFunction = CAValueFunction(name: .translate)
        translate.fromValue = [CGFloat(0), CGFloat(0), CGFloat(0)]
        translate.toValue = [CGFloat(10), CGFloat(20), CGFloat(30)]
        let translated = try presentation(for: translate)
        #expect(abs(translated.m41 - 5) < epsilon)
        #expect(abs(translated.m42 - 10) < epsilon)
        #expect(abs(translated.m43 - 15) < epsilon)
    }

    @Test("Aggregate keyframes preserve cubic and cumulative evaluation")
    func aggregateKeyframesApplyValueFunctions() throws {
        let cubic = CAKeyframeAnimation(keyPath: "transform")
        cubic.valueFunction = CAValueFunction(name: .scale)
        cubic.values = [
            [CGFloat(1), CGFloat(1), CGFloat(1)],
            [CGFloat(3), CGFloat(5), CGFloat(7)],
        ]
        cubic.calculationMode = .cubic
        let scaled = try presentation(for: cubic)
        #expect(abs(scaled.m11 - 2) < epsilon)
        #expect(abs(scaled.m22 - 3) < epsilon)
        #expect(abs(scaled.m33 - 4) < epsilon)

        let cumulative = CAKeyframeAnimation(keyPath: "transform")
        cumulative.valueFunction = CAValueFunction(name: .translate)
        cumulative.values = [
            [CGFloat(0), CGFloat(0), CGFloat(0)],
            [CGFloat(10), CGFloat(20), CGFloat(30)],
        ]
        cumulative.isCumulative = true
        cumulative.repeatCount = 3
        let translated = try presentation(for: cumulative, elapsed: 1.5)
        #expect(abs(translated.m41 - 15) < epsilon)
        #expect(abs(translated.m42 - 30) < epsilon)
        #expect(abs(translated.m43 - 45) < epsilon)
    }

    @Test("Aggregate paced modes preserve QuartzCore keyframe spacing")
    func aggregatePacedModesUseEvenKeyframeSpacing() throws {
        func animation(mode: CAAnimationCalculationMode) -> CAKeyframeAnimation {
            let animation = CAKeyframeAnimation(keyPath: "transform")
            animation.valueFunction = CAValueFunction(name: .translate)
            animation.values = [
                [CGFloat(0), CGFloat(0), CGFloat(0)],
                [CGFloat(10), CGFloat(0), CGFloat(0)],
                [CGFloat(10), CGFloat(30), CGFloat(0)],
            ]
            animation.calculationMode = mode
            return animation
        }

        let paced = try presentation(for: animation(mode: .paced), elapsed: 0.75)
        #expect(abs(paced.m41 - 10) < epsilon)
        #expect(abs(paced.m42 - 15) < epsilon)

        let cubicPaced = try presentation(for: animation(mode: .cubicPaced), elapsed: 0.75)
        #expect(abs(cubicPaced.m41 - 10.625) < epsilon)
        #expect(abs(cubicPaced.m42 - 13.125) < epsilon)
    }

    @Test("Invalid aggregate inputs do not report a synthesized transform")
    func invalidAggregateInputsDoNotApply() throws {
        let animation = CABasicAnimation(keyPath: "transform")
        animation.valueFunction = CAValueFunction(name: .scale)
        animation.fromValue = CGFloat(1)
        animation.toValue = CGFloat(3)

        let transform = try presentation(
            for: animation,
            modelTransform: CATransform3DMakeScale(2, 3, 4)
        )
        #expect(abs(transform.m11 - 2) < epsilon)
        #expect(abs(transform.m22 - 3) < epsilon)
        #expect(abs(transform.m33 - 4) < epsilon)
    }
}
