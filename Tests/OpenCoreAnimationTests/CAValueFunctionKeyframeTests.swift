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
}
