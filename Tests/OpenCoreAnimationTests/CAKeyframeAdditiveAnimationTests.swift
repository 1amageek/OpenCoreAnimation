import Testing
@testable import OpenCoreAnimation

@Suite("CAKeyframeAnimation additive value evaluation")
struct CAKeyframeAdditiveAnimationTests {
    private let epsilon: CGFloat = 0.0001

    @Test("Linear keyframes add scalar, geometry, and color values")
    func linearBaseLayerValues() throws {
        let layer = CALayer()
        layer.opacity = 0.2
        layer.position = CGPoint(x: 10, y: 20)
        layer.bounds = CGRect(x: 1, y: 2, width: 30, height: 40)
        layer.backgroundColor = color(0.1, 0.2, 0.3, 0.4)

        let opacity = animation("opacity", values: [Float(0.1), Float(0.3)])
        let position = animation(
            "position",
            values: [CGPoint(x: 2, y: 4), CGPoint(x: 6, y: 8)]
        )
        let bounds = animation(
            "bounds",
            values: [
                CGRect(x: 1, y: 1, width: 2, height: 2),
                CGRect(x: 3, y: 5, width: 6, height: 10),
            ]
        )
        let background = animation(
            "backgroundColor",
            values: [color(0.1, 0.1, 0.1, 0.1), color(0.3, 0.1, 0.5, 0.1)]
        )

        let result = try presentation(
            of: layer,
            animations: [opacity, position, bounds, background]
        )

        #expect(abs(CGFloat(result.opacity) - 0.4) < epsilon)
        expectEqual(result.position, CGPoint(x: 14, y: 26))
        expectEqual(result.bounds, CGRect(x: 3, y: 5, width: 34, height: 46))
        expectColor(try #require(result.backgroundColor), [0.3, 0.3, 0.6, 0.5])
    }

    @Test("Single, discrete, and cubic keyframes use the same additive contract")
    func allCalculationPaths() throws {
        let singleLayer = CALayer()
        singleLayer.cornerRadius = 5
        let single = animation("cornerRadius", values: [CGFloat(2)])
        let singleResult = try presentation(of: singleLayer, animations: [single])
        #expect(abs(singleResult.cornerRadius - 7) < epsilon)

        let discreteLayer = CALayer()
        discreteLayer.position.x = 10
        let discrete = animation("position.x", values: [CGFloat(2), CGFloat(8)])
        discrete.calculationMode = .discrete
        discrete.keyTimes = [0, 0.8, 1]
        let discreteResult = try presentation(
            of: discreteLayer,
            animations: [discrete],
            elapsed: 0.75
        )
        #expect(abs(discreteResult.position.x - 12) < epsilon)

        let cubicLayer = CALayer()
        cubicLayer.position = CGPoint(x: 10, y: 20)
        let cubic = animation(
            "position",
            values: [CGPoint.zero, CGPoint(x: 2, y: 4)]
        )
        cubic.calculationMode = .cubic
        let cubicResult = try presentation(of: cubicLayer, animations: [cubic])
        expectEqual(cubicResult.position, CGPoint(x: 11, y: 22))
    }

    @Test("Specialized layers add their scalar, geometry, and color values")
    func specializedLayerValues() throws {
        let shape = CAShapeLayer()
        shape.strokeStart = 0.1
        shape.fillColor = color(0.1, 0.2, 0.3, 0.4)
        let stroke = animation("strokeStart", values: [CGFloat(0.1), CGFloat(0.3)])
        let fill = animation(
            "fillColor",
            values: [color(0.1, 0, 0.1, 0), color(0.3, 0.2, 0.1, 0.2)]
        )
        let shapeResult = try #require(
            presentation(of: shape, animations: [stroke, fill]) as? CAShapeLayer
        )
        #expect(abs(shapeResult.strokeStart - 0.3) < epsilon)
        expectColor(try #require(shapeResult.fillColor), [0.3, 0.3, 0.4, 0.5])

        let text = CATextLayer()
        text.fontSize = 10
        text.foregroundColor = color(0.2, 0.2, 0.2, 1)
        let fontSize = animation("fontSize", values: [CGFloat(2), CGFloat(6)])
        let foreground = animation(
            "foregroundColor",
            values: [color(0.1, 0, 0, 0), color(0.3, 0.2, 0, 0)]
        )
        let textResult = try #require(
            presentation(of: text, animations: [fontSize, foreground]) as? CATextLayer
        )
        #expect(abs(textResult.fontSize - 14) < epsilon)
        expectColor(try #require(textResult.foregroundColor), [0.4, 0.3, 0.2, 1])

        let emitter = CAEmitterLayer()
        emitter.birthRate = 2
        emitter.emitterPosition = CGPoint(x: 10, y: 20)
        let birthRate = animation("birthRate", values: [Float(1), Float(3)])
        let emitterPosition = animation(
            "emitterPosition",
            values: [CGPoint(x: 2, y: 4), CGPoint(x: 6, y: 8)]
        )
        let emitterResult = try #require(
            presentation(of: emitter, animations: [birthRate, emitterPosition]) as? CAEmitterLayer
        )
        #expect(abs(CGFloat(emitterResult.birthRate) - 4) < epsilon)
        expectEqual(emitterResult.emitterPosition, CGPoint(x: 14, y: 26))

        let replicator = CAReplicatorLayer()
        replicator.instanceDelay = 1
        replicator.instanceColor = color(0.2, 0.2, 0.2, 1)
        let delay = animation("instanceDelay", values: [CGFloat(0.2), CGFloat(0.6)])
        let instanceColor = animation(
            "instanceColor",
            values: [color(0.1, 0, 0, 0), color(0.3, 0.2, 0, 0)]
        )
        let replicatorResult = try #require(
            presentation(of: replicator, animations: [delay, instanceColor]) as? CAReplicatorLayer
        )
        #expect(abs(replicatorResult.instanceDelay - 1.4) < Double(epsilon))
        expectColor(try #require(replicatorResult.instanceColor), [0.4, 0.3, 0.2, 1])
    }

    @Test("Gradient color and location arrays add element by element")
    func gradientArrays() throws {
        let gradient = CAGradientLayer()
        gradient.colors = [color(0.1, 0.2, 0.3, 0.4), color(0.2, 0.3, 0.4, 0.5)]
        gradient.locations = [0.1, 0.6]

        let colors = animation(
            "colors",
            values: [
                [color(0.1, 0, 0.1, 0), color(0, 0.1, 0, 0.1)],
                [color(0.3, 0.2, 0.1, 0.2), color(0.2, 0.1, 0.2, 0.1)],
            ]
        )
        let locations = animation(
            "locations",
            values: [
                [CGFloat(0.1), CGFloat(0.2)],
                [CGFloat(0.3), CGFloat(0.4)],
            ]
        )

        let result = try #require(
            presentation(of: gradient, animations: [colors, locations]) as? CAGradientLayer
        )
        let colorValues = try #require(result.colors)
        expectColor(try #require(colorValues[0] as? CGColor), [0.3, 0.3, 0.4, 0.5])
        expectColor(try #require(colorValues[1] as? CGColor), [0.3, 0.4, 0.5, 0.6])
        let locationValues = try #require(result.locations)
        expectEqual(locationValues, [0.3, 0.9])
    }

    @Test("Additive gradient arrays can start without model arrays")
    func gradientArraysWithoutModelValues() throws {
        let gradient = CAGradientLayer()
        let colors = animation(
            "colors",
            values: [
                [color(0, 0, 0, 0), color(0, 0, 0, 0)],
                [color(0.4, 0.2, 0.6, 1), color(0.2, 0.6, 0.4, 1)],
            ]
        )
        let locations = animation(
            "locations",
            values: [
                [CGFloat(0), CGFloat(0)],
                [CGFloat(0.4), CGFloat(1.2)],
            ]
        )

        let result = try #require(
            presentation(of: gradient, animations: [colors, locations]) as? CAGradientLayer
        )
        let colorValues = try #require(result.colors)
        expectColor(try #require(colorValues[0] as? CGColor), [0.2, 0.1, 0.3, 0.5])
        expectColor(try #require(colorValues[1] as? CGColor), [0.1, 0.3, 0.2, 0.5])
        expectEqual(try #require(result.locations), [0.2, 0.6])
    }

    @Test("Incompatible gradient keyframes leave complete model arrays unchanged")
    func rejectsIncompatibleGradientArrays() throws {
        let gradient = CAGradientLayer()
        gradient.colors = [color(1, 0, 0, 1), color(0, 1, 0, 1)]
        gradient.locations = [0.2, 0.8]

        let colors = animation(
            "colors",
            values: [
                [color(0, 0, 1, 1)],
                [color(1, 1, 0, 1), color(0, 1, 1, 1)],
            ]
        )
        let locations = animation(
            "locations",
            values: [
                [CGFloat(0)],
                [CGFloat(0.3), CGFloat(0.7)],
            ]
        )

        let result = try #require(
            presentation(of: gradient, animations: [colors, locations]) as? CAGradientLayer
        )
        let colorValues = try #require(result.colors)
        #expect(colorValues.count == 2)
        expectColor(try #require(colorValues[0] as? CGColor), [1, 0, 0, 1])
        expectColor(try #require(colorValues[1] as? CGColor), [0, 1, 0, 1])
        expectEqual(try #require(result.locations), [0.2, 0.8])
    }

    @Test("Cumulative gradient arrays carry terminal values into repeat cycles")
    func cumulativeGradientArrays() throws {
        let gradient = CAGradientLayer()
        gradient.colors = [color(0, 0, 0, 0), color(0, 0, 0, 0)]
        gradient.locations = [0, 0]

        let colors = keyframe(
            "colors",
            values: [
                [color(0, 0, 0, 0), color(0, 0, 0, 0)],
                [color(0.2, 0.4, 0.6, 0.8), color(0.4, 0.2, 0.8, 0.6)],
            ]
        )
        let locations = keyframe(
            "locations",
            values: [
                [CGFloat(0), CGFloat(0)],
                [CGFloat(0.2), CGFloat(0.4)],
            ]
        )
        for item in [colors, locations] {
            item.isCumulative = true
            item.repeatCount = 3
        }

        let result = try #require(
            presentation(
                of: gradient,
                animations: [colors, locations],
                elapsed: 1.5
            ) as? CAGradientLayer
        )
        let colorValues = try #require(result.colors)
        expectColor(try #require(colorValues[0] as? CGColor), [0.3, 0.6, 0.9, 1.2])
        expectColor(try #require(colorValues[1] as? CGColor), [0.6, 0.3, 1.2, 0.9])
        expectEqual(try #require(result.locations), [0.3, 0.6])
    }

    @Test("Full sublayer and replicator transforms are additive")
    func specializedTransforms() throws {
        let layer = CALayer()
        layer.sublayerTransform = CATransform3DMakeTranslation(10, 0, 0)
        let sublayerTransform = animation(
            "sublayerTransform",
            values: [
                CATransform3DIdentity,
                CATransform3DMakeTranslation(20, 0, 0),
            ]
        )
        let layerResult = try presentation(of: layer, animations: [sublayerTransform])
        #expect(abs(layerResult.sublayerTransform.m41 - 20) < epsilon)

        let replicator = CAReplicatorLayer()
        replicator.instanceTransform = CATransform3DMakeTranslation(10, 0, 0)
        let instanceTransform = animation(
            "instanceTransform",
            values: [
                CATransform3DIdentity,
                CATransform3DMakeTranslation(20, 0, 0),
            ]
        )
        let replicatorResult = try #require(
            presentation(of: replicator, animations: [instanceTransform]) as? CAReplicatorLayer
        )
        #expect(abs(replicatorResult.instanceTransform.m41 - 20) < epsilon)
    }

    private func animation(_ keyPath: String, values: [Any]) -> CAKeyframeAnimation {
        let result = keyframe(keyPath, values: values)
        result.isAdditive = true
        return result
    }

    private func keyframe(_ keyPath: String, values: [Any]) -> CAKeyframeAnimation {
        let result = CAKeyframeAnimation(keyPath: keyPath)
        result.values = values
        return result
    }

    private func presentation(
        of layer: CALayer,
        animations: [CAKeyframeAnimation],
        elapsed: CFTimeInterval = 0.5
    ) throws -> CALayer {
        for (index, animation) in animations.enumerated() {
            animation.duration = 1
            animation.speed = 0
            animation.timeOffset = elapsed
            animation.fillMode = .both
            animation.isRemovedOnCompletion = false
            layer.add(animation, forKey: "additive-keyframe-\(index)")
        }
        return try #require(layer.presentation())
    }

    private func color(
        _ red: CGFloat,
        _ green: CGFloat,
        _ blue: CGFloat,
        _ alpha: CGFloat
    ) -> CGColor {
        CGColor(red: red, green: green, blue: blue, alpha: alpha)
    }

    private func expectColor(_ actual: CGColor, _ expected: [CGFloat]) {
        expectEqual(actual.components ?? [], expected)
    }

    private func expectEqual(_ actual: [CGFloat], _ expected: [CGFloat]) {
        #expect(actual.count == expected.count)
        for (actualValue, expectedValue) in zip(actual, expected) {
            #expect(abs(actualValue - expectedValue) < epsilon)
        }
    }

    private func expectEqual(_ actual: CGPoint, _ expected: CGPoint) {
        #expect(abs(actual.x - expected.x) < epsilon)
        #expect(abs(actual.y - expected.y) < epsilon)
    }

    private func expectEqual(_ actual: CGRect, _ expected: CGRect) {
        expectEqual(actual.origin, expected.origin)
        #expect(abs(actual.width - expected.width) < epsilon)
        #expect(abs(actual.height - expected.height) < epsilon)
    }
}
