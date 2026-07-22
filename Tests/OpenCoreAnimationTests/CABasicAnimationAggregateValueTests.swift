import Testing
@testable import OpenCoreAnimation

@Suite("CABasicAnimation aggregate value evaluation")
struct CABasicAnimationAggregateValueTests {
    private let epsilon: CGFloat = 0.0001

    @Test("CGRect byValue resolves relative to the model value")
    func rectangleByValue() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 10, y: 20, width: 100, height: 80)
        let animation = CABasicAnimation(keyPath: "bounds")
        animation.byValue = CGRect(x: 2, y: 4, width: 20, height: 40)

        let result = try presentation(of: layer, animation: animation).bounds

        expectEqual(result, CGRect(x: 11, y: 22, width: 110, height: 100))
    }

    @Test("CGRect from plus by and to plus by resolve both endpoints")
    func rectangleEndpointCombinations() throws {
        let fromAndByLayer = CALayer()
        let fromAndBy = CABasicAnimation(keyPath: "contentsRect")
        fromAndBy.fromValue = CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        fromAndBy.byValue = CGRect(x: 0.2, y: 0.1, width: 0.4, height: 0.2)
        let fromAndByResult = try presentation(
            of: fromAndByLayer,
            animation: fromAndBy
        ).contentsRect
        expectEqual(fromAndByResult, CGRect(x: 0.2, y: 0.25, width: 0.5, height: 0.5))

        let toAndByLayer = CALayer()
        let toAndBy = CABasicAnimation(keyPath: "contentsCenter")
        toAndBy.toValue = CGRect(x: 0.5, y: 0.6, width: 0.7, height: 0.8)
        toAndBy.byValue = CGRect(x: 0.2, y: 0.2, width: 0.2, height: 0.2)
        let toAndByResult = try presentation(
            of: toAndByLayer,
            animation: toAndBy
        ).contentsCenter
        expectEqual(toAndByResult, CGRect(x: 0.4, y: 0.5, width: 0.6, height: 0.7))
    }

    @Test("Color byValue interpolates RGBA components")
    func colorByValue() throws {
        let layer = CALayer()
        layer.backgroundColor = color(0.2, 0.3, 0.4, 0.5)
        let animation = CABasicAnimation(keyPath: "backgroundColor")
        animation.byValue = color(0.2, 0.1, 0.2, 0.1)

        let result = try #require(
            presentation(of: layer, animation: animation).backgroundColor
        )

        expectColor(result, [0.3, 0.35, 0.5, 0.55])
    }

    @Test("Additive color byValue starts from transparent black")
    func additiveColorByValue() throws {
        let layer = CALayer()
        layer.borderColor = color(0.2, 0.3, 0.4, 0.5)
        let animation = CABasicAnimation(keyPath: "borderColor")
        animation.byValue = color(0.2, 0.1, 0.2, 0.1)
        animation.isAdditive = true

        let result = try #require(
            presentation(of: layer, animation: animation).borderColor
        )

        expectColor(result, [0.3, 0.35, 0.5, 0.55])
    }

    @Test("Explicit color endpoints animate without a model color")
    func explicitColorsWithoutModelValue() throws {
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "backgroundColor")
        animation.fromValue = color(1, 0, 0, 1)
        animation.toValue = color(0, 0, 1, 1)

        let result = try #require(
            presentation(of: layer, animation: animation).backgroundColor
        )

        expectColor(result, [0.5, 0, 0.5, 1])
    }

    @Test("Gradient location and color arrays support byValue")
    func gradientArraysByValue() throws {
        let gradient = CAGradientLayer()
        gradient.locations = [0.2, 0.8]
        gradient.colors = [color(0.2, 0.3, 0.4, 0.5), color(0.4, 0.3, 0.2, 0.5)]

        let locations = CABasicAnimation(keyPath: "locations")
        locations.byValue = [CGFloat(0.2), CGFloat(-0.2)]
        let locationsPresentation = try #require(
            presentation(of: gradient, animation: locations) as? CAGradientLayer
        )
        let locationValues = try #require(locationsPresentation.locations)
        #expect(abs(locationValues[0] - 0.3) < epsilon)
        #expect(abs(locationValues[1] - 0.7) < epsilon)

        let colors = CABasicAnimation(keyPath: "colors")
        colors.byValue = [color(0.2, 0.1, 0.2, 0.1), color(0.2, 0.1, 0.2, 0.1)]
        let colorsPresentation = try #require(
            presentation(of: gradient, animation: colors) as? CAGradientLayer
        )
        let colorValues = try #require(colorsPresentation.colors)
        expectColor(try #require(colorValues[0] as? CGColor), [0.3, 0.35, 0.5, 0.55])
        expectColor(try #require(colorValues[1] as? CGColor), [0.5, 0.35, 0.3, 0.55])
    }

    @Test("Explicit gradient array endpoints animate without model arrays")
    func explicitGradientArraysWithoutModelValues() throws {
        let locationGradient = CAGradientLayer()
        let locations = CABasicAnimation(keyPath: "locations")
        locations.fromValue = [CGFloat(0), CGFloat(1)]
        locations.toValue = [CGFloat(0.2), CGFloat(0.8)]

        let locationPresentation = try #require(
            presentation(of: locationGradient, animation: locations) as? CAGradientLayer
        )
        let locationValues = try #require(locationPresentation.locations)
        #expect(locationValues.count == 2)
        #expect(abs(locationValues[0] - 0.1) < epsilon)
        #expect(abs(locationValues[1] - 0.9) < epsilon)

        let colorGradient = CAGradientLayer()
        let colors = CABasicAnimation(keyPath: "colors")
        colors.fromValue = [color(1, 0, 0, 1), color(0, 1, 0, 1)]
        colors.toValue = [color(0, 0, 1, 1), color(1, 1, 0, 1)]

        let colorPresentation = try #require(
            presentation(of: colorGradient, animation: colors) as? CAGradientLayer
        )
        let colorValues = try #require(colorPresentation.colors)
        #expect(colorValues.count == 2)
        expectColor(try #require(colorValues[0] as? CGColor), [0.5, 0, 0.5, 1])
        expectColor(try #require(colorValues[1] as? CGColor), [0.5, 1, 0, 1])
    }

    @Test("Mismatched gradient arrays do not partially replace presentation state")
    func rejectsMismatchedGradientArrays() throws {
        let gradient = CAGradientLayer()
        gradient.locations = [0.2, 0.5, 0.8]
        let animation = CABasicAnimation(keyPath: "locations")
        animation.fromValue = [CGFloat(0), CGFloat(1)]
        animation.toValue = [CGFloat(0), CGFloat(0.5), CGFloat(1)]

        let result = try #require(
            (presentation(of: gradient, animation: animation) as? CAGradientLayer)?.locations
        )

        #expect(result == [0.2, 0.5, 0.8])
    }

    @Test("Full transforms support byValue and inverse endpoint resolution")
    func transformByValue() throws {
        let layer = CALayer()
        layer.transform = CATransform3DMakeTranslation(10, 0, 0)
        let byOnly = CABasicAnimation(keyPath: "transform")
        byOnly.byValue = CATransform3DMakeTranslation(20, 0, 0)
        let byOnlyResult = try presentation(of: layer, animation: byOnly).transform
        #expect(abs(byOnlyResult.m41 - 20) < epsilon)

        let toAndBy = CABasicAnimation(keyPath: "transform")
        toAndBy.toValue = CATransform3DMakeTranslation(30, 0, 0)
        toAndBy.byValue = CATransform3DMakeTranslation(10, 0, 0)
        let toAndByResult = try presentation(of: layer, animation: toAndBy).transform
        #expect(abs(toAndByResult.m41 - 25) < epsilon)
    }

    @Test("Singular transform byValue does not fabricate an inverse endpoint")
    func rejectsSingularTransformSubtraction() throws {
        let layer = CALayer()
        layer.transform = CATransform3DMakeTranslation(10, 0, 0)
        let animation = CABasicAnimation(keyPath: "transform")
        animation.toValue = CATransform3DMakeTranslation(30, 0, 0)
        animation.byValue = CATransform3DMakeScale(0, 1, 1)

        let result = try presentation(of: layer, animation: animation).transform

        #expect(abs(result.m41 - 10) < epsilon)
        #expect(abs(result.m11 - 1) < epsilon)
    }

    private func presentation(
        of layer: CALayer,
        animation: CABasicAnimation
    ) throws -> CALayer {
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        layer.add(animation, forKey: "aggregate")
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
        let components = actual.components ?? []
        #expect(components.count == expected.count)
        for (actualValue, expectedValue) in zip(components, expected) {
            #expect(abs(actualValue - expectedValue) < epsilon)
        }
    }

    private func expectEqual(_ actual: CGRect, _ expected: CGRect) {
        #expect(abs(actual.origin.x - expected.origin.x) < epsilon)
        #expect(abs(actual.origin.y - expected.origin.y) < epsilon)
        #expect(abs(actual.width - expected.width) < epsilon)
        #expect(abs(actual.height - expected.height) < epsilon)
    }
}
