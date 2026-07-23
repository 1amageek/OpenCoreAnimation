import Testing
@testable import OpenCoreAnimation

@Suite("Color animation conversion")
struct CAColorAnimationConversionTests {
    private let epsilon: CGFloat = 0.001

    private func frozenBasic(
        from: CGColor,
        to: CGColor,
        additive: Bool = false
    ) -> CABasicAnimation {
        let animation = CABasicAnimation(keyPath: "backgroundColor")
        animation.fromValue = from
        animation.toValue = to
        animation.isAdditive = additive
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
        return animation
    }

    private func components(of color: CGColor?) throws -> [CGFloat] {
        let components = try #require(color?.components)
        #expect(components.count == 4)
        return components
    }

    private func patternColor() throws -> CGColor {
        let colorSpace = try #require(CGColorSpace(patternBaseSpace: nil))
        return try color(in: colorSpace, components: [1])
    }

    private func color(
        in colorSpace: CGColorSpace,
        components: [CGFloat]
    ) throws -> CGColor {
        try #require(components.count == colorSpace.numberOfComponents + 1)
        return try components.withUnsafeBufferPointer { buffer in
            let baseAddress = try #require(buffer.baseAddress)
            return try #require(CGColor(colorSpace: colorSpace, components: baseAddress))
        }
    }

    @Test("Basic animations convert CMYK endpoints to device RGB")
    func basicAnimationsConvertCMYKEndpoints() throws {
        let layer = CALayer()
        layer.add(
            frozenBasic(
                from: CGColor(
                    genericCMYKCyan: 0,
                    magenta: 0,
                    yellow: 0,
                    black: 0,
                    alpha: 1
                ),
                to: CGColor(
                    genericCMYKCyan: 0,
                    magenta: 1,
                    yellow: 1,
                    black: 0,
                    alpha: 1
                )
            ),
            forKey: "cmykBasic"
        )

        let cmykComponents = try components(of: layer.presentation()?.backgroundColor)
        #expect(abs(cmykComponents[0] - 1) < epsilon)
        #expect(abs(cmykComponents[1] - 0.5) < epsilon)
        #expect(abs(cmykComponents[2] - 0.5) < epsilon)
        #expect(abs(cmykComponents[3] - 1) < epsilon)

        let linearSpace = try #require(CGColorSpace(name: CGColorSpace.linearSRGB))
        let linearColor = try color(
            in: linearSpace,
            components: [0.25, 0.5, 0.75, 1]
        )
        let interpolationSpace = try #require(
            CGColorSpace(name: CGColorSpace.sRGB)
        )
        let expected = try components(
            of: linearColor.converted(
                to: interpolationSpace,
                intent: .defaultIntent,
                options: nil
            )
        )
        let profiledLayer = CALayer()
        profiledLayer.add(
            frozenBasic(from: linearColor, to: linearColor),
            forKey: "profiledRGBBasic"
        )
        let actual = try components(of: profiledLayer.presentation()?.backgroundColor)
        for index in actual.indices {
            #expect(abs(actual[index] - expected[index]) < epsilon)
        }
    }

    @Test("Additive color animations convert before composition")
    func additiveAnimationsConvertBeforeComposition() throws {
        let layer = CALayer()
        layer.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        layer.add(
            frozenBasic(
                from: CGColor(
                    genericCMYKCyan: 0,
                    magenta: 0,
                    yellow: 0,
                    black: 0,
                    alpha: 0
                ),
                to: CGColor(
                    genericCMYKCyan: 0,
                    magenta: 1,
                    yellow: 1,
                    black: 0,
                    alpha: 1
                ),
                additive: true
            ),
            forKey: "cmykAdditive"
        )

        let components = try components(of: layer.presentation()?.backgroundColor)
        #expect(abs(components[0] - 1) < epsilon)
        #expect(abs(components[1] - 1.5) < epsilon)
        #expect(abs(components[2] - 0.5) < epsilon)
        #expect(abs(components[3] - 1.5) < epsilon)
    }

    @Test("Unsupported pattern colors do not alter presentation state")
    func unsupportedPatternColorsDoNotApply() throws {
        let unsupported = try patternColor()
        let modelColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)

        let basicLayer = CALayer()
        basicLayer.backgroundColor = modelColor
        basicLayer.add(
            frozenBasic(
                from: unsupported,
                to: CGColor(red: 1, green: 0, blue: 0, alpha: 1)
            ),
            forKey: "unsupportedBasic"
        )

        let cubicLayer = CALayer()
        cubicLayer.backgroundColor = modelColor
        let cubic = CAKeyframeAnimation(keyPath: "backgroundColor")
        cubic.values = [
            unsupported,
            CGColor(red: 1, green: 0, blue: 0, alpha: 1),
        ]
        cubic.calculationMode = .cubic
        cubic.duration = 1
        cubic.speed = 0
        cubic.timeOffset = 0.5
        cubicLayer.add(cubic, forKey: "unsupportedCubic")

        let pacedLayer = CALayer()
        pacedLayer.backgroundColor = modelColor
        let paced = CAKeyframeAnimation(keyPath: "backgroundColor")
        paced.values = [
            unsupported,
            CGColor(red: 1, green: 0, blue: 0, alpha: 1),
        ]
        paced.calculationMode = .paced
        paced.duration = 1
        paced.speed = 0
        paced.timeOffset = 0.5
        pacedLayer.add(paced, forKey: "unsupportedPaced")

        let nonFiniteLayer = CALayer()
        nonFiniteLayer.backgroundColor = modelColor
        nonFiniteLayer.add(
            frozenBasic(
                from: CGColor(
                    red: .infinity,
                    green: 0,
                    blue: 0,
                    alpha: 1
                ),
                to: CGColor(red: 1, green: 0, blue: 0, alpha: 1)
            ),
            forKey: "nonFiniteBasic"
        )

        for layer in [basicLayer, cubicLayer, pacedLayer, nonFiniteLayer] {
            let components = try components(of: layer.presentation()?.backgroundColor)
            #expect(abs(components[0]) < epsilon)
            #expect(abs(components[1] - 1) < epsilon)
            #expect(abs(components[2]) < epsilon)
            #expect(abs(components[3] - 1) < epsilon)
        }

        let overflowLayer = CALayer()
        overflowLayer.backgroundColor = CGColor(
            red: .greatestFiniteMagnitude,
            green: 0,
            blue: 0,
            alpha: 1
        )
        overflowLayer.add(
            frozenBasic(
                from: CGColor(
                    red: .greatestFiniteMagnitude,
                    green: 0,
                    blue: 0,
                    alpha: 0
                ),
                to: CGColor(
                    red: .greatestFiniteMagnitude,
                    green: 0,
                    blue: 0,
                    alpha: 0
                ),
                additive: true
            ),
            forKey: "overflowingAdditive"
        )
        let overflowComponents = try components(
            of: overflowLayer.presentation()?.backgroundColor
        )
        #expect(overflowComponents[0] == .greatestFiniteMagnitude)
        #expect(overflowComponents[1] == 0)
        #expect(overflowComponents[2] == 0)
        #expect(overflowComponents[3] == 1)
    }

    @Test("Invalid gradient color arrays are rejected atomically")
    func invalidGradientColorArraysAreRejectedAtomically() throws {
        let unsupported = try patternColor()
        let layer = CAGradientLayer()
        layer.colors = [
            CGColor(red: 0, green: 1, blue: 0, alpha: 1),
            CGColor(red: 0, green: 0, blue: 1, alpha: 1),
        ]

        let animation = CAKeyframeAnimation(keyPath: "colors")
        animation.values = [
            [
                CGColor(red: 1, green: 0, blue: 0, alpha: 1),
                unsupported,
            ] as [Any],
            [
                CGColor(red: 1, green: 1, blue: 0, alpha: 1),
                CGColor(red: 1, green: 0, blue: 1, alpha: 1),
            ] as [Any],
        ]
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
        layer.add(animation, forKey: "unsupportedGradient")

        let colors = try #require(layer.presentation()?.colors)
        let first = try components(of: colors[0] as? CGColor)
        let second = try components(of: colors[1] as? CGColor)
        #expect(first == [0, 1, 0, 1])
        #expect(second == [0, 0, 1, 1])
    }
}
