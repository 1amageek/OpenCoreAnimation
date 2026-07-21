import Testing
@testable import OpenCoreAnimation

@Suite("Boolean and discrete animation evaluation")
struct CABooleanAndDiscreteAnimationTests {
    private func add(
        _ animation: CAAnimation,
        to layer: CALayer,
        key: String,
        elapsed: CFTimeInterval
    ) {
        animation.duration = 1
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        layer.add(animation, forKey: key)
        setStoredAnimationBeginTime(
            CACurrentMediaTime() - elapsed,
            on: layer,
            forKey: key
        )
    }

    @Test("Basic animations drive documented Boolean presentation properties")
    func basicBooleanProperties() throws {
        let layer = CALayer()
        layer.isDoubleSided = false

        for keyPath in ["hidden", "masksToBounds", "doubleSided", "shouldRasterize"] {
            let animation = CABasicAnimation(keyPath: keyPath)
            animation.fromValue = false
            animation.toValue = true
            add(animation, to: layer, key: keyPath, elapsed: 0.5)
        }

        let presentation = try #require(layer.presentation())
        #expect(presentation.isHidden)
        #expect(presentation.masksToBounds)
        #expect(presentation.isDoubleSided)
        #expect(presentation.shouldRasterize)
    }

    @Test("Swift Boolean key-path aliases remain executable")
    func swiftBooleanKeyPathAliases() throws {
        let layer = CALayer()
        layer.isDoubleSided = false

        let hidden = CABasicAnimation(keyPath: "isHidden")
        hidden.fromValue = false
        hidden.toValue = true
        add(hidden, to: layer, key: "hidden", elapsed: 0.5)

        let doubleSided = CABasicAnimation(keyPath: "isDoubleSided")
        doubleSided.fromValue = false
        doubleSided.toValue = true
        add(doubleSided, to: layer, key: "doubleSided", elapsed: 0.5)

        let presentation = try #require(layer.presentation())
        #expect(presentation.isHidden)
        #expect(presentation.isDoubleSided)
    }

    @Test("Discrete keyframes select the latest value at or before presentation time")
    func discreteKeyframesSelectLatestValue() throws {
        let middleLayer = CALayer()
        let middle = CAKeyframeAnimation(keyPath: "hidden")
        middle.values = [false, true, false]
        middle.keyTimes = [0, 0.25, 0.75]
        middle.calculationMode = .discrete
        add(middle, to: middleLayer, key: "hidden", elapsed: 0.5)

        let finalLayer = CALayer()
        let final = CAKeyframeAnimation(keyPath: "hidden")
        final.values = [false, true, false]
        final.keyTimes = [0, 0.25, 0.75]
        final.calculationMode = .discrete
        add(final, to: finalLayer, key: "hidden", elapsed: 0.9)

        #expect(try #require(middleLayer.presentation()).isHidden)
        #expect(try #require(finalLayer.presentation()).isHidden == false)
    }

    @Test("Single-value keyframes and scalar component key paths update presentation state")
    func singleValueAndScalarComponentKeyframes() throws {
        let hiddenLayer = CALayer()
        let hidden = CAKeyframeAnimation(keyPath: "hidden")
        hidden.values = [true]
        add(hidden, to: hiddenLayer, key: "hidden", elapsed: 0.5)

        let positionLayer = CALayer()
        positionLayer.position = CGPoint(x: 2, y: 7)
        let position = CAKeyframeAnimation(keyPath: "position.x")
        position.values = [CGFloat(0), CGFloat(20)]
        add(position, to: positionLayer, key: "position", elapsed: 0.5)

        #expect(try #require(hiddenLayer.presentation()).isHidden)
        let positionPresentation = try #require(positionLayer.presentation())
        #expect(abs(positionPresentation.position.x - 10) < 0.01)
        #expect(positionPresentation.position.y == 7)
    }

    @Test("Linear keyframes hold the final value after the final key time")
    func linearKeyframesHoldFinalValue() throws {
        let layer = CALayer()
        let animation = CAKeyframeAnimation(keyPath: "opacity")
        animation.values = [Float(0), Float(1)]
        animation.keyTimes = [0, 0.5]
        add(animation, to: layer, key: "opacity", elapsed: 0.75)

        let presentation = try #require(layer.presentation())
        #expect(presentation.opacity == 1)
    }

    @Test("shouldRasterize mutations participate in custom action resolution")
    func shouldRasterizeRunsCustomAction() {
        CATransaction.flush()
        let layer = CALayer()
        layer.actions = ["shouldRasterize": CABasicAnimation()]

        CATransaction.begin()
        CATransaction.setAnimationDuration(0.4)
        layer.shouldRasterize = true
        CATransaction.commit()

        let animation = layer.animation(forKey: "shouldRasterize") as? CABasicAnimation
        #expect(animation?.keyPath == "shouldRasterize")
        #expect(animation?.fromValue as? Bool == false)
        #expect(animation?.toValue as? Bool == true)
        #expect(animation?.duration == 0.4)
    }
}
