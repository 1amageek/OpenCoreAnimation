import Testing
@testable import OpenCoreAnimation

@Suite("Specialized layer animation evaluation")
struct CASpecializedAnimationTests {
    private func setMidpoint(_ animation: CAAnimation, on layer: CALayer, key: String) {
        animation.duration = 2
        animation.fillMode = .both
        layer.add(animation, forKey: key)
        setStoredAnimationBeginTime(CACurrentMediaTime() - 1, on: layer, forKey: key)
    }

    @Test("text properties interpolate on the presentation layer")
    func textPropertiesInterpolate() {
        let layer = CATextLayer()
        layer.fontSize = 10

        let size = CABasicAnimation(keyPath: "fontSize")
        size.fromValue = CGFloat(10)
        size.toValue = CGFloat(30)
        setMidpoint(size, on: layer, key: "fontSize")

        let color = CABasicAnimation(keyPath: "foregroundColor")
        color.fromValue = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
        color.toValue = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        setMidpoint(color, on: layer, key: "foregroundColor")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected text presentation layer")
            return
        }
        #expect(abs(presentation.fontSize - 20) < 0.01)
        #expect(abs((presentation.foregroundColor?.components?[0] ?? 0) - 0.5) < 0.01)
    }

    @Test("emitter geometry and rates interpolate")
    func emitterPropertiesInterpolate() {
        let layer = CAEmitterLayer()

        let position = CABasicAnimation(keyPath: "emitterPosition")
        position.fromValue = CGPoint(x: 0, y: 10)
        position.toValue = CGPoint(x: 20, y: 30)
        setMidpoint(position, on: layer, key: "position")

        let rate = CABasicAnimation(keyPath: "birthRate")
        rate.fromValue = Float(2)
        rate.toValue = Float(6)
        setMidpoint(rate, on: layer, key: "rate")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected emitter presentation layer")
            return
        }
        #expect(abs(presentation.emitterPosition.x - 10) < 0.01)
        #expect(abs(presentation.emitterPosition.y - 20) < 0.01)
        #expect(abs(presentation.birthRate - 4) < 0.01)
    }

    @Test("replicator transform, delay, and color offsets interpolate")
    func replicatorPropertiesInterpolate() {
        let layer = CAReplicatorLayer()

        let delay = CABasicAnimation(keyPath: "instanceDelay")
        delay.fromValue = CGFloat(0)
        delay.toValue = CGFloat(1)
        setMidpoint(delay, on: layer, key: "delay")

        let alpha = CABasicAnimation(keyPath: "instanceAlphaOffset")
        alpha.fromValue = Float(0)
        alpha.toValue = Float(-0.4)
        setMidpoint(alpha, on: layer, key: "alpha")

        let transform = CABasicAnimation(keyPath: "instanceTransform")
        transform.fromValue = CATransform3DIdentity
        transform.toValue = CATransform3DMakeTranslation(20, 0, 0)
        setMidpoint(transform, on: layer, key: "transform")

        guard let presentation = layer.presentation() else {
            Issue.record("Expected replicator presentation layer")
            return
        }
        #expect(abs(presentation.instanceDelay - 0.5) < 0.01)
        #expect(abs(presentation.instanceAlphaOffset + 0.2) < 0.01)
        #expect(abs(presentation.instanceTransform.m41 - 10) < 0.01)
    }

    @Test("compatible shape paths morph control points")
    func compatiblePathsMorph() {
        let from = CGMutablePath()
        from.addRect(CGRect(x: 0, y: 0, width: 10, height: 10))
        let to = CGMutablePath()
        to.addRect(CGRect(x: 10, y: 0, width: 30, height: 20))

        let layer = CAShapeLayer()
        let animation = CABasicAnimation(keyPath: "path")
        animation.fromValue = from
        animation.toValue = to
        setMidpoint(animation, on: layer, key: "path")

        guard let bounds = layer.presentation()?.path?.boundingBox else {
            Issue.record("Expected interpolated shape path")
            return
        }
        #expect(abs(bounds.minX - 5) < 0.01)
        #expect(abs(bounds.width - 20) < 0.01)
        #expect(abs(bounds.height - 15) < 0.01)
    }
}
