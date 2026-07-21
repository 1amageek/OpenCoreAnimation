import Testing
@testable import OpenCoreAnimation

@Suite("CALayer rasterizationScale animation evaluation")
struct CARasterizationScaleAnimationTests {
    private func add(
        _ animation: CAAnimation,
        to layer: CALayer,
        elapsed: CFTimeInterval
    ) {
        animation.duration = 1
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        layer.add(animation, forKey: "rasterizationScale")
        setStoredAnimationBeginTime(
            CACurrentMediaTime() - elapsed,
            on: layer,
            forKey: "rasterizationScale"
        )
    }

    @Test("Basic animation interpolates rasterization scale")
    func basicAnimationInterpolatesScale() throws {
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "rasterizationScale")
        animation.fromValue = CGFloat(1)
        animation.toValue = CGFloat(2)
        add(animation, to: layer, elapsed: 0.5)

        let presentation = try #require(layer.presentation())
        #expect(abs(presentation.rasterizationScale - 1.5) < 0.01)
    }

    @Test("Keyframe animation interpolates rasterization scale")
    func keyframeAnimationInterpolatesScale() throws {
        let layer = CALayer()
        let animation = CAKeyframeAnimation(keyPath: "rasterizationScale")
        animation.values = [CGFloat(1), CGFloat(3)]
        add(animation, to: layer, elapsed: 0.5)

        let presentation = try #require(layer.presentation())
        #expect(abs(presentation.rasterizationScale - 2) < 0.01)
    }

    @Test("rasterizationScale mutations participate in custom action resolution")
    func mutationRunsCustomAction() {
        CATransaction.flush()
        let layer = CALayer()
        layer.actions = ["rasterizationScale": CABasicAnimation()]

        CATransaction.begin()
        CATransaction.setAnimationDuration(0.4)
        layer.rasterizationScale = 2
        CATransaction.commit()

        let animation = layer.animation(forKey: "rasterizationScale") as? CABasicAnimation
        #expect(animation?.keyPath == "rasterizationScale")
        #expect(animation?.fromValue as? CGFloat == 1)
        #expect(animation?.toValue as? CGFloat == 2)
        #expect(animation?.duration == 0.4)
    }
}
