import Testing
@testable import OpenCoreAnimation

@Suite("CALayer shadowPath keyframe animation evaluation")
struct CAShadowPathKeyframeAnimationTests {
    private func rectangle(x: CGFloat) -> CGPath {
        CGPath(rect: CGRect(x: x, y: 0, width: 20, height: 20))
    }

    private func presentation(
        for animation: CAKeyframeAnimation,
        elapsed: CFTimeInterval
    ) throws -> CALayer {
        let layer = CALayer()
        layer.shadowPath = rectangle(x: 40)
        animation.duration = 1
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        layer.add(animation, forKey: "shadowPath")
        setStoredAnimationBeginTime(
            CACurrentMediaTime() - elapsed,
            on: layer,
            forKey: "shadowPath"
        )
        return try #require(layer.presentation())
    }

    @Test("Single-value shadow paths update presentation state")
    func singleValueAppliesPath() throws {
        let value = rectangle(x: 10)
        let animation = CAKeyframeAnimation(keyPath: "shadowPath")
        animation.values = [value]

        let result = try #require(presentation(for: animation, elapsed: 0.5).shadowPath)

        #expect(result.boundingBox == value.boundingBox)
    }

    @Test("Discrete shadow paths retain the latest reached path")
    func discreteAnimationSelectsReachedPath() throws {
        let first = rectangle(x: 0)
        let second = rectangle(x: 20)
        let animation = CAKeyframeAnimation(keyPath: "shadowPath")
        animation.values = [first, second]
        animation.keyTimes = [0, 0.8, 1]
        animation.calculationMode = .discrete

        let result = try #require(presentation(for: animation, elapsed: 0.75).shadowPath)

        #expect(result.boundingBox == first.boundingBox)
    }

    @Test("Linear shadow paths interpolate compatible control points")
    func linearAnimationInterpolatesPath() throws {
        let animation = CAKeyframeAnimation(keyPath: "shadowPath")
        animation.values = [rectangle(x: 0), rectangle(x: 20)]

        let result = try #require(presentation(for: animation, elapsed: 0.5).shadowPath)

        #expect(abs(result.boundingBox.minX - 10) < 0.01)
        #expect(abs(result.boundingBox.width - 20) < 0.01)
    }

    @Test("Cubic shadow paths apply their interpolated path")
    func cubicAnimationAppliesPath() throws {
        let animation = CAKeyframeAnimation(keyPath: "shadowPath")
        animation.values = [rectangle(x: 0), rectangle(x: 20)]
        animation.calculationMode = .cubic

        let result = try #require(presentation(for: animation, elapsed: 0.5).shadowPath)

        #expect(abs(result.boundingBox.minX - 10) < 0.01)
        #expect(abs(result.boundingBox.width - 20) < 0.01)
    }
}
