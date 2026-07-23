import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CALayer contents animation evaluation")
struct CAContentsAnimationTests {
    private final class ContentsToken {}

    private func presentation(
        for animation: CAAnimation,
        modelContents: Any,
        elapsed: CFTimeInterval
    ) throws -> CALayer {
        let layer = CALayer()
        layer.contents = modelContents
        animation.duration = 1
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        layer.add(animation, forKey: "contents")
        setStoredAnimationBeginTime(
            CACurrentMediaTime() - elapsed,
            on: layer,
            forKey: "contents"
        )
        return try #require(layer.presentation())
    }

    @Test("Basic contents animations switch at the segment midpoint")
    func basicAnimationUsesMidpointSelection() throws {
        let from = ContentsToken()
        let to = ContentsToken()

        let before = CABasicAnimation(keyPath: "contents")
        before.fromValue = from
        before.toValue = to
        let beforePresentation = try presentation(for: before, modelContents: to, elapsed: 0.49)

        let atMidpoint = CABasicAnimation(keyPath: "contents")
        atMidpoint.fromValue = from
        atMidpoint.toValue = to
        let midpointPresentation = try presentation(for: atMidpoint, modelContents: to, elapsed: 0.5)

        #expect(beforePresentation.contents as? ContentsToken === from)
        #expect(midpointPresentation.contents as? ContentsToken === to)
    }

    @Test("Linear contents keyframes select the nearest endpoint in each segment")
    func linearKeyframesUseSegmentMidpoints() throws {
        let first = ContentsToken()
        let second = ContentsToken()
        let third = ContentsToken()
        let animation = CAKeyframeAnimation(keyPath: "contents")
        animation.values = [first, second, third]
        animation.keyTimes = [0, 0.5, 1]

        let presentation = try presentation(for: animation, modelContents: third, elapsed: 0.25)

        #expect(presentation.contents as? ContentsToken === second)
    }

    @Test("Cubic contents keyframes retain object selection semantics")
    func cubicKeyframesUseSegmentMidpoints() throws {
        let first = ContentsToken()
        let second = ContentsToken()
        let third = ContentsToken()
        let animation = CAKeyframeAnimation(keyPath: "contents")
        animation.values = [first, second, third]
        animation.keyTimes = [0, 0.5, 1]
        animation.calculationMode = .cubic

        let presentation = try presentation(for: animation, modelContents: third, elapsed: 0.75)

        #expect(presentation.contents as? ContentsToken === third)
    }

    @Test("Discrete contents keyframes hold the latest reached value")
    func discreteKeyframesHoldReachedValue() throws {
        let first = ContentsToken()
        let second = ContentsToken()
        let third = ContentsToken()
        let animation = CAKeyframeAnimation(keyPath: "contents")
        animation.values = [first, second, third]
        animation.keyTimes = [0, 0.5, 0.8, 1]
        animation.calculationMode = .discrete

        let presentation = try presentation(for: animation, modelContents: third, elapsed: 0.75)

        #expect(presentation.contents as? ContentsToken === second)
    }
}
