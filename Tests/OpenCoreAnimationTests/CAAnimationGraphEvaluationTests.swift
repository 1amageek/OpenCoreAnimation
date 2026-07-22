import Testing
@testable import OpenCoreAnimation

@Suite("CAAnimation graph evaluation")
struct CAAnimationGraphEvaluationTests {
    private let epsilon: CGFloat = 0.0001

    @Test("Group additive children apply after non-additive siblings")
    func groupAdditiveOrdering() throws {
        let layer = CALayer()
        layer.opacity = 0.2
        let group = frozenGroup(
            animations: [additiveOpacity(), replacementOpacity()],
            elapsed: 0.5
        )

        layer.add(group, forKey: "group")
        let result = try #require(layer.presentation())

        #expect(abs(CGFloat(result.opacity) - 0.4) < epsilon)
    }

    @Test("Nested additive groups cannot be overwritten by outer siblings")
    func nestedGroupAdditiveOrdering() throws {
        let layer = CALayer()
        layer.opacity = 0.2
        let nested = CAAnimationGroup()
        nested.duration = 1
        nested.fillMode = .both
        nested.animations = [additiveOpacity()]
        let outer = frozenGroup(
            animations: [nested, replacementOpacity()],
            elapsed: 0.5
        )

        layer.add(outer, forKey: "outer")
        let result = try #require(layer.presentation())

        #expect(abs(CGFloat(result.opacity) - 0.4) < epsilon)
    }

    @Test("Additive group descendants apply after top-level replacements")
    func rootGraphAdditiveOrdering() throws {
        let layer = CALayer()
        layer.opacity = 0.2
        let group = frozenGroup(animations: [additiveOpacity()], elapsed: 0.5)
        let replacement = replacementOpacity()
        freeze(replacement, elapsed: 0.5)

        layer.add(group, forKey: "additive-group")
        layer.add(replacement, forKey: "replacement")
        let result = try #require(layer.presentation())

        #expect(abs(CGFloat(result.opacity) - 0.4) < epsilon)
    }

    @Test("Transitions nested in groups retain snapshots and group basic time")
    func groupedTransition() throws {
        let layer = CALayer()
        layer.backgroundColor = color(1, 0, 0, 1)
        let transition = CATransition()
        transition.type = .fade
        transition.duration = 1
        transition.fillMode = .both
        let group = frozenGroup(animations: [transition], elapsed: 0.25)

        layer.add(group, forKey: "transition-group")
        layer.backgroundColor = color(0, 0, 1, 1)
        let result = try #require(layer.presentation())
        let state = try #require(result._transitionRenderState)

        #expect(abs(CGFloat(state.progress) - 0.25) < epsilon)
        #expect(state.type == .fade)
        expectColor(try #require(state.sourceLayer.backgroundColor), [1, 0, 0, 1])
        expectColor(try #require(result.backgroundColor), [0, 0, 1, 1])
    }

    private func additiveOpacity() -> CABasicAnimation {
        let animation = replacementOpacity()
        animation.isAdditive = true
        return animation
    }

    private func replacementOpacity() -> CABasicAnimation {
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(0.4)
        animation.duration = 1
        animation.fillMode = .both
        return animation
    }

    private func frozenGroup(
        animations: [CAAnimation],
        elapsed: CFTimeInterval
    ) -> CAAnimationGroup {
        let group = CAAnimationGroup()
        group.animations = animations
        group.duration = 1
        freeze(group, elapsed: elapsed)
        return group
    }

    private func freeze(_ animation: CAAnimation, elapsed: CFTimeInterval) {
        animation.speed = 0
        animation.timeOffset = elapsed
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
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
}
