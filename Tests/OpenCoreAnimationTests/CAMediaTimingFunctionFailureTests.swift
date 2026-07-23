import Testing
@testable import OpenCoreAnimation

@Suite("Timing function failure handling")
struct CAMediaTimingFunctionFailureTests {
    private func invalidTimingFunction() -> CAMediaTimingFunction {
        CAMediaTimingFunction(
            controlPoints: 0.25,
            .nan,
            0.75,
            1
        )
    }

    @Test("Non-finite control points remain explicit evaluation failures")
    func publicEvaluationPreservesFailure() {
        let function = invalidTimingFunction()
        #expect(function.evaluate(at: 0.5).isNaN)
        #expect(function.evaluateIfFinite(at: 0.5) == nil)
    }

    @Test("Predefined names resolve to their documented control points")
    func predefinedNamesResolveExactly() {
        let expected: [(CAMediaTimingFunctionName, [Float])] = [
            (.linear, [0, 0, 1, 1]),
            (.easeIn, [0.42, 0, 1, 1]),
            (.easeOut, [0, 0, 0.58, 1]),
            (.easeInEaseOut, [0.42, 0, 0.58, 1]),
            (.default, [0.25, 0.1, 0.25, 1]),
        ]

        for (name, components) in expected {
            let function = CAMediaTimingFunction(name: name)
            var first = [Float](repeating: 0, count: 2)
            var second = [Float](repeating: 0, count: 2)
            function.getControlPoint(at: 1, values: &first)
            function.getControlPoint(at: 2, values: &second)
            #expect(first + second == components)
        }
    }

    @Test("Unknown predefined names fail instead of becoming linear")
    func unknownNameFails() async {
        await #expect(processExitsWith: .failure) {
            _ = CAMediaTimingFunction(
                name: CAMediaTimingFunctionName(rawValue: "future")
            )
        }
    }

    @Test("Basic and keyframe timing failures leave presentation unchanged")
    func propertyAnimationsFailAtomically() throws {
        let basicLayer = CALayer()
        basicLayer.zPosition = 7
        let basic = CABasicAnimation(keyPath: "zPosition")
        basic.fromValue = CGFloat(0)
        basic.toValue = CGFloat(100)
        basic.timingFunction = invalidTimingFunction()
        configure(basic)
        basicLayer.add(basic, forKey: "invalidBasicTiming")
        #expect(try #require(basicLayer.presentation()).zPosition == 7)

        let keyframeLayer = CALayer()
        keyframeLayer.zPosition = 9
        let keyframe = CAKeyframeAnimation(keyPath: "zPosition")
        keyframe.values = [CGFloat(0), CGFloat(100)]
        keyframe.timingFunctions = [invalidTimingFunction()]
        configure(keyframe)
        keyframeLayer.add(keyframe, forKey: "invalidKeyframeTiming")
        #expect(try #require(keyframeLayer.presentation()).zPosition == 9)
    }

    @Test("Path and group timing failures leave presentation unchanged")
    func graphAnimationsFailAtomically() throws {
        let pathLayer = CALayer()
        pathLayer.position = CGPoint(x: 3, y: 4)
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: 100, y: 0))
        let pathAnimation = CAKeyframeAnimation(keyPath: "position")
        pathAnimation.path = path
        pathAnimation.timingFunctions = [invalidTimingFunction()]
        configure(pathAnimation)
        pathLayer.add(pathAnimation, forKey: "invalidPathTiming")
        #expect(try #require(pathLayer.presentation()).position == CGPoint(x: 3, y: 4))

        let groupLayer = CALayer()
        groupLayer.zPosition = 11
        let child = CABasicAnimation(keyPath: "zPosition")
        child.fromValue = CGFloat(0)
        child.toValue = CGFloat(100)
        child.duration = 1
        let group = CAAnimationGroup()
        group.animations = [child]
        group.duration = 1
        group.speed = 0
        group.timeOffset = 0.5
        group.timingFunction = invalidTimingFunction()
        groupLayer.add(group, forKey: "invalidGroupTiming")
        #expect(try #require(groupLayer.presentation()).zPosition == 11)

        let transitionLayer = CALayer()
        let transition = CATransition()
        transition.timingFunction = invalidTimingFunction()
        configure(transition)
        transitionLayer.add(transition, forKey: "invalidTransitionTiming")
        #expect(try #require(transitionLayer.presentation())._transitionRenderState == nil)
    }

    private func configure(_ animation: CAAnimation) {
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
    }
}
