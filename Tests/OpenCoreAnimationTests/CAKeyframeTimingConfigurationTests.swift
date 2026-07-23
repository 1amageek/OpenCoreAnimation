import Testing
@testable import OpenCoreAnimation

@Suite("Keyframe timing configuration")
struct CAKeyframeTimingConfigurationTests {
    private let epsilon: CGFloat = 0.001

    private func sampledZPosition(
        values: [CGFloat],
        keyTimes: [CGFloat]?,
        calculationMode: CAAnimationCalculationMode = .linear,
        progress: CFTimeInterval,
        modelValue: CGFloat = 0
    ) throws -> CGFloat {
        let layer = CALayer()
        layer.zPosition = modelValue
        let animation = CAKeyframeAnimation(keyPath: "zPosition")
        animation.values = values
        animation.keyTimes = keyTimes
        animation.calculationMode = calculationMode
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = progress
        layer.add(animation, forKey: "timingConfiguration")
        return try #require(layer.presentation()).zPosition
    }

    @Test("Linear key times require complete finite unit-range endpoints")
    func linearKeyTimesUseDocumentedValidation() throws {
        let values = [CGFloat(0), CGFloat(10), CGFloat(100)]
        let explicit = try sampledZPosition(
            values: values,
            keyTimes: [0, 0.8, 1],
            progress: 0.25
        )
        #expect(abs(explicit - 3.125) < epsilon)

        let invalidKeyTimes: [[CGFloat]] = [
            [0, 1],
            [-1, 0.5, 1],
            [0.2, 0.5, 1],
            [0, 0.5, 0.8],
            [0, 0.75, 0.5],
            [0, .nan, 1],
        ]
        for keyTimes in invalidKeyTimes {
            let sampled = try sampledZPosition(
                values: values,
                keyTimes: keyTimes,
                progress: 0.25
            )
            #expect(abs(sampled - 5) < epsilon)
        }
    }

    @Test("Discrete values use one more key time than value count")
    func discreteKeyTimesDescribeValueIntervals() throws {
        let values = [CGFloat(10), CGFloat(20)]
        #expect(try sampledZPosition(
            values: values,
            keyTimes: [0, 0.25, 1],
            calculationMode: .discrete,
            progress: 0.2
        ) == 10)
        #expect(try sampledZPosition(
            values: values,
            keyTimes: [0, 0.25, 1],
            calculationMode: .discrete,
            progress: 0.3
        ) == 20)
        #expect(try sampledZPosition(
            values: values,
            keyTimes: [0, 1],
            calculationMode: .discrete,
            progress: 0.3
        ) == 10)
    }

    @Test("Unknown calculation modes do not fabricate linear animation")
    func unknownCalculationModesDoNotApply() throws {
        let unknown = CAAnimationCalculationMode(rawValue: "unknown")
        let sampled = try sampledZPosition(
            values: [0, 100],
            keyTimes: nil,
            calculationMode: unknown,
            progress: 0.5,
            modelValue: 7
        )
        #expect(sampled == 7)

        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: 100, y: 0))
        let layer = CALayer()
        layer.position = CGPoint(x: 3, y: 4)
        let animation = CAKeyframeAnimation(keyPath: "position")
        animation.path = path
        animation.calculationMode = unknown
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
        layer.add(animation, forKey: "unknownPathMode")
        #expect(try #require(layer.presentation()).position == CGPoint(x: 3, y: 4))
    }

    @Test("Path key times share finite unit-range validation")
    func pathKeyTimesShareValidation() throws {
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: 10, y: 0))
        path.addLine(to: CGPoint(x: 10, y: 90))

        func sampledPosition(keyTimes: [CGFloat]) throws -> CGPoint {
            let layer = CALayer()
            let animation = CAKeyframeAnimation(keyPath: "position")
            animation.path = path
            animation.keyTimes = keyTimes
            animation.duration = 1
            animation.speed = 0
            animation.timeOffset = 0.25
            layer.add(animation, forKey: "pathTiming")
            return try #require(layer.presentation()).position
        }

        let valid = try sampledPosition(keyTimes: [0, 0.8, 1])
        #expect(abs(valid.x - 3.125) < epsilon)
        #expect(abs(valid.y) < epsilon)

        let invalidKeyTimes: [[CGFloat]] = [
            [-1, 0.5, 1],
            [0.2, 0.5, 1],
            [0, .infinity, 1],
        ]
        for keyTimes in invalidKeyTimes {
            let sampled = try sampledPosition(keyTimes: keyTimes)
            #expect(abs(sampled.x - 5) < epsilon)
            #expect(abs(sampled.y) < epsilon)
        }
    }
}
