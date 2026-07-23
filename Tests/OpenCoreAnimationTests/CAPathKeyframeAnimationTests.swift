import Testing
@testable import OpenCoreAnimation

@Suite("CAKeyframeAnimation path evaluation")
struct CAPathKeyframeAnimationTests {
    private let epsilon: CGFloat = 0.0001

    private func presentation(
        for animation: CAKeyframeAnimation,
        modelPosition: CGPoint = .zero,
        modelTransform: CATransform3D = CATransform3DIdentity,
        elapsed: CFTimeInterval
    ) throws -> CALayer {
        let layer = CALayer()
        layer.position = modelPosition
        layer.transform = modelTransform
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = elapsed
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        layer.add(animation, forKey: "path")
        return try #require(layer.presentation())
    }

    private func twoSegmentPath() -> CGPath {
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: 10, y: 0))
        path.addLine(to: CGPoint(x: 10, y: 90))
        return path
    }

    @Test("Linear and paced modes use segment time and path distance respectively")
    func linearAndPacedTiming() throws {
        let linear = CAKeyframeAnimation(keyPath: "position")
        linear.path = twoSegmentPath()
        linear.calculationMode = .linear
        let linearQuarter = try presentation(for: linear, elapsed: 0.25)
        #expect(abs(linearQuarter.position.x - 5) < epsilon)
        #expect(abs(linearQuarter.position.y) < epsilon)

        let paced = CAKeyframeAnimation(keyPath: "position")
        paced.path = twoSegmentPath()
        paced.calculationMode = .paced
        let pacedQuarter = try presentation(for: paced, elapsed: 0.25)
        #expect(abs(pacedQuarter.position.x - 10) < epsilon)
        #expect(abs(pacedQuarter.position.y - 15) < epsilon)

        let pacedHalf = try presentation(for: paced, elapsed: 0.5)
        #expect(abs(pacedHalf.position.x - 10) < epsilon)
        #expect(abs(pacedHalf.position.y - 40) < epsilon)
    }

    @Test("Path key times, segment timing functions, and discrete mode are honored")
    func segmentTimingControls() throws {
        let keyed = CAKeyframeAnimation(keyPath: "position")
        keyed.path = twoSegmentPath()
        keyed.keyTimes = [0, 0.8, 1]
        let keyedQuarter = try presentation(for: keyed, elapsed: 0.25)
        #expect(abs(keyedQuarter.position.x - 3.125) < epsilon)
        #expect(abs(keyedQuarter.position.y) < epsilon)

        let eased = CAKeyframeAnimation(keyPath: "position")
        eased.path = twoSegmentPath()
        eased.timingFunctions = [
            CAMediaTimingFunction(name: .easeIn),
            CAMediaTimingFunction(name: .easeOut),
        ]
        let easedQuarter = try presentation(for: eased, elapsed: 0.25)
        #expect(abs(easedQuarter.position.x - 3.153568) < 0.0001)

        let discrete = CAKeyframeAnimation(keyPath: "position")
        discrete.path = twoSegmentPath()
        discrete.calculationMode = .discrete
        let discreteValue = try presentation(for: discrete, elapsed: 0.75)
        #expect(abs(discreteValue.position.x - 10) < epsilon)
        #expect(abs(discreteValue.position.y) < epsilon)

        let invalidTimes = CAKeyframeAnimation(keyPath: "position")
        invalidTimes.path = twoSegmentPath()
        invalidTimes.keyTimes = [0, 0.2]
        let defaultTimed = try presentation(for: invalidTimes, elapsed: 0.25)
        #expect(abs(defaultTimed.position.x - 5) < epsilon)
        #expect(abs(defaultTimed.position.y) < epsilon)
    }

    @Test("Bezier paths preserve parametric and paced evaluation")
    func bezierEvaluation() throws {
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addCurve(
            to: CGPoint(x: 100, y: 0),
            control1: CGPoint(x: 0, y: 100),
            control2: CGPoint(x: 100, y: 100)
        )

        let linear = CAKeyframeAnimation(keyPath: "position")
        linear.path = path
        let linearQuarter = try presentation(for: linear, elapsed: 0.25)
        #expect(abs(linearQuarter.position.x - 15.625) < epsilon)
        #expect(abs(linearQuarter.position.y - 56.25) < epsilon)

        let paced = CAKeyframeAnimation(keyPath: "position")
        paced.path = path
        paced.calculationMode = .paced
        let pacedQuarter = try presentation(for: paced, elapsed: 0.25)
        // QuartzCore approximates path arc length; accept its measured sampling
        // error while requiring the same constant-velocity trajectory.
        #expect(abs(pacedQuarter.position.x - 10.647822) < 0.15)
        #expect(abs(pacedQuarter.position.y - 48.460465) < 0.15)
    }

    @Test("Move-to discontinuities do not become traversed path geometry")
    func subpathDiscontinuities() throws {
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: 10, y: 0))
        path.move(to: CGPoint(x: 100, y: 100))
        path.addLine(to: CGPoint(x: 100, y: 110))

        for mode in [CAAnimationCalculationMode.linear, .paced] {
            let animation = CAKeyframeAnimation(keyPath: "position")
            animation.path = path
            animation.calculationMode = mode

            let quarter = try presentation(for: animation, elapsed: 0.25)
            #expect(abs(quarter.position.x - 5) < epsilon)
            #expect(abs(quarter.position.y) < epsilon)

            let half = try presentation(for: animation, elapsed: 0.5)
            #expect(abs(half.position.x - 100) < epsilon)
            #expect(abs(half.position.y - 100) < epsilon)
        }
    }

    @Test("Closed subpaths include their closing segment")
    func closedSubpath() throws {
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: 10, y: 0))
        path.addLine(to: CGPoint(x: 0, y: 10))
        path.closeSubpath()

        let animation = CAKeyframeAnimation(keyPath: "position")
        animation.path = path
        let value = try presentation(for: animation, elapsed: 5.0 / 6.0)
        #expect(abs(value.position.x) < epsilon)
        #expect(abs(value.position.y - 5) < epsilon)
    }

    @Test("Rotation modes concatenate with the model transform")
    func rotationComposition() throws {
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: 0, y: 100))
        let baseTransform = CATransform3DMakeScale(2, 3, 1)

        let automatic = CAKeyframeAnimation(keyPath: "position")
        automatic.path = path
        automatic.rotationMode = .rotateAuto
        let automaticValue = try presentation(
            for: automatic,
            modelTransform: baseTransform,
            elapsed: 0.5
        )
        #expect(abs(automaticValue.transform.m11) < epsilon)
        #expect(abs(automaticValue.transform.m12 - 3) < epsilon)
        #expect(abs(automaticValue.transform.m21 + 2) < epsilon)
        #expect(abs(automaticValue.transform.m22) < epsilon)

        let reverse = CAKeyframeAnimation(keyPath: "position")
        reverse.path = path
        reverse.rotationMode = .rotateAutoReverse
        let reverseValue = try presentation(
            for: reverse,
            modelTransform: baseTransform,
            elapsed: 0.5
        )
        #expect(abs(reverseValue.transform.m11) < epsilon)
        #expect(abs(reverseValue.transform.m12 + 3) < epsilon)
        #expect(abs(reverseValue.transform.m21 - 2) < epsilon)
        #expect(abs(reverseValue.transform.m22) < epsilon)
    }

    @Test("Additive and cumulative paths preserve model and cycle contributions")
    func additiveAndCumulativePaths() throws {
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: 10, y: 20))

        let additive = CAKeyframeAnimation(keyPath: "position")
        additive.path = path
        additive.isAdditive = true
        let additiveValue = try presentation(
            for: additive,
            modelPosition: CGPoint(x: 50, y: 60),
            elapsed: 0.5
        )
        #expect(abs(additiveValue.position.x - 55) < epsilon)
        #expect(abs(additiveValue.position.y - 70) < epsilon)

        let cumulative = CAKeyframeAnimation(keyPath: "position")
        cumulative.path = path
        cumulative.isCumulative = true
        cumulative.repeatCount = 3
        let cumulativeValue = try presentation(for: cumulative, elapsed: 1.5)
        #expect(abs(cumulativeValue.position.x - 15) < epsilon)
        #expect(abs(cumulativeValue.position.y - 30) < epsilon)

        let combined = CAKeyframeAnimation(keyPath: "position")
        combined.path = path
        combined.isAdditive = true
        combined.isCumulative = true
        combined.repeatCount = 3
        let combinedValue = try presentation(
            for: combined,
            modelPosition: CGPoint(x: 50, y: 60),
            elapsed: 1.5
        )
        #expect(abs(combinedValue.position.x - 65) < epsilon)
        #expect(abs(combinedValue.position.y - 90) < epsilon)
    }

    @Test("Invalid path geometry does not partially mutate presentation state")
    func invalidGeometry() throws {
        let path = CGMutablePath()
        path.move(to: .zero)
        path.addLine(to: CGPoint(x: CGFloat.infinity, y: 10))

        let animation = CAKeyframeAnimation(keyPath: "position")
        animation.path = path
        animation.rotationMode = .rotateAuto
        let modelPosition = CGPoint(x: 12, y: 34)
        let modelTransform = CATransform3DMakeScale(2, 3, 1)
        let value = try presentation(
            for: animation,
            modelPosition: modelPosition,
            modelTransform: modelTransform,
            elapsed: 0.5
        )
        #expect(value.position == modelPosition)
        #expect(CATransform3DEqualToTransform(value.transform, modelTransform))
    }

    @Test("Finite path inputs that overflow during evaluation are rejected atomically")
    func overflowingGeometry() throws {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: -CGFloat.greatestFiniteMagnitude, y: 0))
        path.addLine(to: CGPoint(x: CGFloat.greatestFiniteMagnitude, y: 0))

        for mode in [CAAnimationCalculationMode.linear, .paced] {
            let animation = CAKeyframeAnimation(keyPath: "position")
            animation.path = path
            animation.calculationMode = mode
            animation.rotationMode = .rotateAuto
            let modelPosition = CGPoint(x: 12, y: 34)
            let modelTransform = CATransform3DMakeScale(2, 3, 1)
            let value = try presentation(
                for: animation,
                modelPosition: modelPosition,
                modelTransform: modelTransform,
                elapsed: 0.5
            )
            #expect(value.position == modelPosition)
            #expect(CATransform3DEqualToTransform(value.transform, modelTransform))
        }
    }

    @Test("Overflowing path contributions do not partially mutate presentation state")
    func overflowingContributions() throws {
        let finitePath = CGMutablePath()
        finitePath.move(to: .zero)
        finitePath.addLine(to: CGPoint(x: 10, y: 10))
        let modelPosition = CGPoint(x: CGFloat.greatestFiniteMagnitude, y: 34)
        var overflowingTransform = CATransform3DIdentity
        overflowingTransform.m11 = CGFloat.greatestFiniteMagnitude
        overflowingTransform.m21 = CGFloat.greatestFiniteMagnitude

        let additivePath = CGMutablePath()
        additivePath.move(to: .zero)
        additivePath.addLine(to: CGPoint(x: CGFloat.greatestFiniteMagnitude, y: 0))
        let additive = CAKeyframeAnimation(keyPath: "position")
        additive.path = additivePath
        additive.isAdditive = true
        let additiveValue = try presentation(
            for: additive,
            modelPosition: modelPosition,
            elapsed: 0.5
        )
        #expect(additiveValue.position == modelPosition)

        let rotation = CAKeyframeAnimation(keyPath: "position")
        rotation.path = finitePath
        rotation.rotationMode = .rotateAuto
        let rotationValue = try presentation(
            for: rotation,
            modelPosition: CGPoint(x: 12, y: 34),
            modelTransform: overflowingTransform,
            elapsed: 0.5
        )
        #expect(rotationValue.position == CGPoint(x: 12, y: 34))
        #expect(CATransform3DEqualToTransform(rotationValue.transform, overflowingTransform))

        let cumulativePath = CGMutablePath()
        cumulativePath.move(to: .zero)
        cumulativePath.addLine(to: CGPoint(x: CGFloat.greatestFiniteMagnitude, y: 0))
        let cumulative = CAKeyframeAnimation(keyPath: "position")
        cumulative.path = cumulativePath
        cumulative.isCumulative = true
        cumulative.repeatCount = 3
        let cumulativeValue = try presentation(
            for: cumulative,
            modelPosition: CGPoint(x: 12, y: 34),
            elapsed: 2
        )
        #expect(cumulativeValue.position == CGPoint(x: 12, y: 34))
    }
}
