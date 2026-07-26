import Foundation

/// Describes why an animation graph could not cross the transaction boundary.
public enum CACommittedAnimationCaptureError:
    Error,
    Equatable,
    Sendable
{
    case unsupportedAnimationType(String)
    case unsupportedValueType(String)
    case nonFiniteValue(String)
    case invalidImage(CAImageContentsConversionError)
    case invalidTransitionFilter(CATransitionRenderFailure)
    indirect case invalidTransitionSource(CARendererError)
}

/// Value-owned image storage used only inside a committed animation evaluator.
///
/// Keeping the renderer-ready storage avoids reconstructing a `CGImage` and
/// converting the same pixels a second time on every evaluated frame.
internal struct CACommittedImageAnimationValue: Sendable {
    internal let storage: CGImageTextureStorage
}

/// A validated value accepted by the built-in animation evaluator.
internal enum CACommittedAnimationValue: Sendable {
    case bool(Bool)
    case float(Float)
    case double(Double)
    case scalar(CGFloat)
    case integer(Int)
    case point(CGPoint)
    case size(CGSize)
    case rect(CGRect)
    case transform(CATransform3D)
    case color(SIMD4<Double>)
    case path(CACommittedPath)
    case scalars([CGFloat])
    case doubles([Double])
    case floats([Float])
    case integers([Int])
    case colors([SIMD4<Double>])
    case image(CGImageTextureStorage)

    static func capture(
        _ value: Any
    ) throws(CACommittedAnimationCaptureError) -> Self {
        if let value = value as? Bool {
            return .bool(value)
        }
        if let value = value as? Float {
            guard value.isFinite else {
                throw .nonFiniteValue("Float")
            }
            return .float(value)
        }
        if type(of: value) == CGFloat.self,
           let value = value as? CGFloat {
            guard value.isFinite else {
                throw .nonFiniteValue("CGFloat")
            }
            return .scalar(value)
        }
        if let value = value as? Double {
            guard value.isFinite else {
                throw .nonFiniteValue("Double")
            }
            return .double(value)
        }
        if let value = value as? Int {
            return .integer(value)
        }
        if let value = value as? CGPoint {
            guard value.x.isFinite, value.y.isFinite else {
                throw .nonFiniteValue("CGPoint")
            }
            return .point(value)
        }
        if let value = value as? CGSize {
            guard value.width.isFinite,
                  value.height.isFinite else {
                throw .nonFiniteValue("CGSize")
            }
            return .size(value)
        }
        if let value = value as? CGRect {
            guard value.origin.x.isFinite,
                  value.origin.y.isFinite,
                  value.size.width.isFinite,
                  value.size.height.isFinite else {
                throw .nonFiniteValue("CGRect")
            }
            return .rect(value)
        }
        if let value = value as? CATransform3D {
            guard isFinite(value) else {
                throw .nonFiniteValue("CATransform3D")
            }
            return .transform(value)
        }
        if let value = value as? CGColor {
            return .color(try captureColor(value))
        }
        if let value = value as? CGPath {
            return .path(try CACommittedPath(capturing: value))
        }
        if let value = value as? [CGFloat] {
            guard value.allSatisfy(\.isFinite) else {
                throw .nonFiniteValue("[CGFloat]")
            }
            return .scalars(value)
        }
        if let value = value as? [Double] {
            guard value.allSatisfy(\.isFinite) else {
                throw .nonFiniteValue("[Double]")
            }
            return .doubles(value)
        }
        if let value = value as? [Float] {
            guard value.allSatisfy(\.isFinite) else {
                throw .nonFiniteValue("[Float]")
            }
            return .floats(value)
        }
        if let value = value as? [Int] {
            return .integers(value)
        }
        if let values = value as? [Any] {
            var colors: [SIMD4<Double>] = []
            colors.reserveCapacity(values.count)
            for value in values {
                guard let color = value as? CGColor else {
                    throw .unsupportedValueType(
                        String(reflecting: type(of: value))
                    )
                }
                colors.append(try captureColor(color))
            }
            return .colors(colors)
        }
        if let image = value as? CGImage {
            do {
                return .image(
                    try CGImageTextureStorageConverter.convert(image)
                )
            } catch {
                throw .invalidImage(error)
            }
        }
        throw .unsupportedValueType(
            String(reflecting: type(of: value))
        )
    }

    func materialize()
        throws(CACommittedAnimationCaptureError) -> Any
    {
        switch self {
        case .bool(let value): return value
        case .float(let value): return value
        case .double(let value): return value
        case .scalar(let value): return value
        case .integer(let value): return value
        case .point(let value): return value
        case .size(let value): return value
        case .rect(let value): return value
        case .transform(let value): return value
        case .color(let value):
            return Self.materializeColor(value)
        case .path(let value): return value.materialize()
        case .scalars(let value): return value
        case .doubles(let value): return value
        case .floats(let value): return value
        case .integers(let value): return value
        case .colors(let value):
            return value.map {
                Self.materializeColor($0) as Any
            }
        case .image(let storage):
            return CACommittedImageAnimationValue(
                storage: storage
            )
        }
    }

    private static func captureColor(
        _ color: CGColor
    ) throws(CACommittedAnimationCaptureError)
        -> SIMD4<Double>
    {
        guard let converted = color.converted(
            to: .deviceRGB,
            intent: .defaultIntent,
            options: nil
        ), let components = converted.components,
           components.count == 4,
           components.allSatisfy(\.isFinite) else {
            throw .unsupportedValueType("CGColor")
        }
        return SIMD4(
            Double(components[0]),
            Double(components[1]),
            Double(components[2]),
            Double(components[3])
        )
    }

    private static func materializeColor(
        _ components: SIMD4<Double>
    ) -> CGColor {
        CGColor(
            red: CGFloat(components.x),
            green: CGFloat(components.y),
            blue: CGFloat(components.z),
            alpha: CGFloat(components.w)
        )
    }

    private static func isFinite(
        _ value: CATransform3D
    ) -> Bool {
        value.m11.isFinite && value.m12.isFinite
            && value.m13.isFinite && value.m14.isFinite
            && value.m21.isFinite && value.m22.isFinite
            && value.m23.isFinite && value.m24.isFinite
            && value.m31.isFinite && value.m32.isFinite
            && value.m33.isFinite && value.m34.isFinite
            && value.m41.isFinite && value.m42.isFinite
            && value.m43.isFinite && value.m44.isFinite
    }

}

internal struct CACommittedAnimationSnapshot: Sendable {
    internal struct TimingFunction: Sendable {
        let controlPoints: SIMD4<Float>

        init(
            _ function: CAMediaTimingFunction
        ) throws(CACommittedAnimationCaptureError) {
            controlPoints = function.committedControlPoints
            guard controlPoints.x.isFinite,
                  controlPoints.y.isFinite,
                  controlPoints.z.isFinite,
                  controlPoints.w.isFinite else {
                throw .nonFiniteValue(
                    "CAMediaTimingFunction"
                )
            }
        }

        func materialize() -> CAMediaTimingFunction {
            CAMediaTimingFunction(
                controlPoints: controlPoints.x,
                controlPoints.y,
                controlPoints.z,
                controlPoints.w
            )
        }
    }

    internal struct Common: Sendable {
        let beginTime: CFTimeInterval
        let timeOffset: CFTimeInterval
        let repeatCount: Float
        let repeatDuration: CFTimeInterval
        let duration: CFTimeInterval
        let speed: Float
        let autoreverses: Bool
        let fillMode: CAMediaTimingFillMode
        let timingFunction: TimingFunction?
        let preferredFrameRateRange: CAFrameRateRange
        let isRemovedOnCompletion: Bool
        let isFinished: Bool
        let hasStarted: Bool
    }

    internal struct Property: Sendable {
        let keyPath: String?
        let isAdditive: Bool
        let isCumulative: Bool
        let valueFunctionName: CAValueFunctionName?
    }

    internal struct Basic: Sendable {
        let fromValue: CACommittedAnimationValue?
        let toValue: CACommittedAnimationValue?
        let byValue: CACommittedAnimationValue?
    }

    internal struct Spring: Sendable {
        let mass: CGFloat
        let stiffness: CGFloat
        let damping: CGFloat
        let initialVelocity: CGFloat
        let allowsOverdamping: Bool
    }

    internal struct Keyframe: Sendable {
        let values: [CACommittedAnimationValue]?
        let path: CACommittedPath?
        let keyTimes: [CGFloat]?
        let timingFunctions: [TimingFunction]?
        let calculationMode: CAAnimationCalculationMode
        let tensionValues: [CGFloat]?
        let continuityValues: [CGFloat]?
        let biasValues: [CGFloat]?
        let rotationMode: CAAnimationRotationMode?
    }

    internal struct Transition: Sendable {
        let resourceIdentity: UInt64
        let type: CATransitionType
        let subtype: CATransitionSubtype?
        let startProgress: Float
        let endProgress: Float
        let filter: CARenderSnapshotTransition.Filter?
        let filterCaptureFailure:
            CATransitionRenderFailure?
        let sourceSnapshot: CARenderSnapshot?
    }

    internal enum Kind: Sendable {
        case basic(Property, Basic)
        case spring(Property, Basic, Spring)
        case keyframe(Property, Keyframe)
        case group([CACommittedAnimationSnapshot]?)
        case transition(Transition)
        case base
    }

    let common: Common
    let kind: Kind

    internal var affectsContents: Bool {
        switch kind {
        case .basic(let property, _),
             .spring(let property, _, _),
             .keyframe(let property, _):
            return property.keyPath == "contents"
        case .group(let animations):
            return animations?.contains {
                $0.affectsContents
            } == true
        case .transition, .base:
            return false
        }
    }

    static func capture(
        _ animation: CAAnimation,
        frameToken: UInt64
    ) throws(CACommittedAnimationCaptureError) -> sending Self {
        try validateCommonTiming(animation)
        let capturedTimingFunction: TimingFunction?
        if let timingFunction = animation.timingFunction {
            capturedTimingFunction =
                try TimingFunction(timingFunction)
        } else {
            capturedTimingFunction = nil
        }
        let common = Common(
            beginTime: animation.beginTime,
            timeOffset: animation.timeOffset,
            repeatCount: animation.repeatCount,
            repeatDuration: animation.repeatDuration,
            duration: animation.duration,
            speed: animation.speed,
            autoreverses: animation.autoreverses,
            fillMode: animation.fillMode,
            timingFunction: capturedTimingFunction,
            preferredFrameRateRange:
                animation.preferredFrameRateRange,
            isRemovedOnCompletion:
                animation.isRemovedOnCompletion,
            isFinished: animation.isFinished,
            hasStarted: animation.hasStarted
        )
        if let animation = animation as? CASpringAnimation {
            guard animation.mass.isFinite,
                  animation.stiffness.isFinite,
                  animation.damping.isFinite
                    || (
                        animation.allowsOverdamping
                        && animation.damping == .infinity
                    ),
                  animation.initialVelocity.isFinite else {
                throw .nonFiniteValue(
                    "CASpringAnimation"
                )
            }
            return Self(
                common: common,
                kind: .spring(
                    property(animation),
                    try basic(animation),
                    Spring(
                        mass: animation.mass,
                        stiffness: animation.stiffness,
                        damping: animation.damping,
                        initialVelocity:
                            animation.initialVelocity,
                        allowsOverdamping:
                            animation.allowsOverdamping
                    )
                )
            )
        }
        if let animation = animation as? CABasicAnimation {
            return Self(
                common: common,
                kind: .basic(
                    property(animation),
                    try basic(animation)
                )
            )
        }
        if let animation = animation as? CAKeyframeAnimation {
            var values: [CACommittedAnimationValue]?
            if let sourceValues = animation.values {
                var captured: [CACommittedAnimationValue] = []
                captured.reserveCapacity(sourceValues.count)
                for value in sourceValues {
                    captured.append(try .capture(value))
                }
                values = captured
            }
            let timingFunctions:
                [TimingFunction]?
            if let sourceFunctions =
                    animation.timingFunctions {
                var capturedFunctions:
                    [TimingFunction] = []
                capturedFunctions.reserveCapacity(
                    sourceFunctions.count
                )
                for function in sourceFunctions {
                    capturedFunctions.append(
                        try TimingFunction(function)
                    )
                }
                timingFunctions = capturedFunctions
            } else {
                timingFunctions = nil
            }
            return Self(
                common: common,
                kind: .keyframe(
                    property(animation),
                    Keyframe(
                        values: values,
                        path: try copiedPath(animation.path),
                        keyTimes: animation.keyTimes,
                        timingFunctions: timingFunctions,
                        calculationMode:
                            animation.calculationMode,
                        tensionValues: animation.tensionValues,
                        continuityValues:
                            animation.continuityValues,
                        biasValues: animation.biasValues,
                        rotationMode: animation.rotationMode
                    )
                )
            )
        }
        if let animation = animation as? CAAnimationGroup {
            var animations: [CACommittedAnimationSnapshot]?
            if let sourceAnimations = animation.animations {
                var captured: [CACommittedAnimationSnapshot] = []
                captured.reserveCapacity(sourceAnimations.count)
                for animation in sourceAnimations {
                    captured.append(
                        try capture(
                            animation,
                            frameToken: frameToken
                        )
                    )
                }
                animations = captured
            }
            return Self(
                common: common,
                kind: .group(animations)
            )
        }
        if let animation = animation as? CATransition {
            guard animation.startProgress.isFinite,
                  animation.endProgress.isFinite else {
                throw .nonFiniteValue("CATransition")
            }
            var filter: CARenderSnapshotTransition.Filter?
            var filterCaptureFailure:
                CATransitionRenderFailure?
            do {
                filter = try .capture(animation.filter)
                filterCaptureFailure = nil
            } catch {
                filter = nil
                filterCaptureFailure = error
            }
            return Self(
                common: common,
                kind: .transition(
                    Transition(
                        resourceIdentity:
                            animation.resourceIdentity,
                        type: animation.type,
                        subtype: animation.subtype,
                        startProgress: animation.startProgress,
                        endProgress: animation.endProgress,
                        filter: filter,
                        filterCaptureFailure:
                            filterCaptureFailure,
                        sourceSnapshot:
                            try captureTransitionSource(
                                animation,
                                frameToken: frameToken
                            )
                    )
                )
            )
        }
        guard type(of: animation) == CAAnimation.self else {
            throw .unsupportedAnimationType(
                String(reflecting: type(of: animation))
            )
        }
        return Self(common: common, kind: .base)
    }

    func materialize()
        throws(CACommittedAnimationCaptureError)
        -> sending CAAnimation
    {
        let animation: CAAnimation
        switch kind {
        case .basic(let property, let basic):
            let result = CABasicAnimation()
            apply(property, to: result)
            try apply(basic, to: result)
            animation = result
        case .spring(let property, let basic, let spring):
            let result = CASpringAnimation()
            apply(property, to: result)
            try apply(basic, to: result)
            result.mass = spring.mass
            result.stiffness = spring.stiffness
            result.damping = spring.damping
            result.initialVelocity = spring.initialVelocity
            result.allowsOverdamping = spring.allowsOverdamping
            animation = result
        case .keyframe(let property, let keyframe):
            let result = CAKeyframeAnimation()
            apply(property, to: result)
            if let values = keyframe.values {
                var materialized: [Any] = []
                materialized.reserveCapacity(values.count)
                for value in values {
                    materialized.append(try value.materialize())
                }
                result.values = materialized
            }
            result.path = keyframe.path?.materialize()
            result.keyTimes = keyframe.keyTimes
            result.timingFunctions =
                keyframe.timingFunctions?.map {
                    $0.materialize()
                }
            result.calculationMode = keyframe.calculationMode
            result.tensionValues = keyframe.tensionValues
            result.continuityValues =
                keyframe.continuityValues
            result.biasValues = keyframe.biasValues
            result.rotationMode = keyframe.rotationMode
            animation = result
        case .group(let snapshots):
            let result = CAAnimationGroup()
            if let snapshots {
                var animations: [CAAnimation] = []
                animations.reserveCapacity(snapshots.count)
                for snapshot in snapshots {
                    animations.append(try snapshot.materialize())
                }
                result.animations = consume animations
            }
            animation = result
        case .transition(let transition):
            let result = CATransition()
            result.resourceIdentity =
                transition.resourceIdentity
            result.type = transition.type
            result.subtype = transition.subtype
            result.startProgress = transition.startProgress
            result.endProgress = transition.endProgress
            result.committedFilterSnapshot = transition.filter
            result.committedFilterCaptureFailure =
                transition.filterCaptureFailure
            result.usesCommittedFilterSnapshot = true
            result.committedSourceSnapshot =
                transition.sourceSnapshot
            animation = result
        case .base:
            animation = CAAnimation()
        }
        apply(common, to: animation)
        return animation
    }

    private static func property(
        _ animation: CAPropertyAnimation
    ) -> Property {
        Property(
            keyPath: animation.keyPath,
            isAdditive: animation.isAdditive,
            isCumulative: animation.isCumulative,
            valueFunctionName: animation.valueFunction?.name
        )
    }

    private static func copiedPath(
        _ path: CGPath?
    ) throws(CACommittedAnimationCaptureError) -> CACommittedPath? {
        guard let path else { return nil }
        return try CACommittedPath(capturing: path)
    }

    private static func validateCommonTiming(
        _ animation: CAAnimation
    ) throws(CACommittedAnimationCaptureError) {
        guard animation.beginTime.isFinite else {
            throw .nonFiniteValue("CAAnimation.beginTime")
        }
        guard animation.timeOffset.isFinite else {
            throw .nonFiniteValue("CAAnimation.timeOffset")
        }
        guard animation.speed.isFinite else {
            throw .nonFiniteValue("CAAnimation.speed")
        }
        guard isFiniteOrPositiveInfinity(
            animation.duration
        ) else {
            throw .nonFiniteValue("CAAnimation.duration")
        }
        guard isFiniteOrPositiveInfinity(
            animation.repeatCount
        ) else {
            throw .nonFiniteValue(
                "CAAnimation.repeatCount"
            )
        }
        guard isFiniteOrPositiveInfinity(
            animation.repeatDuration
        ) else {
            throw .nonFiniteValue(
                "CAAnimation.repeatDuration"
            )
        }
        let frameRateRange =
            animation.preferredFrameRateRange
        guard frameRateRange.minimum.isFinite,
              frameRateRange.maximum.isFinite,
              frameRateRange.minimum >= 0,
              frameRateRange.maximum >= 0,
              frameRateRange.preferred?.isFinite
                ?? true else {
            throw .nonFiniteValue(
                "CAAnimation.preferredFrameRateRange"
            )
        }
    }

    private static func isFiniteOrPositiveInfinity<
        Value: BinaryFloatingPoint
    >(_ value: Value) -> Bool {
        value.isFinite && value >= 0
            || value == .infinity
    }

    private static func captureTransitionSource(
        _ transition: CATransition,
        frameToken: UInt64
    ) throws(CACommittedAnimationCaptureError)
        -> CARenderSnapshot?
    {
        guard let source = transition.sourceLayerSnapshot else {
            return transition.committedSourceSnapshot
        }
        do {
            return try CARenderSnapshot.capture(
                source,
                frameToken: frameToken,
                evaluatesAnimations: false
            )
        } catch {
            throw .invalidTransitionSource(error)
        }
    }

    private static func basic(
        _ animation: CABasicAnimation
    ) throws(CACommittedAnimationCaptureError) -> Basic {
        let fromValue: CACommittedAnimationValue?
        if let value = animation.fromValue {
            fromValue = try .capture(value)
        } else {
            fromValue = nil
        }
        let toValue: CACommittedAnimationValue?
        if let value = animation.toValue {
            toValue = try .capture(value)
        } else {
            toValue = nil
        }
        let byValue: CACommittedAnimationValue?
        if let value = animation.byValue {
            byValue = try .capture(value)
        } else {
            byValue = nil
        }
        return Basic(
            fromValue: fromValue,
            toValue: toValue,
            byValue: byValue
        )
    }

    private func apply(
        _ common: Common,
        to animation: CAAnimation
    ) {
        animation.beginTime = common.beginTime
        animation.timeOffset = common.timeOffset
        animation.repeatCount = common.repeatCount
        animation.repeatDuration = common.repeatDuration
        animation.duration = common.duration
        animation.speed = common.speed
        animation.autoreverses = common.autoreverses
        animation.fillMode = common.fillMode
        animation.timingFunction =
            common.timingFunction?.materialize()
        animation.preferredFrameRateRange =
            common.preferredFrameRateRange
        animation.isRemovedOnCompletion =
            common.isRemovedOnCompletion
        animation.isFinished = common.isFinished
        animation.hasStarted = common.hasStarted
        animation.delegate = nil
    }

    private func apply(
        _ property: Property,
        to animation: CAPropertyAnimation
    ) {
        animation.keyPath = property.keyPath
        animation.isAdditive = property.isAdditive
        animation.isCumulative = property.isCumulative
        if let name = property.valueFunctionName {
            animation.valueFunction = CAValueFunction(name: name)
        }
    }

    private func apply(
        _ basic: Basic,
        to animation: CABasicAnimation
    ) throws(CACommittedAnimationCaptureError) {
        animation.fromValue = try basic.fromValue?.materialize()
        animation.toValue = try basic.toValue?.materialize()
        animation.byValue = try basic.byValue?.materialize()
    }
}
