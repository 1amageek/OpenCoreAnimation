import Foundation

internal enum EmitterCellSimulationError: Error, Equatable {
    case nonFiniteTiming
    case invalidTimeInterval
    case nonFiniteDirection
    case zeroParentDirection
}

internal enum EmitterCellSimulation {
    /// Returns the amount of active local cell time covered by a parent-time interval.
    static func activeEmissionDelta(
        for cell: CAEmitterCell,
        from parentStartTime: CFTimeInterval,
        to parentEndTime: CFTimeInterval
    ) throws -> Float {
        try activeEmissionDelta(
            beginTime: cell.beginTime,
            timeOffset: cell.timeOffset,
            duration: cell.duration,
            repeatCount: cell.repeatCount,
            repeatDuration: cell.repeatDuration,
            speed: cell.speed,
            autoreverses: cell.autoreverses,
            from: parentStartTime,
            to: parentEndTime
        )
    }

    static func activeEmissionDelta(
        for cell: CAEmitterCellSnapshot,
        from parentStartTime: CFTimeInterval,
        to parentEndTime: CFTimeInterval
    ) throws -> Float {
        try activeEmissionDelta(
            beginTime: cell.beginTime,
            timeOffset: cell.timeOffset,
            duration: cell.duration,
            repeatCount: cell.repeatCount,
            repeatDuration: cell.repeatDuration,
            speed: cell.speed,
            autoreverses: cell.autoreverses,
            from: parentStartTime,
            to: parentEndTime
        )
    }

    private static func activeEmissionDelta(
        beginTime: CFTimeInterval,
        timeOffset: CFTimeInterval,
        duration: CFTimeInterval,
        repeatCount: Float,
        repeatDuration: CFTimeInterval,
        speed: Float,
        autoreverses: Bool,
        from parentStartTime: CFTimeInterval,
        to parentEndTime: CFTimeInterval
    ) throws -> Float {
        guard parentStartTime.isFinite,
              parentEndTime.isFinite,
              parentStartTime <= parentEndTime else {
            throw EmitterCellSimulationError.invalidTimeInterval
        }
        guard beginTime.isFinite,
              timeOffset.isFinite,
              isFiniteOrPositiveInfinity(duration),
              isFiniteOrPositiveInfinity(repeatDuration),
              isFiniteOrPositiveInfinity(repeatCount),
              speed.isFinite else {
            throw EmitterCellSimulationError.nonFiniteTiming
        }
        guard speed != 0, parentEndTime > beginTime else { return 0 }

        let clippedStart = max(parentStartTime, beginTime)
        let clippedEnd = max(clippedStart, parentEndTime)
        let localSpeed = CFTimeInterval(speed)
        let localStart = (clippedStart - beginTime) * localSpeed + timeOffset
        let localEnd = (clippedEnd - beginTime) * localSpeed + timeOffset
        guard localStart.isFinite, localEnd.isFinite else {
            throw EmitterCellSimulationError.nonFiniteTiming
        }

        let activeUpperBound: CFTimeInterval
        if duration == .infinity {
            activeUpperBound = .infinity
        } else if duration > 0 {
            activeUpperBound = CAMediaTimingEvaluator.activeDuration(
                duration: duration,
                repeatCount: repeatCount,
                repeatDuration: repeatDuration,
                autoreverses: autoreverses
            )
            guard isFiniteOrPositiveInfinity(activeUpperBound) else {
                throw EmitterCellSimulationError.nonFiniteTiming
            }
        } else {
            activeUpperBound = .infinity
        }

        let segmentMinimum = min(localStart, localEnd)
        let segmentMaximum = max(localStart, localEnd)
        let overlapStart = max(0, segmentMinimum)
        let overlapEnd = min(activeUpperBound, segmentMaximum)
        let overlap = max(0, overlapEnd - overlapStart)
        guard overlap.isFinite, overlap <= CFTimeInterval(Float.greatestFiniteMagnitude) else {
            throw EmitterCellSimulationError.nonFiniteTiming
        }
        return Float(overlap)
    }

    private static func isFiniteOrPositiveInfinity<T: BinaryFloatingPoint>(_ value: T) -> Bool {
        value.isFinite || value == .infinity
    }

    /// Rotates a child cell's local emission direction into its parent's direction frame.
    static func childDirection(
        localDirection: SIMD3<Float>,
        parentDirection: SIMD3<Float>
    ) throws -> SIMD3<Float> {
        guard componentsAreFinite(localDirection), componentsAreFinite(parentDirection) else {
            throw EmitterCellSimulationError.nonFiniteDirection
        }
        let parentLength = length(parentDirection)
        guard parentLength > 0 else {
            throw EmitterCellSimulationError.zeroParentDirection
        }
        let forward = parentDirection / parentLength
        let reference = abs(forward.z) < 0.999
            ? SIMD3<Float>(0, 0, 1)
            : SIMD3<Float>(0, 1, 0)
        let tangent = normalized(cross(reference, forward))
        let bitangent = cross(forward, tangent)
        let transformed = tangent * localDirection.x
            + bitangent * localDirection.y
            + forward * localDirection.z
        return normalized(transformed)
    }

    private static func componentsAreFinite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    private static func length(_ value: SIMD3<Float>) -> Float {
        sqrt(value.x * value.x + value.y * value.y + value.z * value.z)
    }

    private static func normalized(_ value: SIMD3<Float>) -> SIMD3<Float> {
        let valueLength = length(value)
        guard valueLength > 0 else { return .zero }
        return value / valueLength
    }

    private static func cross(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x
        )
    }
}
