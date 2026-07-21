import Foundation

/// Maps a CAMediaTiming object from its parent time space into basic local time.
///
/// Core Animation timing is hierarchical: a layer maps global time into the
/// layer's local time, then each animation maps that parent time into active
/// time and finally into its repeating basic timeline. Keeping that mapping in
/// one place prevents rendering and completion checks from disagreeing.
internal enum CAMediaTimingEvaluator {
    internal enum Phase: Equatable {
        case before
        case active
        case after
    }

    internal struct Result {
        let phase: Phase
        let basicTime: CFTimeInterval
        let progress: CFTimeInterval
        let completedCycles: Int

        func applies(fillMode: CAMediaTimingFillMode) -> Bool {
            switch phase {
            case .active:
                return true
            case .before:
                return fillMode == .backwards || fillMode == .both
            case .after:
                return fillMode == .forwards || fillMode == .both
            }
        }
    }

    internal static func activeDuration(
        duration: CFTimeInterval,
        repeatCount: Float,
        repeatDuration: CFTimeInterval,
        autoreverses: Bool
    ) -> CFTimeInterval {
        let basicDuration = max(0, duration)
        guard basicDuration > 0 else { return 0 }

        // repeatDuration is already an active-local-time duration. In
        // particular, autoreversing does not double it.
        if repeatDuration > 0 {
            return repeatDuration
        }

        let cycleDuration = basicDuration * (autoreverses ? 2 : 1)
        if repeatCount > 0 {
            return cycleDuration * CFTimeInterval(repeatCount)
        }
        return cycleDuration
    }

    internal static func evaluate(
        _ animation: CAAnimation,
        parentTime: CFTimeInterval,
        duration: CFTimeInterval
    ) -> Result {
        let activeDuration = activeDuration(
            duration: duration,
            repeatCount: animation.repeatCount,
            repeatDuration: animation.repeatDuration,
            autoreverses: animation.autoreverses
        )

        guard duration > 0, activeDuration > 0 else {
            let phase: Phase = parentTime < animation.beginTime ? .before : .after
            return Result(phase: phase, basicTime: duration, progress: 1, completedCycles: 0)
        }

        let speed = CFTimeInterval(animation.speed)
        let activeTime = (parentTime - animation.beginTime) * speed + animation.timeOffset
        let phase: Phase
        let sampledActiveTime: CFTimeInterval

        if parentTime < animation.beginTime {
            phase = .before
            sampledActiveTime = speed < 0 ? activeDuration : 0
        } else if speed == 0 {
            phase = .active
            sampledActiveTime = min(activeDuration, max(0, animation.timeOffset))
        } else if speed > 0 {
            if activeTime < 0 {
                phase = .before
                sampledActiveTime = 0
            } else if activeTime >= activeDuration {
                phase = .after
                sampledActiveTime = activeDuration
            } else {
                phase = .active
                sampledActiveTime = activeTime
            }
        } else {
            if activeTime > activeDuration {
                phase = .before
                sampledActiveTime = activeDuration
            } else if activeTime <= 0 {
                phase = .after
                sampledActiveTime = 0
            } else {
                phase = .active
                sampledActiveTime = activeTime
            }
        }

        let isPausedAtTerminalOffset = speed == 0
            && abs(sampledActiveTime - activeDuration) < 0.000_000_001
        let isTerminalSample = (phase != .active || isPausedAtTerminalOffset)
            && abs(sampledActiveTime - activeDuration) < 0.000_000_001
        let basicTime = mapActiveTimeToBasicTime(
            sampledActiveTime,
            duration: duration,
            autoreverses: animation.autoreverses,
            isTerminalSample: isTerminalSample
        )
        let cycleDuration = duration * (animation.autoreverses ? 2 : 1)
        let completedCycles = completedCycleCount(
            sampledActiveTime,
            cycleDuration: cycleDuration,
            isTerminalSample: isTerminalSample
        )
        return Result(
            phase: phase,
            basicTime: basicTime,
            progress: min(1, max(0, basicTime / duration)),
            completedCycles: completedCycles
        )
    }

    private static func mapActiveTimeToBasicTime(
        _ activeTime: CFTimeInterval,
        duration: CFTimeInterval,
        autoreverses: Bool,
        isTerminalSample: Bool
    ) -> CFTimeInterval {
        guard activeTime > 0 else { return 0 }
        let cycleDuration = duration * (autoreverses ? 2 : 1)
        guard cycleDuration.isFinite else { return 0 }

        var cycleTime = activeTime.truncatingRemainder(dividingBy: cycleDuration)
        let isCycleBoundary = abs(cycleTime) < 0.000_000_001
        if isCycleBoundary {
            // An interior repeat boundary is the start of the next cycle.
            // Only the terminal sample represents the end of the last cycle.
            if isTerminalSample {
                return autoreverses ? 0 : duration
            }
            return 0
        }

        if autoreverses, cycleTime > duration {
            cycleTime = cycleDuration - cycleTime
        }
        return min(duration, max(0, cycleTime))
    }

    private static func completedCycleCount(
        _ activeTime: CFTimeInterval,
        cycleDuration: CFTimeInterval,
        isTerminalSample: Bool
    ) -> Int {
        guard activeTime > 0, cycleDuration > 0, cycleDuration.isFinite else { return 0 }
        let quotient = activeTime / cycleDuration
        let rounded = quotient.rounded()
        let isBoundary = abs(quotient - rounded) < 0.000_000_001
        if isTerminalSample, isBoundary {
            return max(0, Int(rounded) - 1)
        }
        return max(0, Int(floor(quotient)))
    }
}
