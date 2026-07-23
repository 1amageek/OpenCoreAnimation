import Foundation

internal enum CAKeyframeTimingConfiguration {
    internal static func supports(_ mode: CAAnimationCalculationMode) -> Bool {
        mode == .linear
            || mode == .discrete
            || mode == .paced
            || mode == .cubic
            || mode == .cubicPaced
    }

    /// Invalid explicit key times are ignored, matching the documented
    /// CAKeyframeAnimation contract, and replaced by evenly spaced times.
    internal static func effectiveKeyTimes(
        _ explicitKeyTimes: [CGFloat]?,
        expectedCount: Int
    ) -> [CGFloat] {
        guard expectedCount > 1 else { return [0] }
        if let explicitKeyTimes,
           explicitKeyTimes.count == expectedCount,
           explicitKeyTimes.first == 0,
           explicitKeyTimes.last == 1,
           explicitKeyTimes.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
           zip(explicitKeyTimes, explicitKeyTimes.dropFirst()).allSatisfy({ $0 <= $1 }) {
            return explicitKeyTimes
        }
        let divisor = CGFloat(expectedCount - 1)
        return (0..<expectedCount).map { CGFloat($0) / divisor }
    }
}
