import Foundation

/// Validates browser timer handles without narrowing them to the target's native integer width.
internal enum CATransactionBrowserTimerIdentifierValidator {
    internal static let maximumSafeInteger = 9_007_199_254_740_991.0

    internal static func identifier(
        _ value: Double?
    ) -> Result<Double, CATransactionSchedulingFailure> {
        guard let value else {
            return .failure(.timerIdentifierUnavailable)
        }
        guard value.isFinite else {
            return .failure(.timerIdentifierNonFinite)
        }
        guard value > 0 else {
            return .failure(.timerIdentifierNonPositive(value))
        }
        guard value.rounded(.towardZero) == value else {
            return .failure(.timerIdentifierFractional(value))
        }
        guard value <= maximumSafeInteger else {
            return .failure(.timerIdentifierUnsafeInteger(value))
        }

        return .success(value)
    }
}
