import Foundation

/// Validates untyped browser scheduling values before they enter display-link state.
internal enum CADisplayLinkBrowserValueValidator {
    static func timestamp(
        milliseconds: Double?
    ) -> Result<CFTimeInterval, CADisplayLinkSchedulingFailure> {
        guard let milliseconds else {
            return .failure(.frameTimestampUnavailable)
        }
        guard milliseconds.isFinite else {
            return .failure(.frameTimestampNonFinite)
        }
        guard milliseconds >= 0 else {
            return .failure(.frameTimestampNegative(milliseconds))
        }

        return .success(milliseconds / 1_000)
    }

    static func requestIdentifier(
        _ value: Double?
    ) -> Result<UInt32, CADisplayLinkSchedulingFailure> {
        guard let value else {
            return .failure(.requestIdentifierUnavailable)
        }
        guard value.isFinite else {
            return .failure(.requestIdentifierNonFinite)
        }
        guard value >= 0 else {
            return .failure(.requestIdentifierNegative(value))
        }
        guard value.rounded(.towardZero) == value else {
            return .failure(.requestIdentifierFractional(value))
        }
        guard let identifier = UInt32(exactly: value) else {
            return .failure(.requestIdentifierOutOfRange(value))
        }

        return .success(identifier)
    }
}
