import Foundation

/// Describes why an implicit-transaction commit could not be scheduled or delivered safely.
@_spi(RendererDiagnostics)
public enum CATransactionSchedulingFailure: Error, Equatable, Sendable {
    case setTimeoutUnavailable
    case clearTimeoutUnavailable(identifier: Double)
    case timerIdentifierUnavailable
    case timerIdentifierNonFinite
    case timerIdentifierNonPositive(Double)
    case timerIdentifierFractional(Double)
    case timerIdentifierUnsafeInteger(Double)
}
