import Foundation

/// Describes why a browser implicit-transaction commit could not be scheduled or cancelled safely.
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
