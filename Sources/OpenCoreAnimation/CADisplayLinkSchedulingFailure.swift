import Foundation

/// Describes why browser display-link scheduling could not continue safely.
@_spi(RendererDiagnostics)
public enum CADisplayLinkSchedulingFailure: Error, Equatable, Sendable {
    case requestAnimationFrameUnavailable
    case cancelAnimationFrameUnavailable(identifier: UInt32)
    case frameTimestampUnavailable
    case frameTimestampNonFinite
    case frameTimestampNegative(Double)
    case requestIdentifierUnavailable
    case requestIdentifierNonFinite
    case requestIdentifierNegative(Double)
    case requestIdentifierFractional(Double)
    case requestIdentifierOutOfRange(Double)
}
