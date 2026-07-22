import Foundation

/// Describes why a render-target size cannot be represented safely.
public enum CARenderTargetConfigurationError: Error, Equatable, Sendable {
    case invalidDimensions(width: Double, height: Double)
    case dimensionLimitExceeded(width: Int, height: Int, maximum: Int)
}
