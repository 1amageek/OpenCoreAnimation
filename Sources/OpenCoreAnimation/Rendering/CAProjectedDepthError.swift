import Foundation

/// Describes why a homogeneous projected depth cannot be normalized safely.
public enum CAProjectedDepthError: Error, Equatable, Sendable {
    case nonFiniteHomogeneousCoordinate(z: Float, w: Float)
    case zeroHomogeneousCoordinate
    case nonFiniteNormalizedDepth
}
