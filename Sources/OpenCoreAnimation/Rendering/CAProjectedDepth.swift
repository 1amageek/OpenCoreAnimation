import Foundation

/// Resolves a finite normalized depth from homogeneous clip coordinates.
internal enum CAProjectedDepth {
    static func resolve(z: Float, w: Float) throws(CAProjectedDepthError) -> Float {
        guard z.isFinite, w.isFinite else {
            throw .nonFiniteHomogeneousCoordinate(z: z, w: w)
        }
        guard w != 0 else {
            throw .zeroHomogeneousCoordinate
        }
        let depth = z / w
        guard depth.isFinite else {
            throw .nonFiniteNormalizedDepth
        }
        return depth
    }
}
