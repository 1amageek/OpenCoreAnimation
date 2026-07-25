#if arch(wasm32)
import Foundation

extension CARenderSnapshot.PresentationValues {
    internal func modelMatrix(
        parentMatrix: Matrix4x4 = .identity
    ) -> Matrix4x4 {
        var matrix = parentMatrix
        matrix = matrix * Matrix4x4(translation: position)
        if !CATransform3DIsIdentity(transform) {
            matrix = matrix * transform.matrix4x4
        }
        matrix = matrix * Matrix4x4(translation: anchorOffset)
        return matrix
    }

    internal func sublayerMatrix(modelMatrix: Matrix4x4) -> Matrix4x4 {
        var result = modelMatrix
        if !CATransform3DIsIdentity(sublayerTransform) {
            result = result * sublayerTransform.matrix4x4
        }
        if isGeometryFlipped {
            let flippedBoundsTransform = Matrix4x4(
                translation: SIMD3<Float>(0, boundsSize.y, 0)
            ) * Matrix4x4(columns: (
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, -1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            )) * Matrix4x4(
                translation: SIMD3<Float>(
                    -boundsOrigin.x,
                    -boundsOrigin.y,
                    0
                )
            )
            result = result * flippedBoundsTransform
        } else if boundsOrigin.x != 0 || boundsOrigin.y != 0 {
            result = result * Matrix4x4(
                translation: SIMD3<Float>(
                    -boundsOrigin.x,
                    -boundsOrigin.y,
                    0
                )
            )
        }
        return result
    }
}
#endif
