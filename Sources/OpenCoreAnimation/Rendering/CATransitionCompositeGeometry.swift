import Foundation

/// Value-owned geometry used to composite a frozen transition participant.
internal struct CATransitionCompositeGeometry: Equatable, Sendable {
    internal let bounds: CGRect
    internal let contentsScale: CGFloat
    internal let position: SIMD3<Float>
    internal let anchorOffset: SIMD3<Float>
    internal let transform: CATransform3D
    internal let replicatorInstanceTransform: CATransform3D
    internal let opacity: Float

    internal init(presentationLayer: CALayer) {
        let width = Float(presentationLayer.bounds.width)
        let height = Float(presentationLayer.bounds.height)
        bounds = presentationLayer.bounds
        contentsScale = presentationLayer.contentsScale
        position = SIMD3(
            Float(presentationLayer.position.x),
            Float(presentationLayer.position.y),
            Float(presentationLayer.zPosition)
        )
        anchorOffset = SIMD3(
            -(width * Float(presentationLayer.anchorPoint.x)),
            -(height * Float(presentationLayer.anchorPoint.y)),
            Float(-presentationLayer.anchorPointZ)
        )
        transform = presentationLayer.transform
        replicatorInstanceTransform = CATransform3DIdentity
        opacity = presentationLayer.opacity
    }

    internal init(
        values: CARenderSnapshot.PresentationValues
    ) {
        bounds = values.bounds
        contentsScale = values.contentsScale
        position = values.position
        anchorOffset = values.anchorOffset
        transform = values.transform
        replicatorInstanceTransform =
            values.replicatorInstanceTransform
        opacity = values.opacity
    }

    #if arch(wasm32)
    internal func modelMatrix(
        parentMatrix: Matrix4x4,
        translatedPosition: CGPoint
    ) -> Matrix4x4 {
        var matrix = parentMatrix
        if !CATransform3DIsIdentity(
            replicatorInstanceTransform
        ) {
            matrix =
                matrix * replicatorInstanceTransform.matrix4x4
        }
        matrix = matrix * Matrix4x4(
            translation: SIMD3(
                Float(translatedPosition.x),
                Float(translatedPosition.y),
                position.z
            )
        )
        if !CATransform3DIsIdentity(transform) {
            matrix = matrix * transform.matrix4x4
        }
        return matrix * Matrix4x4(translation: anchorOffset)
    }
    #endif
}
