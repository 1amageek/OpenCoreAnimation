import Foundation

internal struct CATransitionCaptureConfiguration: Equatable {
    let pixelWidth: Int
    let pixelHeight: Int
    let projectionLeft: Float
    let projectionRight: Float
    let projectionBottom: Float
    let projectionTop: Float

    init(
        bounds: CGRect,
        contentsScale: CGFloat,
        pixelSizeOverride: CGSize?,
        maximumTextureDimension: Int,
        role: CATransitionParticipantRole
    ) throws(CATransitionRenderFailure) {
        guard bounds.minX.isFinite,
              bounds.minY.isFinite,
              bounds.width.isFinite,
              bounds.height.isFinite,
              bounds.width > 0,
              bounds.height > 0 else {
            throw .invalidParticipantBounds(role, bounds)
        }
        guard contentsScale.isFinite, contentsScale > 0 else {
            throw .invalidParticipantContentsScale(role, contentsScale)
        }

        projectionLeft = Float(bounds.minX)
        projectionRight = Float(bounds.maxX)
        projectionBottom = Float(bounds.minY)
        projectionTop = Float(bounds.maxY)
        guard projectionLeft.isFinite,
              projectionRight.isFinite,
              projectionBottom.isFinite,
              projectionTop.isFinite,
              projectionLeft < projectionRight,
              projectionBottom < projectionTop else {
            throw .participantProjectionOutOfRange(role, bounds)
        }

        let maximumDimension = CGFloat(max(1, maximumTextureDimension))
        let requestedSize: CGSize
        if let pixelSizeOverride {
            requestedSize = pixelSizeOverride
        } else {
            let scale = max(contentsScale, 1)
            requestedSize = CGSize(
                width: bounds.width * scale,
                height: bounds.height * scale
            )
        }
        guard requestedSize.width.isFinite,
              requestedSize.height.isFinite,
              requestedSize.width > 0,
              requestedSize.height > 0 else {
            throw .invalidParticipantPixelSize(role, requestedSize)
        }

        let fittingScale = min(
            1,
            maximumDimension / max(requestedSize.width, requestedSize.height)
        )
        pixelWidth = max(1, Int((requestedSize.width * fittingScale).rounded(.up)))
        pixelHeight = max(1, Int((requestedSize.height * fittingScale).rounded(.up)))
    }
}
