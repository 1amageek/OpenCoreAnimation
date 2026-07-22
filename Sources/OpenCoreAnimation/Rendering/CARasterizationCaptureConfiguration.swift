import Foundation

internal struct CARasterizationCaptureConfiguration: Equatable {
    let pixelWidth: Int
    let pixelHeight: Int
    let projectionLeft: Float
    let projectionRight: Float
    let projectionBottom: Float
    let projectionTop: Float

    init(
        captureBounds: CGRect,
        rasterizationScale: CGFloat,
        maximumTextureDimension: Int
    ) throws(CARasterizationRenderFailure) {
        guard rasterizationScale.isFinite, rasterizationScale > 0 else {
            throw .invalidRasterizationScale(rasterizationScale)
        }
        let scaledSize = CGSize(
            width: captureBounds.width * rasterizationScale,
            height: captureBounds.height * rasterizationScale
        )
        guard scaledSize.width.isFinite,
              scaledSize.height.isFinite,
              scaledSize.width > 0,
              scaledSize.height > 0 else {
            throw .invalidScaledExtent(scaledSize)
        }

        projectionLeft = Float(captureBounds.minX)
        projectionRight = Float(captureBounds.maxX)
        projectionBottom = Float(captureBounds.minY)
        projectionTop = Float(captureBounds.maxY)
        guard projectionLeft.isFinite,
              projectionRight.isFinite,
              projectionBottom.isFinite,
              projectionTop.isFinite,
              projectionLeft < projectionRight,
              projectionBottom < projectionTop else {
            throw .captureProjectionOutOfRange(captureBounds)
        }

        let maximumDimension = CGFloat(max(1, maximumTextureDimension))
        let fittingScale = min(
            1,
            maximumDimension / max(scaledSize.width, scaledSize.height)
        )
        pixelWidth = max(1, Int((scaledSize.width * fittingScale).rounded(.up)))
        pixelHeight = max(1, Int((scaledSize.height * fittingScale).rounded(.up)))
    }
}
