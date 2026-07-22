import Foundation

internal struct CATransitionCompositeConfiguration: Equatable {
    let size: SIMD2<Float>
    let translatedPosition: CGPoint
    let opacity: Float

    init(
        bounds: CGRect,
        position: CGPoint,
        offset: CGPoint,
        opacity: Float
    ) throws(CATransitionRenderFailure) {
        let width = Float(bounds.width)
        let height = Float(bounds.height)
        let minimumX = Float(bounds.minX)
        let minimumY = Float(bounds.minY)
        guard minimumX.isFinite,
              minimumY.isFinite,
              width.isFinite,
              height.isFinite,
              width > 0,
              height > 0 else {
            throw .invalidCompositeBounds(bounds)
        }

        translatedPosition = CGPoint(
            x: position.x + offset.x,
            y: position.y + offset.y
        )
        let translatedX = Float(translatedPosition.x)
        let translatedY = Float(translatedPosition.y)
        guard offset.x.isFinite,
              offset.y.isFinite,
              translatedX.isFinite,
              translatedY.isFinite else {
            throw .invalidCompositeOffset(offset)
        }
        guard opacity.isFinite, opacity > 0 else {
            throw .invalidCompositeOpacity(opacity)
        }
        self.opacity = opacity
        size = SIMD2(width, height)
    }
}
