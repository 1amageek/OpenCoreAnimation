import Foundation

internal struct CAShadowCompositeConfiguration: Equatable {
    let color: SIMD4<Float>
    let offset: SIMD2<Float>
    let viewportSize: SIMD2<Float>

    init(
        shadow: CAShadowRenderConfiguration,
        effectiveOpacity: Float,
        replicatorColor: SIMD4<Float>,
        viewportSize: CGSize
    ) throws(CAShadowRenderFailure) {
        guard effectiveOpacity.isFinite, effectiveOpacity >= 0 else {
            throw .invalidCompositeOpacity(effectiveOpacity)
        }
        guard shadow.opacity >= 0 else {
            throw .invalidCompositeOpacity(shadow.opacity)
        }
        guard replicatorColor.x.isFinite,
              replicatorColor.y.isFinite,
              replicatorColor.z.isFinite,
              replicatorColor.w.isFinite else {
            throw .invalidReplicatorColor(replicatorColor)
        }

        let viewportWidth = Float(viewportSize.width)
        let viewportHeight = Float(viewportSize.height)
        let offsetX = Float(shadow.offset.width)
        let offsetY = Float(shadow.offset.height)
        guard viewportWidth.isFinite,
              viewportHeight.isFinite,
              viewportWidth > 0,
              viewportHeight > 0,
              offsetX.isFinite,
              offsetY.isFinite else {
            throw .invalidCompositeViewport
        }

        let sourceColor = SIMD4<Float>(
            shadow.color.x,
            shadow.color.y,
            shadow.color.z,
            shadow.color.w * shadow.opacity * effectiveOpacity
        )
        let compositeColor = sourceColor * replicatorColor
        guard compositeColor.x.isFinite,
              compositeColor.y.isFinite,
              compositeColor.z.isFinite,
              compositeColor.w.isFinite else {
            throw .compositeColorOverflow
        }

        color = compositeColor
        offset = SIMD2(offsetX, offsetY)
        self.viewportSize = SIMD2(viewportWidth, viewportHeight)
    }
}
