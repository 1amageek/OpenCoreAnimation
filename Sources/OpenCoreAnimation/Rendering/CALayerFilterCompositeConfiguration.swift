internal struct CALayerFilterCompositeConfiguration: Equatable {
    let opacity: Float
    let colorMultiplier: SIMD4<Float>

    init(
        opacity: Float,
        colorMultiplier: SIMD4<Float>
    ) throws(CALayerFilterRenderFailure) {
        guard opacity.isFinite, opacity >= 0 else {
            throw .invalidCompositeOpacity(opacity)
        }
        guard colorMultiplier.x.isFinite,
              colorMultiplier.y.isFinite,
              colorMultiplier.z.isFinite,
              colorMultiplier.w.isFinite else {
            throw .invalidCompositeColor(colorMultiplier)
        }
        self.opacity = opacity
        self.colorMultiplier = colorMultiplier
    }
}
