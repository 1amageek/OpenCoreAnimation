import Foundation

internal struct CASolidRenderConfiguration: Equatable {
    let size: SIMD2<Float>
    let color: SIMD4<Float>
    let opacity: Float
    let cornerRadius: Float
    let cornerCurveExponent: Float
    let cornerRadii: SIMD4<Float>
    let borderWidth: Float

    init(
        bounds: CGRect,
        color: SIMD4<Float>,
        opacity: Float,
        cornerRadius: CGFloat,
        cornerCurveExponent: Float,
        cornerRadii: SIMD4<Float>,
        borderWidth: CGFloat,
        context: CASolidRenderContext
    ) throws(CASolidRenderFailure) {
        let width = Float(bounds.width)
        let height = Float(bounds.height)
        guard Float(bounds.minX).isFinite,
              Float(bounds.minY).isFinite,
              width.isFinite,
              height.isFinite,
              width > 0,
              height > 0 else {
            throw .invalidGeometry(context, bounds)
        }
        guard color.x.isFinite,
              color.y.isFinite,
              color.z.isFinite,
              color.w.isFinite else {
            throw .invalidColor(context, color)
        }
        guard opacity.isFinite, opacity >= 0 else {
            throw .invalidOpacity(context, opacity)
        }

        let convertedCornerRadius = Float(cornerRadius)
        guard convertedCornerRadius.isFinite,
              convertedCornerRadius >= 0,
              cornerCurveExponent.isFinite,
              cornerCurveExponent > 0,
              cornerRadii.x.isFinite,
              cornerRadii.x >= 0,
              cornerRadii.y.isFinite,
              cornerRadii.y >= 0,
              cornerRadii.z.isFinite,
              cornerRadii.z >= 0,
              cornerRadii.w.isFinite,
              cornerRadii.w >= 0 else {
            throw .invalidCornerGeometry(context)
        }

        let convertedBorderWidth = Float(borderWidth)
        guard convertedBorderWidth.isFinite,
              convertedBorderWidth >= 0 else {
            throw .invalidBorderWidth(convertedBorderWidth)
        }

        size = SIMD2(width, height)
        self.color = color
        self.opacity = opacity
        self.cornerRadius = convertedCornerRadius
        self.cornerCurveExponent = cornerCurveExponent
        self.cornerRadii = cornerRadii
        self.borderWidth = convertedBorderWidth
    }
}
