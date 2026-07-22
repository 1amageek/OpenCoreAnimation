import Foundation

internal struct CAMaskRenderConfiguration: Equatable {
    let size: SIMD2<Float>
    let cornerRadius: Float
    let cornerCurveExponent: Float
    let cornerRadii: SIMD4<Float>

    init(
        bounds: CGRect,
        cornerRadius: CGFloat,
        cornerCurveExponent: Float,
        cornerRadii: SIMD4<Float>,
        context: CAMaskRenderContext
    ) throws(CAMaskRenderFailure) {
        let width = Float(bounds.width)
        let height = Float(bounds.height)
        let minimumX = Float(bounds.minX)
        let minimumY = Float(bounds.minY)
        self.cornerRadius = Float(cornerRadius)
        self.cornerCurveExponent = cornerCurveExponent
        self.cornerRadii = cornerRadii
        guard minimumX.isFinite,
              minimumY.isFinite,
              width.isFinite,
              height.isFinite,
              width > 0,
              height > 0 else {
            throw .invalidGeometry(context, bounds)
        }
        guard self.cornerRadius.isFinite,
              self.cornerRadius >= 0,
              self.cornerCurveExponent.isFinite,
              self.cornerCurveExponent > 0,
              self.cornerRadii.x.isFinite,
              self.cornerRadii.x >= 0,
              self.cornerRadii.y.isFinite,
              self.cornerRadii.y >= 0,
              self.cornerRadii.z.isFinite,
              self.cornerRadii.z >= 0,
              self.cornerRadii.w.isFinite,
              self.cornerRadii.w >= 0 else {
            throw .invalidCornerGeometry(
                context,
                radius: self.cornerRadius,
                exponent: self.cornerCurveExponent,
                radii: self.cornerRadii
            )
        }
        size = SIMD2(width, height)
    }
}
