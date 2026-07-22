import Foundation

internal struct CATextQuadRenderConfiguration: Equatable {
    let size: SIMD2<Float>
    let color: SIMD4<Float>
    let opacity: Float
    let cornerRadius: Float
    let cornerCurveExponent: Float
    let cornerRadii: SIMD4<Float>

    init(
        bounds: CGRect,
        color: SIMD4<Float>,
        opacity: Float,
        masksToBounds: Bool,
        cornerRadius: CGFloat,
        cornerCurveExponent: Float,
        cornerRadii: SIMD4<Float>
    ) throws(CATextRenderFailure) {
        let width = Float(bounds.width)
        let height = Float(bounds.height)
        guard width.isFinite,
              height.isFinite,
              width > 0,
              height > 0 else {
            throw .invalidBounds
        }
        guard color.x.isFinite,
              color.y.isFinite,
              color.z.isFinite,
              color.w.isFinite else {
            throw .invalidReplicatorColor(color)
        }
        guard opacity.isFinite, opacity >= 0 else {
            throw .invalidOpacity(opacity)
        }

        size = SIMD2(width, height)
        self.color = color
        self.opacity = opacity
        if masksToBounds {
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
                throw .invalidCornerGeometry
            }
            self.cornerRadius = convertedCornerRadius
            self.cornerCurveExponent = cornerCurveExponent
            self.cornerRadii = cornerRadii
        } else {
            self.cornerRadius = 0
            self.cornerCurveExponent = Float(CornerCurveRenderConfiguration.circularExponent)
            self.cornerRadii = .zero
        }
    }
}
