import Foundation

/// Validated, renderer-independent shadow input.
internal struct CAShadowRenderConfiguration {
    let color: SIMD4<Float>
    let opacity: Float
    let radius: Float
    let offset: CGSize

    init(layer: CALayer) throws(CAShadowRenderFailure) {
        let convertedRadius = Float(layer.shadowRadius)
        let convertedOffsetX = Float(layer.shadowOffset.width)
        let convertedOffsetY = Float(layer.shadowOffset.height)
        let convertedBoundsX = Float(layer.bounds.origin.x)
        let convertedBoundsY = Float(layer.bounds.origin.y)
        let convertedBoundsWidth = Float(layer.bounds.width)
        let convertedBoundsHeight = Float(layer.bounds.height)
        guard layer.shadowOpacity.isFinite,
              layer.shadowRadius.isFinite,
              layer.shadowOffset.width.isFinite,
              layer.shadowOffset.height.isFinite,
              layer.bounds.origin.x.isFinite,
              layer.bounds.origin.y.isFinite,
              layer.bounds.width.isFinite,
              layer.bounds.height.isFinite,
              convertedRadius.isFinite,
              convertedOffsetX.isFinite,
              convertedOffsetY.isFinite,
              convertedBoundsX.isFinite,
              convertedBoundsY.isFinite,
              convertedBoundsWidth.isFinite,
              convertedBoundsHeight.isFinite else {
            throw .nonFiniteGeometry
        }
        guard let sourceColor = layer.shadowColor,
              let converted = sourceColor.converted(
                to: .deviceRGB,
                intent: .defaultIntent,
                options: nil
              ),
              let components = converted.components,
              components.count == 4,
              components.allSatisfy(\.isFinite) else {
            throw .invalidColor
        }
        let convertedColor = SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
        guard convertedColor.x.isFinite,
              convertedColor.y.isFinite,
              convertedColor.z.isFinite,
              convertedColor.w.isFinite else {
            throw .invalidColor
        }

        color = convertedColor
        opacity = layer.shadowOpacity
        radius = max(0, convertedRadius)
        offset = layer.shadowOffset
    }
}
