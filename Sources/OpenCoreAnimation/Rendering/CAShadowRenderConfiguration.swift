import Foundation

/// Describes why a visible layer shadow could not be rendered.
@_spi(RendererDiagnostics)
public enum CAShadowRenderFailure: Error, Equatable, Sendable {
    case nonFiniteGeometry
    case invalidColor
    case rendererResourcesUnavailable
    case shadowPathTessellationFailed
    case rasterizedShadowResourcesUnavailable
    case vertexCapacityExceeded
    case prerenderedShadowUnavailable
}

/// Validated, renderer-independent shadow input.
internal struct CAShadowRenderConfiguration {
    let color: SIMD4<Float>
    let opacity: Float
    let radius: Float
    let offset: CGSize

    init(layer: CALayer) throws(CAShadowRenderFailure) {
        guard layer.shadowOpacity.isFinite,
              layer.shadowRadius.isFinite,
              layer.shadowOffset.width.isFinite,
              layer.shadowOffset.height.isFinite,
              layer.bounds.origin.x.isFinite,
              layer.bounds.origin.y.isFinite,
              layer.bounds.width.isFinite,
              layer.bounds.height.isFinite else {
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
        radius = max(0, Float(layer.shadowRadius))
        offset = layer.shadowOffset
    }
}
