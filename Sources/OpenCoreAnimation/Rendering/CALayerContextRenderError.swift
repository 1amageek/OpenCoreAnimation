import Foundation

/// Describes why `CALayer.render(in:)` rejected specialized layer content.
@_spi(RendererDiagnostics)
public enum CALayerContextRenderError: Error, Equatable, Sendable {
    case unsupportedCornerCurve(String)
    case nonFiniteShapePath
    case unsupportedShapeFillRule(String)
    case invalidShapeStrokeGeometry
    case invalidShapeDashPattern
    case unsupportedShapeLineCap(String)
    case unsupportedShapeLineJoin(String)
    case unsupportedGradientType(String)
    case nonFiniteGradientGeometry
    case invalidGradientColor(index: Int)
    case invalidGradientColorComponents(index: Int)
    case invalidGradientLocationCount(expected: Int, actual: Int)
    case nonFiniteGradientLocation(index: Int)
    case gradientLocationOutOfRange(index: Int)
    case gradientLocationsNotMonotonic(index: Int)
    case gradientCreationFailed
    case degenerateRadialGradient
    case gradientInterpolationFailed
}
