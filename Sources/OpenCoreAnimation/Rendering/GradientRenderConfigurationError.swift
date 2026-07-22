import Foundation

@_spi(RendererDiagnostics)
public enum GradientRenderConfigurationError: Error, Equatable, Sendable {
    case unsupportedType(String)
    case nonFiniteGeometry
    case invalidColor(index: Int)
    case invalidColorComponents(index: Int)
    case invalidLocationCount(expected: Int, actual: Int)
    case nonFiniteLocation(index: Int)
    case locationOutOfRange(index: Int)
    case locationsNotMonotonic(index: Int)
}
