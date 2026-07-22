import Foundation

enum GradientRenderConfigurationError: Error, Equatable {
    case unsupportedType(String)
    case nonFiniteGeometry
    case invalidColor(index: Int)
    case invalidColorComponents(index: Int)
    case tooManyStops(actual: Int, maximum: Int)
    case invalidLocationCount(expected: Int, actual: Int)
    case nonFiniteLocation(index: Int)
    case locationOutOfRange(index: Int)
    case locationsNotMonotonic(index: Int)
}
