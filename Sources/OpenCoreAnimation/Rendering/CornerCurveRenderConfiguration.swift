import Foundation

struct CornerCurveRenderConfiguration: Equatable {
    static let circularExponent: CGFloat = 2
    static let continuousExponent: CGFloat = 2.2

    let exponent: CGFloat

    init(
        curve: CALayerCornerCurve
    ) throws(CornerCurveRenderConfigurationError) {
        switch curve {
        case .circular:
            exponent = Self.circularExponent
        case .continuous:
            exponent = Self.continuousExponent
        default:
            throw CornerCurveRenderConfigurationError.unsupportedCurve(curve.rawValue)
        }
    }
}
