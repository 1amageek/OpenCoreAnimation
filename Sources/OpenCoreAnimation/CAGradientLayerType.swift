
/// Constants that specify the type of gradient.
public struct CAGradientLayerType: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// A linear gradient that varies along the axis defined by the start and end points.
    public static let axial = CAGradientLayerType(rawValue: "axial")

    /// A conic gradient that varies around a center point.
    public static let conic = CAGradientLayerType(rawValue: "conic")

    /// A radial gradient that varies outward from a center point.
    public static let radial = CAGradientLayerType(rawValue: "radial")
}
