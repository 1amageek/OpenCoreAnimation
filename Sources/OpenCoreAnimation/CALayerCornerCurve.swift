
/// Constants that specify the curve used when drawing a rounded corner.
public struct CALayerCornerCurve: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// A circular corner curve.
    public static let circular = CALayerCornerCurve(rawValue: "circular")

    /// A continuous corner curve.
    public static let continuous = CALayerCornerCurve(rawValue: "continuous")
}
