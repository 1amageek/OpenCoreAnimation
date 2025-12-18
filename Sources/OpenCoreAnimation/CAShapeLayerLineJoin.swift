
/// Constants that specify the shape of the joints between connected segments of a stroked path.
public struct CAShapeLayerLineJoin: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// A mitered line join style.
    public static let miter = CAShapeLayerLineJoin(rawValue: "miter")

    /// A rounded line join style.
    public static let round = CAShapeLayerLineJoin(rawValue: "round")

    /// A beveled line join style.
    public static let bevel = CAShapeLayerLineJoin(rawValue: "bevel")
}
