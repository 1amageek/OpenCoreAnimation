
/// Constants that specify the shape of endpoints for an open path when stroked.
public struct CAShapeLayerLineCap: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The line end extends to the exact endpoint of the path.
    public static let butt = CAShapeLayerLineCap(rawValue: "butt")

    /// A half-circle extension, with diameter equal to line width.
    public static let round = CAShapeLayerLineCap(rawValue: "round")

    /// A half-square extension, with width equal to line width.
    public static let square = CAShapeLayerLineCap(rawValue: "square")
}
