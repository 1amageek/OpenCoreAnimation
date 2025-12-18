
/// Constants that specify the fill rule used when filling the shape's path.
public struct CAShapeLayerFillRule: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The non-zero winding number rule.
    public static let nonZero = CAShapeLayerFillRule(rawValue: "non-zero")

    /// The even-odd rule.
    public static let evenOdd = CAShapeLayerFillRule(rawValue: "even-odd")
}
