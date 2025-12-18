
/// Constants that specify the scrolling mode.
public struct CAScrollLayerScrollMode: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The receiver is unable to scroll.
    public static let none = CAScrollLayerScrollMode(rawValue: "none")

    /// The receiver can scroll vertically.
    public static let vertically = CAScrollLayerScrollMode(rawValue: "vertically")

    /// The receiver can scroll horizontally.
    public static let horizontally = CAScrollLayerScrollMode(rawValue: "horizontally")

    /// The receiver can scroll in both directions.
    public static let both = CAScrollLayerScrollMode(rawValue: "both")
}
