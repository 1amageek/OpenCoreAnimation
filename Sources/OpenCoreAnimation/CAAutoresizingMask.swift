
/// These constants are used by the `autoresizingMask` property.
public struct CAAutoresizingMask: OptionSet, Sendable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    /// The left margin between the receiver and its superview is flexible.
    public static let layerMinXMargin = CAAutoresizingMask(rawValue: 1 << 0)

    /// The receiver's width is flexible.
    public static let layerWidthSizable = CAAutoresizingMask(rawValue: 1 << 1)

    /// The right margin between the receiver and its superview is flexible.
    public static let layerMaxXMargin = CAAutoresizingMask(rawValue: 1 << 2)

    /// The bottom margin between the receiver and its superview is flexible.
    public static let layerMinYMargin = CAAutoresizingMask(rawValue: 1 << 3)

    /// The receiver's height is flexible.
    public static let layerHeightSizable = CAAutoresizingMask(rawValue: 1 << 4)

    /// The top margin between the receiver and its superview is flexible.
    public static let layerMaxYMargin = CAAutoresizingMask(rawValue: 1 << 5)
}
