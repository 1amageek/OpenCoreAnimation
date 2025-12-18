
/// A bitmask that specifies the corners that should be masked.
public struct CACornerMask: OptionSet, Sendable {
    public let rawValue: UInt

    public init(rawValue: UInt) {
        self.rawValue = rawValue
    }

    /// The minimum x, minimum y corner.
    public static let layerMinXMinYCorner = CACornerMask(rawValue: 1 << 0)

    /// The maximum x, minimum y corner.
    public static let layerMaxXMinYCorner = CACornerMask(rawValue: 1 << 1)

    /// The minimum x, maximum y corner.
    public static let layerMinXMaxYCorner = CACornerMask(rawValue: 1 << 2)

    /// The maximum x, maximum y corner.
    public static let layerMaxXMaxYCorner = CACornerMask(rawValue: 1 << 3)
}
