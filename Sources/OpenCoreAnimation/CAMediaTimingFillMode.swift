
/// Constants that define how the timed object behaves once its active duration has completed.
public struct CAMediaTimingFillMode: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The receiver clamps values before zero to zero when the animation is completed.
    public static let backwards = CAMediaTimingFillMode(rawValue: "backwards")

    /// The receiver clamps values at both ends of the object's time space.
    public static let both = CAMediaTimingFillMode(rawValue: "both")

    /// The receiver remains visible in its final state when the animation is completed.
    public static let forwards = CAMediaTimingFillMode(rawValue: "forwards")

    /// The receiver is removed from the presentation when the animation is completed.
    public static let removed = CAMediaTimingFillMode(rawValue: "removed")
}
