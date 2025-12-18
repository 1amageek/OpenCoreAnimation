
/// Constants that specify the predefined timing function names.
public struct CAMediaTimingFunctionName: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The system default timing function.
    /// Use this function to ensure that the timing of your animations matches that of most system animations.
    public static let `default` = CAMediaTimingFunctionName(rawValue: "default")

    /// Ease-in pacing, which causes an animation to begin slowly and then speed up as it progresses.
    public static let easeIn = CAMediaTimingFunctionName(rawValue: "easeIn")

    /// Ease-in-ease-out pacing, which causes an animation to begin slowly, accelerate through
    /// the middle of its duration, and then slow again before completing.
    public static let easeInEaseOut = CAMediaTimingFunctionName(rawValue: "easeInEaseOut")

    /// Ease-out pacing, which causes an animation to begin quickly and then slow as it progresses.
    public static let easeOut = CAMediaTimingFunctionName(rawValue: "easeOut")

    /// Linear pacing, which causes an animation to occur evenly over its duration.
    public static let linear = CAMediaTimingFunctionName(rawValue: "linear")
}
