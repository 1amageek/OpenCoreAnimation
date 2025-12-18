
/// Constants that specify the calculation mode for animation keyframes.
public struct CAAnimationCalculationMode: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// A linear calculation mode.
    public static let linear = CAAnimationCalculationMode(rawValue: "linear")

    /// A discrete calculation mode.
    public static let discrete = CAAnimationCalculationMode(rawValue: "discrete")

    /// A paced calculation mode.
    public static let paced = CAAnimationCalculationMode(rawValue: "paced")

    /// A cubic calculation mode.
    public static let cubic = CAAnimationCalculationMode(rawValue: "cubic")

    /// A cubic paced calculation mode.
    public static let cubicPaced = CAAnimationCalculationMode(rawValue: "cubicPaced")
}
