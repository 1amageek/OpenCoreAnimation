
/// Constants that specify value function names for transform animations.
public struct CAValueFunctionName: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// A value function that rotates by the input value, in radians, around the x-axis.
    public static let rotateX = CAValueFunctionName(rawValue: "rotateX")

    /// A value function that rotates by the input value, in radians, around the y-axis.
    public static let rotateY = CAValueFunctionName(rawValue: "rotateY")

    /// A value function that rotates by the input value, in radians, around the z-axis.
    public static let rotateZ = CAValueFunctionName(rawValue: "rotateZ")

    /// A value function scales by the input value along all three axis.
    public static let scale = CAValueFunctionName(rawValue: "scale")

    /// A value function scales by the input value along the x-axis.
    public static let scaleX = CAValueFunctionName(rawValue: "scaleX")

    /// A value function scales by the input value along the y-axis.
    public static let scaleY = CAValueFunctionName(rawValue: "scaleY")

    /// A value function that scales by the input value along the z-axis.
    public static let scaleZ = CAValueFunctionName(rawValue: "scaleZ")

    /// A value function translates by the input value along the x-axis.
    public static let translateX = CAValueFunctionName(rawValue: "translateX")

    /// A value function translates by the input value along the y-axis.
    public static let translateY = CAValueFunctionName(rawValue: "translateY")

    /// A value function translates by the input value along the z-axis.
    public static let translateZ = CAValueFunctionName(rawValue: "translateZ")
}
