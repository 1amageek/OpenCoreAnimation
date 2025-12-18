
/// Constants that specify the rotation mode for animation keyframes.
public struct CAAnimationRotationMode: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Objects rotate along the tangent to the path.
    public static let rotateAuto = CAAnimationRotationMode(rawValue: "rotateAuto")

    /// Objects rotate along the tangent to the path, but in reverse.
    public static let rotateAutoReverse = CAAnimationRotationMode(rawValue: "rotateAutoReverse")
}
