
/// Constants that specify the transition type.
public struct CATransitionType: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The layer's content fades as it becomes visible or hidden.
    public static let fade = CATransitionType(rawValue: "fade")

    /// The layer's content slides into place over any existing content.
    public static let moveIn = CATransitionType(rawValue: "moveIn")

    /// The layer's content pushes any existing content as it slides into place.
    public static let push = CATransitionType(rawValue: "push")

    /// The layer's content is revealed gradually in the direction indicated by the transition subtype.
    public static let reveal = CATransitionType(rawValue: "reveal")
}

/// Constants that specify the transition subtype.
public struct CATransitionSubtype: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The transition begins at the right side of the layer.
    public static let fromRight = CATransitionSubtype(rawValue: "fromRight")

    /// The transition begins at the left side of the layer.
    public static let fromLeft = CATransitionSubtype(rawValue: "fromLeft")

    /// The transition begins at the top of the layer.
    public static let fromTop = CATransitionSubtype(rawValue: "fromTop")

    /// The transition begins at the bottom of the layer.
    public static let fromBottom = CATransitionSubtype(rawValue: "fromBottom")
}

/// An object that provides an animated transition between a layer's states.
open class CATransition: CAAnimation {

    /// The name of the transition.
    open var type: CATransitionType = .fade

    /// An optional subtype that specifies a direction or configuration for the transition.
    open var subtype: CATransitionSubtype?

    /// The amount of progress through to the transition at which to begin and end execution.
    open var startProgress: Float = 0

    /// The amount of progress through the transition at which to end execution.
    open var endProgress: Float = 1

    /// An optional Core Image filter object that provides the transition.
    open var filter: Any?
}
