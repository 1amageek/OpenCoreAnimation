
/// Constants that specify how a layer's contents are positioned or scaled within its bounds.
public struct CALayerContentsGravity: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The content is horizontally centered at the bottom-edge of the bounds rectangle.
    public static let bottom = CALayerContentsGravity(rawValue: "bottom")

    /// The content is positioned in the bottom-left corner of the bounds rectangle.
    public static let bottomLeft = CALayerContentsGravity(rawValue: "bottomLeft")

    /// The content is positioned in the bottom-right corner of the bounds rectangle.
    public static let bottomRight = CALayerContentsGravity(rawValue: "bottomRight")

    /// The content is horizontally and vertically centered in the bounds rectangle.
    public static let center = CALayerContentsGravity(rawValue: "center")

    /// The content is vertically centered at the left-edge of the bounds rectangle.
    public static let left = CALayerContentsGravity(rawValue: "left")

    /// The content is resized to fit the entire bounds rectangle.
    public static let resize = CALayerContentsGravity(rawValue: "resize")

    /// The content is resized to fit the bounds rectangle, preserving the aspect of the content.
    /// If the content does not completely fill the bounds rectangle, the content is centered in the partial axis.
    public static let resizeAspect = CALayerContentsGravity(rawValue: "resizeAspect")

    /// The content is resized to completely fill the bounds rectangle, while still preserving the aspect of the content.
    /// The content is centered in the axis it exceeds.
    public static let resizeAspectFill = CALayerContentsGravity(rawValue: "resizeAspectFill")

    /// The content is vertically centered at the right-edge of the bounds rectangle.
    public static let right = CALayerContentsGravity(rawValue: "right")

    /// The content is horizontally centered at the top-edge of the bounds rectangle.
    public static let top = CALayerContentsGravity(rawValue: "top")

    /// The content is positioned in the top-left corner of the bounds rectangle.
    public static let topLeft = CALayerContentsGravity(rawValue: "topLeft")

    /// The content is positioned in the top-right corner of the bounds rectangle.
    public static let topRight = CALayerContentsGravity(rawValue: "topRight")
}
