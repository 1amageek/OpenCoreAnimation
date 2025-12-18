
/// Constants that specify text truncation.
public struct CATextLayerTruncationMode: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Do not truncate.
    public static let none = CATextLayerTruncationMode(rawValue: "none")

    /// Truncate at the beginning of the line.
    public static let start = CATextLayerTruncationMode(rawValue: "start")

    /// Truncate at the end of the line.
    public static let end = CATextLayerTruncationMode(rawValue: "end")

    /// Truncate in the middle of the line.
    public static let middle = CATextLayerTruncationMode(rawValue: "middle")
}

/// Constants that specify text alignment.
public struct CATextLayerAlignmentMode: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Text is visually left-aligned.
    public static let left = CATextLayerAlignmentMode(rawValue: "left")

    /// Text is visually right-aligned.
    public static let right = CATextLayerAlignmentMode(rawValue: "right")

    /// Text is visually center-aligned.
    public static let center = CATextLayerAlignmentMode(rawValue: "center")

    /// Text is fully justified.
    public static let justified = CATextLayerAlignmentMode(rawValue: "justified")

    /// Text uses the default alignment associated with the current localization.
    public static let natural = CATextLayerAlignmentMode(rawValue: "natural")
}

/// A layer that provides simple text layout and rendering of plain or attributed strings.
open class CATextLayer: CALayer {

    // MARK: - Text Properties

    /// The text to be rendered by the receiver.
    open var string: Any?

    /// The font used to render the receiver's text.
    open var font: Any?

    /// The font size used to render the receiver's text.
    open var fontSize: CGFloat = 36

    /// The color used to render the receiver's text.
    open var foregroundColor: CGColor?

    // MARK: - Layout Properties

    /// Determines whether the text is wrapped to fit within the receiver's bounds.
    open var isWrapped: Bool = false

    /// The truncation mode to use when the text is too long.
    open var truncationMode: CATextLayerTruncationMode = .none

    /// The text alignment mode.
    open var alignmentMode: CATextLayerAlignmentMode = .natural

    /// Determines whether font smoothing is allowed.
    open var allowsFontSubpixelQuantization: Bool = false
}
