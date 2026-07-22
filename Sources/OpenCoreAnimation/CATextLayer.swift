//
//  CATextLayer.swift
//  OpenCoreAnimation
//
//  Full API compatibility with Apple's CoreAnimation framework.
//

import Foundation

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

    // MARK: - Initialization

    public required init() {
        super.init()
    }

    /// Initializes a new text layer as a copy of the specified layer.
    public required init(layer: Any) {
        super.init(layer: layer)
        if let textLayer = layer as? CATextLayer {
            self.string = textLayer.string
            self.font = textLayer.font
            self._fontSize = textLayer._fontSize
            self._foregroundColor = textLayer._foregroundColor
            self.isWrapped = textLayer.isWrapped
            self.truncationMode = textLayer.truncationMode
            self.alignmentMode = textLayer.alignmentMode
            self.allowsFontSubpixelQuantization = textLayer.allowsFontSubpixelQuantization
        }
    }

    /// Specifies the default value associated with a text-layer property.
    open override class func defaultValue(forKey key: String) -> Any? {
        switch key {
        case "font":
            return "Helvetica"
        case "fontSize":
            return CGFloat(36)
        case "foregroundColor":
            return CGColor(red: 1, green: 1, blue: 1, alpha: 1)
        case "truncationMode":
            return CATextLayerTruncationMode.none
        case "alignmentMode":
            return CATextLayerAlignmentMode.natural
        default:
            return super.defaultValue(forKey: key)
        }
    }

    // MARK: - Text Properties

    /// The text to be rendered by the receiver.
    open var string: Any? {
        didSet { markDirty(.contents) }
    }

    /// The font used to render the receiver's text.
    open var font: Any? = "Helvetica" {
        didSet { markDirty(.contents) }
    }

    internal var _fontSize: CGFloat = 36
    /// The font size used to render the receiver's text. Animatable.
    open var fontSize: CGFloat {
        get { return _fontSize }
        set {
            let oldValue = _fontSize
            _fontSize = newValue
            markDirty(.contents)
            CATransaction.registerChange(layer: self, keyPath: "fontSize", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _foregroundColor: CGColor? = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
    /// The color used to render the receiver's text. Animatable.
    open var foregroundColor: CGColor? {
        get { return _foregroundColor }
        set {
            let oldValue = _foregroundColor
            _foregroundColor = newValue
            markDirty(.appearance)
            CATransaction.registerChange(layer: self, keyPath: "foregroundColor", oldValue: oldValue, newValue: newValue)
        }
    }

    // MARK: - Layout Properties

    /// Determines whether the text is wrapped to fit within the receiver's bounds.
    open var isWrapped: Bool = false {
        didSet {
            guard oldValue != isWrapped else { return }
            markDirty(.contents)
        }
    }

    /// The truncation mode to use when the text is too long.
    open var truncationMode: CATextLayerTruncationMode = .none {
        didSet {
            guard oldValue != truncationMode else { return }
            markDirty(.contents)
        }
    }

    /// The text alignment mode.
    open var alignmentMode: CATextLayerAlignmentMode = .natural {
        didSet {
            guard oldValue != alignmentMode else { return }
            markDirty(.contents)
        }
    }

    /// Determines whether font smoothing is allowed.
    open var allowsFontSubpixelQuantization: Bool = false {
        didSet {
            guard oldValue != allowsFontSubpixelQuantization else { return }
            markDirty(.contents)
        }
    }

}
