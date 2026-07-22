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
            self._string = textLayer._string
            self._font = textLayer._font
            self._fontSize = textLayer._fontSize
            self._foregroundColor = textLayer._foregroundColor
            self._isWrapped = textLayer._isWrapped
            self._truncationMode = textLayer._truncationMode
            self._alignmentMode = textLayer._alignmentMode
            self._allowsFontSubpixelQuantization = textLayer._allowsFontSubpixelQuantization
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

    internal var _string: Any?
    /// The text to be rendered by the receiver.
    open var string: Any? {
        get { _string }
        set {
            guard !CALayer.storedValuesEqual(_string, newValue) else { return }
            _string = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "string") { setNeedsDisplay() }
        }
    }

    internal var _font: Any? = "Helvetica"
    /// The font used to render the receiver's text.
    open var font: Any? {
        get { _font }
        set {
            guard !CALayer.storedValuesEqual(_font, newValue) else { return }
            _font = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "font") { setNeedsDisplay() }
        }
    }

    internal var _fontSize: CGFloat = 36
    /// The font size used to render the receiver's text. Animatable.
    open var fontSize: CGFloat {
        get { return _fontSize }
        set {
            let oldValue = _fontSize
            guard oldValue != newValue else { return }
            _fontSize = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "fontSize") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "fontSize", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _foregroundColor: CGColor? = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
    /// The color used to render the receiver's text. Animatable.
    open var foregroundColor: CGColor? {
        get { return _foregroundColor }
        set {
            let oldValue = _foregroundColor
            guard oldValue != newValue else { return }
            _foregroundColor = newValue
            markDirty(.appearance)
            if Self.needsDisplay(forKey: "foregroundColor") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "foregroundColor", oldValue: oldValue, newValue: newValue)
        }
    }

    // MARK: - Layout Properties

    internal var _isWrapped = false
    /// Determines whether the text is wrapped to fit within the receiver's bounds.
    open var isWrapped: Bool {
        get { _isWrapped }
        set {
            guard _isWrapped != newValue else { return }
            _isWrapped = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "wrapped") { setNeedsDisplay() }
        }
    }

    internal var _truncationMode: CATextLayerTruncationMode = .none
    /// The truncation mode to use when the text is too long.
    open var truncationMode: CATextLayerTruncationMode {
        get { _truncationMode }
        set {
            guard _truncationMode != newValue else { return }
            _truncationMode = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "truncationMode") { setNeedsDisplay() }
        }
    }

    internal var _alignmentMode: CATextLayerAlignmentMode = .natural
    /// The text alignment mode.
    open var alignmentMode: CATextLayerAlignmentMode {
        get { _alignmentMode }
        set {
            guard _alignmentMode != newValue else { return }
            _alignmentMode = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "alignmentMode") { setNeedsDisplay() }
        }
    }

    internal var _allowsFontSubpixelQuantization = false
    /// Determines whether font smoothing is allowed.
    open var allowsFontSubpixelQuantization: Bool {
        get { _allowsFontSubpixelQuantization }
        set {
            guard _allowsFontSubpixelQuantization != newValue else { return }
            _allowsFontSubpixelQuantization = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "allowsFontSubpixelQuantization") { setNeedsDisplay() }
        }
    }

    /// Returns whether a text-layer property change requires content redraw.
    open override class func needsDisplay(forKey key: String) -> Bool {
        switch key {
        case "string", "font", "fontSize", "foregroundColor", "wrapped",
             "truncationMode", "alignmentMode", "allowsFontSubpixelQuantization",
             "style", "contentsScale":
            return true
        default:
            return super.needsDisplay(forKey: key)
        }
    }

    /// Returns whether a text-layer property differs from its archive default.
    open override func shouldArchiveValue(forKey key: String) -> Bool {
        switch key {
        case "string": return string != nil
        case "font": return (font as? String) != "Helvetica"
        case "fontSize": return fontSize != 36
        case "foregroundColor":
            return foregroundColor != CGColor(red: 1, green: 1, blue: 1, alpha: 1)
        case "wrapped": return isWrapped
        case "truncationMode": return truncationMode != .none
        case "alignmentMode": return alignmentMode != .natural
        case "allowsFontSubpixelQuantization": return allowsFontSubpixelQuantization
        default: return super.shouldArchiveValue(forKey: key)
        }
    }

}
