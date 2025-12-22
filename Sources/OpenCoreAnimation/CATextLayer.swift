//
//  CATextLayerTruncationMode.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics

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

    // MARK: - Text Properties

    /// The text to be rendered by the receiver.
    open var string: Any?

    /// The font used to render the receiver's text.
    open var font: Any?

    internal var _fontSize: CGFloat = 36
    /// The font size used to render the receiver's text. Animatable.
    open var fontSize: CGFloat {
        get { return _fontSize }
        set {
            let oldValue = _fontSize
            _fontSize = newValue
            CATransaction.registerChange(layer: self, keyPath: "fontSize", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _foregroundColor: CGColor?
    /// The color used to render the receiver's text. Animatable.
    open var foregroundColor: CGColor? {
        get { return _foregroundColor }
        set {
            let oldValue = _foregroundColor
            _foregroundColor = newValue
            CATransaction.registerChange(layer: self, keyPath: "foregroundColor", oldValue: oldValue, newValue: newValue)
        }
    }

    // MARK: - Layout Properties

    /// Determines whether the text is wrapped to fit within the receiver's bounds.
    open var isWrapped: Bool = false

    /// The truncation mode to use when the text is too long.
    open var truncationMode: CATextLayerTruncationMode = .none

    /// The text alignment mode.
    open var alignmentMode: CATextLayerAlignmentMode = .natural

    /// Determines whether font smoothing is allowed.
    open var allowsFontSubpixelQuantization: Bool = false

    // MARK: - Animatable Keys

    /// The list of animatable property keys for CATextLayer.
    private static let textLayerAnimatableKeys: Set<String> = [
        "fontSize",
        "foregroundColor"
    ]

    /// Returns the default action for the specified key.
    open override class func defaultAction(forKey event: String) -> (any CAAction)? {
        if textLayerAnimatableKeys.contains(event) {
            let animation = CABasicAnimation(keyPath: event)
            animation.duration = CATransaction.animationDuration()
            if let timingFunction = CATransaction.animationTimingFunction() {
                animation.timingFunction = timingFunction
            }
            return animation
        }
        return super.defaultAction(forKey: event)
    }
}
