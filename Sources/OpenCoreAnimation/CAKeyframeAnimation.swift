//
//  CAKeyframeAnimation.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation

/// An object that provides keyframe animation capabilities for a layer object.
open class CAKeyframeAnimation: CAPropertyAnimation {

    /// An array of objects that specify the keyframe values to use for the animation.
    open var values: [Any]? {
        didSet { notifyAttachedLayerOfMutation() }
    }

    /// The path for a point-based property to follow.
    open var path: CGPath? {
        didSet { notifyAttachedLayerOfMutation() }
    }

    /// An optional array of numbers that define the time at which each keyframe value is applied.
    /// Values should be in the range [0, 1].
    open var keyTimes: [CGFloat]? {
        didSet { notifyAttachedLayerOfMutation() }
    }

    /// An optional array of CAMediaTimingFunction objects.
    open var timingFunctions: [CAMediaTimingFunction]? {
        didSet { notifyAttachedLayerOfMutation() }
    }

    /// Specifies how intermediate keyframe values are calculated by the receiver.
    open var calculationMode: CAAnimationCalculationMode = .linear {
        didSet { notifyAttachedLayerOfMutation() }
    }

    /// An array of numbers that define the tightness of the curve.
    /// Used with cubic and Catmull-Rom calculation modes.
    open var tensionValues: [CGFloat]? {
        didSet { notifyAttachedLayerOfMutation() }
    }

    /// An array of numbers that define the sharpness of timing curve corners.
    /// Used with cubic calculation mode.
    open var continuityValues: [CGFloat]? {
        didSet { notifyAttachedLayerOfMutation() }
    }

    /// An array of numbers that define the position of the curve relative to a control point.
    /// Used with cubic calculation mode.
    open var biasValues: [CGFloat]? {
        didSet { notifyAttachedLayerOfMutation() }
    }

    /// Determines whether objects animating along the path rotate to match the path tangent.
    open var rotationMode: CAAnimationRotationMode? {
        didSet { notifyAttachedLayerOfMutation() }
    }

    public required init() {
        super.init()
    }

    public required init(animation: CAAnimation) {
        super.init(animation: animation)
        if let source = animation as? CAKeyframeAnimation {
            self.values = source.values
            self.path = source.path
            self.keyTimes = source.keyTimes
            self.timingFunctions = source.timingFunctions
            self.calculationMode = source.calculationMode
            self.tensionValues = source.tensionValues
            self.continuityValues = source.continuityValues
            self.biasValues = source.biasValues
            self.rotationMode = source.rotationMode
        }
    }

    open override func shouldArchiveValue(forKey key: String) -> Bool {
        switch key {
        case "values":
            return values != nil
        case "path":
            return path != nil
        case "keyTimes":
            return keyTimes != nil
        case "timingFunctions":
            return timingFunctions != nil
        case "calculationMode":
            return calculationMode != .linear
        case "tensionValues":
            return tensionValues != nil
        case "continuityValues":
            return continuityValues != nil
        case "biasValues":
            return biasValues != nil
        case "rotationMode":
            return rotationMode != nil
        default:
            return super.shouldArchiveValue(forKey: key)
        }
    }

    /// Returns the default value for a keyframe animation property.
    open override class func defaultValue(forKey key: String) -> Any? {
        switch key {
        case "calculationMode":
            return CAAnimationCalculationMode.linear
        default:
            return super.defaultValue(forKey: key)
        }
    }
}
