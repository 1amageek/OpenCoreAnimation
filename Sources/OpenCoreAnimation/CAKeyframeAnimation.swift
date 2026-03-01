//
//  CAKeyframeAnimation.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics

/// An object that provides keyframe animation capabilities for a layer object.
open class CAKeyframeAnimation: CAPropertyAnimation {

    /// An array of objects that specify the keyframe values to use for the animation.
    open var values: [Any]?

    /// The path for a point-based property to follow.
    open var path: CGPath?

    /// An optional array of numbers that define the time at which each keyframe value is applied.
    /// Values should be in the range [0, 1].
    open var keyTimes: [CGFloat]?

    /// An optional array of CAMediaTimingFunction objects.
    open var timingFunctions: [CAMediaTimingFunction]?

    /// Specifies how intermediate keyframe values are calculated by the receiver.
    open var calculationMode: CAAnimationCalculationMode = .linear

    /// An array of numbers that define the tightness of the curve.
    /// Used with cubic and Catmull-Rom calculation modes.
    open var tensionValues: [CGFloat]?

    /// An array of numbers that define the sharpness of timing curve corners.
    /// Used with cubic calculation mode.
    open var continuityValues: [CGFloat]?

    /// An array of numbers that define the position of the curve relative to a control point.
    /// Used with cubic calculation mode.
    open var biasValues: [CGFloat]?

    /// Determines whether objects animating along the path rotate to match the path tangent.
    open var rotationMode: CAAnimationRotationMode?
}
