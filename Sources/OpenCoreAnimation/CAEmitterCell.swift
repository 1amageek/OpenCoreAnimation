// CAEmitterCell.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework

import Foundation

/// The definition of a particle emitted by a particle layer.
open class CAEmitterCell: CAMediaTiming {

    public init() {}

    // MARK: - Cell Content

    /// The contents of the cell.
    open var contents: Any?

    /// The bounds of the cell's contents rectangle.
    open var contentsRect: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)

    /// The scale factor applied to the contents of the cell.
    open var contentsScale: CGFloat = 1.0

    /// The filter used when the contents are enlarged.
    open var magnificationFilter: String = CALayerContentsFilter.linear.rawValue

    /// The filter used when the contents are reduced.
    open var minificationFilter: String = CALayerContentsFilter.linear.rawValue

    /// The bias applied when selecting a minification mip level.
    open var minificationFilterBias: Float = 0

    // MARK: - Emitter Behavior

    /// The number of emitted objects created every second.
    open var birthRate: Float = 0

    /// The lifetime of the cell, in seconds.
    open var lifetime: Float = 0

    /// The range of values used by lifetime.
    open var lifetimeRange: Float = 0

    /// An optional array containing the sub-cells of this cell.
    open var emitterCells: [CAEmitterCell]?

    // MARK: - Color Properties

    /// The color of each emitted object.
    open var color: CGColor? = CGColor(red: 1, green: 1, blue: 1, alpha: 1)

    /// The amount by which the red color component of the cell can vary.
    open var redRange: Float = 0

    /// The amount by which the green color component of the cell can vary.
    open var greenRange: Float = 0

    /// The amount by which the blue color component of the cell can vary.
    open var blueRange: Float = 0

    /// The amount by which the alpha component of the cell can vary.
    open var alphaRange: Float = 0

    /// The speed at which the red color component changes over the lifetime of the cell.
    open var redSpeed: Float = 0

    /// The speed at which the green color component changes over the lifetime of the cell.
    open var greenSpeed: Float = 0

    /// The speed at which the blue color component changes over the lifetime of the cell.
    open var blueSpeed: Float = 0

    /// The speed at which the alpha component changes over the lifetime of the cell.
    open var alphaSpeed: Float = 0

    // MARK: - Geometry

    /// The initial velocity of the cell.
    open var velocity: CGFloat = 0

    /// The amount by which the velocity of the cell can vary.
    open var velocityRange: CGFloat = 0

    /// The x-component of the acceleration vector applied to emitted objects.
    open var xAcceleration: CGFloat = 0

    /// The y-component of the acceleration vector applied to emitted objects.
    open var yAcceleration: CGFloat = 0

    /// The z-component of the acceleration vector applied to emitted objects.
    open var zAcceleration: CGFloat = 0

    /// The initial scale factor applied to the cell.
    open var scale: CGFloat = 1

    /// The amount by which the scale of the cell can vary.
    open var scaleRange: CGFloat = 0

    /// The speed at which the scale changes over the lifetime of the cell.
    open var scaleSpeed: CGFloat = 0

    /// The rotational velocity, measured in radians per second, to apply to the cell.
    open var spin: CGFloat = 0

    /// The amount by which the spin of the cell can vary.
    open var spinRange: CGFloat = 0

    /// The orientation of the emission angle.
    open var emissionLatitude: CGFloat = 0

    /// The emission angle of the cell.
    open var emissionLongitude: CGFloat = 0

    /// The emission range of the cell.
    open var emissionRange: CGFloat = 0

    // MARK: - Identifying

    /// The name of the cell.
    open var name: String?

    /// A Boolean indicating whether or not cells from this emitter are rendered.
    open var isEnabled: Bool = true

    /// An optional dictionary of style properties for the cell.
    open var style: [AnyHashable: Any]?

    // MARK: - CAMediaTiming

    open var beginTime: CFTimeInterval = 0
    open var timeOffset: CFTimeInterval = 0
    open var repeatCount: Float = 0
    open var repeatDuration: CFTimeInterval = 0
    open var duration: CFTimeInterval = .infinity
    open var speed: Float = 1
    open var autoreverses: Bool = false
    open var fillMode: CAMediaTimingFillMode = .removed

    // MARK: - Default Value

    /// Returns the default value for a given property key.
    ///
    /// - Parameter key: The key of the property.
    /// - Returns: The default value for the property, or `nil` if no default is defined.
    open class func defaultValue(forKey key: String) -> Any? {
        switch key {
        case "contentsRect":
            return CGRect(x: 0, y: 0, width: 1, height: 1)
        case "contentsScale":
            return CGFloat(1.0)
        case "magnificationFilter", "minificationFilter":
            return CALayerContentsFilter.linear.rawValue
        case "color":
            return CGColor(red: 1, green: 1, blue: 1, alpha: 1)
        case "scale":
            return CGFloat(1)
        case "enabled":
            return true
        case "duration":
            return CFTimeInterval.infinity
        case "speed":
            return Float(1)
        case "fillMode":
            return CAMediaTimingFillMode.removed
        default:
            return nil
        }
    }
}
