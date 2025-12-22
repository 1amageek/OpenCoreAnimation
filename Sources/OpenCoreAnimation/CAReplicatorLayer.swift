//
//  CAReplicatorLayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics



/// A layer that creates a specified number of sublayer copies with varying geometric,
/// temporal, and color transformations.
open class CAReplicatorLayer: CALayer {

    // MARK: - Initialization

    public required init() {
        super.init()
    }

    /// Initializes a new replicator layer as a copy of the specified layer.
    public required init(layer: Any) {
        super.init(layer: layer)
        if let replicatorLayer = layer as? CAReplicatorLayer {
            self.instanceCount = replicatorLayer.instanceCount
            self.preservesDepth = replicatorLayer.preservesDepth
            self._instanceDelay = replicatorLayer._instanceDelay
            self._instanceTransform = replicatorLayer._instanceTransform
            self._instanceColor = replicatorLayer._instanceColor
            self._instanceRedOffset = replicatorLayer._instanceRedOffset
            self._instanceGreenOffset = replicatorLayer._instanceGreenOffset
            self._instanceBlueOffset = replicatorLayer._instanceBlueOffset
            self._instanceAlphaOffset = replicatorLayer._instanceAlphaOffset
        }
    }

    // MARK: - Instance Properties

    /// The number of copies to create, including the source layer.
    open var instanceCount: Int = 1

    /// Specifies whether this layer flattens its sublayers into its plane.
    open var preservesDepth: Bool = false

    internal var _instanceDelay: CFTimeInterval = 0
    /// The delay, in seconds, between replicated copies. Animatable.
    open var instanceDelay: CFTimeInterval {
        get { return _instanceDelay }
        set {
            let oldValue = _instanceDelay
            _instanceDelay = newValue
            CATransaction.registerChange(layer: self, keyPath: "instanceDelay", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _instanceTransform: CATransform3D = CATransform3DIdentity
    /// The transform matrix applied to the previous instance to produce the current instance. Animatable.
    open var instanceTransform: CATransform3D {
        get { return _instanceTransform }
        set {
            let oldValue = _instanceTransform
            _instanceTransform = newValue
            CATransaction.registerChange(layer: self, keyPath: "instanceTransform", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _instanceColor: CGColor?
    /// The color used to multiply the source object. Animatable.
    open var instanceColor: CGColor? {
        get { return _instanceColor }
        set {
            let oldValue = _instanceColor
            _instanceColor = newValue
            CATransaction.registerChange(layer: self, keyPath: "instanceColor", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _instanceRedOffset: Float = 0
    /// Defines the offset added to the red component of the color for each replicated instance. Animatable.
    open var instanceRedOffset: Float {
        get { return _instanceRedOffset }
        set {
            let oldValue = _instanceRedOffset
            _instanceRedOffset = newValue
            CATransaction.registerChange(layer: self, keyPath: "instanceRedOffset", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _instanceGreenOffset: Float = 0
    /// Defines the offset added to the green component of the color for each replicated instance. Animatable.
    open var instanceGreenOffset: Float {
        get { return _instanceGreenOffset }
        set {
            let oldValue = _instanceGreenOffset
            _instanceGreenOffset = newValue
            CATransaction.registerChange(layer: self, keyPath: "instanceGreenOffset", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _instanceBlueOffset: Float = 0
    /// Defines the offset added to the blue component of the color for each replicated instance. Animatable.
    open var instanceBlueOffset: Float {
        get { return _instanceBlueOffset }
        set {
            let oldValue = _instanceBlueOffset
            _instanceBlueOffset = newValue
            CATransaction.registerChange(layer: self, keyPath: "instanceBlueOffset", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _instanceAlphaOffset: Float = 0
    /// Defines the offset added to the alpha component of the color for each replicated instance. Animatable.
    open var instanceAlphaOffset: Float {
        get { return _instanceAlphaOffset }
        set {
            let oldValue = _instanceAlphaOffset
            _instanceAlphaOffset = newValue
            CATransaction.registerChange(layer: self, keyPath: "instanceAlphaOffset", oldValue: oldValue, newValue: newValue)
        }
    }

    // MARK: - Animatable Keys

    /// The list of animatable property keys for CAReplicatorLayer.
    private static let replicatorLayerAnimatableKeys: Set<String> = [
        "instanceDelay",
        "instanceTransform",
        "instanceColor",
        "instanceRedOffset",
        "instanceGreenOffset",
        "instanceBlueOffset",
        "instanceAlphaOffset"
    ]

    /// Returns the default action for the specified key.
    open override class func defaultAction(forKey event: String) -> (any CAAction)? {
        if replicatorLayerAnimatableKeys.contains(event) {
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
