
/// A layer that emits, animates, and renders a particle system.
open class CAEmitterLayer: CALayer {

    // MARK: - Initialization

    public required init() {
        super.init()
    }

    /// Initializes a new emitter layer as a copy of the specified layer.
    public required init(layer: Any) {
        super.init(layer: layer)
        if let emitterLayer = layer as? CAEmitterLayer {
            self.emitterCells = emitterLayer.emitterCells
            self._emitterPosition = emitterLayer._emitterPosition
            self._emitterZPosition = emitterLayer._emitterZPosition
            self._emitterSize = emitterLayer._emitterSize
            self.emitterDepth = emitterLayer.emitterDepth
            self.emitterShape = emitterLayer.emitterShape
            self.emitterMode = emitterLayer.emitterMode
            self.renderMode = emitterLayer.renderMode
            self.preservesDepth = emitterLayer.preservesDepth
            self._birthRate = emitterLayer._birthRate
            self._lifetime = emitterLayer._lifetime
            self._velocity = emitterLayer._velocity
            self.scale = emitterLayer.scale
            self._spin = emitterLayer._spin
            self.seed = emitterLayer.seed
        }
    }

    // MARK: - Emitter Cells

    /// An array of CAEmitterCell objects that define the types of emitted objects.
    open var emitterCells: [CAEmitterCell]?

    // MARK: - Emitter Geometry

    internal var _emitterPosition: CGPoint = .zero
    /// The position of the center of the particle emitter. Animatable.
    open var emitterPosition: CGPoint {
        get { return _emitterPosition }
        set {
            let oldValue = _emitterPosition
            _emitterPosition = newValue
            CATransaction.registerChange(layer: self, keyPath: "emitterPosition", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _emitterZPosition: CGFloat = 0
    /// The depth of the particle emitter. Animatable.
    open var emitterZPosition: CGFloat {
        get { return _emitterZPosition }
        set {
            let oldValue = _emitterZPosition
            _emitterZPosition = newValue
            CATransaction.registerChange(layer: self, keyPath: "emitterZPosition", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _emitterSize: CGSize = .zero
    /// The size of the particle emitter. Animatable.
    open var emitterSize: CGSize {
        get { return _emitterSize }
        set {
            let oldValue = _emitterSize
            _emitterSize = newValue
            CATransaction.registerChange(layer: self, keyPath: "emitterSize", oldValue: oldValue, newValue: newValue)
        }
    }

    /// The depth of the particle emitter. Animatable.
    open var emitterDepth: CGFloat = 0

    /// The shape of the particle emitter.
    open var emitterShape: CAEmitterLayerEmitterShape = .point

    /// The emission mode of the particle emitter.
    open var emitterMode: CAEmitterLayerEmitterMode = .volume

    // MARK: - Emitter Behavior

    /// The render mode of the particle emitter.
    open var renderMode: CAEmitterLayerRenderMode = .unordered

    /// Specifies whether this layer flattens its sublayers into its plane.
    open var preservesDepth: Bool = false

    internal var _birthRate: Float = 1
    /// Defines a multiplier applied to the cell-defined birth rate. Animatable.
    open var birthRate: Float {
        get { return _birthRate }
        set {
            let oldValue = _birthRate
            _birthRate = newValue
            CATransaction.registerChange(layer: self, keyPath: "birthRate", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _lifetime: Float = 1
    /// Defines a multiplier applied to the cell-defined lifetime. Animatable.
    open var lifetime: Float {
        get { return _lifetime }
        set {
            let oldValue = _lifetime
            _lifetime = newValue
            CATransaction.registerChange(layer: self, keyPath: "lifetime", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _velocity: Float = 1
    /// Defines a multiplier applied to the cell-defined velocity. Animatable.
    open var velocity: Float {
        get { return _velocity }
        set {
            let oldValue = _velocity
            _velocity = newValue
            CATransaction.registerChange(layer: self, keyPath: "velocity", oldValue: oldValue, newValue: newValue)
        }
    }

    /// Defines a multiplier applied to the cell-defined scale.
    open var scale: Float = 1

    internal var _spin: Float = 1
    /// Defines a multiplier applied to the cell-defined spin. Animatable.
    open var spin: Float {
        get { return _spin }
        set {
            let oldValue = _spin
            _spin = newValue
            CATransaction.registerChange(layer: self, keyPath: "spin", oldValue: oldValue, newValue: newValue)
        }
    }

    /// Specifies the seed used to initialize the random number generator.
    open var seed: UInt32 = 0

    // MARK: - Animatable Keys

    /// The list of animatable property keys for CAEmitterLayer.
    private static let emitterLayerAnimatableKeys: Set<String> = [
        "emitterPosition",
        "emitterZPosition",
        "emitterSize",
        "birthRate",
        "lifetime",
        "velocity",
        "spin"
    ]

    /// Returns the default action for the specified key.
    open override class func defaultAction(forKey event: String) -> (any CAAction)? {
        if emitterLayerAnimatableKeys.contains(event) {
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
