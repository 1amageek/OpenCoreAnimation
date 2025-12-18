
/// A layer that draws a color gradient over its background color, filling the shape of the layer.
open class CAGradientLayer: CALayer {

    // MARK: - Initialization

    public required init() {
        super.init()
    }

    /// Initializes a new gradient layer as a copy of the specified layer.
    public required init(layer: Any) {
        super.init(layer: layer)
        if let gradientLayer = layer as? CAGradientLayer {
            self._colors = gradientLayer._colors
            self._locations = gradientLayer._locations
            self._startPoint = gradientLayer._startPoint
            self._endPoint = gradientLayer._endPoint
            self.type = gradientLayer.type
        }
    }

    // MARK: - Gradient Properties

    internal var _colors: [Any]?
    /// An array of CGColor objects defining the color of each gradient stop. Animatable.
    open var colors: [Any]? {
        get { return _colors }
        set {
            let oldValue = _colors
            _colors = newValue
            CATransaction.registerChange(layer: self, keyPath: "colors", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _locations: [Float]?
    /// An optional array of numbers defining the location of each gradient stop. Animatable.
    /// Values should be in the range [0, 1] and monotonically increasing.
    open var locations: [Float]? {
        get { return _locations }
        set {
            let oldValue = _locations
            _locations = newValue
            CATransaction.registerChange(layer: self, keyPath: "locations", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _startPoint: CGPoint = CGPoint(x: 0.5, y: 0)
    /// The start point of the gradient when drawn in the layer's coordinate space. Animatable.
    open var startPoint: CGPoint {
        get { return _startPoint }
        set {
            let oldValue = _startPoint
            _startPoint = newValue
            CATransaction.registerChange(layer: self, keyPath: "startPoint", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _endPoint: CGPoint = CGPoint(x: 0.5, y: 1)
    /// The end point of the gradient when drawn in the layer's coordinate space. Animatable.
    open var endPoint: CGPoint {
        get { return _endPoint }
        set {
            let oldValue = _endPoint
            _endPoint = newValue
            CATransaction.registerChange(layer: self, keyPath: "endPoint", oldValue: oldValue, newValue: newValue)
        }
    }

    /// Style of gradient drawn by the layer.
    open var type: CAGradientLayerType = .axial

    // MARK: - Animatable Keys

    /// The list of animatable property keys for CAGradientLayer.
    private static let gradientLayerAnimatableKeys: Set<String> = [
        "colors",
        "locations",
        "startPoint",
        "endPoint"
    ]

    /// Returns the default action for the specified key.
    open override class func defaultAction(forKey event: String) -> (any CAAction)? {
        // First check CAGradientLayer-specific keys
        if gradientLayerAnimatableKeys.contains(event) {
            let animation = CABasicAnimation(keyPath: event)
            animation.duration = CATransaction.animationDuration()
            if let timingFunction = CATransaction.animationTimingFunction() {
                animation.timingFunction = timingFunction
            }
            return animation
        }
        // Fall back to parent class
        return super.defaultAction(forKey: event)
    }
}
