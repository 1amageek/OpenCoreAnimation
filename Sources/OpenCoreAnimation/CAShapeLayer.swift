
/// A layer that draws a cubic Bezier spline in its coordinate space.
///
/// The shape is composited between the layer's contents and its first sublayer.
/// The shape will be drawn antialiased, and whenever possible it will be mapped into screen space
/// before being rasterized to preserve resolution independence.
open class CAShapeLayer: CALayer {

    // MARK: - Initialization

    public required init() {
        super.init()
    }

    /// Initializes a new shape layer as a copy of the specified layer.
    public required init(layer: Any) {
        super.init(layer: layer)
        if let shapeLayer = layer as? CAShapeLayer {
            // Copy shape path
            self._path = shapeLayer._path

            // Copy style properties
            self._fillColor = shapeLayer._fillColor
            self.fillRule = shapeLayer.fillRule
            self.lineCap = shapeLayer.lineCap
            self.lineDashPattern = shapeLayer.lineDashPattern
            self._lineDashPhase = shapeLayer._lineDashPhase
            self.lineJoin = shapeLayer.lineJoin
            self._lineWidth = shapeLayer._lineWidth
            self._miterLimit = shapeLayer._miterLimit
            self._strokeColor = shapeLayer._strokeColor
            self._strokeStart = shapeLayer._strokeStart
            self._strokeEnd = shapeLayer._strokeEnd
        }
    }

    // MARK: - Specifying the Shape Path

    internal var _path: CGPath?
    /// The path defining the shape to be rendered. Animatable.
    open var path: CGPath? {
        get { return _path }
        set {
            let oldValue = _path
            _path = newValue
            CATransaction.registerChange(layer: self, keyPath: "path", oldValue: oldValue, newValue: newValue)
        }
    }

    // MARK: - Accessing Shape Style Properties

    internal var _fillColor: CGColor?
    /// The color used to fill the shape's path. Animatable.
    open var fillColor: CGColor? {
        get { return _fillColor }
        set {
            let oldValue = _fillColor
            _fillColor = newValue
            CATransaction.registerChange(layer: self, keyPath: "fillColor", oldValue: oldValue, newValue: newValue)
        }
    }

    /// The fill rule used when filling the shape's path.
    open var fillRule: CAShapeLayerFillRule = .nonZero

    /// Specifies the line cap style for the shape's path.
    open var lineCap: CAShapeLayerLineCap = .butt

    /// The dash pattern applied to the shape's path when stroked.
    /// Each element specifies the length of a dash or gap in the pattern.
    open var lineDashPattern: [CGFloat]?

    internal var _lineDashPhase: CGFloat = 0
    /// The dash phase applied to the shape's path when stroked. Animatable.
    open var lineDashPhase: CGFloat {
        get { return _lineDashPhase }
        set {
            let oldValue = _lineDashPhase
            _lineDashPhase = newValue
            CATransaction.registerChange(layer: self, keyPath: "lineDashPhase", oldValue: oldValue, newValue: newValue)
        }
    }

    /// Specifies the line join style for the shape's path.
    open var lineJoin: CAShapeLayerLineJoin = .miter

    internal var _lineWidth: CGFloat = 1
    /// Specifies the line width of the shape's path. Animatable.
    open var lineWidth: CGFloat {
        get { return _lineWidth }
        set {
            let oldValue = _lineWidth
            _lineWidth = newValue
            CATransaction.registerChange(layer: self, keyPath: "lineWidth", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _miterLimit: CGFloat = 10
    /// The miter limit used when stroking the shape's path. Animatable.
    open var miterLimit: CGFloat {
        get { return _miterLimit }
        set {
            let oldValue = _miterLimit
            _miterLimit = newValue
            CATransaction.registerChange(layer: self, keyPath: "miterLimit", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _strokeColor: CGColor?
    /// The color used to stroke the shape's path. Animatable.
    open var strokeColor: CGColor? {
        get { return _strokeColor }
        set {
            let oldValue = _strokeColor
            _strokeColor = newValue
            CATransaction.registerChange(layer: self, keyPath: "strokeColor", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _strokeStart: CGFloat = 0
    /// The relative location at which to begin stroking the path. Animatable.
    open var strokeStart: CGFloat {
        get { return _strokeStart }
        set {
            let oldValue = _strokeStart
            _strokeStart = max(0, min(1, newValue))
            CATransaction.registerChange(layer: self, keyPath: "strokeStart", oldValue: oldValue, newValue: _strokeStart)
        }
    }

    internal var _strokeEnd: CGFloat = 1
    /// The relative location at which to stop stroking the path. Animatable.
    open var strokeEnd: CGFloat {
        get { return _strokeEnd }
        set {
            let oldValue = _strokeEnd
            _strokeEnd = max(0, min(1, newValue))
            CATransaction.registerChange(layer: self, keyPath: "strokeEnd", oldValue: oldValue, newValue: _strokeEnd)
        }
    }

    // MARK: - Animatable Keys

    /// The list of animatable property keys for CAShapeLayer.
    private static let shapeLayerAnimatableKeys: Set<String> = [
        "path",
        "fillColor",
        "strokeColor",
        "strokeStart",
        "strokeEnd",
        "lineWidth",
        "lineDashPhase",
        "miterLimit"
    ]

    /// Returns the default action for the specified key.
    open override class func defaultAction(forKey event: String) -> (any CAAction)? {
        // First check CAShapeLayer-specific keys
        if shapeLayerAnimatableKeys.contains(event) {
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
