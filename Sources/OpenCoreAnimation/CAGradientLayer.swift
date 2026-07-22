//
//  CAGradientLayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation


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

    /// Specifies the default value associated with a gradient-layer property.
    open override class func defaultValue(forKey key: String) -> Any? {
        switch key {
        case "startPoint":
            return CGPoint(x: 0.5, y: 0)
        case "endPoint":
            return CGPoint(x: 0.5, y: 1)
        case "type":
            return CAGradientLayerType.axial
        default:
            return super.defaultValue(forKey: key)
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
            markDirty(.contents)
            CATransaction.registerChange(layer: self, keyPath: "colors", oldValue: oldValue, newValue: newValue)
        }
    }

    internal var _locations: [CGFloat]?
    /// An optional array of numbers defining the location of each gradient stop. Animatable.
    /// Values should be in the range [0, 1] and monotonically increasing.
    open var locations: [CGFloat]? {
        get { return _locations }
        set {
            let oldValue = _locations
            _locations = newValue
            markDirty(.contents)
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
            markDirty(.geometry)
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
            markDirty(.geometry)
            CATransaction.registerChange(layer: self, keyPath: "endPoint", oldValue: oldValue, newValue: newValue)
        }
    }

    /// Style of gradient drawn by the layer.
    open var type: CAGradientLayerType = .axial {
        didSet {
            guard oldValue != type else { return }
            markDirty(.contents)
        }
    }

    /// Returns whether a gradient-layer property differs from its archive default.
    open override func shouldArchiveValue(forKey key: String) -> Bool {
        switch key {
        case "colors": return colors != nil
        case "locations": return locations != nil
        case "startPoint": return startPoint != CGPoint(x: 0.5, y: 0)
        case "endPoint": return endPoint != CGPoint(x: 0.5, y: 1)
        case "type": return type != .axial
        default: return super.shouldArchiveValue(forKey: key)
        }
    }

}
