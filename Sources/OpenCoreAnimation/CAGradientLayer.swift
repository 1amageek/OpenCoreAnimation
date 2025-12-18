
/// A layer that draws a color gradient over its background color, filling the shape of the layer.
open class CAGradientLayer: CALayer {

    // MARK: - Gradient Properties

    /// An array of CGColor objects defining the color of each gradient stop. Animatable.
    open var colors: [Any]?

    /// An optional array of numbers defining the location of each gradient stop. Animatable.
    /// Values should be in the range [0, 1] and monotonically increasing.
    open var locations: [Float]?

    /// The start point of the gradient when drawn in the layer's coordinate space. Animatable.
    open var startPoint: CGPoint = CGPoint(x: 0.5, y: 0)

    /// The end point of the gradient when drawn in the layer's coordinate space. Animatable.
    open var endPoint: CGPoint = CGPoint(x: 0.5, y: 1)

    /// Style of gradient drawn by the layer.
    open var type: CAGradientLayerType = .axial
}
