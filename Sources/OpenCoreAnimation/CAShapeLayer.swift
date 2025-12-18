
/// A layer that draws a cubic Bezier spline in its coordinate space.
///
/// The shape is composited between the layer's contents and its first sublayer.
/// The shape will be drawn antialiased, and whenever possible it will be mapped into screen space
/// before being rasterized to preserve resolution independence.
open class CAShapeLayer: CALayer {

    // MARK: - Specifying the Shape Path

    /// The path defining the shape to be rendered. Animatable.
    open var path: CGPath?

    // MARK: - Accessing Shape Style Properties

    /// The color used to fill the shape's path. Animatable.
    open var fillColor: CGColor?

    /// The fill rule used when filling the shape's path.
    open var fillRule: CAShapeLayerFillRule = .nonZero

    /// Specifies the line cap style for the shape's path.
    open var lineCap: CAShapeLayerLineCap = .butt

    /// The dash pattern applied to the shape's path when stroked.
    /// Each element specifies the length of a dash or gap in the pattern.
    open var lineDashPattern: [CGFloat]?

    /// The dash phase applied to the shape's path when stroked. Animatable.
    open var lineDashPhase: CGFloat = 0

    /// Specifies the line join style for the shape's path.
    open var lineJoin: CAShapeLayerLineJoin = .miter

    /// Specifies the line width of the shape's path. Animatable.
    open var lineWidth: CGFloat = 1

    /// The miter limit used when stroking the shape's path. Animatable.
    open var miterLimit: CGFloat = 10

    /// The color used to stroke the shape's path. Animatable.
    open var strokeColor: CGColor?

    /// The relative location at which to begin stroking the path. Animatable.
    open var strokeStart: CGFloat = 0

    /// The relative location at which to stop stroking the path. Animatable.
    open var strokeEnd: CGFloat = 1
}
