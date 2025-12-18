
/// A layer that creates a specified number of sublayer copies with varying geometric,
/// temporal, and color transformations.
open class CAReplicatorLayer: CALayer {

    /// The number of copies to create, including the source layer.
    open var instanceCount: Int = 1

    /// Specifies whether this layer flattens its sublayers into its plane.
    open var preservesDepth: Bool = false

    /// The delay, in seconds, between replicated copies. Animatable.
    open var instanceDelay: CFTimeInterval = 0

    /// The transform matrix applied to the previous instance to produce the current instance. Animatable.
    open var instanceTransform: CATransform3D = CATransform3DIdentity

    /// The color used to multiply the source object. Animatable.
    open var instanceColor: CGColor?

    /// Defines the offset added to the red component of the color for each replicated instance. Animatable.
    open var instanceRedOffset: Float = 0

    /// Defines the offset added to the green component of the color for each replicated instance. Animatable.
    open var instanceGreenOffset: Float = 0

    /// Defines the offset added to the blue component of the color for each replicated instance. Animatable.
    open var instanceBlueOffset: Float = 0

    /// Defines the offset added to the alpha component of the color for each replicated instance. Animatable.
    open var instanceAlphaOffset: Float = 0
}
