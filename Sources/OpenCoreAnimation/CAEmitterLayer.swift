
/// A layer that emits, animates, and renders a particle system.
open class CAEmitterLayer: CALayer {

    // MARK: - Emitter Cells

    /// An array of CAEmitterCell objects that define the types of emitted objects.
    open var emitterCells: [CAEmitterCell]?

    // MARK: - Emitter Geometry

    /// The position of the center of the particle emitter. Animatable.
    open var emitterPosition: CGPoint = .zero

    /// The depth of the particle emitter. Animatable.
    open var emitterZPosition: CGFloat = 0

    /// The size of the particle emitter. Animatable.
    open var emitterSize: CGSize = .zero

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

    /// Defines a multiplier applied to the cell-defined birth rate.
    open var birthRate: Float = 1

    /// Defines a multiplier applied to the cell-defined lifetime.
    open var lifetime: Float = 1

    /// Defines a multiplier applied to the cell-defined velocity.
    open var velocity: Float = 1

    /// Defines a multiplier applied to the cell-defined scale.
    open var scale: Float = 1

    /// Defines a multiplier applied to the cell-defined spin.
    open var spin: Float = 1

    /// Specifies the seed used to initialize the random number generator.
    open var seed: UInt32 = 0
}
