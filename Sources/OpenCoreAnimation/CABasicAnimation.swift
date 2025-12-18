
/// An object that provides basic, single-keyframe animation capabilities for a layer property.
open class CABasicAnimation: CAPropertyAnimation {

    /// The value at the start of the animation.
    open var fromValue: Any?

    /// The value at the end of the animation.
    open var toValue: Any?

    /// The value at which the animation will interpolate between.
    open var byValue: Any?
}
