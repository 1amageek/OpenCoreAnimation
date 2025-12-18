
/// Methods your app can implement to respond when animations start and stop.
public protocol CAAnimationDelegate: AnyObject {
    /// Tells the delegate the animation has started.
    ///
    /// - Parameter anim: The animation that has started.
    func animationDidStart(_ anim: CAAnimation)

    /// Tells the delegate the animation has ended.
    ///
    /// - Parameters:
    ///   - anim: The animation that has ended.
    ///   - flag: A Boolean value that indicates whether the animation ran to completion
    ///           before it stopped. The value is true if the animation ran to completion
    ///           before it stopped or false if it did not.
    func animationDidStop(_ anim: CAAnimation, finished flag: Bool)
}

// Default implementations - all methods are optional
public extension CAAnimationDelegate {
    func animationDidStart(_ anim: CAAnimation) {}
    func animationDidStop(_ anim: CAAnimation, finished flag: Bool) {}
}
