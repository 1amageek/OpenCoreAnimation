
/// Objects used to create true 3D layer hierarchies, rather than the flattened hierarchy
/// rendering model used by other layer types.
///
/// Unlike normal layers, transform layers don't flatten their sublayers into the plane
/// at z = 0. This makes them useful for constructing objects that have 3D depth.
open class CATransformLayer: CALayer {

    /// Transform layers do not render their content, so hitTest always returns nil.
    open override func hitTest(_ p: CGPoint) -> CALayer? {
        return nil
    }
}
