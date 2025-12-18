
/// Methods that allow an object to manage the layout of a layer and its sublayers.
public protocol CALayoutManager: AnyObject {
    /// Invalidates the layout of a layer so it knows to refresh its content on the next frame.
    func invalidateLayout(of layer: CALayer)

    /// Override to customize layout of sublayers whenever the layer needs redrawing.
    func layoutSublayers(of layer: CALayer)

    /// Override to customize layer size.
    func preferredSize(of layer: CALayer) -> CGSize
}

// Default implementations
public extension CALayoutManager {
    func invalidateLayout(of layer: CALayer) {}
    func preferredSize(of layer: CALayer) -> CGSize { return .zero }
}
