import Foundation

/// Render-time inputs for compositing the layer states on both sides of a transition.
internal struct CATransitionRenderState {
    internal let sourceLayer: CALayer
    internal let type: CATransitionType
    internal let subtype: CATransitionSubtype?
    internal let progress: CFTimeInterval
}
