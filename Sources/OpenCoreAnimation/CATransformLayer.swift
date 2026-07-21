//
//  CATransformLayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation


/// Objects used to create true 3D layer hierarchies, rather than the flattened hierarchy
/// rendering model used by other layer types.
///
/// Unlike normal layers, transform layers don't flatten their sublayers into the plane
/// at z = 0. This makes them useful for constructing objects that have 3D depth.
open class CATransformLayer: CALayer {

    // MARK: - Initialization

    public required init() {
        super.init()
    }

    /// Initializes a new transform layer as a copy of the specified layer.
    public required init(layer: Any) {
        super.init(layer: layer)
        // CATransformLayer has no additional properties to copy
    }

    // MARK: - Hit Testing

    /// Transform layers do not render their own content, but forward hit testing
    /// to sublayers. A CATransformLayer never returns itself from hitTest.
    open override func hitTest(_ p: CGPoint) -> CALayer? {
        guard !isHidden, opacity > 0 else { return nil }

        // `p` is expressed in this layer's superlayer space. Convert it once to
        // this transform layer's local space; each child then performs its own
        // single superlayer-to-local conversion.
        let localPoint = convert(p, from: superlayer)
        for sublayer in sortedSublayers().reversed() {
            if let hit = sublayer.hitTest(localPoint) {
                return hit
            }
        }

        // Never return self for CATransformLayer
        return nil
    }
}
