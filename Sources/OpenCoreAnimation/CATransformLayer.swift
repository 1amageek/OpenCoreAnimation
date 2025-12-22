//
//  CATransformLayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics


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

    /// Transform layers do not render their content, so hitTest always returns nil.
    open override func hitTest(_ p: CGPoint) -> CALayer? {
        return nil
    }
}
