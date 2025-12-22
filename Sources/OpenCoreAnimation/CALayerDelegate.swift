// CGImageMetadataTag.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework

import Foundation
import OpenCoreGraphics

/// Methods your app can implement to respond to layer-related events.
///
/// You can implement the methods of this protocol to provide the layer's content, handle the layout of sublayers,
/// and provide custom animation actions to perform. The object that implements this protocol must be assigned
/// to the delegate property of the layer object.
public protocol CALayerDelegate: AnyObject {
    /// Tells the delegate to implement the display process.
    func display(_ layer: CALayer)

    /// Tells the delegate to implement the display process using the layer's context.
    func draw(_ layer: CALayer, in ctx: CGContext)

    /// Notifies the delegate of an imminent draw.
    func layerWillDraw(_ layer: CALayer)

    /// Tells the delegate a layer's bounds have changed.
    func layoutSublayers(of layer: CALayer)

    /// Returns the default action of the action(forKey:) method.
    func action(for layer: CALayer, forKey event: String) -> (any CAAction)?
}

// Default implementations - all methods are optional
public extension CALayerDelegate {
    func display(_ layer: CALayer) {}
    func draw(_ layer: CALayer, in ctx: CGContext) {}
    func layerWillDraw(_ layer: CALayer) {}
    func layoutSublayers(of layer: CALayer) {}
    func action(for layer: CALayer, forKey event: String) -> (any CAAction)? { return nil }
}
