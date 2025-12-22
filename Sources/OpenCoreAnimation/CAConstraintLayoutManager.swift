//
//  CAConstraintLayoutManager.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics


/// An object that provides a constraint-based layout manager.
open class CAConstraintLayoutManager: CALayoutManager {

    /// Returns the shared constraint layout manager.
    public static func shared() -> CAConstraintLayoutManager {
        return _shared
    }

    nonisolated(unsafe) private static let _shared = CAConstraintLayoutManager()

    public init() {}

    // MARK: - CALayoutManager

    public func invalidateLayout(of layer: CALayer) {
        layer.setNeedsLayout()
    }

    public func layoutSublayers(of layer: CALayer) {
        guard let sublayers = layer.sublayers else { return }

        // Each sublayer has its own constraints that define how it's positioned
        // relative to its siblings or superlayer
        for sublayer in sublayers {
            guard let constraints = sublayer.constraints else { continue }

            for constraint in constraints {
                // Find the source layer by name
                let sourceLayer: CALayer?
                if constraint.sourceName == "superlayer" {
                    sourceLayer = layer
                } else {
                    sourceLayer = sublayers.first(where: { $0.name == constraint.sourceName })
                }

                guard let source = sourceLayer else { continue }

                // Apply the constraint to this sublayer
                applyConstraint(constraint, to: sublayer, relativeTo: source)
            }
        }
    }

    public func preferredSize(of layer: CALayer) -> CGSize {
        return layer.bounds.size
    }

    // MARK: - Private

    private func applyConstraint(_ constraint: CAConstraint, to layer: CALayer, relativeTo source: CALayer) {
        let sourceValue = getValue(for: constraint.sourceAttribute, from: source)
        let targetValue = sourceValue * constraint.scale + constraint.offset

        setValue(targetValue, for: constraint.attribute, on: layer)
    }

    /// Gets the value for a constraint attribute from a layer.
    ///
    /// Uses model values (bounds/position) rather than frame to ensure constraints
    /// work correctly regardless of transforms.
    private func getValue(for attribute: CAConstraintAttribute, from layer: CALayer) -> CGFloat {
        // Calculate the untransformed frame from bounds, position, and anchorPoint
        let width = layer.bounds.width
        let height = layer.bounds.height
        let minX = layer.position.x - width * layer.anchorPoint.x
        let minY = layer.position.y - height * layer.anchorPoint.y

        switch attribute {
        case .minX:
            return minX
        case .midX:
            return minX + width / 2
        case .maxX:
            return minX + width
        case .width:
            return width
        case .minY:
            return minY
        case .midY:
            return minY + height / 2
        case .maxY:
            return minY + height
        case .height:
            return height
        }
    }

    /// Sets the value for a constraint attribute on a layer.
    ///
    /// Modifies bounds and position directly rather than frame to ensure constraints
    /// work correctly regardless of transforms.
    private func setValue(_ value: CGFloat, for attribute: CAConstraintAttribute, on layer: CALayer) {
        let width = layer.bounds.width
        let height = layer.bounds.height

        switch attribute {
        case .minX:
            // minX = position.x - width * anchorPoint.x
            // position.x = minX + width * anchorPoint.x
            layer.position.x = value + width * layer.anchorPoint.x
        case .midX:
            // midX = position.x - width * anchorPoint.x + width / 2
            // For center, this simplifies when anchorPoint.x = 0.5
            layer.position.x = value + width * (layer.anchorPoint.x - 0.5)
        case .maxX:
            // maxX = position.x - width * anchorPoint.x + width
            // position.x = maxX - width + width * anchorPoint.x = maxX - width * (1 - anchorPoint.x)
            layer.position.x = value - width * (1 - layer.anchorPoint.x)
        case .width:
            var bounds = layer.bounds
            bounds.size.width = value
            layer.bounds = bounds
        case .minY:
            layer.position.y = value + height * layer.anchorPoint.y
        case .midY:
            layer.position.y = value + height * (layer.anchorPoint.y - 0.5)
        case .maxY:
            layer.position.y = value - height * (1 - layer.anchorPoint.y)
        case .height:
            var bounds = layer.bounds
            bounds.size.height = value
            layer.bounds = bounds
        }
    }
}
