//
//  CAScrollLayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation


/// A layer that displays scrollable content larger than its own bounds.
///
/// `CAScrollLayer` is useful for displaying a portion of a larger layer's content.
/// It clips its sublayers to its bounds and allows scrolling through the content.
open class CAScrollLayer: CALayer {

    // MARK: - Initialization

    public required init() {
        super.init()
    }

    /// Initializes a new scroll layer as a copy of the specified layer.
    public required init(layer: Any) {
        super.init(layer: layer)
        if let scrollLayer = layer as? CAScrollLayer {
            self.scrollMode = scrollLayer.scrollMode
        }
    }

    /// Specifies the default value associated with a scroll-layer property.
    open override class func defaultValue(forKey key: String) -> Any? {
        switch key {
        case "scrollMode":
            return CAScrollLayerScrollMode.both
        default:
            return super.defaultValue(forKey: key)
        }
    }

    // MARK: - Scroll Properties

    /// The scroll mode.
    ///
    /// Determines which directions the layer can be scrolled.
    open var scrollMode: CAScrollLayerScrollMode = .both {
        didSet { markDirty(.geometry) }
    }

    /// Scroll to the specified point.
    ///
    /// This method adjusts the layer's bounds origin to scroll the content.
    /// The scrolling is constrained by the `scrollMode` property.
    ///
    /// - Parameter p: The point to scroll to.
    open func scroll(to p: CGPoint) {
        var newOrigin = bounds.origin

        switch scrollMode {
        case .none:
            return
        case .horizontally:
            newOrigin.x = p.x
        case .vertically:
            newOrigin.y = p.y
        case .both:
            newOrigin = p
        default:
            newOrigin = p
        }

        bounds.origin = newOrigin
    }

    /// Scroll to make the specified rectangle visible.
    ///
    /// This method adjusts the scroll position to make as much of the rectangle
    /// visible as possible. The scrolling is constrained by the `scrollMode` property.
    ///
    /// - Parameter r: The rectangle to make visible.
    open func scroll(to r: CGRect) {
        var newOrigin = bounds.origin
        let visibleRect = CGRect(origin: bounds.origin, size: bounds.size)

        // Calculate the minimum scroll needed to make the rectangle visible
        switch scrollMode {
        case .none:
            return
        case .horizontally:
            newOrigin.x = calculateScrollToMakeVisible(
                targetMin: r.minX,
                targetMax: r.maxX,
                visibleMin: visibleRect.minX,
                visibleMax: visibleRect.maxX,
                visibleSize: bounds.width
            )
        case .vertically:
            newOrigin.y = calculateScrollToMakeVisible(
                targetMin: r.minY,
                targetMax: r.maxY,
                visibleMin: visibleRect.minY,
                visibleMax: visibleRect.maxY,
                visibleSize: bounds.height
            )
        case .both:
            newOrigin.x = calculateScrollToMakeVisible(
                targetMin: r.minX,
                targetMax: r.maxX,
                visibleMin: visibleRect.minX,
                visibleMax: visibleRect.maxX,
                visibleSize: bounds.width
            )
            newOrigin.y = calculateScrollToMakeVisible(
                targetMin: r.minY,
                targetMax: r.maxY,
                visibleMin: visibleRect.minY,
                visibleMax: visibleRect.maxY,
                visibleSize: bounds.height
            )
        default:
            newOrigin = r.origin
        }

        bounds.origin = newOrigin
    }

    // MARK: - Private

    /// Calculates the scroll offset needed to make a target range visible.
    private func calculateScrollToMakeVisible(
        targetMin: CGFloat,
        targetMax: CGFloat,
        visibleMin: CGFloat,
        visibleMax: CGFloat,
        visibleSize: CGFloat
    ) -> CGFloat {
        var newOffset = visibleMin

        // If target is completely visible, don't scroll
        if targetMin >= visibleMin && targetMax <= visibleMax {
            return visibleMin
        }

        // If target is larger than visible area, scroll to show the start
        if targetMax - targetMin >= visibleSize {
            newOffset = targetMin
        }
        // If target is above/left of visible area, scroll to show it
        else if targetMin < visibleMin {
            newOffset = targetMin
        }
        // If target is below/right of visible area, scroll to show it
        else if targetMax > visibleMax {
            newOffset = targetMax - visibleSize
        }

        return newOffset
    }
}
