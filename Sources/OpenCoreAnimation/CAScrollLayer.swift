
/// A layer that displays scrollable content larger than its own bounds.
///
/// `CAScrollLayer` is useful for displaying a portion of a larger layer's content.
/// It clips its sublayers to its bounds and allows scrolling through the content.
open class CAScrollLayer: CALayer {

    /// The scroll mode.
    ///
    /// Determines which directions the layer can be scrolled.
    open var scrollMode: CAScrollLayerScrollMode = .both

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
            newOrigin.x = constrainScrollOffset(p.x, contentSize: contentSize.width, visibleSize: bounds.width)
        case .vertically:
            newOrigin.y = constrainScrollOffset(p.y, contentSize: contentSize.height, visibleSize: bounds.height)
        case .both:
            newOrigin.x = constrainScrollOffset(p.x, contentSize: contentSize.width, visibleSize: bounds.width)
            newOrigin.y = constrainScrollOffset(p.y, contentSize: contentSize.height, visibleSize: bounds.height)
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
                contentSize: contentSize.width,
                visibleSize: bounds.width
            )
        case .vertically:
            newOrigin.y = calculateScrollToMakeVisible(
                targetMin: r.minY,
                targetMax: r.maxY,
                visibleMin: visibleRect.minY,
                visibleMax: visibleRect.maxY,
                contentSize: contentSize.height,
                visibleSize: bounds.height
            )
        case .both:
            newOrigin.x = calculateScrollToMakeVisible(
                targetMin: r.minX,
                targetMax: r.maxX,
                visibleMin: visibleRect.minX,
                visibleMax: visibleRect.maxX,
                contentSize: contentSize.width,
                visibleSize: bounds.width
            )
            newOrigin.y = calculateScrollToMakeVisible(
                targetMin: r.minY,
                targetMax: r.maxY,
                visibleMin: visibleRect.minY,
                visibleMax: visibleRect.maxY,
                contentSize: contentSize.height,
                visibleSize: bounds.height
            )
        default:
            newOrigin = r.origin
        }

        bounds.origin = newOrigin
    }

    // MARK: - Private

    /// The total content size based on sublayers.
    private var contentSize: CGSize {
        guard let sublayers = sublayers else { return bounds.size }

        var maxX: CGFloat = 0
        var maxY: CGFloat = 0

        for sublayer in sublayers {
            let frame = sublayer.frame
            maxX = max(maxX, frame.maxX)
            maxY = max(maxY, frame.maxY)
        }

        return CGSize(width: max(maxX, bounds.width), height: max(maxY, bounds.height))
    }

    /// Constrains a scroll offset to valid bounds.
    private func constrainScrollOffset(_ offset: CGFloat, contentSize: CGFloat, visibleSize: CGFloat) -> CGFloat {
        let maxOffset = max(0, contentSize - visibleSize)
        return max(0, min(offset, maxOffset))
    }

    /// Calculates the scroll offset needed to make a target range visible.
    private func calculateScrollToMakeVisible(
        targetMin: CGFloat,
        targetMax: CGFloat,
        visibleMin: CGFloat,
        visibleMax: CGFloat,
        contentSize: CGFloat,
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

        return constrainScrollOffset(newOffset, contentSize: contentSize, visibleSize: visibleSize)
    }
}
