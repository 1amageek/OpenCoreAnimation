import Foundation

extension CALayer {
    internal var supportsDelegateBackingStore: Bool {
        !(self is CATiledLayer)
            && !(self is CATransformLayer)
            && !(self is CAEmitterLayer)
            && !(self is CATextLayer)
            && !(self is CAShapeLayer)
            && !(self is CAGradientLayer)
    }

    internal func prepareDelegateBackingStore(
        maximumPixelDimension: Int
    ) throws(CADelegateBackingStoreError) {
        guard supportsDelegateBackingStore else { return }
        guard let invalidation = pendingDisplayInvalidation else { return }
        guard let delegate else {
            delegateBackingStore = nil
            return
        }

        let contentsAssignmentBeforeDisplay = _contentsAssignmentGeneration
        displayIfNeeded()
        if _contentsAssignmentGeneration != contentsAssignmentBeforeDisplay {
            delegateBackingStore = nil
            return
        }

        do {
            delegateBackingStore = try CADelegateBackingStore.render(
                layer: self,
                delegate: delegate,
                invalidation: invalidation,
                previous: delegateBackingStore,
                maximumPixelDimension: maximumPixelDimension
            )
        } catch {
            delegateBackingStore = nil
            restorePendingDisplayInvalidation(invalidation)
            throw error
        }
    }
}
