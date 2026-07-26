import Foundation

/// Render-time inputs for compositing the layer states on both sides of a transition.
internal struct CATransitionRenderState {
    internal let resourceIdentity: UInt64
    internal let sourceLayer: CALayer?
    internal let committedSourceSnapshot: CARenderSnapshot?
    internal let type: CATransitionType
    internal let subtype: CATransitionSubtype?
    internal let filter: Any?
    internal let committedFilterSnapshot:
        CARenderSnapshotTransition.Filter?
    internal let committedFilterCaptureFailure:
        CATransitionRenderFailure?
    internal let usesCommittedFilterSnapshot: Bool
    internal let progress: CFTimeInterval

    internal init(
        resourceIdentity: UInt64,
        sourceLayer: CALayer? = nil,
        committedSourceSnapshot: CARenderSnapshot? = nil,
        type: CATransitionType,
        subtype: CATransitionSubtype?,
        filter: Any?,
        committedFilterSnapshot:
            CARenderSnapshotTransition.Filter? = nil,
        committedFilterCaptureFailure:
            CATransitionRenderFailure? = nil,
        usesCommittedFilterSnapshot: Bool = false,
        progress: CFTimeInterval
    ) {
        self.resourceIdentity = resourceIdentity
        self.sourceLayer = sourceLayer
        self.committedSourceSnapshot = committedSourceSnapshot
        self.type = type
        self.subtype = subtype
        self.filter = filter
        self.committedFilterSnapshot =
            committedFilterSnapshot
        self.committedFilterCaptureFailure =
            committedFilterCaptureFailure
        self.usesCommittedFilterSnapshot =
            usesCommittedFilterSnapshot
        self.progress = progress
    }

    internal func resolvedFilterSnapshot()
        throws(CATransitionRenderFailure)
        -> CARenderSnapshotTransition.Filter? {
        if usesCommittedFilterSnapshot {
            if let committedFilterCaptureFailure {
                throw committedFilterCaptureFailure
            }
            return committedFilterSnapshot
        }
        return try CARenderSnapshotTransition.Filter.capture(
            filter
        )
    }
}
