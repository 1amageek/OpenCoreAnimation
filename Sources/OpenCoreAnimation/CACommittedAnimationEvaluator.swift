import Synchronization

/// Owns a model-independent animation tree behind one synchronization
/// boundary and produces value-only render snapshots for individual frames.
internal final class CACommittedAnimationEvaluator: Sendable {
    private let rootLayer: Mutex<CALayer>

    internal init(
        rootLayer: CALayer,
        frameToken: UInt64
    ) throws(CARendererError) {
        self.rootLayer = Mutex(
            try rootLayer
                .makeCommittedAnimationEvaluatorCopy(
                    frameToken: frameToken
                )
        )
    }

    internal func snapshot(
        frameToken: UInt64
    ) throws(CARendererError) -> CARenderSnapshot {
        try rootLayer.withLock {
            (
                rootLayer: inout CALayer
            ) throws(CARendererError) -> CARenderSnapshot in
            try CARenderSnapshot.capture(
                rootLayer,
                frameToken: frameToken
            )
        }
    }
}
