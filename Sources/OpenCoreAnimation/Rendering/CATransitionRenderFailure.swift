import Foundation

public enum CATransitionParticipantRole: String, Equatable, Sendable {
    case source
    case target
}

public enum CATransitionParticipantSnapshotStage:
    String, Equatable, Sendable {
    case solid
    case mask
    case contents
    case contentMask
    case groupOpacity
    case rasterization
    case filter
    case backdropComposition
    case shadow
    case gradient
    case shape
    case text
    case transformDepth
    case replicator
    case emitter
    case tiled
    case transition
}

/// Describes why a transition capture or filter could not be rendered.
public enum CATransitionRenderFailure: Error, Equatable, Sendable {
    case unsupportedFilterValue(String)
    case filterSnapshotCaptureFailed(CARenderSnapshotFilterError)
    case filterProcessorUnavailable
    case unsupportedFilter(String)
    case unsupportedTransitionType(String)
    case unsupportedTransitionSubtype(String)
    case invalidParticipantBounds(CATransitionParticipantRole, CGRect)
    case invalidParticipantContentsScale(CATransitionParticipantRole, CGFloat)
    case invalidParticipantPixelSize(CATransitionParticipantRole, CGSize)
    case participantProjectionOutOfRange(CATransitionParticipantRole, CGRect)
    case participantReplicatorFailed(
        CATransitionParticipantRole,
        CAReplicatorRenderFailure
    )
    case participantSnapshotEncodingFailed(
        CATransitionParticipantRole,
        stage: CATransitionParticipantSnapshotStage,
        reason: String
    )
    case filterExecutionCreationFailed(String)
    case invalidProgress(CFTimeInterval)
    case filterDispatchFailed(String)
    case compositeResourcesUnavailable
    case invalidCompositeBounds(CGRect)
    case invalidCompositeOffset(CGPoint)
    case invalidCompositeOpacity(Float)
    case invalidCompositeTransform
    case compositeVertexCapacityExceeded(Int)
    case compositePipelineUnavailable
}
