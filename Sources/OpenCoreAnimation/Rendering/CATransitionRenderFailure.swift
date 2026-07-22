import Foundation

@_spi(RendererDiagnostics)
public enum CATransitionParticipantRole: String, Equatable, Sendable {
    case source
    case target
}

/// Describes why a transition capture or filter could not be rendered.
@_spi(RendererDiagnostics)
public enum CATransitionRenderFailure: Error, Equatable, Sendable {
    case unsupportedFilterValue(String)
    case filterProcessorUnavailable
    case unsupportedFilter(String)
    case unsupportedTransitionType(String)
    case unsupportedTransitionSubtype(String)
    case invalidParticipantBounds(CATransitionParticipantRole, CGRect)
    case invalidParticipantContentsScale(CATransitionParticipantRole, CGFloat)
    case invalidParticipantPixelSize(CATransitionParticipantRole, CGSize)
    case participantProjectionOutOfRange(CATransitionParticipantRole, CGRect)
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
