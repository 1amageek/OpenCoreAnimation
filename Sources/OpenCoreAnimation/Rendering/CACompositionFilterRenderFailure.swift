import Foundation

/// Describes why backdrop filtering or layer composition could not be rendered.
@_spi(RendererDiagnostics)
public enum CACompositionFilterRenderFailure: Error, Equatable, Sendable {
    case backgroundFilterPlanningFailed(CALayerFilterRenderFailure)
    case contentMaskFilterPlanningFailed(CALayerFilterRenderFailure)
    case unsupportedCompositingFilterValue(String)
    case defaultCompositingFilterUnavailable
    case invalidBackdropPrefix
    case unsupportedCompositingFilter(String)
    case sourceCaptureUnavailable
    case clipMaskFailed
    case sourceAdjustmentFailed
    case backdropCaptureIncomplete
    case backdropReplicatorFailed(CAReplicatorRenderFailure)
    case alphaConversionFailed
    case backgroundFilterExecutionFailed(CALayerFilterRenderFailure)
    case contentMaskFilterExecutionFailed(CALayerFilterRenderFailure)
    case backgroundFilterMaskFailed
    case backgroundFilterMixFailed
    case compositionExecutionFailed
    case displayResourcesUnavailable
    case invalidDisplayGeometry
    case invalidDisplayTransform
    case invalidSamplingTransform
    case displayVertexCapacityExceeded
}
