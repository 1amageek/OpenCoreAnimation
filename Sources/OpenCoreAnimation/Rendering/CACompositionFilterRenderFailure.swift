import Foundation

/// Describes why backdrop filtering or layer composition could not be rendered.
@_spi(RendererDiagnostics)
public enum CACompositionFilterRenderFailure: Error, Equatable, Sendable {
    case backgroundFilterPlanningFailed(CALayerFilterRenderFailure)
    case unsupportedCompositingFilterValue(String)
    case defaultCompositingFilterUnavailable
    case invalidBackdropPrefix
    case unsupportedCompositingFilter(String)
    case sourceCaptureUnavailable
    case clipMaskFailed
    case sourceAdjustmentFailed
    case backdropCaptureIncomplete
    case alphaConversionFailed
    case backgroundFilterExecutionFailed
    case backgroundFilterMaskFailed
    case backgroundFilterMixFailed
    case compositionExecutionFailed
}
