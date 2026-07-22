import Foundation

/// Describes why a requested layer-filter chain could not be rendered.
@_spi(RendererDiagnostics)
public enum CALayerFilterRenderFailure: Error, Equatable, Sendable {
    case invalidConfiguration(CAFilterConfigurationError)
    case unsupportedFilterValue(String)
    case unavailableCoreImageFilter(String)
    case rendererResourcesUnavailable
    case alphaConversionFailed
    case rendererOperationFailed
    case coreImageProcessorUnavailable
    case coreImageExecutionFailed
    case contentMaskUnavailable
    case contentMaskCaptureFailed
    case contentMaskCompositeFailed
    case invalidCompositeOpacity(Float)
    case invalidCompositeColor(SIMD4<Float>)
    case compositeResourcesUnavailable
    case compositeStencilPipelineUnavailable
    case compositeRestorationPipelineUnavailable
}
