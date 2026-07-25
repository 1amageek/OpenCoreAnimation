/// Describes why layer contents could not be submitted to the renderer.
@_spi(RendererDiagnostics)
public enum CAContentsRenderFailure: Error, Equatable, Sendable {
    case rendererResourcesUnavailable
    case textureManagerUnavailable
    case nineSliceConfiguration(ContentsRenderConfigurationError)
    case standardConfiguration(ContentsRenderConfigurationError)
    case imageConversion(CAImageContentsConversionError)
    case invalidSamplingFilters(
        magnification: CALayerContentsFilter,
        minification: CALayerContentsFilter
    )
    case invalidMinificationFilterBias(Float)
    case textureCreationFailed
    case invalidTransform
    case nineSliceVertexCapacityExceeded
    case standardVertexCapacityExceeded
}
