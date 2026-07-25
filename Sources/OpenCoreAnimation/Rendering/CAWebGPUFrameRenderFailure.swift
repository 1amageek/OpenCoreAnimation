import Foundation

@_spi(RendererDiagnostics)
public enum CACommittedSnapshotEncodingFailure: Error, Equatable, Sendable {
    case solid(CASolidRenderFailure)
    case mask(CAMaskRenderFailure)
    case contents(CAContentsRenderFailure)
    case contentMask(CALayerFilterRenderFailure)
    case groupOpacity(CALayerFilterRenderFailure)
    case rasterization(CARasterizationRenderFailure)
    case filter(CALayerFilterRenderFailure)
    case backdropComposition(CACompositionFilterRenderFailure)
    case shadow(CAShadowRenderFailure)
    case gradient(CAGradientRenderFailure)
    case shape(CAShapeRenderFailure)
}

/// Describes why a WebGPU frame could not begin rendering.
@_spi(RendererDiagnostics)
public enum CAWebGPUFrameRenderFailure: Error, Equatable, Sendable {
    case invalidRenderTarget(CARenderTargetConfigurationError)
    case deviceUnavailable
    case contextUnavailable
    case canvasConfigurationFailed
    case basePipelineUnavailable
    case baseBindGroupUnavailable
    case depthTextureUnavailable
    case depthTextureViewUnavailable
    case layerFilterProcessorUnavailable
    case rasterizationCacheUnavailable
    case delegateBackingStoreFailed(CADelegateBackingStoreError)
    case contentMaskPreparationFailed(CAContentMaskPreparationFailure)
    case committedSnapshotCaptureFailed(CARendererError)
    case committedSnapshotEncodingFailed(
        CACommittedSnapshotEncodingFailure
    )
    case layerRevisionCaptureFailed(CARendererError)
}
