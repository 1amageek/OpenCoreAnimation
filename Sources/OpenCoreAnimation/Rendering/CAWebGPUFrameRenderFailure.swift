import Foundation

@_spi(RendererDiagnostics)
public enum CACommittedSnapshotEncodingFailure: Error, Equatable, Sendable {
    case solid(CASolidRenderFailure)
    case mask(CAMaskRenderFailure)
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
    case committedSnapshotCaptureFailed(CARendererError)
    case committedSnapshotEncodingFailed(
        CACommittedSnapshotEncodingFailure
    )
    case layerRevisionCaptureFailed(CARendererError)
}
