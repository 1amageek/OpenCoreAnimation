import Foundation

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
}
