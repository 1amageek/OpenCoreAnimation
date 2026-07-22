/// Describes which renderer path rejected an unsupported corner curve.
@_spi(RendererDiagnostics)
public enum CACornerCurveRenderFailure: Error, Equatable, Sendable {
    case layer(CornerCurveRenderConfigurationError)
    case mask(CornerCurveRenderConfigurationError)
    case roundedClip(CornerCurveRenderConfigurationError)
}
