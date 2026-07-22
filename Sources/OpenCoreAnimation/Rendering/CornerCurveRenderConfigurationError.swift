@_spi(RendererDiagnostics)
public enum CornerCurveRenderConfigurationError: Error, Equatable, Sendable {
    case unsupportedCurve(String)
}
