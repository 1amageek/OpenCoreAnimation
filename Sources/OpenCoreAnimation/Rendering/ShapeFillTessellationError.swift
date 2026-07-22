@_spi(RendererDiagnostics)
public enum ShapeFillTessellationError: Error, Equatable, Sendable {
    case unsupportedFillRule(String)
    case nonFinitePath
}
