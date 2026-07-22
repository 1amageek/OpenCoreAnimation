@_spi(RendererDiagnostics)
public enum ShapeStrokeTessellationError: Error, Equatable, Sendable {
    case invalidGeometry
    case invalidDashPattern
    case unsupportedLineCap(String)
    case unsupportedLineJoin(String)
}
