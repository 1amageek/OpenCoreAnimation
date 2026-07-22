/// Describes why a shape fill or stroke could not be submitted.
@_spi(RendererDiagnostics)
public enum CAShapeRenderFailure: Error, Equatable, Sendable {
    case rendererResourcesUnavailable
    case pathValidationFailed(ShapeFillTessellationError)
    case fillTessellationFailed(ShapeFillTessellationError)
    case strokeTessellationFailed(ShapeStrokeTessellationError)
    case invalidFillColor
    case invalidStrokeColor
    case nonFiniteLineWidth
    case fillVertexCapacityExceeded
    case strokeVertexCapacityExceeded
}
