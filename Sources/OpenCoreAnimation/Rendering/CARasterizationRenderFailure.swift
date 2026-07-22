import Foundation

/// Describes why a rasterization capture or composite could not be rendered.
@_spi(RendererDiagnostics)
public enum CARasterizationRenderFailure: Error, Equatable, Sendable {
    case invalidCaptureBounds
    case invalidRasterizationScale(CGFloat)
    case invalidScaledExtent(CGSize)
    case captureProjectionOutOfRange(CGRect)
    case compositeResourcesUnavailable
    case invalidCompositeBounds(CGRect)
    case compositeVertexCapacityExceeded
    case compositePipelineUnavailable
}
