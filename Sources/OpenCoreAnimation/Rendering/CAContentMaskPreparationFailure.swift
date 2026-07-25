import Foundation

/// Describes why a masked subtree could not be prepared for compositing.
@_spi(RendererDiagnostics)
public enum CAContentMaskPreparationFailure: Error, Equatable, Sendable {
    case rasterization(CARasterizationRenderFailure)
    case layerFilter(CALayerFilterRenderFailure)
    case shadow(CAShadowRenderFailure)
}
