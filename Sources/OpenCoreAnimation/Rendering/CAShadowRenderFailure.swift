import Foundation

/// Describes why a visible layer shadow could not be rendered.
@_spi(RendererDiagnostics)
public enum CAShadowRenderFailure: Error, Equatable, Sendable {
    case nonFiniteGeometry
    case invalidColor
    case rendererResourcesUnavailable
    case shadowPathTessellationFailed
    case rasterizedShadowResourcesUnavailable
    case vertexCapacityExceeded
    case prerenderedShadowUnavailable
    case invalidCompositeOpacity(Float)
    case invalidReplicatorColor(SIMD4<Float>)
    case invalidCompositeViewport
    case compositeColorOverflow
    case compositeStencilPipelineUnavailable
    case compositeRestorationPipelineUnavailable
}
