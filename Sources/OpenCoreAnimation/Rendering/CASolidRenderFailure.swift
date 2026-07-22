import Foundation

/// Describes why a background or border solid quad could not be rendered.
@_spi(RendererDiagnostics)
public enum CASolidRenderFailure: Error, Equatable, Sendable {
    case resourcesUnavailable(CASolidRenderContext)
    case invalidGeometry(CASolidRenderContext, CGRect)
    case invalidCornerGeometry(CASolidRenderContext)
    case invalidColor(CASolidRenderContext, SIMD4<Float>)
    case invalidOpacity(CASolidRenderContext, Float)
    case invalidBorderWidth(Float)
    case invalidTransform(CASolidRenderContext)
    case vertexCapacityExceeded(CASolidRenderContext)
    case pipelineUnavailable(CASolidRenderContext)
}
