import Foundation

@_spi(RendererDiagnostics)
public enum CAMaskRenderContext: String, Equatable, Sendable {
    case contentMask
    case roundedClip
    case activeStencil
    case restoration
}

/// Describes why a stencil mask or rounded clip could not be rendered safely.
@_spi(RendererDiagnostics)
public enum CAMaskRenderFailure: Error, Equatable, Sendable {
    case resourcesUnavailable(CAMaskRenderContext)
    case invalidGeometry(CAMaskRenderContext, CGRect)
    case invalidCornerGeometry(
        CAMaskRenderContext,
        radius: Float,
        exponent: Float,
        radii: SIMD4<Float>
    )
    case invalidTransform(CAMaskRenderContext)
    case unsupportedCornerCurve(CAMaskRenderContext, String)
    case vertexCapacityExceeded(CAMaskRenderContext)
    case stencilReferenceOverflow(CAMaskRenderContext)
    case invalidStencilState(depth: Int, reference: UInt32)
    case pipelineUnavailable(CAMaskRenderContext)
}
