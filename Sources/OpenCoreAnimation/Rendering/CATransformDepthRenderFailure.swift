import Foundation

/// Describes why a transform-layer depth group could not be rendered safely.
@_spi(RendererDiagnostics)
public enum CATransformDepthRenderFailure: Error, Equatable, Sendable {
    case invalidNestingDepth(Int)
    case nestingDepthOverflow
    case depthClearPipelineUnavailable
    case invalidProjectedDepth(sublayerIndex: Int, reason: CAProjectedDepthError)
}
