/// Describes why a gradient could not be submitted to the renderer.
@_spi(RendererDiagnostics)
public enum CAGradientRenderFailure: Error, Equatable, Sendable {
    case rendererResourcesUnavailable
    case invalidConfiguration(GradientRenderConfigurationError)
    case stopByteCountOverflow(colorCount: Int)
    case stopCapacityOverflow(byteOffset: UInt64, byteCount: UInt64)
    case stopBufferFailure(GradientStopBufferPoolError)
    case stopOffsetOutOfRange(UInt64)
    case vertexCapacityExceeded
}
