@_spi(RendererDiagnostics)
public enum GradientStopBufferPoolError: Error, Equatable, Sendable {
    case capacityExceeded(required: UInt64, maximum: UInt64)
}
