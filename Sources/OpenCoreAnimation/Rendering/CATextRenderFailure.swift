import Foundation

/// Describes why a text layer could not enter or complete the renderer's text pipeline.
@_spi(RendererDiagnostics)
public enum CATextRenderFailure: Error, Equatable, Sendable {
    case unsupportedStringValue
    case unsupportedFontValue
    case invalidFontSize
    case invalidContentsScale
    case invalidBounds
    case invalidForegroundColor
    case unsupportedAlignmentMode(String)
    case unsupportedTruncationMode(String)
    case invalidOpacity(Float)
    case invalidReplicatorColor(SIMD4<Float>)
    case invalidCornerGeometry
    case invalidTransform
    case rendererResourcesUnavailable
    case canvas2DUnavailable
    case textMeasurementUnavailable
    case textureDimensionsUnsupported
    case imageDataUnavailable
    case imageDataStorageUnavailable
    case vertexCapacityExceeded
}
