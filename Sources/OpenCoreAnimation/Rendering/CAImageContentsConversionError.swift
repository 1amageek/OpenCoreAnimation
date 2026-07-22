import Foundation

/// Describes why a `CGImage` could not be normalized for GPU texture upload.
public enum CAImageContentsConversionError: Error, Equatable, Sendable {
    case invalidDimensions(width: Int, height: Int)
    case dimensionsExceedTextureLimit(width: Int, height: Int, maximum: Int)
    case unsupportedPixelLayout(bitsPerComponent: Int, bitsPerPixel: Int)
    case invalidBytesPerRow(minimum: Int, actual: Int)
    case pixelStorageOverflow
    case missingPixelData
    case insufficientPixelData(required: Int, actual: Int)
    case missingColorSpace
    case unsupportedColorSpace
    case nonFinitePixelComponent(pixelIndex: Int, componentIndex: Int)
    case invalidAlphaComponent(pixelIndex: Int)
    case conversionFailed
}
