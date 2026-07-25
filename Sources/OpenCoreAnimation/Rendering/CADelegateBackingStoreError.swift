import Foundation

/// Describes why a layer delegate could not produce a bitmap backing store.
public enum CADelegateBackingStoreError: Error, Equatable, Sendable {
    case invalidGeometry
    case dimensionsExceedTextureLimit(width: Int, height: Int, maximum: Int)
    case pixelStorageSizeOverflow(width: Int, height: Int, bitsPerPixel: Int)
    case unsupportedContentsFormat(String)
    case extendedColorSpaceUnavailable
    case contextCreationFailed
    case extendedHeadroomRejected(CGFloat)
    case snapshotFailed
}
