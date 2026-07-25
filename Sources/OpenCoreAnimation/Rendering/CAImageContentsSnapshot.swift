import Foundation

/// Value-owned image contents prepared at transaction commit time.
///
/// Pixel conversion happens while the presentation layer is being captured.
/// The renderer therefore consumes only immutable, tightly packed storage and
/// never retains or reads the source `CGImage` after the commit boundary.
internal struct CAImageContentsSnapshot: Equatable, Hashable, Sendable {
    internal enum Origin: Equatable, Hashable, Sendable {
        case layerContents
        case delegateBackingStore(CALayerContentsFormat)
    }

    internal let storage: CGImageTextureStorage
    internal let origin: Origin
    internal let contentsRect: CGRect
    internal let contentsCenter: CGRect
    internal let contentsScale: CGFloat
    internal let gravity: CALayerContentsGravity
    internal let sampling: CAContentsSampling
    internal let minificationFilterBias: Float
    internal let isOpaque: Bool
}

/// Describes why layer image contents could not cross the commit boundary.
public enum CAImageContentsSnapshotError: Error, Equatable, Sendable {
    case unsupportedContentsType(String)
    case imageConversion(CAImageContentsConversionError)
    case invalidSamplingFilters(
        magnification: CALayerContentsFilter,
        minification: CALayerContentsFilter
    )
    case invalidMinificationFilterBias(Float)
}
