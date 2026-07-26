import Foundation

/// Immutable geometry supplied to a captured tile-content implementation.
public struct CATiledLayerTileDrawingInfo: Equatable, Sendable {
    /// The committed bounds of the tiled layer.
    public let layerBounds: CGRect

    /// The logical rectangle covered by the requested tile.
    public let tileRect: CGRect

    /// The selected level of detail, where zero is the highest detail.
    public let levelOfDetail: Int

    /// The number of output pixels per logical point for this tile request.
    public let pixelScale: CGFloat

    internal init(
        layerBounds: CGRect,
        tileRect: CGRect,
        levelOfDetail: Int,
        pixelScale: CGFloat
    ) {
        self.layerBounds = layerBounds
        self.tileRect = tileRect
        self.levelOfDetail = levelOfDetail
        self.pixelScale = pixelScale
    }
}
