
/// A layer that provides a way to asynchronously provide tiles of the layer's content,
/// potentially cached at multiple levels of detail.
///
/// ## Tile Drawing
///
/// To provide tile content, set a delegate that implements `draw(_:in:)`.
/// The delegate will be called with a CGContext configured for each visible tile.
/// The context is already translated and scaled appropriately for the tile's position
/// and the current level of detail.
///
/// ## Usage Example
///
/// ```swift
/// class TileProvider: CALayerDelegate {
///     func draw(_ layer: CALayer, in ctx: CGContext) {
///         // Draw content for the current tile
///         // The context is pre-transformed for the tile's position
///         ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1))
///         ctx.fill(layer.bounds)
///     }
/// }
///
/// let tiledLayer = CATiledLayer()
/// tiledLayer.delegate = TileProvider()
/// tiledLayer.tileSize = CGSize(width: 256, height: 256)
/// ```
open class CATiledLayer: CALayer {

    // MARK: - Initialization

    public required init() {
        super.init()
    }

    /// Initializes a new tiled layer as a copy of the specified layer.
    public required init(layer: Any) {
        super.init(layer: layer)
        if let tiledLayer = layer as? CATiledLayer {
            self.levelsOfDetail = tiledLayer.levelsOfDetail
            self.levelsOfDetailBias = tiledLayer.levelsOfDetailBias
            self.tileSize = tiledLayer.tileSize
            // Note: tileCache and loadingTiles are not copied as they are internal state
        }
    }

    // MARK: - Tile Properties

    /// The number of levels of detail maintained by this layer.
    ///
    /// Each level of detail is rendered at half the resolution of the previous level.
    /// For example, if levelsOfDetail is 3, the layer maintains tiles at full resolution,
    /// half resolution, and quarter resolution.
    open var levelsOfDetail: Int = 1

    /// The number of magnified levels of detail for this layer.
    ///
    /// Positive values add levels of detail for zooming in beyond the layer's normal size.
    /// A value of 2 means the layer can display tiles at 2x and 4x the normal resolution.
    open var levelsOfDetailBias: Int = 0

    /// The maximum size of each tile.
    ///
    /// Tiles are the unit of asynchronous loading. Larger tiles require fewer draw calls
    /// but use more memory and take longer to render.
    open var tileSize: CGSize = CGSize(width: 256, height: 256)

    /// Returns the fading duration for a given view.
    ///
    /// Newly loaded tiles fade in over this duration for smooth appearance.
    open class func fadeDuration() -> CFTimeInterval {
        return 0.25
    }

    // MARK: - Tile Cache

    /// Represents a unique identifier for a tile.
    public struct TileKey: Hashable {
        public let column: Int
        public let row: Int
        public let lodLevel: Int

        public init(column: Int, row: Int, lodLevel: Int) {
            self.column = column
            self.row = row
            self.lodLevel = lodLevel
        }
    }

    /// Cache of rendered tile images.
    ///
    /// Keys are TileKey structs, values are CGImage representations of tiles.
    /// The renderer uses this cache to avoid re-rendering tiles unnecessarily.
    internal var tileCache: [TileKey: CGImage] = [:]

    /// Set of tiles currently being loaded.
    internal var loadingTiles: Set<TileKey> = []

    /// Clears all cached tiles.
    ///
    /// Call this when the layer's content needs to be completely redrawn,
    /// such as when the underlying data changes.
    public func clearTileCache() {
        tileCache.removeAll()
        loadingTiles.removeAll()
        setNeedsDisplay()
    }

    /// Clears a specific tile from the cache.
    ///
    /// Use this to invalidate individual tiles when only part of the content changes.
    public func clearTile(at key: TileKey) {
        tileCache.removeValue(forKey: key)
        loadingTiles.remove(key)
    }

    /// Returns the cached image for a tile, or nil if not cached.
    public func cachedImage(for key: TileKey) -> CGImage? {
        return tileCache[key]
    }

    /// Stores a rendered tile image in the cache.
    public func cacheImage(_ image: CGImage, for key: TileKey) {
        tileCache[key] = image
        loadingTiles.remove(key)
    }
}
