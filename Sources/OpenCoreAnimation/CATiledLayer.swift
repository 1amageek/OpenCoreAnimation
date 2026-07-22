//
//  CATiledLayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation

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

    /// Specifies the default value associated with a tiled-layer property.
    open override class func defaultValue(forKey key: String) -> Any? {
        switch key {
        case "levelsOfDetail":
            return 1
        case "levelsOfDetailBias":
            return 0
        case "tileSize":
            return CGSize(width: 256, height: 256)
        default:
            return super.defaultValue(forKey: key)
        }
    }

    // MARK: - Tile Properties

    /// The number of levels of detail maintained by this layer.
    ///
    /// Each level of detail is rendered at half the resolution of the previous level.
    /// For example, if levelsOfDetail is 3, the layer maintains tiles at full resolution,
    /// half resolution, and quarter resolution.
    open var levelsOfDetail: Int = 1 {
        didSet { markDirty(.contents) }
    }

    /// The number of magnified levels of detail for this layer.
    ///
    /// Positive values add levels of detail for zooming in beyond the layer's normal size.
    /// A value of 2 means the layer can display tiles at 2x and 4x the normal resolution.
    open var levelsOfDetailBias: Int = 0 {
        didSet { markDirty(.contents) }
    }

    /// The maximum size of each tile.
    ///
    /// Tiles are the unit of asynchronous loading. Larger tiles require fewer draw calls
    /// but use more memory and take longer to render.
    open var tileSize: CGSize = CGSize(width: 256, height: 256) {
        didSet { markDirty(.contents) }
    }

    /// Returns the fading duration for a given view.
    ///
    /// Newly loaded tiles fade in over this duration for smooth appearance.
    open class func fadeDuration() -> CFTimeInterval {
        return 0.25
    }

    // MARK: - Tile Cache

    /// Represents a unique identifier for a tile.
    internal struct TileKey: Hashable {
        internal let column: Int
        internal let row: Int
        internal let lodLevel: Int

        internal init(column: Int, row: Int, lodLevel: Int) {
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

    /// Media times at which cached tiles became available for display.
    internal var tileFadeStartTimes: [TileKey: CFTimeInterval] = [:]

    /// Set of tiles currently being loaded.
    internal var loadingTiles: Set<TileKey> = []

    /// Clears all cached tiles.
    ///
    /// Call this when the layer's content needs to be completely redrawn,
    /// such as when the underlying data changes.
    internal func clearTileCache() {
        tileCache.removeAll()
        tileFadeStartTimes.removeAll()
        loadingTiles.removeAll()
        setNeedsDisplay()
    }

    /// Clears a specific tile from the cache.
    ///
    /// Use this to invalidate individual tiles when only part of the content changes.
    internal func clearTile(at key: TileKey) {
        tileCache.removeValue(forKey: key)
        tileFadeStartTimes.removeValue(forKey: key)
        loadingTiles.remove(key)
    }

    /// Returns the cached image for a tile, or nil if not cached.
    internal func cachedImage(for key: TileKey) -> CGImage? {
        return tileCache[key]
    }

    /// Stores a rendered tile image in the cache.
    internal func cacheImage(
        _ image: CGImage,
        for key: TileKey,
        at mediaTime: CFTimeInterval = CACurrentMediaTime()
    ) {
        tileCache[key] = image
        tileFadeStartTimes[key] = mediaTime
        loadingTiles.remove(key)
    }

    /// Returns the opacity for a newly cached tile at the supplied media time.
    internal func tileOpacity(for key: TileKey, at mediaTime: CFTimeInterval) -> Float {
        guard let startTime = tileFadeStartTimes[key] else { return 1 }
        let duration = type(of: self).fadeDuration()
        guard duration > 0 else { return 1 }
        return Float(min(max((mediaTime - startTime) / duration, 0), 1))
    }

    /// Selects a signed detail level for a screen-space scale.
    /// Negative levels represent magnified detail supplied by `levelsOfDetailBias`.
    internal func lodLevel(forScreenScale screenScale: CGFloat) -> Int {
        let safeScale = screenScale.isFinite && screenScale > 0 ? screenScale : 1
        let requestedLevel = Int(floor(-log2(safeScale)))
        let minimumLevel = -max(0, levelsOfDetailBias)

        let pixelWidth = max(0, bounds.width * max(contentsScale, 0))
        let pixelHeight = max(0, bounds.height * max(contentsScale, 0))
        let minimumPixelDimension = min(pixelWidth, pixelHeight)
        let requestedMaximum = max(0, levelsOfDetail - 1)
        let dimensionLimit: Int
        if minimumPixelDimension.isFinite, minimumPixelDimension >= 1 {
            dimensionLimit = max(0, Int(floor(log2(minimumPixelDimension))))
        } else if minimumPixelDimension == .infinity {
            dimensionLimit = requestedMaximum
        } else {
            dimensionLimit = 0
        }
        let maximumLevel = min(requestedMaximum, dimensionLimit)
        return min(max(requestedLevel, minimumLevel), maximumLevel)
    }
}
