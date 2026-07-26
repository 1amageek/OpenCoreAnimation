//
//  CATiledLayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import Synchronization

/// A layer that provides a way to asynchronously provide tiles of the layer's content,
/// potentially cached at multiple levels of detail.
///
/// ## Tile Drawing
///
/// To provide tile content, set a delegate that implements
/// `CATiledLayerContentProvider`. OpenCoreAnimation captures a Sendable content
/// snapshot at transaction commit and invokes that snapshot for every visible
/// tile. The context is already translated and scaled for the tile position and
/// current level of detail.
///
/// ## Usage Example
///
/// ```swift
/// struct TileContent: CATiledLayerContentSnapshot {
///     func drawTile(
///         _ tile: CATiledLayerTileDrawingInfo,
///         in context: CGContext
///     ) {
///         context.setFillColor(
///             CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1)
///         )
///         context.fill(tile.tileRect)
///     }
/// }
///
/// final class TileProvider: CATiledLayerContentProvider {
///     func makeTileContentSnapshot()
///         -> any CATiledLayerContentSnapshot {
///         TileContent()
///     }
/// }
///
/// let tiledLayer = CATiledLayer()
/// tiledLayer.delegate = TileProvider()
/// tiledLayer.tileSize = CGSize(width: 256, height: 256)
/// ```
open class CATiledLayer: CALayer {
    private static let resourceIdentityStorage =
        Mutex<UInt64>(0)

    private struct TileState {
        var cache = CATileKeyMap<CGImageTextureStorage>()
        var fadeStartTimes = CATileKeyMap<CFTimeInterval>()
        var loadingTiles = CATileKeySet()
        var loadingGenerations = CATileKeyMap<UInt64>()
        var generation: UInt64 = 0
    }

    private let tileState = Mutex(TileState())
    internal private(set) var resourceIdentity: UInt64

    // MARK: - Initialization

    public required init() {
        resourceIdentity = Self.nextResourceIdentity()
        super.init()
    }

    /// Initializes a new tiled layer as a copy of the specified layer.
    public required init(layer: Any) {
        resourceIdentity = Self.nextResourceIdentity()
        super.init(layer: layer)
        if let tiledLayer = layer as? CATiledLayer {
            self._levelsOfDetail = tiledLayer._levelsOfDetail
            self._levelsOfDetailBias = tiledLayer._levelsOfDetailBias
            self._tileSize = tiledLayer._tileSize
            // Compatibility cache and request state belong to the new copy.
        }
    }

    private static func nextResourceIdentity() -> UInt64 {
        resourceIdentityStorage.withLock { identity in
            identity &+= 1
            if identity == 0 {
                identity = 1
            }
            return identity
        }
    }

    internal func adoptPresentationResourceState(
        from modelLayer: CATiledLayer
    ) {
        resourceIdentity = modelLayer.resourceIdentity
        let generation = modelLayer.tileCacheGeneration
        tileState.withLock { state in
            state.generation = generation
            state.cache.removeAll(keepingCapacity: false)
            state.fadeStartTimes.removeAll(keepingCapacity: false)
            state.loadingTiles.removeAll(keepingCapacity: false)
            state.loadingGenerations.removeAll(
                keepingCapacity: false
            )
        }
    }

    internal func _copyPresentationConfiguration(
        from modelLayer: CATiledLayer
    ) {
        _levelsOfDetail = modelLayer._levelsOfDetail
        _levelsOfDetailBias = modelLayer._levelsOfDetailBias
        _tileSize = modelLayer._tileSize
        adoptPresentationResourceState(from: modelLayer)
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
    private var _levelsOfDetail = 1
    open var levelsOfDetail: Int {
        get { _levelsOfDetail }
        set {
            guard _levelsOfDetail != newValue else { return }
            _levelsOfDetail = newValue
            setNeedsDisplay()
        }
    }

    /// The number of magnified levels of detail for this layer.
    ///
    /// Positive values add levels of detail for zooming in beyond the layer's normal size.
    /// A value of 2 means the layer can display tiles at 2x and 4x the normal resolution.
    private var _levelsOfDetailBias = 0
    open var levelsOfDetailBias: Int {
        get { _levelsOfDetailBias }
        set {
            guard _levelsOfDetailBias != newValue else { return }
            _levelsOfDetailBias = newValue
            setNeedsDisplay()
        }
    }

    /// The maximum size of each tile.
    ///
    /// Tiles are the unit of asynchronous loading. Larger tiles require fewer draw calls
    /// but use more memory and take longer to render.
    private var _tileSize = CGSize(width: 256, height: 256)
    open var tileSize: CGSize {
        get { _tileSize }
        set {
            guard _tileSize != newValue else { return }
            _tileSize = newValue
            setNeedsDisplay()
        }
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

    /// Advances whenever cached content becomes invalid.
    internal var tileCacheGeneration: UInt64 {
        tileState.withLock(\.generation)
    }

    internal var isTileCacheEmpty: Bool {
        tileState.withLock(\.cache.isEmpty)
    }

    /// Clears all cached tiles.
    ///
    /// Call this when the layer's content needs to be completely redrawn,
    /// such as when the underlying data changes.
    internal func clearTileCache() {
        invalidateTileStorage()
        super.setNeedsDisplay()
    }

    /// Clears a specific tile from the cache.
    ///
    /// Use this to invalidate individual tiles when only part of the content changes.
    internal func clearTile(at key: TileKey) {
        // Advancing the generation prevents a replacement request for this key
        // from aliasing an older request that completes later. Other cached tiles
        // remain valid, while in-flight requests are conservatively restarted.
        tileState.withLock { state in
            state.generation &+= 1
            state.loadingTiles.removeAll(keepingCapacity: true)
            state.loadingGenerations.removeAll(keepingCapacity: true)
            state.cache.removeValue(forKey: key)
            state.fadeStartTimes.removeValue(forKey: key)
        }
    }

    /// Returns immutable renderer-ready pixels for a tile.
    internal func cachedStorage(
        for key: TileKey
    ) -> CGImageTextureStorage? {
        tileState.withLock { $0.cache[key] }
    }

    internal func tileFadeStartTime(
        for key: TileKey
    ) -> CFTimeInterval? {
        tileState.withLock { $0.fadeStartTimes[key] }
    }

    internal func hasLoadingTile(_ key: TileKey) -> Bool {
        tileState.withLock { $0.loadingTiles.contains(key) }
    }

    internal func loadingGeneration(
        for key: TileKey
    ) -> UInt64? {
        tileState.withLock { $0.loadingGenerations[key] }
    }

    /// Atomically reserves one tile request in the current cache generation.
    internal func beginTileRequest(
        for key: TileKey
    ) -> UInt64? {
        tileState.withLock { state in
            guard !state.loadingTiles.contains(key) else {
                return nil
            }
            let generation = state.generation
            state.loadingTiles.insert(key)
            state.loadingGenerations[key] = generation
            return generation
        }
    }

    /// Releases a failed request only if it still owns the recorded generation.
    internal func cancelTileRequest(
        for key: TileKey,
        generation: UInt64
    ) {
        tileState.withLock { state in
            guard state.loadingGenerations[key] == generation else {
                return
            }
            state.loadingTiles.remove(key)
            state.loadingGenerations.removeValue(forKey: key)
        }
    }

    /// Stores a rendered tile image in the cache.
    @discardableResult
    internal func cacheImage(
        _ image: CGImage,
        for key: TileKey,
        requestGeneration: UInt64? = nil,
        at mediaTime: CFTimeInterval = CACurrentMediaTime()
    ) throws(CAImageContentsConversionError) -> Bool {
        if let requestGeneration,
           !ownsTileRequest(
                for: key,
                generation: requestGeneration
           ) {
            return false
        }

        // Conversion is intentionally outside the critical section. The
        // ownership check above avoids work for known-stale results, while the
        // check in the insertion transaction closes an invalidation race.
        let storage: CGImageTextureStorage
        do {
            storage = try CGImageTextureStorageConverter.convert(
                image
            )
        } catch {
            // Invalidation owns the result once this request becomes stale.
            // Do not surface a conversion failure from work that can no
            // longer affect the current cache generation.
            if let requestGeneration,
               !ownsTileRequest(
                    for: key,
                    generation: requestGeneration
               ) {
                return false
            }
            throw error
        }
        return tileState.withLock { state in
            if let requestGeneration {
                guard requestGeneration == state.generation,
                      state.loadingGenerations[key]
                        == requestGeneration else {
                    return false
                }
            }
            state.cache[key] = storage
            state.fadeStartTimes[key] = mediaTime
            state.loadingTiles.remove(key)
            state.loadingGenerations.removeValue(forKey: key)
            return true
        }
    }

    private func ownsTileRequest(
        for key: TileKey,
        generation: UInt64
    ) -> Bool {
        tileState.withLock { state in
            generation == state.generation
                && state.loadingGenerations[key] == generation
        }
    }

    /// Invalidates all cached and in-flight tile content.
    open override func setNeedsDisplay() {
        invalidateTileStorage()
        super.setNeedsDisplay()
    }

    /// Invalidates tile content after a regional display request.
    ///
    /// Device-clamped tile rectangles depend on renderer limits, so a regional
    /// request conservatively advances the complete generation. This guarantees
    /// that no stale cached or in-flight tile survives the requested update.
    open override func setNeedsDisplay(_ r: CGRect) {
        invalidateTileStorage()
        super.setNeedsDisplay(r)
    }

    open override class func needsDisplay(forKey key: String) -> Bool {
        switch key {
        case "bounds", "contentsScale":
            return true
        default:
            return super.needsDisplay(forKey: key)
        }
    }

    private func invalidateTileStorage() {
        tileState.withLock { state in
            state.generation &+= 1
            state.cache.removeAll(keepingCapacity: true)
            state.fadeStartTimes.removeAll(keepingCapacity: true)
            state.loadingTiles.removeAll(keepingCapacity: true)
            state.loadingGenerations.removeAll(keepingCapacity: true)
        }
    }

    /// Returns the opacity for a newly cached tile at the supplied media time.
    internal func tileOpacity(for key: TileKey, at mediaTime: CFTimeInterval) -> Float {
        guard let startTime = tileFadeStartTime(for: key) else {
            return 1
        }
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

    /// Returns whether a tiled-layer property differs from its archive default.
    open override func shouldArchiveValue(forKey key: String) -> Bool {
        switch key {
        case "levelsOfDetail": return levelsOfDetail != 1
        case "levelsOfDetailBias": return levelsOfDetailBias != 0
        case "tileSize": return tileSize != CGSize(width: 256, height: 256)
        default: return super.shouldArchiveValue(forKey: key)
        }
    }
}
