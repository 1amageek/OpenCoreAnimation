#if arch(wasm32)
import Foundation

// MARK: - Geometry Cache

/// A cache entry with access tracking for LRU eviction.
private struct GeometryCacheEntry {
    let geometry: TessellatedGeometry
    var lastAccessFrame: UInt64
    var accessCount: UInt64
}

/// A cache for tessellated path geometry with LRU eviction.
///
/// Path tessellation (converting bezier curves to triangles) is computationally
/// expensive. This cache stores tessellated geometry so paths that haven't changed
/// don't need to be re-tessellated every frame.
///
/// ## Usage
///
/// ```swift
/// let cache = GeometryCache(maxEntries: 256, maxMemoryBytes: 64 * 1024 * 1024)
///
/// // Create a cache key for a shape layer's path
/// let key = GeometryCacheKey(
///     pathHash: path.hashValue,
///     isStroke: true,
///     lineWidth: shapeLayer.lineWidth,
///     lineCap: shapeLayer.lineCap,
///     lineJoin: shapeLayer.lineJoin
/// )
///
/// // Get cached geometry or compute it
/// if let cached = cache.getGeometry(for: key) {
///     // Use cached geometry
/// } else {
///     let geometry = tessellate(path)
///     cache.cacheGeometry(geometry, for: key)
/// }
///
/// // At end of frame
/// cache.advanceFrame()
/// ```
public final class GeometryCache {

    // MARK: - Properties

    /// Cache of tessellated geometry keyed by GeometryCacheKey.
    private var cache: [GeometryCacheKey: GeometryCacheEntry] = [:]

    /// Current frame number for LRU tracking.
    private var currentFrame: UInt64 = 0

    /// Maximum number of entries in the cache.
    public let maxEntries: Int

    /// Maximum total memory in bytes for cached geometry.
    public let maxMemoryBytes: Int

    /// Current number of entries in the cache.
    public var entryCount: Int {
        return cache.count
    }

    /// Current total memory usage in bytes.
    public private(set) var currentMemoryBytes: Int = 0

    /// Number of cache hits since creation.
    public private(set) var cacheHits: UInt64 = 0

    /// Number of cache misses since creation.
    public private(set) var cacheMisses: UInt64 = 0

    /// Cache hit rate (0.0 to 1.0).
    public var hitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0.0
    }

    // MARK: - Initialization

    /// Creates a new geometry cache.
    ///
    /// - Parameters:
    ///   - maxEntries: Maximum number of entries to cache (default: 256).
    ///   - maxMemoryBytes: Maximum memory in bytes (default: 64MB).
    public init(maxEntries: Int = 256, maxMemoryBytes: Int = 64 * 1024 * 1024) {
        self.maxEntries = maxEntries
        self.maxMemoryBytes = maxMemoryBytes
    }

    // MARK: - Public Methods

    /// Gets cached tessellated geometry if it exists.
    ///
    /// - Parameter key: The cache key.
    /// - Returns: The cached geometry, or nil if not found.
    public func getGeometry(for key: GeometryCacheKey) -> TessellatedGeometry? {
        guard var entry = cache[key] else {
            cacheMisses += 1
            return nil
        }

        // Update access tracking
        entry.lastAccessFrame = currentFrame
        entry.accessCount += 1
        cache[key] = entry
        cacheHits += 1

        return entry.geometry
    }

    /// Caches tessellated geometry.
    ///
    /// - Parameters:
    ///   - geometry: The tessellated geometry to cache.
    ///   - key: The cache key.
    public func cacheGeometry(_ geometry: TessellatedGeometry, for key: GeometryCacheKey) {
        // Remove existing entry if present
        if let existing = cache[key] {
            currentMemoryBytes -= existing.geometry.memorySize
        }

        let memorySize = geometry.memorySize
        evictIfNeeded(forNewMemory: memorySize)

        let entry = GeometryCacheEntry(
            geometry: geometry,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        cache[key] = entry
        currentMemoryBytes += memorySize
    }

    /// Gets cached geometry or creates it using the provided factory.
    ///
    /// - Parameters:
    ///   - key: The cache key.
    ///   - factory: A closure that creates the geometry if not cached.
    /// - Returns: The cached or newly created geometry.
    public func getOrCreateGeometry(
        for key: GeometryCacheKey,
        factory: () -> TessellatedGeometry
    ) -> TessellatedGeometry {
        if let cached = getGeometry(for: key) {
            return cached
        }

        let geometry = factory()
        cacheGeometry(geometry, for: key)
        return geometry
    }

    /// Removes a specific geometry from the cache.
    ///
    /// - Parameter key: The cache key.
    public func removeGeometry(for key: GeometryCacheKey) {
        if let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.geometry.memorySize
        }
    }

    /// Advances the frame counter for LRU tracking.
    ///
    /// Call this at the end of each frame.
    public func advanceFrame() {
        currentFrame += 1
    }

    /// Clears all cached geometry.
    public func clearAll() {
        cache.removeAll()
        currentMemoryBytes = 0
    }

    /// Invalidates the cache.
    public func invalidate() {
        clearAll()
    }

    /// Evicts geometry that hasn't been used for the specified number of frames.
    ///
    /// - Parameter frameThreshold: Number of frames after which unused geometry is evicted.
    public func evictStale(olderThan frameThreshold: UInt64) {
        let cutoffFrame = currentFrame > frameThreshold ? currentFrame - frameThreshold : 0

        var keysToRemove: [GeometryCacheKey] = []
        for (key, entry) in cache {
            if entry.lastAccessFrame < cutoffFrame {
                keysToRemove.append(key)
            }
        }

        for key in keysToRemove {
            if let entry = cache.removeValue(forKey: key) {
                currentMemoryBytes -= entry.geometry.memorySize
            }
        }
    }

    // MARK: - Private Methods

    /// Evicts least recently used entries if the cache is over capacity.
    private func evictIfNeeded(forNewMemory newMemory: Int) {
        // Check if we need to evict based on count
        while cache.count >= maxEntries {
            evictLeastRecentlyUsed()
        }

        // Check if we need to evict based on memory
        while currentMemoryBytes + newMemory > maxMemoryBytes && !cache.isEmpty {
            evictLeastRecentlyUsed()
        }
    }

    /// Evicts the single least recently used entry.
    private func evictLeastRecentlyUsed() {
        guard !cache.isEmpty else { return }

        // Find the entry with the oldest last access frame
        var oldestKey: GeometryCacheKey?
        var oldestFrame: UInt64 = .max

        for (key, entry) in cache {
            if entry.lastAccessFrame < oldestFrame {
                oldestFrame = entry.lastAccessFrame
                oldestKey = key
            }
        }

        if let key = oldestKey, let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.geometry.memorySize
        }
    }
}

#endif
