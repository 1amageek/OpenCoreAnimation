#if arch(wasm32)
import Foundation
import SwiftWebGPU

// MARK: - Texture Manager (LRU Cache)

/// A texture cache entry with access tracking for LRU eviction.
///
/// The entry holds a strong reference to the source `CGImage` so the
/// `ObjectIdentifier(cgImage)` used as the dictionary key remains unique
/// for the lifetime of the cache entry. Without this strong reference the
/// CGImage may be deallocated by ARC, its heap address reused by a fresh
/// allocation, and a subsequent lookup would return the wrong cached
/// `GPUTexture` (cross-image identity collision).
private struct TextureCacheEntry {
    let cgImage: CGImage
    let texture: GPUTexture
    let width: Int
    let height: Int
    var lastAccessFrame: UInt64
    var accessCount: UInt64

    /// Approximate memory usage in bytes (4 bytes per pixel for RGBA8).
    var memorySize: UInt64 {
        return UInt64(width * height * 4)
    }
}

/// A texture manager with LRU (Least Recently Used) cache eviction.
///
/// This class manages GPU textures with automatic memory management.
/// When the cache exceeds its capacity, the least recently used textures
/// are evicted to make room for new ones.
///
/// ## Identity & Ownership
///
/// The cache is keyed by `ObjectIdentifier(CGImage)`, but a raw pointer
/// identifier is only stable while the underlying object is alive. To
/// guarantee key uniqueness, every cache entry holds a strong reference
/// to its `CGImage`; downstream caches keyed by the same identity (e.g.
/// `GPUTextureView` / `GPUBindGroup` caches in the renderer) are kept in
/// sync via the `onEvict` callback, which fires for every entry the
/// manager removes.
///
/// ## Usage
///
/// ```swift
/// let manager = GPUTextureManager(device: device, maxTextures: 256, maxMemory: 256 * 1024 * 1024)
///
/// // Get or create a texture (manager retains the CGImage while cached).
/// let texture = manager.getOrCreateTexture(for: cgImage, width: w, height: h) {
///     return createTextureFromImage(cgImage, device: device)
/// }
///
/// // Receive eviction notifications so downstream caches stay in sync.
/// manager.onEvict = { evictedImage in
///     downstreamCache.removeValue(forKey: ObjectIdentifier(evictedImage))
/// }
///
/// // At end of frame, update frame counter
/// manager.advanceFrame()
/// ```
public final class GPUTextureManager {

    // MARK: - Properties

    /// The GPU device for creating textures.
    private weak var device: GPUDevice?

    /// Cache of textures keyed by `ObjectIdentifier(CGImage)`.
    ///
    /// Each entry retains its `CGImage` (see `TextureCacheEntry.cgImage`)
    /// so the identifier remains unique for the cached lifetime.
    private var cache: [ObjectIdentifier: TextureCacheEntry] = [:]

    /// Current frame number for LRU tracking.
    private var currentFrame: UInt64 = 0

    /// Maximum number of textures in the cache.
    public let maxTextures: Int

    /// Maximum total memory in bytes for cached textures.
    public let maxMemoryBytes: UInt64

    /// Current number of textures in the cache.
    public var textureCount: Int {
        return cache.count
    }

    /// Current total memory usage in bytes.
    public private(set) var currentMemoryBytes: UInt64 = 0

    /// Number of cache hits since creation.
    public private(set) var cacheHits: UInt64 = 0

    /// Number of cache misses since creation.
    public private(set) var cacheMisses: UInt64 = 0

    /// Cache hit rate (0.0 to 1.0).
    public var hitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0.0
    }

    /// Called for each `CGImage` whose cached texture is removed.
    ///
    /// Downstream caches keyed by the same `ObjectIdentifier(CGImage)`
    /// (texture views, bind groups) MUST drop their entries here,
    /// otherwise they may end up serving stale `GPUTextureView`s for a
    /// future image whose heap address happens to alias the evicted one.
    public var onEvict: ((CGImage) -> Void)?

    // MARK: - Initialization

    /// Creates a new texture manager.
    ///
    /// - Parameters:
    ///   - device: The GPU device for creating textures.
    ///   - maxTextures: Maximum number of textures to cache (default: 256).
    ///   - maxMemoryBytes: Maximum memory in bytes (default: 256MB).
    public init(device: GPUDevice, maxTextures: Int = 256, maxMemoryBytes: UInt64 = 256 * 1024 * 1024) {
        self.device = device
        self.maxTextures = maxTextures
        self.maxMemoryBytes = maxMemoryBytes
    }

    // MARK: - Public Methods

    /// Gets a cached texture or creates a new one using the provided factory.
    ///
    /// The manager retains `cgImage` for as long as the texture stays in
    /// the cache so that `ObjectIdentifier(cgImage)` remains a unique key.
    ///
    /// - Parameters:
    ///   - cgImage: The source image. Used both as the cache key
    ///     (`ObjectIdentifier(cgImage)`) and as the retained owner.
    ///   - width: Width of the texture (used for memory tracking).
    ///   - height: Height of the texture (used for memory tracking).
    ///   - factory: A closure that creates the texture if not cached.
    /// - Returns: The cached or newly created texture.
    public func getOrCreateTexture(
        for cgImage: CGImage,
        width: Int,
        height: Int,
        factory: () -> GPUTexture?
    ) -> GPUTexture? {
        let key = ObjectIdentifier(cgImage)

        // Check cache first
        if var entry = cache[key] {
            // Update access tracking
            entry.lastAccessFrame = currentFrame
            entry.accessCount += 1
            cache[key] = entry
            cacheHits += 1
            return entry.texture
        }

        // Cache miss - create new texture
        cacheMisses += 1

        guard let texture = factory() else {
            return nil
        }

        let memorySize = UInt64(width * height * 4)

        // Evict if necessary
        evictIfNeeded(forNewMemory: memorySize)

        // Add to cache (entry retains cgImage)
        let entry = TextureCacheEntry(
            cgImage: cgImage,
            texture: texture,
            width: width,
            height: height,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        cache[key] = entry
        currentMemoryBytes += memorySize

        return texture
    }

    /// Gets a cached texture if it exists.
    ///
    /// - Parameter cgImage: The source image identifying the cached texture.
    /// - Returns: The cached texture, or nil if not found.
    public func getCachedTexture(for cgImage: CGImage) -> GPUTexture? {
        let key = ObjectIdentifier(cgImage)
        guard var entry = cache[key] else {
            return nil
        }

        // Update access tracking
        entry.lastAccessFrame = currentFrame
        entry.accessCount += 1
        cache[key] = entry
        cacheHits += 1

        return entry.texture
    }

    /// Manually adds a texture to the cache.
    ///
    /// - Parameters:
    ///   - texture: The texture to cache.
    ///   - cgImage: The source image. Used as the cache key
    ///     (`ObjectIdentifier(cgImage)`) and retained by the entry.
    ///   - width: Width of the texture.
    ///   - height: Height of the texture.
    public func cacheTexture(_ texture: GPUTexture, for cgImage: CGImage, width: Int, height: Int) {
        let key = ObjectIdentifier(cgImage)

        // Remove existing entry if present (notify downstream caches)
        if let existing = cache.removeValue(forKey: key) {
            currentMemoryBytes -= existing.memorySize
            onEvict?(existing.cgImage)
        }

        let memorySize = UInt64(width * height * 4)
        evictIfNeeded(forNewMemory: memorySize)

        let entry = TextureCacheEntry(
            cgImage: cgImage,
            texture: texture,
            width: width,
            height: height,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        cache[key] = entry
        currentMemoryBytes += memorySize
    }

    /// Removes a specific texture from the cache.
    ///
    /// - Parameter cgImage: The source image whose cached texture should
    ///   be removed.
    public func removeTexture(for cgImage: CGImage) {
        let key = ObjectIdentifier(cgImage)
        if let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.memorySize
            onEvict?(entry.cgImage)
        }
    }

    /// Advances the frame counter for LRU tracking.
    ///
    /// Call this at the end of each frame.
    public func advanceFrame() {
        currentFrame += 1
    }

    /// Clears all cached textures.
    public func clearAll() {
        // Snapshot the entries we are about to drop so the eviction
        // callback can run *after* the dictionary is empty. This avoids
        // iteration-during-mutation if the callback indirectly inserts
        // back into the cache.
        let evicted = cache.values.map { $0.cgImage }
        cache.removeAll()
        currentMemoryBytes = 0

        if let onEvict = onEvict {
            for cgImage in evicted {
                onEvict(cgImage)
            }
        }
    }

    /// Invalidates the texture manager.
    public func invalidate() {
        clearAll()
        device = nil
    }

    /// Evicts textures that haven't been used for the specified number of frames.
    ///
    /// - Parameter frameThreshold: Number of frames after which unused textures are evicted.
    public func evictStale(olderThan frameThreshold: UInt64) {
        let cutoffFrame = currentFrame > frameThreshold ? currentFrame - frameThreshold : 0

        // Two-phase: collect victim keys, drop them, then fire callbacks.
        var keysToRemove: [ObjectIdentifier] = []
        for (key, entry) in cache {
            if entry.lastAccessFrame < cutoffFrame {
                keysToRemove.append(key)
            }
        }

        var evictedImages: [CGImage] = []
        evictedImages.reserveCapacity(keysToRemove.count)
        for key in keysToRemove {
            if let entry = cache.removeValue(forKey: key) {
                currentMemoryBytes -= entry.memorySize
                evictedImages.append(entry.cgImage)
            }
        }

        if let onEvict = onEvict {
            for cgImage in evictedImages {
                onEvict(cgImage)
            }
        }
    }

    // MARK: - Private Methods

    /// Evicts least recently used textures if the cache is over capacity.
    private func evictIfNeeded(forNewMemory newMemory: UInt64) {
        // Check if we need to evict based on count
        while cache.count >= maxTextures {
            evictLeastRecentlyUsed()
        }

        // Check if we need to evict based on memory
        while currentMemoryBytes + newMemory > maxMemoryBytes && !cache.isEmpty {
            evictLeastRecentlyUsed()
        }
    }

    /// Evicts the single least recently used texture.
    private func evictLeastRecentlyUsed() {
        guard !cache.isEmpty else { return }

        // Find the entry with the oldest last access frame
        var oldestKey: ObjectIdentifier?
        var oldestFrame: UInt64 = .max

        for (key, entry) in cache {
            if entry.lastAccessFrame < oldestFrame {
                oldestFrame = entry.lastAccessFrame
                oldestKey = key
            }
        }

        if let key = oldestKey, let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.memorySize
            onEvict?(entry.cgImage)
        }
    }
}

#endif
