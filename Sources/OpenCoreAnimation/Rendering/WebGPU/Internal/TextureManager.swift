#if arch(wasm32)
import Foundation
import SwiftWebGPU

// MARK: - Texture Manager (LRU Cache)

/// A texture cache entry with access tracking for LRU eviction.
private struct TextureCacheEntry {
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
/// ## Usage
///
/// ```swift
/// let manager = GPUTextureManager(device: device, maxTextures: 256, maxMemory: 256 * 1024 * 1024)
///
/// // Get or create a texture
/// let texture = manager.getOrCreateTexture(for: imageId) { device in
///     return createTextureFromImage(image, device: device)
/// }
///
/// // At end of frame, update frame counter
/// manager.advanceFrame()
/// ```
public final class GPUTextureManager {

    // MARK: - Properties

    /// The GPU device for creating textures.
    private weak var device: GPUDevice?

    /// Cache of textures keyed by ObjectIdentifier.
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
    /// - Parameters:
    ///   - key: The unique identifier for the texture (typically ObjectIdentifier of source image).
    ///   - width: Width of the texture (used for memory tracking).
    ///   - height: Height of the texture (used for memory tracking).
    ///   - factory: A closure that creates the texture if not cached.
    /// - Returns: The cached or newly created texture.
    public func getOrCreateTexture(
        for key: ObjectIdentifier,
        width: Int,
        height: Int,
        factory: () -> GPUTexture?
    ) -> GPUTexture? {
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

        // Add to cache
        let entry = TextureCacheEntry(
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
    /// - Parameter key: The unique identifier for the texture.
    /// - Returns: The cached texture, or nil if not found.
    public func getCachedTexture(for key: ObjectIdentifier) -> GPUTexture? {
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
    ///   - key: The unique identifier for the texture.
    ///   - width: Width of the texture.
    ///   - height: Height of the texture.
    public func cacheTexture(_ texture: GPUTexture, for key: ObjectIdentifier, width: Int, height: Int) {
        // Remove existing entry if present
        if let existing = cache[key] {
            currentMemoryBytes -= existing.memorySize
        }

        let memorySize = UInt64(width * height * 4)
        evictIfNeeded(forNewMemory: memorySize)

        let entry = TextureCacheEntry(
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
    /// - Parameter key: The unique identifier of the texture to remove.
    public func removeTexture(for key: ObjectIdentifier) {
        if let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.memorySize
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
        cache.removeAll()
        currentMemoryBytes = 0
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

        var keysToRemove: [ObjectIdentifier] = []
        for (key, entry) in cache {
            if entry.lastAccessFrame < cutoffFrame {
                keysToRemove.append(key)
            }
        }

        for key in keysToRemove {
            if let entry = cache.removeValue(forKey: key) {
                currentMemoryBytes -= entry.memorySize
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
        }
    }
}

#endif
