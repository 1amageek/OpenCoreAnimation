#if arch(wasm32)
import Foundation
import SwiftWebGPU

// MARK: - Texture Manager (LRU Cache)

private struct TextureCacheEntry {
    /// Retains the identity-keyed image for the complete cache lifetime.
    let cgImage: CGImage
    let texture: GPUTexture
    let width: Int
    let height: Int
    let memorySize: UInt64
    var lastAccessFrame: UInt64
    var accessCount: UInt64
}

private struct ImmutableTextureCacheEntry {
    let texture: GPUTexture
    let memorySize: UInt64
    var lastAccessFrame: UInt64
    var accessCount: UInt64
}

private enum TextureEvictionCandidate {
    case image(ObjectIdentifier)
    case immutableStorage(CGImageTextureStorage)
}

/// An LRU cache for identity-owned `CGImage` textures and commit-owned pixels.
///
/// Identity keys and value keys intentionally use separate dictionaries. This
/// preserves the compact, established `ObjectIdentifier` storage path for live
/// images while both stores share one texture-count and GPU-memory budget.
public final class GPUTextureManager {
    private weak var device: GPUDevice?
    private var imageCache: [ObjectIdentifier: TextureCacheEntry] = [:]
    private var immutableStorageCache =
        OpenAddressingHashMap<
            CGImageTextureStorage,
            ImmutableTextureCacheEntry
        >()
    private var currentFrame: UInt64 = 0

    public let maxTextures: Int
    public let maxMemoryBytes: UInt64

    public var textureCount: Int {
        imageCache.count + immutableStorageCache.count
    }

    public private(set) var currentMemoryBytes: UInt64 = 0
    public private(set) var cacheHits: UInt64 = 0
    public private(set) var cacheMisses: UInt64 = 0

    public var hitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0
    }

    /// Runs after an identity-owned texture has left the cache.
    public var onEvict: ((CGImage) -> Void)?

    /// Runs after commit-owned pixel storage has left the cache.
    internal var onImmutableStorageEvict: ((CGImageTextureStorage) -> Void)?

    public init(
        device: GPUDevice,
        maxTextures: Int = 256,
        maxMemoryBytes: UInt64 = 256 * 1024 * 1024
    ) {
        precondition(maxTextures > 0)
        precondition(maxMemoryBytes > 0)
        self.device = device
        self.maxTextures = maxTextures
        self.maxMemoryBytes = maxMemoryBytes
    }

    public func getOrCreateTexture(
        for cgImage: CGImage,
        width: Int,
        height: Int,
        memorySizeBytes: UInt64? = nil,
        factory: () -> GPUTexture?
    ) -> GPUTexture? {
        let key = ObjectIdentifier(cgImage)
        if var entry = imageCache[key] {
            entry.lastAccessFrame = currentFrame
            entry.accessCount += 1
            imageCache[key] = entry
            cacheHits += 1
            return entry.texture
        }

        cacheMisses += 1
        guard let texture = factory() else { return nil }
        let memorySize = memorySizeBytes ?? UInt64(width * height * 4)
        evictIfNeeded(forNewMemory: memorySize)
        imageCache[key] = TextureCacheEntry(
            cgImage: cgImage,
            texture: texture,
            width: width,
            height: height,
            memorySize: memorySize,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        currentMemoryBytes += memorySize
        return texture
    }

    internal func getOrCreateTexture(
        for storage: CGImageTextureStorage,
        memorySizeBytes: UInt64,
        factory: () -> GPUTexture?
    ) -> GPUTexture? {
        if var entry = immutableStorageCache[storage] {
            entry.lastAccessFrame = currentFrame
            entry.accessCount += 1
            immutableStorageCache[storage] = entry
            cacheHits += 1
            return entry.texture
        }

        cacheMisses += 1
        guard let texture = factory() else { return nil }
        evictIfNeeded(forNewMemory: memorySizeBytes)
        immutableStorageCache[storage] = ImmutableTextureCacheEntry(
            texture: texture,
            memorySize: memorySizeBytes,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        currentMemoryBytes += memorySizeBytes
        return texture
    }

    public func getCachedTexture(for cgImage: CGImage) -> GPUTexture? {
        let key = ObjectIdentifier(cgImage)
        guard var entry = imageCache[key] else { return nil }
        entry.lastAccessFrame = currentFrame
        entry.accessCount += 1
        imageCache[key] = entry
        cacheHits += 1
        return entry.texture
    }

    public func cacheTexture(
        _ texture: GPUTexture,
        for cgImage: CGImage,
        width: Int,
        height: Int
    ) {
        let key = ObjectIdentifier(cgImage)
        let replaced = imageCache.removeValue(forKey: key)
        if let replaced {
            currentMemoryBytes -= replaced.memorySize
        }

        let memorySize = UInt64(width * height * 4)
        evictIfNeeded(forNewMemory: memorySize)
        imageCache[key] = TextureCacheEntry(
            cgImage: cgImage,
            texture: texture,
            width: width,
            height: height,
            memorySize: memorySize,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        currentMemoryBytes += memorySize
        if let replaced {
            onEvict?(replaced.cgImage)
        }
    }

    public func removeTexture(for cgImage: CGImage) {
        let key = ObjectIdentifier(cgImage)
        guard let entry = imageCache.removeValue(forKey: key) else {
            return
        }
        currentMemoryBytes -= entry.memorySize
        onEvict?(entry.cgImage)
    }

    public func advanceFrame() {
        currentFrame += 1
    }

    public func clearAll() {
        let evictedImages = imageCache.values.map(\.cgImage)
        let evictedStorage = Array(immutableStorageCache.keys)
        imageCache.removeAll()
        immutableStorageCache.removeAll()
        currentMemoryBytes = 0

        if let onEvict {
            for image in evictedImages {
                onEvict(image)
            }
        }
        if let onImmutableStorageEvict {
            for storage in evictedStorage {
                onImmutableStorageEvict(storage)
            }
        }
    }

    public func invalidate() {
        clearAll()
        device = nil
    }

    public func evictStale(olderThan frameThreshold: UInt64) {
        let cutoffFrame = currentFrame > frameThreshold
            ? currentFrame - frameThreshold
            : 0
        let imageKeys = imageCache.compactMap { key, entry in
            entry.lastAccessFrame < cutoffFrame ? key : nil
        }
        let storageKeys = immutableStorageCache.compactMap { key, entry in
            entry.lastAccessFrame < cutoffFrame ? key : nil
        }

        var evictedImages: [CGImage] = []
        for key in imageKeys {
            if let entry = imageCache.removeValue(forKey: key) {
                currentMemoryBytes -= entry.memorySize
                evictedImages.append(entry.cgImage)
            }
        }
        var evictedStorage: [CGImageTextureStorage] = []
        for key in storageKeys {
            if let entry = immutableStorageCache.removeValue(forKey: key) {
                currentMemoryBytes -= entry.memorySize
                evictedStorage.append(key)
            }
        }
        if let onEvict {
            for image in evictedImages {
                onEvict(image)
            }
        }
        if let onImmutableStorageEvict {
            for storage in evictedStorage {
                onImmutableStorageEvict(storage)
            }
        }
    }

    private func evictIfNeeded(forNewMemory newMemory: UInt64) {
        while textureCount >= maxTextures {
            evictLeastRecentlyUsed()
        }
        while currentMemoryBytes + newMemory > maxMemoryBytes,
              textureCount > 0 {
            evictLeastRecentlyUsed()
        }
    }

    private func evictLeastRecentlyUsed() {
        guard textureCount > 0 else { return }
        var oldestFrame = UInt64.max
        var candidate: TextureEvictionCandidate?
        for (key, entry) in imageCache
        where entry.lastAccessFrame < oldestFrame {
            oldestFrame = entry.lastAccessFrame
            candidate = .image(key)
        }
        immutableStorageCache.forEach { key, entry in
            if entry.lastAccessFrame < oldestFrame {
                oldestFrame = entry.lastAccessFrame
                candidate = .immutableStorage(key)
            }
        }

        switch candidate {
        case .image(let key):
            guard let entry = imageCache.removeValue(forKey: key) else {
                preconditionFailure("Selected image cache entry disappeared")
            }
            currentMemoryBytes -= entry.memorySize
            onEvict?(entry.cgImage)
        case .immutableStorage(let storage):
            guard let entry = immutableStorageCache.removeValue(
                forKey: storage
            ) else {
                preconditionFailure(
                    "Selected immutable texture entry disappeared"
                )
            }
            currentMemoryBytes -= entry.memorySize
            onImmutableStorageEvict?(storage)
        case nil:
            preconditionFailure(
                "A non-empty texture cache must have an eviction candidate"
            )
        }
    }
}

#endif
