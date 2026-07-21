// Platform-agnostic rasterization-cache bookkeeping for Phase 3 (R3.2 /
// R3.4). The cache is generic over the renderer's texture handle type so
// CAWebGPURenderer can store `GPUTexture` while native tests substitute a
// stub. Pure Swift, no GPU symbols — the renderer is responsible for
// allocating, uploading, and destroying the actual texture; the cache
// only tracks identity, byte cost, and last-used frame.
//
// See PERFORMANCE_DESIGN.md §5.2.

import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif

// MARK: - Key

internal struct ReplicatorInstancePathComponent: Hashable, Sendable {
    internal let replicator: ObjectIdentifier
    internal let instanceIndex: Int
}

internal struct LayerRenderKey: Hashable, Sendable {
    internal let layer: ObjectIdentifier
    internal let replicatorPath: [ReplicatorInstancePathComponent]

    internal init(
        layer: ObjectIdentifier,
        replicatorPath: [ReplicatorInstancePathComponent] = []
    ) {
        self.layer = layer
        self.replicatorPath = replicatorPath
    }
}

/// A stable identity for a cache entry. Production keys combine model-layer
/// identity with the enclosing replicator instance path; tests can construct
/// keys from raw integers without manufacturing CALayers.
internal struct RasterizationCacheKey: Hashable, Sendable {
    private enum Storage: Hashable {
        case layer(LayerRenderKey)
        case raw(Int)
    }
    private let storage: Storage

    internal init(_ identifier: ObjectIdentifier) {
        self.storage = .layer(LayerRenderKey(layer: identifier))
    }
    internal init(_ renderKey: LayerRenderKey) {
        self.storage = .layer(renderKey)
    }
    internal init(raw: Int) {
        self.storage = .raw(raw)
    }
}

// MARK: - Entry

/// One cached rasterization. The cache owns the entry's bookkeeping
/// fields; the texture handle is opaque to the cache and is the
/// renderer's responsibility to release if it holds GPU resources.
internal struct RasterizedEntry<TextureRef> {
    internal var texture: TextureRef
    /// Captured pixel dimensions (`bounds.size × rasterizationScale`).
    internal var pixelSize: CGSize
    /// Hash of the inputs that determine the captured pixels (typically
    /// `bounds + transform`). The renderer compares this to detect
    /// content invalidation independent of the dirty-bit pathway.
    internal var contentBoundsHash: Int
    /// Frame at which this entry was last looked up or inserted; used by
    /// idle-eviction and byte-budget eviction.
    internal var lastUsedFrame: UInt64
    /// Approximate GPU byte cost (`width × height × 4` for RGBA8). Used
    /// only for eviction accounting; not for actual allocation.
    internal var byteCost: Int
}

// MARK: - Cache

/// LRU + byte-budget cache for `shouldRasterize` captures. Generic over
/// the renderer's texture handle type — `GPUTexture` for the WebGPU
/// renderer, a stub struct for unit tests.
///
/// The cache is **not thread-safe by itself**. WASM is single-threaded
/// and OpenCoreAnimation runs all rendering on the main actor in native
/// tests too, so callers do not need additional synchronisation.
internal final class RasterizationCache<TextureRef> {

    // MARK: Storage

    private var entries: [RasterizationCacheKey: RasterizedEntry<TextureRef>] = [:]
    /// The byte-cost ceiling. The cache enforces this only when the
    /// caller invokes `evictToBudget()`; inserts past the ceiling are
    /// allowed (so a single oversize entry can still land before an
    /// eviction pass runs).
    internal let maxBytes: Int

    // MARK: Counters (test/observability only)

    internal private(set) var hits: Int = 0
    internal private(set) var misses: Int = 0

    // MARK: Init

    internal init(maxBytes: Int) {
        self.maxBytes = maxBytes
    }

    // MARK: Public surface

    /// Aggregate byte cost of all live entries.
    internal var bytes: Int {
        entries.values.reduce(0) { $0 + $1.byteCost }
    }

    /// Number of live entries.
    internal var count: Int { entries.count }

    /// Direct entry access for tests / inspection. Production callers
    /// should use `lookup` so that hit-counters and `lastUsedFrame` get
    /// updated.
    internal func entry(_ key: RasterizationCacheKey) -> RasterizedEntry<TextureRef>? {
        entries[key]
    }

    /// Cache lookup. Increments `hits`/`misses` and updates the entry's
    /// `lastUsedFrame` to `frame` on a hit.
    internal func lookup(_ key: RasterizationCacheKey, atFrame frame: UInt64)
        -> RasterizedEntry<TextureRef>?
    {
        guard var entry = entries[key] else {
            misses &+= 1
            return nil
        }
        hits &+= 1
        entry.lastUsedFrame = frame
        entries[key] = entry
        return entry
    }

    /// Insert or replace an entry for `key`. Replacing reuses the slot
    /// without double-counting bytes.
    internal func insert(
        _ key: RasterizationCacheKey,
        texture: TextureRef,
        pixelSize: CGSize,
        contentBoundsHash: Int,
        atFrame frame: UInt64
    ) {
        let byteCost = byteCostOf(pixelSize: pixelSize)
        let entry = RasterizedEntry(
            texture: texture,
            pixelSize: pixelSize,
            contentBoundsHash: contentBoundsHash,
            lastUsedFrame: frame,
            byteCost: byteCost
        )
        entries[key] = entry
    }

    /// Drop a single entry; no-op if absent.
    internal func remove(_ key: RasterizationCacheKey) {
        entries.removeValue(forKey: key)
    }

    /// Drop every entry. Counters are preserved (they are observability
    /// data, not state). Tests that need pristine counters reset the
    /// whole instance.
    internal func removeAll() {
        entries.removeAll(keepingCapacity: true)
    }

    /// Drop entries whose `lastUsedFrame + olderThan < currentFrame`.
    /// PERFORMANCE_DESIGN.md §5.2 sets `olderThan = 6` (~100 ms @ 60 Hz).
    internal func evictIdle(currentFrame: UInt64, olderThan threshold: UInt64) {
        guard currentFrame > threshold else { return }
        let cutoff = currentFrame - threshold
        entries = entries.filter { _, entry in
            entry.lastUsedFrame >= cutoff
        }
    }

    /// Drop entries oldest-first (`lastUsedFrame` ascending) until the
    /// total byte cost is at or below `maxBytes`. Stable: ties on
    /// `lastUsedFrame` are broken by hash order, which is good enough
    /// because entries with identical `lastUsedFrame` were inserted in
    /// the same frame and are equally cold.
    internal func evictToBudget() {
        var live = bytes
        guard live > maxBytes else { return }
        let ordered = entries.sorted { $0.value.lastUsedFrame < $1.value.lastUsedFrame }
        for (key, entry) in ordered {
            if live <= maxBytes { break }
            entries.removeValue(forKey: key)
            live -= entry.byteCost
        }
    }

    // MARK: Internal helpers

    /// 4 bytes per pixel for RGBA8. Phase 3 only stores RGBA8 captures —
    /// the BGRA / sRGB question is decided by the renderer's pipeline,
    /// not by the cache.
    private func byteCostOf(pixelSize: CGSize) -> Int {
        let w = max(0, Int(pixelSize.width.rounded(.up)))
        let h = max(0, Int(pixelSize.height.rounded(.up)))
        return w * h * 4
    }
}
