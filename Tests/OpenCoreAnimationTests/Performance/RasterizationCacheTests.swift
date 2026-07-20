// Phase 3 — Cache-level unit tests for the platform-agnostic
// RasterizationCache. Drives cache-internal logic only; the renderer
// integration tests live in RasterizationTests.swift (later commits).
//
// Tests mirror PERFORMANCE_DESIGN.md §5.2 / §5.7 sequence 3.5–3.6 plus
// pre-requisite hit/miss/touch invariants the higher-level tests rely on.

import Testing
import Foundation
@testable import OpenCoreAnimation

#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif

// A trivially-`Sendable` placeholder for the texture handle the cache
// hands back. The cache does not interpret it, only stores it.
private struct StubTexture: Equatable, Sendable {
    let id: Int
}

extension PerformanceTests {
@Suite
struct RasterizationCacheTests {

    init() { resetPerformanceTestState() }

    private func makeKey(_ raw: Int) -> RasterizationCacheKey {
        // The cache keys real entries by ObjectIdentifier in production,
        // but `RasterizationCacheKey` exposes a raw-Int initializer for
        // tests so we can construct keys without manufacturing CALayers.
        RasterizationCacheKey(raw: raw)
    }

    // C.1 — A miss returns nil and does not increment the hit counter.
    @Test func missReturnsNil() {
        let cache = RasterizationCache<StubTexture>(maxBytes: 1024)
        let result = cache.lookup(makeKey(1), atFrame: 0)
        #expect(result == nil)
        #expect(cache.hits == 0)
        #expect(cache.misses == 1)
    }

    // C.2 — Insert + lookup at the same frame returns the entry and
    // increments hits.
    @Test func insertThenLookupHits() {
        let cache = RasterizationCache<StubTexture>(maxBytes: 1024)
        let key = makeKey(1)
        cache.insert(
            key,
            texture: StubTexture(id: 7),
            pixelSize: CGSize(width: 4, height: 4),
            contentBoundsHash: 0,
            atFrame: 0
        )
        let entry = cache.lookup(key, atFrame: 0)
        #expect(entry?.texture == StubTexture(id: 7))
        #expect(cache.hits == 1)
        #expect(cache.misses == 0)
    }

    // C.3 — Inserting with a known per-pixel byte cost updates `bytes`.
    @Test func bytesAccountedOnInsert() {
        let cache = RasterizationCache<StubTexture>(maxBytes: 1_000_000)
        let pixels = CGSize(width: 8, height: 8)
        let expectedBytes = 8 * 8 * 4
        cache.insert(
            makeKey(1),
            texture: StubTexture(id: 1),
            pixelSize: pixels,
            contentBoundsHash: 0,
            atFrame: 0
        )
        #expect(cache.bytes == expectedBytes)
    }

    // C.4 — `lookup(... atFrame:)` updates `lastUsedFrame`. This is what
    // protects an entry from idle-eviction the next frame.
    @Test func lookupTouchesLastUsedFrame() {
        let cache = RasterizationCache<StubTexture>(maxBytes: 1024)
        let key = makeKey(1)
        cache.insert(
            key,
            texture: StubTexture(id: 1),
            pixelSize: CGSize(width: 1, height: 1),
            contentBoundsHash: 0,
            atFrame: 0
        )
        _ = cache.lookup(key, atFrame: 5)
        #expect(cache.entry(key)?.lastUsedFrame == 5)
    }

    // 3.5 — Idle eviction: an entry untouched for >6 frames is dropped.
    @Test func evictIdleDropsStaleEntries() {
        let cache = RasterizationCache<StubTexture>(maxBytes: 1024)
        cache.insert(
            makeKey(1),
            texture: StubTexture(id: 1),
            pixelSize: CGSize(width: 4, height: 4),
            contentBoundsHash: 0,
            atFrame: 0
        )
        cache.insert(
            makeKey(2),
            texture: StubTexture(id: 2),
            pixelSize: CGSize(width: 4, height: 4),
            contentBoundsHash: 0,
            atFrame: 5
        )
        // Current frame is 7; threshold is 6 frames. Entry 1 (last used 0)
        // is older than 6 → drop. Entry 2 (last used 5) is still fresh.
        cache.evictIdle(currentFrame: 7, olderThan: 6)
        #expect(cache.entry(makeKey(1)) == nil)
        #expect(cache.entry(makeKey(2)) != nil)
    }

    // 3.6 — Byte-budget eviction: when `bytes > maxBytes` after an insert,
    // drop oldest by `lastUsedFrame` until under budget. (Eviction is a
    // separate call so callers can decide *when* to trim — typically right
    // after `submit`.)
    @Test func evictToBudgetDropsOldestFirst() {
        // Budget = 128 bytes. Each 4x4 entry is 64 bytes. Inserting three
        // of them puts the cache at 192 bytes — over budget.
        let cache = RasterizationCache<StubTexture>(maxBytes: 128)
        let pix = CGSize(width: 4, height: 4)
        cache.insert(makeKey(1), texture: .init(id: 1),
                     pixelSize: pix, contentBoundsHash: 0, atFrame: 0)
        cache.insert(makeKey(2), texture: .init(id: 2),
                     pixelSize: pix, contentBoundsHash: 0, atFrame: 1)
        cache.insert(makeKey(3), texture: .init(id: 3),
                     pixelSize: pix, contentBoundsHash: 0, atFrame: 2)
        #expect(cache.bytes == 192)

        cache.evictToBudget()
        #expect(cache.bytes <= 128)
        // Oldest (lastUsedFrame == 0) must go first.
        #expect(cache.entry(makeKey(1)) == nil)
        #expect(cache.entry(makeKey(3)) != nil)
    }

    // C.5 — Re-inserting at the same key replaces the entry and keeps the
    // byte accounting consistent (no double-count).
    @Test func reinsertReplacesNotAccumulates() {
        let cache = RasterizationCache<StubTexture>(maxBytes: 1024)
        let key = makeKey(1)
        cache.insert(
            key,
            texture: StubTexture(id: 1),
            pixelSize: CGSize(width: 4, height: 4),
            contentBoundsHash: 0,
            atFrame: 0
        )
        cache.insert(
            key,
            texture: StubTexture(id: 2),
            pixelSize: CGSize(width: 8, height: 8),
            contentBoundsHash: 1,
            atFrame: 1
        )
        #expect(cache.bytes == 8 * 8 * 4)
        #expect(cache.entry(key)?.texture == StubTexture(id: 2))
    }

    // C.6 — `removeAll()` drops every entry and zeros the byte counter.
    @Test func removeAllClearsCacheAndBytes() {
        let cache = RasterizationCache<StubTexture>(maxBytes: 1024)
        cache.insert(makeKey(1), texture: .init(id: 1),
                     pixelSize: CGSize(width: 4, height: 4),
                     contentBoundsHash: 0, atFrame: 0)
        cache.removeAll()
        #expect(cache.bytes == 0)
        #expect(cache.entry(makeKey(1)) == nil)
    }
}
}
