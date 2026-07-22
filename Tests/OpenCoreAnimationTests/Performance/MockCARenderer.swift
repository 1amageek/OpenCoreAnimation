// Phase 3 — A native-only renderer-backend implementation that records the
// command stream a production WebGPU renderer would issue. Phase 3 / 4
// tests assert against this stream rather than against pixels.
//
// The mock shares its decision logic with `CAWebGPURenderer` via the
// platform-agnostic `RasterizationDecisions` enum and stores cached
// captures in the same `RasterizationCache` type. That way the mock can
// only diverge from production in I/O (no real GPU), not in *what* it
// would record.
//
// See PERFORMANCE_DESIGN.md §10 (test infra) and §5.7 for the test
// sequence the mock unblocks.

import Foundation
@testable import OpenCoreAnimation
@testable import OpenCoreGraphics

#if canImport(Metal)

// MARK: - Recorded command stream

/// Stand-in for a GPU texture handle. The mock never allocates a real
/// texture; it only tracks identity + byte cost so the cache can verify
/// reuse and budget accounting.
internal struct MockTextureRef: Equatable, Sendable {
    let id: ObjectIdentifier
    let byteCount: Int
}

/// One operation the renderer would issue. Matches the surface of
/// `RenderCommandSink` 1:1 so tests can pattern-match on the call log.
internal enum RecordedCommand: Equatable {
    case writeTexture(key: ObjectIdentifier, byteCount: Int)
    case setPipeline(RenderPipelineSnapshot)
    case writeUniform(UniformPayload)
    case dispatchQuad(layerID: ObjectIdentifier)
    case submit
    case frameDidEnd(frameToken: UInt64)
}

/// Concrete `RenderCommandSink` that appends every call into a buffer.
internal final class RecordingCommandSink: RenderCommandSink {
    internal private(set) var commands: [RecordedCommand] = []

    internal func writeTexture(key: ObjectIdentifier, byteCount: Int) {
        commands.append(.writeTexture(key: key, byteCount: byteCount))
    }
    internal func setPipeline(_ snapshot: RenderPipelineSnapshot) {
        commands.append(.setPipeline(snapshot))
    }
    internal func writeUniform(_ payload: UniformPayload) {
        commands.append(.writeUniform(payload))
    }
    internal func dispatchQuad(layerID: ObjectIdentifier) {
        commands.append(.dispatchQuad(layerID: layerID))
    }
    internal func submit() {
        commands.append(.submit)
    }
    internal func frameDidEnd(frameToken: UInt64) {
        commands.append(.frameDidEnd(frameToken: frameToken))
    }

    internal func clear() {
        commands.removeAll(keepingCapacity: true)
    }

    /// Test convenience: how many times we asked the GPU to upload bytes
    /// for a given source identity. R3.1 hinges on this being exactly 1
    /// across N frames of the same image. R3.2 reuses the same predicate
    /// against a layer's `ObjectIdentifier` (rasterized capture key).
    internal func uploadCount(for key: ObjectIdentifier) -> Int {
        commands.reduce(0) { acc, cmd in
            if case .writeTexture(let recordedKey, _) = cmd, recordedKey == key {
                return acc + 1
            }
            return acc
        }
    }
}

// MARK: - MockCARenderer

/// Native-only renderer backend that drives the same Phase 3 decision logic
/// as `CAWebGPURenderer` but records commands into a `RecordingCommandSink`
/// instead of issuing real WebGPU calls.
internal final class MockCARenderer: CARendererDelegate {

    internal var size: CGSize = CGSize(width: 0, height: 0)

    /// Exposed for tests — assert against `sink.commands`.
    internal let sink: RecordingCommandSink

    /// Cache of already-uploaded contents textures, keyed by image
    /// identity. Mirrors the production renderer's `GPUTextureManager`
    /// cache (R3.1) at the level of detail this mock needs.
    private var uploadedImageIdentities: Set<ObjectIdentifier> = []

    /// Rasterization cache (R3.2 / R3.4). Allocated per-instance so each
    /// test starts with a clean cache without touching globals. Exposed
    /// (read-only conceptually — tests should not mutate it) so 3.5/3.6
    /// can assert on byte/entry counts after eviction passes.
    internal let rasterizationCache: RasterizationCache<MockTextureRef>

    /// Identities of layers whose blurred shadow texture has been
    /// prerendered and is still valid. R3.7 — present means "skip the
    /// blur pass and reuse"; absent or invalidated means "blur again".
    private var prerenderedShadowIdentities: Set<ObjectIdentifier> = []

    internal init(
        sink: RecordingCommandSink = RecordingCommandSink(),
        rasterizationCacheMaxBytes: Int = 4 * 1024 * 1024
    ) {
        self.sink = sink
        self.rasterizationCache = RasterizationCache(maxBytes: rasterizationCacheMaxBytes)
    }

    // MARK: CARendererDelegate

    internal func initialize() async throws {
        // Nothing to do — no real device/queue/pipeline behind this mock.
    }

    internal func resize(width: Int, height: Int) {
        size = CGSize(width: width, height: height)
    }

    internal func render(layer rootLayer: CALayer) {
        // Mirror the production renderer's commit-cycle housekeeping
        // (PERFORMANCE_DESIGN.md §3.6 / §3.8): bump the per-frame token
        // *before* any cache lookup, and clear the dirty subtree *after*
        // submit so setters that fire on the same tick re-mark for the
        // next frame.
        let frameToken = CALayer.advanceFrameToken()

        renderLayer(rootLayer)

        sink.submit()
        // R3.4: idle + budget eviction runs after submit. The threshold
        // (6 frames ≈ 100 ms @ 60 Hz) is the design-doc default.
        rasterizationCache.evictIdle(currentFrame: frameToken, olderThan: 6)
        rasterizationCache.evictToBudget()
        sink.frameDidEnd(frameToken: frameToken)
        rootLayer.recursivelyClearDirtyAfterCommit()
    }

    internal func invalidate() {
        rasterizationCache.removeAll()
        uploadedImageIdentities.removeAll()
        prerenderedShadowIdentities.removeAll()
        sink.clear()
    }

    // MARK: Layer walk

    private func renderLayer(_ layer: CALayer) {
        // Rasterization cache path (R3.2 / R3.3 / R3.4). Checked first:
        // a `shouldRasterize` layer captures itself + its subtree into a
        // single offscreen texture, so we must NOT recurse into children
        // (they're "inside" the cached texture).
        if layer.shouldRasterize {
            emitRasterizedSubtree(layer)
            return
        }

        // Shadow prerender path (R3.6 / R3.7). The shadow texture is
        // produced upstream of the layer's own quad — emit it before the
        // contents pass so tests can match on command order.
        if layer.shadowOpacity > 0 {
            emitShadow(for: layer)
        }

        // Backing-store path (R3.1). The mock only models the contents
        // texture — flat-colour layers do not exercise the upload path
        // we want to assert on.
        if let image = layer.contents as? CGImage {
            emitContentsCommands(for: layer, image: image)
        }

        if let sublayers = layer.sublayers {
            for sub in sublayers {
                renderLayer(sub)
            }
        }
    }

    /// Shadow rendering (R3.6 silhouette path + R3.7 blur cache).
    /// Emits one of two silhouette pipelines depending on whether
    /// `shadowPath` is set, then a blur pipeline iff the prerender
    /// cache cannot be reused. Tests 3.8 / 3.9 match on these labels.
    private func emitShadow(for layer: CALayer) {
        let layerID = ObjectIdentifier(layer)
        let canReuse = RasterizationDecisions.canReusePrerenderCache(
            contributorLayer: layer,
            hasCachedTexture: prerenderedShadowIdentities.contains(layerID)
        )

        if canReuse {
            // Cache hit — composite the cached blurred texture without
            // re-running silhouette extraction or blur passes.
            sink.setPipeline(RenderPipelineSnapshot(
                blendEnabled: true,
                clearAlpha: RasterizationDecisions.captureClearAlpha(),
                label: "shadow-composite"
            ))
            return
        }

        // Cache miss — silhouette + blur. R3.6: shadowPath skips the
        // contents-derived `shadow-mask` pipeline in favour of a
        // tessellated `shadow-path` pipeline.
        let silhouetteLabel = RasterizationDecisions.useShadowPathFastPath(for: layer)
            ? "shadow-path"
            : "shadow-mask"
        sink.setPipeline(RenderPipelineSnapshot(
            blendEnabled: false,
            clearAlpha: RasterizationDecisions.captureClearAlpha(),
            label: silhouetteLabel
        ))

        sink.setPipeline(RenderPipelineSnapshot(
            blendEnabled: false,
            clearAlpha: RasterizationDecisions.captureClearAlpha(),
            label: "shadow-blur"
        ))

        prerenderedShadowIdentities.insert(layerID)
    }

    private func emitContentsCommands(for layer: CALayer, image: CGImage) {
        let imageID = ObjectIdentifier(image as AnyObject)
        let bytes = image.width * image.height * 4

        // R3.1: skip the upload iff the layer's `.contentsRedraw` bit
        // is clean AND the renderer has already uploaded this image.
        let canReuse = RasterizationDecisions.canReuseContentsTexture(layer: layer)
            && uploadedImageIdentities.contains(imageID)

        if !canReuse {
            sink.writeTexture(key: imageID, byteCount: bytes)
            uploadedImageIdentities.insert(imageID)
        }

        let pipeline = RenderPipelineSnapshot(
            blendEnabled: RasterizationDecisions.blendEnabled(for: layer),
            clearAlpha: RasterizationDecisions.captureClearAlpha(),
            label: "contents"
        )
        sink.setPipeline(pipeline)
        sink.writeUniform(UniformPayload(
            opacity: RasterizationDecisions.compositeOpacity(for: layer),
            hasTexture: true
        ))
        sink.dispatchQuad(layerID: ObjectIdentifier(layer))
    }

    private func emitRasterizedSubtree(_ layer: CALayer) {
        let key = RasterizationCacheKey(ObjectIdentifier(layer))
        let pixelSize = CGSize(
            width: max(1, layer.bounds.width * layer.rasterizationScale),
            height: max(1, layer.bounds.height * layer.rasterizationScale)
        )
        let hash = contentBoundsHash(for: layer)
        let cached = rasterizationCache.entry(key)

        let canReuse = RasterizationDecisions.canReuseRasterizedTexture(
            layer: layer,
            cached: cached,
            currentContentBoundsHash: hash
        )

        if canReuse {
            // Cache hit — touch lastUsedFrame so eviction sees this entry
            // as fresh, then composite the cached texture without a
            // capture pass.
            _ = rasterizationCache.lookup(key, atFrame: CALayer._currentFrameToken)
        } else {
            // Cache miss — capture pass. R3.3 mandates the offscreen
            // pipeline use clearAlpha = 1.0 regardless of layer.opacity
            // (opacity is a *composite-time* multiplier, applied below).
            let bytes = max(1, Int(pixelSize.width) * Int(pixelSize.height) * 4)
            let texture = MockTextureRef(id: ObjectIdentifier(layer), byteCount: bytes)

            let capturePipeline = RenderPipelineSnapshot(
                blendEnabled: false,
                clearAlpha: RasterizationDecisions.captureClearAlpha(),
                label: "rasterize-capture"
            )
            sink.setPipeline(capturePipeline)
            sink.writeTexture(key: ObjectIdentifier(layer), byteCount: bytes)

            rasterizationCache.insert(
                key,
                texture: texture,
                pixelSize: pixelSize,
                contentBoundsHash: hash,
                atFrame: CALayer._currentFrameToken
            )
        }

        // Composite pass — same shape whether we just captured or hit
        // the cache. The composite uniform multiplies in the *current*
        // layer.opacity (R3.3 composite path).
        let compositePipeline = RenderPipelineSnapshot(
            blendEnabled: RasterizationDecisions.blendEnabled(for: layer),
            clearAlpha: RasterizationDecisions.captureClearAlpha(),
            label: "rasterize-composite"
        )
        sink.setPipeline(compositePipeline)
        sink.writeUniform(UniformPayload(
            opacity: RasterizationDecisions.compositeOpacity(for: layer),
            hasTexture: true
        ))
        sink.dispatchQuad(layerID: ObjectIdentifier(layer))
    }

    /// Hash of the inputs that determine the captured pixels. Bounds and
    /// rasterization scale are the only knobs the *self* layer exposes
    /// that affect its own captured pixels (the parent transform is a
    /// composite-time uniform, not a capture-time input — see §5.6).
    private func contentBoundsHash(for layer: CALayer) -> Int {
        var hasher = Hasher()
        hasher.combine(layer.bounds.origin.x)
        hasher.combine(layer.bounds.origin.y)
        hasher.combine(layer.bounds.size.width)
        hasher.combine(layer.bounds.size.height)
        hasher.combine(layer.rasterizationScale)
        return hasher.finalize()
    }
}

#endif
