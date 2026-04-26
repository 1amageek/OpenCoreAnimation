// Phase 3 — Renderer-level integration tests for backing store reuse and
// the `shouldRasterize` cache. These drive `MockCARenderer` (which
// records commands instead of issuing GPU work) so we can assert on
// "did the renderer issue a second upload?" without standing up a real
// GPU.
//
// Tests trace PERFORMANCE_DESIGN.md §5.7 sequence 3.1 → 3.9.

import Testing
import Foundation
@testable import OpenCoreAnimation
@testable import OpenCoreGraphics

#if canImport(Metal)

@Suite(.serialized)
struct RasterizationTests {

    init() { resetPerformanceTestState() }

    // MARK: Helpers

    /// A minimal opaque RGBA8 image. Identity is what the cache keys on,
    /// so two calls produce two distinct images by design.
    private func makeImage(width: Int = 4, height: Int = 4) -> CGImage {
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let bytesPerRow = width * 4
        let pixelData = Data(repeating: 255, count: bytesPerRow * height)
        let provider = CGDataProvider(data: pixelData)
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: .deviceRGB,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            fatalError("Expected CGImage")
        }
        return image
    }

    private func makeRenderer() async throws -> MockCARenderer {
        let renderer = MockCARenderer()
        try await renderer.initialize()
        renderer.resize(width: 64, height: 64)
        return renderer
    }

    // MARK: 3.1 — Backing store reuse (R3.1)

    /// Same image, no `setNeedsDisplay`, two consecutive frames must
    /// produce exactly one `writeTexture` call. R3.1 — once the renderer
    /// has uploaded a layer's `contents`, it must reuse the existing
    /// GPU texture until something invalidates it.
    @Test func contentsReuseAcrossFrames() async throws {
        let renderer = try await makeRenderer()
        let image = makeImage()
        let imageID = ObjectIdentifier(image as AnyObject)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 4, height: 4)
        layer.contents = image

        renderer.render(layer: layer)
        #expect(renderer.sink.uploadCount(for: imageID) == 1)

        renderer.render(layer: layer)
        #expect(renderer.sink.uploadCount(for: imageID) == 1,
                "second frame must reuse the cached texture")
    }

    /// Calling `setNeedsDisplay()` between frames flips the
    /// `.contentsRedraw` bit, which forces the renderer to re-upload.
    /// This is the negative case for R3.1 — it proves the reuse decision
    /// actually consults the dirty mask, not just "have we ever uploaded
    /// this image".
    @Test func setNeedsDisplayForcesReUpload() async throws {
        let renderer = try await makeRenderer()
        let image = makeImage()
        let imageID = ObjectIdentifier(image as AnyObject)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 4, height: 4)
        layer.contents = image

        renderer.render(layer: layer)
        #expect(renderer.sink.uploadCount(for: imageID) == 1)

        layer.setNeedsDisplay()
        renderer.render(layer: layer)
        #expect(renderer.sink.uploadCount(for: imageID) == 2)
    }

    // MARK: 3.2 / 3.3 — Rasterization cache (R3.2)

    /// `shouldRasterize` captures the layer + its subtree on the first
    /// frame, then a clean second frame must reuse the cached texture
    /// (no second capture). R3.2.
    @Test func shouldRasterizeCachesSubtree() async throws {
        let renderer = try await makeRenderer()

        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        parent.shouldRasterize = true

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        child.contents = makeImage()
        parent.addSublayer(child)

        let parentID = ObjectIdentifier(parent)

        renderer.render(layer: parent)
        #expect(renderer.sink.uploadCount(for: parentID) == 1)

        renderer.render(layer: parent)
        #expect(renderer.sink.uploadCount(for: parentID) == 1,
                "second clean frame must reuse the rasterized texture")
    }

    /// Mutating any descendant of a `shouldRasterize` layer between
    /// frames must invalidate the cached texture: `_subtreeDirtyCount`
    /// at the rasterized root is now > 0, so frame 2 recaptures. R3.2 +
    /// the dirty-propagation hook from Phase 1.
    @Test func descendantDirtyEvictsRasterizationCache() async throws {
        let renderer = try await makeRenderer()

        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        parent.shouldRasterize = true

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        parent.addSublayer(child)

        let parentID = ObjectIdentifier(parent)

        renderer.render(layer: parent)
        #expect(renderer.sink.uploadCount(for: parentID) == 1)

        // Mutate a descendant. Phase 1's propagation must bump
        // `parent._subtreeDirtyCount`, and the cache decision must
        // notice and re-capture.
        child.opacity = 0.5
        renderer.render(layer: parent)

        #expect(renderer.sink.uploadCount(for: parentID) == 2,
                "descendant mutation must force a recapture")
    }

    // MARK: 3.4a / 3.4b — Capture vs. composite opacity (R3.3)

    /// 3.4a — The pipeline state used to render *into* the offscreen
    /// rasterization texture must clear with α = 1.0 regardless of the
    /// layer's opacity. R3.3 separates per-pixel alpha (captured fully
    /// opaque) from the composite-time opacity multiplier.
    @Test func rasterizationPipelineExcludesOpacityFromCapture() async throws {
        let renderer = try await makeRenderer()

        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        parent.shouldRasterize = true
        parent.opacity = 0.3   // half-transparent

        renderer.render(layer: parent)

        let captureSnapshots: [RenderPipelineSnapshot] = renderer.sink.commands.compactMap {
            if case .setPipeline(let s) = $0, s.label == "rasterize-capture" {
                return s
            }
            return nil
        }
        let snapshot = try #require(captureSnapshots.first,
                                    "expected one rasterize-capture pipeline")
        #expect(snapshot.clearAlpha == 1.0,
                "capture-pass clearAlpha must be 1.0 even when layer.opacity < 1")
    }

    /// 3.4b — When the renderer composites the rasterized quad, the
    /// per-quad uniform `opacity` must equal the *current* layer opacity.
    /// The capture itself bakes nothing — opacity is a render-time
    /// multiplier applied at composite. R3.3 composite path.
    @Test func compositeMultipliesLayerOpacityAtRenderTime() async throws {
        let renderer = try await makeRenderer()

        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        parent.shouldRasterize = true
        parent.opacity = 0.3

        renderer.render(layer: parent)

        // The composite uniform follows the composite pipeline (last
        // setPipeline of the frame). Its `opacity` field must mirror
        // the model layer's *current* opacity.
        let compositeUniform: UniformPayload? = renderer.sink.commands.compactMap {
            if case .writeUniform(let p) = $0 { return p }
            return nil
        }.last
        let payload = try #require(compositeUniform)
        #expect(payload.opacity == Float(0.3))
    }

    // MARK: 3.5 / 3.6 — Cache eviction (R3.4)

    /// 3.5 — An entry untouched for more than 6 frames must be evicted
    /// by the post-submit idle pass (≈100 ms @ 60 Hz per WWDC 2014 #419).
    /// Re-rendering the same rasterized layer after the gap forces a
    /// recapture, proving the cache no longer holds the entry.
    @Test func cacheEvictsAfterIdleFrames() async throws {
        let renderer = try await makeRenderer()

        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        parent.shouldRasterize = true
        let parentID = ObjectIdentifier(parent)

        renderer.render(layer: parent)
        #expect(renderer.sink.uploadCount(for: parentID) == 1)

        // Render 7 frames against an unrelated root so the cached entry
        // sits idle and the post-submit eviction pass drops it.
        let other = CALayer()
        for _ in 0..<7 {
            renderer.render(layer: other)
        }

        renderer.render(layer: parent)
        #expect(renderer.sink.uploadCount(for: parentID) == 2,
                "idle entry must have been evicted, forcing a recapture")
    }

    /// 3.6 — When the cache exceeds its byte budget, the post-submit
    /// pass drops oldest-by-`lastUsedFrame` entries until under bound.
    /// Configure a tight budget that fits one entry but not two.
    @Test func cacheRespectsByteBudget() async throws {
        // Each 16x16 entry = 16*16*4 = 1024 bytes. Budget of 1500 fits
        // exactly one entry; a second 1024-byte entry pushes us to 2048
        // and must trigger eviction of the oldest.
        let renderer = MockCARenderer(rasterizationCacheMaxBytes: 1500)
        try await renderer.initialize()
        renderer.resize(width: 64, height: 64)

        let first = CALayer()
        first.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        first.shouldRasterize = true
        let firstID = ObjectIdentifier(first)

        let second = CALayer()
        second.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        second.shouldRasterize = true
        let secondID = ObjectIdentifier(second)

        renderer.render(layer: first)
        renderer.render(layer: second)

        // After frame 2 the cache should be back under budget — the
        // older `first` entry must have been evicted.
        #expect(renderer.rasterizationCache.bytes <= 1500)
        #expect(renderer.rasterizationCache.entry(RasterizationCacheKey(firstID)) == nil,
                "oldest entry must be evicted to honor the byte budget")
        #expect(renderer.rasterizationCache.entry(RasterizationCacheKey(secondID)) != nil)
    }

    // MARK: 3.7 — isOpaque (R3.5)

    /// 3.7 — When `isOpaque == true` and `opacity == 1.0`, the
    /// per-quad pipeline must skip alpha blending. R3.5 — `isOpaque`
    /// is a hint that lets the renderer drop the source-over blend.
    @Test func isOpaqueOmitsAlphaBlend() async throws {
        let renderer = try await makeRenderer()
        let image = makeImage()

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 4, height: 4)
        layer.contents = image
        layer.isOpaque = true

        renderer.render(layer: layer)

        let contentsPipelines: [RenderPipelineSnapshot] = renderer.sink.commands.compactMap {
            if case .setPipeline(let s) = $0, s.label == "contents" {
                return s
            }
            return nil
        }
        let snapshot = try #require(contentsPipelines.first,
                                    "expected one contents pipeline")
        #expect(snapshot.blendEnabled == false,
                "isOpaque layer at full opacity must not enable blend")
    }

    // MARK: 3.8 — shadowPath fast path (R3.6)

    /// 3.8 — When `shadowPath != nil`, the renderer must skip the
    /// contents-derived `shadow-mask` silhouette pipeline and instead
    /// tessellate the path directly via the `shadow-path` pipeline.
    /// R3.6 fast path.
    @Test func shadowPathSkipsSilhouetteExtraction() async throws {
        let renderer = try await makeRenderer()

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        layer.shadowOpacity = 1
        layer.shadowColor = CGColor(red: 0, green: 0, blue: 0, alpha: 1)

        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 16, height: 16))
        layer.shadowPath = path

        renderer.render(layer: layer)

        let labels: [String] = renderer.sink.commands.compactMap {
            if case .setPipeline(let s) = $0 { return s.label }
            return nil
        }
        #expect(!labels.contains("shadow-mask"),
                "shadowPath set — silhouette must not run")
        #expect(labels.contains("shadow-path"),
                "shadowPath fast path must emit the tessellated pipeline")
    }

    // MARK: 3.9 — Shadow prerender cache (R3.7)

    /// 3.9 — After a clean frame, the next frame's shadow prerender
    /// must reuse the cached blurred texture instead of re-running
    /// silhouette extraction + blur. R3.7 — same skip semantics as
    /// the rasterization cache, but for the blurred-shadow texture.
    @Test func shadowPrerenderSkipsCleanSubtree() async throws {
        let renderer = try await makeRenderer()

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        layer.shadowOpacity = 1
        layer.shadowColor = CGColor(red: 0, green: 0, blue: 0, alpha: 1)

        renderer.render(layer: layer)
        let frame1Labels: [String] = renderer.sink.commands.compactMap {
            if case .setPipeline(let s) = $0 { return s.label }
            return nil
        }
        #expect(frame1Labels.contains("shadow-blur"),
                "first frame must run the blur pass")

        renderer.sink.clear()
        renderer.render(layer: layer)
        let frame2Labels: [String] = renderer.sink.commands.compactMap {
            if case .setPipeline(let s) = $0 { return s.label }
            return nil
        }
        #expect(!frame2Labels.contains("shadow-blur"),
                "clean frame must reuse the cached blurred texture")
        #expect(!frame2Labels.contains("shadow-mask"),
                "clean frame must not re-extract the silhouette")
        #expect(frame2Labels.contains("shadow-composite"),
                "cache hit must composite the cached texture")
    }
}

#endif
