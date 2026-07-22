import Testing
@testable import OpenCoreAnimation

#if canImport(Metal)
import Metal

@MainActor
@Suite("CAAnimationEngine native renderer")
struct CAAnimationEngineRendererTests {
    @Test("renderFrame submits the root layer to a real offscreen Metal target")
    func renderFrameSubmitsMetalPixels() throws {
        let engine = CAAnimationEngine()
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        engine.rootLayer = root

        engine.renderFrame()

        let renderer = try #require(engine.rendererDelegate as? CAMetalRenderer)
        #expect(renderer.lastRenderError == nil)
        let commandBuffer = try #require(renderer.lastCommandBuffer)
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let texture = try #require(renderer.targetTexture)

        var pixel = [UInt8](repeating: 0, count: 4)
        pixel.withUnsafeMutableBytes { bytes in
            guard let destination = bytes.baseAddress else { return }
            texture.getBytes(
                destination,
                bytesPerRow: 4,
                from: MTLRegionMake2D(8, 8, 1, 1),
                mipmapLevel: 0
            )
        }
        #expect(pixel == [0, 255, 0, 255])
        #expect(root._dirtyMask.isEmpty)
    }

    @Test("invalid root dimensions fail without consuming dirty state")
    func invalidRootDimensionsFailExplicitly() throws {
        let engine = CAAnimationEngine()
        let root = CALayer()
        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        engine.rootLayer = root

        engine.renderFrame()

        let renderer = try #require(engine.rendererDelegate as? CAMetalRenderer)
        guard case .renderingFailed = renderer.lastRenderError else {
            Issue.record("Expected invalid root dimensions to produce a rendering error")
            return
        }
        #expect(renderer.lastCommandBuffer == nil)
        #expect(renderer.targetTexture == nil)
        #expect(root._dirtyMask.isEmpty == false)
    }

    @Test("invalid explicit resize is reported and does not allocate a target")
    func invalidExplicitResizeFailsExplicitly() async throws {
        let renderer = CAMetalRenderer()
        try await renderer.initialize()

        renderer.resize(width: 0, height: 16)

        guard case .renderingFailed = renderer.lastRenderError else {
            Issue.record("Expected an invalid resize to produce a rendering error")
            return
        }
        #expect(renderer.targetTexture == nil)
    }
}
#endif
