import Testing
@testable import OpenCoreAnimation

#if canImport(Metal)
import Metal
#endif

@MainActor
@Suite("CARenderer frame lifecycle")
struct CARendererLifecycleTests {
    private final class RecordingBackend: CARendererDelegate {
        var size: CGSize = CGSize(width: 64, height: 32)
        private(set) var renderCount = 0
        private(set) var renderedOpacities: [Float] = []

        func initialize() async throws {}

        func resize(width: Int, height: Int) {
            size = CGSize(width: width, height: height)
        }

        func render(layer rootLayer: CALayer) {
            CALayer.advanceFrameToken()
            renderCount += 1
            renderedOpacities.append(rootLayer._renderTimePresentation().opacity)
            rootLayer.recursivelyClearDirtyAfterCommit()
            rootLayer.completeTransactionsAfterRenderRecursively()
        }

        func invalidate() {}
    }

    private final class RecordingLayoutManager: CALayoutManager {
        private(set) var layoutCount = 0

        func layoutSublayers(of layer: CALayer) {
            layoutCount += 1
        }
    }

    @Test("Frame regions are discovered, unioned, rendered, and released")
    func updateLifecycle() {
        let backend = RecordingBackend()
        let renderer = CARenderer(backend: backend)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)
        root.position = CGPoint(x: 20, y: 10)
        renderer.layer = root
        renderer.bounds = CGRect(x: 4, y: 5, width: 40, height: 20)

        renderer.render()
        #expect(backend.renderCount == 0)

        renderer.beginFrame(atTime: 10, timeStamp: nil)
        #expect(renderer.updateBounds() == CGRect(x: 4, y: 5, width: 36, height: 15))
        renderer.render()
        #expect(backend.renderCount == 1)
        renderer.endFrame()
        #expect(isNull(renderer.updateBounds()))

        renderer.beginFrame(atTime: 11, timeStamp: nil)
        #expect(isNull(renderer.updateBounds()))
        renderer.render()
        #expect(backend.renderCount == 1)

        renderer.addUpdate(CGRect(x: -20, y: -20, width: 10, height: 10))
        renderer.addUpdate(CGRect(x: 2, y: 3, width: 5, height: 6))
        #expect(renderer.updateBounds() == CGRect(x: -20, y: -20, width: 27, height: 29))
        renderer.render()
        #expect(backend.renderCount == 2)
        renderer.endFrame()
    }

    @Test("Frame time drives presentation values and next-frame scheduling")
    func animationTiming() {
        let backend = RecordingBackend()
        let renderer = CARenderer(backend: backend)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)
        root.position = CGPoint(x: 20, y: 10)
        renderer.layer = root
        renderer.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)

        let start = CACurrentMediaTime() + 5
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.beginTime = start
        animation.duration = 2
        root.add(animation, forKey: "opacity")

        renderer.beginFrame(atTime: start - 1, timeStamp: nil)
        #expect(abs(renderer.nextFrameTime() - start) < 0.000_001)
        renderer.render()
        #expect(backend.renderedOpacities.last == 1)
        renderer.endFrame()

        renderer.beginFrame(atTime: start + 0.5, timeStamp: nil)
        #expect(abs(renderer.nextFrameTime() - (start + 0.5)) < 0.000_001)
        #expect(!isNull(renderer.updateBounds()))
        renderer.render()
        #expect(abs((backend.renderedOpacities.last ?? -1) - 0.25) < 0.000_001)
        renderer.endFrame()

        let paused = CABasicAnimation(keyPath: "opacity")
        paused.fromValue = Float(0)
        paused.toValue = Float(1)
        paused.beginTime = start + 3
        paused.duration = 10
        paused.speed = 0
        paused.timeOffset = 2
        root.removeAllAnimations()
        root.add(paused, forKey: "paused")
        renderer.beginFrame(atTime: start + 4, timeStamp: nil)
        #expect(renderer.nextFrameTime() == .infinity)
        renderer.endFrame()
    }

    @Test("Mask animations participate in scheduling and terminal rendering")
    func maskAnimationScheduling() {
        let backend = RecordingBackend()
        let renderer = CARenderer(backend: backend)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)
        root.position = CGPoint(x: 20, y: 10)
        let mask = CALayer()
        mask.bounds = root.bounds
        mask.position = root.position
        root.mask = mask
        renderer.layer = root
        renderer.bounds = root.bounds

        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.beginTime = 20
        animation.duration = 1
        mask.add(animation, forKey: "mask-opacity")

        renderer.beginFrame(atTime: 19, timeStamp: nil)
        #expect(renderer.nextFrameTime() == 20)
        renderer.render()
        renderer.endFrame()

        setStoredAnimationBeginTime(17, on: mask, forKey: "mask-opacity")
        renderer.beginFrame(atTime: 19, timeStamp: nil)
        #expect(renderer.nextFrameTime() == 19)
        #expect(renderer.updateBounds() == root.bounds)
        renderer.render()
        renderer.endFrame()

        #expect(mask.animation(forKey: "mask-opacity") == nil)
    }

    @Test("Mask trees complete pending layout before explicit rendering")
    func maskLayout() {
        let backend = RecordingBackend()
        let renderer = CARenderer(backend: backend)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)
        root.position = CGPoint(x: 20, y: 10)
        let mask = CALayer()
        let layoutManager = RecordingLayoutManager()
        mask.layoutManager = layoutManager
        root.mask = mask
        renderer.layer = root
        renderer.bounds = root.bounds

        renderer.beginFrame(atTime: 1, timeStamp: nil)
        renderer.render()
        renderer.endFrame()

        #expect(layoutManager.layoutCount == 1)
        #expect(!mask.needsLayout())
    }

    @Test("Update bounds include overflowing descendants and their previous pixels")
    func descendantUpdateExtent() {
        let backend = RecordingBackend()
        let renderer = CARenderer(backend: backend)
        renderer.bounds = CGRect(x: 0, y: 0, width: 60, height: 20)

        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
        root.position = CGPoint(x: 10, y: 10)
        renderer.layer = root

        renderer.beginFrame(atTime: 1, timeStamp: nil)
        #expect(renderer.updateBounds() == CGRect(x: 0, y: 0, width: 20, height: 20))
        renderer.render()
        renderer.endFrame()

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 30, height: 20)
        child.position = CGPoint(x: 35, y: 10)
        root.addSublayer(child)
        renderer.beginFrame(atTime: 2, timeStamp: nil)
        #expect(renderer.updateBounds() == CGRect(x: 0, y: 0, width: 50, height: 20))
        renderer.render()
        renderer.endFrame()

        child.removeFromSuperlayer()
        renderer.beginFrame(atTime: 3, timeStamp: nil)
        #expect(renderer.updateBounds() == CGRect(x: 0, y: 0, width: 50, height: 20))
        renderer.render()
        renderer.endFrame()

        renderer.beginFrame(atTime: 4, timeStamp: nil)
        #expect(isNull(renderer.updateBounds()))
        renderer.endFrame()
    }

    #if canImport(Metal)
    @Test("Public Metal renderer submits pixels to its destination texture")
    func metalDestination() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererTestError.metalUnavailable
        }
        let texture = try makeTexture(device: device)

        let renderer = CARenderer(mtlTexture: texture)
        renderer.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        let root = CALayer()
        root.bounds = renderer.bounds
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        renderer.layer = root

        renderer.beginFrame(atTime: CACurrentMediaTime(), timeStamp: nil)
        renderer.render()
        renderer.endFrame()

        let metalBackend = try #require(renderer.backend as? CAMetalRenderer)
        let commandBuffer = try #require(metalBackend.lastCommandBuffer)
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

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
        #expect(pixel == [0, 0, 255, 255])
    }

    @Test("Metal renderer reports invalid options without trapping")
    func metalOptionFailures() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererTestError.metalUnavailable
        }
        let texture = try makeTexture(device: device)

        let unknownOptionRenderer = CARenderer(
            mtlTexture: texture,
            options: ["futureOption": true]
        )
        #expect(
            unknownOptionRenderer.lastRenderError
                == .unsupportedRendererOption("futureOption")
        )
        unknownOptionRenderer.setDestination(texture)
        #expect(unknownOptionRenderer.lastRenderError == nil)

        let invalidQueueRenderer = CARenderer(
            mtlTexture: texture,
            options: [kCARendererMetalCommandQueue: 42]
        )
        if case .invalidRendererOption(
            let key,
            let expected,
            _
        ) = invalidQueueRenderer.lastRenderError {
            #expect(key == kCARendererMetalCommandQueue)
            #expect(expected == "MTLCommandQueue")
        } else {
            Issue.record("Expected an invalid renderer option error")
        }

        let unsupportedColorRenderer = CARenderer(
            mtlTexture: texture,
            options: [
                kCARendererColorSpace:
                    OpenCoreGraphics.CGColorSpace.deviceGray
            ]
        )
        if case .unsupportedRendererColorSpace =
                unsupportedColorRenderer.lastRenderError {
            #expect(true)
        } else {
            Issue.record(
                "Expected an unsupported renderer color-space error"
            )
        }

        let invalidDestinationRenderer = CARenderer(
            mtlTexture: try makeTexture(
                device: device,
                usage: [.shaderRead]
            )
        )
        #expect(
            invalidDestinationRenderer.lastRenderError
                == .metalDestinationMissingRenderTargetUsage
        )
    }

    @Test("Metal renderer uses a client-owned command queue")
    func metalCommandQueueOption() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererTestError.metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw CARendererTestError.commandQueueUnavailable
        }
        let texture = try makeTexture(device: device)
        let renderer = CARenderer(
            mtlTexture: texture,
            options: [
                kCARendererMetalCommandQueue: commandQueue
            ]
        )

        #expect(renderer.lastRenderError == nil)
        let backend = try #require(
            renderer.backend as? CAMetalRenderer
        )
        let configuredQueue = try #require(
            backend.configuredCommandQueue
        )
        #expect(
            ObjectIdentifier(configuredQueue)
                == ObjectIdentifier(commandQueue)
        )
    }

    @Test("Metal destination replacement is atomic")
    func metalDestinationReplacement() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererTestError.metalUnavailable
        }
        let originalTexture = try makeTexture(device: device)
        let renderer = CARenderer(mtlTexture: originalTexture)
        let backend = try #require(
            renderer.backend as? CAMetalRenderer
        )

        let invalidTexture = try makeTexture(
            device: device,
            usage: [.shaderRead]
        )
        renderer.setDestination(invalidTexture)
        #expect(
            renderer.lastRenderError
                == .metalDestinationMissingRenderTargetUsage
        )
        let retainedTexture = try #require(backend.targetTexture)
        #expect(
            ObjectIdentifier(retainedTexture)
                == ObjectIdentifier(originalTexture)
        )

        let replacementTexture = try makeTexture(device: device)
        renderer.setDestination(replacementTexture)
        #expect(renderer.lastRenderError == nil)
        let configuredTexture = try #require(backend.targetTexture)
        #expect(
            ObjectIdentifier(configuredTexture)
                == ObjectIdentifier(replacementTexture)
        )
    }

    @Test("Metal destination rejects an incompatible backend without trapping")
    func incompatibleMetalDestination() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererTestError.metalUnavailable
        }
        let renderer = CARenderer(backend: RecordingBackend())
        renderer.setDestination(try makeTexture(device: device))
        #expect(
            renderer.lastRenderError
                == .incompatibleRendererBackend
        )
    }

    @Test("Metal renderer retains immutable resources for every layer draw")
    func metalMultiLayerResources() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererTestError.metalUnavailable
        }
        let texture = try makeTexture(device: device)
        let candidateColorSpace =
            OpenCoreGraphics.CGColorSpace(
                name: OpenCoreGraphics.CGColorSpace.sRGB
            )
        let outputColorSpace = try #require(
            candidateColorSpace
        )
        let renderer = CARenderer(
            mtlTexture: texture,
            options: [
                kCARendererColorSpace: outputColorSpace
            ]
        )
        renderer.bounds = CGRect(
            x: 0,
            y: 0,
            width: 16,
            height: 16
        )

        let root = CALayer()
        root.bounds = renderer.bounds
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        let child = CALayer()
        child.bounds = CGRect(
            x: 0,
            y: 0,
            width: 8,
            height: 8
        )
        child.position = CGPoint(x: 4, y: 4)
        child.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        root.addSublayer(child)
        renderer.layer = root

        renderer.beginFrame(
            atTime: CACurrentMediaTime(),
            timeStamp: nil
        )
        renderer.render()
        renderer.endFrame()

        let backend = try #require(
            renderer.backend as? CAMetalRenderer
        )
        let commandBuffer = try #require(
            backend.lastCommandBuffer
        )
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        var redPixelCount = 0
        var bluePixelCount = 0
        for y in 0..<texture.height {
            for x in 0..<texture.width {
                switch readPixel(texture, x: x, y: y) {
                case [0, 0, 255, 255]:
                    redPixelCount += 1
                case [255, 0, 0, 255]:
                    bluePixelCount += 1
                default:
                    break
                }
            }
        }
        #expect(redPixelCount == 64)
        #expect(bluePixelCount == 192)
    }

    @Test("Metal renderer explicitly rejects unsupported layer semantics")
    func unsupportedMetalSemantics() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CARendererTestError.metalUnavailable
        }
        let renderer = CARenderer(
            mtlTexture: try makeTexture(device: device)
        )
        renderer.bounds = CGRect(
            x: 0,
            y: 0,
            width: 16,
            height: 16
        )
        let root = CALayer()
        root.bounds = renderer.bounds
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        root.masksToBounds = true
        renderer.layer = root

        renderer.beginFrame(
            atTime: CACurrentMediaTime(),
            timeStamp: nil
        )
        renderer.render()
        renderer.endFrame()

        #expect(
            renderer.lastRenderError
                == .unsupportedCommittedSnapshotFeature(.clipping)
        )
    }

    private func makeTexture(
        device: any MTLDevice,
        usage: MTLTextureUsage = [
            .renderTarget,
            .shaderRead
        ]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor
            .texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: 16,
                height: 16,
                mipmapped: false
            )
        descriptor.usage = usage
        descriptor.storageMode = .shared
        guard let texture =
                device.makeTexture(descriptor: descriptor) else {
            throw CARendererTestError.textureUnavailable
        }
        return texture
    }

    private func readPixel(
        _ texture: any MTLTexture,
        x: Int,
        y: Int
    ) -> [UInt8] {
        var pixel = [UInt8](repeating: 0, count: 4)
        pixel.withUnsafeMutableBytes { bytes in
            guard let destination = bytes.baseAddress else { return }
            texture.getBytes(
                destination,
                bytesPerRow: 4,
                from: MTLRegionMake2D(x, y, 1, 1),
                mipmapLevel: 0
            )
        }
        return pixel
    }
    #endif

    private func isNull(_ rect: CGRect) -> Bool {
        rect.origin.x == CGFloat.infinity && rect.origin.y == CGFloat.infinity
    }
}

private enum CARendererTestError: Error {
    case metalUnavailable
    case commandQueueUnavailable
    case textureUnavailable
}
