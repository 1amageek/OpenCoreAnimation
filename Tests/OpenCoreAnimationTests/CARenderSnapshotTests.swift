import Testing
@testable import OpenCoreAnimation

@MainActor
@Suite("Immutable render snapshots")
struct CARenderSnapshotTests {
    @Test("Captured values and hierarchy do not follow later model mutations")
    func captureIsIndependentFromModelTree() throws {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 4, height: 4)
        child.position = CGPoint(x: 2, y: 2)
        root.addSublayer(child)

        let snapshot = try CARenderSnapshot.capture(root, frameToken: 41)
        requireSendable(snapshot)

        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        root.bounds = CGRect(x: 0, y: 0, width: 32, height: 32)
        child.removeFromSuperlayer()

        let rootNode = snapshot.nodes[snapshot.rootIndex]
        #expect(snapshot.frameToken == 41)
        #expect(snapshot.nodes.count == 2)
        #expect(rootNode.childIndices == [1])
        #expect(rootNode.presentationValues.bounds == CGRect(
            x: 0,
            y: 0,
            width: 16,
            height: 16
        ))
        #expect(rootNode.presentationValues.backgroundColor == SIMD4<Float>(0, 1, 0, 1))
    }

    @Test("Gray colors are converted to device RGB instead of becoming transparent")
    func grayBackgroundColorIsConverted() throws {
        let root = CALayer()
        root.backgroundColor = CGColor(gray: 0.25, alpha: 0.75)

        let snapshot = try CARenderSnapshot.capture(root, frameToken: 42)
        let color = snapshot.nodes[snapshot.rootIndex].presentationValues.backgroundColor

        #expect(color == SIMD4<Float>(0.25, 0.25, 0.25, 0.75))
    }

    @Test("Non-finite colors fail capture explicitly")
    func nonFiniteBackgroundColorFailsCapture() {
        let root = CALayer()
        let components = [CGFloat.nan, 0, 0, 1]
        root.backgroundColor = components.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return nil }
            return CGColor(colorSpace: .deviceRGB, components: baseAddress)
        }

        #expect(throws: CARendererError.invalidLayerBackgroundColor) {
            try CARenderSnapshot.capture(root, frameToken: 43)
        }
    }

    @Test("Non-finite geometry fails capture explicitly")
    func nonFiniteGeometryFailsCapture() {
        let root = CALayer()
        root.position.x = .infinity

        #expect(throws: CARendererError.nonFiniteLayerGeometry) {
            try CARenderSnapshot.capture(root, frameToken: 44)
        }
    }

    private func requireSendable<T: Sendable>(_ value: T) {
        _ = value
    }
}

#if canImport(Metal)
import Metal

extension CARenderSnapshotTests {
    @Test("Metal encoding reads the captured frame instead of the mutated model")
    func metalEncodingUsesCapturedValues() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 16,
            height: 16,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let renderer = try CAMetalRenderer(destination: texture)

        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        let snapshot = try CARenderSnapshot.capture(root, frameToken: 45)

        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)

        #expect(renderer.render(snapshot: snapshot))
        let commandBuffer = try #require(renderer.lastCommandBuffer)
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
        #expect(pixel == [0, 255, 0, 255])
    }
}
#endif
