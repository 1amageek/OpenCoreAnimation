import Testing
@testable import OpenCoreAnimation

@MainActor
@Suite("Immutable render snapshots", .serialized)
struct CARenderSnapshotTests {
    @Test("Outermost transaction publishes an immutable root snapshot")
    func commitPublishesSnapshot() throws {
        CATransaction.flush()
        let root = CALayer()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        CATransaction.commit()

        guard case .snapshot(let snapshot) = root.pendingCommittedRenderState else {
            Issue.record("Expected the outermost transaction to publish a snapshot")
            return
        }
        #expect(snapshot.rootBounds == root.bounds)
        #expect(snapshot.nodes[snapshot.rootIndex].presentationValues.backgroundColor
            == SIMD4<Float>(0, 1, 0, 1))
    }

    @Test("Common solid state and z-ordered hierarchy are value-owned")
    func commonSolidStateIsCaptured() throws {
        let root = CALayer()
        root.borderWidth = 2
        root.borderColor = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        root.cornerRadius = 4
        root.maskedCorners = [.layerMinXMinYCorner]
        root.masksToBounds = true
        root.isGeometryFlipped = true
        root.toneMapMode = .never
        root.preferredDynamicRange = .high
        root.contentsHeadroom = 2

        let front = CALayer()
        front.zPosition = 2
        front.isDoubleSided = false
        let back = CALayer()
        back.zPosition = -1
        root.addSublayer(front)
        root.addSublayer(back)

        let snapshot = try CARenderSnapshot.capture(root, frameToken: 45)
        let rootNode = snapshot.nodes[snapshot.rootIndex]
        let values = rootNode.presentationValues

        #expect(snapshot.liveTreeRequirement == nil)
        #expect(values.borderWidth == 2)
        #expect(values.borderColor == SIMD4<Float>(0, 0, 1, 1))
        #expect(values.cornerRadii == SIMD4<Float>(4, 0, 0, 0))
        #expect(values.masksToBounds)
        #expect(values.isGeometryFlipped)
        #expect(values.toneMapMode == .never)
        #expect(values.preferredDynamicRange == .high)
        #expect(values.contentsHeadroom == 2)
        let frontIndex = try #require(
            rootNode.childIndices.first {
                snapshot.nodes[$0].identity == ObjectIdentifier(front)
            }
        )
        #expect(
            !snapshot.nodes[frontIndex]
                .presentationValues.isDoubleSided
        )
        #expect(
            rootNode.childIndices.map { snapshot.nodes[$0].identity }
                == [ObjectIdentifier(back), ObjectIdentifier(front)]
        )
    }

    @Test("Static resources remain an explicit live-tree state")
    func unsupportedStaticResourceIsExplicit() {
        CATransaction.flush()
        let root = CALayer()
        let mask = CALayer()

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.mask = mask
        CATransaction.commit()

        guard case .requiresLiveResourceCapture(_, let requirement) =
                root.pendingCommittedRenderState else {
            Issue.record("Expected an explicit live-resource capture state")
            return
        }
        #expect(requirement == .mask)
    }

    @Test("Only overlapping group opacity requires live resource capture")
    func opacityRequirementsMatchCompositionSemantics() throws {
        let leaf = CALayer()
        leaf.opacity = 0.5
        let leafSnapshot = try CARenderSnapshot.capture(
            leaf,
            frameToken: 46
        )
        #expect(leafSnapshot.liveTreeRequirement == nil)

        let distributedRoot = CALayer()
        distributedRoot.opacity = 0.5
        distributedRoot.allowsGroupOpacity = false
        distributedRoot.addSublayer(CALayer())
        let distributedSnapshot = try CARenderSnapshot.capture(
            distributedRoot,
            frameToken: 47
        )
        #expect(distributedSnapshot.liveTreeRequirement == nil)

        let groupedRoot = CALayer()
        groupedRoot.opacity = 0.5
        groupedRoot.addSublayer(CALayer())
        let groupedSnapshot = try CARenderSnapshot.capture(
            groupedRoot,
            frameToken: 48
        )
        #expect(groupedSnapshot.liveTreeRequirement == .opacityGroup)
    }

    @Test("CGImage contents become value-owned commit resources")
    func imageContentsAreCapturedByValue() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 4)
        layer.contentsRect = CGRect(x: 0.25, y: 0, width: 0.5, height: 1)
        layer.contentsCenter = CGRect(
            x: 0.2,
            y: 0.3,
            width: 0.4,
            height: 0.5
        )
        layer.contentsScale = 2
        layer.contentsGravity = .resizeAspect
        layer.magnificationFilter = .nearest
        layer.minificationFilter = .trilinear
        layer.minificationFilterBias = 20
        layer.isOpaque = true

        weak var sourceImage: CGImage?
        let snapshot: CARenderSnapshot
        do {
            let image = try makeImage(
                width: 2,
                height: 1,
                pixels: [255, 0, 0, 255, 0, 255, 0, 255]
            )
            sourceImage = image
            layer.contents = image
            snapshot = try CARenderSnapshot.capture(
                layer,
                frameToken: 49
            )
            layer.contents = nil
        }
        layer.removeAllAnimations()
        CATransaction.flush()

        #expect(sourceImage == nil)
        #expect(snapshot.liveTreeRequirement == nil)
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )
        #expect(contents.storage.format == .rgba8Unorm)
        #expect(contents.storage.width == 2)
        #expect(contents.storage.height == 1)
        #expect(contents.storage.data == Data([
            255, 0, 0, 255,
            0, 255, 0, 255,
        ]))
        #expect(contents.contentsRect == layer.contentsRect)
        #expect(contents.contentsCenter == layer.contentsCenter)
        #expect(contents.contentsScale == 2)
        #expect(contents.gravity == .resizeAspect)
        #expect(contents.sampling == .nearestTrilinear)
        #expect(contents.minificationFilterBias == 15.99)
        #expect(contents.isOpaque)
        requireSendable(snapshot)
    }

    @Test("Captured image state does not follow later layer mutations")
    func capturedImageStateIsImmutable() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        layer.contents = try makeImage(
            width: 2,
            height: 1,
            pixels: [255, 0, 0, 255, 0, 255, 0, 255]
        )
        layer.magnificationFilter = .nearest
        layer.minificationFilter = .nearest

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 50
        )
        layer.contents = try makeImage(
            width: 2,
            height: 1,
            pixels: [0, 0, 255, 255, 255, 255, 255, 255]
        )
        layer.contentsRect = CGRect(x: 0.5, y: 0, width: 0.5, height: 1)
        layer.magnificationFilter = .linear

        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )
        #expect(contents.storage.data == Data([
            255, 0, 0, 255,
            0, 255, 0, 255,
        ]))
        #expect(contents.contentsRect == CGRect(
            x: 0,
            y: 0,
            width: 1,
            height: 1
        ))
        #expect(contents.sampling == .nearestNearest)
    }

    @Test("Unknown image sampling filters fail capture explicitly")
    func invalidImageSamplingFailsCapture() throws {
        let layer = CALayer()
        layer.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 255, 255]
        )
        layer.magnificationFilter = CALayerContentsFilter(
            rawValue: "future"
        )

        #expect(throws: CARendererError.invalidLayerContents(
            .invalidSamplingFilters(
                magnification: CALayerContentsFilter(rawValue: "future"),
                minification: .linear
            )
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 51)
        }
    }

    @Test(
        "Image sampling captures every supported filter combination",
        arguments: [
            (
                CALayerContentsFilter.nearest,
                CALayerContentsFilter.nearest,
                CAContentsSampling.nearestNearest
            ),
            (
                CALayerContentsFilter.nearest,
                CALayerContentsFilter.linear,
                CAContentsSampling.nearestLinear
            ),
            (
                CALayerContentsFilter.nearest,
                CALayerContentsFilter.trilinear,
                CAContentsSampling.nearestTrilinear
            ),
            (
                CALayerContentsFilter.linear,
                CALayerContentsFilter.nearest,
                CAContentsSampling.linearNearest
            ),
            (
                CALayerContentsFilter.linear,
                CALayerContentsFilter.linear,
                CAContentsSampling.linearLinear
            ),
            (
                CALayerContentsFilter.trilinear,
                CALayerContentsFilter.trilinear,
                CAContentsSampling.linearTrilinear
            ),
        ]
    )
    func imageSamplingCapturesSupportedFilters(
        magnification: CALayerContentsFilter,
        minification: CALayerContentsFilter,
        expected: CAContentsSampling
    ) throws {
        let layer = CALayer()
        layer.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 255, 255]
        )
        layer.magnificationFilter = magnification
        layer.minificationFilter = minification

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 53
        )
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )

        #expect(contents.sampling == expected)
        #expect(contents.sampling.usesMipmaps == (minification == .trilinear))
    }

    @Test("Non-finite image mip bias fails capture explicitly")
    func nonFiniteImageMipBiasFailsCapture() throws {
        let layer = CALayer()
        layer.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 255, 255]
        )
        layer.minificationFilterBias = .infinity

        #expect(throws: CARendererError.invalidLayerContents(
            .invalidMinificationFilterBias(.infinity)
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 54)
        }
    }

    @Test("Non-image contents remain an explicit live-tree requirement")
    func nonImageContentsRemainExplicit() throws {
        let layer = CALayer()
        layer.contents = SnapshotContentsToken()

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 55
        )
        #expect(snapshot.liveTreeRequirement == .contents)
        #expect(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents == nil
        )
    }

    @Test("Animated commits request explicit live evaluation until evaluators are immutable")
    func animatedCommitPublishesExplicitEvaluationState() {
        CATransaction.flush()
        let root = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 1

        CATransaction.begin()
        root.add(animation, forKey: "opacity")
        CATransaction.commit()

        guard case .requiresLiveAnimationEvaluation = root.pendingCommittedRenderState else {
            Issue.record("Expected an explicit live-animation evaluation state")
            return
        }
        root.removeAllAnimations()
    }

    @Test("Layout-pending commits cannot publish stale geometry")
    func layoutPendingCommitPublishesExplicitPreparationState() {
        CATransaction.flush()
        let root = CALayer()
        root.layoutManager = SnapshotLayoutManager()

        CATransaction.begin()
        root.addSublayer(CALayer())
        CATransaction.commit()

        guard case .requiresLiveTreePreparation = root.pendingCommittedRenderState else {
            Issue.record("Expected an explicit live-tree preparation state")
            return
        }
    }

    @Test("Capture failure is retained as a typed committed state")
    func commitRetainsCaptureFailure() {
        CATransaction.flush()
        let root = CALayer()
        let components = [CGFloat.nan, 0, 0, 1]
        let invalidColor: CGColor? = components.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return nil }
            return CGColor(colorSpace: .deviceRGB, components: baseAddress)
        }

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.backgroundColor = invalidColor
        CATransaction.commit()

        guard case .captureFailure(_, let error) = root.pendingCommittedRenderState else {
            Issue.record("Expected a committed capture failure")
            return
        }
        #expect(error == .invalidLayerBackgroundColor)
    }

    @Test("Captured values and hierarchy do not follow later model mutations")
    func captureIsIndependentFromModelTree() throws {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 4, height: 4)
        child.position = CGPoint(x: 2, y: 2)
        child.isDoubleSided = false
        root.addSublayer(child)

        let snapshot = try CARenderSnapshot.capture(root, frameToken: 41)
        requireSendable(snapshot)

        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        root.bounds = CGRect(x: 0, y: 0, width: 32, height: 32)
        child.isDoubleSided = true
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
        #expect(!snapshot.nodes[1].presentationValues.isDoubleSided)
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

    @Test("Revision snapshots preserve later layer and detached-mask mutations")
    func revisionSnapshotClearsOnlyCapturedMutations() throws {
        let root = CALayer()
        let child = CALayer()
        let mask = CALayer()
        root.addSublayer(child)
        root.mask = mask

        let submitted = try CARenderRevisionSnapshot.capture(root)
        root.recursivelyClearDirtyAfterCommit(matching: submitted)
        #expect(root._dirtyMask.isEmpty)
        #expect(child._dirtyMask.isEmpty)
        #expect(mask._dirtyMask.isEmpty)

        let nextSubmitted = try CARenderRevisionSnapshot.capture(root)
        child.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        mask.opacity = 0.5

        root.recursivelyClearDirtyAfterCommit(matching: nextSubmitted)
        #expect(root._dirtyMask.isEmpty)
        #expect(child._dirtyMask.isEmpty == false)
        #expect(mask._dirtyMask.isEmpty == false)
    }

    @Test("Revision capture rejects a cyclic detached-mask graph")
    func revisionSnapshotRejectsMaskCycles() {
        let root = CALayer()
        let mask = CALayer()
        root.mask = mask
        mask.mask = root

        #expect(throws: CARendererError.cyclicLayerHierarchy) {
            try CARenderRevisionSnapshot.capture(root)
        }

        mask.mask = nil
        root.mask = nil
        CATransaction.flush()
    }

    private func requireSendable<T: Sendable>(_ value: T) {
        _ = value
    }

    private func makeImage(
        width: Int,
        height: Int,
        pixels: [UInt8]
    ) throws -> CGImage {
        let bytesPerRow = width * 4
        #expect(pixels.count == bytesPerRow * height)
        return try #require(CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: .deviceRGB,
            bitmapInfo: CGBitmapInfo(
                rawValue: CGImageAlphaInfo.last.rawValue
            ),
            provider: CGDataProvider(data: Data(pixels)),
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ))
    }
}

private final class SnapshotLayoutManager: CALayoutManager {
    func layoutSublayers(of layer: CALayer) {}
}

private final class SnapshotContentsToken {}

#if canImport(Metal)
import Metal

extension CARenderSnapshotTests {
    @Test("Metal reports committed image contents instead of dropping them")
    func metalRejectsUnsupportedImageSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let renderer = try CAMetalRenderer(destination: texture)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.position = CGPoint(x: 1, y: 0.5)
        root.contents = try makeImage(
            width: 2,
            height: 1,
            pixels: [255, 0, 0, 255, 0, 255, 0, 255]
        )

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.imageContents))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal submits committed pixels without clearing a later mutation")
    func metalSubmissionPreservesPostCommitMutation() throws {
        CATransaction.flush()
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

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        CATransaction.commit()

        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        renderer.render(layer: root)

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
        #expect(root._dirtyMask.isEmpty == false)
        if case .some = root.pendingCommittedRenderState {
            Issue.record("Expected the submitted committed snapshot to be acknowledged")
        }

        CATransaction.flush()
    }

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
