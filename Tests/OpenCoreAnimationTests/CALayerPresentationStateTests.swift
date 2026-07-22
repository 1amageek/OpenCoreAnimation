import Testing
@testable import OpenCoreAnimation

@Suite("CALayer presentation state synchronization")
struct CALayerPresentationStateTests {
    @Test("model mutation invalidates a same-frame presentation cache")
    func sameFrameMutationRefreshesPresentation() {
        let layer = CALayer()
        layer.opacity = 0.25

        let first = layer.presentation()
        #expect(first?.opacity == 0.25)

        layer.opacity = 0.75
        let second = layer.presentation()

        #expect(first === second)
        #expect(second?.opacity == 0.75)
    }

    @Test("base rendering configuration stays synchronized")
    func baseRenderingConfigurationRefreshes() {
        let layer = CALayer()
        _ = layer.presentation()

        let mask = CALayer()
        layer.mask = mask
        layer.isDoubleSided = false
        layer.isOpaque = true
        layer.shouldRasterize = true
        layer.rasterizationScale = 2
        layer.cornerCurve = .continuous
        layer.maskedCorners = [.layerMinXMinYCorner]
        layer.contentsFormat = .RGBA16Float

        guard let presentation = layer.presentation() else {
            Issue.record("Expected presentation layer")
            return
        }
        #expect(presentation.mask === mask)
        #expect(presentation.isDoubleSided == false)
        #expect(presentation.isOpaque)
        #expect(presentation.shouldRasterize)
        #expect(presentation.rasterizationScale == 2)
        #expect(presentation.cornerCurve == .continuous)
        #expect(presentation.maskedCorners == [.layerMinXMinYCorner])
        #expect(presentation.contentsFormat == .RGBA16Float)
    }

    @Test("shape style stays synchronized after presentation allocation")
    func shapeStyleRefreshes() {
        let layer = CAShapeLayer()
        _ = layer.presentation()

        layer.lineWidth = 7
        layer.lineCap = .round
        layer.lineJoin = .bevel
        layer.lineDashPattern = [3, 2]
        layer.fillRule = .evenOdd

        guard let presentation = layer.presentation() else {
            Issue.record("Expected shape presentation layer")
            return
        }
        #expect(presentation.lineWidth == 7)
        #expect(presentation.lineCap == .round)
        #expect(presentation.lineJoin == .bevel)
        #expect(presentation.lineDashPattern == [3, 2])
        #expect(presentation.fillRule == .evenOdd)
    }

    @Test("specialized layer configuration stays synchronized")
    func specializedLayerStateRefreshes() {
        let emitter = CAEmitterLayer()
        _ = emitter.presentation()
        emitter.emitterPosition = CGPoint(x: 12, y: 34)
        emitter.birthRate = 4
        emitter.seed = 99
        emitter.preservesDepth = true

        let replicator = CAReplicatorLayer()
        _ = replicator.presentation()
        replicator.instanceCount = 8
        replicator.instanceDelay = 0.2
        replicator.instanceAlphaOffset = -0.1
        replicator.preservesDepth = true

        let emitterPresentation = emitter.presentation()
        let replicatorPresentation = replicator.presentation()
        #expect(emitterPresentation?.emitterPosition == CGPoint(x: 12, y: 34))
        #expect(emitterPresentation?.birthRate == 4)
        #expect(emitterPresentation?.seed == 99)
        #expect(emitterPresentation?.preservesDepth == true)
        #expect(replicatorPresentation?.instanceCount == 8)
        #expect(replicatorPresentation?.instanceDelay == 0.2)
        #expect(replicatorPresentation?.instanceAlphaOffset == -0.1)
        #expect(replicatorPresentation?.preservesDepth == true)
    }

    @Test("replacing sublayers transfers dirty subtree accounting")
    func sublayerReplacementTransfersDirtyCounts() {
        let parent = CALayer()
        parent._testClearDirty()

        let first = CALayer()
        let second = CALayer()
        parent.sublayers = [first]
        let firstCount = parent._subtreeDirtyCount

        parent.sublayers = [second]

        #expect(first.superlayer == nil)
        #expect(second.superlayer === parent)
        #expect(parent._subtreeDirtyCount == firstCount)
    }
}
