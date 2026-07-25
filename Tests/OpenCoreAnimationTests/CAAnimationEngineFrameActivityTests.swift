import Testing
@testable import OpenCoreAnimation

@MainActor
@Suite("CAAnimationEngine frame activity", .serialized)
struct CAAnimationEngineFrameActivityTests {
    @Test("Display-link frames skip a clean static tree")
    func cleanStaticTreeSkipsSubmit() {
        let fixture = makeFixture()

        fixture.fireDisplayLink()
        #expect(fixture.renderer.renderCount == 1)

        fixture.fireDisplayLink()
        #expect(fixture.renderer.renderCount == 1)

        fixture.root.setNeedsDisplay()
        fixture.fireDisplayLink()
        #expect(fixture.renderer.renderCount == 2)
    }

    @Test("Future animations skip clean frames until their active interval")
    func futureAnimationSkipsCleanFrames() {
        let fixture = makeFixture()
        fixture.fireDisplayLink()

        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(1)
        animation.toValue = Float(0)
        animation.beginTime = fixture.root.convertTime(
            CACurrentMediaTime() + 60,
            from: nil
        )
        animation.duration = 1
        fixture.root.add(animation, forKey: "future-opacity")

        fixture.fireDisplayLink()
        fixture.fireDisplayLink()

        #expect(fixture.renderer.renderCount == 2)

        setStoredAnimationBeginTime(
            fixture.root.convertTime(CACurrentMediaTime() - 0.5, from: nil),
            on: fixture.root,
            forKey: "future-opacity"
        )
        fixture.fireDisplayLink()

        #expect(fixture.renderer.renderCount == 3)
    }

    @Test("Progressing animations keep display-link rendering active")
    func progressingAnimationRequiresFrames() {
        let fixture = makeFixture()
        fixture.fireDisplayLink()

        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(1)
        animation.toValue = Float(0)
        animation.duration = 60
        fixture.root.add(animation, forKey: "active-opacity")

        fixture.fireDisplayLink()
        fixture.fireDisplayLink()

        #expect(fixture.renderer.renderCount == 3)
    }

    @Test("Paused animations skip clean display-link frames")
    func pausedAnimationSkipsCleanFrames() {
        let fixture = makeFixture()
        fixture.fireDisplayLink()

        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(1)
        animation.toValue = Float(0)
        animation.duration = 60
        animation.speed = 0
        animation.timeOffset = 0.5
        fixture.root.add(animation, forKey: "paused-opacity")

        fixture.fireDisplayLink()
        fixture.fireDisplayLink()

        #expect(fixture.renderer.renderCount == 2)
    }

    @Test("Renderer-owned work can request a clean-tree frame")
    func rendererOwnedWorkRequiresFrame() {
        let fixture = makeFixture()
        fixture.fireDisplayLink()
        fixture.fireDisplayLink()
        #expect(fixture.renderer.renderCount == 1)

        fixture.renderer.hasPendingFrameWork = true
        fixture.fireDisplayLink()

        #expect(fixture.renderer.renderCount == 2)
    }

    @Test("Manual renderFrame remains an unconditional submission")
    func manualRenderRemainsUnconditional() {
        let fixture = makeFixture()

        fixture.engine.renderFrame()
        fixture.engine.renderFrame()

        #expect(fixture.renderer.renderCount == 2)
    }

    @Test("Mask-tree animations participate in scheduling and completion")
    func maskAnimationsParticipateInEngineTraversal() {
        let fixture = makeFixture()
        let mask = CALayer()
        fixture.root.mask = mask
        fixture.fireDisplayLink()

        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(1)
        animation.toValue = Float(0)
        animation.duration = 0.01
        animation.preferredFrameRateRange = CAFrameRateRange(
            minimum: 24,
            maximum: 48,
            preferred: 48
        )
        mask.add(animation, forKey: "mask-opacity")

        let range = fixture.engine.resolvedFrameRateRange(at: CACurrentMediaTime())
        #expect(range.minimum == 24)
        #expect(range.maximum == 48)
        #expect(range.preferred == 48)

        setStoredAnimationBeginTime(
            mask.convertTime(CACurrentMediaTime() - 1, from: nil),
            on: mask,
            forKey: "mask-opacity"
        )
        fixture.fireDisplayLink()

        #expect(mask.animation(forKey: "mask-opacity") == nil)
    }

    private func makeFixture() -> EngineFrameFixture {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 32, height: 32)
        let renderer = FrameActivityRenderer()
        let engine = CAAnimationEngine()
        engine.rootLayer = root
        engine.rendererDelegate = renderer
        let displayLink = CADisplayLink(
            target: engine,
            selector: Selector("displayLinkDidFire")
        )
        return EngineFrameFixture(
            root: root,
            renderer: renderer,
            engine: engine,
            displayLink: displayLink
        )
    }
}

@MainActor
private struct EngineFrameFixture {
    let root: CALayer
    let renderer: FrameActivityRenderer
    let engine: CAAnimationEngine
    let displayLink: CADisplayLink

    func fireDisplayLink() {
        engine.displayLinkDidFire(displayLink)
    }
}

@MainActor
private final class FrameActivityRenderer: CARendererDelegate {
    var size = CGSize(width: 32, height: 32)
    var hasPendingFrameWork = false
    private(set) var renderCount = 0

    var requiresFrameWhenLayerTreeIsClean: Bool {
        hasPendingFrameWork
    }

    func initialize() async throws {}

    func invalidate() {}

    func resize(width: Int, height: Int) {
        size = CGSize(width: width, height: height)
    }

    func render(layer rootLayer: CALayer) {
        renderCount += 1
        hasPendingFrameWork = false
        rootLayer.recursivelyClearDirtyAfterCommit()
        rootLayer.completeTransactionsAfterRenderRecursively()
    }
}
