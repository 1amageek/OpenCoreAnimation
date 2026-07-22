import Testing
@testable import OpenCoreAnimation

@Suite("CALayer dynamic range")
struct CALayerDynamicRangeTests {
    @Test("Typed values match QuartzCore raw values")
    func rawValues() {
        #expect(CALayer.ToneMapMode.automatic.rawValue == "automatic")
        #expect(CALayer.ToneMapMode.never.rawValue == "never")
        #expect(CALayer.ToneMapMode.ifSupported.rawValue == "ifSupported")
        #expect(CALayer.DynamicRange.automatic.rawValue == "automatic")
        #expect(CALayer.DynamicRange.standard.rawValue == "standard")
        #expect(CALayer.DynamicRange.constrainedHigh.rawValue == "constrainedHigh")
        #expect(CALayer.DynamicRange.high.rawValue == "high")
    }

    @Test("Defaults match QuartzCore")
    func defaults() {
        let layer = CALayer()

        #expect(layer.toneMapMode == .automatic)
        #expect(layer.preferredDynamicRange == .standard)
        #expect(layer.contentsHeadroom == 0)
        #expect(CALayer.defaultValue(forKey: "toneMapMode") as? CALayer.ToneMapMode == .automatic)
        #expect(CALayer.defaultValue(forKey: "preferredDynamicRange") as? CALayer.DynamicRange == .standard)
        #expect(CALayer.defaultValue(forKey: "contentsHeadroom") as? CGFloat == 0)
    }

    @Test("Layer and presentation copies retain rendering policy")
    func copies() throws {
        let layer = CALayer()
        layer.toneMapMode = .never
        layer.preferredDynamicRange = .high
        layer.contentsHeadroom = 4

        let copy = CALayer(layer: layer)
        #expect(copy.toneMapMode == .never)
        #expect(copy.preferredDynamicRange == .high)
        #expect(copy.contentsHeadroom == 4)

        let presentation = try #require(layer.presentation())
        #expect(presentation.toneMapMode == .never)
        #expect(presentation.preferredDynamicRange == .high)
        #expect(presentation.contentsHeadroom == 4)
    }

    @Test("Mutations invalidate render state without forcing delegate redraw")
    func dirtyState() {
        let layer = CALayer()
        layer.recursivelyClearDirtyAfterCommit()

        layer.preferredDynamicRange = .high
        #expect(layer._dirtyMask.contains(.contents))
        #expect(!layer.needsDisplay())

        layer.recursivelyClearDirtyAfterCommit()
        layer.toneMapMode = .never
        #expect(layer._dirtyMask.contains(.contents))

        layer.recursivelyClearDirtyAfterCommit()
        layer.contentsHeadroom = 3
        #expect(layer._dirtyMask.contains(.contents))
    }
}
