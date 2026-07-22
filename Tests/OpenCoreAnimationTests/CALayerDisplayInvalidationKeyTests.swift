import Testing
@testable import OpenCoreAnimation

@Suite("CALayer display invalidation keys")
struct CALayerDisplayInvalidationKeyTests {
    @Test("Base and non-text specialized layers do not request display")
    func nonTextLayerKeys() {
        let cases: [(CALayer.Type, [String])] = [
            (CALayer.self, [
                "bounds", "position", "opacity", "backgroundColor", "contents",
                "style", "contentsScale", "futureKey",
            ]),
            (CAShapeLayer.self, ["path", "fillColor", "lineWidth"]),
            (CAGradientLayer.self, ["colors", "locations", "startPoint"]),
            (CAReplicatorLayer.self, ["instanceCount", "instanceTransform"]),
            (CAEmitterLayer.self, ["birthRate", "emitterPosition"]),
            (CATiledLayer.self, ["levelsOfDetail", "tileSize"]),
            (CAScrollLayer.self, ["scrollMode", "bounds"]),
        ]

        for (layerType, keys) in cases {
            for key in keys {
                #expect(!layerType.needsDisplay(forKey: key), "\(layerType).\(key)")
            }
        }
    }

    @Test("Text layer redraw keys match QuartzCore")
    func textLayerKeys() {
        let redrawKeys = [
            "string", "font", "fontSize", "foregroundColor", "wrapped",
            "truncationMode", "alignmentMode", "allowsFontSubpixelQuantization",
            "style", "contentsScale",
        ]
        let inheritedNonRedrawKeys = [
            "bounds", "position", "opacity", "backgroundColor", "contents",
            "contentsRect", "contentsGravity", "futureKey",
        ]

        for key in redrawKeys {
            #expect(CATextLayer.needsDisplay(forKey: key), "\(key)")
        }
        for key in inheritedNonRedrawKeys {
            #expect(!CATextLayer.needsDisplay(forKey: key), "\(key)")
        }
    }

    @Test("Text property mutations request display exactly once per value change")
    func textPropertyMutations() {
        assertMutationRequestsDisplay { $0.string = "text" }
        assertMutationRequestsDisplay { $0.font = "Other" }
        assertMutationRequestsDisplay { $0.fontSize = 12 }
        assertMutationRequestsDisplay {
            $0.foregroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        }
        assertMutationRequestsDisplay { $0.isWrapped = true }
        assertMutationRequestsDisplay { $0.truncationMode = .end }
        assertMutationRequestsDisplay { $0.alignmentMode = .center }
        assertMutationRequestsDisplay { $0.allowsFontSubpixelQuantization = true }
        assertMutationRequestsDisplay { $0.contentsScale = 2 }
        assertMutationRequestsDisplay { $0.style = ["value": 1] }
    }

    @Test("Text layer copies preserve clean display state")
    func textLayerCopy() {
        let source = CATextLayer()
        source.string = "text"
        source.font = "Other"
        source.fontSize = 12
        source.foregroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        source.isWrapped = true
        source.truncationMode = .end
        source.alignmentMode = .center
        source.allowsFontSubpixelQuantization = true
        source.displayIfNeeded()

        let copy = CATextLayer(layer: source)

        #expect(!source.needsDisplay())
        #expect(!copy.needsDisplay())
        #expect(copy.string as? String == "text")
        #expect(copy.font as? String == "Other")
        #expect(copy.fontSize == 12)
        #expect(copy.isWrapped)
        #expect(copy.truncationMode == .end)
        #expect(copy.alignmentMode == .center)
        #expect(copy.allowsFontSubpixelQuantization)
    }

    private func assertMutationRequestsDisplay(
        _ mutate: (CATextLayer) -> Void
    ) {
        let layer = CATextLayer()
        #expect(!layer.needsDisplay())

        mutate(layer)
        #expect(layer.needsDisplay())

        layer.displayIfNeeded()
        #expect(!layer.needsDisplay())
        mutate(layer)
        #expect(!layer.needsDisplay())
    }
}
