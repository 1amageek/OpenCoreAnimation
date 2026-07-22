import Foundation
import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

@Suite("CATextLayer render configuration")
struct CATextRenderConfigurationTests {
    @Test("Valid text input preserves layout and converts gray foreground colors")
    func validConfiguration() throws {
        let layer = CATextLayer()
        layer.string = "Hello"
        layer.font = "Test Sans"
        layer.fontSize = 18
        layer.contentsScale = 2
        layer.foregroundColor = CGColor(gray: 0.25, alpha: 0.5)
        layer.bounds = CGRect(x: 2, y: 3, width: 80, height: 24)
        layer.alignmentMode = .center
        layer.truncationMode = .middle
        layer.isWrapped = true

        let configuration = try CATextRenderConfiguration(layer: layer)

        #expect(configuration.text == "Hello")
        #expect(configuration.fontFamily == "Test Sans")
        #expect(configuration.cssFontFamily == "\"Test Sans\"")
        #expect(configuration.contentsScale == 2)
        #expect(configuration.foregroundRGBA == SIMD4<Float>(0.25, 0.25, 0.25, 0.5))
        #expect(configuration.bounds == layer.bounds)
        #expect(configuration.alignmentMode == .center)
        #expect(configuration.truncationMode == .middle)
        #expect(configuration.isWrapped)
    }

    @Test("CSS generic families remain generic and concrete names are escaped")
    func cssFontFamily() throws {
        let layer = CATextLayer()
        layer.string = "Text"
        layer.font = "Monospace"
        #expect(try CATextRenderConfiguration(layer: layer).cssFontFamily == "monospace")

        layer.font = "Quoted \\\" Font"
        #expect(
            try CATextRenderConfiguration(layer: layer).cssFontFamily
                == "\"Quoted \\\\\\\" Font\""
        )
    }

    @Test("Unsupported text and font values fail instead of becoming descriptions")
    func unsupportedValues() {
        let layer = CATextLayer()
        layer.string = 42
        #expect(throws: CATextRenderFailure.unsupportedStringValue) {
            try CATextRenderConfiguration(layer: layer)
        }

        layer.string = "Text"
        layer.font = 42
        #expect(throws: CATextRenderFailure.unsupportedFontValue) {
            try CATextRenderConfiguration(layer: layer)
        }
    }

    @Test("Invalid geometry and unknown layout modes fail with typed reasons")
    func invalidGeometryAndModes() {
        let layer = CATextLayer()
        layer.string = "Text"
        layer.fontSize = .nan
        #expect(throws: CATextRenderFailure.invalidFontSize) {
            try CATextRenderConfiguration(layer: layer)
        }

        layer.fontSize = CGFloat.greatestFiniteMagnitude
        #expect(throws: CATextRenderFailure.invalidFontSize) {
            try CATextRenderConfiguration(layer: layer)
        }

        layer.fontSize = 12
        layer.contentsScale = 0
        #expect(throws: CATextRenderFailure.invalidContentsScale) {
            try CATextRenderConfiguration(layer: layer)
        }

        layer.contentsScale = 1
        layer.bounds = CGRect(x: 0, y: 0, width: CGFloat.infinity, height: 20)
        #expect(throws: CATextRenderFailure.invalidBounds) {
            try CATextRenderConfiguration(layer: layer)
        }

        layer.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)
        layer.alignmentMode = CATextLayerAlignmentMode(rawValue: "future-alignment")
        #expect(throws: CATextRenderFailure.unsupportedAlignmentMode("future-alignment")) {
            try CATextRenderConfiguration(layer: layer)
        }

        layer.alignmentMode = .left
        layer.truncationMode = CATextLayerTruncationMode(rawValue: "future-truncation")
        #expect(throws: CATextRenderFailure.unsupportedTruncationMode("future-truncation")) {
            try CATextRenderConfiguration(layer: layer)
        }
    }
}
