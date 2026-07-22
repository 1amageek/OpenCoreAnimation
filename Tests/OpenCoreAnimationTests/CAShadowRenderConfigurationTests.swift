import Foundation
import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

@Suite("CALayer shadow render configuration")
struct CAShadowRenderConfigurationTests {
    @Test("Valid shadows preserve converted color and geometry")
    func validConfiguration() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 1, y: 2, width: 30, height: 40)
        layer.shadowColor = CGColor(gray: 0.25, alpha: 0.5)
        layer.shadowOpacity = 0.75
        layer.shadowRadius = -4
        layer.shadowOffset = CGSize(width: 5, height: -6)

        let configuration = try CAShadowRenderConfiguration(layer: layer)

        #expect(configuration.color == SIMD4<Float>(0.25, 0.25, 0.25, 0.5))
        #expect(configuration.opacity == 0.75)
        #expect(configuration.radius == 0)
        #expect(configuration.offset == CGSize(width: 5, height: -6))
    }

    @Test("Non-finite geometry fails explicitly")
    func nonFiniteGeometry() {
        let layer = CALayer()
        layer.shadowOpacity = 1
        layer.shadowRadius = .nan

        #expect(throws: CAShadowRenderFailure.nonFiniteGeometry) {
            try CAShadowRenderConfiguration(layer: layer)
        }
    }

    @Test("Missing and non-finite colors fail explicitly")
    func invalidColor() {
        let layer = CALayer()
        layer.shadowOpacity = 1
        layer.shadowColor = nil
        #expect(throws: CAShadowRenderFailure.invalidColor) {
            try CAShadowRenderConfiguration(layer: layer)
        }

        layer.shadowColor = CGColor(red: .nan, green: 0, blue: 0, alpha: 1)
        #expect(throws: CAShadowRenderFailure.invalidColor) {
            try CAShadowRenderConfiguration(layer: layer)
        }
    }
}
