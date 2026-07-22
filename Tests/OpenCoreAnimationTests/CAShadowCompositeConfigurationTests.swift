import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Shadow composite configuration")
struct CAShadowCompositeConfigurationTests {
    @Test("Finite extended-range compositing values are retained")
    func validConfiguration() throws {
        let shadow = try makeShadowConfiguration()
        let configuration = try CAShadowCompositeConfiguration(
            shadow: shadow,
            effectiveOpacity: 0.5,
            replicatorColor: SIMD4(2, 0.5, -0.25, 2),
            viewportSize: CGSize(width: 400, height: 300)
        )
        #expect(configuration.color == SIMD4<Float>(0.5, 0.125, -0.0625, 0.5))
        #expect(configuration.offset == SIMD2<Float>(3, -4))
        #expect(configuration.viewportSize == SIMD2<Float>(400, 300))
    }

    @Test("Invalid opacity and replicator color retain exact typed reasons")
    func invalidCompositingState() throws {
        let shadow = try makeShadowConfiguration()
        #expect(throws: CAShadowRenderFailure.invalidCompositeOpacity(-1)) {
            try CAShadowCompositeConfiguration(
                shadow: shadow,
                effectiveOpacity: -1,
                replicatorColor: SIMD4(repeating: 1),
                viewportSize: CGSize(width: 400, height: 300)
            )
        }

        let invalidColor = SIMD4<Float>(1, .infinity, 1, 1)
        #expect(throws: CAShadowRenderFailure.invalidReplicatorColor(invalidColor)) {
            try CAShadowCompositeConfiguration(
                shadow: shadow,
                effectiveOpacity: 1,
                replicatorColor: invalidColor,
                viewportSize: CGSize(width: 400, height: 300)
            )
        }
    }

    @Test("Invalid viewport and color multiplication are rejected")
    func invalidOutputState() throws {
        let shadow = try makeShadowConfiguration()
        #expect(throws: CAShadowRenderFailure.invalidCompositeViewport) {
            try CAShadowCompositeConfiguration(
                shadow: shadow,
                effectiveOpacity: 1,
                replicatorColor: SIMD4(repeating: 1),
                viewportSize: CGSize(width: 0, height: 300)
            )
        }
        #expect(throws: CAShadowRenderFailure.compositeColorOverflow) {
            try CAShadowCompositeConfiguration(
                shadow: shadow,
                effectiveOpacity: Float.greatestFiniteMagnitude,
                replicatorColor: SIMD4(repeating: Float.greatestFiniteMagnitude),
                viewportSize: CGSize(width: 400, height: 300)
            )
        }
    }

    private func makeShadowConfiguration() throws -> CAShadowRenderConfiguration {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
        layer.shadowColor = CGColor(red: 0.25, green: 0.25, blue: 0.25, alpha: 1)
        layer.shadowOpacity = 0.5
        layer.shadowRadius = 2
        layer.shadowOffset = CGSize(width: 3, height: -4)
        return try CAShadowRenderConfiguration(layer: layer)
    }
}
