import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Layer filter composite configuration")
struct CALayerFilterCompositeConfigurationTests {
    @Test("Finite opacity and extended-range color are retained")
    func validConfiguration() throws {
        let configuration = try CALayerFilterCompositeConfiguration(
            opacity: 0.75,
            colorMultiplier: SIMD4(2, 0.5, -0.25, 1)
        )
        #expect(configuration.opacity == 0.75)
        #expect(configuration.colorMultiplier == SIMD4<Float>(2, 0.5, -0.25, 1))
    }

    @Test("Invalid compositing values retain their exact typed reason")
    func invalidConfiguration() {
        #expect(throws: CALayerFilterRenderFailure.invalidCompositeOpacity(-1)) {
            try CALayerFilterCompositeConfiguration(
                opacity: -1,
                colorMultiplier: SIMD4(repeating: 1)
            )
        }
        #expect(throws: CALayerFilterRenderFailure.invalidCompositeOpacity(.infinity)) {
            try CALayerFilterCompositeConfiguration(
                opacity: .infinity,
                colorMultiplier: SIMD4(repeating: 1)
            )
        }

        let invalidColor = SIMD4<Float>(1, 1, .infinity, 1)
        #expect(throws: CALayerFilterRenderFailure.invalidCompositeColor(invalidColor)) {
            try CALayerFilterCompositeConfiguration(
                opacity: 1,
                colorMultiplier: invalidColor
            )
        }
    }
}
