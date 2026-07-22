import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Corner Curve Render Configuration Tests")
struct CornerCurveRenderConfigurationTests {
    @Test("known corner curves resolve to distinct finite exponents")
    func knownCurvesResolveToDistinctExponents() throws {
        let circular = try CornerCurveRenderConfiguration(curve: .circular)
        let continuous = try CornerCurveRenderConfiguration(curve: .continuous)

        #expect(circular.exponent == 2)
        #expect(continuous.exponent == 2.2)
        #expect(circular.exponent.isFinite)
        #expect(continuous.exponent.isFinite)
        #expect(continuous.exponent > circular.exponent)
    }

    @Test("unknown corner curves fail with their raw value")
    func unknownCurveFailsExplicitly() {
        let curve = CALayerCornerCurve(rawValue: "future-curve")

        #expect(throws: CornerCurveRenderConfigurationError.unsupportedCurve("future-curve")) {
            try CornerCurveRenderConfiguration(curve: curve)
        }
    }
}
