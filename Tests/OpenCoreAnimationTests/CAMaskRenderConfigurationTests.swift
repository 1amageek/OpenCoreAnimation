import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Mask render configuration")
struct CAMaskRenderConfigurationTests {
    @Test("Finite layer geometry becomes GPU mask inputs")
    func validGeometry() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 4, y: 8, width: 20, height: 10)
        layer.cornerRadius = 3
        let configuration = try CAMaskRenderConfiguration(
            bounds: layer.bounds,
            cornerRadius: layer.cornerRadius,
            cornerCurveExponent: 2,
            cornerRadii: SIMD4(repeating: 3),
            context: .roundedClip
        )
        #expect(configuration.size == SIMD2<Float>(20, 10))
        #expect(configuration.cornerRadius == 3)
        #expect(configuration.cornerCurveExponent.isFinite)
    }

    @Test("Float-overflowing geometry retains mask context")
    func invalidGeometry() {
        let layer = CALayer()
        layer.bounds = CGRect(
            x: 0,
            y: 0,
            width: CGFloat(Float.greatestFiniteMagnitude) * 2,
            height: 10
        )
        #expect(throws: CAMaskRenderFailure.invalidGeometry(.contentMask, layer.bounds)) {
            try CAMaskRenderConfiguration(
                bounds: layer.bounds,
                cornerRadius: 0,
                cornerCurveExponent: 2,
                cornerRadii: .zero,
                context: .contentMask
            )
        }
        #expect(throws: CAMaskRenderFailure.invalidCornerGeometry(
            .roundedClip,
            radius: 2,
            exponent: 0,
            radii: SIMD4(repeating: 2)
        )) {
            try CAMaskRenderConfiguration(
                bounds: CGRect(x: 0, y: 0, width: 10, height: 10),
                cornerRadius: 2,
                cornerCurveExponent: 0,
                cornerRadii: SIMD4(repeating: 2),
                context: .roundedClip
            )
        }
    }
}
