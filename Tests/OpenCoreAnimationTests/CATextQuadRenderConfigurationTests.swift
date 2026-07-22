import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Text quad render configuration")
struct CATextQuadRenderConfigurationTests {
    @Test("Valid clipped and unclipped quads preserve only active corner geometry")
    func validConfigurations() throws {
        let clipped = try makeConfiguration(
            color: SIMD4(1.5, 0.5, 0.25, 1),
            opacity: 0.75,
            masksToBounds: true,
            cornerRadius: 4,
            cornerCurveExponent: 4,
            cornerRadii: SIMD4(1, 2, 3, 4)
        )
        #expect(clipped.size == SIMD2<Float>(40, 20))
        #expect(clipped.color == SIMD4<Float>(1.5, 0.5, 0.25, 1))
        #expect(clipped.opacity == 0.75)
        #expect(clipped.cornerRadius == 4)
        #expect(clipped.cornerCurveExponent == 4)
        #expect(clipped.cornerRadii == SIMD4<Float>(1, 2, 3, 4))

        let unclipped = try makeConfiguration(
            masksToBounds: false,
            cornerRadius: -.infinity,
            cornerCurveExponent: .nan,
            cornerRadii: SIMD4(repeating: .nan)
        )
        #expect(unclipped.cornerRadius == 0)
        #expect(unclipped.cornerRadii == .zero)
    }

    @Test("Non-Float geometry is rejected before GPU upload")
    func invalidGeometry() {
        #expect(throws: CATextRenderFailure.invalidBounds) {
            try makeConfiguration(
                bounds: CGRect(
                    x: 0,
                    y: 0,
                    width: CGFloat.greatestFiniteMagnitude,
                    height: 20
                )
            )
        }
    }

    @Test("Non-finite compositing values retain their exact typed reason")
    func invalidCompositingValues() {
        let invalidColor = SIMD4<Float>(1, .infinity, 1, 1)
        #expect(throws: CATextRenderFailure.invalidReplicatorColor(invalidColor)) {
            try makeConfiguration(color: invalidColor)
        }
        #expect(throws: CATextRenderFailure.invalidOpacity(-1)) {
            try makeConfiguration(opacity: -1)
        }
        #expect(throws: CATextRenderFailure.invalidOpacity(.infinity)) {
            try makeConfiguration(opacity: .infinity)
        }
    }

    @Test("Invalid active corner geometry is rejected")
    func invalidCornerGeometry() {
        #expect(throws: CATextRenderFailure.invalidCornerGeometry) {
            try makeConfiguration(masksToBounds: true, cornerRadius: -1)
        }
        #expect(throws: CATextRenderFailure.invalidCornerGeometry) {
            try makeConfiguration(masksToBounds: true, cornerCurveExponent: 0)
        }
        #expect(throws: CATextRenderFailure.invalidCornerGeometry) {
            try makeConfiguration(
                masksToBounds: true,
                cornerRadii: SIMD4(0, -1, 0, 0)
            )
        }
    }

    private func makeConfiguration(
        bounds: CGRect = CGRect(x: 0, y: 0, width: 40, height: 20),
        color: SIMD4<Float> = SIMD4(repeating: 1),
        opacity: Float = 1,
        masksToBounds: Bool = false,
        cornerRadius: CGFloat = 0,
        cornerCurveExponent: Float = 2,
        cornerRadii: SIMD4<Float> = .zero
    ) throws(CATextRenderFailure) -> CATextQuadRenderConfiguration {
        try CATextQuadRenderConfiguration(
            bounds: bounds,
            color: color,
            opacity: opacity,
            masksToBounds: masksToBounds,
            cornerRadius: cornerRadius,
            cornerCurveExponent: cornerCurveExponent,
            cornerRadii: cornerRadii
        )
    }
}
