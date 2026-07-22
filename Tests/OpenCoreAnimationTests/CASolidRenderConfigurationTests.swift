import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Solid render configuration")
struct CASolidRenderConfigurationTests {
    @Test("Finite rounded border inputs retain GPU values")
    func validConfiguration() throws {
        let configuration = try CASolidRenderConfiguration(
            bounds: CGRect(x: 4, y: 8, width: 20, height: 10),
            color: SIMD4(2, 0.5, 0.25, 1),
            opacity: 0.75,
            cornerRadius: 3,
            cornerCurveExponent: 2,
            cornerRadii: SIMD4(repeating: 3),
            borderWidth: 2,
            context: .border
        )
        #expect(configuration.size == SIMD2<Float>(20, 10))
        #expect(configuration.color == SIMD4<Float>(2, 0.5, 0.25, 1))
        #expect(configuration.opacity == 0.75)
        #expect(configuration.borderWidth == 2)
    }

    @Test("Invalid geometry retains solid-quad context")
    func invalidGeometry() {
        let zeroWidthBounds = CGRect(x: 0, y: 0, width: 0, height: 10)
        #expect(throws: CASolidRenderFailure.invalidGeometry(.background, zeroWidthBounds)) {
            try makeConfiguration(bounds: zeroWidthBounds, context: .background)
        }

        let overflowingBounds = CGRect(
            x: 0,
            y: 0,
            width: CGFloat.greatestFiniteMagnitude,
            height: 10
        )
        #expect(throws: CASolidRenderFailure.invalidGeometry(.border, overflowingBounds)) {
            try makeConfiguration(bounds: overflowingBounds, context: .border)
        }
    }

    @Test("Invalid color and opacity retain solid-quad context")
    func invalidAppearance() {
        let bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        let invalidColor = SIMD4<Float>(1, .infinity, 1, 1)
        #expect(throws: CASolidRenderFailure.invalidColor(.border, invalidColor)) {
            try makeConfiguration(bounds: bounds, color: invalidColor, context: .border)
        }

        #expect(throws: CASolidRenderFailure.invalidOpacity(.background, -1)) {
            try makeConfiguration(bounds: bounds, opacity: -1, context: .background)
        }
        #expect(throws: CASolidRenderFailure.invalidOpacity(.border, .infinity)) {
            try makeConfiguration(bounds: bounds, opacity: .infinity, context: .border)
        }
    }

    @Test("Invalid corner inputs retain solid-quad context")
    func invalidCorners() {
        let bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        #expect(throws: CASolidRenderFailure.invalidCornerGeometry(.background)) {
            try makeConfiguration(bounds: bounds, cornerRadius: -1, context: .background)
        }
        #expect(throws: CASolidRenderFailure.invalidCornerGeometry(.border)) {
            try makeConfiguration(bounds: bounds, cornerCurveExponent: 0, context: .border)
        }
        #expect(throws: CASolidRenderFailure.invalidCornerGeometry(.background)) {
            try makeConfiguration(
                bounds: bounds,
                cornerRadii: SIMD4(0, 0, -1, 0),
                context: .background
            )
        }
    }

    @Test("Invalid border width retains its rejected GPU value")
    func invalidBorderWidth() {
        let bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        #expect(throws: CASolidRenderFailure.invalidBorderWidth(-1)) {
            try makeConfiguration(bounds: bounds, borderWidth: -1, context: .border)
        }
        #expect(throws: CASolidRenderFailure.invalidBorderWidth(.infinity)) {
            try makeConfiguration(bounds: bounds, borderWidth: .infinity, context: .border)
        }
    }

    private func makeConfiguration(
        bounds: CGRect,
        color: SIMD4<Float> = SIMD4(repeating: 1),
        opacity: Float = 1,
        cornerRadius: CGFloat = 0,
        cornerCurveExponent: Float = 2,
        cornerRadii: SIMD4<Float> = .zero,
        borderWidth: CGFloat = 0,
        context: CASolidRenderContext
    ) throws(CASolidRenderFailure) -> CASolidRenderConfiguration {
        try CASolidRenderConfiguration(
            bounds: bounds,
            color: color,
            opacity: opacity,
            cornerRadius: cornerRadius,
            cornerCurveExponent: cornerCurveExponent,
            cornerRadii: cornerRadii,
            borderWidth: borderWidth,
            context: context
        )
    }
}
