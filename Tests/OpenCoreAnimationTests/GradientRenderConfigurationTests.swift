import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Gradient render configuration")
struct GradientRenderConfigurationTests {
    private let red = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
    private let green = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
    private let blue = CGColor(red: 0, green: 0, blue: 1, alpha: 1)

    @Test("Every public gradient type selects a distinct renderer mode")
    func renderModes() throws {
        let colors: [Any] = [red, blue]
        let axial = try configuration(type: .axial, colors: colors)
        let radial = try configuration(type: .radial, colors: colors)
        let conic = try configuration(type: .conic, colors: colors)

        #expect(axial.renderMode == 2)
        #expect(radial.renderMode == 3)
        #expect(conic.renderMode == 4)
        #expect(axial.colors.count == 2)
    }

    @Test("Missing locations are distributed across every color stop")
    func uniformLocations() throws {
        let result = try configuration(type: .axial, colors: [red, green, blue])
        #expect(result.locations == [0, 0.5, 1])

        let single = try configuration(type: .axial, colors: [red])
        #expect(single.locations == [0])
    }

    @Test("Explicit valid locations preserve coincident stops")
    func explicitLocations() throws {
        let result = try GradientRenderConfiguration(
            type: .radial,
            colors: [red, green, blue],
            locations: [0.25, 0.25, 0.75],
            startPoint: CGPoint(x: 0.5, y: 0.5),
            endPoint: CGPoint(x: 1, y: 1)
        )

        #expect(result.locations == [0.25, 0.25, 0.75])
    }

    @Test("Colors convert to finite device-RGB upload components")
    func deviceRGBComponents() throws {
        let result = try configuration(
            type: .axial,
            colors: [CGColor(gray: 0.25, alpha: 0.5)]
        )

        #expect(result.colors.first?.colorSpace == .deviceRGB)
        #expect(result.colorComponents == [SIMD4<Float>(0.25, 0.25, 0.25, 0.5)])
    }

    @Test("Unknown types and invalid colors are rejected without limiting valid stop counts")
    func invalidStops() throws {
        let nonFiniteColor = CGColor(red: .nan, green: 0, blue: 0, alpha: 1)

        #expect(throws: GradientRenderConfigurationError.unsupportedType("future")) {
            try configuration(type: CAGradientLayerType(rawValue: "future"), colors: [red])
        }
        #expect(throws: GradientRenderConfigurationError.invalidColor(index: 1)) {
            try configuration(type: .axial, colors: [red, "not-a-color"])
        }
        #expect(throws: GradientRenderConfigurationError.invalidColorComponents(index: 0)) {
            try configuration(type: .axial, colors: [nonFiniteColor])
        }
        let manyStops = try configuration(
            type: .axial,
            colors: Array(repeating: red as Any, count: 257)
        )
        #expect(manyStops.colors.count == 257)
        #expect(manyStops.locations.count == 257)
        #expect(manyStops.locations.last == 1)
        #expect(throws: GradientRenderConfigurationError.nonFiniteGeometry) {
            try GradientRenderConfiguration(
                type: .axial,
                colors: [red],
                locations: nil,
                startPoint: CGPoint(x: CGFloat.infinity, y: 0),
                endPoint: CGPoint(x: 1, y: 1)
            )
        }
    }

    @Test("Locations must match colors, remain finite, stay in range, and be monotonic")
    func invalidLocations() {
        #expect(throws: GradientRenderConfigurationError.invalidLocationCount(expected: 2, actual: 1)) {
            try configuration(type: .axial, colors: [red, blue], locations: [0])
        }
        #expect(throws: GradientRenderConfigurationError.nonFiniteLocation(index: 1)) {
            try configuration(type: .axial, colors: [red, blue], locations: [0, .nan])
        }
        #expect(throws: GradientRenderConfigurationError.locationOutOfRange(index: 1)) {
            try configuration(type: .axial, colors: [red, blue], locations: [0, 1.1])
        }
        #expect(throws: GradientRenderConfigurationError.locationsNotMonotonic(index: 2)) {
            try configuration(type: .axial, colors: [red, green, blue], locations: [0, 0.8, 0.4])
        }
    }

    @Test("Axial and radial parameters use unit-coordinate geometry")
    func axialAndRadialParameters() throws {
        let axial = try GradientRenderConfiguration.parameter(
            at: CGPoint(x: 0.75, y: 0.5),
            type: .axial,
            startPoint: CGPoint(x: 0.25, y: 0.5),
            endPoint: CGPoint(x: 1.25, y: 0.5)
        )
        #expect(axial == 0.5)

        let radialCenter = try radialParameter(at: CGPoint(x: 0.5, y: 0.5))
        let radialRight = try radialParameter(at: CGPoint(x: 1, y: 0.5))
        let radialTop = try radialParameter(at: CGPoint(x: 0.5, y: 1))
        let radialCorner = try radialParameter(at: CGPoint(x: 1, y: 1))
        #expect(radialCenter == 0)
        #expect(radialRight == 1)
        #expect(radialTop == 1)
        #expect(abs(try #require(radialCorner) - sqrt(2)) < 0.000_001)

        let degenerate = try GradientRenderConfiguration.parameter(
            at: CGPoint(x: 0.5, y: 0.5),
            type: .radial,
            startPoint: CGPoint(x: 0.5, y: 0.5),
            endPoint: CGPoint(x: 1, y: 0.5)
        )
        #expect(degenerate == nil)
    }

    @Test("Conic zero aligns with the start-to-end ray and advances counterclockwise")
    func conicParameter() throws {
        let points = [
            CGPoint(x: 1, y: 0.5),
            CGPoint(x: 0.5, y: 1),
            CGPoint(x: 0, y: 0.5),
            CGPoint(x: 0.5, y: 0),
        ]
        let expected: [CGFloat] = [0, 0.25, 0.5, 0.75]

        for (point, expectedValue) in zip(points, expected) {
            let value = try GradientRenderConfiguration.parameter(
                at: point,
                type: .conic,
                startPoint: CGPoint(x: 0.5, y: 0.5),
                endPoint: CGPoint(x: 1, y: 0.5)
            )
            #expect(abs(try #require(value) - expectedValue) < 0.000_001)
        }
    }

    private func configuration(
        type: CAGradientLayerType,
        colors: [Any],
        locations: [CGFloat]? = nil
    ) throws -> GradientRenderConfiguration {
        try GradientRenderConfiguration(
            type: type,
            colors: colors,
            locations: locations,
            startPoint: CGPoint(x: 0.5, y: 0),
            endPoint: CGPoint(x: 0.5, y: 1)
        )
    }

    private func radialParameter(at point: CGPoint) throws -> CGFloat? {
        try GradientRenderConfiguration.parameter(
            at: point,
            type: .radial,
            startPoint: CGPoint(x: 0.5, y: 0.5),
            endPoint: CGPoint(x: 1, y: 1)
        )
    }
}
