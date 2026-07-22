import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Composition plane render configuration")
struct CACompositionPlaneRenderConfigurationTests {
    private let identityColumns = (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(0, 0, 0, 1)
    )

    @Test("Valid geometry retains layer and viewport sizes")
    func validGeometry() throws {
        let configuration = try CACompositionPlaneRenderConfiguration(
            bounds: CGRect(x: 2, y: 3, width: 40, height: 20),
            viewportSize: CGSize(width: 400, height: 300)
        )
        #expect(configuration.size == SIMD2<Float>(40, 20))
        #expect(configuration.viewportSize == SIMD2<Float>(400, 300))
        #expect(configuration.standardVertices().count == 6)
        try configuration.validateDisplayTransform(columns: identityColumns)
    }

    @Test("Invalid layer or viewport geometry is rejected")
    func invalidGeometry() {
        #expect(throws: CACompositionFilterRenderFailure.invalidDisplayGeometry) {
            try CACompositionPlaneRenderConfiguration(
                bounds: CGRect(x: 0, y: 0, width: CGFloat.greatestFiniteMagnitude, height: 20),
                viewportSize: CGSize(width: 400, height: 300)
            )
        }
        #expect(throws: CACompositionFilterRenderFailure.invalidDisplayGeometry) {
            try CACompositionPlaneRenderConfiguration(
                bounds: CGRect(x: 0, y: 0, width: 40, height: 20),
                viewportSize: CGSize(width: 0, height: 300)
            )
        }
    }

    @Test("Non-finite display transforms are rejected")
    func invalidDisplayTransform() throws {
        let configuration = try makeConfiguration()
        let invalidColumns = (
            SIMD4<Float>(.infinity, 0, 0, 0),
            identityColumns.1,
            identityColumns.2,
            identityColumns.3
        )
        #expect(throws: CACompositionFilterRenderFailure.invalidDisplayTransform) {
            try configuration.validateDisplayTransform(columns: invalidColumns)
        }
    }

    @Test("Captured vertices reject invalid perspective division")
    func capturedVertices() throws {
        let configuration = try makeConfiguration()
        let vertices = try configuration.capturedVertices(samplingColumns: identityColumns)
        #expect(vertices.count == 6)
        #expect(vertices[0].texCoord == SIMD2<Float>(0.5, 0.5))

        let zeroWColumns = (
            identityColumns.0,
            identityColumns.1,
            identityColumns.2,
            SIMD4<Float>(0, 0, 0, 0)
        )
        #expect(throws: CACompositionFilterRenderFailure.invalidSamplingTransform) {
            try configuration.capturedVertices(samplingColumns: zeroWColumns)
        }
    }

    private func makeConfiguration() throws -> CACompositionPlaneRenderConfiguration {
        try CACompositionPlaneRenderConfiguration(
            bounds: CGRect(x: 0, y: 0, width: 40, height: 20),
            viewportSize: CGSize(width: 400, height: 300)
        )
    }
}
