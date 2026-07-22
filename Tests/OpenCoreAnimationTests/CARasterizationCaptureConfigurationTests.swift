import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Rasterization capture configuration")
struct CARasterizationCaptureConfigurationTests {
    @Test("Scale and device limits determine the capture texture size")
    func pixelSize() throws {
        let regular = try CARasterizationCaptureConfiguration(
            captureBounds: CGRect(x: -10, y: 5, width: 40, height: 20),
            rasterizationScale: 2,
            maximumTextureDimension: 1_024
        )
        #expect(regular.pixelWidth == 80)
        #expect(regular.pixelHeight == 40)

        let limited = try CARasterizationCaptureConfiguration(
            captureBounds: CGRect(x: 0, y: 0, width: 200, height: 100),
            rasterizationScale: 2,
            maximumTextureDimension: 100
        )
        #expect(limited.pixelWidth == 100)
        #expect(limited.pixelHeight == 50)
    }

    @Test("Invalid scale and scaled extent retain their exact failure")
    func failures() {
        #expect(throws: CARasterizationRenderFailure.invalidRasterizationScale(0)) {
            try CARasterizationCaptureConfiguration(
                captureBounds: CGRect(x: 0, y: 0, width: 10, height: 10),
                rasterizationScale: 0,
                maximumTextureDimension: 100
            )
        }
        #expect(throws: CARasterizationRenderFailure.invalidScaledExtent(
            CGSize(width: CGFloat.infinity, height: 10)
        )) {
            try CARasterizationCaptureConfiguration(
                captureBounds: CGRect(x: 0, y: 0, width: CGFloat.infinity, height: 10),
                rasterizationScale: 1,
                maximumTextureDimension: 100
            )
        }

        let oversizedBounds = CGRect(
            x: 0,
            y: 0,
            width: CGFloat(Float.greatestFiniteMagnitude) * 2,
            height: 10
        )
        #expect(throws: CARasterizationRenderFailure.captureProjectionOutOfRange(
            oversizedBounds
        )) {
            try CARasterizationCaptureConfiguration(
                captureBounds: oversizedBounds,
                rasterizationScale: 1,
                maximumTextureDimension: 100
            )
        }
    }
}
