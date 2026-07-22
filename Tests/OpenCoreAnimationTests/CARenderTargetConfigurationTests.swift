import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("Render target configuration")
struct CARenderTargetConfigurationTests {
    @Test("Valid dimensions preserve exact integer and viewport values")
    func validDimensions() throws {
        let configuration = try CARenderTargetConfiguration(
            width: 1920,
            height: 1080,
            maximumTextureDimension: 8192
        )
        #expect(configuration.width == 1920)
        #expect(configuration.height == 1080)
        #expect(configuration.viewportSize == SIMD2<Float>(1920, 1080))
    }

    @Test("Non-finite, fractional, and non-positive dimensions are rejected")
    func invalidDimensions() {
        for dimensions in [
            SIMD2<Double>(.infinity, 100),
            SIMD2<Double>(100.5, 100),
            SIMD2<Double>(100, 0),
            SIMD2<Double>(-1, 100),
        ] {
            #expect(throws: CARenderTargetConfigurationError.invalidDimensions(
                width: dimensions.x,
                height: dimensions.y
            )) {
                try CARenderTargetConfiguration(
                    width: dimensions.x,
                    height: dimensions.y,
                    maximumTextureDimension: 8192
                )
            }
        }
    }

    @Test("Device dimension limits retain the exact rejected values")
    func deviceLimit() {
        #expect(throws: CARenderTargetConfigurationError.dimensionLimitExceeded(
            width: 8193,
            height: 4096,
            maximum: 8192
        )) {
            try CARenderTargetConfiguration(
                width: 8193,
                height: 4096,
                maximumTextureDimension: 8192
            )
        }
    }
}
