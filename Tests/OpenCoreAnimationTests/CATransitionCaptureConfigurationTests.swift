import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Transition capture configuration")
struct CATransitionCaptureConfigurationTests {
    @Test("Capture size is scale-aware and device-limited")
    func pixelSize() throws {
        let configuration = try CATransitionCaptureConfiguration(
            bounds: CGRect(x: -5, y: 10, width: 200, height: 100),
            contentsScale: 2,
            pixelSizeOverride: nil,
            maximumTextureDimension: 100,
            role: .source
        )
        #expect(configuration.pixelWidth == 100)
        #expect(configuration.pixelHeight == 50)
        #expect(configuration.projectionLeft == -5)
        #expect(configuration.projectionRight == 195)
    }

    @Test("Invalid values retain participant role and exact reason")
    func failures() {
        let bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        #expect(throws: CATransitionRenderFailure.invalidParticipantContentsScale(
            .target,
            0
        )) {
            try CATransitionCaptureConfiguration(
                bounds: bounds,
                contentsScale: 0,
                pixelSizeOverride: nil,
                maximumTextureDimension: 100,
                role: .target
            )
        }

        let oversizedBounds = CGRect(
            x: 0,
            y: 0,
            width: CGFloat(Float.greatestFiniteMagnitude) * 2,
            height: 10
        )
        #expect(throws: CATransitionRenderFailure.participantProjectionOutOfRange(
            .source,
            oversizedBounds
        )) {
            try CATransitionCaptureConfiguration(
                bounds: oversizedBounds,
                contentsScale: 1,
                pixelSizeOverride: nil,
                maximumTextureDimension: 100,
                role: .source
            )
        }
    }
}
