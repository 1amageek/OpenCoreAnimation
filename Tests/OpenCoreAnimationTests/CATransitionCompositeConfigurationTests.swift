import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Transition composite configuration")
struct CATransitionCompositeConfigurationTests {
    @Test("Bounds and offset produce finite GPU inputs")
    func validConfiguration() throws {
        let configuration = try CATransitionCompositeConfiguration(
            bounds: CGRect(x: 4, y: 8, width: 20, height: 10),
            position: CGPoint(x: 5, y: 6),
            offset: CGPoint(x: -2, y: 3),
            opacity: 0.5
        )
        #expect(configuration.size == SIMD2<Float>(20, 10))
        #expect(configuration.translatedPosition == CGPoint(x: 3, y: 9))
        #expect(configuration.opacity == 0.5)
    }

    @Test("Invalid GPU inputs retain their exact failure")
    func failures() {
        let oversizedBounds = CGRect(
            x: 0,
            y: 0,
            width: CGFloat(Float.greatestFiniteMagnitude) * 2,
            height: 10
        )
        #expect(throws: CATransitionRenderFailure.invalidCompositeBounds(oversizedBounds)) {
            try CATransitionCompositeConfiguration(
                bounds: oversizedBounds,
                position: .zero,
                offset: .zero,
                opacity: 1
            )
        }
        #expect(throws: CATransitionRenderFailure.invalidCompositeOpacity(0)) {
            try CATransitionCompositeConfiguration(
                bounds: CGRect(x: 0, y: 0, width: 10, height: 10),
                position: .zero,
                offset: .zero,
                opacity: 0
            )
        }
        let oversizedOffset = CGPoint(
            x: CGFloat(Float.greatestFiniteMagnitude) * 2,
            y: 0
        )
        #expect(throws: CATransitionRenderFailure.invalidCompositeOffset(oversizedOffset)) {
            try CATransitionCompositeConfiguration(
                bounds: CGRect(x: 0, y: 0, width: 10, height: 10),
                position: .zero,
                offset: oversizedOffset,
                opacity: 1
            )
        }
    }
}
