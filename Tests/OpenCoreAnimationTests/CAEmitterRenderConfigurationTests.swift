import Foundation
import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

@Suite("CAEmitterLayer render configuration")
struct CAEmitterRenderConfigurationTests {
    @Test("Valid input preserves simulation and geometry state")
    func validConfiguration() throws {
        let cell = CAEmitterCell()
        let layer = CAEmitterLayer()
        layer.emitterCells = [cell]
        layer.emitterPosition = CGPoint(x: 10, y: 20)
        layer.emitterZPosition = 30
        layer.emitterSize = CGSize(width: -40, height: 50)
        layer.emitterDepth = -60
        layer.emitterShape = .cuboid
        layer.emitterMode = .surface
        layer.renderMode = .backToFront
        layer.preservesDepth = true
        layer.birthRate = 2
        layer.lifetime = 3
        layer.velocity = 4
        layer.scale = 5
        layer.spin = 6
        layer.seed = 7

        let configuration = try CAEmitterRenderConfiguration(layer: layer)

        #expect(configuration.emitterCells.count == 1)
        #expect(configuration.emitterCells[0] === cell)
        #expect(configuration.emitterPosition == CGPoint(x: 10, y: 20))
        #expect(configuration.emitterZPosition == 30)
        #expect(configuration.emitterSize == CGSize(width: -40, height: 50))
        #expect(configuration.emitterDepth == -60)
        #expect(configuration.emitterShape == .cuboid)
        #expect(configuration.emitterMode == .surface)
        #expect(configuration.renderMode == .backToFront)
        #expect(configuration.preservesDepth)
        #expect(configuration.birthRate == 2)
        #expect(configuration.lifetime == 3)
        #expect(configuration.velocity == 4)
        #expect(configuration.scale == 5)
        #expect(configuration.spin == 6)
        #expect(configuration.seed == 7)
    }

    @Test("Unknown modes fail before simulation or GPU work")
    func unsupportedModes() {
        let layer = CAEmitterLayer()
        layer.emitterShape = CAEmitterLayerEmitterShape(rawValue: "future-shape")
        #expect(throws: CAEmitterFailure.unsupportedEmitterShape("future-shape")) {
            try CAEmitterRenderConfiguration(layer: layer)
        }

        layer.emitterShape = .point
        layer.emitterMode = CAEmitterLayerEmitterMode(rawValue: "future-mode")
        #expect(throws: CAEmitterFailure.unsupportedEmitterMode("future-mode")) {
            try CAEmitterRenderConfiguration(layer: layer)
        }

        layer.emitterMode = .volume
        layer.renderMode = CAEmitterLayerRenderMode(rawValue: "future-render")
        #expect(throws: CAEmitterFailure.unsupportedRenderMode("future-render")) {
            try CAEmitterRenderConfiguration(layer: layer)
        }
    }

    @Test("Non-finite geometry and simulation values fail explicitly")
    func nonFiniteValues() {
        let layer = CAEmitterLayer()
        layer.emitterPosition.x = .infinity
        #expect(throws: CAEmitterFailure.nonFiniteLayerGeometry) {
            try CAEmitterRenderConfiguration(layer: layer)
        }

        layer.emitterPosition = .zero
        layer.birthRate = .nan
        #expect(throws: CAEmitterFailure.nonFiniteLayerSimulationValue) {
            try CAEmitterRenderConfiguration(layer: layer)
        }
    }
}
