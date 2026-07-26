import Foundation
import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

@Suite("CAEmitterLayer render configuration")
struct CAEmitterRenderConfigurationTests {
    @Test("Presentation copies retain identity while new emitters are distinct")
    func simulationIdentity() {
        let original = CAEmitterLayer()
        let presentationCopy = CAEmitterLayer(layer: original)
        let independent = CAEmitterLayer()
        let firstCell = CAEmitterCell()
        let secondCell = CAEmitterCell()

        #expect(
            presentationCopy.simulationIdentity
                == original.simulationIdentity
        )
        #expect(
            independent.simulationIdentity
                != original.simulationIdentity
        )
        #expect(
            firstCell.simulationIdentity
                != secondCell.simulationIdentity
        )
    }

    @Test("Cell mutations invalidate every owning layer")
    func cellMutationsInvalidateOwners() throws {
        let child = CAEmitterCell()
        child.birthRate = 2
        let parent = CAEmitterCell()
        parent.birthRate = 1
        parent.emitterCells = [child]
        let firstLayer = CAEmitterLayer()
        let secondLayer = CAEmitterLayer()

        firstLayer.emitterCells = [parent]
        secondLayer.emitterCells = [parent]
        let firstRevision = firstLayer._contentRevision
        let secondRevision = secondLayer._contentRevision
        let firstSnapshot = try CARenderSnapshot.capture(
            firstLayer,
            frameToken: 1
        )
        #expect(
            firstSnapshot.nodes[firstSnapshot.rootIndex]
                .presentationValues.emitter?
                .emitterCells.first?
                .childCells.first?
                .birthRate == 2
        )

        child.birthRate = 9
        #expect(firstLayer._contentRevision == firstRevision &+ 1)
        #expect(secondLayer._contentRevision == secondRevision &+ 1)
        let updatedFirstSnapshot = try CARenderSnapshot.capture(
            firstLayer,
            frameToken: 2
        )
        let updatedSecondSnapshot = try CARenderSnapshot.capture(
            secondLayer,
            frameToken: 3
        )
        #expect(
            updatedFirstSnapshot.nodes[updatedFirstSnapshot.rootIndex]
                .presentationValues.emitter?
                .emitterCells.first?
                .childCells.first?
                .birthRate == 9
        )
        #expect(
            updatedSecondSnapshot.nodes[updatedSecondSnapshot.rootIndex]
                .presentationValues.emitter?
                .emitterCells.first?
                .childCells.first?
                .birthRate == 9
        )

        firstLayer.emitterCells = []
        secondLayer.emitterCells = []
        let detachedFirstRevision = firstLayer._contentRevision
        let detachedSecondRevision = secondLayer._contentRevision

        child.birthRate = 10
        #expect(
            firstLayer._contentRevision
                == detachedFirstRevision
        )
        #expect(
            secondLayer._contentRevision
                == detachedSecondRevision
        )
    }

    @Test("Valid input preserves simulation and geometry state")
    func validConfiguration() throws {
        let cell = CAEmitterCell()
        cell.birthRate = 8
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
        #expect(
            configuration.emitterCells[0].identity
                == cell.simulationIdentity
        )
        #expect(configuration.emitterCells[0].birthRate == 8)
        #expect(
            configuration.simulationIdentity
                == layer.simulationIdentity
        )
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

        cell.birthRate = 99
        layer.emitterCells = []
        #expect(configuration.emitterCells.count == 1)
        #expect(configuration.emitterCells[0].birthRate == 8)
    }

    @Test("Unknown modes fail before simulation or GPU work")
    func unsupportedModes() {
        let layer = CAEmitterLayer()
        layer.emitterShape = CAEmitterLayerEmitterShape(rawValue: "future-shape")
        #expect(throws: CARenderSnapshotEmitterError
            .unsupportedEmitterShape("future-shape")) {
            try CAEmitterRenderConfiguration(layer: layer)
        }

        layer.emitterShape = .point
        layer.emitterMode = CAEmitterLayerEmitterMode(rawValue: "future-mode")
        #expect(throws: CARenderSnapshotEmitterError
            .unsupportedEmitterMode("future-mode")) {
            try CAEmitterRenderConfiguration(layer: layer)
        }

        layer.emitterMode = .volume
        layer.renderMode = CAEmitterLayerRenderMode(rawValue: "future-render")
        #expect(throws: CARenderSnapshotEmitterError
            .unsupportedRenderMode("future-render")) {
            try CAEmitterRenderConfiguration(layer: layer)
        }
    }

    @Test("Non-finite geometry and simulation values fail explicitly")
    func nonFiniteValues() {
        let layer = CAEmitterLayer()
        layer.emitterPosition.x = .infinity
        #expect(throws: CARenderSnapshotEmitterError
            .nonFiniteLayerGeometry) {
            try CAEmitterRenderConfiguration(layer: layer)
        }

        layer.emitterPosition = .zero
        layer.birthRate = .nan
        #expect(throws: CARenderSnapshotEmitterError
            .nonFiniteLayerSimulationValue) {
            try CAEmitterRenderConfiguration(layer: layer)
        }
    }
}
