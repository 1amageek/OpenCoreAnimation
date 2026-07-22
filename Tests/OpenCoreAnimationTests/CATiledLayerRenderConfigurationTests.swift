import Foundation
import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

@Suite("CATiledLayer render configuration")
struct CATiledLayerRenderConfigurationTests {
    @Test("Valid tile input preserves renderer geometry")
    func validConfiguration() throws {
        let layer = CATiledLayer()
        layer.bounds = CGRect(x: 4, y: 8, width: 512, height: 256)
        layer.contentsScale = 2
        layer.tileSize = CGSize(width: 128, height: 64)
        layer.levelsOfDetail = 4
        layer.levelsOfDetailBias = 2

        let configuration = try CATiledLayerRenderConfiguration(layer: layer)

        #expect(configuration.bounds == layer.bounds)
        #expect(configuration.contentsScale == 2)
        #expect(configuration.tileSize == CGSize(width: 128, height: 64))
        #expect(configuration.levelsOfDetail == 4)
        #expect(configuration.levelsOfDetailBias == 2)
    }

    @Test("Invalid detail levels fail with their public values")
    func invalidDetailLevels() {
        let layer = CATiledLayer()
        layer.levelsOfDetail = 0
        #expect(throws: CATiledLayerRenderFailure.invalidLevelsOfDetail(0)) {
            try CATiledLayerRenderConfiguration(layer: layer)
        }

        layer.levelsOfDetail = 1
        layer.levelsOfDetailBias = -1
        #expect(throws: CATiledLayerRenderFailure.invalidLevelsOfDetailBias(-1)) {
            try CATiledLayerRenderConfiguration(layer: layer)
        }
    }

    @Test("Invalid tile geometry and scale fail instead of producing empty tiles")
    func invalidGeometry() {
        let layer = CATiledLayer()
        layer.tileSize = CGSize(width: 0, height: 256)
        #expect(throws: CATiledLayerRenderFailure.invalidTileSize(layer.tileSize)) {
            try CATiledLayerRenderConfiguration(layer: layer)
        }

        layer.tileSize = CGSize(width: 256, height: 256)
        layer.contentsScale = -1
        #expect(throws: CATiledLayerRenderFailure.invalidContentsScale(-1)) {
            try CATiledLayerRenderConfiguration(layer: layer)
        }

        layer.contentsScale = 1
        layer.bounds = CGRect(x: 0, y: 0, width: CGFloat.infinity, height: 10)
        #expect(throws: CATiledLayerRenderFailure.invalidBounds(layer.bounds)) {
            try CATiledLayerRenderConfiguration(layer: layer)
        }
    }
}
