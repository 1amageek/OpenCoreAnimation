import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CATiledLayer Tests")
struct CATiledLayerTests {
    @Test("LOD selection includes magnified and minified detail levels")
    func signedLODSelection() {
        let layer = CATiledLayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 256, height: 128)
        layer.levelsOfDetail = 4
        layer.levelsOfDetailBias = 2

        #expect(layer.lodLevel(forScreenScale: 4) == -2)
        #expect(layer.lodLevel(forScreenScale: 2) == -1)
        #expect(layer.lodLevel(forScreenScale: 1) == 0)
        #expect(layer.lodLevel(forScreenScale: 0.5) == 1)
        #expect(layer.lodLevel(forScreenScale: 0.25) == 2)
        #expect(layer.lodLevel(forScreenScale: 0.01) == 3)
    }

    @Test("LOD selection clamps to a single-pixel minimum dimension")
    func lodSelectionClampsToLayerDimensions() {
        let layer = CATiledLayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 2, height: 128)
        layer.levelsOfDetail = 10

        #expect(layer.lodLevel(forScreenScale: 0.001) == 1)

        layer.bounds.size.width = 0.5
        #expect(layer.lodLevel(forScreenScale: 0.001) == 0)

        layer.bounds.size.width = .infinity
        layer.bounds.size.height = .infinity
        #expect(layer.lodLevel(forScreenScale: 0.001) == 9)
        #expect(layer.lodLevel(forScreenScale: .nan) == 0)
        #expect(layer.lodLevel(forScreenScale: -1) == 0)
    }

    @Test("Newly cached tiles fade in over the class duration")
    func cachedTileFadeProgress() throws {
        let layer = CATiledLayer()
        let key = CATiledLayer.TileKey(column: 1, row: 2, lodLevel: 0)
        let image = try makeImage()

        layer.loadingTiles.insert(key)
        layer.cacheImage(image, for: key, at: 10)

        #expect(layer.cachedImage(for: key) === image)
        #expect(!layer.loadingTiles.contains(key))
        #expect(layer.tileOpacity(for: key, at: 10) == 0)
        #expect(abs(layer.tileOpacity(for: key, at: 10.125) - 0.5) < 0.001)
        #expect(layer.tileOpacity(for: key, at: 10.25) == 1)
        #expect(layer.tileOpacity(for: key, at: 11) == 1)
    }

    @Test("Clearing a tile removes its image and fade state")
    func clearingTileRemovesAllCachedState() throws {
        let layer = CATiledLayer()
        let key = CATiledLayer.TileKey(column: 0, row: 0, lodLevel: -1)
        layer.cacheImage(try makeImage(), for: key, at: 10)

        layer.clearTile(at: key)

        #expect(layer.cachedImage(for: key) == nil)
        #expect(layer.tileFadeStartTimes[key] == nil)
        #expect(layer.tileOpacity(for: key, at: 10) == 1)
    }

    private func makeImage() throws -> CGImage {
        let pixelData = Data([255, 0, 255, 255])
        let provider = CGDataProvider(data: pixelData)
        return try #require(CGImage(
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 4,
            space: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ))
    }
}
