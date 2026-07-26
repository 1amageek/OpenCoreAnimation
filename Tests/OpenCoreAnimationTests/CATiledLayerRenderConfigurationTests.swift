import Foundation
import Synchronization
import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

private struct CapturedTileContent:
    CATiledLayerContentSnapshot {
    let value: Int

    func drawTile(
        _ tile: CATiledLayerTileDrawingInfo,
        in context: CGContext
    ) {}
}

private final class SendableTileProvider:
    CATiledLayerContentProvider {
    private let value = Mutex(1)

    func makeTileContentSnapshot()
        -> any CATiledLayerContentSnapshot {
        CapturedTileContent(
            value: value.withLock { $0 }
        )
    }

    func setValue(_ newValue: Int) {
        value.withLock { value in
            value = newValue
        }
    }
}

private enum TileProviderTestError: Error {
    case unavailable
}

private final class FailingTileProvider:
    CATiledLayerContentProvider {
    func makeTileContentSnapshot()
        throws -> any CATiledLayerContentSnapshot {
        throw TileProviderTestError.unavailable
    }
}

private final class NonSendableTileDelegate:
    CALayerDelegate {
    var drawCount = 0
}

private final class InvalidFadeTiledLayer: CATiledLayer {
    override class func fadeDuration() -> CFTimeInterval {
        .nan
    }
}

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
        let provider = SendableTileProvider()
        layer.delegate = provider
        let expectedGeneration = layer.tileCacheGeneration

        let configuration = try CATiledLayerRenderConfiguration(layer: layer)

        #expect(
            configuration.resourceIdentity
                == layer.resourceIdentity
        )
        #expect(
            configuration.cacheGeneration
                == expectedGeneration
        )
        #expect(configuration.bounds == layer.bounds)
        #expect(configuration.contentsScale == 2)
        #expect(configuration.tileSize == CGSize(width: 128, height: 64))
        #expect(configuration.levelsOfDetail == 4)
        #expect(configuration.levelsOfDetailBias == 2)
        #expect(configuration.fadeDuration == 0.25)
        let capturedContent = try #require(
            configuration.capturedContent
        )
        let snapshot = try #require(
            capturedContent.snapshot as? CapturedTileContent
        )
        #expect(snapshot.value == 1)
        provider.setValue(2)
        #expect(snapshot.value == 1)
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

    @Test("Committed and live LOD selection stay equivalent")
    func committedLODMatchesLayer() throws {
        let layer = CATiledLayer()
        layer.bounds = CGRect(
            x: 0,
            y: 0,
            width: 256,
            height: 128
        )
        layer.contentsScale = 2
        layer.levelsOfDetail = 6
        layer.levelsOfDetailBias = 3
        let configuration =
            try CATiledLayerRenderConfiguration(
                layer: layer
            )

        for scale in [
            CGFloat.nan,
            -1,
            0,
            0.01,
            0.25,
            0.5,
            1,
            2,
            8,
        ] {
            #expect(
                configuration.lodLevel(
                    forScreenScale: scale
                )
                    == layer.lodLevel(
                        forScreenScale: scale
                    )
            )
        }
    }

    @Test("Invalid fade timing fails immutable capture")
    func invalidFadeDurationFails() {
        let layer = InvalidFadeTiledLayer()
        do {
            _ = try CATiledLayerRenderConfiguration(
                layer: layer
            )
            Issue.record(
                "Expected an invalid fade duration failure"
            )
        } catch {
            guard case .invalidFadeDuration(let value) =
                    error else {
                Issue.record(
                    "Unexpected tile failure: \(error)"
                )
                return
            }
            #expect(value.isNaN)
        }
    }

    @Test("A non-Sendable tile delegate fails immutable capture")
    func nonSendableDelegateFails() {
        let layer = CATiledLayer()
        let delegate = NonSendableTileDelegate()
        layer.delegate = delegate

        #expect(throws: CATiledLayerRenderFailure
            .delegateRequiresSendableTileProvider) {
            try CATiledLayerRenderConfiguration(layer: layer)
        }
    }

    @Test("A provider capture failure is not treated as empty content")
    func providerCaptureFailureIsTyped() {
        let layer = CATiledLayer()
        let provider = FailingTileProvider()
        layer.delegate = provider

        #expect(throws: CATiledLayerRenderFailure
            .contentSnapshotCreationFailed("unavailable")) {
            try CATiledLayerRenderConfiguration(layer: layer)
        }
    }
}
