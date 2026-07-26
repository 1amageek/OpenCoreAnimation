import Foundation
import Synchronization

/// Describes why a tiled layer could not complete its rendering pipeline.
public enum CATiledLayerRenderFailure: Error, Equatable, Sendable {
    case invalidLevelsOfDetail(Int)
    case invalidLevelsOfDetailBias(Int)
    case invalidTileSize(CGSize)
    case invalidContentsScale(CGFloat)
    case invalidBounds(CGRect)
    case tileCountExceedsRendererCapacity(Int)
    case rendererResourcesUnavailable
    case drawingContextCreationFailed
    case imageCreationFailed
    case imageConversionFailed(CAImageContentsConversionError)
    case delegateRequiresSendableTileProvider
    case contentSnapshotCreationFailed(String)
    case contentDrawingFailed(String)
    case invalidFadeDuration(CFTimeInterval)
}

internal struct CATiledLayerCapturedContent: Sendable {
    private static let identityStorage = Mutex<UInt64>(0)

    let identity: UInt64
    let snapshot: any CATiledLayerContentSnapshot

    init(snapshot: any CATiledLayerContentSnapshot) {
        identity = Self.identityStorage.withLock { identity in
            identity &+= 1
            if identity == 0 {
                identity = 1
            }
            return identity
        }
        self.snapshot = snapshot
    }
}

/// Validated, renderer-independent tiled-layer input.
internal struct CATiledLayerRenderConfiguration:
    Equatable,
    Sendable {
    let resourceIdentity: UInt64
    let cacheGeneration: UInt64
    let levelsOfDetail: Int
    let levelsOfDetailBias: Int
    let tileSize: CGSize
    let contentsScale: CGFloat
    let bounds: CGRect
    let fadeDuration: CFTimeInterval
    let capturedContent: CATiledLayerCapturedContent?

    init(layer: CATiledLayer) throws(CATiledLayerRenderFailure) {
        try self.init(
            layer: layer,
            contentDelegate: layer.delegate
        )
    }

    init(
        layer: CATiledLayer,
        contentDelegate: (any CALayerDelegate)?
    ) throws(CATiledLayerRenderFailure) {
        guard layer.levelsOfDetail >= 1 else {
            throw .invalidLevelsOfDetail(layer.levelsOfDetail)
        }
        guard layer.levelsOfDetailBias >= 0 else {
            throw .invalidLevelsOfDetailBias(layer.levelsOfDetailBias)
        }
        guard layer.tileSize.width.isFinite,
              layer.tileSize.height.isFinite,
              layer.tileSize.width > 0,
              layer.tileSize.height > 0 else {
            throw .invalidTileSize(layer.tileSize)
        }
        guard layer.contentsScale.isFinite, layer.contentsScale > 0 else {
            throw .invalidContentsScale(layer.contentsScale)
        }
        guard layer.bounds.origin.x.isFinite,
              layer.bounds.origin.y.isFinite,
              layer.bounds.width.isFinite,
              layer.bounds.height.isFinite,
              layer.bounds.width >= 0,
              layer.bounds.height >= 0 else {
            throw .invalidBounds(layer.bounds)
        }
        let fadeDuration = type(of: layer).fadeDuration()
        guard fadeDuration.isFinite, fadeDuration >= 0 else {
            throw .invalidFadeDuration(fadeDuration)
        }
        let capturedContent: CATiledLayerCapturedContent?
        if let delegate = contentDelegate {
            guard let provider =
                    delegate as? any CATiledLayerContentProvider else {
                throw .delegateRequiresSendableTileProvider
            }
            do {
                capturedContent = CATiledLayerCapturedContent(
                    snapshot:
                        try provider.makeTileContentSnapshot()
                )
            } catch {
                throw .contentSnapshotCreationFailed(
                    String(describing: error)
                )
            }
        } else {
            capturedContent = nil
        }

        resourceIdentity = layer.resourceIdentity
        cacheGeneration = layer.tileCacheGeneration
        levelsOfDetail = layer.levelsOfDetail
        levelsOfDetailBias = layer.levelsOfDetailBias
        tileSize = layer.tileSize
        contentsScale = layer.contentsScale
        bounds = layer.bounds
        self.fadeDuration = fadeDuration
        self.capturedContent = capturedContent
    }

    static func == (
        lhs: Self,
        rhs: Self
    ) -> Bool {
        lhs.resourceIdentity == rhs.resourceIdentity
            && lhs.cacheGeneration == rhs.cacheGeneration
            && lhs.levelsOfDetail == rhs.levelsOfDetail
            && lhs.levelsOfDetailBias
                == rhs.levelsOfDetailBias
            && lhs.tileSize == rhs.tileSize
            && lhs.contentsScale == rhs.contentsScale
            && lhs.bounds == rhs.bounds
            && lhs.fadeDuration == rhs.fadeDuration
            && lhs.capturedContent?.identity
                == rhs.capturedContent?.identity
    }

    func lodLevel(forScreenScale screenScale: CGFloat) -> Int {
        let safeScale =
            screenScale.isFinite && screenScale > 0
                ? screenScale
                : 1
        let requestedLevel = Int(floor(-log2(safeScale)))
        let minimumLevel = -max(0, levelsOfDetailBias)

        let pixelWidth =
            max(0, bounds.width * max(contentsScale, 0))
        let pixelHeight =
            max(0, bounds.height * max(contentsScale, 0))
        let minimumPixelDimension = min(
            pixelWidth,
            pixelHeight
        )
        let requestedMaximum = max(0, levelsOfDetail - 1)
        let dimensionLimit: Int
        if minimumPixelDimension.isFinite,
           minimumPixelDimension >= 1 {
            dimensionLimit = max(
                0,
                Int(floor(log2(minimumPixelDimension)))
            )
        } else if minimumPixelDimension == .infinity {
            dimensionLimit = requestedMaximum
        } else {
            dimensionLimit = 0
        }
        let maximumLevel = min(
            requestedMaximum,
            dimensionLimit
        )
        return min(
            max(requestedLevel, minimumLevel),
            maximumLevel
        )
    }
}
