import Foundation

/// Describes why a tiled layer could not complete its rendering pipeline.
@_spi(RendererDiagnostics)
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
}

/// Validated, renderer-independent tiled-layer input.
internal struct CATiledLayerRenderConfiguration {
    let levelsOfDetail: Int
    let levelsOfDetailBias: Int
    let tileSize: CGSize
    let contentsScale: CGFloat
    let bounds: CGRect

    init(layer: CATiledLayer) throws(CATiledLayerRenderFailure) {
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

        levelsOfDetail = layer.levelsOfDetail
        levelsOfDetailBias = layer.levelsOfDetailBias
        tileSize = layer.tileSize
        contentsScale = layer.contentsScale
        bounds = layer.bounds
    }
}
