import Foundation

/// Describes why an emitter could not simulate or render its particles.
@_spi(RendererDiagnostics)
public enum CAEmitterFailure: Error, Equatable, Sendable {
    case unsupportedEmitterShape(String)
    case unsupportedEmitterMode(String)
    case unsupportedRenderMode(String)
    case nonFiniteLayerGeometry
    case nonFiniteLayerSimulationValue
    case invalidCellTiming
    case invalidCellBirthRate
    case invalidCellContents
    case invalidCellColor
    case nonFiniteEmissionDirection
    case invalidChildDirection
    case nonFiniteParticleState
    case particleCapacityExceeded(maximum: Int)
    case flatteningCaptureUnavailable
    case depthResourcesUnavailable
    case invalidDepthNesting(Int)
    case depthNestingOverflow
    case rendererResourcesUnavailable
    case additivePipelineUnavailable
    case imageConversionFailed(CAImageContentsConversionError)
    case textureResourcesUnavailable
    case vertexCapacityExceeded

    internal static func depthGroupStateFailure(
        _ failure: CADepthGroupStateFailure
    ) -> Self {
        switch failure {
        case .invalidNestingDepth(let depth):
            return .invalidDepthNesting(depth)
        case .nestingDepthOverflow:
            return .depthNestingOverflow
        }
    }
}

/// Validated, renderer-independent emitter-layer input.
internal struct CAEmitterRenderConfiguration {
    let emitterCells: [CAEmitterCell]
    let emitterPosition: CGPoint
    let emitterZPosition: CGFloat
    let emitterSize: CGSize
    let emitterDepth: CGFloat
    let emitterShape: CAEmitterLayerEmitterShape
    let emitterMode: CAEmitterLayerEmitterMode
    let renderMode: CAEmitterLayerRenderMode
    let preservesDepth: Bool
    let birthRate: Float
    let lifetime: Float
    let velocity: Float
    let scale: Float
    let spin: Float
    let seed: UInt32

    init(layer: CAEmitterLayer) throws(CAEmitterFailure) {
        switch layer.emitterShape {
        case .point, .line, .rectangle, .cuboid, .circle, .sphere:
            break
        default:
            throw .unsupportedEmitterShape(layer.emitterShape.rawValue)
        }
        switch layer.emitterMode {
        case .points, .outline, .surface, .volume:
            break
        default:
            throw .unsupportedEmitterMode(layer.emitterMode.rawValue)
        }
        switch layer.renderMode {
        case .unordered, .oldestFirst, .oldestLast, .backToFront, .additive:
            break
        default:
            throw .unsupportedRenderMode(layer.renderMode.rawValue)
        }
        guard layer.emitterPosition.x.isFinite,
              layer.emitterPosition.y.isFinite,
              layer.emitterZPosition.isFinite,
              layer.emitterSize.width.isFinite,
              layer.emitterSize.height.isFinite,
              layer.emitterDepth.isFinite else {
            throw .nonFiniteLayerGeometry
        }
        guard layer.birthRate.isFinite,
              layer.lifetime.isFinite,
              layer.velocity.isFinite,
              layer.scale.isFinite,
              layer.spin.isFinite else {
            throw .nonFiniteLayerSimulationValue
        }

        emitterCells = layer.emitterCells ?? []
        emitterPosition = layer.emitterPosition
        emitterZPosition = layer.emitterZPosition
        emitterSize = layer.emitterSize
        emitterDepth = layer.emitterDepth
        emitterShape = layer.emitterShape
        emitterMode = layer.emitterMode
        renderMode = layer.renderMode
        preservesDepth = layer.preservesDepth
        birthRate = layer.birthRate
        lifetime = layer.lifetime
        velocity = layer.velocity
        scale = layer.scale
        spin = layer.spin
        seed = layer.seed
    }
}
