import Foundation

/// Describes why a replicator layer could not complete its rendering pipeline.
@_spi(RendererDiagnostics)
public enum CAReplicatorRenderFailure: Error, Equatable, Sendable {
    case instanceCountExceedsRendererCapacity(actual: Int, maximum: Int)
    case nonFiniteInstanceDelay
    case nonFiniteInstanceTransform
    case invalidInstanceColor
    case nonFiniteInstanceColorOffset
    case instanceTimeOffsetOverflow(instanceIndex: Int)
    case instanceColorOverflow(instanceIndex: Int)
    case cumulativeTransformOverflow(instanceIndex: Int)
    case depthResourcesUnavailable
    case invalidDepthNesting(Int)
    case depthNestingOverflow
    case invalidProjectedDepth(
        instanceIndex: Int,
        sublayerIndex: Int,
        reason: CAProjectedDepthError
    )

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

/// Validated, renderer-independent replicator input.
internal struct CAReplicatorRenderConfiguration {
    let instanceCount: Int
    let preservesDepth: Bool
    let instanceDelay: CFTimeInterval
    let instanceTransform: CATransform3D
    let baseColor: SIMD4<Float>
    let colorOffset: SIMD4<Float>

    init(
        layer: CAReplicatorLayer,
        maximumInstanceCount: Int
    ) throws(CAReplicatorRenderFailure) {
        let normalizedInstanceCount = max(0, layer.instanceCount)
        guard normalizedInstanceCount <= maximumInstanceCount else {
            throw .instanceCountExceedsRendererCapacity(
                actual: normalizedInstanceCount,
                maximum: maximumInstanceCount
            )
        }
        guard layer.instanceDelay.isFinite else {
            throw .nonFiniteInstanceDelay
        }
        guard Self.isFinite(layer.instanceTransform) else {
            throw .nonFiniteInstanceTransform
        }
        guard layer.instanceRedOffset.isFinite,
              layer.instanceGreenOffset.isFinite,
              layer.instanceBlueOffset.isFinite,
              layer.instanceAlphaOffset.isFinite else {
            throw .nonFiniteInstanceColorOffset
        }

        let resolvedBaseColor: SIMD4<Float>
        if let instanceColor = layer.instanceColor {
            guard let converted = instanceColor.converted(
                to: .deviceRGB,
                intent: .defaultIntent,
                options: nil
            ), let components = converted.components,
               components.count == 4,
               components.allSatisfy(\.isFinite) else {
                throw .invalidInstanceColor
            }
            resolvedBaseColor = SIMD4(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
            guard resolvedBaseColor.x.isFinite,
                  resolvedBaseColor.y.isFinite,
                  resolvedBaseColor.z.isFinite,
                  resolvedBaseColor.w.isFinite else {
                throw .invalidInstanceColor
            }
        } else {
            resolvedBaseColor = SIMD4(repeating: 1)
        }

        instanceCount = normalizedInstanceCount
        preservesDepth = layer.preservesDepth
        instanceDelay = layer.instanceDelay
        instanceTransform = layer.instanceTransform
        baseColor = resolvedBaseColor
        colorOffset = SIMD4(
            layer.instanceRedOffset,
            layer.instanceGreenOffset,
            layer.instanceBlueOffset,
            layer.instanceAlphaOffset
        )
    }

    func color(at instanceIndex: Int) throws(CAReplicatorRenderFailure) -> SIMD4<Float> {
        let offsetMultiplier = Float(instanceIndex)
        let color = baseColor + colorOffset * offsetMultiplier
        guard color.x.isFinite,
              color.y.isFinite,
              color.z.isFinite,
              color.w.isFinite else {
            throw .instanceColorOverflow(instanceIndex: instanceIndex)
        }
        return SIMD4(
            min(max(color.x, 0), 1),
            min(max(color.y, 0), 1),
            min(max(color.z, 0), 1),
            min(max(color.w, 0), 1)
        )
    }

    func timeOffset(at instanceIndex: Int) throws(CAReplicatorRenderFailure) -> CFTimeInterval {
        let offset = CFTimeInterval(instanceIndex) * instanceDelay
        guard offset.isFinite else {
            throw .instanceTimeOffsetOverflow(instanceIndex: instanceIndex)
        }
        return offset
    }

    func nextTransform(
        after transform: CATransform3D,
        nextInstanceIndex: Int
    ) throws(CAReplicatorRenderFailure) -> CATransform3D {
        let result = CATransform3DConcat(transform, instanceTransform)
        guard Self.isFinite(result) else {
            throw .cumulativeTransformOverflow(instanceIndex: nextInstanceIndex)
        }
        return result
    }

    static func isFinite(_ transform: CATransform3D) -> Bool {
        transform.m11.isFinite && transform.m12.isFinite
            && transform.m13.isFinite && transform.m14.isFinite
            && transform.m21.isFinite && transform.m22.isFinite
            && transform.m23.isFinite && transform.m24.isFinite
            && transform.m31.isFinite && transform.m32.isFinite
            && transform.m33.isFinite && transform.m34.isFinite
            && transform.m41.isFinite && transform.m42.isFinite
            && transform.m43.isFinite && transform.m44.isFinite
    }
}
