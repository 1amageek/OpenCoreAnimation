import Foundation

/// An immutable, value-owned view of the presentation state required by a
/// renderer for one frame.
///
/// The snapshot intentionally stores layer identity without retaining a
/// `CALayer`. This prevents mutations made after capture from changing the
/// frame that is already being encoded.
// FIXME(INCOMPLETE_IMPLEMENTATION): The immutable snapshot currently contains
// every value consumed by the production CAMetalRenderer path, while the WASM
// CAWebGPURenderer still renders directly from CALayer. The active native path
// captures this value in CAMetalRenderer.render(layer:). Phase 4 must not be
// considered complete until WebGPU resources, masks, specialized layer state,
// copied animation evaluators, and commit-time layout preparation are
// represented here, the explicit live-evaluation commit states are removed,
// and WebGPU no longer reads mutable model layers after capture.
internal struct CARenderSnapshot: Sendable {
    internal struct PresentationValues: Sendable, Equatable {
        internal let bounds: CGRect
        internal let boundsSize: SIMD2<Float>
        internal let boundsOrigin: SIMD2<Float>
        internal let position: SIMD3<Float>
        internal let anchorOffset: SIMD3<Float>
        internal let transform: CATransform3D
        internal let sublayerTransform: CATransform3D
        internal let opacity: Float
        internal let isHidden: Bool
        internal let cornerRadius: Float
        internal let backgroundColor: SIMD4<Float>?
    }

    internal struct Node: Sendable, Equatable {
        internal let identity: ObjectIdentifier
        internal let contentRevision: UInt64
        internal let presentationValues: PresentationValues
        internal let childIndices: [Int]
    }

    internal let nodes: [Node]
    internal let rootIndex: Int
    internal let frameToken: UInt64
    internal let rootBounds: CGRect
    internal let rootContentsScale: CGFloat
    internal let capturedContentRevisions: [ObjectIdentifier: UInt64]

    internal static func capture(
        _ rootLayer: CALayer,
        frameToken: UInt64
    ) throws(CARendererError) -> CARenderSnapshot {
        var nodes: [Node] = []
        var visited: Set<ObjectIdentifier> = []
        let rootIndex = try captureNode(
            rootLayer,
            nodes: &nodes,
            visited: &visited
        )
        var capturedContentRevisions: [ObjectIdentifier: UInt64] = [:]
        capturedContentRevisions.reserveCapacity(nodes.count)
        for node in nodes {
            capturedContentRevisions[node.identity] = node.contentRevision
        }
        return CARenderSnapshot(
            nodes: nodes,
            rootIndex: rootIndex,
            frameToken: frameToken,
            rootBounds: rootLayer.bounds,
            rootContentsScale: rootLayer.contentsScale,
            capturedContentRevisions: capturedContentRevisions
        )
    }

    private static func captureNode(
        _ layer: CALayer,
        nodes: inout [Node],
        visited: inout Set<ObjectIdentifier>
    ) throws(CARendererError) -> Int {
        let identity = ObjectIdentifier(layer)
        guard visited.insert(identity).inserted else {
            throw .cyclicLayerHierarchy
        }

        let contentRevision = layer._contentRevision
        let presentationLayer = layer._renderTimePresentation()
        let values = try presentationValues(from: presentationLayer)
        let nodeIndex = nodes.count
        nodes.append(
            Node(
                identity: identity,
                contentRevision: contentRevision,
                presentationValues: values,
                childIndices: []
            )
        )

        var childIndices: [Int] = []
        childIndices.reserveCapacity(layer.sublayers?.count ?? 0)
        for child in layer.sublayers ?? [] {
            childIndices.append(
                try captureNode(
                    child,
                    nodes: &nodes,
                    visited: &visited
                )
            )
        }

        nodes[nodeIndex] = Node(
            identity: identity,
            contentRevision: contentRevision,
            presentationValues: values,
            childIndices: childIndices
        )
        return nodeIndex
    }

    private static func presentationValues(
        from layer: CALayer
    ) throws(CARendererError) -> PresentationValues {
        guard layer.bounds.origin.x.isFinite,
              layer.bounds.origin.y.isFinite,
              layer.bounds.width.isFinite,
              layer.bounds.height.isFinite,
              layer.position.x.isFinite,
              layer.position.y.isFinite,
              layer.zPosition.isFinite,
              layer.anchorPoint.x.isFinite,
              layer.anchorPoint.y.isFinite,
              layer.anchorPointZ.isFinite,
              layer.opacity.isFinite,
              layer.cornerRadius.isFinite,
              isFinite(layer.transform),
              isFinite(layer.sublayerTransform) else {
            throw .nonFiniteLayerGeometry
        }
        let boundsWidth = Float(layer.bounds.width)
        let boundsHeight = Float(layer.bounds.height)
        let anchorX = Float(layer.anchorPoint.x)
        let anchorY = Float(layer.anchorPoint.y)
        let boundsOrigin = SIMD2<Float>(
            Float(layer.bounds.origin.x),
            Float(layer.bounds.origin.y)
        )
        let position = SIMD3<Float>(
            Float(layer.position.x),
            Float(layer.position.y),
            Float(layer.zPosition)
        )
        let anchorOffset = SIMD3<Float>(
            -(boundsWidth * anchorX),
            -(boundsHeight * anchorY),
            Float(-layer.anchorPointZ)
        )
        guard boundsWidth.isFinite,
              boundsHeight.isFinite,
              boundsOrigin.x.isFinite,
              boundsOrigin.y.isFinite,
              position.x.isFinite,
              position.y.isFinite,
              position.z.isFinite,
              anchorOffset.x.isFinite,
              anchorOffset.y.isFinite,
              anchorOffset.z.isFinite else {
            throw .nonFiniteLayerGeometry
        }
        return PresentationValues(
            bounds: layer.bounds,
            boundsSize: SIMD2<Float>(boundsWidth, boundsHeight),
            boundsOrigin: boundsOrigin,
            position: position,
            anchorOffset: anchorOffset,
            transform: layer.transform,
            sublayerTransform: layer.sublayerTransform,
            opacity: layer.opacity,
            isHidden: layer.isHidden,
            cornerRadius: Float(layer.cornerRadius),
            backgroundColor: try colorComponents(layer.backgroundColor)
        )
    }

    private static func colorComponents(
        _ color: CGColor?
    ) throws(CARendererError) -> SIMD4<Float>? {
        guard let color else { return nil }
        guard let converted = color.converted(
            to: .deviceRGB,
            intent: .defaultIntent,
            options: nil
        ),
        let components = converted.components,
        components.count == 4,
        components.allSatisfy(\.isFinite) else {
            throw .invalidLayerBackgroundColor
        }
        let result = SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
        guard result.x.isFinite,
              result.y.isFinite,
              result.z.isFinite,
              result.w.isFinite else {
            throw .invalidLayerBackgroundColor
        }
        return result
    }

    private static func isFinite(_ transform: CATransform3D) -> Bool {
        transform.m11.isFinite
            && transform.m12.isFinite
            && transform.m13.isFinite
            && transform.m14.isFinite
            && transform.m21.isFinite
            && transform.m22.isFinite
            && transform.m23.isFinite
            && transform.m24.isFinite
            && transform.m31.isFinite
            && transform.m32.isFinite
            && transform.m33.isFinite
            && transform.m34.isFinite
            && transform.m41.isFinite
            && transform.m42.isFinite
            && transform.m43.isFinite
            && transform.m44.isFinite
    }
}

/// The exact state published by the outermost transaction for one render root.
internal enum CACommittedRenderState: Sendable {
    case snapshot(CARenderSnapshot)
    case captureFailure(frameToken: UInt64, error: CARendererError)
    case requiresLiveAnimationEvaluation(frameToken: UInt64)
    case requiresLiveTreePreparation(frameToken: UInt64)

    internal var frameToken: UInt64 {
        switch self {
        case .snapshot(let snapshot):
            snapshot.frameToken
        case .captureFailure(let frameToken, _),
             .requiresLiveAnimationEvaluation(let frameToken),
             .requiresLiveTreePreparation(let frameToken):
            frameToken
        }
    }
}
