import Foundation

/// An immutable, value-owned view of the presentation state required by a
/// renderer for one frame.
///
/// The snapshot intentionally stores layer identity without retaining a
/// `CALayer`. This prevents mutations made after capture from changing the
/// frame that is already being encoded.
// FIXME(INCOMPLETE_IMPLEMENTATION): The immutable snapshot contains every value
// consumed by CAMetalRenderer and by CAWebGPURenderer's static snapshot path,
// including nested rectangular and rounded clipping, ordinary CGImage
// contents, layer filter and backdrop-composition plans, gradient inputs,
// tessellated shape geometry, validated text configuration, and emitter cells
// with their converted image bytes.
// Production WebGPU still uses an explicitly typed live-tree branch for
// animation evaluation.
// Phase 4 must not be considered complete until those
// values and resources are owned here, the animation live-tree commit state is
// removed,
// and every WebGPU frame encodes without reading mutable model layers after
// capture.
internal struct CARenderSnapshot: Sendable {
    internal struct PresentationValues: Sendable, Equatable {
        internal struct Shadow: Sendable, Equatable {
            internal let color: SIMD4<Float>
            internal let opacity: Float
            internal let radius: Float
            internal let offset: SIMD2<Float>
            internal let pathVertices: [SIMD2<Float>]?
        }

        internal struct Shape: Sendable, Equatable {
            internal struct Primitive: Sendable, Equatable {
                internal let vertices: [SIMD2<Float>]
                internal let color: SIMD4<Float>
            }

            internal let fill: Primitive?
            internal let stroke: Primitive?
        }

        internal struct Text: Sendable, Equatable {
            internal let configuration: CATextRenderConfiguration?
        }

        internal let replicator:
            CAReplicatorRenderConfiguration?
        internal let emitter:
            CAEmitterRenderConfiguration?
        internal let tiled:
            CATiledLayerRenderConfiguration?
        internal let transition:
            CARenderSnapshotTransition?
        internal private(set) var replicatorInstanceTransform:
            CATransform3D
        internal private(set) var effectiveReplicatorColor:
            SIMD4<Float>
        internal private(set) var effectiveReplicatorTimeOffset:
            CFTimeInterval
        internal let bounds: CGRect
        internal let contentsScale: CGFloat
        internal let boundsSize: SIMD2<Float>
        internal let boundsOrigin: SIMD2<Float>
        internal let position: SIMD3<Float>
        internal let anchorOffset: SIMD3<Float>
        internal let transform: CATransform3D
        internal let sublayerTransform: CATransform3D
        internal let isTransformLayer: Bool
        internal let isGeometryFlipped: Bool
        internal let isDoubleSided: Bool
        internal let isOpaque: Bool
        internal let masksToBounds: Bool
        internal let allowsGroupOpacity: Bool
        internal let shouldRasterize: Bool
        internal let rasterizationScale: CGFloat
        internal let opacity: Float
        internal let isHidden: Bool
        internal let cornerRadius: Float
        internal let cornerCurveExponent: Float
        internal let cornerRadii: SIMD4<Float>
        internal let edgeAntialiasingMask: Float
        internal let backgroundColor: SIMD4<Float>?
        internal let borderWidth: Float
        internal let borderColor: SIMD4<Float>?
        internal let toneMapMode: CALayer.ToneMapMode
        internal let preferredDynamicRange: CALayer.DynamicRange
        internal let contentsHeadroom: Float
        internal let imageContents: CAImageContentsSnapshot?
        internal let filters: [CARenderSnapshotFilterStage]
        internal let compositingFilter:
            CARenderSnapshotCompositingFilter?
        internal let backgroundFilters: [CARenderSnapshotFilterStage]
        internal let gradient: GradientRenderConfiguration?
        internal let shape: Shape?
        internal let text: Text?
        internal let shadow: Shadow?

        internal func applyingReplicatorInstance(
            transform: CATransform3D?,
            color: SIMD4<Float>,
            timeOffset: CFTimeInterval,
            instanceIndex: Int
        ) throws(CAReplicatorRenderFailure) -> Self {
            var result = self
            if let transform {
                let combinedTransform = CATransform3DConcat(
                    transform,
                    result.replicatorInstanceTransform
                )
                guard CAReplicatorRenderConfiguration.isFinite(
                    combinedTransform
                ) else {
                    throw .cumulativeTransformOverflow(
                        instanceIndex: instanceIndex
                    )
                }
                result.replicatorInstanceTransform =
                    combinedTransform
            }
            result.effectiveReplicatorColor *= color
            let combinedTimeOffset =
                result.effectiveReplicatorTimeOffset + timeOffset
            guard combinedTimeOffset.isFinite else {
                throw .instanceTimeOffsetOverflow(
                    instanceIndex: instanceIndex
                )
            }
            result.effectiveReplicatorTimeOffset =
                combinedTimeOffset
            return result
        }
    }

    internal struct Node: Sendable, Equatable {
        internal let identity: ObjectIdentifier
        internal let contentRevision: UInt64
        internal let presentationValues: PresentationValues
        internal let childIndices: [Int]
        internal let maskIndex: Int?
        internal let replicatorSourceChildCount: Int?
    }

    internal let nodes: [Node]
    internal let rootIndex: Int
    internal let frameToken: UInt64
    internal let rootBounds: CGRect
    internal let rootContentsScale: CGFloat
    internal let capturedContentRevisions: [ObjectIdentifier: UInt64]

    internal func rooted(at nodeIndex: Int) -> Self {
        let values = nodes[nodeIndex].presentationValues
        return Self(
            nodes: nodes,
            rootIndex: nodeIndex,
            frameToken: frameToken,
            rootBounds: values.bounds,
            rootContentsScale: values.contentsScale,
            capturedContentRevisions:
                capturedContentRevisions
        )
    }

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

        let isDepthContainer =
            layer is CATransformLayer
            || (layer as? CAReplicatorLayer)?.preservesDepth == true
        if !isDepthContainer {
            do {
                try layer.prepareDelegateBackingStore(
                    maximumPixelDimension: Int.max
                )
            } catch {
                throw .invalidDelegateBackingStore(error)
            }
        }
        let contentRevision = layer._contentRevision
        let presentationLayer = layer._renderTimePresentation()
        let transition: CARenderSnapshotTransition?
        if let transitionState =
                presentationLayer._transitionRenderState {
            let sourceRootIndex = try captureNode(
                transitionState.sourceLayer,
                nodes: &nodes,
                visited: &visited
            )
            do {
                transition = try CARenderSnapshotTransition.capture(
                    transitionState,
                    sourceRootIndex: sourceRootIndex
                )
            } catch {
                throw .invalidLayerTransition(error)
            }
        } else {
            transition = nil
        }
        let values = try presentationValues(
            from: presentationLayer,
            delegateBackingStore: layer.delegateBackingStore,
            tiledContentDelegate:
                (layer as? CATiledLayer)?.delegate,
            transition: transition
        )
        let nodeIndex = nodes.count
        nodes.append(
            Node(
                identity: identity,
                contentRevision: contentRevision,
                presentationValues: values,
                childIndices: [],
                maskIndex: nil,
                replicatorSourceChildCount: nil
            )
        )

        var childIndices: [Int] = []
        let orderedChildren = isDepthContainer
            ? layer.sublayers ?? []
            : layer.sortedSublayers()
        childIndices.reserveCapacity(orderedChildren.count)
        for child in orderedChildren {
            childIndices.append(
                try captureNode(
                    child,
                    nodes: &nodes,
                    visited: &visited
                )
            )
        }
        if let replicator = values.replicator {
            do {
                childIndices = try expandedReplicatorChildren(
                    sourceChildIndices: childIndices,
                    configuration: replicator,
                    nodes: &nodes
                )
            } catch {
                throw .invalidLayerReplicator(
                    snapshotReplicatorError(from: error)
                )
            }
        }
        let maskIndex: Int?
        if !isDepthContainer, let mask = layer.mask {
            maskIndex = try captureNode(
                mask,
                nodes: &nodes,
                visited: &visited
            )
        } else {
            maskIndex = nil
        }

        nodes[nodeIndex] = Node(
            identity: identity,
            contentRevision: contentRevision,
            presentationValues: values,
            childIndices: childIndices,
            maskIndex: maskIndex,
            replicatorSourceChildCount:
                values.replicator == nil
                    ? nil
                    : orderedChildren.count
        )
        return nodeIndex
    }

    private static func expandedReplicatorChildren(
        sourceChildIndices: [Int],
        configuration: CAReplicatorRenderConfiguration,
        nodes: inout [Node]
    ) throws(CAReplicatorRenderFailure) -> [Int] {
        guard configuration.instanceCount > 0,
              !sourceChildIndices.isEmpty else {
            return []
        }

        var instanceChildIndices: [[Int]] = []
        instanceChildIndices.reserveCapacity(
            configuration.instanceCount
        )
        instanceChildIndices.append(sourceChildIndices)
        if configuration.instanceCount > 1 {
            for _ in 1..<configuration.instanceCount {
                instanceChildIndices.append(
                    sourceChildIndices.map {
                        cloneSubtree(at: $0, nodes: &nodes)
                    }
                )
            }
        }

        var cumulativeTransform = CATransform3DIdentity
        var expandedChildIndices: [Int] = []
        for instanceIndex in 0..<configuration.instanceCount {
            let color = try configuration.color(
                at: instanceIndex
            )
            let timeOffset = try configuration.timeOffset(
                at: instanceIndex
            )
            for childIndex in instanceChildIndices[instanceIndex] {
                try applyReplicatorInstance(
                    to: childIndex,
                    rootTransform: cumulativeTransform,
                    color: color,
                    timeOffset: timeOffset,
                    instanceIndex: instanceIndex,
                    nodes: &nodes
                )
                expandedChildIndices.append(childIndex)
            }
            if instanceIndex + 1 < configuration.instanceCount {
                cumulativeTransform =
                    try configuration.nextTransform(
                        after: cumulativeTransform,
                        nextInstanceIndex: instanceIndex + 1
                    )
            }
        }
        return expandedChildIndices
    }

    private static func cloneSubtree(
        at sourceIndex: Int,
        nodes: inout [Node]
    ) -> Int {
        let source = nodes[sourceIndex]
        let cloneIndex = nodes.count
        nodes.append(
            Node(
                identity: source.identity,
                contentRevision: source.contentRevision,
                presentationValues: source.presentationValues,
                childIndices: [],
                maskIndex: nil,
                replicatorSourceChildCount:
                    source.replicatorSourceChildCount
            )
        )
        let childIndices = source.childIndices.map {
            cloneSubtree(at: $0, nodes: &nodes)
        }
        let maskIndex = source.maskIndex.map {
            cloneSubtree(at: $0, nodes: &nodes)
        }
        nodes[cloneIndex] = Node(
            identity: source.identity,
            contentRevision: source.contentRevision,
            presentationValues: source.presentationValues,
            childIndices: childIndices,
            maskIndex: maskIndex,
            replicatorSourceChildCount:
                source.replicatorSourceChildCount
        )
        return cloneIndex
    }

    private static func applyReplicatorInstance(
        to nodeIndex: Int,
        rootTransform: CATransform3D,
        color: SIMD4<Float>,
        timeOffset: CFTimeInterval,
        instanceIndex: Int,
        nodes: inout [Node]
    ) throws(CAReplicatorRenderFailure) {
        let source = nodes[nodeIndex]
        let values = try source.presentationValues
            .applyingReplicatorInstance(
                transform: rootTransform,
                color: color,
                timeOffset: timeOffset,
                instanceIndex: instanceIndex
            )
        nodes[nodeIndex] = Node(
            identity: source.identity,
            contentRevision: source.contentRevision,
            presentationValues: values,
            childIndices: source.childIndices,
            maskIndex: source.maskIndex,
            replicatorSourceChildCount:
                source.replicatorSourceChildCount
        )
        for childIndex in source.childIndices {
            try applyReplicatorInstance(
                to: childIndex,
                rootTransform: CATransform3DIdentity,
                color: color,
                timeOffset: timeOffset,
                instanceIndex: instanceIndex,
                nodes: &nodes
            )
        }
        if let maskIndex = source.maskIndex {
            try applyReplicatorInstance(
                to: maskIndex,
                rootTransform: CATransform3DIdentity,
                color: color,
                timeOffset: timeOffset,
                instanceIndex: instanceIndex,
                nodes: &nodes
            )
        }
    }

    private static func snapshotReplicatorError(
        from error: CAReplicatorRenderFailure
    ) -> CARenderSnapshotReplicatorError {
        switch error {
        case .instanceCountExceedsRendererCapacity(
            let actual,
            let maximum
        ):
            return .instanceCountExceedsRendererCapacity(
                actual: actual,
                maximum: maximum
            )
        case .nonFiniteInstanceDelay:
            return .nonFiniteInstanceDelay
        case .nonFiniteInstanceTransform:
            return .nonFiniteInstanceTransform
        case .invalidInstanceColor:
            return .invalidInstanceColor
        case .nonFiniteInstanceColorOffset:
            return .nonFiniteInstanceColorOffset
        case .instanceTimeOffsetOverflow(let instanceIndex):
            return .instanceTimeOffsetOverflow(
                instanceIndex: instanceIndex
            )
        case .instanceColorOverflow(let instanceIndex):
            return .instanceColorOverflow(
                instanceIndex: instanceIndex
            )
        case .cumulativeTransformOverflow(let instanceIndex):
            return .cumulativeTransformOverflow(
                instanceIndex: instanceIndex
            )
        case .depthResourcesUnavailable:
            return .depthResourcesUnavailable
        case .invalidDepthNesting(let depth):
            return .invalidDepthNesting(depth)
        case .depthNestingOverflow:
            return .depthNestingOverflow
        case .invalidProjectedDepth(
            let instanceIndex,
            let sublayerIndex,
            let reason
        ):
            return .invalidProjectedDepth(
                instanceIndex: instanceIndex,
                sublayerIndex: sublayerIndex,
                reason: snapshotProjectedDepthError(
                    from: reason
                )
            )
        }
    }

    private static func snapshotProjectedDepthError(
        from error: CAProjectedDepthError
    ) -> CARenderSnapshotProjectedDepthError {
        switch error {
        case .nonFiniteHomogeneousCoordinate(let z, let w):
            return .nonFiniteHomogeneousCoordinate(z: z, w: w)
        case .zeroHomogeneousCoordinate:
            return .zeroHomogeneousCoordinate
        case .nonFiniteNormalizedDepth:
            return .nonFiniteNormalizedDepth
        }
    }

    private static func presentationValues(
        from layer: CALayer,
        delegateBackingStore: CADelegateBackingStore? = nil,
        tiledContentDelegate:
            (any CALayerDelegate)? = nil,
        transition: CARenderSnapshotTransition? = nil
    ) throws(CARendererError) -> PresentationValues {
        let isTransformLayer = layer is CATransformLayer
        let replicator: CAReplicatorRenderConfiguration?
        if let replicatorLayer = layer as? CAReplicatorLayer {
            do {
                replicator = try CAReplicatorRenderConfiguration(
                    layer: replicatorLayer,
                    maximumInstanceCount:
                        CAReplicatorRenderConfiguration
                            .maximumInstanceCount
                )
            } catch {
                throw .invalidLayerReplicator(
                    snapshotReplicatorError(from: error)
                )
            }
        } else {
            replicator = nil
        }
        let emitter: CAEmitterRenderConfiguration?
        if let emitterLayer = layer as? CAEmitterLayer {
            do {
                emitter = try CAEmitterRenderConfiguration(
                    layer: emitterLayer
                )
            } catch {
                throw .invalidLayerEmitter(error)
            }
        } else {
            emitter = nil
        }
        let tiled: CATiledLayerRenderConfiguration?
        if let tiledLayer = layer as? CATiledLayer {
            do {
                tiled = try CATiledLayerRenderConfiguration(
                    layer: tiledLayer,
                    contentDelegate: tiledContentDelegate
                )
            } catch {
                throw .invalidLayerTiled(error)
            }
        } else {
            tiled = nil
        }
        let isDepthContainer =
            isTransformLayer || replicator?.preservesDepth == true
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
              isFinite(layer.transform),
              isFinite(layer.sublayerTransform) else {
            throw .nonFiniteLayerGeometry
        }
        if !isDepthContainer {
            guard layer.cornerRadius.isFinite,
                  layer.borderWidth.isFinite,
                  layer.contentsHeadroom.isFinite else {
                throw .nonFiniteLayerGeometry
            }
            guard layer.cornerRadius >= 0 else {
                throw .invalidLayerCornerGeometry
            }
        }
        if !isDepthContainer, layer.shouldRasterize {
            guard layer.rasterizationScale.isFinite,
                  layer.rasterizationScale > 0 else {
                throw .invalidLayerRasterization(
                    .invalidRasterizationScale(
                        layer.rasterizationScale
                    )
                )
            }
        }
        let cornerCurveExponent: Float
        if isDepthContainer {
            cornerCurveExponent = Float(
                CornerCurveRenderConfiguration.circularExponent
            )
        } else {
            do {
                cornerCurveExponent = Float(
                    try CornerCurveRenderConfiguration(
                        curve: layer.cornerCurve
                    ).exponent
                )
            } catch {
                throw .invalidLayerCornerGeometry
            }
        }
        let cornerRadii = isDepthContainer
            ? SIMD4<Float>.zero
            : cornerRadiiComponents(from: layer)
        guard cornerCurveExponent.isFinite,
              cornerCurveExponent > 0,
              cornerRadii.x.isFinite,
              cornerRadii.y.isFinite,
              cornerRadii.z.isFinite,
              cornerRadii.w.isFinite else {
            throw .invalidLayerCornerGeometry
        }
        let borderWidth = isDepthContainer
            ? 0
            : Float(layer.borderWidth)
        guard borderWidth.isFinite, borderWidth >= 0 else {
            throw .invalidLayerBorderWidth
        }
        let contentsHeadroom = isDepthContainer
            ? 0
            : Float(layer.contentsHeadroom)
        guard contentsHeadroom.isFinite else {
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
        let imageContents: CAImageContentsSnapshot?
        if isDepthContainer
            || layer is CAShapeLayer
            || layer is CATextLayer {
            imageContents = nil
        } else {
            do {
                imageContents = try captureImageContents(
                    from: layer,
                    delegateBackingStore: delegateBackingStore
                )
            } catch {
                throw .invalidLayerContents(error)
            }
        }
        let filters: [CARenderSnapshotFilterStage]
        let compositingFilter: CARenderSnapshotCompositingFilter?
        let backgroundFilters: [CARenderSnapshotFilterStage]
        if isDepthContainer {
            filters = []
            compositingFilter = nil
            backgroundFilters = []
        } else {
            do {
                filters = try CARenderSnapshotFilterStage.capture(
                    layer.filters ?? []
                )
            } catch {
                throw .invalidLayerFilter(error)
            }
            do {
                compositingFilter =
                    try CARenderSnapshotCompositingFilter.capture(
                        layer.compositingFilter
                    )
            } catch {
                throw .invalidLayerCompositingFilter(error)
            }
            do {
                backgroundFilters =
                    try CARenderSnapshotFilterStage.capture(
                        layer.backgroundFilters ?? []
                    )
            } catch {
                throw .invalidLayerBackgroundFilter(error)
            }
        }
        let shadow: PresentationValues.Shadow?
        if !isDepthContainer,
           layer.shadowOpacity > 0,
           layer.shadowColor != nil {
            let configuration: CAShadowRenderConfiguration
            do {
                configuration = try CAShadowRenderConfiguration(layer: layer)
            } catch {
                switch error {
                case .invalidColor:
                    throw .invalidLayerShadow(.invalidColor)
                default:
                    throw .invalidLayerShadow(.nonFiniteGeometry)
                }
            }
            let pathVertices: [SIMD2<Float>]?
            if let shadowPath = layer.shadowPath {
                do {
                    let vertices = try ShapeFillTessellator.triangles(
                        for: shadowPath,
                        rule: .nonZero
                    ).map {
                        SIMD2(Float($0.x), Float($0.y))
                    }
                    guard vertices.allSatisfy({
                        $0.x.isFinite && $0.y.isFinite
                    }) else {
                        throw CARenderSnapshotShadowError
                            .nonFiniteGeometry
                    }
                    pathVertices = vertices
                } catch let error as CARenderSnapshotShadowError {
                    throw .invalidLayerShadow(error)
                } catch {
                    throw .invalidLayerShadow(
                        .pathTessellationFailed
                    )
                }
            } else {
                pathVertices = nil
            }
            shadow = PresentationValues.Shadow(
                color: configuration.color,
                opacity: configuration.opacity,
                radius: configuration.radius,
                offset: SIMD2(
                    Float(configuration.offset.width),
                    Float(configuration.offset.height)
                ),
                pathVertices: pathVertices
            )
        } else {
            shadow = nil
        }
        let gradient: GradientRenderConfiguration?
        if let gradientLayer = layer as? CAGradientLayer,
           let colors = gradientLayer.colors,
           !colors.isEmpty {
            do {
                gradient = try GradientRenderConfiguration(
                    type: gradientLayer.type,
                    colors: colors,
                    locations: gradientLayer.locations,
                    startPoint: gradientLayer.startPoint,
                    endPoint: gradientLayer.endPoint
                )
            } catch {
                throw .invalidLayerGradient(
                    snapshotGradientError(from: error)
                )
            }
        } else {
            gradient = nil
        }
        let shape = try captureShape(from: layer)
        let text = try captureText(from: layer)
        return PresentationValues(
            replicator: replicator,
            emitter: emitter,
            tiled: tiled,
            transition: transition,
            replicatorInstanceTransform:
                CATransform3DIdentity,
            effectiveReplicatorColor:
                SIMD4<Float>(repeating: 1),
            effectiveReplicatorTimeOffset: 0,
            bounds: layer.bounds,
            contentsScale: layer.contentsScale,
            boundsSize: SIMD2<Float>(boundsWidth, boundsHeight),
            boundsOrigin: boundsOrigin,
            position: position,
            anchorOffset: anchorOffset,
            transform: layer.transform,
            sublayerTransform: layer.sublayerTransform,
            isTransformLayer: isTransformLayer,
            isGeometryFlipped: layer.isGeometryFlipped,
            isDoubleSided:
                isDepthContainer ? true : layer.isDoubleSided,
            isOpaque: isDepthContainer ? false : layer.isOpaque,
            masksToBounds:
                isDepthContainer ? false : layer.masksToBounds,
            allowsGroupOpacity:
                isDepthContainer ? false : layer.allowsGroupOpacity,
            shouldRasterize:
                isDepthContainer ? false : layer.shouldRasterize,
            rasterizationScale:
                isDepthContainer ? 1 : layer.rasterizationScale,
            opacity: layer.opacity,
            isHidden: layer.isHidden,
            cornerRadius:
                isDepthContainer ? 0 : Float(layer.cornerRadius),
            cornerCurveExponent: cornerCurveExponent,
            cornerRadii: cornerRadii,
            edgeAntialiasingMask: !isDepthContainer
                && layer.allowsEdgeAntialiasing
                ? Float(layer.edgeAntialiasingMask.rawValue & 0xF)
                : 0,
            backgroundColor: isDepthContainer
                ? nil
                : try colorComponents(
                    layer.backgroundColor,
                    failure: .invalidLayerBackgroundColor
                ),
            borderWidth: borderWidth,
            borderColor: isDepthContainer
                ? nil
                : try colorComponents(
                    layer.borderColor,
                    failure: .invalidLayerBorderColor
                ),
            toneMapMode:
                isDepthContainer ? .automatic : layer.toneMapMode,
            preferredDynamicRange:
                isDepthContainer
                    ? .automatic
                    : layer.preferredDynamicRange,
            contentsHeadroom: contentsHeadroom,
            imageContents: imageContents,
            filters: filters,
            compositingFilter: compositingFilter,
            backgroundFilters: backgroundFilters,
            gradient: gradient,
            shape: shape,
            text: text,
            shadow: shadow
        )
    }

    private static func captureText(
        from layer: CALayer
    ) throws(CARendererError) -> PresentationValues.Text? {
        guard let textLayer = layer as? CATextLayer else {
            return nil
        }
        guard textLayer.string != nil else {
            return PresentationValues.Text(configuration: nil)
        }
        do {
            return PresentationValues.Text(
                configuration: try CATextRenderConfiguration(
                    layer: textLayer
                )
            )
        } catch {
            throw .invalidLayerText(textError(from: error))
        }
    }

    private static func textError(
        from error: CATextRenderConfigurationError
    ) -> CARenderSnapshotTextError {
        switch error {
        case .unsupportedStringValue:
            return .unsupportedStringValue
        case .unsupportedFontValue:
            return .unsupportedFontValue
        case .invalidFontSize:
            return .invalidFontSize
        case .invalidContentsScale:
            return .invalidContentsScale
        case .invalidBounds:
            return .invalidBounds
        case .invalidForegroundColor:
            return .invalidForegroundColor
        case .unsupportedAlignmentMode(let value):
            return .unsupportedAlignmentMode(value)
        case .unsupportedTruncationMode(let value):
            return .unsupportedTruncationMode(value)
        }
    }

    private static func captureShape(
        from layer: CALayer
    ) throws(CARendererError) -> PresentationValues.Shape? {
        guard let shapeLayer = layer as? CAShapeLayer else {
            return nil
        }
        guard let path = shapeLayer.path else {
            return PresentationValues.Shape(fill: nil, stroke: nil)
        }
        do {
            try ShapeFillTessellator.validate(path)
        } catch {
            throw .invalidLayerShape(shapeError(from: error))
        }
        let fill: PresentationValues.Shape.Primitive?
        if let fillColor = shapeLayer.fillColor {
            let points: [CGPoint]
            do {
                points = try ShapeFillTessellator.triangles(
                    for: path,
                    rule: shapeLayer.fillRule
                )
            } catch {
                throw .invalidLayerShape(shapeError(from: error))
            }
            fill = try shapePrimitive(
                points: points,
                color: fillColor,
                invalidColor: .invalidFillColor
            )
        } else {
            fill = nil
        }
        let stroke: PresentationValues.Shape.Primitive?
        if let strokeColor = shapeLayer.strokeColor {
            guard shapeLayer.lineWidth.isFinite else {
                throw .invalidLayerShape(.invalidStrokeGeometry)
            }
            guard shapeLayer.lineWidth > 0 else {
                return PresentationValues.Shape(
                    fill: fill,
                    stroke: nil
                )
            }
            let points: [CGPoint]
            do {
                points = try ShapeStrokeTessellator.triangles(
                    for: path,
                    lineWidth: shapeLayer.lineWidth,
                    lineCap: shapeLayer.lineCap,
                    lineJoin: shapeLayer.lineJoin,
                    miterLimit: shapeLayer.miterLimit,
                    dashPattern: shapeLayer.lineDashPattern,
                    dashPhase: shapeLayer.lineDashPhase,
                    strokeStart: shapeLayer.strokeStart,
                    strokeEnd: shapeLayer.strokeEnd
                )
            } catch {
                throw .invalidLayerShape(shapeError(from: error))
            }
            stroke = try shapePrimitive(
                points: points,
                color: strokeColor,
                invalidColor: .invalidStrokeColor
            )
        } else {
            stroke = nil
        }
        return PresentationValues.Shape(fill: fill, stroke: stroke)
    }

    private static func shapePrimitive(
        points: [CGPoint],
        color: CGColor,
        invalidColor: CARenderSnapshotShapeError
    ) throws(CARendererError) -> PresentationValues.Shape.Primitive? {
        guard !points.isEmpty else { return nil }
        guard let converted = color.converted(
            to: .deviceRGB,
            intent: .defaultIntent,
            options: nil
        ), let components = converted.components,
              components.count == 4,
              components.allSatisfy(\.isFinite) else {
            throw .invalidLayerShape(invalidColor)
        }
        let vertices = points.map { SIMD2(Float($0.x), Float($0.y)) }
        guard vertices.allSatisfy({
            $0.x.isFinite && $0.y.isFinite
        }) else {
            throw .invalidLayerShape(.nonFinitePath)
        }
        return PresentationValues.Shape.Primitive(
            vertices: vertices,
            color: SIMD4(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
        )
    }

    private static func shapeError(
        from error: ShapeFillTessellationError
    ) -> CARenderSnapshotShapeError {
        switch error {
        case .unsupportedFillRule(let value):
            return .unsupportedFillRule(value)
        case .nonFinitePath:
            return .nonFinitePath
        }
    }

    private static func shapeError(
        from error: ShapeStrokeTessellationError
    ) -> CARenderSnapshotShapeError {
        switch error {
        case .invalidGeometry:
            return .invalidStrokeGeometry
        case .invalidDashPattern:
            return .invalidDashPattern
        case .unsupportedLineCap(let value):
            return .unsupportedLineCap(value)
        case .unsupportedLineJoin(let value):
            return .unsupportedLineJoin(value)
        }
    }

    private static func snapshotGradientError(
        from error: GradientRenderConfigurationError
    ) -> CARenderSnapshotGradientError {
        switch error {
        case .unsupportedType(let value):
            return .unsupportedType(value)
        case .nonFiniteGeometry:
            return .nonFiniteGeometry
        case .invalidColor(let index):
            return .invalidColor(index: index)
        case .invalidColorComponents(let index):
            return .invalidColorComponents(index: index)
        case .invalidLocationCount(let expected, let actual):
            return .invalidLocationCount(
                expected: expected,
                actual: actual
            )
        case .nonFiniteLocation(let index):
            return .nonFiniteLocation(index: index)
        case .locationOutOfRange(let index):
            return .locationOutOfRange(index: index)
        case .locationsNotMonotonic(let index):
            return .locationsNotMonotonic(index: index)
        }
    }

    private static func captureImageContents(
        from layer: CALayer,
        delegateBackingStore: CADelegateBackingStore?
    ) throws(CAImageContentsSnapshotError) -> CAImageContentsSnapshot? {
        let image: CGImage
        if let delegateImage = delegateBackingStore?.image {
            image = delegateImage
        } else if let contents = layer.contents {
            guard let contentsImage = contents as? CGImage else {
                throw .unsupportedContentsType(
                    String(reflecting: type(of: contents))
                )
            }
            image = contentsImage
        } else {
            return nil
        }
        guard let sampling = CAContentsSampling(
            magnificationFilter: layer.magnificationFilter,
            minificationFilter: layer.minificationFilter
        ) else {
            throw .invalidSamplingFilters(
                magnification: layer.magnificationFilter,
                minification: layer.minificationFilter
            )
        }
        guard layer.minificationFilterBias.isFinite else {
            throw .invalidMinificationFilterBias(
                layer.minificationFilterBias
            )
        }
        let storage: CGImageTextureStorage
        do {
            storage = try CGImageTextureStorageConverter.convert(image)
        } catch {
            throw .imageConversion(error)
        }
        return CAImageContentsSnapshot(
            storage: storage,
            origin: delegateBackingStore.map {
                .delegateBackingStore($0.format.contentsFormat)
            } ?? .layerContents,
            contentsRect: layer.contentsRect,
            contentsCenter: layer.contentsCenter,
            contentsScale: layer.contentsScale,
            gravity: layer.contentsGravity,
            sampling: sampling,
            minificationFilterBias: min(
                max(layer.minificationFilterBias, -16),
                15.99
            ),
            isOpaque: layer.isOpaque
        )
    }

    private static func cornerRadiiComponents(
        from layer: CALayer
    ) -> SIMD4<Float> {
        let radius = Float(layer.cornerRadius)
        guard radius > 0 else { return .zero }
        return SIMD4<Float>(
            layer.maskedCorners.contains(.layerMinXMinYCorner) ? radius : 0,
            layer.maskedCorners.contains(.layerMaxXMinYCorner) ? radius : 0,
            layer.maskedCorners.contains(.layerMinXMaxYCorner) ? radius : 0,
            layer.maskedCorners.contains(.layerMaxXMaxYCorner) ? radius : 0
        )
    }

    private static func colorComponents(
        _ color: CGColor?,
        failure: CARendererError
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
            throw failure
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
            throw failure
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

/// Captures the exact model revisions a live-tree renderer submitted.
///
/// This is a transitional frame-boundary contract for WebGPU while its
/// complete immutable value/resource snapshot is implemented. It does not
/// authorize the renderer to read mutable layer state after capture; it only
/// prevents a successful submission from clearing later mutations.
internal struct CARenderRevisionSnapshot: Sendable {
    internal let capturedContentRevisions: [ObjectIdentifier: UInt64]

    internal static func capture(
        _ rootLayer: CALayer
    ) throws(CARendererError) -> CARenderRevisionSnapshot {
        var revisions: [ObjectIdentifier: UInt64] = [:]
        var activePath: Set<ObjectIdentifier> = []
        try captureNode(
            rootLayer,
            revisions: &revisions,
            activePath: &activePath
        )
        return CARenderRevisionSnapshot(
            capturedContentRevisions: revisions
        )
    }

    private static func captureNode(
        _ layer: CALayer,
        revisions: inout [ObjectIdentifier: UInt64],
        activePath: inout Set<ObjectIdentifier>
    ) throws(CARendererError) {
        let identity = ObjectIdentifier(layer)
        guard activePath.insert(identity).inserted else {
            throw .cyclicLayerHierarchy
        }
        defer {
            activePath.remove(identity)
        }

        if revisions[identity] == nil {
            revisions[identity] = layer._contentRevision
        }
        if let mask = layer._maskForDirty {
            try captureNode(
                mask,
                revisions: &revisions,
                activePath: &activePath
            )
        }
        for sublayer in layer._sublayersForDirty ?? [] {
            try captureNode(
                sublayer,
                revisions: &revisions,
                activePath: &activePath
            )
        }
    }
}

/// The exact state published by the outermost transaction for one render root.
internal enum CACommittedRenderState: Sendable {
    case snapshot(CARenderSnapshot)
    case captureFailure(frameToken: UInt64, error: CARendererError)
    case requiresLiveAnimationEvaluation(frameToken: UInt64)

    internal var frameToken: UInt64 {
        switch self {
        case .snapshot(let snapshot):
            snapshot.frameToken
        case .captureFailure(let frameToken, _),
             .requiresLiveAnimationEvaluation(let frameToken):
            frameToken
        }
    }
}
