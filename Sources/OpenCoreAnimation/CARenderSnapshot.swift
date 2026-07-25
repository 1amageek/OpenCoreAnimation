import Foundation

internal enum CARenderSnapshotLiveTreeRequirement: Equatable, Sendable {
    case specializedLayer
    case contents
    case mask
    case opacityGroup
    case shadow
    case filters
    case backdropComposition
    case rasterization
    case transition
}

/// An immutable, value-owned view of the presentation state required by a
/// renderer for one frame.
///
/// The snapshot intentionally stores layer identity without retaining a
/// `CALayer`. This prevents mutations made after capture from changing the
/// frame that is already being encoded.
// FIXME(INCOMPLETE_IMPLEMENTATION): The immutable snapshot contains every value
// consumed by CAMetalRenderer and by CAWebGPURenderer's static snapshot path,
// including nested rectangular and rounded clipping and ordinary CGImage
// contents. Production WebGPU still uses explicitly typed live-tree branches
// for non-image contents, masks, specialized layers, animation evaluation, and
// layout preparation. Phase 4 must not be considered complete until those
// values and resources are owned here, the live-tree commit states are removed,
// and every WebGPU frame encodes without reading mutable model layers after
// capture.
internal struct CARenderSnapshot: Sendable {
    internal struct PresentationValues: Sendable, Equatable {
        internal let bounds: CGRect
        internal let boundsSize: SIMD2<Float>
        internal let boundsOrigin: SIMD2<Float>
        internal let position: SIMD3<Float>
        internal let anchorOffset: SIMD3<Float>
        internal let transform: CATransform3D
        internal let sublayerTransform: CATransform3D
        internal let isGeometryFlipped: Bool
        internal let isDoubleSided: Bool
        internal let masksToBounds: Bool
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
    internal let liveTreeRequirement: CARenderSnapshotLiveTreeRequirement?

    internal static func capture(
        _ rootLayer: CALayer,
        frameToken: UInt64
    ) throws(CARendererError) -> CARenderSnapshot {
        var nodes: [Node] = []
        var visited: Set<ObjectIdentifier> = []
        var liveTreeRequirement: CARenderSnapshotLiveTreeRequirement?
        let rootIndex = try captureNode(
            rootLayer,
            nodes: &nodes,
            visited: &visited,
            liveTreeRequirement: &liveTreeRequirement
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
            capturedContentRevisions: capturedContentRevisions,
            liveTreeRequirement: liveTreeRequirement
        )
    }

    private static func captureNode(
        _ layer: CALayer,
        nodes: inout [Node],
        visited: inout Set<ObjectIdentifier>,
        liveTreeRequirement: inout CARenderSnapshotLiveTreeRequirement?
    ) throws(CARendererError) -> Int {
        let identity = ObjectIdentifier(layer)
        guard visited.insert(identity).inserted else {
            throw .cyclicLayerHierarchy
        }

        do {
            try layer.prepareDelegateBackingStore(
                maximumPixelDimension: Int.max
            )
        } catch {
            throw .invalidDelegateBackingStore(error)
        }
        let contentRevision = layer._contentRevision
        let presentationLayer = layer._renderTimePresentation()
        if liveTreeRequirement == nil {
            liveTreeRequirement = requiredLiveTreeFeature(
                modelLayer: layer,
                presentationLayer: presentationLayer
            )
        }
        let values = try presentationValues(
            from: presentationLayer,
            delegateBackingStore: layer.delegateBackingStore
        )
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
        for child in layer.sortedSublayers() {
            childIndices.append(
                try captureNode(
                    child,
                    nodes: &nodes,
                    visited: &visited,
                    liveTreeRequirement: &liveTreeRequirement
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

    private static func requiredLiveTreeFeature(
        modelLayer: CALayer,
        presentationLayer: CALayer
    ) -> CARenderSnapshotLiveTreeRequirement? {
        if ObjectIdentifier(type(of: modelLayer)) != ObjectIdentifier(CALayer.self) {
            return .specializedLayer
        }
        if presentationLayer.contents != nil,
           !(presentationLayer.contents is CGImage) {
            return .contents
        }
        if presentationLayer.mask != nil {
            return .mask
        }
        if presentationLayer.allowsGroupOpacity,
           presentationLayer.opacity < 1,
           modelLayer.sublayers?.isEmpty == false {
            return .opacityGroup
        }
        if presentationLayer.shadowOpacity > 0,
           presentationLayer.shadowColor != nil {
            return .shadow
        }
        if presentationLayer.filters?.isEmpty == false {
            return .filters
        }
        if presentationLayer.compositingFilter != nil
            || presentationLayer.backgroundFilters?.isEmpty == false {
            return .backdropComposition
        }
        if presentationLayer.shouldRasterize {
            return .rasterization
        }
        if presentationLayer._transitionRenderState != nil {
            return .transition
        }
        return nil
    }

    private static func presentationValues(
        from layer: CALayer,
        delegateBackingStore: CADelegateBackingStore? = nil
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
              layer.borderWidth.isFinite,
              layer.contentsHeadroom.isFinite,
              isFinite(layer.transform),
              isFinite(layer.sublayerTransform) else {
            throw .nonFiniteLayerGeometry
        }
        guard layer.cornerRadius >= 0 else {
            throw .invalidLayerCornerGeometry
        }
        let cornerCurveExponent: Float
        do {
            cornerCurveExponent = Float(
                try CornerCurveRenderConfiguration(
                    curve: layer.cornerCurve
                ).exponent
            )
        } catch {
            throw .invalidLayerCornerGeometry
        }
        let cornerRadii = cornerRadiiComponents(from: layer)
        guard cornerCurveExponent.isFinite,
              cornerCurveExponent > 0,
              cornerRadii.x.isFinite,
              cornerRadii.y.isFinite,
              cornerRadii.z.isFinite,
              cornerRadii.w.isFinite else {
            throw .invalidLayerCornerGeometry
        }
        let borderWidth = Float(layer.borderWidth)
        guard borderWidth.isFinite, borderWidth >= 0 else {
            throw .invalidLayerBorderWidth
        }
        let contentsHeadroom = Float(layer.contentsHeadroom)
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
        do {
            imageContents = try captureImageContents(
                from: layer,
                delegateBackingStore: delegateBackingStore
            )
        } catch {
            throw .invalidLayerContents(error)
        }
        return PresentationValues(
            bounds: layer.bounds,
            boundsSize: SIMD2<Float>(boundsWidth, boundsHeight),
            boundsOrigin: boundsOrigin,
            position: position,
            anchorOffset: anchorOffset,
            transform: layer.transform,
            sublayerTransform: layer.sublayerTransform,
            isGeometryFlipped: layer.isGeometryFlipped,
            isDoubleSided: layer.isDoubleSided,
            masksToBounds: layer.masksToBounds,
            opacity: layer.opacity,
            isHidden: layer.isHidden,
            cornerRadius: Float(layer.cornerRadius),
            cornerCurveExponent: cornerCurveExponent,
            cornerRadii: cornerRadii,
            edgeAntialiasingMask: layer.allowsEdgeAntialiasing
                ? Float(layer.edgeAntialiasingMask.rawValue & 0xF)
                : 0,
            backgroundColor: try colorComponents(
                layer.backgroundColor,
                failure: .invalidLayerBackgroundColor
            ),
            borderWidth: borderWidth,
            borderColor: try colorComponents(
                layer.borderColor,
                failure: .invalidLayerBorderColor
            ),
            toneMapMode: layer.toneMapMode,
            preferredDynamicRange: layer.preferredDynamicRange,
            contentsHeadroom: contentsHeadroom,
            imageContents: imageContents
        )
    }

    private static func captureImageContents(
        from layer: CALayer,
        delegateBackingStore: CADelegateBackingStore?
    ) throws(CAImageContentsSnapshotError) -> CAImageContentsSnapshot? {
        let image = delegateBackingStore?.image
            ?? (layer.contents as? CGImage)
        guard let image else { return nil }
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
    case requiresLiveTreePreparation(frameToken: UInt64)
    // FIXME(INCOMPLETE_IMPLEMENTATION): Static trees using this feature still
    // reach production WebGPU through the live-tree renderer. This branch must
    // not be treated as snapshot success until the named resource category is
    // value-owned by CARenderSnapshot and encoded without CALayer reads.
    case requiresLiveResourceCapture(
        frameToken: UInt64,
        requirement: CARenderSnapshotLiveTreeRequirement
    )

    internal var frameToken: UInt64 {
        switch self {
        case .snapshot(let snapshot):
            snapshot.frameToken
        case .captureFailure(let frameToken, _),
             .requiresLiveAnimationEvaluation(let frameToken),
             .requiresLiveTreePreparation(let frameToken),
             .requiresLiveResourceCapture(let frameToken, _):
            frameToken
        }
    }
}
