import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

private struct SnapshotTileContent:
    CATiledLayerContentSnapshot {
    let value: Int

    func drawTile(
        _ tile: CATiledLayerTileDrawingInfo,
        in context: CGContext
    ) {}
}

private final class SnapshotTileProvider:
    CATiledLayerContentProvider {
    func makeTileContentSnapshot()
        -> any CATiledLayerContentSnapshot {
        SnapshotTileContent(value: 42)
    }
}

private final class SnapshotNonSendableTileDelegate:
    CALayerDelegate {}

@MainActor
@Suite("Immutable render snapshots", .serialized)
struct CARenderSnapshotTests {
    @Test("Hierarchy-only mutations publish through implicit transactions")
    func hierarchyMutationPublishesImplicitSnapshot() {
        CATransaction.flush()
        let root = CALayer()
        let child = CALayer()

        root.addSublayer(child)
        #expect(root.pendingCommittedRenderState == nil)

        CATransaction.deliverScheduledImplicitCommitForTesting()

        guard case .snapshot(let snapshot) =
                root.pendingCommittedRenderState else {
            Issue.record(
                "Expected hierarchy mutation to publish an implicit snapshot"
            )
            return
        }
        #expect(
            snapshot.nodes.contains {
                $0.identity == ObjectIdentifier(child)
            }
        )
    }

    @Test("Outermost transaction publishes an immutable root snapshot")
    func commitPublishesSnapshot() throws {
        CATransaction.flush()
        let root = CALayer()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        CATransaction.commit()

        guard case .snapshot(let snapshot) = root.pendingCommittedRenderState else {
            Issue.record("Expected the outermost transaction to publish a snapshot")
            return
        }
        #expect(snapshot.rootBounds == root.bounds)
        #expect(snapshot.nodes[snapshot.rootIndex].presentationValues.backgroundColor
            == SIMD4<Float>(0, 1, 0, 1))
    }

    @Test("Committed emitter cells remain independent of later model mutation")
    func committedEmitterCellsRemainImmutable() throws {
        CATransaction.flush()
        let root = CALayer()
        let emitter = CAEmitterLayer()
        let cell = CAEmitterCell()
        var completionRan = false

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        CATransaction.setCompletionBlock {
            completionRan = true
        }
        root.bounds = CGRect(x: 0, y: 0, width: 64, height: 64)
        emitter.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        cell.birthRate = 60
        cell.lifetime = 3
        emitter.emitterCells = [cell]
        root.addSublayer(emitter)
        CATransaction.commit()

        cell.birthRate = 0

        guard case .snapshot(let snapshot) =
                root.pendingCommittedRenderState else {
            Issue.record("Expected a committed emitter snapshot")
            CATransaction.flush()
            return
        }
        let emitterNode = try #require(
            snapshot.nodes.first {
                $0.identity == ObjectIdentifier(emitter)
            }
        )
        let configuration = try #require(
            emitterNode.presentationValues.emitter
        )
        #expect(configuration.emitterCells.count == 1)
        #expect(configuration.emitterCells[0].birthRate == 60)
        #expect(cell.birthRate == 0)
        #expect(!completionRan)

        CATransaction.flush()
    }

    @Test("Detached mask descendant mutations publish the owning root")
    func detachedMaskMutationPublishesOwningRoot() throws {
        CATransaction.flush()
        let root = CALayer()
        let maskRoot = CALayer()
        let maskChild = CALayer()
        maskRoot.addSublayer(maskChild)
        root.mask = maskRoot
        CATransaction.flush()

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        maskChild.opacity = 0.25
        CATransaction.commit()

        guard case .snapshot(let snapshot) =
                root.pendingCommittedRenderState else {
            Issue.record(
                "Expected the mask owner to publish a snapshot"
            )
            return
        }
        let maskChildIdentity = ObjectIdentifier(maskChild)
        let capturedMaskChild = snapshot.nodes.first {
            $0.identity == maskChildIdentity
        }
        #expect(capturedMaskChild?.presentationValues.opacity == 0.25)
        #expect(maskRoot.pendingCommittedRenderState == nil)
    }

    @Test("Common solid state and z-ordered hierarchy are value-owned")
    func commonSolidStateIsCaptured() throws {
        let root = CALayer()
        root.borderWidth = 2
        root.borderColor = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        root.cornerRadius = 4
        root.maskedCorners = [.layerMinXMinYCorner]
        root.masksToBounds = true
        root.isGeometryFlipped = true
        root.toneMapMode = .never
        root.preferredDynamicRange = .high
        root.contentsHeadroom = 2

        let front = CALayer()
        front.zPosition = 2
        front.isDoubleSided = false
        let back = CALayer()
        back.zPosition = -1
        root.addSublayer(front)
        root.addSublayer(back)

        let snapshot = try CARenderSnapshot.capture(root, frameToken: 45)
        let rootNode = snapshot.nodes[snapshot.rootIndex]
        let values = rootNode.presentationValues
        #expect(values.borderWidth == 2)
        #expect(values.borderColor == SIMD4<Float>(0, 0, 1, 1))
        #expect(values.cornerRadii == SIMD4<Float>(4, 0, 0, 0))
        #expect(values.masksToBounds)
        #expect(values.isGeometryFlipped)
        #expect(values.toneMapMode == .never)
        #expect(values.preferredDynamicRange == .high)
        #expect(values.contentsHeadroom == 2)
        let frontIndex = try #require(
            rootNode.childIndices.first {
                snapshot.nodes[$0].identity == ObjectIdentifier(front)
            }
        )
        #expect(
            !snapshot.nodes[frontIndex]
                .presentationValues.isDoubleSided
        )
        #expect(
            rootNode.childIndices.map { snapshot.nodes[$0].identity }
                == [ObjectIdentifier(back), ObjectIdentifier(front)]
        )
    }

    @Test("Committed evaluator preserves scalar transform keyframes")
    func committedEvaluatorPreservesScalarTransformKeyframes() throws {
        CATransaction.flush()
        let root = CALayer()
        let translated = CALayer()
        translated.bounds = CGRect(
            x: 0,
            y: 0,
            width: 20,
            height: 20
        )
        translated.position = CGPoint(x: 60, y: 60)
        root.addSublayer(translated)

        let animation = CAKeyframeAnimation(
            keyPath: "transform.translation.x"
        )
        animation.values = [CGFloat(0), CGFloat(40)]
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        translated.add(animation, forKey: "translation")

        let evaluator = try CACommittedAnimationEvaluator(
            rootLayer: root,
            frameToken: 71
        )
        let snapshot = try evaluator.snapshot(frameToken: 72)
        let translatedIndex = try #require(
            snapshot.nodes[snapshot.rootIndex].childIndices.first
        )
        let transform = snapshot.nodes[translatedIndex]
            .presentationValues.transform

        #expect(abs(transform.m41 - 20) < 0.001)
        CATransaction.flush()
    }

    @Test("Transition source and configuration become immutable snapshot values")
    func transitionStateIsCaptured() throws {
        let source = CALayer()
        source.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        source.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        let sourceChild = CALayer()
        sourceChild.bounds = CGRect(
            x: 0,
            y: 0,
            width: 4,
            height: 4
        )
        sourceChild.backgroundColor = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        source.addSublayer(sourceChild)

        let target = CALayer()
        target.bounds = source.bounds
        target.contentsScale = 2
        target.recursivelyClearDirtyAfterCommit()
        let transition = CATransition()
        target._transitionRenderState = CATransitionRenderState(
            resourceIdentity: transition.resourceIdentity,
            sourceLayer: source,
            type: .push,
            subtype: .fromRight,
            filter: nil,
            progress: 0.25
        )

        let snapshot = try CARenderSnapshot.capture(
            target,
            frameToken: 47
        )
        let targetNode = snapshot.nodes[snapshot.rootIndex]
        let capturedTransition = try #require(
            targetNode.presentationValues.transition
        )
        let sourceNode = snapshot.nodes[
            capturedTransition.sourceRootIndex
        ]
        #expect(
            capturedTransition.resourceIdentity
                == transition.resourceIdentity
        )
        #expect(capturedTransition.type == .push)
        #expect(capturedTransition.subtype == .fromRight)
        #expect(capturedTransition.progress == 0.25)
        #expect(targetNode.presentationValues.contentsScale == 2)
        #expect(
            sourceNode.presentationValues.backgroundColor
                == SIMD4<Float>(1, 0, 0, 1)
        )
        #expect(sourceNode.childIndices.count == 1)
        #expect(
            !targetNode.childIndices.contains(
                capturedTransition.sourceRootIndex
            )
        )

        source.backgroundColor = CGColor(
            red: 0,
            green: 1,
            blue: 0,
            alpha: 1
        )
        sourceChild.backgroundColor = CGColor(
            red: 1,
            green: 1,
            blue: 0,
            alpha: 1
        )
        #expect(
            sourceNode.presentationValues.backgroundColor
                == SIMD4<Float>(1, 0, 0, 1)
        )
        let capturedChild = snapshot.nodes[
            try #require(sourceNode.childIndices.first)
        ]
        #expect(
            capturedChild.presentationValues.backgroundColor
                == SIMD4<Float>(0, 0, 1, 1)
        )
    }

    @Test("Invalid built-in transition is isolated in its snapshot node")
    func invalidTransitionIsIsolated() throws {
        let source = CALayer()
        let target = CALayer()
        target.recursivelyClearDirtyAfterCommit()
        let transition = CATransition()
        target._transitionRenderState = CATransitionRenderState(
            resourceIdentity: transition.resourceIdentity,
            sourceLayer: source,
            type: CATransitionType(rawValue: "unsupported"),
            subtype: nil,
            filter: nil,
            progress: 0.5
        )

        let snapshot = try CARenderSnapshot.capture(
            target,
            frameToken: 48
        )
        #expect(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.transition?
                .preparationFailure
                == .unsupportedTransitionType("unsupported")
        )
    }

    @Test("Nonportable transition filter is isolated in its snapshot node")
    func nonportableTransitionFilterIsIsolated() throws {
        let source = CALayer()
        let target = CALayer()
        target.recursivelyClearDirtyAfterCommit()
        let transition = CATransition()
        target._transitionRenderState = CATransitionRenderState(
            resourceIdentity: transition.resourceIdentity,
            sourceLayer: source,
            type: .fade,
            subtype: nil,
            filter: "not-a-filter",
            progress: 0.5
        )

        let snapshot = try CARenderSnapshot.capture(
            target,
            frameToken: 49
        )
        #expect(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.transition?
                .preparationFailure
                == .unsupportedFilterValue("Swift.String")
        )
    }

    @Test("Invalid committed transitions do not block sibling evaluation")
    func invalidCommittedTransitionsDoNotBlockSiblings() throws {
        let root = CALayer()
        let filteredLayer = CALayer()
        let typedLayer = CALayer()
        root.addSublayer(filteredLayer)
        root.addSublayer(typedLayer)

        let filteredTransition = CATransition()
        filteredTransition.filter = "not-a-filter"
        filteredTransition.duration = 1
        filteredTransition.speed = 0
        filteredTransition.timeOffset = 0.5
        filteredTransition.fillMode = .both
        filteredTransition.isRemovedOnCompletion = false
        filteredLayer.add(
            filteredTransition,
            forKey: "invalid-filter"
        )

        let typedTransition = CATransition()
        typedTransition.type =
            CATransitionType(rawValue: "unsupported")
        typedTransition.duration = 1
        typedTransition.speed = 0
        typedTransition.timeOffset = 0.5
        typedTransition.fillMode = .both
        typedTransition.isRemovedOnCompletion = false
        typedLayer.add(
            typedTransition,
            forKey: "invalid-type"
        )

        let evaluator = try CACommittedAnimationEvaluator(
            rootLayer: root,
            frameToken: 50
        )
        let snapshot = try evaluator.snapshot(
            frameToken: 50
        )
        let failures = snapshot.nodes.compactMap {
            $0.presentationValues.transition?
                .preparationFailure
        }

        #expect(
            failures.contains(
                .unsupportedFilterValue("Swift.String")
            )
        )
        #expect(
            failures.contains(
                .unsupportedTransitionType("unsupported")
            )
        )
    }

    @Test("Filter vector arrays become finite immutable values")
    func filterVectorArraysAreCaptured() throws {
        #expect(
            try CARenderSnapshotFilterParameter.capture(
                [Float(40), Float(20)],
                filterName: "VectorFilter",
                key: "inputCenter"
            ) == .vector([40, 20])
        )
        #expect(
            try CARenderSnapshotFilterParameter.capture(
                [Double(8), Double(4)],
                filterName: "VectorFilter",
                key: "inputExtent"
            ) == .vector([8, 4])
        )
        #expect(throws: CARenderSnapshotFilterError.nonFiniteCoreImageParameter(
            filter: "VectorFilter",
            key: "inputCenter"
        )) {
            _ = try CARenderSnapshotFilterParameter.capture(
                [Float.infinity, 1],
                filterName: "VectorFilter",
                key: "inputCenter"
            )
        }
    }

    @Test("Tiled content becomes immutable committed input")
    func tiledContentIsCaptured() throws {
        let layer = CATiledLayer()
        layer.bounds = CGRect(
            x: 4,
            y: 8,
            width: 128,
            height: 64
        )
        layer.tileSize = CGSize(width: 32, height: 16)
        layer.levelsOfDetail = 3
        let provider = SnapshotTileProvider()
        layer.delegate = provider
        let expectedGeneration = layer.tileCacheGeneration

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 46
        )
        let values = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues
        let configuration = try #require(values.tiled)
        let capturedContent = try #require(
            configuration.capturedContent
        )
        let content = try #require(
            capturedContent.snapshot
                as? SnapshotTileContent
        )
        #expect(
            configuration.resourceIdentity
                == layer.resourceIdentity
        )
        #expect(
            configuration.cacheGeneration
                == expectedGeneration
        )
        #expect(configuration.bounds == layer.bounds)
        #expect(configuration.tileSize == layer.tileSize)
        #expect(content.value == 42)
    }

    @Test("Unsafe tile delegates fail committed capture exactly")
    func unsafeTileDelegateFailsCapture() {
        let layer = CATiledLayer()
        let delegate = SnapshotNonSendableTileDelegate()
        layer.delegate = delegate

        #expect(throws: CARendererError.invalidLayerTiled(
            .delegateRequiresSendableTileProvider
        )) {
            try CARenderSnapshot.capture(
                layer,
                frameToken: 47
            )
        }
    }

    @Test("Base-only subclasses and scroll layers use immutable snapshots")
    func baseOnlySubclassesUseSnapshots() throws {
        let root = SnapshotBaseLayer()
        root.backgroundColor = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        let scroll = CAScrollLayer()
        scroll.bounds = CGRect(
            x: 12,
            y: 8,
            width: 40,
            height: 30
        )
        scroll.scrollMode = .horizontally
        root.addSublayer(scroll)

        let snapshot = try CARenderSnapshot.capture(
            root,
            frameToken: 45
        )
        let rootNode = snapshot.nodes[snapshot.rootIndex]
        let scrollIndex = try #require(
            rootNode.childIndices.first
        )
        let scrollNode = snapshot.nodes[scrollIndex]
        #expect(scroll.masksToBounds)
        #expect(
            rootNode.presentationValues.backgroundColor
                == SIMD4<Float>(0, 0, 1, 1)
        )
        #expect(
            scrollNode.presentationValues.boundsOrigin
                == SIMD2<Float>(12, 8)
        )
        #expect(scrollNode.presentationValues.masksToBounds)

        root.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        scroll.bounds.origin = .zero
        #expect(
            rootNode.presentationValues.backgroundColor
                == SIMD4<Float>(0, 0, 1, 1)
        )
        #expect(
            scrollNode.presentationValues.boundsOrigin
                == SIMD2<Float>(12, 8)
        )
    }

    @Test("Transform layers capture only their 3D container contract")
    func transformLayerValuesUseSnapshots() throws {
        let transform = CATransformLayer()
        transform.bounds = CGRect(
            x: 3,
            y: 4,
            width: 40,
            height: 30
        )
        transform.position = CGPoint(x: 20, y: 15)
        transform.opacity = 0.5
        transform.transform = CATransform3DMakeTranslation(
            2,
            3,
            4
        )
        transform.sublayerTransform = CATransform3DMakeScale(
            2,
            2,
            1
        )

        let image = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 255, 255]
        )
        let delegate = SnapshotDisplayDelegate(image: image)
        transform.delegate = delegate
        transform.setNeedsDisplay()
        transform.contents = SnapshotContentsToken()
        transform.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        transform.borderWidth = .nan
        transform.cornerRadius = .nan
        transform.contentsHeadroom = .nan
        transform.filters = [SnapshotContentsToken()]
        transform.mask = CALayer()

        let first = CALayer()
        first.zPosition = 4
        let second = CALayer()
        second.zPosition = -2
        transform.addSublayer(first)
        transform.addSublayer(second)

        let snapshot = try CARenderSnapshot.capture(
            transform,
            frameToken: 46
        )
        let node = snapshot.nodes[snapshot.rootIndex]
        let values = node.presentationValues
        #expect(snapshot.nodes.count == 3)
        #expect(values.isTransformLayer)
        #expect(values.opacity == 0.5)
        #expect(values.transform.m41 == 2)
        #expect(values.sublayerTransform.m11 == 2)
        #expect(values.backgroundColor == nil)
        #expect(values.borderWidth == 0)
        #expect(values.imageContents == nil)
        #expect(values.filters.isEmpty)
        #expect(values.compositingFilter == nil)
        #expect(values.backgroundFilters.isEmpty)
        #expect(values.shadow == nil)
        #expect(node.maskIndex == nil)
        #expect(delegate.displayCount == 0)
        #expect(
            node.childIndices.map {
                snapshot.nodes[$0].identity
            } == [
                ObjectIdentifier(first),
                ObjectIdentifier(second),
            ]
        )

        transform.opacity = 1
        transform.transform = CATransform3DIdentity
        transform.sublayerTransform = CATransform3DIdentity
        first.removeFromSuperlayer()

        #expect(values.opacity == 0.5)
        #expect(values.transform.m41 == 2)
        #expect(values.sublayerTransform.m11 == 2)
        #expect(node.childIndices.count == 2)
    }

    @Test("Replicator instances become independent immutable subtrees")
    func replicatorInstancesUseSnapshots() throws {
        let replicator = CAReplicatorLayer()
        replicator.instanceCount = 3
        replicator.instanceDelay = 0.25
        replicator.instanceTransform =
            CATransform3DMakeTranslation(12, 0, 0)
        replicator.instanceColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        replicator.instanceRedOffset = -0.5
        replicator.instanceGreenOffset = 0.5

        let source = CALayer()
        source.bounds = CGRect(
            x: 0,
            y: 0,
            width: 8,
            height: 8
        )
        source.backgroundColor = CGColor(
            red: 1,
            green: 1,
            blue: 1,
            alpha: 1
        )
        let descendant = CALayer()
        descendant.bounds = source.bounds
        source.addSublayer(descendant)
        replicator.addSublayer(source)

        let snapshot = try CARenderSnapshot.capture(
            replicator,
            frameToken: 47
        )
        let root = snapshot.nodes[snapshot.rootIndex]
        #expect(snapshot.nodes.count == 7)
        #expect(root.presentationValues.replicator != nil)
        #expect(root.replicatorSourceChildCount == 1)
        #expect(root.childIndices.count == 3)

        let instances = root.childIndices.map {
            snapshot.nodes[$0]
        }
        #expect(instances.allSatisfy {
            $0.identity == ObjectIdentifier(source)
        })
        #expect(
            instances.map {
                $0.presentationValues
                    .replicatorInstanceTransform.m41
            } == [0, 12, 24]
        )
        #expect(
            instances.map {
                $0.presentationValues
                    .effectiveReplicatorColor
            } == [
                SIMD4<Float>(1, 0, 0, 1),
                SIMD4<Float>(0.5, 0.5, 0, 1),
                SIMD4<Float>(0, 1, 0, 1),
            ]
        )
        #expect(
            instances.map {
                $0.presentationValues
                    .effectiveReplicatorTimeOffset
            } == [0, 0.25, 0.5]
        )
        #expect(
            instances.map {
                snapshot.nodes[$0.childIndices[0]]
                    .presentationValues
                    .effectiveReplicatorColor
            } == instances.map {
                $0.presentationValues
                    .effectiveReplicatorColor
            }
        )

        replicator.instanceCount = 0
        replicator.instanceTransform = CATransform3DIdentity
        replicator.instanceColor = nil
        source.backgroundColor = CGColor(
            red: 0,
            green: 0,
            blue: 0,
            alpha: 1
        )
        source.removeFromSuperlayer()

        #expect(root.childIndices.count == 3)
        #expect(
            instances.map {
                $0.presentationValues
                    .replicatorInstanceTransform.m41
            } == [0, 12, 24]
        )
        #expect(
            instances[0].presentationValues.backgroundColor
                == SIMD4<Float>(1, 1, 1, 1)
        )
    }

    @Test("Nested replicator snapshots compose inherited instance values")
    func nestedReplicatorInstancesUseSnapshots() throws {
        let outer = CAReplicatorLayer()
        outer.instanceCount = 2
        outer.instanceDelay = 0.5
        outer.instanceTransform =
            CATransform3DMakeTranslation(10, 0, 0)
        outer.instanceColor = CGColor(
            red: 1,
            green: 0.5,
            blue: 1,
            alpha: 1
        )

        let inner = CAReplicatorLayer()
        inner.instanceCount = 2
        inner.instanceDelay = 0.25
        inner.instanceTransform =
            CATransform3DMakeTranslation(3, 0, 0)
        inner.instanceColor = CGColor(
            red: 0.5,
            green: 1,
            blue: 1,
            alpha: 1
        )
        inner.addSublayer(CALayer())
        outer.addSublayer(inner)

        let snapshot = try CARenderSnapshot.capture(
            outer,
            frameToken: 48
        )
        let outerNode = snapshot.nodes[snapshot.rootIndex]
        let innerInstances = outerNode.childIndices.map {
            snapshot.nodes[$0]
        }
        #expect(innerInstances.count == 2)
        #expect(
            innerInstances.map {
                $0.presentationValues
                    .replicatorInstanceTransform.m41
            } == [0, 10]
        )
        #expect(
            innerInstances.map {
                $0.presentationValues
                    .effectiveReplicatorTimeOffset
            } == [0, 0.5]
        )
        #expect(
            innerInstances.allSatisfy {
                $0.presentationValues.effectiveReplicatorColor
                    == SIMD4<Float>(1, 0.5, 1, 1)
            }
        )

        let sourceInstances = innerInstances.map { innerNode in
            innerNode.childIndices.map {
                snapshot.nodes[$0]
            }
        }
        #expect(
            sourceInstances.map {
                $0.map {
                    $0.presentationValues
                        .replicatorInstanceTransform.m41
                }
            } == [[0, 3], [0, 3]]
        )
        #expect(
            sourceInstances.map {
                $0.map {
                    $0.presentationValues
                        .effectiveReplicatorTimeOffset
                }
            } == [[0, 0.25], [0.5, 0.75]]
        )
        #expect(
            sourceInstances
                .flatMap { $0 }
                .allSatisfy {
                    $0.presentationValues
                        .effectiveReplicatorColor
                        == SIMD4<Float>(0.5, 0.5, 1, 1)
                }
        )
    }

    @Test("Depth-preserving replicators capture only their container contract")
    func depthPreservingReplicatorUsesSnapshots() throws {
        let replicator = CAReplicatorLayer()
        replicator.preservesDepth = true
        replicator.instanceCount = 1
        let image = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 255, 255]
        )
        let delegate = SnapshotDisplayDelegate(image: image)
        replicator.delegate = delegate
        replicator.setNeedsDisplay()
        replicator.contents = SnapshotContentsToken()
        replicator.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        replicator.borderWidth = .nan
        replicator.cornerRadius = .nan
        replicator.contentsHeadroom = .nan
        replicator.filters = [SnapshotContentsToken()]
        replicator.mask = CALayer()

        let first = CALayer()
        first.zPosition = 4
        let second = CALayer()
        second.zPosition = -2
        replicator.addSublayer(first)
        replicator.addSublayer(second)

        let snapshot = try CARenderSnapshot.capture(
            replicator,
            frameToken: 48
        )
        let node = snapshot.nodes[snapshot.rootIndex]
        let values = node.presentationValues
        #expect(values.replicator?.preservesDepth == true)
        #expect(values.backgroundColor == nil)
        #expect(values.borderWidth == 0)
        #expect(values.imageContents == nil)
        #expect(values.filters.isEmpty)
        #expect(values.shadow == nil)
        #expect(node.maskIndex == nil)
        #expect(delegate.displayCount == 0)
        #expect(node.replicatorSourceChildCount == 2)
        #expect(
            node.childIndices.map {
                snapshot.nodes[$0].identity
            } == [
                ObjectIdentifier(first),
                ObjectIdentifier(second),
            ]
        )
    }

    @Test("Replicator delay evaluates each immutable animation instance")
    func replicatorDelayEvaluatesEachInstance() throws {
        CATransaction.flush()
        let replicator = CAReplicatorLayer()
        replicator.bounds = CGRect(
            x: 0,
            y: 0,
            width: 100,
            height: 100
        )
        replicator.instanceCount = 2
        replicator.instanceDelay = 0.5
        let child = CALayer()
        child.bounds = CGRect(
            x: 0,
            y: 0,
            width: 10,
            height: 10
        )
        child.opacity = 1
        replicator.addSublayer(child)
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 2
        animation.beginTime = child.convertTime(
            CACurrentMediaTime(),
            from: nil
        ) - 1
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        child.add(animation, forKey: "opacity")

        let snapshot = try CARenderSnapshot.capture(
            replicator,
            frameToken: CALayer.advanceFrameToken()
        )
        let rootNode = snapshot.nodes[
            snapshot.rootIndex
        ]
        #expect(rootNode.childIndices.count == 2)
        let firstOpacity = snapshot.nodes[
            rootNode.childIndices[0]
        ].presentationValues.opacity
        let secondOpacity = snapshot.nodes[
            rootNode.childIndices[1]
        ].presentationValues.opacity
        #expect(firstOpacity > secondOpacity + 0.2)

        child.removeAllAnimations()
        CATransaction.flush()
    }

    @Test("Emitter cells and image bytes become immutable snapshot values")
    func emitterValuesUseSnapshots() throws {
        let image = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 0, 0, 255]
        )
        let child = CAEmitterCell()
        child.birthRate = 3
        child.lifetime = 4
        let cell = CAEmitterCell()
        cell.contents = image
        cell.contentsRect = CGRect(
            x: 0.25,
            y: 0,
            width: 0.5,
            height: 1
        )
        cell.contentsScale = 2
        cell.birthRate = 6
        cell.lifetime = 7
        cell.velocity = 8
        cell.color = CGColor(
            red: 0,
            green: 1,
            blue: 0,
            alpha: 1
        )
        cell.emitterCells = [child]

        let emitter = CAEmitterLayer()
        emitter.emitterCells = [cell]
        emitter.emitterPosition = CGPoint(x: 12, y: 14)
        emitter.emitterSize = CGSize(width: 16, height: 18)
        emitter.birthRate = 2
        emitter.seed = 99

        let snapshot = try CARenderSnapshot.capture(
            emitter,
            frameToken: 51
        )
        let values =
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues
        let configuration = try #require(values.emitter)
        let capturedCell = try #require(
            configuration.emitterCells.first
        )
        let capturedImage = try #require(capturedCell.image)
        #expect(
            configuration.simulationIdentity
                == emitter.simulationIdentity
        )
        #expect(configuration.emitterPosition == CGPoint(x: 12, y: 14))
        #expect(configuration.emitterSize == CGSize(width: 16, height: 18))
        #expect(configuration.birthRate == 2)
        #expect(configuration.seed == 99)
        #expect(capturedCell.birthRate == 6)
        #expect(capturedCell.lifetime == 7)
        #expect(capturedCell.velocity == 8)
        #expect(capturedCell.color == SIMD4<Float>(0, 1, 0, 1))
        #expect(capturedCell.childCells.first?.birthRate == 3)
        #expect(capturedImage.storage.data == Data([255, 0, 0, 255]))
        #expect(
            capturedImage.contentsRect
                == CGRect(x: 0.25, y: 0, width: 0.5, height: 1)
        )
        #expect(capturedImage.contentsScale == 2)

        emitter.emitterCells = []
        emitter.birthRate = 0
        cell.birthRate = 100
        cell.contents = nil
        child.birthRate = 200

        #expect(configuration.emitterCells.count == 1)
        #expect(capturedCell.birthRate == 6)
        #expect(capturedCell.childCells.first?.birthRate == 3)
        #expect(capturedImage.storage.data == Data([255, 0, 0, 255]))
    }

    @Test("Invalid emitter graphs fail immutable capture exactly")
    func invalidEmitterCaptureFails() {
        let cyclic = CAEmitterCell()
        cyclic.emitterCells = [cyclic]
        let emitter = CAEmitterLayer()
        emitter.emitterCells = [cyclic]

        #expect(throws: CARendererError.invalidLayerEmitter(
            .cyclicCellHierarchy(path: [0, 0])
        )) {
            try CARenderSnapshot.capture(
                emitter,
                frameToken: 52
            )
        }

        cyclic.emitterCells = nil
        cyclic.contents = SnapshotContentsToken()
        #expect(throws: CARendererError.invalidLayerEmitter(
            .invalidCellContents(path: [0])
        )) {
            try CARenderSnapshot.capture(
                emitter,
                frameToken: 53
            )
        }
    }

    @Test("Invalid replicator input remains a deferred typed value")
    func invalidReplicatorCaptureIsDeferred() throws {
        let replicator = CAReplicatorLayer()
        replicator.instanceCount =
            CAReplicatorRenderConfiguration
                .maximumInstanceCount + 1
        var snapshot = try CARenderSnapshot.capture(
            replicator,
            frameToken: 49
        )
        #expect(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues
                .replicatorCaptureFailure
            ==
            .instanceCountExceedsRendererCapacity(
                actual:
                    CAReplicatorRenderConfiguration
                        .maximumInstanceCount + 1,
                maximum:
                    CAReplicatorRenderConfiguration
                        .maximumInstanceCount
            )
        )

        replicator.instanceCount = 2
        replicator.instanceDelay = .nan
        snapshot = try CARenderSnapshot.capture(
            replicator,
            frameToken: 50
        )
        #expect(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues
                .replicatorCaptureFailure
            == .nonFiniteInstanceDelay
        )
    }

    @Test("Gradient stops and geometry become immutable snapshot values")
    func gradientValuesUseSnapshots() throws {
        let gradient = CAGradientLayer()
        gradient.bounds = CGRect(x: 0, y: 0, width: 40, height: 30)
        gradient.colors = [
            CGColor(red: 1, green: 0, blue: 0, alpha: 1),
            CGColor(red: 0, green: 0, blue: 1, alpha: 1),
        ]
        gradient.locations = [0.25, 0.75]
        gradient.startPoint = CGPoint(x: 0.1, y: 0.2)
        gradient.endPoint = CGPoint(x: 0.8, y: 0.9)

        let snapshot = try CARenderSnapshot.capture(
            gradient,
            frameToken: 46
        )
        let configuration = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.gradient
        )
        #expect(configuration.colorComponents == [
            SIMD4<Float>(1, 0, 0, 1),
            SIMD4<Float>(0, 0, 1, 1),
        ])
        #expect(configuration.locations == [0.25, 0.75])
        #expect(configuration.startPoint == SIMD2<Float>(0.1, 0.2))
        #expect(configuration.endPoint == SIMD2<Float>(0.8, 0.9))

        gradient.colors = [
            CGColor(red: 0, green: 1, blue: 0, alpha: 1),
        ]
        gradient.locations = nil
        gradient.startPoint = .zero
        gradient.endPoint = .zero
        #expect(configuration.colorCount == 2)
        #expect(configuration.locations == [0.25, 0.75])
        #expect(configuration.startPoint == SIMD2<Float>(0.1, 0.2))
        #expect(configuration.endPoint == SIMD2<Float>(0.8, 0.9))
    }

    @Test("Invalid gradients fail immutable capture explicitly")
    func invalidGradientFailsCapture() {
        let gradient = CAGradientLayer()
        gradient.colors = [
            CGColor(red: 1, green: 0, blue: 0, alpha: 1),
        ]
        gradient.type = CAGradientLayerType(rawValue: "future")

        #expect(throws: CARendererError.invalidLayerGradient(
            .unsupportedType("future")
        )) {
            try CARenderSnapshot.capture(gradient, frameToken: 47)
        }
    }

    @Test("Shape fill and stroke become immutable tessellated values")
    func shapeValuesUseSnapshots() throws {
        let committedPath = CGMutablePath()
        committedPath.addRect(
            CGRect(x: 2, y: 3, width: 20, height: 10)
        )
        let shape = CAShapeLayer()
        shape.bounds = CGRect(x: 0, y: 0, width: 24, height: 16)
        shape.path = committedPath
        shape.fillColor = CGColor(
            red: 0,
            green: 1,
            blue: 1,
            alpha: 1
        )
        shape.strokeColor = CGColor(
            red: 1,
            green: 0,
            blue: 1,
            alpha: 1
        )
        shape.lineWidth = 2
        shape.lineCap = .round
        shape.lineJoin = .bevel
        shape.lineDashPattern = [3, 2]
        shape.lineDashPhase = 1
        shape.strokeStart = 0.1
        shape.strokeEnd = 0.9

        let snapshot = try CARenderSnapshot.capture(
            shape,
            frameToken: 48
        )
        let captured = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.shape
        )
        let fill = try #require(captured.fill)
        let stroke = try #require(captured.stroke)
        #expect(fill.vertices.count == 6)
        #expect(fill.color == SIMD4<Float>(0, 1, 1, 1))
        #expect(!stroke.vertices.isEmpty)
        #expect(stroke.vertices.count.isMultiple(of: 3))
        #expect(stroke.color == SIMD4<Float>(1, 0, 1, 1))

        let capturedFillVertices = fill.vertices
        let capturedStrokeVertices = stroke.vertices
        committedPath.addRect(
            CGRect(x: 0, y: 0, width: 24, height: 16)
        )
        shape.path = CGMutablePath()
        shape.fillColor = CGColor(
            red: 1,
            green: 1,
            blue: 0,
            alpha: 1
        )
        shape.strokeColor = nil
        shape.lineWidth = 8
        shape.lineDashPattern = nil

        #expect(fill.vertices == capturedFillVertices)
        #expect(stroke.vertices == capturedStrokeVertices)
        #expect(fill.color == SIMD4<Float>(0, 1, 1, 1))
        #expect(stroke.color == SIMD4<Float>(1, 0, 1, 1))
    }

    @Test("A pathless shape remains an empty shape foreground")
    func pathlessShapeDoesNotFallBackToImageContents() throws {
        let shape = CAShapeLayer()
        shape.contents = SnapshotContentsToken()

        let snapshot = try CARenderSnapshot.capture(
            shape,
            frameToken: 49
        )
        let values = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues
        let captured = try #require(values.shape)
        #expect(captured.fill == nil)
        #expect(captured.stroke == nil)
        #expect(values.imageContents == nil)
    }

    @Test("Invalid shape values fail immutable capture explicitly")
    func invalidShapeFailsCapture() {
        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 10, height: 10))
        let shape = CAShapeLayer()
        shape.path = path
        shape.fillRule = CAShapeLayerFillRule(rawValue: "future")

        #expect(throws: CARendererError.invalidLayerShape(
            .unsupportedFillRule("future")
        )) {
            try CARenderSnapshot.capture(shape, frameToken: 50)
        }

        shape.fillRule = .nonZero
        shape.strokeColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        shape.lineDashPattern = [2, 0]
        #expect(throws: CARendererError.invalidLayerShape(
            .invalidDashPattern
        )) {
            try CARenderSnapshot.capture(shape, frameToken: 51)
        }

        shape.lineDashPattern = nil
        shape.lineCap = CAShapeLayerLineCap(rawValue: "future")
        #expect(throws: CARendererError.invalidLayerShape(
            .unsupportedLineCap("future")
        )) {
            try CARenderSnapshot.capture(shape, frameToken: 52)
        }
    }

    @Test("Text layout and style become immutable snapshot values")
    func textValuesUseSnapshots() throws {
        let text = CATextLayer()
        text.bounds = CGRect(x: 2, y: 3, width: 80, height: 24)
        text.string = "Committed"
        text.font = "Snapshot Sans"
        text.fontSize = 18
        text.contentsScale = 2
        text.foregroundColor = CGColor(
            red: 0,
            green: 1,
            blue: 1,
            alpha: 0.5
        )
        text.alignmentMode = .center
        text.truncationMode = .middle
        text.isWrapped = true

        let snapshot = try CARenderSnapshot.capture(
            text,
            frameToken: 53
        )
        let values = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues
        let capturedText = try #require(values.text)
        let configuration = try #require(
            capturedText.configuration
        )
        #expect(configuration.text == "Committed")
        #expect(configuration.fontFamily == "Snapshot Sans")
        #expect(configuration.fontSize == 18)
        #expect(configuration.contentsScale == 2)
        #expect(
            configuration.foregroundRGBA
                == SIMD4<Float>(0, 1, 1, 0.5)
        )
        #expect(configuration.bounds == text.bounds)
        #expect(configuration.alignmentMode == .center)
        #expect(configuration.truncationMode == .middle)
        #expect(configuration.isWrapped)

        text.string = "Mutated"
        text.font = "Other"
        text.fontSize = 30
        text.contentsScale = 1
        text.foregroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        text.alignmentMode = .right
        text.truncationMode = .end
        text.isWrapped = false

        #expect(configuration.text == "Committed")
        #expect(configuration.fontFamily == "Snapshot Sans")
        #expect(configuration.fontSize == 18)
        #expect(configuration.contentsScale == 2)
        #expect(
            configuration.foregroundRGBA
                == SIMD4<Float>(0, 1, 1, 0.5)
        )
        #expect(configuration.alignmentMode == .center)
        #expect(configuration.truncationMode == .middle)
        #expect(configuration.isWrapped)
    }

    @Test("An empty text foreground does not fall back to image contents")
    func emptyTextDoesNotFallBackToImageContents() throws {
        let text = CATextLayer()
        text.contents = SnapshotContentsToken()

        let snapshot = try CARenderSnapshot.capture(
            text,
            frameToken: 54
        )
        let values = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues
        let capturedText = try #require(values.text)
        #expect(capturedText.configuration == nil)
        #expect(values.imageContents == nil)
    }

    @Test("Invalid text values fail immutable capture with exact reasons")
    func invalidTextFailsCapture() {
        let text = CATextLayer()
        text.string = 42

        #expect(throws: CARendererError.invalidLayerText(
            .unsupportedStringValue
        )) {
            try CARenderSnapshot.capture(text, frameToken: 55)
        }

        text.string = "Text"
        text.font = 42
        #expect(throws: CARendererError.invalidLayerText(
            .unsupportedFontValue
        )) {
            try CARenderSnapshot.capture(text, frameToken: 56)
        }

        text.font = "sans-serif"
        text.alignmentMode = CATextLayerAlignmentMode(
            rawValue: "future-alignment"
        )
        #expect(throws: CARendererError.invalidLayerText(
            .unsupportedAlignmentMode("future-alignment")
        )) {
            try CARenderSnapshot.capture(text, frameToken: 57)
        }
    }

    @Test("Detached mask trees become value-owned snapshot nodes")
    func maskTreeIsCapturedByValue() throws {
        CATransaction.flush()
        let root = CALayer()
        let mask = CALayer()

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        mask.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        mask.backgroundColor = CGColor(
            red: 1,
            green: 1,
            blue: 1,
            alpha: 0.5
        )
        root.mask = mask
        CATransaction.commit()

        guard case .snapshot(let snapshot) =
                root.pendingCommittedRenderState else {
            Issue.record("Expected the mask tree to publish a snapshot")
            return
        }
        let rootNode = snapshot.nodes[snapshot.rootIndex]
        let maskIndex = try #require(rootNode.maskIndex)
        #expect(snapshot.nodes[maskIndex].identity
            == ObjectIdentifier(mask))
        #expect(
            snapshot.nodes[maskIndex]
                .presentationValues.backgroundColor
                == SIMD4<Float>(1, 1, 1, 0.5)
        )

        mask.backgroundColor = CGColor(
            red: 0,
            green: 0,
            blue: 0,
            alpha: 1
        )
        #expect(
            snapshot.nodes[maskIndex]
                .presentationValues.backgroundColor
                == SIMD4<Float>(1, 1, 1, 0.5)
        )
    }

    @Test("Rasterization policy becomes immutable snapshot values")
    func rasterizationIsCapturedByValue() throws {
        let layer = CALayer()
        layer.shouldRasterize = true
        layer.rasterizationScale = 2.5

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 46
        )
        let values = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues
        #expect(values.shouldRasterize)
        #expect(values.rasterizationScale == 2.5)

        layer.shouldRasterize = false
        layer.rasterizationScale = 1
        #expect(values.shouldRasterize)
        #expect(values.rasterizationScale == 2.5)
    }

    @Test("Invalid rasterization scale remains renderer-visible")
    func invalidRasterizationScaleIsCapturedForRendererValidation() throws {
        let layer = CALayer()
        layer.shouldRasterize = true
        layer.rasterizationScale = 0

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 46
        )
        let values = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues

        #expect(values.shouldRasterize)
        #expect(values.rasterizationScale == 0)
    }

    @Test("Layer and mask filters become value-owned snapshot plans")
    func filtersAreCapturedByValue() throws {
        let root = CALayer()
        var rootFilter = CAFilter.brightness(0.25)
        root.filters = [rootFilter]
        let mask = CALayer()
        mask.filters = [CAFilter.blur(radius: 4)]
        root.mask = mask

        let snapshot = try CARenderSnapshot.capture(
            root,
            frameToken: 45
        )

        let rootNode = snapshot.nodes[snapshot.rootIndex]
        let maskIndex = try #require(rootNode.maskIndex)
        #expect(
            rootNode.presentationValues.filters
                == [.renderer(.brightness(amount: 0.25))]
        )
        #expect(
            snapshot.nodes[maskIndex].presentationValues.filters
                == [.renderer(.gaussianBlur(radius: 4))]
        )

        rootFilter.parameters["inputBrightness"] = -0.75
        root.filters = [rootFilter]
        mask.filters = [CAFilter.blur(radius: 12)]
        #expect(
            rootNode.presentationValues.filters
                == [.renderer(.brightness(amount: 0.25))]
        )
        #expect(
            snapshot.nodes[maskIndex].presentationValues.filters
                == [.renderer(.gaussianBlur(radius: 4))]
        )
    }

    @Test("Backdrop filters become value-owned snapshot plans")
    func backdropFiltersAreCapturedByValue() throws {
        let layer = CALayer()
        var backgroundFilter = CAFilter.brightness(0.25)
        layer.backgroundFilters = [backgroundFilter]

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 46
        )
        let values = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues
        #expect(values.compositingFilter == nil)
        #expect(
            values.backgroundFilters
                == [.renderer(.brightness(amount: 0.25))]
        )

        backgroundFilter.parameters["inputBrightness"] = -0.75
        layer.backgroundFilters = [backgroundFilter]
        #expect(
            values.backgroundFilters
                == [.renderer(.brightness(amount: 0.25))]
        )
    }

    @Test("Invalid filter plans fail snapshot capture explicitly")
    func invalidFilterCaptureIsTyped() {
        let layer = CALayer()
        layer.filters = [
            CAFilter(
                type: .brightness,
                parameters: ["inputBrightness": "invalid"]
            ),
        ]

        #expect(throws: CARendererError.invalidLayerFilter(
            .invalidConfiguration(
                .invalidParameterType("inputBrightness")
            )
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 46)
        }
    }

    @Test("Invalid backdrop plans retain their owning property")
    func invalidBackdropCaptureIsTyped() throws {
        let backgroundLayer = CALayer()
        backgroundLayer.backgroundFilters = [
            CAFilter(
                type: .brightness,
                parameters: ["inputBrightness": "invalid"]
            ),
        ]
        #expect(throws: CARendererError.invalidLayerBackgroundFilter(
            .invalidConfiguration(
                .invalidParameterType("inputBrightness")
            )
        )) {
            try CARenderSnapshot.capture(
                backgroundLayer,
                frameToken: 47
            )
        }

        let compositionLayer = CALayer()
        compositionLayer.compositingFilter = "invalid"
        let compositionSnapshot = try CARenderSnapshot.capture(
            compositionLayer,
            frameToken: 48
        )
        let compositionValues = compositionSnapshot.nodes[
            compositionSnapshot.rootIndex
        ].presentationValues
        #expect(compositionValues.compositingFilter == nil)
        #expect(
            compositionValues.compositingFilterCaptureFailure
                == .unsupportedCompositingFilterValue(
                    "Swift.String"
                )
        )
    }

    @Test("Group opacity becomes value-owned snapshot state")
    func groupOpacityIsCapturedByValue() throws {
        let leaf = CALayer()
        leaf.opacity = 0.5
        _ = try CARenderSnapshot.capture(
            leaf,
            frameToken: 46
        )

        let distributedRoot = CALayer()
        distributedRoot.opacity = 0.5
        distributedRoot.allowsGroupOpacity = false
        distributedRoot.addSublayer(CALayer())
        _ = try CARenderSnapshot.capture(
            distributedRoot,
            frameToken: 47
        )

        let groupedRoot = CALayer()
        groupedRoot.opacity = 0.5
        groupedRoot.addSublayer(CALayer())
        let groupedSnapshot = try CARenderSnapshot.capture(
            groupedRoot,
            frameToken: 48
        )
        let groupedValues = groupedSnapshot.nodes[
            groupedSnapshot.rootIndex
        ].presentationValues
        #expect(groupedValues.allowsGroupOpacity)
        #expect(groupedValues.opacity == 0.5)
        #expect(
            groupedSnapshot.nodes[groupedSnapshot.rootIndex]
                .childIndices.count == 1
        )
    }

    @Test("Static shadow values and paths are captured by value")
    func shadowIsCapturedByValue() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 20, height: 10)
        layer.shadowColor = CGColor(
            red: 0.25,
            green: 0.5,
            blue: 0.75,
            alpha: 0.8
        )
        layer.shadowOpacity = 0.6
        layer.shadowRadius = 4
        layer.shadowOffset = CGSize(width: 3, height: -2)
        let path = CGMutablePath()
        path.addRect(CGRect(x: 1, y: 2, width: 8, height: 4))
        layer.shadowPath = path

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 49
        )
        let shadow = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.shadow
        )
        #expect(shadow.color == SIMD4<Float>(0.25, 0.5, 0.75, 0.8))
        #expect(shadow.opacity == 0.6)
        #expect(shadow.radius == 4)
        #expect(shadow.offset == SIMD2<Float>(3, -2))
        #expect(shadow.pathVertices?.count == 6)

        path.addRect(CGRect(x: 0, y: 0, width: 20, height: 10))
        layer.shadowColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        layer.shadowOpacity = 1
        #expect(shadow.color == SIMD4<Float>(0.25, 0.5, 0.75, 0.8))
        #expect(shadow.opacity == 0.6)
        #expect(shadow.pathVertices?.count == 6)
    }

    @Test("Invalid visible shadows fail snapshot capture explicitly")
    func invalidShadowCaptureIsTyped() {
        let layer = CALayer()
        layer.shadowOpacity = 1
        layer.shadowRadius = .nan

        #expect(throws: CARendererError.invalidLayerShadow(
            .nonFiniteGeometry
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 50)
        }
    }

    @Test("Invalid shadow composite opacity is typed at commit capture")
    func invalidShadowCompositeOpacityIsTyped() {
        let layer = CALayer()
        layer.shadowOpacity = 1
        layer.opacity = .infinity

        #expect(throws: CARendererError.invalidLayerShadow(
            .invalidCompositeOpacity(.infinity)
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 51)
        }
    }

    @Test("CGImage contents become value-owned commit resources")
    func imageContentsAreCapturedByValue() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 4)
        layer.contentsRect = CGRect(x: 0.25, y: 0, width: 0.5, height: 1)
        layer.contentsCenter = CGRect(
            x: 0.2,
            y: 0.3,
            width: 0.4,
            height: 0.5
        )
        layer.contentsScale = 2
        layer.contentsGravity = .resizeAspect
        layer.magnificationFilter = .nearest
        layer.minificationFilter = .trilinear
        layer.minificationFilterBias = 20
        layer.isOpaque = true

        weak var sourceImage: CGImage?
        let snapshot: CARenderSnapshot
        do {
            let image = try makeImage(
                width: 2,
                height: 1,
                pixels: [255, 0, 0, 255, 0, 255, 0, 255]
            )
            sourceImage = image
            layer.contents = image
            snapshot = try CARenderSnapshot.capture(
                layer,
                frameToken: 49
            )
            layer.contents = nil
        }
        layer.removeAllAnimations()
        CATransaction.flush()

        #expect(sourceImage == nil)
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )
        #expect(contents.storage.format == .rgba8Unorm)
        #expect(contents.storage.width == 2)
        #expect(contents.storage.height == 1)
        #expect(contents.storage.data == Data([
            255, 0, 0, 255,
            0, 255, 0, 255,
        ]))
        #expect(contents.contentsRect == layer.contentsRect)
        #expect(contents.contentsCenter == layer.contentsCenter)
        #expect(contents.contentsScale == 2)
        #expect(contents.gravity == .resizeAspect)
        #expect(contents.sampling == .nearestTrilinear)
        #expect(contents.minificationFilterBias == 15.99)
        #expect(contents.isOpaque)
        requireSendable(snapshot)
    }

    @Test("Captured image state does not follow later layer mutations")
    func capturedImageStateIsImmutable() throws {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        layer.contents = try makeImage(
            width: 2,
            height: 1,
            pixels: [255, 0, 0, 255, 0, 255, 0, 255]
        )
        layer.magnificationFilter = .nearest
        layer.minificationFilter = .nearest

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 50
        )
        layer.contents = try makeImage(
            width: 2,
            height: 1,
            pixels: [0, 0, 255, 255, 255, 255, 255, 255]
        )
        layer.contentsRect = CGRect(x: 0.5, y: 0, width: 0.5, height: 1)
        layer.magnificationFilter = .linear

        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )
        #expect(contents.storage.data == Data([
            255, 0, 0, 255,
            0, 255, 0, 255,
        ]))
        #expect(contents.contentsRect == CGRect(
            x: 0,
            y: 0,
            width: 1,
            height: 1
        ))
        #expect(contents.sampling == .nearestNearest)
    }

    @Test("Unknown image sampling filters fail capture explicitly")
    func invalidImageSamplingFailsCapture() throws {
        let layer = CALayer()
        layer.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 255, 255]
        )
        layer.magnificationFilter = CALayerContentsFilter(
            rawValue: "future"
        )

        #expect(throws: CARendererError.invalidLayerContents(
            .invalidSamplingFilters(
                magnification: CALayerContentsFilter(rawValue: "future"),
                minification: .linear
            )
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 51)
        }
    }

    @Test(
        "Image sampling captures every supported filter combination",
        arguments: [
            (
                CALayerContentsFilter.nearest,
                CALayerContentsFilter.nearest,
                CAContentsSampling.nearestNearest
            ),
            (
                CALayerContentsFilter.nearest,
                CALayerContentsFilter.linear,
                CAContentsSampling.nearestLinear
            ),
            (
                CALayerContentsFilter.nearest,
                CALayerContentsFilter.trilinear,
                CAContentsSampling.nearestTrilinear
            ),
            (
                CALayerContentsFilter.linear,
                CALayerContentsFilter.nearest,
                CAContentsSampling.linearNearest
            ),
            (
                CALayerContentsFilter.linear,
                CALayerContentsFilter.linear,
                CAContentsSampling.linearLinear
            ),
            (
                CALayerContentsFilter.trilinear,
                CALayerContentsFilter.trilinear,
                CAContentsSampling.linearTrilinear
            ),
        ]
    )
    func imageSamplingCapturesSupportedFilters(
        magnification: CALayerContentsFilter,
        minification: CALayerContentsFilter,
        expected: CAContentsSampling
    ) throws {
        let layer = CALayer()
        layer.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 255, 255]
        )
        layer.magnificationFilter = magnification
        layer.minificationFilter = minification

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 53
        )
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )

        #expect(contents.sampling == expected)
        #expect(contents.sampling.usesMipmaps == (minification == .trilinear))
    }

    @Test("Non-finite image mip bias fails capture explicitly")
    func nonFiniteImageMipBiasFailsCapture() throws {
        let layer = CALayer()
        layer.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 255, 255]
        )
        layer.minificationFilterBias = .infinity

        #expect(throws: CARendererError.invalidLayerContents(
            .invalidMinificationFilterBias(.infinity)
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 54)
        }
    }

    @Test("Non-image contents fail instead of becoming an empty live draw")
    func nonImageContentsFailCapture() {
        let layer = CALayer()
        layer.contents = SnapshotContentsToken()

        #expect(throws: CARendererError.invalidLayerContents(
            .unsupportedContentsType(
                String(reflecting: SnapshotContentsToken.self)
            )
        )) {
            _ = try CARenderSnapshot.capture(
                layer,
                frameToken: 55
            )
        }
    }

    @Test("Delegate bitmap supersedes unsupported model contents")
    func delegateBitmapSupersedesUnsupportedContents() throws {
        let delegate = SnapshotDrawingDelegate()
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        layer.contents = SnapshotContentsToken()
        layer.delegate = delegate
        layer.setNeedsDisplay()

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 56
        )
        #expect(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents != nil
        )
    }

    @Test("Delegate drawing becomes a value-owned image snapshot")
    func delegateDrawingBecomesImageSnapshot() throws {
        let delegate = SnapshotDrawingDelegate()
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        layer.delegate = delegate
        layer.setNeedsDisplay()

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 56
        )
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )
        #expect(contents.origin == .delegateBackingStore(.RGBA8Uint))
        #expect(contents.storage.data == Data([
            255, 0, 0, 255,
            0, 255, 0, 255,
        ]))
        #expect(delegate.willDrawCount == 1)
        #expect(delegate.drawCount == 1)
        #expect(!layer.needsDisplay())

        let repeatedSnapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 57
        )
        #expect(
            repeatedSnapshot.nodes[repeatedSnapshot.rootIndex]
                .presentationValues.imageContents?.origin
                == .delegateBackingStore(.RGBA8Uint)
        )
        #expect(delegate.drawCount == 1)
    }

    @Test("Partial delegate redraw preserves untouched committed pixels")
    func partialDelegateRedrawPreservesPixels() throws {
        let delegate = SnapshotDrawingDelegate()
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        layer.delegate = delegate
        layer.setNeedsDisplay()
        _ = try CARenderSnapshot.capture(layer, frameToken: 58)

        delegate.leftColor = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        layer.setNeedsDisplay(CGRect(x: 0, y: 0, width: 1, height: 1))
        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 59
        )
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )

        #expect(contents.storage.data == Data([
            0, 0, 255, 255,
            0, 255, 0, 255,
        ]))
        #expect(delegate.drawCount == 2)
    }

    @Test("Delegate display contents supersede the software backing store")
    func delegateDisplayContentsTakePriority() throws {
        let image = try makeImage(
            width: 1,
            height: 1,
            pixels: [0, 0, 255, 255]
        )
        let delegate = SnapshotDisplayDelegate(image: image)
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
        layer.delegate = delegate
        layer.setNeedsDisplay()

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 60
        )
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )

        #expect(contents.origin == .layerContents)
        #expect(contents.storage.data == Data([0, 0, 255, 255]))
        #expect(delegate.displayCount == 1)
        #expect(delegate.drawCount == 0)
        #expect(layer.delegateBackingStore == nil)
    }

    @Test("Delegate display re-invalidation remains pending after current drawing")
    func delegateDisplayReinvalidationRemainsPending() throws {
        let delegate = SnapshotDrawingDelegate()
        delegate.invalidatesDuringDisplay = true
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        layer.delegate = delegate
        layer.setNeedsDisplay()

        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 61
        )
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )

        #expect(contents.origin == .delegateBackingStore(.RGBA8Uint))
        #expect(delegate.drawCount == 1)
        #expect(layer.needsDisplay())
    }

    @Test("Invalid delegate geometry fails snapshot capture explicitly")
    func invalidDelegateGeometryFailsCapture() throws {
        let delegate = SnapshotDrawingDelegate()
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
        layer.contentsScale = .infinity
        layer.delegate = delegate
        layer.setNeedsDisplay()

        #expect(throws: CARendererError.invalidDelegateBackingStore(
            .invalidGeometry
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 61)
        }
        #expect(layer.delegateBackingStore == nil)
        #expect(layer.needsDisplay())

        layer.contentsScale = 1
        _ = try CARenderSnapshot.capture(layer, frameToken: 62)
        #expect(layer.delegateBackingStore != nil)
        #expect(!layer.needsDisplay())
    }

    @Test("Delegate storage overflow fails before bitmap allocation")
    func delegateStorageOverflowFailsBeforeAllocation() {
        let delegate = SnapshotDrawingDelegate()
        let overflowingWidth = 1 << (Int.bitWidth - 5)
        let layer = CALayer()
        layer.bounds = CGRect(
            x: 0,
            y: 0,
            width: CGFloat(overflowingWidth),
            height: 1
        )
        layer.delegate = delegate
        layer.setNeedsDisplay()

        #expect(throws: CARendererError.invalidDelegateBackingStore(
            .pixelStorageSizeOverflow(
                width: overflowingWidth,
                height: 1,
                bitsPerPixel: 32
            )
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 62)
        }
        #expect(layer.delegateBackingStore == nil)
    }

    @Test("Unknown delegate storage format fails snapshot capture explicitly")
    func unknownDelegateStorageFailsCapture() {
        let delegate = SnapshotDrawingDelegate()
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
        layer.contentsFormat = CALayerContentsFormat(rawValue: "future")
        layer.delegate = delegate
        layer.setNeedsDisplay()

        #expect(throws: CARendererError.invalidDelegateBackingStore(
            .unsupportedContentsFormat("future")
        )) {
            try CARenderSnapshot.capture(layer, frameToken: 62)
        }
        #expect(layer.delegateBackingStore == nil)
    }

    @Test("Explicit contents release an older delegate backing store")
    func explicitContentsReleaseDelegateBackingStore() throws {
        let delegate = SnapshotDrawingDelegate()
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
        layer.delegate = delegate
        layer.setNeedsDisplay()
        _ = try CARenderSnapshot.capture(layer, frameToken: 63)
        #expect(layer.delegateBackingStore != nil)

        layer.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 255, 0, 255]
        )
        let snapshot = try CARenderSnapshot.capture(
            layer,
            frameToken: 64
        )
        let contents = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )

        #expect(layer.delegateBackingStore == nil)
        #expect(contents.origin == .layerContents)
        #expect(contents.storage.data == Data([255, 255, 0, 255]))
    }

    @Test("Animated commits publish a model-independent evaluator")
    func animatedCommitPublishesIndependentEvaluator() throws {
        CATransaction.flush()
        let root = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.25
        CATransaction.flush()

        CATransaction.begin()
        root.add(animation, forKey: "opacity")
        CATransaction.commit()

        guard case .animationEvaluator(
            _,
            let evaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record("Expected an immutable animation evaluator")
            return
        }
        let committed = try evaluator.snapshot(frameToken: 80)
        let committedValues = committed.nodes[
            committed.rootIndex
        ].presentationValues
        #expect(abs(committedValues.opacity - 0.25) < 0.0001)

        root.opacity = 0.9
        let stored = try #require(
            root.animation(forKey: "opacity")
                as? CABasicAnimation
        )
        stored.fromValue = Float(1)
        stored.toValue = Float(0)
        stored.timeOffset = 0.75

        let afterMutation = try evaluator.snapshot(
            frameToken: 81
        )
        let afterMutationValues = afterMutation.nodes[
            afterMutation.rootIndex
        ].presentationValues
        #expect(
            afterMutationValues.opacity
                == committedValues.opacity
        )
        #expect(
            afterMutation.nodes[afterMutation.rootIndex].identity
                == ObjectIdentifier(root)
        )
        root.removeAllAnimations()
    }

    @Test("Committed evaluator retains an initially hidden animated shadow")
    func animatedShadowRetainsHiddenModelValues() throws {
        CATransaction.flush()
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
        root.shadowColor = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        root.shadowOpacity = 0
        root.shadowRadius = 4
        root.shadowOffset = CGSize(width: 8, height: 2)
        CATransaction.flush()

        let animation = CABasicAnimation(keyPath: "shadowOpacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false

        CATransaction.begin()
        root.add(animation, forKey: "shadowOpacity")
        CATransaction.commit()

        guard case .animationEvaluator(
            _,
            let evaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record("Expected a committed shadow evaluator")
            return
        }
        let snapshot = try evaluator.snapshot(frameToken: 82)
        let shadow = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.shadow
        )

        #expect(abs(shadow.opacity - 0.5) < 0.0001)
        #expect(shadow.color == SIMD4<Float>(0, 0, 1, 1))
        #expect(shadow.radius == 4)
        #expect(shadow.offset == SIMD2<Float>(8, 2))
        root.removeAllAnimations()
    }

    @Test("Committed evaluator preserves invalid replicator model values")
    func committedEvaluatorPreservesInvalidReplicatorValues() throws {
        CATransaction.flush()
        let replicator = CAReplicatorLayer()
        replicator.instanceCount = 2
        replicator.instanceDelay = .nan
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.5
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false
        CATransaction.flush()

        CATransaction.begin()
        replicator.add(animation, forKey: "opacity")
        CATransaction.commit()

        guard case .animationEvaluator(
            let frameToken,
            let evaluator
        ) = replicator.pendingCommittedRenderState else {
            Issue.record("Expected a committed replicator evaluator")
            return
        }
        let snapshot = try evaluator.snapshot(
            frameToken: frameToken
        )
        let values = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues

        #expect(
            values.replicatorCaptureFailure
                == .nonFiniteInstanceDelay
        )
        #expect(values.replicator == nil)
        replicator.removeAllAnimations()
        CATransaction.flush()
    }

    @Test("Mutating a stored animation republishes the committed evaluator")
    func storedAnimationMutationRepublishesEvaluator() throws {
        CATransaction.flush()
        let root = CALayer()
        let child = CALayer()
        child.position = CGPoint(x: 20, y: 25)
        root.addSublayer(child)
        CATransaction.flush()

        let animation = CABasicAnimation(keyPath: "position")
        animation.fromValue = CGPoint(x: 20, y: 25)
        animation.toValue = CGPoint(x: 60, y: 25)
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0
        animation.fillMode = .both
        animation.isRemovedOnCompletion = false

        CATransaction.begin()
        child.add(animation, forKey: "position")
        CATransaction.commit()

        guard case .animationEvaluator(
            let firstToken,
            let firstEvaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record("Expected the first committed evaluator")
            return
        }
        let firstSnapshot = try firstEvaluator.snapshot(
            frameToken: firstToken
        )
        let firstNode = try #require(
            firstSnapshot.nodes.first {
                $0.identity == ObjectIdentifier(child)
            }
        )
        #expect(firstNode.presentationValues.position.x == 20)

        let stored = try #require(
            child.animation(forKey: "position")
        )
        stored.timeOffset = 1
        CATransaction.commitPendingImplicitTransactions()

        guard case .animationEvaluator(
            let secondToken,
            let secondEvaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record("Expected the updated committed evaluator")
            return
        }
        #expect(secondToken != firstToken)
        let secondSnapshot = try secondEvaluator.snapshot(
            frameToken: secondToken
        )
        let secondNode = try #require(
            secondSnapshot.nodes.first {
                $0.identity == ObjectIdentifier(child)
            }
        )
        #expect(secondNode.presentationValues.position.x == 60)
        child.removeAllAnimations()
    }

    @Test("Removing a detached mask animation republishes static mask state")
    func detachedMaskAnimationRemovalRepublishesState() throws {
        CATransaction.flush()
        let root = CALayer()
        let animatedSibling = CALayer()
        let keeper = CABasicAnimation(keyPath: "opacity")
        keeper.fromValue = Float(1)
        keeper.toValue = Float(1)
        keeper.duration = 1
        keeper.speed = 0
        keeper.fillMode = .both
        keeper.isRemovedOnCompletion = false
        animatedSibling.add(keeper, forKey: "keeper")

        let masked = CALayer()
        let maskRoot = CALayer()
        let maskChild = CALayer()
        maskChild.backgroundColor = CGColor(
            red: 1,
            green: 1,
            blue: 1,
            alpha: 1
        )
        let transition = CATransition()
        transition.duration = 1
        transition.speed = 0
        transition.timeOffset = 0.5
        transition.fillMode = .both
        transition.isRemovedOnCompletion = false
        maskChild.add(transition, forKey: "fade")
        maskChild.backgroundColor = CGColor(
            red: 0,
            green: 0,
            blue: 0,
            alpha: 0
        )
        maskRoot.addSublayer(maskChild)
        masked.mask = maskRoot
        root.addSublayer(animatedSibling)
        root.addSublayer(masked)
        CATransaction.flush()

        guard case .animationEvaluator(
            let firstToken,
            let firstEvaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record("Expected the first mask evaluator")
            return
        }
        let firstSnapshot = try firstEvaluator.snapshot(
            frameToken: firstToken
        )
        let firstMaskChild = try #require(
            firstSnapshot.nodes.first {
                $0.identity == ObjectIdentifier(maskChild)
            }
        )
        #expect(firstMaskChild.presentationValues.transition != nil)

        maskChild.removeAnimation(forKey: "fade")
        maskChild.backgroundColor = CGColor(
            red: 1,
            green: 1,
            blue: 1,
            alpha: 1
        )
        CATransaction.commitPendingImplicitTransactions()

        guard case .animationEvaluator(
            let secondToken,
            let secondEvaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record("Expected the updated mask evaluator")
            return
        }
        #expect(secondToken != firstToken)
        let secondSnapshot = try secondEvaluator.snapshot(
            frameToken: secondToken
        )
        let secondMaskChild = try #require(
            secondSnapshot.nodes.first {
                $0.identity == ObjectIdentifier(maskChild)
            }
        )
        #expect(secondMaskChild.presentationValues.transition == nil)
        #expect(
            secondMaskChild.presentationValues.backgroundColor
                == SIMD4<Float>(1, 1, 1, 1)
        )
        animatedSibling.removeAllAnimations()
    }

    @Test("Animated tiled commits retain the committed cache contract")
    func animatedTiledCommitRetainsCommittedConfiguration() throws {
        CATransaction.flush()
        let root = CALayer()
        let tiled = CATiledLayer()
        let provider = SnapshotTileProvider()
        tiled.bounds = CGRect(x: 0, y: 0, width: 32, height: 32)
        tiled.tileSize = CGSize(width: 16, height: 16)
        tiled.delegate = provider
        root.addSublayer(tiled)
        CATransaction.flush()

        let animation = CABasicAnimation(keyPath: "position.x")
        animation.fromValue = CGFloat(0)
        animation.toValue = CGFloat(0)
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.25

        CATransaction.begin()
        tiled.add(animation, forKey: "position.x")
        CATransaction.commit()

        guard case .animationEvaluator(
            _,
            let evaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record("Expected an immutable animation evaluator")
            return
        }

        let first = try evaluator.snapshot(frameToken: 82)
        let second = try evaluator.snapshot(frameToken: 83)
        let firstChildIndex = try #require(
            first.nodes[first.rootIndex].childIndices.first
        )
        let secondChildIndex = try #require(
            second.nodes[second.rootIndex].childIndices.first
        )
        let firstConfiguration = try #require(
            first.nodes[firstChildIndex].presentationValues.tiled
        )
        let secondConfiguration = try #require(
            second.nodes[secondChildIndex].presentationValues.tiled
        )

        #expect(
            firstConfiguration.resourceIdentity
                == tiled.resourceIdentity
        )
        #expect(
            secondConfiguration.resourceIdentity
                == firstConfiguration.resourceIdentity
        )
        #expect(
            secondConfiguration.cacheGeneration
                == firstConfiguration.cacheGeneration
        )
        #expect(
            secondConfiguration.capturedContent?.identity
                == firstConfiguration.capturedContent?.identity
        )
        tiled.removeAllAnimations()
    }

    @Test("Committed contents animation owns its model and endpoint pixels")
    func committedContentsAnimationOwnsPixels() throws {
        CATransaction.flush()
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
        root.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 0, 0, 255]
        )
        let animation = CABasicAnimation(keyPath: "contents")
        animation.toValue = try makeImage(
            width: 1,
            height: 1,
            pixels: [0, 0, 255, 255]
        )
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.25
        CATransaction.flush()

        CATransaction.begin()
        root.add(animation, forKey: "contents")
        CATransaction.commit()

        guard case .animationEvaluator(
            _,
            let evaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record(
                "Expected a committed animation evaluator, got \(String(describing: root.pendingCommittedRenderState))"
            )
            return
        }
        root.contents = try makeImage(
            width: 1,
            height: 1,
            pixels: [0, 255, 0, 255]
        )
        let stored = try #require(
            root.animation(forKey: "contents")
                as? CABasicAnimation
        )
        stored.toValue = root.contents
        stored.timeOffset = 0.75

        let snapshot = try evaluator.snapshot(frameToken: 82)
        let image = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )
        #expect(image.storage.data == Data([255, 0, 0, 255]))
        root.removeAllAnimations()
    }

    @Test("Committed keyframe contents own every endpoint image")
    func committedKeyframeContentsOwnPixels() throws {
        CATransaction.flush()
        let red = try makeImage(
            width: 1,
            height: 1,
            pixels: [255, 0, 0, 255]
        )
        let blue = try makeImage(
            width: 1,
            height: 1,
            pixels: [0, 0, 255, 255]
        )
        let green = try makeImage(
            width: 1,
            height: 1,
            pixels: [0, 255, 0, 255]
        )
        let root = CALayer()
        root.bounds = CGRect(
            x: 0,
            y: 0,
            width: 1,
            height: 1
        )
        root.contents = green
        let animation = CAKeyframeAnimation(
            keyPath: "contents"
        )
        animation.values = [red, blue]
        animation.calculationMode = .discrete
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.25
        CATransaction.flush()

        CATransaction.begin()
        root.add(animation, forKey: "contents")
        CATransaction.commit()

        guard case .animationEvaluator(
            _,
            let evaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record(
                "Expected a committed animation evaluator"
            )
            return
        }
        let stored = try #require(
            root.animation(forKey: "contents")
                as? CAKeyframeAnimation
        )
        stored.values = [green, green]
        stored.timeOffset = 0.75
        root.contents = green

        let snapshot = try evaluator.snapshot(
            frameToken: 83
        )
        let image = try #require(
            snapshot.nodes[snapshot.rootIndex]
                .presentationValues.imageContents
        )
        #expect(
            image.storage.data
                == Data([255, 0, 0, 255])
        )
        root.removeAllAnimations()
    }

    @Test("Committed nested groups own their child graphs")
    func committedNestedGroupsOwnChildren() throws {
        CATransaction.flush()
        let root = CALayer()
        root.position = .zero
        let child = CABasicAnimation(
            keyPath: "position"
        )
        child.fromValue = CGPoint.zero
        child.toValue = CGPoint(x: 100, y: 40)
        child.duration = 1
        let inner = CAAnimationGroup()
        inner.animations = [child]
        inner.duration = 1
        let outer = CAAnimationGroup()
        outer.animations = [inner]
        outer.duration = 1
        outer.speed = 0
        outer.timeOffset = 0.25
        CATransaction.flush()

        CATransaction.begin()
        root.add(outer, forKey: "nested")
        CATransaction.commit()

        guard case .animationEvaluator(
            _,
            let evaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record(
                "Expected a committed animation evaluator"
            )
            return
        }
        let storedOuter = try #require(
            root.animation(forKey: "nested")
                as? CAAnimationGroup
        )
        let storedInner = try #require(
            storedOuter.animations?.first
                as? CAAnimationGroup
        )
        let storedChild = try #require(
            storedInner.animations?.first
                as? CABasicAnimation
        )
        storedChild.toValue = CGPoint(x: 400, y: 400)
        storedOuter.timeOffset = 0.75
        root.position = CGPoint(x: 900, y: 900)

        let snapshot = try evaluator.snapshot(
            frameToken: 84
        )
        let position = snapshot.nodes[
            snapshot.rootIndex
        ].presentationValues.position
        #expect(abs(position.x - 25) < 0.0001)
        #expect(abs(position.y - 10) < 0.0001)
        root.removeAllAnimations()
    }

    @Test("Committed evaluator serializes concurrent frame capture")
    func committedEvaluatorSerializesCapture() async throws {
        CATransaction.flush()
        let root = CALayer()
        root.opacity = 1
        let animation = CABasicAnimation(
            keyPath: "opacity"
        )
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.375
        CATransaction.flush()

        CATransaction.begin()
        root.add(animation, forKey: "opacity")
        CATransaction.commit()

        guard case .animationEvaluator(
            _,
            let evaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record(
                "Expected a committed animation evaluator"
            )
            return
        }
        let opacities = try await withThrowingTaskGroup(
            of: Float.self,
            returning: [Float].self
        ) { group in
            for token in UInt64(100)..<UInt64(116) {
                group.addTask {
                    let snapshot = try evaluator.snapshot(
                        frameToken: token
                    )
                    return snapshot.nodes[
                        snapshot.rootIndex
                    ].presentationValues.opacity
                }
            }
            var values: [Float] = []
            for try await value in group {
                values.append(value)
            }
            return values
        }
        #expect(opacities.count == 16)
        #expect(
            opacities.allSatisfy {
                abs($0 - 0.375) < 0.0001
            }
        )
        root.removeAllAnimations()
    }

    @Test("Committed transition owns source, target, and timing")
    func committedTransitionOwnsParticipants() throws {
        CATransaction.flush()
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
        root.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        let transition = CATransition()
        transition.duration = 1
        transition.speed = 0
        transition.timeOffset = 0.25
        CATransaction.flush()

        CATransaction.begin()
        root.add(transition, forKey: "transition")
        root.backgroundColor = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        CATransaction.commit()

        guard case .animationEvaluator(
            _,
            let evaluator
        ) = root.pendingCommittedRenderState else {
            Issue.record(
                "Expected a committed animation evaluator, got \(String(describing: root.pendingCommittedRenderState))"
            )
            return
        }
        root.backgroundColor = CGColor(
            red: 0,
            green: 1,
            blue: 0,
            alpha: 1
        )
        let stored = try #require(
            root.animation(forKey: "transition")
                as? CATransition
        )
        stored.timeOffset = 0.75

        let snapshot = try evaluator.snapshot(frameToken: 83)
        let target = snapshot.nodes[snapshot.rootIndex]
        let capturedTransition = try #require(
            target.presentationValues.transition
        )
        let source = snapshot.nodes[
            capturedTransition.sourceRootIndex
        ]
        #expect(
            source.presentationValues.backgroundColor
                == SIMD4<Float>(1, 0, 0, 1)
        )
        #expect(
            target.presentationValues.backgroundColor
                == SIMD4<Float>(0, 0, 1, 1)
        )
        #expect(
            abs(capturedTransition.progress - 0.25)
                < 0.0001
        )
        root.removeAllAnimations()
    }

    @Test("Unsupported animation endpoints publish a typed failure")
    func unsupportedAnimationEndpointFailsCommit() {
        CATransaction.flush()
        let root = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = "unsupported"
        animation.toValue = Float(1)
        animation.duration = 1

        CATransaction.begin()
        root.add(animation, forKey: "opacity")
        CATransaction.commit()

        guard case .captureFailure(
            _,
            let error
        ) = root.pendingCommittedRenderState else {
            Issue.record("Expected a typed committed capture failure")
            return
        }
        #expect(
            error == .invalidCommittedAnimation(
                .unsupportedValueType("Swift.String")
            )
        )
        root.removeAllAnimations()
    }

    @Test("Non-finite animation endpoints publish a typed failure")
    func nonFiniteAnimationEndpointFailsCommit() {
        CATransaction.flush()
        let root = CALayer()
        let animation = CABasicAnimation(
            keyPath: "position"
        )
        animation.fromValue = CGPoint(
            x: CGFloat.infinity,
            y: 0
        )
        animation.toValue = CGPoint(x: 1, y: 1)
        animation.duration = 1

        CATransaction.begin()
        root.add(animation, forKey: "position")
        CATransaction.commit()

        guard case .captureFailure(
            _,
            let error
        ) = root.pendingCommittedRenderState else {
            Issue.record(
                "Expected a typed committed capture failure"
            )
            return
        }
        #expect(
            error == .invalidCommittedAnimation(
                .nonFiniteValue("CGPoint")
            )
        )
        root.removeAllAnimations()
    }

    @Test("Layout-pending commits publish post-layout snapshot geometry")
    func layoutPendingCommitPublishesPostLayoutSnapshot() throws {
        CATransaction.flush()
        let root = CALayer()
        let child = CALayer()
        let layoutManager = SnapshotLayoutManager { layer in
            layer.sublayers?.first?.bounds = CGRect(
                x: 0,
                y: 0,
                width: 12,
                height: 8
            )
            layer.sublayers?.first?.position = CGPoint(x: 7, y: 5)
        }
        root.layoutManager = layoutManager

        CATransaction.begin()
        root.addSublayer(child)
        CATransaction.commit()

        guard case .snapshot(let snapshot) =
                root.pendingCommittedRenderState else {
            Issue.record("Expected an immutable post-layout snapshot")
            return
        }
        let rootNode = snapshot.nodes[snapshot.rootIndex]
        let childIndex = try #require(rootNode.childIndices.first)
        let childValues = snapshot.nodes[childIndex].presentationValues
        #expect(childValues.bounds == CGRect(
            x: 0,
            y: 0,
            width: 12,
            height: 8
        ))
        #expect(childValues.position == SIMD3<Float>(7, 5, 0))
        #expect(layoutManager.layoutCount == 2)
        #expect(!root.needsLayout())
    }

    @Test("Capture failure is retained as a typed committed state")
    func commitRetainsCaptureFailure() {
        CATransaction.flush()
        let root = CALayer()
        let components = [CGFloat.nan, 0, 0, 1]
        let invalidColor: CGColor? = components.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return nil }
            return CGColor(colorSpace: .deviceRGB, components: baseAddress)
        }

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.backgroundColor = invalidColor
        CATransaction.commit()

        guard case .captureFailure(_, let error) = root.pendingCommittedRenderState else {
            Issue.record("Expected a committed capture failure")
            return
        }
        #expect(error == .invalidLayerBackgroundColor)
    }

    @Test("Captured values and hierarchy do not follow later model mutations")
    func captureIsIndependentFromModelTree() throws {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 4, height: 4)
        child.position = CGPoint(x: 2, y: 2)
        child.isDoubleSided = false
        root.addSublayer(child)

        let snapshot = try CARenderSnapshot.capture(root, frameToken: 41)
        requireSendable(snapshot)

        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        root.bounds = CGRect(x: 0, y: 0, width: 32, height: 32)
        child.isDoubleSided = true
        child.removeFromSuperlayer()

        let rootNode = snapshot.nodes[snapshot.rootIndex]
        #expect(snapshot.frameToken == 41)
        #expect(snapshot.nodes.count == 2)
        #expect(rootNode.childIndices == [1])
        #expect(rootNode.presentationValues.bounds == CGRect(
            x: 0,
            y: 0,
            width: 16,
            height: 16
        ))
        #expect(rootNode.presentationValues.backgroundColor == SIMD4<Float>(0, 1, 0, 1))
        #expect(!snapshot.nodes[1].presentationValues.isDoubleSided)
    }

    @Test("Gray colors are converted to device RGB instead of becoming transparent")
    func grayBackgroundColorIsConverted() throws {
        let root = CALayer()
        root.backgroundColor = CGColor(gray: 0.25, alpha: 0.75)

        let snapshot = try CARenderSnapshot.capture(root, frameToken: 42)
        let color = snapshot.nodes[snapshot.rootIndex].presentationValues.backgroundColor

        #expect(color == SIMD4<Float>(0.25, 0.25, 0.25, 0.75))
    }

    @Test("Non-finite colors fail capture explicitly")
    func nonFiniteBackgroundColorFailsCapture() {
        let root = CALayer()
        let components = [CGFloat.nan, 0, 0, 1]
        root.backgroundColor = components.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return nil }
            return CGColor(colorSpace: .deviceRGB, components: baseAddress)
        }

        #expect(throws: CARendererError.invalidLayerBackgroundColor) {
            try CARenderSnapshot.capture(root, frameToken: 43)
        }
    }

    @Test("Non-finite geometry fails capture explicitly")
    func nonFiniteGeometryFailsCapture() {
        let root = CALayer()
        root.position.x = .infinity

        #expect(throws: CARendererError.nonFiniteLayerGeometry) {
            try CARenderSnapshot.capture(root, frameToken: 44)
        }
    }

    @Test("Invalid corner geometry preserves its exact capture reason")
    func invalidCornerGeometryPreservesReason() {
        let root = CALayer()
        root.cornerRadius = -1

        #expect(throws: CARendererError.invalidLayerCornerGeometry(
            .negativeCornerRadius(-1)
        )) {
            try CARenderSnapshot.capture(root, frameToken: 45)
        }

        root.cornerRadius = 1
        root.cornerCurve = CALayerCornerCurve(
            rawValue: "future-curve"
        )
        #expect(throws: CARendererError.invalidLayerCornerGeometry(
            .unsupportedCurve("future-curve")
        )) {
            try CARenderSnapshot.capture(root, frameToken: 46)
        }
    }

    @Test("Revision snapshots preserve later layer and detached-mask mutations")
    func revisionSnapshotClearsOnlyCapturedMutations() throws {
        let root = CALayer()
        let child = CALayer()
        let mask = CALayer()
        root.addSublayer(child)
        root.mask = mask

        let submitted = try CARenderRevisionSnapshot.capture(root)
        root.recursivelyClearDirtyAfterCommit(matching: submitted)
        #expect(root._dirtyMask.isEmpty)
        #expect(child._dirtyMask.isEmpty)
        #expect(mask._dirtyMask.isEmpty)

        let nextSubmitted = try CARenderRevisionSnapshot.capture(root)
        child.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        mask.opacity = 0.5

        root.recursivelyClearDirtyAfterCommit(matching: nextSubmitted)
        #expect(root._dirtyMask.isEmpty)
        #expect(child._dirtyMask.isEmpty == false)
        #expect(mask._dirtyMask.isEmpty == false)
    }

    @Test("Revision capture rejects a cyclic detached-mask graph")
    func revisionSnapshotRejectsMaskCycles() {
        let root = CALayer()
        let mask = CALayer()
        root.mask = mask
        mask.mask = root

        #expect(throws: CARendererError.cyclicLayerHierarchy) {
            try CARenderRevisionSnapshot.capture(root)
        }

        mask.mask = nil
        root.mask = nil
        CATransaction.flush()
    }

    private func requireSendable<T: Sendable>(_ value: T) {
        _ = value
    }

    private func makeImage(
        width: Int,
        height: Int,
        pixels: [UInt8]
    ) throws -> CGImage {
        let bytesPerRow = width * 4
        #expect(pixels.count == bytesPerRow * height)
        return try #require(CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: .deviceRGB,
            bitmapInfo: CGBitmapInfo(
                rawValue: CGImageAlphaInfo.last.rawValue
            ),
            provider: CGDataProvider(data: Data(pixels)),
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ))
    }
}

private final class SnapshotLayoutManager: CALayoutManager {
    private let operation: (CALayer) -> Void
    private(set) var layoutCount = 0

    init(operation: @escaping (CALayer) -> Void = { _ in }) {
        self.operation = operation
    }

    func layoutSublayers(of layer: CALayer) {
        layoutCount += 1
        operation(layer)
    }
}

private final class SnapshotContentsToken {}
private final class SnapshotBaseLayer: CALayer {}

private final class SnapshotDrawingDelegate: CALayerDelegate {
    var leftColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
    var rightColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
    var invalidatesDuringDisplay = false
    private(set) var willDrawCount = 0
    private(set) var drawCount = 0

    func display(_ layer: CALayer) {
        if invalidatesDuringDisplay {
            layer.setNeedsDisplay()
        }
    }

    func layerWillDraw(_ layer: CALayer) {
        willDrawCount += 1
    }

    func draw(_ layer: CALayer, in context: CGContext) {
        drawCount += 1
        let halfWidth = layer.bounds.width / 2
        context.setFillColor(leftColor)
        context.fill(CGRect(
            x: layer.bounds.minX,
            y: layer.bounds.minY,
            width: halfWidth,
            height: layer.bounds.height
        ))
        context.setFillColor(rightColor)
        context.fill(CGRect(
            x: layer.bounds.minX + halfWidth,
            y: layer.bounds.minY,
            width: halfWidth,
            height: layer.bounds.height
        ))
    }
}

private final class SnapshotDisplayDelegate: CALayerDelegate {
    let image: CGImage
    private(set) var displayCount = 0
    private(set) var drawCount = 0

    init(image: CGImage) {
        self.image = image
    }

    func display(_ layer: CALayer) {
        displayCount += 1
        layer.contents = image
    }

    func draw(_ layer: CALayer, in context: CGContext) {
        drawCount += 1
    }
}

#if canImport(Metal)
import Metal

extension CARenderSnapshotTests {
    @Test("Metal reports committed rasterization instead of dropping it")
    func metalRejectsUnsupportedRasterizationSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(
            device.makeTexture(descriptor: descriptor)
        )
        let renderer = CAMetalRenderer(destination: texture)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.shouldRasterize = true
        root.rasterizationScale = 2

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.rasterization))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed filters instead of dropping them")
    func metalRejectsUnsupportedFilterSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(
            device.makeTexture(descriptor: descriptor)
        )
        let renderer = CAMetalRenderer(destination: texture)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.filters = [CAFilter.colorInvert()]

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.filters))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed backdrop filters instead of dropping them")
    func metalRejectsUnsupportedBackdropSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(
            device.makeTexture(descriptor: descriptor)
        )
        let renderer = CAMetalRenderer(destination: texture)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.backgroundFilters = [CAFilter.colorInvert()]

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(
                .backdropComposition
            ))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed shadows instead of dropping them")
    func metalRejectsUnsupportedShadowSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(
            device.makeTexture(descriptor: descriptor)
        )
        let renderer = CAMetalRenderer(destination: texture)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.shadowOpacity = 1
        root.shadowColor = CGColor(
            red: 0,
            green: 0,
            blue: 0,
            alpha: 1
        )

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.shadow))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed group opacity instead of distributing it")
    func metalRejectsUnsupportedGroupOpacitySnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(
            device.makeTexture(descriptor: descriptor)
        )
        let renderer = CAMetalRenderer(destination: texture)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.opacity = 0.5
        root.allowsGroupOpacity = true
        root.addSublayer(CALayer())

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.groupOpacity))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed content masks instead of dropping them")
    func metalRejectsUnsupportedContentMaskSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(
            device.makeTexture(descriptor: descriptor)
        )
        let renderer = CAMetalRenderer(destination: texture)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.position = CGPoint(x: 1, y: 0.5)
        root.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        let mask = CALayer()
        mask.frame = root.bounds
        mask.backgroundColor = CGColor(
            red: 1,
            green: 1,
            blue: 1,
            alpha: 1
        )
        root.mask = mask

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.contentMask))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed image contents instead of dropping them")
    func metalRejectsUnsupportedImageSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let renderer = CAMetalRenderer(destination: texture)
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.position = CGPoint(x: 1, y: 0.5)
        root.contents = try makeImage(
            width: 2,
            height: 1,
            pixels: [255, 0, 0, 255, 0, 255, 0, 255]
        )

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.imageContents))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed gradients instead of dropping them")
    func metalRejectsUnsupportedGradientSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let renderer = CAMetalRenderer(destination: texture)
        let root = CAGradientLayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.position = CGPoint(x: 1, y: 0.5)
        root.colors = [
            CGColor(red: 1, green: 0, blue: 0, alpha: 1),
            CGColor(red: 0, green: 0, blue: 1, alpha: 1),
        ]

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.gradient))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed shapes instead of dropping them")
    func metalRejectsUnsupportedShapeSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 2,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(
            descriptor: descriptor
        ))
        let renderer = CAMetalRenderer(destination: texture)
        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 2, height: 1))
        let root = CAShapeLayer()
        root.bounds = CGRect(x: 0, y: 0, width: 2, height: 1)
        root.position = CGPoint(x: 1, y: 0.5)
        root.path = path
        root.fillColor = CGColor(
            red: 0,
            green: 1,
            blue: 0,
            alpha: 1
        )

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.shape))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed text instead of dropping it")
    func metalRejectsUnsupportedTextSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 16,
            height: 16,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(
            descriptor: descriptor
        ))
        let renderer = CAMetalRenderer(destination: texture)
        let root = CATextLayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.string = "Text"
        root.font = "sans-serif"

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.text))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed transform depth instead of flattening it")
    func metalRejectsUnsupportedTransformSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 16,
            height: 16,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(
            descriptor: descriptor
        ))
        let renderer = CAMetalRenderer(destination: texture)
        let root = CATransformLayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.addSublayer(CALayer())

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(
                .transformDepth
            ))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed replicator instances instead of omitting them")
    func metalRejectsUnsupportedReplicatorSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 16,
            height: 16,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(
            descriptor: descriptor
        ))
        let renderer = CAMetalRenderer(destination: texture)
        let root = CAReplicatorLayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.instanceCount = 2
        root.addSublayer(CALayer())

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(
                .replicatorInstances
            ))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed emitters instead of omitting them")
    func metalRejectsUnsupportedEmitterSnapshot() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 16,
            height: 16,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(
            descriptor: descriptor
        ))
        let renderer = CAMetalRenderer(destination: texture)
        let root = CAEmitterLayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.emitterCells = [CAEmitterCell()]

        renderer.render(layer: root)

        #expect(renderer.lastRenderError
            == .unsupportedCommittedSnapshotFeature(.emitter))
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal reports committed tiled content instead of omitting it")
    func metalRejectsUnsupportedTiledSnapshot() throws {
        let device = try #require(
            MTLCreateSystemDefaultDevice()
        )
        let descriptor =
            MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: 16,
                height: 16,
                mipmapped: false
            )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(
            device.makeTexture(descriptor: descriptor)
        )
        let renderer = CAMetalRenderer(
            destination: texture
        )
        let root = CATiledLayer()
        root.bounds = CGRect(
            x: 0,
            y: 0,
            width: 16,
            height: 16
        )
        root.position = CGPoint(x: 8, y: 8)
        let provider = SnapshotTileProvider()
        root.delegate = provider

        renderer.render(layer: root)

        #expect(
            renderer.lastRenderError
                == .unsupportedCommittedSnapshotFeature(
                    .tiledLayer
                )
        )
        #expect(renderer.lastCommandBuffer == nil)
    }

    @Test("Metal submits committed pixels without clearing a later mutation")
    func metalSubmissionPreservesPostCommitMutation() throws {
        CATransaction.flush()
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 16,
            height: 16,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let renderer = CAMetalRenderer(destination: texture)
        let root = CALayer()

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        CATransaction.commit()

        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        renderer.render(layer: root)

        let commandBuffer = try #require(renderer.lastCommandBuffer)
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        var pixel = [UInt8](repeating: 0, count: 4)
        pixel.withUnsafeMutableBytes { bytes in
            guard let destination = bytes.baseAddress else { return }
            texture.getBytes(
                destination,
                bytesPerRow: 4,
                from: MTLRegionMake2D(8, 8, 1, 1),
                mipmapLevel: 0
            )
        }
        #expect(pixel == [0, 255, 0, 255])
        #expect(root._dirtyMask.isEmpty == false)
        if case .some = root.pendingCommittedRenderState {
            Issue.record("Expected the submitted committed snapshot to be acknowledged")
        }

        CATransaction.flush()
    }

    @Test("Metal retains the committed evaluator after acknowledging its transaction")
    func metalRetainsCommittedAnimationEvaluator() throws {
        CATransaction.flush()
        let device = try #require(
            MTLCreateSystemDefaultDevice()
        )
        let descriptor =
            MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: 16,
                height: 16,
                mipmapped: false
            )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(
            device.makeTexture(descriptor: descriptor)
        )
        let renderer = CAMetalRenderer(
            destination: texture
        )
        let root = CALayer()
        root.bounds = CGRect(
            x: 0,
            y: 0,
            width: 16,
            height: 16
        )
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        let animation = CABasicAnimation(
            keyPath: "backgroundColor"
        )
        animation.fromValue = CGColor(
            red: 1,
            green: 0,
            blue: 0,
            alpha: 1
        )
        animation.toValue = CGColor(
            red: 0,
            green: 0,
            blue: 1,
            alpha: 1
        )
        animation.duration = 1
        animation.speed = 0
        animation.timeOffset = 0.25
        CATransaction.flush()

        CATransaction.begin()
        root.add(
            animation,
            forKey: "backgroundColor"
        )
        CATransaction.commit()

        root.backgroundColor = CGColor(
            red: 0,
            green: 1,
            blue: 0,
            alpha: 1
        )
        let stored = try #require(
            root.animation(forKey: "backgroundColor")
                as? CABasicAnimation
        )
        stored.toValue = root.backgroundColor
        stored.timeOffset = 0.75

        renderer.render(layer: root)
        let firstCommandBuffer = try #require(
            renderer.lastCommandBuffer
        )
        firstCommandBuffer.waitUntilCompleted()
        #expect(firstCommandBuffer.status == .completed)
        #expect(root.pendingCommittedRenderState == nil)
        let firstPixel = readPixel(
            texture,
            x: 8,
            y: 8
        )
        #expect(firstPixel == [64, 0, 191, 255])

        root.backgroundColor = CGColor(
            red: 1,
            green: 1,
            blue: 0,
            alpha: 1
        )
        renderer.render(layer: root)
        let secondCommandBuffer = try #require(
            renderer.lastCommandBuffer
        )
        secondCommandBuffer.waitUntilCompleted()
        #expect(secondCommandBuffer.status == .completed)
        #expect(
            readPixel(texture, x: 8, y: 8)
                == firstPixel
        )
        CATransaction.flush()
        root.removeAllAnimations()
    }

    @Test("Metal encoding reads the captured frame instead of the mutated model")
    func metalEncodingUsesCapturedValues() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 16,
            height: 16,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let renderer = CAMetalRenderer(destination: texture)

        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 16, height: 16)
        root.position = CGPoint(x: 8, y: 8)
        root.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        let snapshot = try CARenderSnapshot.capture(root, frameToken: 45)

        root.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)

        #expect(renderer.render(snapshot: snapshot))
        let commandBuffer = try #require(renderer.lastCommandBuffer)
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        var pixel = [UInt8](repeating: 0, count: 4)
        pixel.withUnsafeMutableBytes { bytes in
            guard let destination = bytes.baseAddress else { return }
            texture.getBytes(
                destination,
                bytesPerRow: 4,
                from: MTLRegionMake2D(8, 8, 1, 1),
                mipmapLevel: 0
            )
        }
        #expect(pixel == [0, 255, 0, 255])
    }

    private func readPixel(
        _ texture: any MTLTexture,
        x: Int,
        y: Int
    ) -> [UInt8] {
        var pixel = [UInt8](repeating: 0, count: 4)
        pixel.withUnsafeMutableBytes { bytes in
            guard let destination = bytes.baseAddress else {
                return
            }
            texture.getBytes(
                destination,
                bytesPerRow: 4,
                from: MTLRegionMake2D(x, y, 1, 1),
                mipmapLevel: 0
            )
        }
        return pixel
    }
}
#endif
