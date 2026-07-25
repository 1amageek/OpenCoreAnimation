import Testing
@testable import OpenCoreAnimation

@MainActor
@Suite("Immutable render snapshots", .serialized)
struct CARenderSnapshotTests {
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

        #expect(snapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
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
        #expect(snapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
        #expect(values.shouldRasterize)
        #expect(values.rasterizationScale == 2.5)

        layer.shouldRasterize = false
        layer.rasterizationScale = 1
        #expect(values.shouldRasterize)
        #expect(values.rasterizationScale == 2.5)
    }

    @Test("Invalid active rasterization scale fails snapshot capture")
    func invalidRasterizationScaleFailsCapture() {
        let layer = CALayer()
        layer.shouldRasterize = true
        layer.rasterizationScale = 0

        #expect(throws: CARendererError.invalidLayerRasterization(
            .invalidRasterizationScale(0)
        )) {
            _ = try CARenderSnapshot.capture(
                layer,
                frameToken: 46
            )
        }
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
        #expect(snapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
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
    func invalidBackdropCaptureIsTyped() {
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
        #expect(throws: CARendererError.invalidLayerCompositingFilter(
            .unsupportedFilterValue("Swift.String")
        )) {
            try CARenderSnapshot.capture(
                compositionLayer,
                frameToken: 48
            )
        }
    }

    @Test("Group opacity becomes value-owned snapshot state")
    func groupOpacityIsCapturedByValue() throws {
        let leaf = CALayer()
        leaf.opacity = 0.5
        let leafSnapshot = try CARenderSnapshot.capture(
            leaf,
            frameToken: 46
        )
        #expect(leafSnapshot.liveTreeRequirement == nil)

        let distributedRoot = CALayer()
        distributedRoot.opacity = 0.5
        distributedRoot.allowsGroupOpacity = false
        distributedRoot.addSublayer(CALayer())
        let distributedSnapshot = try CARenderSnapshot.capture(
            distributedRoot,
            frameToken: 47
        )
        #expect(distributedSnapshot.liveTreeRequirement == nil)

        let groupedRoot = CALayer()
        groupedRoot.opacity = 0.5
        groupedRoot.addSublayer(CALayer())
        let groupedSnapshot = try CARenderSnapshot.capture(
            groupedRoot,
            frameToken: 48
        )
        #expect(groupedSnapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
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
        #expect(snapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
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

        #expect(snapshot.liveTreeRequirement == nil)
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

    @Test("Animated commits request explicit live evaluation until evaluators are immutable")
    func animatedCommitPublishesExplicitEvaluationState() {
        CATransaction.flush()
        let root = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = Float(0)
        animation.toValue = Float(1)
        animation.duration = 1

        CATransaction.begin()
        root.add(animation, forKey: "opacity")
        CATransaction.commit()

        guard case .requiresLiveAnimationEvaluation = root.pendingCommittedRenderState else {
            Issue.record("Expected an explicit live-animation evaluation state")
            return
        }
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)
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
        let renderer = try CAMetalRenderer(destination: texture)

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
}
#endif
