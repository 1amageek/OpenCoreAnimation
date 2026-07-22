import Testing
@testable import OpenCoreAnimation

@Suite("CALayer archive decisions")
struct CALayerArchivingTests {
    private final class Delegate: CALayerDelegate {}

    @Test("Fresh base-layer decisions match QuartzCore")
    func baseDefaults() {
        let layer = CALayer()
        let alwaysArchived = [
            "cornerCurve",
            "contentsFormat",
            "allowsEdgeAntialiasing",
            "allowsGroupOpacity",
        ]
        let omitted = [
            "anchorPoint", "anchorPointZ", "bounds", "position", "zPosition",
            "opacity", "hidden", "masksToBounds", "mask", "doubleSided",
            "cornerRadius", "maskedCorners", "borderWidth", "borderColor",
            "backgroundColor", "shadowOpacity", "shadowRadius", "shadowOffset",
            "shadowColor", "shadowPath", "style", "filters", "compositingFilter",
            "backgroundFilters", "contents", "contentsRect", "contentsCenter",
            "contentsGravity", "contentsScale", "minificationFilter",
            "magnificationFilter", "minificationFilterBias", "opaque",
            "edgeAntialiasingMask", "geometryFlipped", "drawsAsynchronously",
            "shouldRasterize", "rasterizationScale", "toneMapMode",
            "preferredDynamicRange", "contentsHeadroom", "transform",
            "sublayerTransform", "sublayers", "needsDisplayOnBoundsChange",
            "layoutManager", "autoresizingMask", "constraints", "actions",
            "delegate", "name", "beginTime", "timeOffset", "repeatCount",
            "repeatDuration", "duration", "speed", "autoreverses", "fillMode",
            "frame", "superlayer", "visibleRect", "futureKey",
        ]

        for key in alwaysArchived {
            #expect(layer.shouldArchiveValue(forKey: key), "\(key)")
        }
        for key in omitted {
            #expect(!layer.shouldArchiveValue(forKey: key), "\(key)")
        }
    }

    @Test("Changed base-layer values are archived")
    func baseMutations() {
        let layer = CALayer()
        let delegate = Delegate()
        layer.anchorPoint = CGPoint(x: 0, y: 0)
        layer.anchorPointZ = 1
        layer.bounds = CGRect(x: 1, y: 2, width: 3, height: 4)
        layer.position = CGPoint(x: 5, y: 6)
        layer.zPosition = 1
        layer.opacity = 0.5
        layer.isHidden = true
        layer.masksToBounds = true
        layer.mask = CALayer()
        layer.isDoubleSided = false
        layer.cornerRadius = 1
        layer.maskedCorners = [.layerMinXMinYCorner]
        layer.borderWidth = 1
        layer.borderColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        layer.shadowOpacity = 1
        layer.shadowRadius = 1
        layer.shadowOffset = .zero
        layer.shadowColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        let shadowPath = CGMutablePath()
        shadowPath.addRect(CGRect(x: 0, y: 0, width: 1, height: 1))
        layer.shadowPath = shadowPath
        layer.style = ["value": 1]
        layer.filters = ["filter"]
        layer.compositingFilter = "filter"
        layer.backgroundFilters = ["filter"]
        layer.contents = "contents"
        layer.contentsRect = CGRect(x: 0, y: 0, width: 0.5, height: 1)
        layer.contentsCenter = CGRect(x: 0, y: 0, width: 0.5, height: 1)
        layer.contentsGravity = .center
        layer.contentsScale = 2
        layer.minificationFilter = .nearest
        layer.magnificationFilter = .nearest
        layer.minificationFilterBias = 1
        layer.isOpaque = true
        layer.edgeAntialiasingMask = []
        layer.isGeometryFlipped = true
        layer.drawsAsynchronously = true
        layer.shouldRasterize = true
        layer.rasterizationScale = 2
        layer.toneMapMode = .never
        layer.preferredDynamicRange = .high
        layer.contentsHeadroom = 2
        layer.transform = CATransform3DMakeTranslation(1, 0, 0)
        layer.sublayerTransform = CATransform3DMakeScale(2, 2, 1)
        layer.addSublayer(CALayer())
        layer.needsDisplayOnBoundsChange = true
        layer.layoutManager = CAConstraintLayoutManager()
        layer.autoresizingMask = [.layerWidthSizable]
        layer.constraints = [CAConstraint(
            attribute: .minX,
            relativeTo: "superlayer",
            attribute: .minX
        )]
        layer.actions = ["opacity": CABasicAnimation(keyPath: "opacity")]
        layer.delegate = delegate
        layer.name = "layer"
        layer.beginTime = 1
        layer.timeOffset = 1
        layer.repeatCount = 1
        layer.repeatDuration = 1
        layer.duration = 1
        layer.speed = 2
        layer.autoreverses = true
        layer.fillMode = .forwards

        let archived = [
            "anchorPoint", "anchorPointZ", "bounds", "position", "zPosition",
            "opacity", "hidden", "masksToBounds", "mask", "doubleSided",
            "cornerRadius", "maskedCorners", "borderWidth", "borderColor",
            "backgroundColor", "shadowOpacity", "shadowRadius", "shadowOffset",
            "shadowColor", "shadowPath", "style", "filters", "compositingFilter",
            "backgroundFilters", "contents", "contentsRect", "contentsCenter",
            "contentsGravity", "contentsScale", "minificationFilter",
            "magnificationFilter", "minificationFilterBias", "opaque",
            "edgeAntialiasingMask", "geometryFlipped", "drawsAsynchronously",
            "shouldRasterize", "rasterizationScale", "toneMapMode",
            "preferredDynamicRange", "contentsHeadroom", "transform",
            "sublayerTransform", "sublayers", "needsDisplayOnBoundsChange",
            "layoutManager", "autoresizingMask", "constraints", "actions",
            "delegate", "name", "beginTime", "timeOffset", "repeatCount",
            "repeatDuration", "duration", "speed", "autoreverses", "fillMode",
        ]
        for key in archived {
            #expect(layer.shouldArchiveValue(forKey: key), "\(key)")
        }
        #expect(!layer.shouldArchiveValue(forKey: "frame"))
        #expect(!layer.shouldArchiveValue(forKey: "superlayer"))
        #expect(!layer.shouldArchiveValue(forKey: "visibleRect"))
        #expect(!layer.shouldArchiveValue(forKey: "futureKey"))
    }

    @Test("Fresh specialized-layer values are omitted")
    func specializedDefaults() {
        let layersAndKeys: [(CALayer, [String])] = [
            (CAShapeLayer(), [
                "path", "fillColor", "fillRule", "lineCap", "lineDashPattern",
                "lineDashPhase", "lineJoin", "lineWidth", "miterLimit",
                "strokeColor", "strokeStart", "strokeEnd",
            ]),
            (CAGradientLayer(), ["colors", "locations", "startPoint", "endPoint", "type"]),
            (CAReplicatorLayer(), [
                "instanceCount", "preservesDepth", "instanceDelay", "instanceTransform",
                "instanceColor", "instanceRedOffset", "instanceGreenOffset",
                "instanceBlueOffset", "instanceAlphaOffset",
            ]),
            (CAEmitterLayer(), [
                "emitterCells", "emitterPosition", "emitterZPosition", "emitterSize",
                "emitterDepth", "emitterShape", "emitterMode", "renderMode",
                "preservesDepth", "birthRate", "lifetime", "velocity", "scale",
                "spin", "seed",
            ]),
            (CATextLayer(), [
                "string", "font", "fontSize", "foregroundColor", "wrapped",
                "truncationMode", "alignmentMode", "allowsFontSubpixelQuantization",
            ]),
            (CATiledLayer(), ["levelsOfDetail", "levelsOfDetailBias", "tileSize"]),
            (CAScrollLayer(), ["scrollMode"]),
        ]

        for (layer, keys) in layersAndKeys {
            for key in keys {
                #expect(!layer.shouldArchiveValue(forKey: key), "\(type(of: layer)).\(key)")
            }
        }
    }

    @Test("Changed specialized-layer values are archived")
    func specializedMutations() {
        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 1, height: 1))
        let shape = CAShapeLayer()
        shape.path = path
        shape.fillColor = nil
        shape.fillRule = .evenOdd
        shape.lineCap = .round
        shape.lineDashPattern = [1]
        shape.lineDashPhase = 1
        shape.lineJoin = .round
        shape.lineWidth = 2
        shape.miterLimit = 2
        shape.strokeColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        shape.strokeStart = 0.25
        shape.strokeEnd = 0.75

        let gradient = CAGradientLayer()
        gradient.colors = [CGColor(red: 1, green: 0, blue: 0, alpha: 1)]
        gradient.locations = [0]
        gradient.startPoint = .zero
        gradient.endPoint = CGPoint(x: 1, y: 1)
        gradient.type = .radial

        let replicator = CAReplicatorLayer()
        replicator.instanceCount = 2
        replicator.preservesDepth = true
        replicator.instanceDelay = 1
        replicator.instanceTransform = CATransform3DMakeTranslation(1, 0, 0)
        replicator.instanceColor = nil
        replicator.instanceRedOffset = 1
        replicator.instanceGreenOffset = 1
        replicator.instanceBlueOffset = 1
        replicator.instanceAlphaOffset = 1

        let emitter = CAEmitterLayer()
        emitter.emitterCells = [CAEmitterCell()]
        emitter.emitterPosition = CGPoint(x: 1, y: 1)
        emitter.emitterZPosition = 1
        emitter.emitterSize = CGSize(width: 1, height: 1)
        emitter.emitterDepth = 1
        emitter.emitterShape = .line
        emitter.emitterMode = .outline
        emitter.renderMode = .additive
        emitter.preservesDepth = true
        emitter.birthRate = 2
        emitter.lifetime = 2
        emitter.velocity = 2
        emitter.scale = 2
        emitter.spin = 2
        emitter.seed = 1

        let text = CATextLayer()
        text.string = "text"
        text.font = "Other"
        text.fontSize = 12
        text.foregroundColor = nil
        text.isWrapped = true
        text.truncationMode = .end
        text.alignmentMode = .center
        text.allowsFontSubpixelQuantization = true

        let tiled = CATiledLayer()
        tiled.levelsOfDetail = 2
        tiled.levelsOfDetailBias = 1
        tiled.tileSize = CGSize(width: 128, height: 128)

        let scroll = CAScrollLayer()
        scroll.scrollMode = .horizontally

        let layersAndKeys: [(CALayer, [String])] = [
            (shape, [
                "path", "fillColor", "fillRule", "lineCap", "lineDashPattern",
                "lineDashPhase", "lineJoin", "lineWidth", "miterLimit",
                "strokeColor", "strokeStart", "strokeEnd",
            ]),
            (gradient, ["colors", "locations", "startPoint", "endPoint", "type"]),
            (replicator, [
                "instanceCount", "preservesDepth", "instanceDelay", "instanceTransform",
                "instanceColor", "instanceRedOffset", "instanceGreenOffset",
                "instanceBlueOffset", "instanceAlphaOffset",
            ]),
            (emitter, [
                "emitterCells", "emitterPosition", "emitterZPosition", "emitterSize",
                "emitterDepth", "emitterShape", "emitterMode", "renderMode",
                "preservesDepth", "birthRate", "lifetime", "velocity", "scale",
                "spin", "seed",
            ]),
            (text, [
                "string", "font", "fontSize", "foregroundColor", "wrapped",
                "truncationMode", "alignmentMode", "allowsFontSubpixelQuantization",
            ]),
            (tiled, ["levelsOfDetail", "levelsOfDetailBias", "tileSize"]),
            (scroll, ["scrollMode"]),
        ]
        for (layer, keys) in layersAndKeys {
            for key in keys {
                #expect(layer.shouldArchiveValue(forKey: key), "\(type(of: layer)).\(key)")
            }
        }
    }
}
