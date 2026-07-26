import Foundation

/// Sendable model values required to evaluate a committed animation tree.
///
/// Renderer resources live in `presentation`. Reference-typed public API
/// values are normalized by the render-snapshot capture before this value
/// crosses the transaction boundary.
internal struct CACommittedLayerModelValues: Sendable {
    internal struct Timing: Sendable {
        let beginTime: CFTimeInterval
        let timeOffset: CFTimeInterval
        let repeatCount: Float
        let repeatDuration: CFTimeInterval
        let duration: CFTimeInterval
        let speed: Float
        let autoreverses: Bool
        let fillMode: CAMediaTimingFillMode
    }

    internal struct Shape: Sendable {
        let path: CGPath?
        let fillColor: CGColor?
        let fillRule: CAShapeLayerFillRule
        let lineCap: CAShapeLayerLineCap
        let lineDashPattern: [CGFloat]?
        let lineDashPhase: CGFloat
        let lineJoin: CAShapeLayerLineJoin
        let lineWidth: CGFloat
        let miterLimit: CGFloat
        let strokeColor: CGColor?
        let strokeStart: CGFloat
        let strokeEnd: CGFloat
    }

    internal struct Replicator: Sendable {
        let instanceCount: Int
        let preservesDepth: Bool
        let instanceDelay: CFTimeInterval
        let instanceTransform: CATransform3D
        let instanceColor: CGColor?
        let instanceRedOffset: Float
        let instanceGreenOffset: Float
        let instanceBlueOffset: Float
        let instanceAlphaOffset: Float
    }

    let presentation: CARenderSnapshot.PresentationValues
    let anchorPoint: CGPoint
    let anchorPointZ: CGFloat
    let contentsRect: CGRect
    let contentsCenter: CGRect
    let contentsGravity: CALayerContentsGravity
    let contentsFormat: CALayerContentsFormat
    let maskedCorners: CACornerMask
    let cornerCurve: CALayerCornerCurve
    let minificationFilter: CALayerContentsFilter
    let minificationFilterBias: Float
    let magnificationFilter: CALayerContentsFilter
    let allowsEdgeAntialiasing: Bool
    let autoresizingMask: CAAutoresizingMask
    let needsDisplayOnBoundsChange: Bool
    let name: String?
    let timing: Timing
    let shadowColor: CGColor?
    let shadowOpacity: Float
    let shadowRadius: CGFloat
    let shadowOffset: CGSize
    let shadowPath: CGPath?
    let shape: Shape?
    let replicator: Replicator?

    init(
        layer: CALayer,
        presentation: CARenderSnapshot.PresentationValues
    ) throws(CACommittedAnimationCaptureError) {
        self.presentation = presentation
        anchorPoint = layer.anchorPoint
        anchorPointZ = layer.anchorPointZ
        contentsRect = layer.contentsRect
        contentsCenter = layer.contentsCenter
        contentsGravity = layer.contentsGravity
        contentsFormat = layer.contentsFormat
        maskedCorners = layer.maskedCorners
        cornerCurve = layer.cornerCurve
        minificationFilter = layer.minificationFilter
        minificationFilterBias = layer.minificationFilterBias
        magnificationFilter = layer.magnificationFilter
        allowsEdgeAntialiasing =
            layer.allowsEdgeAntialiasing
        autoresizingMask = layer.autoresizingMask
        needsDisplayOnBoundsChange =
            layer.needsDisplayOnBoundsChange
        name = layer.name
        timing = Timing(
            beginTime: layer.beginTime,
            timeOffset: layer.timeOffset,
            repeatCount: layer.repeatCount,
            repeatDuration: layer.repeatDuration,
            duration: layer.duration,
            speed: layer.speed,
            autoreverses: layer.autoreverses,
            fillMode: layer.fillMode
        )
        shadowColor = try layer.shadowColor.map(Self.copy)
        shadowOpacity = layer.shadowOpacity
        shadowRadius = layer.shadowRadius
        shadowOffset = layer.shadowOffset
        if let path = layer.shadowPath {
            shadowPath = try Self.copy(path)
        } else {
            shadowPath = nil
        }
        if let layer = layer as? CAShapeLayer {
            shape = Shape(
                path: try layer.path.map(Self.copy),
                fillColor: try layer.fillColor.map(
                    Self.copy
                ),
                fillRule: layer.fillRule,
                lineCap: layer.lineCap,
                lineDashPattern: layer.lineDashPattern,
                lineDashPhase: layer.lineDashPhase,
                lineJoin: layer.lineJoin,
                lineWidth: layer.lineWidth,
                miterLimit: layer.miterLimit,
                strokeColor: try layer.strokeColor.map(
                    Self.copy
                ),
                strokeStart: layer.strokeStart,
                strokeEnd: layer.strokeEnd
            )
        } else {
            shape = nil
        }
        if let layer = layer as? CAReplicatorLayer {
            replicator = Replicator(
                instanceCount: layer.instanceCount,
                preservesDepth: layer.preservesDepth,
                instanceDelay: layer._instanceDelay,
                instanceTransform: layer._instanceTransform,
                instanceColor: try layer._instanceColor.map(
                    Self.copy
                ),
                instanceRedOffset: layer._instanceRedOffset,
                instanceGreenOffset: layer._instanceGreenOffset,
                instanceBlueOffset: layer._instanceBlueOffset,
                instanceAlphaOffset: layer._instanceAlphaOffset
            )
        } else {
            replicator = nil
        }
    }

    private static func copy(
        _ path: CGPath
    ) throws(CACommittedAnimationCaptureError) -> CGPath {
        guard let copy = path.copy() else {
            throw .unsupportedValueType("CGPath")
        }
        return copy
    }

    private static func copy(
        _ color: CGColor
    ) throws(CACommittedAnimationCaptureError) -> CGColor {
        guard let copy = color.copy() else {
            throw .unsupportedValueType("CGColor")
        }
        return copy
    }
}
