import Foundation

/// Specialized layer content that can draw into a Core Graphics context.
internal protocol CALayerContextRenderable: AnyObject {
    func renderSpecializedContent(in context: CGContext) throws(CALayerContextRenderError)
}

extension CAShapeLayer: CALayerContextRenderable {
    internal func renderSpecializedContent(
        in context: CGContext
    ) throws(CALayerContextRenderError) {
        guard let path else { return }

        do {
            try ShapeFillTessellator.validate(path)
        } catch {
            throw .nonFiniteShapePath
        }

        if let fillColor {
            let fillRule: CGPathFillRule
            switch self.fillRule {
            case .nonZero:
                fillRule = .winding
            case .evenOdd:
                fillRule = .evenOdd
            default:
                throw .unsupportedShapeFillRule(self.fillRule.rawValue)
            }
            context.setFillColor(fillColor)
            context.addPath(path)
            context.fillPath(using: fillRule)
        }

        guard let strokeColor, lineWidth > 0 else { return }
        let triangles: [CGPoint]
        do {
            triangles = try ShapeStrokeTessellator.triangles(
                for: path,
                lineWidth: lineWidth,
                lineCap: lineCap,
                lineJoin: lineJoin,
                miterLimit: miterLimit,
                dashPattern: lineDashPattern,
                dashPhase: lineDashPhase,
                strokeStart: strokeStart,
                strokeEnd: strokeEnd
            )
        } catch {
            switch error {
            case .invalidGeometry:
                throw .invalidShapeStrokeGeometry
            case .invalidDashPattern:
                throw .invalidShapeDashPattern
            case .unsupportedLineCap(let value):
                throw .unsupportedShapeLineCap(value)
            case .unsupportedLineJoin(let value):
                throw .unsupportedShapeLineJoin(value)
            }
        }

        let strokeOutline = CGMutablePath()
        for index in stride(from: 0, to: triangles.count, by: 3) {
            guard index + 2 < triangles.count else { break }
            strokeOutline.move(to: triangles[index])
            strokeOutline.addLine(to: triangles[index + 1])
            strokeOutline.addLine(to: triangles[index + 2])
            strokeOutline.closeSubpath()
        }
        context.setFillColor(strokeColor)
        context.addPath(strokeOutline)
        context.fillPath()
    }
}

extension CAGradientLayer: CALayerContextRenderable {
    internal func renderSpecializedContent(
        in context: CGContext
    ) throws(CALayerContextRenderError) {
        guard let colorValues = colors, !colorValues.isEmpty else { return }

        let configuration: GradientRenderConfiguration
        do {
            configuration = try GradientRenderConfiguration(
                type: type,
                colors: colorValues,
                locations: locations,
                startPoint: startPoint,
                endPoint: endPoint
            )
        } catch {
            throw Self.contextRenderError(for: error)
        }
        guard let gradient = CGGradient(
            colors: configuration.colors,
            locations: configuration.locations.map { CGFloat($0) }
        ) else {
            throw .gradientCreationFailed
        }

        context.saveGState()
        defer { context.restoreGState() }
        context.addPath(try contextRenderShapePath())
        context.clip()

        switch type {
        case .axial:
            context.drawLinearGradient(
                gradient,
                start: pointInBounds(startPoint),
                end: pointInBounds(endPoint),
                options: [.drawsBeforeStartLocation, .drawsAfterEndLocation]
            )
        case .radial:
            try drawRadialGradient(gradient, in: context)
        case .conic:
            try drawConicGradient(gradient, in: context)
        default:
            throw .unsupportedGradientType(type.rawValue)
        }
    }

    private func drawRadialGradient(
        _ gradient: CGGradient,
        in context: CGContext
    ) throws(CALayerContextRenderError) {
        let center = pointInBounds(startPoint)
        let radiusX = abs((endPoint.x - startPoint.x) * bounds.width)
        let radiusY = abs((endPoint.y - startPoint.y) * bounds.height)
        guard radiusX > 0, radiusY > 0 else {
            throw .degenerateRadialGradient
        }
        let stepCount = gradientStepCount
        for step in stride(from: stepCount, through: 1, by: -1) {
            let outerParameter = CGFloat(step) / CGFloat(stepCount)
            let innerParameter = CGFloat(step - 1) / CGFloat(stepCount)
            let sampleParameter = innerParameter
            guard let color = gradient.color(at: sampleParameter) else {
                throw .gradientInterpolationFailed
            }
            let band = CGMutablePath()
            band.addEllipse(in: ellipseRect(
                center: center,
                radiusX: radiusX * outerParameter,
                radiusY: radiusY * outerParameter
            ))
            if innerParameter > 0 {
                band.addEllipse(in: ellipseRect(
                    center: center,
                    radiusX: radiusX * innerParameter,
                    radiusY: radiusY * innerParameter
                ))
            }
            context.setFillColor(color)
            context.addPath(band)
            context.fillPath(using: .evenOdd)
        }
    }

    private func drawConicGradient(
        _ gradient: CGGradient,
        in context: CGContext
    ) throws(CALayerContextRenderError) {
        let center = pointInBounds(startPoint)
        let directionX = (endPoint.x - startPoint.x) * bounds.width
        let directionY = (endPoint.y - startPoint.y) * bounds.height
        let startAngle = atan2(directionY, directionX)
        let corners = [
            CGPoint(x: bounds.minX, y: bounds.minY),
            CGPoint(x: bounds.maxX, y: bounds.minY),
            CGPoint(x: bounds.minX, y: bounds.maxY),
            CGPoint(x: bounds.maxX, y: bounds.maxY),
        ]
        let radius = corners.reduce(CGFloat.zero) { current, corner in
            max(current, hypot(corner.x - center.x, corner.y - center.y))
        } + 1
        let stepCount = max(128, gradientStepCount)
        for step in 0..<stepCount {
            let lowerParameter = CGFloat(step) / CGFloat(stepCount)
            let upperParameter = CGFloat(step + 1) / CGFloat(stepCount)
            let sampleParameter = (lowerParameter + upperParameter) / 2
            guard let color = gradient.color(at: sampleParameter) else {
                throw .gradientInterpolationFailed
            }
            let lowerAngle = startAngle + lowerParameter * 2 * .pi
            let upperAngle = startAngle + upperParameter * 2 * .pi
            let wedge = CGMutablePath()
            wedge.move(to: center)
            wedge.addLine(to: CGPoint(
                x: center.x + cos(lowerAngle) * radius,
                y: center.y + sin(lowerAngle) * radius
            ))
            wedge.addLine(to: CGPoint(
                x: center.x + cos(upperAngle) * radius,
                y: center.y + sin(upperAngle) * radius
            ))
            wedge.closeSubpath()
            context.setFillColor(color)
            context.addPath(wedge)
            context.fillPath()
        }
    }

    private var gradientStepCount: Int {
        min(720, max(1, Int(ceil(max(abs(bounds.width), abs(bounds.height))))))
    }

    private func pointInBounds(_ unitPoint: CGPoint) -> CGPoint {
        CGPoint(
            x: bounds.minX + unitPoint.x * bounds.width,
            y: bounds.minY + unitPoint.y * bounds.height
        )
    }

    private func ellipseRect(
        center: CGPoint,
        radiusX: CGFloat,
        radiusY: CGFloat
    ) -> CGRect {
        CGRect(
            x: center.x - radiusX,
            y: center.y - radiusY,
            width: radiusX * 2,
            height: radiusY * 2
        )
    }

    private static func contextRenderError(
        for error: GradientRenderConfigurationError
    ) -> CALayerContextRenderError {
        switch error {
        case .unsupportedType(let value):
            return .unsupportedGradientType(value)
        case .nonFiniteGeometry:
            return .nonFiniteGradientGeometry
        case .invalidColor(let index):
            return .invalidGradientColor(index: index)
        case .invalidColorComponents(let index):
            return .invalidGradientColorComponents(index: index)
        case .invalidLocationCount(let expected, let actual):
            return .invalidGradientLocationCount(expected: expected, actual: actual)
        case .nonFiniteLocation(let index):
            return .nonFiniteGradientLocation(index: index)
        case .locationOutOfRange(let index):
            return .gradientLocationOutOfRange(index: index)
        case .locationsNotMonotonic(let index):
            return .gradientLocationsNotMonotonic(index: index)
        }
    }
}
