import Foundation

/// Converts CAShapeLayer stroke state into fill triangles through the shared
/// OpenCoreGraphics dash and stroke-outline operations.
internal enum ShapeStrokeTessellator {
    private struct Contour {
        let points: [CGPoint]
        let isClosed: Bool

        var segments: [(CGPoint, CGPoint, CGFloat)] {
            guard points.count >= 2 else { return [] }
            let count = isClosed ? points.count : points.count - 1
            return (0..<count).compactMap { index in
                let start = points[index]
                let end = points[(index + 1) % points.count]
                let length = hypot(end.x - start.x, end.y - start.y)
                return length > 1e-9 ? (start, end, length) : nil
            }
        }
    }

    private struct TrimmedSubpath {
        let path: CGPath
        let dashOffset: CGFloat
    }

    static func triangles(
        for path: CGPath,
        lineWidth: CGFloat,
        lineCap: CAShapeLayerLineCap,
        lineJoin: CAShapeLayerLineJoin,
        miterLimit: CGFloat,
        dashPattern: [CGFloat]?,
        dashPhase: CGFloat,
        strokeStart: CGFloat,
        strokeEnd: CGFloat,
        flatness: CGFloat = 0.5
    ) throws(ShapeStrokeTessellationError) -> [CGPoint] {
        guard lineWidth.isFinite, lineWidth > 0,
              miterLimit.isFinite,
              dashPhase.isFinite,
              strokeStart.isFinite,
              strokeEnd.isFinite,
              flatness.isFinite,
              flatness > 0 else {
            throw ShapeStrokeTessellationError.invalidGeometry
        }
        do {
            try ShapeFillTessellator.validate(path)
        } catch {
            throw .invalidGeometry
        }
        let cap = try coreGraphicsLineCap(lineCap)
        let join = try coreGraphicsLineJoin(lineJoin)
        let pattern = dashPattern ?? []
        guard pattern.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw ShapeStrokeTessellationError.invalidDashPattern
        }

        let start = min(1, max(0, strokeStart))
        let end = min(1, max(0, strokeEnd))
        guard start < end else { return [] }

        let subpaths = try trimmedSubpaths(
            path,
            start: start,
            end: end,
            flatness: flatness
        )
        guard !subpaths.isEmpty else { return [] }

        let centerline = CGMutablePath()
        for subpath in subpaths {
            if pattern.isEmpty {
                centerline.addPath(subpath.path)
            } else {
                centerline.addPath(subpath.path.copy(
                    dashingWithPhase: dashPhase + subpath.dashOffset,
                    lengths: pattern
                ))
            }
        }
        let outline = centerline.copy(
            strokingWithWidth: lineWidth,
            lineCap: cap,
            lineJoin: join,
            miterLimit: miterLimit
        )
        do {
            return try ShapeFillTessellator.triangles(for: outline, rule: .nonZero)
        } catch {
            throw .invalidGeometry
        }
    }

    private static func trimmedSubpaths(
        _ path: CGPath,
        start: CGFloat,
        end: CGFloat,
        flatness: CGFloat
    ) throws(ShapeStrokeTessellationError) -> [TrimmedSubpath] {
        let contours = contours(in: path.flattened(threshold: flatness))
        let lengths = contours.map { contour in
            contour.segments.reduce(CGFloat.zero) { $0 + $1.2 }
        }
        let totalLength = lengths.reduce(0, +)
        guard totalLength.isFinite else {
            throw ShapeStrokeTessellationError.invalidGeometry
        }
        guard totalLength > 1e-9 else { return [] }

        let lowerBound = totalLength * start
        let upperBound = totalLength * end
        var globalOffset: CGFloat = 0
        var result: [TrimmedSubpath] = []

        for (contour, contourLength) in zip(contours, lengths) {
            defer { globalOffset += contourLength }
            guard contourLength > 1e-9 else { continue }
            let localLower = max(0, lowerBound - globalOffset)
            let localUpper = min(contourLength, upperBound - globalOffset)
            guard localLower < localUpper else { continue }

            if contour.isClosed,
               localLower <= 1e-9,
               localUpper >= contourLength - 1e-9 {
                let complete = CGMutablePath()
                guard let first = contour.points.first else { continue }
                complete.move(to: first)
                for point in contour.points.dropFirst() { complete.addLine(to: point) }
                complete.closeSubpath()
                result.append(TrimmedSubpath(path: complete, dashOffset: 0))
                continue
            }

            let trimmed = CGMutablePath()
            var traversed: CGFloat = 0
            var hasPoint = false
            for (segmentStart, segmentEnd, segmentLength) in contour.segments {
                let segmentLower = traversed
                let segmentUpper = traversed + segmentLength
                defer { traversed = segmentUpper }
                let overlapLower = max(localLower, segmentLower)
                let overlapUpper = min(localUpper, segmentUpper)
                guard overlapLower < overlapUpper else { continue }
                let startT = (overlapLower - segmentLower) / segmentLength
                let endT = (overlapUpper - segmentLower) / segmentLength
                let first = interpolate(segmentStart, segmentEnd, startT)
                let last = interpolate(segmentStart, segmentEnd, endT)
                if !hasPoint {
                    trimmed.move(to: first)
                    hasPoint = true
                }
                trimmed.addLine(to: last)
            }
            if hasPoint {
                result.append(TrimmedSubpath(path: trimmed, dashOffset: localLower))
            }
        }
        return result
    }

    private static func contours(in path: CGPath) -> [Contour] {
        var result: [Contour] = []
        var points: [CGPoint] = []

        func finish(closed: Bool) {
            if points.count >= 2 {
                result.append(Contour(points: points, isClosed: closed))
            }
            points = []
        }

        path.applyWithBlock { pointer in
            let element = pointer.pointee
            switch element.type {
            case .moveToPoint:
                finish(closed: false)
                if let values = element.points { points = [values[0]] }
            case .addLineToPoint:
                if let values = element.points { points.append(values[0]) }
            case .closeSubpath:
                finish(closed: true)
            case .addQuadCurveToPoint, .addCurveToPoint:
                break
            @unknown default:
                break
            }
        }
        finish(closed: false)
        return result
    }

    private static func interpolate(_ start: CGPoint, _ end: CGPoint, _ t: CGFloat) -> CGPoint {
        CGPoint(
            x: start.x + (end.x - start.x) * t,
            y: start.y + (end.y - start.y) * t
        )
    }

    private static func coreGraphicsLineCap(
        _ value: CAShapeLayerLineCap
    ) throws(ShapeStrokeTessellationError) -> CGLineCap {
        switch value {
        case .butt: return .butt
        case .round: return .round
        case .square: return .square
        default: throw ShapeStrokeTessellationError.unsupportedLineCap(value.rawValue)
        }
    }

    private static func coreGraphicsLineJoin(
        _ value: CAShapeLayerLineJoin
    ) throws(ShapeStrokeTessellationError) -> CGLineJoin {
        switch value {
        case .miter: return .miter
        case .round: return .round
        case .bevel: return .bevel
        default: throw ShapeStrokeTessellationError.unsupportedLineJoin(value.rawValue)
        }
    }
}
