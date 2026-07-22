import Foundation

/// Converts a complete path into non-overlapping triangles while preserving
/// even-odd and non-zero winding semantics across every subpath.
internal enum ShapeFillTessellator {
    private struct Edge {
        let start: CGPoint
        let end: CGPoint

        var windingDelta: Int { end.y > start.y ? 1 : -1 }
        var minimumY: CGFloat { min(start.y, end.y) }
        var maximumY: CGFloat { max(start.y, end.y) }

        func x(at y: CGFloat) -> CGFloat {
            start.x + (y - start.y) * (end.x - start.x) / (end.y - start.y)
        }
    }

    private struct Crossing {
        let x: CGFloat
        let edge: Edge
    }

    static func validate(_ path: CGPath) throws {
        var failure: ShapeFillTessellationError?
        path.applyWithBlock { elementPointer in
            guard failure == nil else { return }
            let element = elementPointer.pointee
            let pointCount: Int
            switch element.type {
            case .moveToPoint, .addLineToPoint:
                pointCount = 1
            case .addQuadCurveToPoint:
                pointCount = 2
            case .addCurveToPoint:
                pointCount = 3
            case .closeSubpath:
                pointCount = 0
            @unknown default:
                failure = .nonFinitePath
                return
            }
            guard pointCount > 0 else { return }
            guard let points = element.points else {
                failure = .nonFinitePath
                return
            }
            for index in 0..<pointCount {
                let point = points[index]
                guard point.x.isFinite, point.y.isFinite else {
                    failure = .nonFinitePath
                    return
                }
            }
        }
        if let failure { throw failure }
    }

    static func triangles(
        for path: CGPath,
        rule: CAShapeLayerFillRule,
        flatness: CGFloat = 0.5
    ) throws -> [CGPoint] {
        guard flatness.isFinite, flatness > 0 else {
            throw ShapeFillTessellationError.nonFinitePath
        }
        guard rule == .nonZero || rule == .evenOdd else {
            throw ShapeFillTessellationError.unsupportedFillRule(rule.rawValue)
        }
        try validate(path)

        let contours = try flattenedContours(path, flatness: flatness)
        let edges = contours.flatMap(makeEdges)
        guard !edges.isEmpty else { return [] }

        var criticalY = edges.flatMap { [$0.start.y, $0.end.y] }
        for firstIndex in edges.indices {
            for secondIndex in edges.indices where secondIndex > firstIndex {
                if let intersectionY = properIntersectionY(
                    edges[firstIndex],
                    edges[secondIndex]
                ) {
                    criticalY.append(intersectionY)
                }
            }
        }
        criticalY.sort()
        criticalY = uniqueValues(criticalY)

        var result: [CGPoint] = []
        for index in 0..<(criticalY.count - 1) {
            let lowerY = criticalY[index]
            let upperY = criticalY[index + 1]
            guard upperY - lowerY > 1e-9 else { continue }
            let middleY = (lowerY + upperY) / 2
            let crossings = edges.compactMap { edge -> Crossing? in
                guard middleY > edge.minimumY, middleY < edge.maximumY else { return nil }
                return Crossing(x: edge.x(at: middleY), edge: edge)
            }.sorted { lhs, rhs in
                if abs(lhs.x - rhs.x) > 1e-9 { return lhs.x < rhs.x }
                return lhs.edge.windingDelta < rhs.edge.windingDelta
            }
            appendSlab(
                crossings: crossings,
                lowerY: lowerY,
                upperY: upperY,
                rule: rule,
                to: &result
            )
        }
        guard result.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
            throw ShapeFillTessellationError.nonFinitePath
        }
        return result
    }

    private static func appendSlab(
        crossings: [Crossing],
        lowerY: CGFloat,
        upperY: CGFloat,
        rule: CAShapeLayerFillRule,
        to result: inout [CGPoint]
    ) {
        var winding = 0
        var parity = false
        var leftBoundary: Edge?
        var index = 0

        while index < crossings.count {
            let groupX = crossings[index].x
            var groupEnd = index + 1
            while groupEnd < crossings.count,
                  abs(crossings[groupEnd].x - groupX) <= 1e-9 {
                groupEnd += 1
            }

            let wasInside = rule == .evenOdd ? parity : winding != 0
            if rule == .evenOdd {
                if (groupEnd - index).isMultiple(of: 2) == false {
                    parity.toggle()
                }
            } else {
                for crossing in crossings[index..<groupEnd] {
                    winding += crossing.edge.windingDelta
                }
            }
            let isInside = rule == .evenOdd ? parity : winding != 0

            if !wasInside, isInside {
                leftBoundary = crossings[index].edge
            } else if wasInside, !isInside, let boundary = leftBoundary {
                appendTrapezoid(
                    left: boundary,
                    right: crossings[index].edge,
                    lowerY: lowerY,
                    upperY: upperY,
                    to: &result
                )
                leftBoundary = nil
            }
            index = groupEnd
        }
    }

    private static func appendTrapezoid(
        left: Edge,
        right: Edge,
        lowerY: CGFloat,
        upperY: CGFloat,
        to result: inout [CGPoint]
    ) {
        let lowerLeft = CGPoint(x: left.x(at: lowerY), y: lowerY)
        let lowerRight = CGPoint(x: right.x(at: lowerY), y: lowerY)
        let upperLeft = CGPoint(x: left.x(at: upperY), y: upperY)
        let upperRight = CGPoint(x: right.x(at: upperY), y: upperY)
        appendTriangle(lowerLeft, lowerRight, upperRight, to: &result)
        appendTriangle(lowerLeft, upperRight, upperLeft, to: &result)
    }

    private static func appendTriangle(
        _ first: CGPoint,
        _ second: CGPoint,
        _ third: CGPoint,
        to result: inout [CGPoint]
    ) {
        let area = (second.x - first.x) * (third.y - first.y)
            - (second.y - first.y) * (third.x - first.x)
        guard abs(area) > 1e-9 else { return }
        result.append(contentsOf: [first, second, third])
    }

    private static func makeEdges(_ contour: [CGPoint]) -> [Edge] {
        guard contour.count >= 3 else { return [] }
        var edges: [Edge] = []
        for index in contour.indices {
            let start = contour[index]
            let end = contour[(index + 1) % contour.count]
            guard abs(start.y - end.y) > 1e-9 else { continue }
            edges.append(Edge(start: start, end: end))
        }
        return edges
    }

    private static func properIntersectionY(_ lhs: Edge, _ rhs: Edge) -> CGFloat? {
        let lhsVector = CGPoint(x: lhs.end.x - lhs.start.x, y: lhs.end.y - lhs.start.y)
        let rhsVector = CGPoint(x: rhs.end.x - rhs.start.x, y: rhs.end.y - rhs.start.y)
        let denominator = cross(lhsVector, rhsVector)
        guard abs(denominator) > 1e-12 else { return nil }
        let offset = CGPoint(x: rhs.start.x - lhs.start.x, y: rhs.start.y - lhs.start.y)
        let lhsParameter = cross(offset, rhsVector) / denominator
        let rhsParameter = cross(offset, lhsVector) / denominator
        guard lhsParameter > 1e-9, lhsParameter < 1 - 1e-9,
              rhsParameter > 1e-9, rhsParameter < 1 - 1e-9 else { return nil }
        return lhs.start.y + lhsParameter * lhsVector.y
    }

    private static func cross(_ lhs: CGPoint, _ rhs: CGPoint) -> CGFloat {
        lhs.x * rhs.y - lhs.y * rhs.x
    }

    private static func uniqueValues(_ values: [CGFloat]) -> [CGFloat] {
        var result: [CGFloat] = []
        for value in values where result.last.map({ abs($0 - value) > 1e-9 }) ?? true {
            result.append(value)
        }
        return result
    }

    private static func flattenedContours(
        _ path: CGPath,
        flatness: CGFloat
    ) throws -> [[CGPoint]] {
        var contours: [[CGPoint]] = []
        var current: [CGPoint] = []
        var currentPoint = CGPoint.zero
        var failure: ShapeFillTessellationError?

        func accept(_ point: CGPoint) -> Bool {
            guard point.x.isFinite, point.y.isFinite else {
                failure = .nonFinitePath
                return false
            }
            return true
        }

        path.applyWithBlock { elementPointer in
            guard failure == nil else { return }
            let element = elementPointer.pointee
            switch element.type {
            case .moveToPoint:
                if current.count >= 3 { contours.append(current) }
                current = []
                guard let points = element.points, accept(points[0]) else { return }
                current.append(points[0])
                currentPoint = points[0]
            case .addLineToPoint:
                guard let points = element.points, accept(points[0]) else { return }
                current.append(points[0])
                currentPoint = points[0]
            case .addQuadCurveToPoint:
                guard let points = element.points,
                      accept(points[0]), accept(points[1]) else { return }
                flattenQuadratic(
                    from: currentPoint,
                    control: points[0],
                    to: points[1],
                    flatness: flatness,
                    depth: 0,
                    into: &current
                )
                currentPoint = points[1]
            case .addCurveToPoint:
                guard let points = element.points,
                      accept(points[0]), accept(points[1]), accept(points[2]) else { return }
                flattenCubic(
                    from: currentPoint,
                    control1: points[0],
                    control2: points[1],
                    to: points[2],
                    flatness: flatness,
                    depth: 0,
                    into: &current
                )
                currentPoint = points[2]
            case .closeSubpath:
                if current.count >= 3 { contours.append(current) }
                current = []
            @unknown default:
                failure = .nonFinitePath
            }
        }
        if let failure { throw failure }
        if current.count >= 3 { contours.append(current) }
        return contours
    }

    private static func flattenQuadratic(
        from start: CGPoint,
        control: CGPoint,
        to end: CGPoint,
        flatness: CGFloat,
        depth: Int,
        into points: inout [CGPoint]
    ) {
        let distance = pointLineDistance(control, start: start, end: end)
        guard distance > flatness, depth < 20 else {
            points.append(end)
            return
        }
        let first = midpoint(start, control)
        let second = midpoint(control, end)
        let middle = midpoint(first, second)
        flattenQuadratic(from: start, control: first, to: middle, flatness: flatness, depth: depth + 1, into: &points)
        flattenQuadratic(from: middle, control: second, to: end, flatness: flatness, depth: depth + 1, into: &points)
    }

    private static func flattenCubic(
        from start: CGPoint,
        control1: CGPoint,
        control2: CGPoint,
        to end: CGPoint,
        flatness: CGFloat,
        depth: Int,
        into points: inout [CGPoint]
    ) {
        let distance = max(
            pointLineDistance(control1, start: start, end: end),
            pointLineDistance(control2, start: start, end: end)
        )
        guard distance > flatness, depth < 20 else {
            points.append(end)
            return
        }
        let first = midpoint(start, control1)
        let center = midpoint(control1, control2)
        let last = midpoint(control2, end)
        let firstCenter = midpoint(first, center)
        let lastCenter = midpoint(center, last)
        let middle = midpoint(firstCenter, lastCenter)
        flattenCubic(from: start, control1: first, control2: firstCenter, to: middle, flatness: flatness, depth: depth + 1, into: &points)
        flattenCubic(from: middle, control1: lastCenter, control2: last, to: end, flatness: flatness, depth: depth + 1, into: &points)
    }

    private static func midpoint(_ lhs: CGPoint, _ rhs: CGPoint) -> CGPoint {
        CGPoint(x: lhs.x / 2 + rhs.x / 2, y: lhs.y / 2 + rhs.y / 2)
    }

    private static func pointLineDistance(_ point: CGPoint, start: CGPoint, end: CGPoint) -> CGFloat {
        let delta = CGPoint(x: end.x - start.x, y: end.y - start.y)
        let length = hypot(delta.x, delta.y)
        guard length > 1e-12 else { return hypot(point.x - start.x, point.y - start.y) }
        return abs(cross(CGPoint(x: point.x - start.x, y: point.y - start.y), delta)) / length
    }
}
