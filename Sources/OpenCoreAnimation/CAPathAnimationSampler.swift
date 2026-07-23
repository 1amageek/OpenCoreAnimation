import Foundation

/// Evaluates a Core Animation path while preserving its original segment and
/// subpath boundaries.
internal struct CAPathAnimationSampler {
    internal struct Sample {
        let point: CGPoint
        let tangent: CGFloat
    }

    private struct ArcSample {
        let parameter: CGFloat
        let cumulativeLength: CGFloat
    }

    private enum Segment {
        case line(start: CGPoint, end: CGPoint)
        case quadratic(start: CGPoint, control: CGPoint, end: CGPoint)
        case cubic(
            start: CGPoint,
            firstControl: CGPoint,
            secondControl: CGPoint,
            end: CGPoint
        )

        var start: CGPoint {
            switch self {
            case .line(let start, _),
                 .quadratic(let start, _, _),
                 .cubic(let start, _, _, _):
                return start
            }
        }

        var end: CGPoint {
            switch self {
            case .line(_, let end),
                 .quadratic(_, _, let end),
                 .cubic(_, _, _, let end):
                return end
            }
        }

        var coordinateScale: CGFloat {
            let points: [CGPoint]
            switch self {
            case .line(let start, let end):
                points = [start, end]
            case .quadratic(let start, let control, let end):
                points = [start, control, end]
            case .cubic(let start, let firstControl, let secondControl, let end):
                points = [start, firstControl, secondControl, end]
            }
            return points.reduce(CGFloat(1)) { scale, point in
                max(scale, max(abs(point.x), abs(point.y)))
            }
        }

        func point(at parameter: CGFloat) -> CGPoint {
            let t = min(max(parameter, 0), 1)
            let inverse = 1 - t
            switch self {
            case .line(let start, let end):
                return CGPoint(
                    x: start.x + t * (end.x - start.x),
                    y: start.y + t * (end.y - start.y)
                )
            case .quadratic(let start, let control, let end):
                return CGPoint(
                    x: inverse * inverse * start.x
                        + 2 * inverse * t * control.x
                        + t * t * end.x,
                    y: inverse * inverse * start.y
                        + 2 * inverse * t * control.y
                        + t * t * end.y
                )
            case .cubic(let start, let firstControl, let secondControl, let end):
                return CGPoint(
                    x: inverse * inverse * inverse * start.x
                        + 3 * inverse * inverse * t * firstControl.x
                        + 3 * inverse * t * t * secondControl.x
                        + t * t * t * end.x,
                    y: inverse * inverse * inverse * start.y
                        + 3 * inverse * inverse * t * firstControl.y
                        + 3 * inverse * t * t * secondControl.y
                        + t * t * t * end.y
                )
            }
        }

        func derivative(at parameter: CGFloat) -> CGPoint {
            let t = min(max(parameter, 0), 1)
            let inverse = 1 - t
            switch self {
            case .line(let start, let end):
                return CGPoint(x: end.x - start.x, y: end.y - start.y)
            case .quadratic(let start, let control, let end):
                return CGPoint(
                    x: 2 * inverse * (control.x - start.x)
                        + 2 * t * (end.x - control.x),
                    y: 2 * inverse * (control.y - start.y)
                        + 2 * t * (end.y - control.y)
                )
            case .cubic(let start, let firstControl, let secondControl, let end):
                return CGPoint(
                    x: 3 * inverse * inverse * (firstControl.x - start.x)
                        + 6 * inverse * t * (secondControl.x - firstControl.x)
                        + 3 * t * t * (end.x - secondControl.x),
                    y: 3 * inverse * inverse * (firstControl.y - start.y)
                        + 6 * inverse * t * (secondControl.y - firstControl.y)
                        + 3 * t * t * (end.y - secondControl.y)
                )
            }
        }

        func tangent(at parameter: CGFloat) -> CGFloat {
            var direction = derivative(at: parameter)
            if !Self.hasDirection(direction) {
                let lowerPoint = point(at: max(0, parameter - 0.0001))
                let upperPoint = point(at: min(1, parameter + 0.0001))
                direction = CGPoint(
                    x: upperPoint.x - lowerPoint.x,
                    y: upperPoint.y - lowerPoint.y
                )
            }
            if !Self.hasDirection(direction) {
                direction = CGPoint(x: end.x - start.x, y: end.y - start.y)
            }
            guard Self.hasDirection(direction) else { return 0 }
            return atan2(direction.y, direction.x)
        }

        func arcSamples() -> [ArcSample]? {
            let firstPoint = point(at: 0)
            let lastPoint = point(at: 1)
            guard CAPathAnimationSampler.isFinite(firstPoint),
                  CAPathAnimationSampler.isFinite(lastPoint) else {
                return nil
            }
            var points: [(parameter: CGFloat, point: CGPoint)] = [(0, firstPoint)]
            guard appendArcSamples(
                from: 0,
                startPoint: firstPoint,
                to: 1,
                endPoint: lastPoint,
                tolerance: max(0.0000001, coordinateScale * 0.00000001),
                depth: 0,
                into: &points
            ) else {
                return nil
            }

            var cumulativeLength: CGFloat = 0
            var result: [ArcSample] = []
            result.reserveCapacity(points.count)
            for index in points.indices {
                if index > points.startIndex {
                    cumulativeLength += Self.distance(
                        from: points[index - 1].point,
                        to: points[index].point
                    )
                    guard cumulativeLength.isFinite else { return nil }
                }
                result.append(ArcSample(
                    parameter: points[index].parameter,
                    cumulativeLength: cumulativeLength
                ))
            }
            return result
        }

        private func appendArcSamples(
            from startParameter: CGFloat,
            startPoint: CGPoint,
            to endParameter: CGFloat,
            endPoint: CGPoint,
            tolerance: CGFloat,
            depth: Int,
            into samples: inout [(parameter: CGFloat, point: CGPoint)]
        ) -> Bool {
            let interval = endParameter - startParameter
            let quarterParameter = startParameter + interval * 0.25
            let middleParameter = startParameter + interval * 0.5
            let threeQuarterParameter = startParameter + interval * 0.75
            let quarterPoint = point(at: quarterParameter)
            let middlePoint = point(at: middleParameter)
            let threeQuarterPoint = point(at: threeQuarterParameter)
            guard CAPathAnimationSampler.isFinite(quarterPoint),
                  CAPathAnimationSampler.isFinite(middlePoint),
                  CAPathAnimationSampler.isFinite(threeQuarterPoint) else {
                return false
            }
            let polylineLength = Self.distance(from: startPoint, to: quarterPoint)
                + Self.distance(from: quarterPoint, to: middlePoint)
                + Self.distance(from: middlePoint, to: threeQuarterPoint)
                + Self.distance(from: threeQuarterPoint, to: endPoint)
            let chordLength = Self.distance(from: startPoint, to: endPoint)
            guard polylineLength.isFinite, chordLength.isFinite else { return false }

            if depth >= 16 || polylineLength - chordLength <= tolerance {
                samples.append((endParameter, endPoint))
                return true
            }
            guard appendArcSamples(
                from: startParameter,
                startPoint: startPoint,
                to: middleParameter,
                endPoint: middlePoint,
                tolerance: tolerance,
                depth: depth + 1,
                into: &samples
            ) else {
                return false
            }
            return appendArcSamples(
                from: middleParameter,
                startPoint: middlePoint,
                to: endParameter,
                endPoint: endPoint,
                tolerance: tolerance,
                depth: depth + 1,
                into: &samples
            )
        }

        private static func hasDirection(_ point: CGPoint) -> Bool {
            point.x.isFinite
                && point.y.isFinite
                && (point.x != 0 || point.y != 0)
        }

        private static func distance(from start: CGPoint, to end: CGPoint) -> CGFloat {
            hypot(end.x - start.x, end.y - start.y)
        }
    }

    private struct PreparedSegment {
        let geometry: Segment
        let arcSamples: [ArcSample]

        var length: CGFloat {
            arcSamples.last?.cumulativeLength ?? 0
        }

        func parameter(atDistance distance: CGFloat) -> CGFloat {
            guard length > 0 else { return 0 }
            let target = min(max(distance, 0), length)
            if target <= 0 { return 0 }
            if target >= length { return 1 }

            var lowerIndex = 0
            var upperIndex = arcSamples.count - 1
            while lowerIndex + 1 < upperIndex {
                let middleIndex = (lowerIndex + upperIndex) / 2
                if arcSamples[middleIndex].cumulativeLength <= target {
                    lowerIndex = middleIndex
                } else {
                    upperIndex = middleIndex
                }
            }
            let lower = arcSamples[lowerIndex]
            let upper = arcSamples[upperIndex]
            let interval = upper.cumulativeLength - lower.cumulativeLength
            guard interval > 0 else { return upper.parameter }
            let progress = (target - lower.cumulativeLength) / interval
            return lower.parameter + progress * (upper.parameter - lower.parameter)
        }
    }

    private let segments: [Segment]

    internal init?(path: CGPath) {
        var parsedSegments: [Segment] = []
        var currentPoint: CGPoint?
        var subpathStart: CGPoint?
        var isValid = true

        path.applyWithBlock { elementPointer in
            guard isValid else { return }
            let element = elementPointer.pointee
            switch element.type {
            case .moveToPoint:
                guard let points = element.points, Self.isFinite(points[0]) else {
                    isValid = false
                    return
                }
                currentPoint = points[0]
                subpathStart = points[0]
            case .addLineToPoint:
                guard let points = element.points, Self.isFinite(points[0]) else {
                    isValid = false
                    return
                }
                let start = currentPoint ?? .zero
                parsedSegments.append(.line(start: start, end: points[0]))
                currentPoint = points[0]
                if subpathStart == nil { subpathStart = start }
            case .addQuadCurveToPoint:
                guard let points = element.points,
                      Self.isFinite(points[0]),
                      Self.isFinite(points[1]) else {
                    isValid = false
                    return
                }
                let start = currentPoint ?? .zero
                parsedSegments.append(.quadratic(
                    start: start,
                    control: points[0],
                    end: points[1]
                ))
                currentPoint = points[1]
                if subpathStart == nil { subpathStart = start }
            case .addCurveToPoint:
                guard let points = element.points,
                      Self.isFinite(points[0]),
                      Self.isFinite(points[1]),
                      Self.isFinite(points[2]) else {
                    isValid = false
                    return
                }
                let start = currentPoint ?? .zero
                parsedSegments.append(.cubic(
                    start: start,
                    firstControl: points[0],
                    secondControl: points[1],
                    end: points[2]
                ))
                currentPoint = points[2]
                if subpathStart == nil { subpathStart = start }
            case .closeSubpath:
                if let start = currentPoint, let end = subpathStart, start != end {
                    parsedSegments.append(.line(start: start, end: end))
                }
                currentPoint = subpathStart
            @unknown default:
                isValid = false
            }
        }

        guard isValid, !parsedSegments.isEmpty else { return nil }
        segments = parsedSegments
    }

    internal var cycleDisplacement: CGPoint? {
        guard let first = segments.first, let last = segments.last else { return nil }
        let displacement = CGPoint(
            x: last.end.x - first.start.x,
            y: last.end.y - first.start.y
        )
        return Self.isFinite(displacement) ? displacement : nil
    }

    internal func sample(
        at progress: CGFloat,
        calculationMode: CAAnimationCalculationMode,
        keyTimes: [CGFloat]?,
        timingFunctions: [CAMediaTimingFunction]?
    ) -> Sample? {
        guard progress.isFinite else { return nil }
        let clampedProgress = min(max(progress, 0), 1)
        let result: Sample?
        switch calculationMode {
        case .paced, .cubicPaced:
            result = pacedSample(at: clampedProgress)
        case .linear, .cubic, .discrete:
            result = segmentTimedSample(
                at: clampedProgress,
                calculationMode: calculationMode,
                keyTimes: keyTimes,
                timingFunctions: timingFunctions
            )
        default:
            return nil
        }
        guard let result,
              Self.isFinite(result.point),
              result.tangent.isFinite else {
            return nil
        }
        return result
    }

    private func pacedSample(at progress: CGFloat) -> Sample? {
        var preparedSegments: [PreparedSegment] = []
        preparedSegments.reserveCapacity(segments.count)
        for segment in segments {
            guard let arcSamples = segment.arcSamples() else { return nil }
            preparedSegments.append(
                PreparedSegment(geometry: segment, arcSamples: arcSamples)
            )
        }
        let totalLength = preparedSegments.reduce(CGFloat(0)) { $0 + $1.length }
        guard totalLength.isFinite else {
            return nil
        }
        guard totalLength > 0 else {
            let geometry = segments[0]
            return Sample(point: geometry.start, tangent: geometry.tangent(at: 0))
        }
        if progress >= 1 {
            let geometry = segments[segments.count - 1]
            return Sample(point: geometry.end, tangent: geometry.tangent(at: 1))
        }

        let targetDistance = progress * totalLength
        var traversedLength: CGFloat = 0
        for index in preparedSegments.indices {
            let segment = preparedSegments[index]
            let segmentEnd = traversedLength + segment.length
            if targetDistance < segmentEnd
                || index == preparedSegments.index(before: preparedSegments.endIndex) {
                let parameter = segment.parameter(
                    atDistance: targetDistance - traversedLength
                )
                return Sample(
                    point: segment.geometry.point(at: parameter),
                    tangent: segment.geometry.tangent(at: parameter)
                )
            }
            traversedLength = segmentEnd
        }

        let geometry = segments[segments.count - 1]
        return Sample(point: geometry.end, tangent: geometry.tangent(at: 1))
    }

    private func segmentTimedSample(
        at progress: CGFloat,
        calculationMode: CAAnimationCalculationMode,
        keyTimes: [CGFloat]?,
        timingFunctions: [CAMediaTimingFunction]?
    ) -> Sample {
        if progress >= 1 {
            let geometry = segments[segments.count - 1]
            return Sample(point: geometry.end, tangent: geometry.tangent(at: 1))
        }

        let effectiveKeyTimes = validatedKeyTimes(keyTimes) ?? defaultKeyTimes()
        let segmentIndex = min(
            effectiveKeyTimes.lastIndex(where: { $0 <= progress }) ?? 0,
            segments.count - 1
        )
        let segment = segments[segmentIndex]
        let startTime = effectiveKeyTimes[segmentIndex]
        let endTime = effectiveKeyTimes[segmentIndex + 1]
        let rawParameter = endTime > startTime
            ? (progress - startTime) / (endTime - startTime)
            : CGFloat(1)
        let parameter: CGFloat
        if calculationMode == .discrete {
            parameter = 0
        } else if let timingFunctions, segmentIndex < timingFunctions.count {
            parameter = CGFloat(timingFunctions[segmentIndex].evaluate(at: Float(rawParameter)))
        } else {
            parameter = rawParameter
        }
        return Sample(
            point: segment.point(at: parameter),
            tangent: segment.tangent(at: parameter)
        )
    }

    private func validatedKeyTimes(_ keyTimes: [CGFloat]?) -> [CGFloat]? {
        guard let keyTimes,
              keyTimes.count == segments.count + 1,
              keyTimes.first == 0,
              keyTimes.last == 1,
              keyTimes.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
              zip(keyTimes, keyTimes.dropFirst()).allSatisfy({ $0 <= $1 }) else {
            return nil
        }
        return keyTimes
    }

    private func defaultKeyTimes() -> [CGFloat] {
        let divisor = CGFloat(segments.count)
        return (0...segments.count).map { CGFloat($0) / divisor }
    }

    private static func isFinite(_ point: CGPoint) -> Bool {
        point.x.isFinite && point.y.isFinite
    }
}
