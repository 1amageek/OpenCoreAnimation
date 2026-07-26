import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("Committed Path Snapshot Tests")
struct CACommittedPathTests {
    private enum Element: Equatable {
        case move(CGPoint)
        case line(CGPoint)
        case quadratic(CGPoint, CGPoint)
        case cubic(CGPoint, CGPoint, CGPoint)
        case close
    }

    @Test("Capture owns every command independently of the source path")
    func captureIsImmutable() throws {
        let source = CGMutablePath()
        source.move(to: CGPoint(x: 1, y: 2))
        source.addLine(to: CGPoint(x: 3, y: 4))
        source.addQuadCurve(
            to: CGPoint(x: 7, y: 8),
            control: CGPoint(x: 5, y: 6)
        )
        source.addCurve(
            to: CGPoint(x: 13, y: 14),
            control1: CGPoint(x: 9, y: 10),
            control2: CGPoint(x: 11, y: 12)
        )
        source.closeSubpath()

        let snapshot = try CACommittedPath(capturing: source)
        source.addRect(
            CGRect(x: 100, y: 100, width: 20, height: 20)
        )

        #expect(elements(of: snapshot.materialize()) == [
            .move(CGPoint(x: 1, y: 2)),
            .line(CGPoint(x: 3, y: 4)),
            .quadratic(
                CGPoint(x: 5, y: 6),
                CGPoint(x: 7, y: 8)
            ),
            .cubic(
                CGPoint(x: 9, y: 10),
                CGPoint(x: 11, y: 12),
                CGPoint(x: 13, y: 14)
            ),
            .close,
        ])
    }

    @Test("Capture rejects a non-finite command")
    func captureRejectsNonFinitePath() {
        let source = CGMutablePath()
        source.move(to: CGPoint(x: CGFloat.infinity, y: 0))

        #expect(
            throws: CACommittedAnimationCaptureError.nonFiniteValue(
                "CGPath"
            )
        ) {
            _ = try CACommittedPath(capturing: source)
        }
    }

    private func elements(of path: CGPath) -> [Element] {
        var result: [Element] = []
        path.applyWithBlock { elementPointer in
            let element = elementPointer.pointee
            switch element.type {
            case .moveToPoint:
                if let points = element.points {
                    result.append(.move(points[0]))
                }
            case .addLineToPoint:
                if let points = element.points {
                    result.append(.line(points[0]))
                }
            case .addQuadCurveToPoint:
                if let points = element.points {
                    result.append(
                        .quadratic(points[0], points[1])
                    )
                }
            case .addCurveToPoint:
                if let points = element.points {
                    result.append(
                        .cubic(
                            points[0],
                            points[1],
                            points[2]
                        )
                    )
                }
            case .closeSubpath:
                result.append(.close)
            @unknown default:
                break
            }
        }
        return result
    }
}
