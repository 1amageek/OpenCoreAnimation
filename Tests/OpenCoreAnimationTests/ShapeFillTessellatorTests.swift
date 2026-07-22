import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Shape fill tessellation")
struct ShapeFillTessellatorTests {
    @Test("Even-odd and non-zero rules distinguish same-winding holes")
    func sameWindingContours() throws {
        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 10, height: 10))
        path.addRect(CGRect(x: 3, y: 3, width: 4, height: 4))

        let nonZero = try ShapeFillTessellator.triangles(for: path, rule: .nonZero)
        let evenOdd = try ShapeFillTessellator.triangles(for: path, rule: .evenOdd)

        #expect(contains(CGPoint(x: 1, y: 1), triangles: nonZero))
        #expect(contains(CGPoint(x: 1, y: 1), triangles: evenOdd))
        #expect(contains(CGPoint(x: 5, y: 5), triangles: nonZero))
        #expect(!contains(CGPoint(x: 5, y: 5), triangles: evenOdd))
    }

    @Test("Opposite winding creates a non-zero hole")
    func oppositeWindingContours() throws {
        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 10, height: 10))
        path.move(to: CGPoint(x: 3, y: 3))
        path.addLine(to: CGPoint(x: 3, y: 7))
        path.addLine(to: CGPoint(x: 7, y: 7))
        path.addLine(to: CGPoint(x: 7, y: 3))
        path.closeSubpath()

        let triangles = try ShapeFillTessellator.triangles(for: path, rule: .nonZero)

        #expect(contains(CGPoint(x: 1, y: 1), triangles: triangles))
        #expect(!contains(CGPoint(x: 5, y: 5), triangles: triangles))
    }

    @Test("Even-odd removes the overlap between independent contours")
    func overlappingContours() throws {
        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 8, height: 8))
        path.addRect(CGRect(x: 4, y: 0, width: 8, height: 8))

        let nonZero = try ShapeFillTessellator.triangles(for: path, rule: .nonZero)
        let evenOdd = try ShapeFillTessellator.triangles(for: path, rule: .evenOdd)

        #expect(contains(CGPoint(x: 2, y: 4), triangles: evenOdd))
        #expect(contains(CGPoint(x: 6, y: 4), triangles: nonZero))
        #expect(!contains(CGPoint(x: 6, y: 4), triangles: evenOdd))
        #expect(contains(CGPoint(x: 10, y: 4), triangles: evenOdd))
    }

    @Test("Coincident contours preserve winding and cancel parity")
    func coincidentContours() throws {
        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 10, height: 10))
        path.addRect(CGRect(x: 0, y: 0, width: 10, height: 10))

        let nonZero = try ShapeFillTessellator.triangles(for: path, rule: .nonZero)
        let evenOdd = try ShapeFillTessellator.triangles(for: path, rule: .evenOdd)

        #expect(contains(CGPoint(x: 5, y: 5), triangles: nonZero))
        #expect(evenOdd.isEmpty)
    }

    @Test("Self intersections split at crossing boundaries")
    func selfIntersectingContour() throws {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 0, y: 0))
        path.addLine(to: CGPoint(x: 10, y: 10))
        path.addLine(to: CGPoint(x: 0, y: 10))
        path.addLine(to: CGPoint(x: 10, y: 0))
        path.closeSubpath()

        for rule in [CAShapeLayerFillRule.nonZero, .evenOdd] {
            let triangles = try ShapeFillTessellator.triangles(for: path, rule: rule)
            #expect(contains(CGPoint(x: 5, y: 2), triangles: triangles))
            #expect(contains(CGPoint(x: 5, y: 8), triangles: triangles))
            #expect(!contains(CGPoint(x: 2, y: 5), triangles: triangles))
        }
    }

    @Test("Curves and open subpaths produce finite fill triangles")
    func curveAndOpenSubpath() throws {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 0, y: 0))
        path.addCurve(
            to: CGPoint(x: 10, y: 0),
            control1: CGPoint(x: 0, y: 10),
            control2: CGPoint(x: 10, y: 10)
        )
        path.addLine(to: CGPoint(x: 5, y: -5))

        let triangles = try ShapeFillTessellator.triangles(for: path, rule: .nonZero)

        #expect(!triangles.isEmpty)
        #expect(triangles.allSatisfy { $0.x.isFinite && $0.y.isFinite })
        #expect(contains(CGPoint(x: 5, y: 2), triangles: triangles))
    }

    @Test("Unknown fill rules and non-finite paths fail explicitly")
    func invalidInput() {
        let validPath = CGPath(rect: CGRect(x: 0, y: 0, width: 10, height: 10), transform: nil)
        #expect(throws: ShapeFillTessellationError.unsupportedFillRule("future-rule")) {
            try ShapeFillTessellator.triangles(
                for: validPath,
                rule: CAShapeLayerFillRule(rawValue: "future-rule")
            )
        }

        let invalidPath = CGMutablePath()
        invalidPath.move(to: CGPoint(x: CGFloat.nan, y: 0))
        invalidPath.addLine(to: CGPoint(x: 10, y: 0))
        invalidPath.addLine(to: CGPoint(x: 0, y: 10))
        invalidPath.closeSubpath()
        #expect(throws: ShapeFillTessellationError.nonFinitePath) {
            try ShapeFillTessellator.validate(invalidPath)
        }
        #expect(throws: ShapeFillTessellationError.nonFinitePath) {
            try ShapeFillTessellator.triangles(for: invalidPath, rule: .nonZero)
        }
    }

    private func contains(_ point: CGPoint, triangles: [CGPoint]) -> Bool {
        stride(from: 0, to: triangles.count, by: 3).contains { index in
            pointInTriangle(
                point,
                triangles[index],
                triangles[index + 1],
                triangles[index + 2]
            )
        }
    }

    private func pointInTriangle(
        _ point: CGPoint,
        _ first: CGPoint,
        _ second: CGPoint,
        _ third: CGPoint
    ) -> Bool {
        let firstSign = cross(point, first, second)
        let secondSign = cross(point, second, third)
        let thirdSign = cross(point, third, first)
        let hasNegative = firstSign < -1e-9 || secondSign < -1e-9 || thirdSign < -1e-9
        let hasPositive = firstSign > 1e-9 || secondSign > 1e-9 || thirdSign > 1e-9
        return !(hasNegative && hasPositive)
    }

    private func cross(_ point: CGPoint, _ first: CGPoint, _ second: CGPoint) -> CGFloat {
        (point.x - second.x) * (first.y - second.y)
            - (first.x - second.x) * (point.y - second.y)
    }
}
