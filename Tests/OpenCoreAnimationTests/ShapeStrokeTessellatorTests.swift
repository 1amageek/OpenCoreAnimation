import Foundation
import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Shape stroke tessellation")
struct ShapeStrokeTessellatorTests {
    @Test("Stroke start and end trim by total path length")
    func trimsStroke() throws {
        let path = line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 100, y: 0))
        let triangles = try tessellate(path, start: 0.25, end: 0.75)

        #expect(!contains(CGPoint(x: 20, y: 0), triangles: triangles))
        #expect(contains(CGPoint(x: 50, y: 0), triangles: triangles))
        #expect(!contains(CGPoint(x: 80, y: 0), triangles: triangles))
    }

    @Test("Dash pattern alternates painted path intervals")
    func dashPattern() throws {
        let triangles = try tessellate(
            line(from: .zero, to: CGPoint(x: 60, y: 0)),
            dashPattern: [10, 10]
        )

        #expect(contains(CGPoint(x: 5, y: 0), triangles: triangles))
        #expect(!contains(CGPoint(x: 15, y: 0), triangles: triangles))
        #expect(contains(CGPoint(x: 25, y: 0), triangles: triangles))
    }

    @Test("Trimmed dashes retain phase from the original subpath")
    func trimmedDashPhase() throws {
        let triangles = try tessellate(
            line(from: .zero, to: CGPoint(x: 100, y: 0)),
            dashPattern: [10, 10],
            start: 0.1,
            end: 0.4
        )

        #expect(!contains(CGPoint(x: 12, y: 0), triangles: triangles))
        #expect(contains(CGPoint(x: 22, y: 0), triangles: triangles))
    }

    @Test("Stroke trimming spans subpaths without resetting fractions")
    func trimsAcrossSubpaths() throws {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 0, y: 0))
        path.addLine(to: CGPoint(x: 100, y: 0))
        path.move(to: CGPoint(x: 0, y: 20))
        path.addLine(to: CGPoint(x: 100, y: 20))
        let triangles = try tessellate(path, start: 0.25, end: 0.75)

        #expect(!contains(CGPoint(x: 40, y: 0), triangles: triangles))
        #expect(contains(CGPoint(x: 75, y: 0), triangles: triangles))
        #expect(contains(CGPoint(x: 25, y: 20), triangles: triangles))
        #expect(!contains(CGPoint(x: 60, y: 20), triangles: triangles))
    }

    @Test("Cap styles use OpenCoreGraphics stroke outlines")
    func capStyles() throws {
        let path = line(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 20, y: 0))
        let butt = try tessellate(path, cap: .butt)
        let round = try tessellate(path, cap: .round)
        let square = try tessellate(path, cap: .square)

        #expect(!contains(CGPoint(x: -2, y: 0), triangles: butt))
        #expect(contains(CGPoint(x: -2, y: 0), triangles: round))
        #expect(contains(CGPoint(x: -2, y: 0), triangles: square))
    }

    @Test("Join styles and miter limits reach OpenCoreGraphics outlines")
    func joinStylesAndMiterLimit() throws {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 0, y: 0))
        path.addLine(to: CGPoint(x: 20, y: 0))
        path.addLine(to: CGPoint(x: 20, y: 20))
        let outsideBevel = CGPoint(x: 21.5, y: -1.5)

        let miter = try tessellate(path, lineWidth: 4, join: .miter, miterLimit: 10)
        let limited = try tessellate(path, lineWidth: 4, join: .miter, miterLimit: 1)
        let bevel = try tessellate(path, lineWidth: 4, join: .bevel)
        let round = try tessellate(path, lineWidth: 4, join: .round)

        #expect(contains(outsideBevel, triangles: miter))
        #expect(!contains(outsideBevel, triangles: limited))
        #expect(!contains(outsideBevel, triangles: bevel))
        #expect(!contains(outsideBevel, triangles: round))
    }

    @Test("Invalid geometry and unknown styles fail explicitly")
    func rejectsInvalidInputs() {
        let path = line(from: .zero, to: CGPoint(x: 20, y: 0))
        #expect(throws: ShapeStrokeTessellationError.invalidDashPattern) {
            try tessellate(path, dashPattern: [4, 0])
        }
        #expect(throws: ShapeStrokeTessellationError.invalidGeometry) {
            try tessellate(path, lineWidth: .nan)
        }
        #expect(throws: ShapeStrokeTessellationError.unsupportedLineCap("future-cap")) {
            try tessellate(path, cap: CAShapeLayerLineCap(rawValue: "future-cap"))
        }
        #expect(throws: ShapeStrokeTessellationError.unsupportedLineJoin("future-join")) {
            try tessellate(path, join: CAShapeLayerLineJoin(rawValue: "future-join"))
        }
    }

    private func line(from start: CGPoint, to end: CGPoint) -> CGPath {
        let path = CGMutablePath()
        path.move(to: start)
        path.addLine(to: end)
        return path
    }

    private func tessellate(
        _ path: CGPath,
        lineWidth: CGFloat = 10,
        cap: CAShapeLayerLineCap = .butt,
        join: CAShapeLayerLineJoin = .miter,
        miterLimit: CGFloat = 10,
        dashPattern: [CGFloat]? = nil,
        start: CGFloat = 0,
        end: CGFloat = 1
    ) throws -> [CGPoint] {
        try ShapeStrokeTessellator.triangles(
            for: path,
            lineWidth: lineWidth,
            lineCap: cap,
            lineJoin: join,
            miterLimit: miterLimit,
            dashPattern: dashPattern,
            dashPhase: 0,
            strokeStart: start,
            strokeEnd: end
        )
    }

    private func contains(_ point: CGPoint, triangles: [CGPoint]) -> Bool {
        stride(from: 0, to: triangles.count, by: 3).contains { index in
            let a = cross(point, triangles[index], triangles[index + 1])
            let b = cross(point, triangles[index + 1], triangles[index + 2])
            let c = cross(point, triangles[index + 2], triangles[index])
            return !([a, b, c].contains { $0 < -1e-9 }
                && [a, b, c].contains { $0 > 1e-9 })
        }
    }

    private func cross(_ point: CGPoint, _ first: CGPoint, _ second: CGPoint) -> CGFloat {
        (point.x - second.x) * (first.y - second.y)
            - (first.x - second.x) * (point.y - second.y)
    }
}
