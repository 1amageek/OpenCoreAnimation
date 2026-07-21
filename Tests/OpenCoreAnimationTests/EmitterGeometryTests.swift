import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("Emitter Geometry Tests")
struct EmitterGeometryTests {
    private static let tolerance: Float = 0.0001

    @Test("Point shape always emits from its documented center")
    func pointShapeUsesCenter() throws {
        for mode in modes {
            var random = EmitterRandomSource(seed: 1)
            let point = try #require(EmitterGeometry.position(
                shape: .point,
                mode: mode,
                position: CGPoint(x: 3, y: 4),
                zPosition: 5,
                size: CGSize(width: 20, height: 30),
                depth: 40,
                random: &random
            ))
            #expect(point == SIMD3(3, 4, 5))
        }
    }

    @Test("Line and rectangle modes select endpoints, outlines, and interiors")
    func planarModesRespectBoundaries() throws {
        var random = EmitterRandomSource(seed: 7)
        for _ in 0..<64 {
            let linePoint = try sample(
                shape: .line,
                mode: .points,
                size: CGSize(width: 20, height: 10),
                depth: 0,
                random: &random
            )
            #expect(abs(abs(linePoint.x) - 10) < Self.tolerance)
            #expect(linePoint.y == 0)

            let outline = try sample(
                shape: .rectangle,
                mode: .outline,
                size: CGSize(width: 20, height: 10),
                depth: 0,
                random: &random
            )
            let onVerticalEdge = abs(abs(outline.x) - 10) < Self.tolerance
            let onHorizontalEdge = abs(abs(outline.y) - 5) < Self.tolerance
            #expect(onVerticalEdge || onHorizontalEdge)
            #expect(abs(outline.x) <= 10 + Self.tolerance)
            #expect(abs(outline.y) <= 5 + Self.tolerance)

            let interior = try sample(
                shape: .rectangle,
                mode: .volume,
                size: CGSize(width: 20, height: 10),
                depth: 0,
                random: &random
            )
            #expect(abs(interior.x) <= 10)
            #expect(abs(interior.y) <= 5)
            #expect(interior.z == 0)
        }
    }

    @Test("Cuboid modes select vertices, edges, faces, and volume")
    func cuboidModesRespectDimensions() throws {
        var random = EmitterRandomSource(seed: 19)
        for _ in 0..<64 {
            let point = try sample(
                shape: .cuboid,
                mode: .points,
                size: CGSize(width: 20, height: 10),
                depth: 6,
                random: &random
            )
            #expect(abs(abs(point.x) - 10) < Self.tolerance)
            #expect(abs(abs(point.y) - 5) < Self.tolerance)
            #expect(abs(abs(point.z) - 3) < Self.tolerance)

            let edge = try sample(
                shape: .cuboid,
                mode: .outline,
                size: CGSize(width: 20, height: 10),
                depth: 6,
                random: &random
            )
            let edgeBoundaryCount = [
                abs(abs(edge.x) - 10),
                abs(abs(edge.y) - 5),
                abs(abs(edge.z) - 3),
            ].count { $0 < Self.tolerance }
            #expect(edgeBoundaryCount == 2)

            let surface = try sample(
                shape: .cuboid,
                mode: .surface,
                size: CGSize(width: 20, height: 10),
                depth: 6,
                random: &random
            )
            let onFace = abs(abs(surface.x) - 10) < Self.tolerance
                || abs(abs(surface.y) - 5) < Self.tolerance
                || abs(abs(surface.z) - 3) < Self.tolerance
            #expect(onFace)

            let volume = try sample(
                shape: .cuboid,
                mode: .volume,
                size: CGSize(width: 20, height: 10),
                depth: 6,
                random: &random
            )
            #expect(abs(volume.x) <= 10)
            #expect(abs(volume.y) <= 5)
            #expect(abs(volume.z) <= 3)
        }
    }

    @Test("Circle and sphere use emitterSize.width as their radius")
    func radialShapesUseDocumentedRadius() throws {
        var random = EmitterRandomSource(seed: 23)
        for _ in 0..<64 {
            let circle = try sample(
                shape: .circle,
                mode: .outline,
                size: CGSize(width: 12, height: 2),
                depth: 1,
                random: &random
            )
            #expect(abs(length(circle) - 12) < Self.tolerance)
            #expect(circle.z == 0)

            let disk = try sample(
                shape: .circle,
                mode: .surface,
                size: CGSize(width: 12, height: 2),
                depth: 1,
                random: &random
            )
            #expect(length(disk) <= 12 + Self.tolerance)

            let sphere = try sample(
                shape: .sphere,
                mode: .surface,
                size: CGSize(width: 12, height: 2),
                depth: 1,
                random: &random
            )
            #expect(abs(length(sphere) - 12) < Self.tolerance)

            let volume = try sample(
                shape: .sphere,
                mode: .volume,
                size: CGSize(width: 12, height: 2),
                depth: 1,
                random: &random
            )
            #expect(length(volume) <= 12 + Self.tolerance)
        }
    }

    @Test("Longitude and colatitude define the three-dimensional emission axis")
    func directionUsesSphericalAngles() throws {
        var random = EmitterRandomSource(seed: 29)
        let z = try #require(EmitterGeometry.direction(
            longitude: 0,
            latitude: 0,
            range: 0,
            random: &random
        ))
        let x = try #require(EmitterGeometry.direction(
            longitude: 0,
            latitude: .pi / 2,
            range: 0,
            random: &random
        ))
        let y = try #require(EmitterGeometry.direction(
            longitude: .pi / 2,
            latitude: .pi / 2,
            range: 0,
            random: &random
        ))

        #expect(approximatelyEqual(z, SIMD3(0, 0, 1)))
        #expect(approximatelyEqual(x, SIMD3(1, 0, 0)))
        #expect(approximatelyEqual(y, SIMD3(0, 1, 0)))
    }

    @Test("Emission range is sampled uniformly inside its cone")
    func directionStaysInsideCone() throws {
        var random = EmitterRandomSource(seed: 31)
        let halfAngle = Float.pi / 4
        let minimumCosine = cos(halfAngle)

        for _ in 0..<256 {
            let direction = try #require(EmitterGeometry.direction(
                longitude: 0,
                latitude: 0,
                range: .pi / 2,
                random: &random
            ))
            #expect(abs(length(direction) - 1) < Self.tolerance)
            #expect(direction.z >= minimumCosine - Self.tolerance)
        }
    }

    @Test("Unknown shape or mode rejects the particle instead of falling back")
    func unknownGeometryFailsExplicitly() {
        var random = EmitterRandomSource(seed: 37)
        let unknownShape = EmitterGeometry.position(
            shape: CAEmitterLayerEmitterShape(rawValue: "unsupported"),
            mode: .volume,
            position: .zero,
            zPosition: 0,
            size: .zero,
            depth: 0,
            random: &random
        )
        let unknownMode = EmitterGeometry.position(
            shape: .point,
            mode: CAEmitterLayerEmitterMode(rawValue: "unsupported"),
            position: .zero,
            zPosition: 0,
            size: .zero,
            depth: 0,
            random: &random
        )

        #expect(unknownShape == nil)
        #expect(unknownMode == nil)
    }

    @Test("Non-finite geometry and angles reject the particle")
    func nonFiniteInputsFailExplicitly() {
        var random = EmitterRandomSource(seed: 41)
        let invalidPosition = EmitterGeometry.position(
            shape: .rectangle,
            mode: .volume,
            position: CGPoint(x: CGFloat.nan, y: 0),
            zPosition: 0,
            size: CGSize(width: 10, height: 10),
            depth: 0,
            random: &random
        )
        let invalidSize = EmitterGeometry.position(
            shape: .sphere,
            mode: .surface,
            position: .zero,
            zPosition: 0,
            size: CGSize(width: CGFloat.infinity, height: 0),
            depth: 0,
            random: &random
        )
        let invalidDirection = EmitterGeometry.direction(
            longitude: 0,
            latitude: CGFloat.nan,
            range: 0,
            random: &random
        )

        #expect(invalidPosition == nil)
        #expect(invalidSize == nil)
        #expect(invalidDirection == nil)
    }

    private var modes: [CAEmitterLayerEmitterMode] {
        [.points, .outline, .surface, .volume]
    }

    private func sample(
        shape: CAEmitterLayerEmitterShape,
        mode: CAEmitterLayerEmitterMode,
        size: CGSize,
        depth: CGFloat,
        random: inout EmitterRandomSource
    ) throws -> SIMD3<Float> {
        try #require(EmitterGeometry.position(
            shape: shape,
            mode: mode,
            position: .zero,
            zPosition: 0,
            size: size,
            depth: depth,
            random: &random
        ))
    }

    private func length(_ value: SIMD3<Float>) -> Float {
        sqrt(value.x * value.x + value.y * value.y + value.z * value.z)
    }

    private func approximatelyEqual(
        _ lhs: SIMD3<Float>,
        _ rhs: SIMD3<Float>
    ) -> Bool {
        abs(lhs.x - rhs.x) < Self.tolerance
            && abs(lhs.y - rhs.y) < Self.tolerance
            && abs(lhs.z - rhs.z) < Self.tolerance
    }
}
