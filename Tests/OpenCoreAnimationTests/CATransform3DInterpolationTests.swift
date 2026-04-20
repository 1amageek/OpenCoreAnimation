import Testing
import Foundation
#if canImport(CoreGraphics)
import CoreGraphics
#else
import OpenCoreGraphics
#endif
@testable import OpenCoreAnimation

@Suite("CATransform3DInterpolation")
struct CATransform3DInterpolationTests {

    // MARK: - Helpers

    private static let epsilon: CGFloat = 1e-6

    private static func isApproximatelyEqual(_ a: CATransform3D, _ b: CATransform3D, tolerance: CGFloat = epsilon) -> Bool {
        let deltas: [CGFloat] = [
            abs(a.m11 - b.m11), abs(a.m12 - b.m12), abs(a.m13 - b.m13), abs(a.m14 - b.m14),
            abs(a.m21 - b.m21), abs(a.m22 - b.m22), abs(a.m23 - b.m23), abs(a.m24 - b.m24),
            abs(a.m31 - b.m31), abs(a.m32 - b.m32), abs(a.m33 - b.m33), abs(a.m34 - b.m34),
            abs(a.m41 - b.m41), abs(a.m42 - b.m42), abs(a.m43 - b.m43), abs(a.m44 - b.m44)
        ]
        return deltas.allSatisfy { $0 <= tolerance }
    }

    // MARK: - Endpoint behaviour

    @Test("Progress 0 returns from transform")
    func progressZero() {
        let from = CATransform3DMakeRotation(.pi / 4, 0, 0, 1)
        let to = CATransform3DMakeRotation(.pi, 0, 0, 1)
        let result = CATransform3DInterpolation.interpolate(from: from, to: to, progress: 0)
        #expect(Self.isApproximatelyEqual(result, from))
    }

    @Test("Progress 1 returns to transform")
    func progressOne() {
        let from = CATransform3DMakeRotation(.pi / 4, 0, 0, 1)
        let to = CATransform3DMakeRotation(.pi, 0, 0, 1)
        let result = CATransform3DInterpolation.interpolate(from: from, to: to, progress: 1)
        #expect(Self.isApproximatelyEqual(result, to))
    }

    // MARK: - Round-trip decomposition

    @Test("Decompose and recompose of identity is identity")
    func decomposeRecomposeIdentity() throws {
        let decomp = try #require(CATransform3DInterpolation.decompose(CATransform3DIdentity))
        let reconstructed = CATransform3DInterpolation.recompose(decomp)
        #expect(Self.isApproximatelyEqual(reconstructed, CATransform3DIdentity))
    }

    @Test("Decompose and recompose of translation is preserved")
    func decomposeRecomposeTranslation() throws {
        let original = CATransform3DMakeTranslation(10, 20, 30)
        let decomp = try #require(CATransform3DInterpolation.decompose(original))
        let reconstructed = CATransform3DInterpolation.recompose(decomp)
        #expect(Self.isApproximatelyEqual(reconstructed, original))
    }

    @Test("Decompose and recompose of Z rotation is preserved")
    func decomposeRecomposeRotationZ() throws {
        let original = CATransform3DMakeRotation(.pi / 3, 0, 0, 1)
        let decomp = try #require(CATransform3DInterpolation.decompose(original))
        let reconstructed = CATransform3DInterpolation.recompose(decomp)
        #expect(Self.isApproximatelyEqual(reconstructed, original))
    }

    @Test("Decompose and recompose of X rotation is preserved")
    func decomposeRecomposeRotationX() throws {
        let original = CATransform3DMakeRotation(.pi / 4, 1, 0, 0)
        let decomp = try #require(CATransform3DInterpolation.decompose(original))
        let reconstructed = CATransform3DInterpolation.recompose(decomp)
        #expect(Self.isApproximatelyEqual(reconstructed, original))
    }

    @Test("Decompose and recompose of scale is preserved")
    func decomposeRecomposeScale() throws {
        let original = CATransform3DMakeScale(2, 3, 4)
        let decomp = try #require(CATransform3DInterpolation.decompose(original))
        let reconstructed = CATransform3DInterpolation.recompose(decomp)
        #expect(Self.isApproximatelyEqual(reconstructed, original))
    }

    @Test("Decompose and recompose of combined transform is preserved")
    func decomposeRecomposeCombined() throws {
        // Translate, then rotate, then scale — a realistic case where naive
        // element-wise interpolation would fail.
        var m = CATransform3DMakeTranslation(10, 20, 0)
        m = CATransform3DRotate(m, .pi / 3, 0, 0, 1)
        m = CATransform3DScale(m, 2, 2, 1)

        let decomp = try #require(CATransform3DInterpolation.decompose(m))
        let reconstructed = CATransform3DInterpolation.recompose(decomp)
        #expect(Self.isApproximatelyEqual(reconstructed, m, tolerance: 1e-5))
    }

    // MARK: - Rotation interpolation correctness

    @Test("Rotation midpoint is a valid rotation matrix")
    func rotationMidpointIsValid() {
        let from = CATransform3DIdentity
        let to = CATransform3DMakeRotation(.pi / 2, 0, 0, 1)
        let mid = CATransform3DInterpolation.interpolate(from: from, to: to, progress: 0.5)

        // Halfway between 0 and 90 degrees is 45 degrees.
        let expected = CATransform3DMakeRotation(.pi / 4, 0, 0, 1)
        #expect(Self.isApproximatelyEqual(mid, expected, tolerance: 1e-5))

        // The midpoint must still have orthonormal rows (det == 1 for a rotation).
        // Naive element-wise lerp produces rows of length cos(pi/4) ≈ 0.707,
        // which is the bug this class fixes.
        let row0Length = sqrt(mid.m11 * mid.m11 + mid.m12 * mid.m12 + mid.m13 * mid.m13)
        #expect(abs(row0Length - 1) < 1e-5)
    }

    @Test("180 degree rotation interpolates along the short arc")
    func slerpShortArc() {
        // Rotating from 0 to 180 around Z: midpoint should be 90 degrees.
        let from = CATransform3DIdentity
        let to = CATransform3DMakeRotation(.pi, 0, 0, 1)
        let mid = CATransform3DInterpolation.interpolate(from: from, to: to, progress: 0.5)

        let expected = CATransform3DMakeRotation(.pi / 2, 0, 0, 1)
        #expect(Self.isApproximatelyEqual(mid, expected, tolerance: 1e-5))
    }

    // MARK: - Translation + rotation interpolation

    @Test("Translation interpolates linearly while rotation slerps")
    func combinedTranslationRotationMidpoint() {
        // Destination matrix has m41..m43 = (0, 100, 0) and upper-left 3x3 = R(pi/2, Z).
        // Build as R then Translate so concat order (row-vector) is R * T — the translation
        // ends up directly in m41..m43 without being pre-rotated, which matches how the
        // industry-standard decomposition (Unmatrix / CSS) extracts the translation component.
        let from = CATransform3DIdentity
        var to = CATransform3DMakeRotation(.pi / 2, 0, 0, 1)
        to = CATransform3DTranslate(to, 0, 100, 0)

        let mid = CATransform3DInterpolation.interpolate(from: from, to: to, progress: 0.5)

        // At progress 0.5: translation lerps to (0, 50, 0), rotation slerps to pi/4 around Z.
        var expected = CATransform3DMakeRotation(.pi / 4, 0, 0, 1)
        expected = CATransform3DTranslate(expected, 0, 50, 0)
        #expect(Self.isApproximatelyEqual(mid, expected, tolerance: 1e-5))
    }
}
