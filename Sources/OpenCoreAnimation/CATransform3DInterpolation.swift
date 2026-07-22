//
//  CATransform3DInterpolation.swift
//  OpenCoreAnimation
//
//  Decomposition-based interpolation for CATransform3D.
//

import Foundation

/// Decomposition-based interpolation for `CATransform3D`, adapted from the
/// W3C CSS Transforms Level 2 algorithm for CATransform3D's row-vector
/// storage convention (translation at m41/m42/m43, rotation matrix rows are
/// the images of the basis vectors).
///
/// Element-wise linear interpolation of a 4x4 matrix produces invalid
/// intermediate matrices whenever a rotation is involved — the rows lose
/// orthonormality, so the animated layer appears to shear and shrink instead
/// of rotating. Decomposing into translation, rotation (as a quaternion),
/// scale, skew, and perspective lets each component be interpolated on its
/// own manifold (slerp for rotation, linear for everything else) and then
/// recomposed into a valid transform at each frame.
internal enum CATransform3DInterpolation {

    /// Synthetic transform key paths exposed by Core Animation. These values
    /// are read from and written back through matrix decomposition so changing
    /// one component preserves the remaining translation, rotation, scale,
    /// skew, and perspective components.
    static func componentValue(for keyPath: String, in transform: CATransform3D) -> Any? {
        guard let decomposition = decompose(transform) else { return nil }
        let rotation = eulerAngles(from: decomposition.quaternion)

        switch keyPath {
        case "transform.scale":
            return (decomposition.scale.x + decomposition.scale.y + decomposition.scale.z) / 3
        case "transform.scale.x":
            return decomposition.scale.x
        case "transform.scale.y":
            return decomposition.scale.y
        case "transform.scale.z":
            return decomposition.scale.z
        case "transform.rotation", "transform.rotation.z":
            return rotation.z
        case "transform.rotation.x":
            return rotation.x
        case "transform.rotation.y":
            return rotation.y
        case "transform.translation":
            return CGSize(width: decomposition.translation.x, height: decomposition.translation.y)
        case "transform.translation.x":
            return decomposition.translation.x
        case "transform.translation.y":
            return decomposition.translation.y
        case "transform.translation.z":
            return decomposition.translation.z
        default:
            return nil
        }
    }

    static func applyingComponent(
        _ value: Any,
        for keyPath: String,
        to transform: CATransform3D,
        additive: Bool
    ) -> CATransform3D? {
        guard var decomposition = decompose(transform) else { return nil }

        switch keyPath {
        case "transform.scale":
            guard let scalar = scalarValue(value) else { return nil }
            if additive {
                decomposition.scale.x += scalar
                decomposition.scale.y += scalar
                decomposition.scale.z += scalar
            } else {
                decomposition.scale = (scalar, scalar, scalar)
            }
        case "transform.scale.x":
            guard let scalar = scalarValue(value) else { return nil }
            decomposition.scale.x = additive ? decomposition.scale.x + scalar : scalar
        case "transform.scale.y":
            guard let scalar = scalarValue(value) else { return nil }
            decomposition.scale.y = additive ? decomposition.scale.y + scalar : scalar
        case "transform.scale.z":
            guard let scalar = scalarValue(value) else { return nil }
            decomposition.scale.z = additive ? decomposition.scale.z + scalar : scalar
        case "transform.rotation", "transform.rotation.z":
            guard let scalar = scalarValue(value) else { return nil }
            var rotation = eulerAngles(from: decomposition.quaternion)
            rotation.z = additive ? rotation.z + scalar : scalar
            guard let quaternion = quaternion(from: rotation) else { return nil }
            decomposition.quaternion = quaternion
        case "transform.rotation.x":
            guard let scalar = scalarValue(value) else { return nil }
            var rotation = eulerAngles(from: decomposition.quaternion)
            rotation.x = additive ? rotation.x + scalar : scalar
            guard let quaternion = quaternion(from: rotation) else { return nil }
            decomposition.quaternion = quaternion
        case "transform.rotation.y":
            guard let scalar = scalarValue(value) else { return nil }
            var rotation = eulerAngles(from: decomposition.quaternion)
            rotation.y = additive ? rotation.y + scalar : scalar
            guard let quaternion = quaternion(from: rotation) else { return nil }
            decomposition.quaternion = quaternion
        case "transform.translation":
            guard let size = value as? CGSize else { return nil }
            decomposition.translation.x = additive
                ? decomposition.translation.x + size.width
                : size.width
            decomposition.translation.y = additive
                ? decomposition.translation.y + size.height
                : size.height
        case "transform.translation.x":
            guard let scalar = scalarValue(value) else { return nil }
            decomposition.translation.x = additive ? decomposition.translation.x + scalar : scalar
        case "transform.translation.y":
            guard let scalar = scalarValue(value) else { return nil }
            decomposition.translation.y = additive ? decomposition.translation.y + scalar : scalar
        case "transform.translation.z":
            guard let scalar = scalarValue(value) else { return nil }
            decomposition.translation.z = additive ? decomposition.translation.z + scalar : scalar
        default:
            return nil
        }

        return recompose(decomposition)
    }

    /// Components of a decomposed transform, one component per independently
    /// interpolatable degree of freedom.
    struct Decomposition {
        var translation: (x: CGFloat, y: CGFloat, z: CGFloat)
        var scale: (x: CGFloat, y: CGFloat, z: CGFloat)
        /// Shear factors: xy contributes row 0 into row 1, xz and yz contribute
        /// row 0 and row 1 into row 2.
        var skew: (xy: CGFloat, xz: CGFloat, yz: CGFloat)
        /// Perspective column of the matrix (m14, m24, m34, m44 after solving
        /// the perspective equation).
        var perspective: (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat)
        /// Unit quaternion (x, y, z, w).
        var quaternion: (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat)
    }

    /// Interpolates between two transforms by decomposing each, interpolating
    /// the components, and recomposing the result. Falls back to element-wise
    /// linear interpolation only when a matrix is singular and cannot be
    /// decomposed.
    static func interpolate(from: CATransform3D, to: CATransform3D, progress: CGFloat) -> CATransform3D {
        if progress <= 0 { return from }
        if progress >= 1 { return to }

        guard let fromDecomp = decompose(from),
              let toDecomp = decompose(to) else {
            return elementWiseLerp(from: from, to: to, progress: progress)
        }

        let interp = interpolate(from: fromDecomp, to: toDecomp, progress: progress)
        return recompose(interp)
    }

    // MARK: - Decomposition

    static func decompose(_ transform: CATransform3D) -> Decomposition? {
        guard transform.m44 != 0 else { return nil }
        let inv44 = 1.0 / transform.m44

        let m11 = transform.m11 * inv44
        let m12 = transform.m12 * inv44
        let m13 = transform.m13 * inv44
        let m14 = transform.m14 * inv44
        let m21 = transform.m21 * inv44
        let m22 = transform.m22 * inv44
        let m23 = transform.m23 * inv44
        let m24 = transform.m24 * inv44
        let m31 = transform.m31 * inv44
        let m32 = transform.m32 * inv44
        let m33 = transform.m33 * inv44
        let m34 = transform.m34 * inv44
        let m41 = transform.m41 * inv44
        let m42 = transform.m42 * inv44
        let m43 = transform.m43 * inv44

        let perspFree = CATransform3D(
            m11: m11, m12: m12, m13: m13, m14: 0,
            m21: m21, m22: m22, m23: m23, m24: 0,
            m31: m31, m32: m32, m33: m33, m34: 0,
            m41: m41, m42: m42, m43: m43, m44: 1
        )
        guard matrix4Determinant(perspFree) != 0 else { return nil }

        let perspective: (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat)
        if m14 != 0 || m24 != 0 || m34 != 0 {
            let inv = CATransform3DInvert(perspFree)
            let rx = m14, ry = m24, rz = m34
            let rw: CGFloat = 1
            perspective = (
                x: rx * inv.m11 + ry * inv.m21 + rz * inv.m31 + rw * inv.m41,
                y: rx * inv.m12 + ry * inv.m22 + rz * inv.m32 + rw * inv.m42,
                z: rx * inv.m13 + ry * inv.m23 + rz * inv.m33 + rw * inv.m43,
                w: rx * inv.m14 + ry * inv.m24 + rz * inv.m34 + rw * inv.m44
            )
        } else {
            perspective = (0, 0, 0, 1)
        }

        let translation = (x: m41, y: m42, z: m43)

        var r0x = m11, r0y = m12, r0z = m13
        var r1x = m21, r1y = m22, r1z = m23
        var r2x = m31, r2y = m32, r2z = m33

        var sx = sqrt(r0x * r0x + r0y * r0y + r0z * r0z)
        if sx != 0 { r0x /= sx; r0y /= sx; r0z /= sx }

        var skewXY = r0x * r1x + r0y * r1y + r0z * r1z
        r1x -= skewXY * r0x
        r1y -= skewXY * r0y
        r1z -= skewXY * r0z

        var sy = sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
        if sy != 0 {
            r1x /= sy; r1y /= sy; r1z /= sy
            skewXY /= sy
        }

        var skewXZ = r0x * r2x + r0y * r2y + r0z * r2z
        r2x -= skewXZ * r0x
        r2y -= skewXZ * r0y
        r2z -= skewXZ * r0z

        var skewYZ = r1x * r2x + r1y * r2y + r1z * r2z
        r2x -= skewYZ * r1x
        r2y -= skewYZ * r1y
        r2z -= skewYZ * r1z

        var sz = sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
        if sz != 0 {
            r2x /= sz; r2y /= sz; r2z /= sz
            skewXZ /= sz
            skewYZ /= sz
        }

        // If det(rotation) < 0 the coordinate system was flipped; negate
        // scales and rows to restore a right-handed rotation.
        let crossX = r1y * r2z - r1z * r2y
        let crossY = r1z * r2x - r1x * r2z
        let crossZ = r1x * r2y - r1y * r2x
        if r0x * crossX + r0y * crossY + r0z * crossZ < 0 {
            sx = -sx; sy = -sy; sz = -sz
            r0x = -r0x; r0y = -r0y; r0z = -r0z
            r1x = -r1x; r1y = -r1y; r1z = -r1z
            r2x = -r2x; r2y = -r2y; r2z = -r2z
        }

        var qx = 0.5 * sqrt(max(1 + r0x - r1y - r2z, 0))
        var qy = 0.5 * sqrt(max(1 - r0x + r1y - r2z, 0))
        var qz = 0.5 * sqrt(max(1 - r0x - r1y + r2z, 0))
        let qw = 0.5 * sqrt(max(1 + r0x + r1y + r2z, 0))

        if r2y > r1z { qx = -qx }
        if r0z > r2x { qy = -qy }
        if r1x > r0y { qz = -qz }

        return Decomposition(
            translation: translation,
            scale: (x: sx, y: sy, z: sz),
            skew: (xy: skewXY, xz: skewXZ, yz: skewYZ),
            perspective: perspective,
            quaternion: (x: qx, y: qy, z: qz, w: qw)
        )
    }

    // MARK: - Interpolation

    static func interpolate(from a: Decomposition, to b: Decomposition, progress t: CGFloat) -> Decomposition {
        return Decomposition(
            translation: (
                x: lerp(a.translation.x, b.translation.x, t),
                y: lerp(a.translation.y, b.translation.y, t),
                z: lerp(a.translation.z, b.translation.z, t)
            ),
            scale: (
                x: lerp(a.scale.x, b.scale.x, t),
                y: lerp(a.scale.y, b.scale.y, t),
                z: lerp(a.scale.z, b.scale.z, t)
            ),
            skew: (
                xy: lerp(a.skew.xy, b.skew.xy, t),
                xz: lerp(a.skew.xz, b.skew.xz, t),
                yz: lerp(a.skew.yz, b.skew.yz, t)
            ),
            perspective: (
                x: lerp(a.perspective.x, b.perspective.x, t),
                y: lerp(a.perspective.y, b.perspective.y, t),
                z: lerp(a.perspective.z, b.perspective.z, t),
                w: lerp(a.perspective.w, b.perspective.w, t)
            ),
            quaternion: slerp(a.quaternion, b.quaternion, t)
        )
    }

    // MARK: - Recomposition

    static func recompose(_ d: Decomposition) -> CATransform3D {
        var m = CATransform3DIdentity

        m.m14 = d.perspective.x
        m.m24 = d.perspective.y
        m.m34 = d.perspective.z
        m.m44 = d.perspective.w

        let tx = d.translation.x, ty = d.translation.y, tz = d.translation.z
        m.m41 += tx * m.m11 + ty * m.m21 + tz * m.m31
        m.m42 += tx * m.m12 + ty * m.m22 + tz * m.m32
        m.m43 += tx * m.m13 + ty * m.m23 + tz * m.m33
        m.m44 += tx * m.m14 + ty * m.m24 + tz * m.m34

        // Rotation in row-vector form (transpose of the CSS spec's column-vector
        // rotation matrix); required because CATransform3D applies as p' = p * M.
        let qx = d.quaternion.x, qy = d.quaternion.y
        let qz = d.quaternion.z, qw = d.quaternion.w
        var rotation = CATransform3DIdentity
        rotation.m11 = 1 - 2 * (qy * qy + qz * qz)
        rotation.m12 = 2 * (qx * qy + qz * qw)
        rotation.m13 = 2 * (qx * qz - qy * qw)
        rotation.m21 = 2 * (qx * qy - qz * qw)
        rotation.m22 = 1 - 2 * (qx * qx + qz * qz)
        rotation.m23 = 2 * (qy * qz + qx * qw)
        rotation.m31 = 2 * (qx * qz + qy * qw)
        rotation.m32 = 2 * (qy * qz - qx * qw)
        rotation.m33 = 1 - 2 * (qx * qx + qy * qy)

        m = CATransform3DConcat(rotation, m)

        if d.skew.yz != 0 {
            var s = CATransform3DIdentity
            s.m32 = d.skew.yz
            m = CATransform3DConcat(s, m)
        }
        if d.skew.xz != 0 {
            var s = CATransform3DIdentity
            s.m31 = d.skew.xz
            m = CATransform3DConcat(s, m)
        }
        if d.skew.xy != 0 {
            var s = CATransform3DIdentity
            s.m21 = d.skew.xy
            m = CATransform3DConcat(s, m)
        }

        m.m11 *= d.scale.x; m.m12 *= d.scale.x; m.m13 *= d.scale.x
        m.m21 *= d.scale.y; m.m22 *= d.scale.y; m.m23 *= d.scale.y
        m.m31 *= d.scale.z; m.m32 *= d.scale.z; m.m33 *= d.scale.z

        return m
    }

    private static func scalarValue(_ value: Any) -> CGFloat? {
        if let value = value as? CGFloat { return value }
        if let value = value as? Double { return CGFloat(value) }
        if let value = value as? Float { return CGFloat(value) }
        return nil
    }

    /// Extracts the XYZ Euler representation used by Core Animation's
    /// transform.rotation component key paths. The corresponding composition
    /// order is Rx * Ry * Rz for CATransform3D's row-vector convention.
    private static func eulerAngles(
        from quaternion: (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat)
    ) -> (x: CGFloat, y: CGFloat, z: CGFloat) {
        let qx = quaternion.x
        let qy = quaternion.y
        let qz = quaternion.z
        let qw = quaternion.w
        let r11 = 1 - 2 * (qy * qy + qz * qz)
        let r12 = 2 * (qx * qy + qz * qw)
        let r13 = 2 * (qx * qz - qy * qw)
        let r21 = 2 * (qx * qy - qz * qw)
        let r22 = 1 - 2 * (qx * qx + qz * qz)
        let r23 = 2 * (qy * qz + qx * qw)
        let r33 = 1 - 2 * (qx * qx + qy * qy)

        let y = asin(max(-1, min(1, -r13)))
        let cosY = cos(y)
        if abs(cosY) > 1e-8 {
            return (atan2(r23, r33), y, atan2(r12, r11))
        }

        // At gimbal lock infinitely many X/Z pairs represent the same
        // rotation. Canonicalizing Z to zero preserves the matrix exactly.
        let x = y > 0 ? atan2(r21, r22) : atan2(-r21, r22)
        return (x, y, 0)
    }

    private static func quaternion(
        from rotation: (x: CGFloat, y: CGFloat, z: CGFloat)
    ) -> (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat)? {
        let transform = CATransform3DConcat(
            CATransform3DConcat(
                CATransform3DMakeRotation(rotation.x, 1, 0, 0),
                CATransform3DMakeRotation(rotation.y, 0, 1, 0)
            ),
            CATransform3DMakeRotation(rotation.z, 0, 0, 1)
        )
        return decompose(transform)?.quaternion
    }

    // MARK: - Helpers

    private static func lerp(_ a: CGFloat, _ b: CGFloat, _ t: CGFloat) -> CGFloat {
        return a + (b - a) * t
    }

    private static func slerp(
        _ a: (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat),
        _ b: (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat),
        _ t: CGFloat
    ) -> (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat) {
        var cosHalfTheta = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w

        var bx = b.x, by = b.y, bz = b.z, bw = b.w
        if cosHalfTheta < 0 {
            cosHalfTheta = -cosHalfTheta
            bx = -bx; by = -by; bz = -bz; bw = -bw
        }

        if cosHalfTheta >= 1 - 1e-6 {
            let x = a.x + (bx - a.x) * t
            let y = a.y + (by - a.y) * t
            let z = a.z + (bz - a.z) * t
            let w = a.w + (bw - a.w) * t
            return normalizeQuaternion(x: x, y: y, z: z, w: w)
        }

        let halfTheta = acos(min(max(cosHalfTheta, -1), 1))
        let sinHalfTheta = sin(halfTheta)
        let wa = sin((1 - t) * halfTheta) / sinHalfTheta
        let wb = sin(t * halfTheta) / sinHalfTheta

        return (
            x: a.x * wa + bx * wb,
            y: a.y * wa + by * wb,
            z: a.z * wa + bz * wb,
            w: a.w * wa + bw * wb
        )
    }

    private static func normalizeQuaternion(
        x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat
    ) -> (x: CGFloat, y: CGFloat, z: CGFloat, w: CGFloat) {
        let length = sqrt(x * x + y * y + z * z + w * w)
        if length == 0 { return (0, 0, 0, 1) }
        return (x / length, y / length, z / length, w / length)
    }

    private static func matrix4Determinant(_ t: CATransform3D) -> CGFloat {
        return t.m11 * (t.m22 * (t.m33 * t.m44 - t.m34 * t.m43) - t.m23 * (t.m32 * t.m44 - t.m34 * t.m42) + t.m24 * (t.m32 * t.m43 - t.m33 * t.m42))
             - t.m12 * (t.m21 * (t.m33 * t.m44 - t.m34 * t.m43) - t.m23 * (t.m31 * t.m44 - t.m34 * t.m41) + t.m24 * (t.m31 * t.m43 - t.m33 * t.m41))
             + t.m13 * (t.m21 * (t.m32 * t.m44 - t.m34 * t.m42) - t.m22 * (t.m31 * t.m44 - t.m34 * t.m41) + t.m24 * (t.m31 * t.m42 - t.m32 * t.m41))
             - t.m14 * (t.m21 * (t.m32 * t.m43 - t.m33 * t.m42) - t.m22 * (t.m31 * t.m43 - t.m33 * t.m41) + t.m23 * (t.m31 * t.m42 - t.m32 * t.m41))
    }

    private static func elementWiseLerp(
        from: CATransform3D, to: CATransform3D, progress: CGFloat
    ) -> CATransform3D {
        return CATransform3D(
            m11: lerp(from.m11, to.m11, progress),
            m12: lerp(from.m12, to.m12, progress),
            m13: lerp(from.m13, to.m13, progress),
            m14: lerp(from.m14, to.m14, progress),
            m21: lerp(from.m21, to.m21, progress),
            m22: lerp(from.m22, to.m22, progress),
            m23: lerp(from.m23, to.m23, progress),
            m24: lerp(from.m24, to.m24, progress),
            m31: lerp(from.m31, to.m31, progress),
            m32: lerp(from.m32, to.m32, progress),
            m33: lerp(from.m33, to.m33, progress),
            m34: lerp(from.m34, to.m34, progress),
            m41: lerp(from.m41, to.m41, progress),
            m42: lerp(from.m42, to.m42, progress),
            m43: lerp(from.m43, to.m43, progress),
            m44: lerp(from.m44, to.m44, progress)
        )
    }
}
