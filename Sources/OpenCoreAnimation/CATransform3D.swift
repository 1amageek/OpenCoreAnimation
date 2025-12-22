//
//  CATransform3D.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics


/// The standard transform matrix used throughout Core Animation.
///
/// The transform matrix is used to rotate, scale, translate, skew, and project the layer content.
/// Functions are provided for creating, concatenating, and modifying CATransform3D data.
public struct CATransform3D: Sendable {
    /// The entry at position 1,1 in the matrix.
    public var m11: CGFloat
    /// The entry at position 1,2 in the matrix.
    public var m12: CGFloat
    /// The entry at position 1,3 in the matrix.
    public var m13: CGFloat
    /// The entry at position 1,4 in the matrix.
    public var m14: CGFloat
    /// The entry at position 2,1 in the matrix.
    public var m21: CGFloat
    /// The entry at position 2,2 in the matrix.
    public var m22: CGFloat
    /// The entry at position 2,3 in the matrix.
    public var m23: CGFloat
    /// The entry at position 2,4 in the matrix.
    public var m24: CGFloat
    /// The entry at position 3,1 in the matrix.
    public var m31: CGFloat
    /// The entry at position 3,2 in the matrix.
    public var m32: CGFloat
    /// The entry at position 3,3 in the matrix.
    public var m33: CGFloat
    /// The entry at position 3,4 in the matrix.
    public var m34: CGFloat
    /// The entry at position 4,1 in the matrix.
    public var m41: CGFloat
    /// The entry at position 4,2 in the matrix.
    public var m42: CGFloat
    /// The entry at position 4,3 in the matrix.
    public var m43: CGFloat
    /// The entry at position 4,4 in the matrix.
    public var m44: CGFloat

    public init() {
        m11 = 1; m12 = 0; m13 = 0; m14 = 0
        m21 = 0; m22 = 1; m23 = 0; m24 = 0
        m31 = 0; m32 = 0; m33 = 1; m34 = 0
        m41 = 0; m42 = 0; m43 = 0; m44 = 1
    }

    public init(
        m11: CGFloat, m12: CGFloat, m13: CGFloat, m14: CGFloat,
        m21: CGFloat, m22: CGFloat, m23: CGFloat, m24: CGFloat,
        m31: CGFloat, m32: CGFloat, m33: CGFloat, m34: CGFloat,
        m41: CGFloat, m42: CGFloat, m43: CGFloat, m44: CGFloat
    ) {
        self.m11 = m11; self.m12 = m12; self.m13 = m13; self.m14 = m14
        self.m21 = m21; self.m22 = m22; self.m23 = m23; self.m24 = m24
        self.m31 = m31; self.m32 = m32; self.m33 = m33; self.m34 = m34
        self.m41 = m41; self.m42 = m42; self.m43 = m43; self.m44 = m44
    }
}

extension CATransform3D: Equatable {
    public static func == (lhs: CATransform3D, rhs: CATransform3D) -> Bool {
        return lhs.m11 == rhs.m11 && lhs.m12 == rhs.m12 && lhs.m13 == rhs.m13 && lhs.m14 == rhs.m14 &&
               lhs.m21 == rhs.m21 && lhs.m22 == rhs.m22 && lhs.m23 == rhs.m23 && lhs.m24 == rhs.m24 &&
               lhs.m31 == rhs.m31 && lhs.m32 == rhs.m32 && lhs.m33 == rhs.m33 && lhs.m34 == rhs.m34 &&
               lhs.m41 == rhs.m41 && lhs.m42 == rhs.m42 && lhs.m43 == rhs.m43 && lhs.m44 == rhs.m44
    }
}

// MARK: - Identity Transform

/// The identity transform: [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1].
public let CATransform3DIdentity = CATransform3D()

// MARK: - Creating Transforms

/// Returns a transform that translates by (tx, ty, tz).
public func CATransform3DMakeTranslation(_ tx: CGFloat, _ ty: CGFloat, _ tz: CGFloat) -> CATransform3D {
    var t = CATransform3DIdentity
    t.m41 = tx
    t.m42 = ty
    t.m43 = tz
    return t
}

/// Returns a transform that scales by (sx, sy, sz).
public func CATransform3DMakeScale(_ sx: CGFloat, _ sy: CGFloat, _ sz: CGFloat) -> CATransform3D {
    var t = CATransform3DIdentity
    t.m11 = sx
    t.m22 = sy
    t.m33 = sz
    return t
}

/// Returns a transform that rotates by `angle` radians about the vector (x, y, z).
public func CATransform3DMakeRotation(_ angle: CGFloat, _ x: CGFloat, _ y: CGFloat, _ z: CGFloat) -> CATransform3D {
    let length = sqrt(x * x + y * y + z * z)
    guard length > 0 else { return CATransform3DIdentity }

    let nx = x / length
    let ny = y / length
    let nz = z / length

    let c = cos(angle)
    let s = sin(angle)
    let t = 1 - c

    return CATransform3D(
        m11: t * nx * nx + c,      m12: t * nx * ny + nz * s, m13: t * nx * nz - ny * s, m14: 0,
        m21: t * nx * ny - nz * s, m22: t * ny * ny + c,      m23: t * ny * nz + nx * s, m24: 0,
        m31: t * nx * nz + ny * s, m32: t * ny * nz - nx * s, m33: t * nz * nz + c,      m34: 0,
        m41: 0,                    m42: 0,                    m43: 0,                    m44: 1
    )
}

// MARK: - Chaining Transforms

/// Concatenates b to a and returns the result: t = a * b.
public func CATransform3DConcat(_ a: CATransform3D, _ b: CATransform3D) -> CATransform3D {
    return CATransform3D(
        m11: a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31 + a.m14 * b.m41,
        m12: a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32 + a.m14 * b.m42,
        m13: a.m11 * b.m13 + a.m12 * b.m23 + a.m13 * b.m33 + a.m14 * b.m43,
        m14: a.m11 * b.m14 + a.m12 * b.m24 + a.m13 * b.m34 + a.m14 * b.m44,
        m21: a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31 + a.m24 * b.m41,
        m22: a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32 + a.m24 * b.m42,
        m23: a.m21 * b.m13 + a.m22 * b.m23 + a.m23 * b.m33 + a.m24 * b.m43,
        m24: a.m21 * b.m14 + a.m22 * b.m24 + a.m23 * b.m34 + a.m24 * b.m44,
        m31: a.m31 * b.m11 + a.m32 * b.m21 + a.m33 * b.m31 + a.m34 * b.m41,
        m32: a.m31 * b.m12 + a.m32 * b.m22 + a.m33 * b.m32 + a.m34 * b.m42,
        m33: a.m31 * b.m13 + a.m32 * b.m23 + a.m33 * b.m33 + a.m34 * b.m43,
        m34: a.m31 * b.m14 + a.m32 * b.m24 + a.m33 * b.m34 + a.m34 * b.m44,
        m41: a.m41 * b.m11 + a.m42 * b.m21 + a.m43 * b.m31 + a.m44 * b.m41,
        m42: a.m41 * b.m12 + a.m42 * b.m22 + a.m43 * b.m32 + a.m44 * b.m42,
        m43: a.m41 * b.m13 + a.m42 * b.m23 + a.m43 * b.m33 + a.m44 * b.m43,
        m44: a.m41 * b.m14 + a.m42 * b.m24 + a.m43 * b.m34 + a.m44 * b.m44
    )
}

/// Translates t by (tx, ty, tz) and returns the result: t' = t * translate(tx, ty, tz).
///
/// The translation is applied after the existing transform, meaning points are first
/// transformed by t, then translated.
public func CATransform3DTranslate(_ t: CATransform3D, _ tx: CGFloat, _ ty: CGFloat, _ tz: CGFloat) -> CATransform3D {
    return CATransform3DConcat(t, CATransform3DMakeTranslation(tx, ty, tz))
}

/// Scales t by (sx, sy, sz) and returns the result: t' = t * scale(sx, sy, sz).
///
/// The scaling is applied after the existing transform, meaning points are first
/// transformed by t, then scaled.
public func CATransform3DScale(_ t: CATransform3D, _ sx: CGFloat, _ sy: CGFloat, _ sz: CGFloat) -> CATransform3D {
    return CATransform3DConcat(t, CATransform3DMakeScale(sx, sy, sz))
}

/// Rotates t by `angle` radians about the vector (x, y, z) and returns the result: t' = t * rotate(angle, x, y, z).
///
/// The rotation is applied after the existing transform, meaning points are first
/// transformed by t, then rotated.
public func CATransform3DRotate(_ t: CATransform3D, _ angle: CGFloat, _ x: CGFloat, _ y: CGFloat, _ z: CGFloat) -> CATransform3D {
    return CATransform3DConcat(t, CATransform3DMakeRotation(angle, x, y, z))
}

// MARK: - Inverting a Transform

/// Inverts t and returns the result.
public func CATransform3DInvert(_ t: CATransform3D) -> CATransform3D {
    // Calculate the determinant
    let det = t.m11 * (t.m22 * (t.m33 * t.m44 - t.m34 * t.m43) - t.m23 * (t.m32 * t.m44 - t.m34 * t.m42) + t.m24 * (t.m32 * t.m43 - t.m33 * t.m42))
             - t.m12 * (t.m21 * (t.m33 * t.m44 - t.m34 * t.m43) - t.m23 * (t.m31 * t.m44 - t.m34 * t.m41) + t.m24 * (t.m31 * t.m43 - t.m33 * t.m41))
             + t.m13 * (t.m21 * (t.m32 * t.m44 - t.m34 * t.m42) - t.m22 * (t.m31 * t.m44 - t.m34 * t.m41) + t.m24 * (t.m31 * t.m42 - t.m32 * t.m41))
             - t.m14 * (t.m21 * (t.m32 * t.m43 - t.m33 * t.m42) - t.m22 * (t.m31 * t.m43 - t.m33 * t.m41) + t.m23 * (t.m31 * t.m42 - t.m32 * t.m41))

    guard det != 0 else { return t }

    let invDet = 1 / det

    // Calculate the adjugate matrix and multiply by 1/det
    return CATransform3D(
        m11: invDet * (t.m22 * (t.m33 * t.m44 - t.m34 * t.m43) - t.m23 * (t.m32 * t.m44 - t.m34 * t.m42) + t.m24 * (t.m32 * t.m43 - t.m33 * t.m42)),
        m12: invDet * -(t.m12 * (t.m33 * t.m44 - t.m34 * t.m43) - t.m13 * (t.m32 * t.m44 - t.m34 * t.m42) + t.m14 * (t.m32 * t.m43 - t.m33 * t.m42)),
        m13: invDet * (t.m12 * (t.m23 * t.m44 - t.m24 * t.m43) - t.m13 * (t.m22 * t.m44 - t.m24 * t.m42) + t.m14 * (t.m22 * t.m43 - t.m23 * t.m42)),
        m14: invDet * -(t.m12 * (t.m23 * t.m34 - t.m24 * t.m33) - t.m13 * (t.m22 * t.m34 - t.m24 * t.m32) + t.m14 * (t.m22 * t.m33 - t.m23 * t.m32)),
        m21: invDet * -(t.m21 * (t.m33 * t.m44 - t.m34 * t.m43) - t.m23 * (t.m31 * t.m44 - t.m34 * t.m41) + t.m24 * (t.m31 * t.m43 - t.m33 * t.m41)),
        m22: invDet * (t.m11 * (t.m33 * t.m44 - t.m34 * t.m43) - t.m13 * (t.m31 * t.m44 - t.m34 * t.m41) + t.m14 * (t.m31 * t.m43 - t.m33 * t.m41)),
        m23: invDet * -(t.m11 * (t.m23 * t.m44 - t.m24 * t.m43) - t.m13 * (t.m21 * t.m44 - t.m24 * t.m41) + t.m14 * (t.m21 * t.m43 - t.m23 * t.m41)),
        m24: invDet * (t.m11 * (t.m23 * t.m34 - t.m24 * t.m33) - t.m13 * (t.m21 * t.m34 - t.m24 * t.m31) + t.m14 * (t.m21 * t.m33 - t.m23 * t.m31)),
        m31: invDet * (t.m21 * (t.m32 * t.m44 - t.m34 * t.m42) - t.m22 * (t.m31 * t.m44 - t.m34 * t.m41) + t.m24 * (t.m31 * t.m42 - t.m32 * t.m41)),
        m32: invDet * -(t.m11 * (t.m32 * t.m44 - t.m34 * t.m42) - t.m12 * (t.m31 * t.m44 - t.m34 * t.m41) + t.m14 * (t.m31 * t.m42 - t.m32 * t.m41)),
        m33: invDet * (t.m11 * (t.m22 * t.m44 - t.m24 * t.m42) - t.m12 * (t.m21 * t.m44 - t.m24 * t.m41) + t.m14 * (t.m21 * t.m42 - t.m22 * t.m41)),
        m34: invDet * -(t.m11 * (t.m22 * t.m34 - t.m24 * t.m32) - t.m12 * (t.m21 * t.m34 - t.m24 * t.m31) + t.m14 * (t.m21 * t.m32 - t.m22 * t.m31)),
        m41: invDet * -(t.m21 * (t.m32 * t.m43 - t.m33 * t.m42) - t.m22 * (t.m31 * t.m43 - t.m33 * t.m41) + t.m23 * (t.m31 * t.m42 - t.m32 * t.m41)),
        m42: invDet * (t.m11 * (t.m32 * t.m43 - t.m33 * t.m42) - t.m12 * (t.m31 * t.m43 - t.m33 * t.m41) + t.m13 * (t.m31 * t.m42 - t.m32 * t.m41)),
        m43: invDet * -(t.m11 * (t.m22 * t.m43 - t.m23 * t.m42) - t.m12 * (t.m21 * t.m43 - t.m23 * t.m41) + t.m13 * (t.m21 * t.m42 - t.m22 * t.m41)),
        m44: invDet * (t.m11 * (t.m22 * t.m33 - t.m23 * t.m32) - t.m12 * (t.m21 * t.m33 - t.m23 * t.m31) + t.m13 * (t.m21 * t.m32 - t.m22 * t.m31))
    )
}

// MARK: - Determining Transform Properties

/// Returns a Boolean value that indicates whether a transform can be exactly represented by an affine transform.
public func CATransform3DIsAffine(_ t: CATransform3D) -> Bool {
    return t.m13 == 0 && t.m14 == 0 &&
           t.m23 == 0 && t.m24 == 0 &&
           t.m31 == 0 && t.m32 == 0 && t.m33 == 1 && t.m34 == 0 &&
           t.m43 == 0 && t.m44 == 1
}

/// Returns a Boolean value that indicates whether the transform is the identity transform.
public func CATransform3DIsIdentity(_ t: CATransform3D) -> Bool {
    return t == CATransform3DIdentity
}

/// Returns a Boolean value that indicates whether the two transforms are exactly equal.
public func CATransform3DEqualToTransform(_ a: CATransform3D, _ b: CATransform3D) -> Bool {
    return a == b
}

// MARK: - Converting to and from Core Graphics Affine Transforms

/// Returns a transform with the same effect as affine transform m.
public func CATransform3DMakeAffineTransform(_ m: CGAffineTransform) -> CATransform3D {
    return CATransform3D(
        m11: m.a,  m12: m.b,  m13: 0, m14: 0,
        m21: m.c,  m22: m.d,  m23: 0, m24: 0,
        m31: 0,    m32: 0,    m33: 1, m34: 0,
        m41: m.tx, m42: m.ty, m43: 0, m44: 1
    )
}

/// Returns the affine transform represented by t.
public func CATransform3DGetAffineTransform(_ t: CATransform3D) -> CGAffineTransform {
    return CGAffineTransform(a: t.m11, b: t.m12, c: t.m21, d: t.m22, tx: t.m41, ty: t.m42)
}
