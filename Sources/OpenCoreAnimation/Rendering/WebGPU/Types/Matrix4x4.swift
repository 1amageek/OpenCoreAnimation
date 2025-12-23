#if arch(wasm32)
import Foundation

// MARK: - WASM Matrix Types (simd replacement)

/// A 4x4 matrix of Float values for WASM environments.
/// This replaces simd_float4x4 which is not available on WASM.
public struct Matrix4x4 {
    public var columns: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)

    public init(columns: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)) {
        self.columns = columns
    }

    /// Identity matrix
    public static var identity: Matrix4x4 {
        Matrix4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }

    /// Creates a translation matrix.
    public init(translation: SIMD3<Float>) {
        self = .identity
        self.columns.3 = SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    }

    /// Creates an orthographic projection matrix for WebGPU (depth range 0 to 1).
    ///
    /// WebGPU uses a depth range of [0, 1] in clip space, unlike OpenGL which uses [-1, 1].
    /// This matrix maps:
    /// - X: [left, right] → [-1, 1]
    /// - Y: [bottom, top] → [-1, 1]
    /// - Z: [near, far] → [0, 1]
    public static func orthographic(
        left: Float,
        right: Float,
        bottom: Float,
        top: Float,
        near: Float,
        far: Float
    ) -> Matrix4x4 {
        let width = right - left
        let height = top - bottom
        let depth = far - near

        // WebGPU depth range [0, 1]:
        // z_ndc = (z_eye - near) / (far - near)
        //       = z_eye / depth - near / depth
        return Matrix4x4(columns: (
            SIMD4<Float>(2 / width, 0, 0, 0),
            SIMD4<Float>(0, 2 / height, 0, 0),
            SIMD4<Float>(0, 0, 1 / depth, 0),
            SIMD4<Float>(-(right + left) / width, -(top + bottom) / height, -near / depth, 1)
        ))
    }

    /// Matrix multiplication
    public static func * (lhs: Matrix4x4, rhs: Matrix4x4) -> Matrix4x4 {
        var result = Matrix4x4.identity

        for i in 0..<4 {
            let col = getColumn(rhs, i)
            let x = lhs.columns.0 * col.x
            let y = lhs.columns.1 * col.y
            let z = lhs.columns.2 * col.z
            let w = lhs.columns.3 * col.w
            setColumn(&result, i, x + y + z + w)
        }

        return result
    }

    private static func getColumn(_ m: Matrix4x4, _ i: Int) -> SIMD4<Float> {
        switch i {
        case 0: return m.columns.0
        case 1: return m.columns.1
        case 2: return m.columns.2
        case 3: return m.columns.3
        default: return .zero
        }
    }

    private static func setColumn(_ m: inout Matrix4x4, _ i: Int, _ v: SIMD4<Float>) {
        switch i {
        case 0: m.columns.0 = v
        case 1: m.columns.1 = v
        case 2: m.columns.2 = v
        case 3: m.columns.3 = v
        default: break
        }
    }

    /// Matrix-vector multiplication
    public static func * (lhs: Matrix4x4, rhs: SIMD4<Float>) -> SIMD4<Float> {
        let x = lhs.columns.0 * rhs.x
        let y = lhs.columns.1 * rhs.y
        let z = lhs.columns.2 * rhs.z
        let w = lhs.columns.3 * rhs.w
        return x + y + z + w
    }
}

#endif
