#if arch(wasm32)
import Foundation
import OpenCoreGraphics

// MARK: - WASM Helper Extensions

extension CALayer {
    /// Converts the layer's background color to SIMD4<Float>.
    internal var backgroundColorComponents: SIMD4<Float> {
        guard let color = backgroundColor,
              let components = color.components,
              components.count >= 4 else {
            return SIMD4<Float>(0, 0, 0, 0)
        }
        return SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
    }

    /// Converts the layer's border color to SIMD4<Float>.
    internal var borderColorComponents: SIMD4<Float> {
        guard let color = borderColor,
              let components = color.components,
              components.count >= 4 else {
            return SIMD4<Float>(0, 0, 0, 0)
        }
        return SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
    }

    /// Calculates the model matrix for this layer.
    ///
    /// The z-coordinate is negated to match CoreAnimation's convention where
    /// higher zPosition values appear "in front" (closer to the viewer).
    /// With WebGPU's depth range [0, 1] and lessEqual comparison:
    /// - Lower z in eye space → lower clip z → passes depth test → in front
    /// - So we negate zPosition: higher zPosition → lower z_eye → in front
    internal func modelMatrix(parentMatrix: Matrix4x4 = .identity) -> Matrix4x4 {
        var matrix = parentMatrix

        // Negate zPosition so higher values appear in front (CoreAnimation convention)
        let translation = Matrix4x4(translation: SIMD3<Float>(
            Float(position.x),
            Float(position.y),
            Float(-zPosition)  // Negated for correct z-ordering
        ))
        matrix = matrix * translation

        if !CATransform3DIsIdentity(transform) {
            let layerTransform = transform.matrix4x4
            matrix = matrix * layerTransform
        }

        // Negate anchorPointZ to match the z-coordinate convention
        let anchorOffset = Matrix4x4(translation: SIMD3<Float>(
            Float(-bounds.width * anchorPoint.x),
            Float(-bounds.height * anchorPoint.y),
            Float(anchorPointZ)  // Negated (double negation with the minus sign)
        ))
        matrix = matrix * anchorOffset

        return matrix
    }

    /// Calculates the parent matrix for sublayer positioning.
    ///
    /// This includes the layer's model matrix, sublayerTransform, and bounds.origin offset.
    /// The bounds.origin offset ensures that sublayers are correctly positioned when the
    /// layer's bounds origin is non-zero (e.g., for CAScrollLayer scrolling).
    ///
    /// - Parameter modelMatrix: The layer's model matrix
    /// - Returns: The matrix to use as parentMatrix for sublayer rendering
    internal func sublayerMatrix(modelMatrix: Matrix4x4) -> Matrix4x4 {
        var result = modelMatrix

        // Apply sublayerTransform if not identity
        if !CATransform3DIsIdentity(sublayerTransform) {
            result = result * sublayerTransform.matrix4x4
        }

        // Apply bounds.origin offset
        // In CoreAnimation, bounds.origin defines where the coordinate system origin is
        // within the layer. A sublayer at position (0,0) with parent's bounds.origin = (50, 50)
        // should appear at (-50, -50) relative to the parent's visible top-left.
        // This is the scrolling behavior used by CAScrollLayer.
        if bounds.origin.x != 0 || bounds.origin.y != 0 {
            let boundsOriginOffset = Matrix4x4(translation: SIMD3<Float>(
                Float(-bounds.origin.x),
                Float(-bounds.origin.y),
                0
            ))
            result = result * boundsOriginOffset
        }

        return result
    }
}

extension CATransform3D {
    /// Converts CATransform3D to Matrix4x4.
    internal var matrix4x4: Matrix4x4 {
        return Matrix4x4(columns: (
            SIMD4<Float>(Float(m11), Float(m21), Float(m31), Float(m41)),
            SIMD4<Float>(Float(m12), Float(m22), Float(m32), Float(m42)),
            SIMD4<Float>(Float(m13), Float(m23), Float(m33), Float(m43)),
            SIMD4<Float>(Float(m14), Float(m24), Float(m34), Float(m44))
        ))
    }
}

#endif
