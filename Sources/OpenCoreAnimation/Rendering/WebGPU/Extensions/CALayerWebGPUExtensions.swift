#if arch(wasm32)
import Foundation

// MARK: - WASM Helper Extensions

extension CALayer {
    /// Raw edge-antialiasing bits consumed by the WebGPU fragment shaders.
    internal var edgeAntialiasingMaskValue: Float {
        guard allowsEdgeAntialiasing else { return 0 }
        return Float(edgeAntialiasingMask.rawValue & 0xF)
    }

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

    /// Calculates per-corner radii based on maskedCorners and cornerRadius.
    ///
    /// Returns a SIMD4<Float> with the radius for each corner:
    /// - x: minXminY (bottom-left)
    /// - y: maxXminY (bottom-right)
    /// - z: minXmaxY (top-left)
    /// - w: maxXmaxY (top-right)
    ///
    /// If maskedCorners contains all four corners (or is empty with a non-zero cornerRadius),
    /// all four values will be the same. Otherwise, unmasked corners will have 0 radius.
    internal var cornerRadiiComponents: SIMD4<Float> {
        let radius = Float(cornerRadius)

        // If cornerRadius is 0, return all zeros regardless of mask
        guard radius > 0 else {
            return .zero
        }

        // Check which corners should have the radius applied
        let minXminY: Float = maskedCorners.contains(.layerMinXMinYCorner) ? radius : 0
        let maxXminY: Float = maskedCorners.contains(.layerMaxXMinYCorner) ? radius : 0
        let minXmaxY: Float = maskedCorners.contains(.layerMinXMaxYCorner) ? radius : 0
        let maxXmaxY: Float = maskedCorners.contains(.layerMaxXMaxYCorner) ? radius : 0

        return SIMD4<Float>(minXminY, maxXminY, minXmaxY, maxXmaxY)
    }

    /// Calculates the model matrix for this layer.
    ///
    /// Normal layer trees still use stable painter ordering, while transform-layer
    /// descendants additionally consume the generated z coordinate in the depth buffer.
    internal func modelMatrix(parentMatrix: Matrix4x4 = .identity) -> Matrix4x4 {
        var matrix = parentMatrix

        let translation = Matrix4x4(translation: SIMD3<Float>(
            Float(position.x),
            Float(position.y),
            Float(zPosition)
        ))
        matrix = matrix * translation

        if !CATransform3DIsIdentity(transform) {
            let layerTransform = transform.matrix4x4
            matrix = matrix * layerTransform
        }

        let anchorOffset = Matrix4x4(translation: SIMD3<Float>(
            Float(-bounds.width * anchorPoint.x),
            Float(-bounds.height * anchorPoint.y),
            Float(-anchorPointZ)
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

        // Convert the layer's bounds coordinate space into its drawable plane.
        // A geometry-flipped layer reflects descendant geometry around the
        // horizontal midpoint of its bounds without changing its own contents.
        if isGeometryFlipped {
            let flippedBoundsTransform = Matrix4x4(translation: SIMD3<Float>(
                0,
                Float(bounds.height),
                0
            )) * Matrix4x4(columns: (
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, -1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))
                * Matrix4x4(translation: SIMD3<Float>(
                    Float(-bounds.origin.x),
                    Float(-bounds.origin.y),
                    0
                ))
            result = result * flippedBoundsTransform
        } else if bounds.origin.x != 0 || bounds.origin.y != 0 {
            result = result * Matrix4x4(translation: SIMD3<Float>(
                Float(-bounds.origin.x),
                Float(-bounds.origin.y),
                0
            ))
        }

        return result
    }
}

extension CATransform3D {
    /// Converts CATransform3D to Matrix4x4.
    internal var matrix4x4: Matrix4x4 {
        // CATransform3D uses row-vector semantics (translation is m41/m42/m43).
        // Transpose its rows into WebGPU columns so matrix * columnVector is equivalent.
        return Matrix4x4(columns: (
            SIMD4<Float>(Float(m11), Float(m12), Float(m13), Float(m14)),
            SIMD4<Float>(Float(m21), Float(m22), Float(m23), Float(m24)),
            SIMD4<Float>(Float(m31), Float(m32), Float(m33), Float(m34)),
            SIMD4<Float>(Float(m41), Float(m42), Float(m43), Float(m44))
        ))
    }
}

#endif
