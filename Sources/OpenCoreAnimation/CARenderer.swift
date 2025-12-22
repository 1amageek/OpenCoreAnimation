// CGImageMetadataTag.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework

import Foundation
import OpenCoreGraphics

/// Errors that can occur during renderer operations.
public enum CARendererError: Error {
    /// The GPU/graphics device is not available.
    case deviceNotAvailable
    /// Failed to create the render pipeline.
    case pipelineCreationFailed
    /// Failed to create shader module.
    case shaderCompilationFailed(String)
    /// Failed to create buffer.
    case bufferCreationFailed
    /// Failed to create texture.
    case textureCreationFailed
    /// Failed to create GPU resource.
    case resourceCreationFailed
    /// The canvas/view is not configured.
    case canvasNotConfigured
    /// General rendering error.
    case renderingFailed(String)
}

/// A protocol that defines the interface for rendering layer trees.
///
/// Implementations of this protocol provide platform-specific rendering:
/// - `CAMetalRenderer`: Uses Metal for Apple platforms (macOS/iOS)
/// - `CAWebGPURenderer`: Uses WebGPU for WASM/Web platforms
public protocol CARenderer: AnyObject {

    /// The size of the render target in pixels.
    var size: CGSize { get }

    /// Initializes the renderer asynchronously.
    ///
    /// This method sets up the GPU device, creates render pipelines,
    /// and prepares all resources needed for rendering.
    ///
    /// - Throws: `CARendererError` if initialization fails.
    func initialize() async throws

    /// Resizes the render target.
    ///
    /// Call this method when the canvas/view size changes.
    ///
    /// - Parameters:
    ///   - width: The new width in pixels.
    ///   - height: The new height in pixels.
    func resize(width: Int, height: Int)

    /// Renders the layer tree starting from the root layer.
    ///
    /// This method traverses the layer hierarchy and renders each layer
    /// according to its properties (transform, opacity, background color, etc.).
    ///
    /// - Parameter rootLayer: The root layer of the tree to render.
    func render(layer rootLayer: CALayer)

    /// Releases all GPU resources.
    ///
    /// Call this method when the renderer is no longer needed.
    func invalidate()
}

// MARK: - SIMD-dependent types and helpers (Apple platforms only)

#if canImport(simd)
import simd

/// A structure representing a vertex for layer rendering.
public struct CARendererVertex {
    /// Position in normalized device coordinates.
    public var position: SIMD2<Float>
    /// Texture coordinates (0-1 range).
    public var texCoord: SIMD2<Float>
    /// Vertex color (RGBA).
    public var color: SIMD4<Float>

    public init(position: SIMD2<Float>, texCoord: SIMD2<Float>, color: SIMD4<Float>) {
        self.position = position
        self.texCoord = texCoord
        self.color = color
    }
}

/// Uniform data passed to shaders for each layer.
public struct CARendererUniforms {
    /// Model-view-projection matrix.
    public var mvpMatrix: simd_float4x4
    /// Layer opacity (0-1).
    public var opacity: Float
    /// Corner radius in pixels.
    public var cornerRadius: Float
    /// Padding for alignment.
    public var padding: SIMD2<Float>

    public init(
        mvpMatrix: simd_float4x4 = matrix_identity_float4x4,
        opacity: Float = 1.0,
        cornerRadius: Float = 0.0
    ) {
        self.mvpMatrix = mvpMatrix
        self.opacity = opacity
        self.cornerRadius = cornerRadius
        self.padding = .zero
    }
}

// MARK: - Helper Extensions

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

    /// Calculates the model matrix for this layer.
    internal func modelMatrix(parentMatrix: simd_float4x4 = matrix_identity_float4x4) -> simd_float4x4 {
        // Start with translation to position
        var matrix = parentMatrix

        // Translate to layer position
        let translation = simd_float4x4(translation: SIMD3<Float>(
            Float(position.x),
            Float(position.y),
            Float(zPosition)
        ))
        matrix = matrix * translation

        // Apply layer transform if not identity
        if !CATransform3DIsIdentity(transform) {
            let layerTransform = transform.simdMatrix
            matrix = matrix * layerTransform
        }

        // Translate by anchor point offset
        let anchorOffset = simd_float4x4(translation: SIMD3<Float>(
            Float(-bounds.width * anchorPoint.x),
            Float(-bounds.height * anchorPoint.y),
            Float(-anchorPointZ)
        ))
        matrix = matrix * anchorOffset

        return matrix
    }

    /// Calculates the parent matrix for sublayer positioning.
    ///
    /// This includes the layer's sublayerTransform and bounds.origin offset.
    /// The bounds.origin offset ensures that sublayers are correctly positioned when the
    /// layer's bounds origin is non-zero (e.g., for CAScrollLayer scrolling).
    ///
    /// - Parameter modelMatrix: The layer's model matrix
    /// - Returns: The matrix to use as parentMatrix for sublayer rendering
    internal func sublayerMatrix(modelMatrix: simd_float4x4) -> simd_float4x4 {
        var result = modelMatrix

        // Apply sublayerTransform if not identity
        if !CATransform3DIsIdentity(sublayerTransform) {
            result = result * sublayerTransform.simdMatrix
        }

        // Apply bounds.origin offset
        // In CoreAnimation, bounds.origin defines where the coordinate system origin is
        // within the layer. A sublayer at position (0,0) with parent's bounds.origin = (50, 50)
        // should appear at (-50, -50) relative to the parent's visible top-left.
        // This is the scrolling behavior used by CAScrollLayer.
        if bounds.origin.x != 0 || bounds.origin.y != 0 {
            let boundsOriginOffset = simd_float4x4(translation: SIMD3<Float>(
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
    /// Converts CATransform3D to simd_float4x4.
    internal var simdMatrix: simd_float4x4 {
        // simd_float4x4 is column-major, so we construct from columns
        return simd_float4x4(columns: (
            SIMD4<Float>(Float(m11), Float(m21), Float(m31), Float(m41)),
            SIMD4<Float>(Float(m12), Float(m22), Float(m32), Float(m42)),
            SIMD4<Float>(Float(m13), Float(m23), Float(m33), Float(m43)),
            SIMD4<Float>(Float(m14), Float(m24), Float(m34), Float(m44))
        ))
    }
}

extension simd_float4x4 {
    /// Creates a translation matrix.
    internal init(translation: SIMD3<Float>) {
        self = matrix_identity_float4x4
        self.columns.3 = SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    }

    /// Creates an orthographic projection matrix.
    internal static func orthographic(
        left: Float,
        right: Float,
        bottom: Float,
        top: Float,
        near: Float,
        far: Float
    ) -> simd_float4x4 {
        let width = right - left
        let height = top - bottom
        let depth = far - near

        // simd_float4x4 is column-major, so we construct from columns
        return simd_float4x4(columns: (
            SIMD4<Float>(2 / width, 0, 0, 0),
            SIMD4<Float>(0, 2 / height, 0, 0),
            SIMD4<Float>(0, 0, -2 / depth, 0),
            SIMD4<Float>(-(right + left) / width, -(top + bottom) / height, -(far + near) / depth, 1)
        ))
    }
}

#endif
