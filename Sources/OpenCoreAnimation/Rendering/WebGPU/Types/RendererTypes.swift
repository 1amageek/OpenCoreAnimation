#if arch(wasm32)
import Foundation

// MARK: - WASM Renderer Types

/// A structure representing a vertex for layer rendering (WASM version).
public struct CARendererVertex {
    public var position: SIMD2<Float>
    public var texCoord: SIMD2<Float>
    public var color: SIMD4<Float>

    public init(position: SIMD2<Float>, texCoord: SIMD2<Float>, color: SIMD4<Float>) {
        self.position = position
        self.texCoord = texCoord
        self.color = color
    }
}

/// Maximum number of gradient color stops supported.
public let kMaxGradientStops: Int = 8

/// Uniform data passed to shaders for each layer (WASM version).
public struct CARendererUniforms {
    public var mvpMatrix: Matrix4x4
    public var opacity: Float
    public var cornerRadius: Float
    public var layerSize: SIMD2<Float>
    public var borderWidth: Float
    public var renderMode: Float  // 0 = fill, 1 = border, 2 = gradient
    public var gradientStartPoint: SIMD2<Float>
    public var gradientEndPoint: SIMD2<Float>
    public var gradientColorCount: Float
    public var padding3: SIMD3<Float>
    // Gradient color stops: each is vec4 color + vec4 (location, 0, 0, 0) = 8 bytes per stop
    // For simplicity, we'll pack 8 colors and 8 locations separately
    public var gradientColors: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>,
                                SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    public var gradientLocations: SIMD4<Float>  // First 4 locations
    public var gradientLocations2: SIMD4<Float> // Next 4 locations

    public init(
        mvpMatrix: Matrix4x4 = .identity,
        opacity: Float = 1.0,
        cornerRadius: Float = 0.0,
        layerSize: SIMD2<Float> = .zero,
        borderWidth: Float = 0.0,
        renderMode: Float = 0.0,
        gradientStartPoint: SIMD2<Float> = .zero,
        gradientEndPoint: SIMD2<Float> = SIMD2(0, 1),
        gradientColorCount: Float = 0
    ) {
        self.mvpMatrix = mvpMatrix
        self.opacity = opacity
        self.cornerRadius = cornerRadius
        self.layerSize = layerSize
        self.borderWidth = borderWidth
        self.renderMode = renderMode
        self.gradientStartPoint = gradientStartPoint
        self.gradientEndPoint = gradientEndPoint
        self.gradientColorCount = gradientColorCount
        self.padding3 = .zero
        self.gradientColors = (.zero, .zero, .zero, .zero, .zero, .zero, .zero, .zero)
        self.gradientLocations = .zero
        self.gradientLocations2 = .zero
    }
}

/// Uniform data for textured layer rendering.
public struct TexturedUniforms {
    public var mvpMatrix: Matrix4x4
    public var opacity: Float
    public var cornerRadius: Float
    public var layerSize: SIMD2<Float>

    public init(
        mvpMatrix: Matrix4x4 = .identity,
        opacity: Float = 1.0,
        cornerRadius: Float = 0.0,
        layerSize: SIMD2<Float> = .zero
    ) {
        self.mvpMatrix = mvpMatrix
        self.opacity = opacity
        self.cornerRadius = cornerRadius
        self.layerSize = layerSize
    }
}

#endif
