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

/// Uniform data passed to shaders for each layer (WASM version).
public struct CARendererUniforms {
    public var mvpMatrix: Matrix4x4
    public var opacity: Float
    public var cornerRadius: Float
    public var layerSize: SIMD2<Float>
    public var borderWidth: Float
    public var renderMode: Float  // 0 = fill, 1 = border, 2 = axial, 3 = radial, 4 = conic
    public var gradientStartPoint: SIMD2<Float>
    public var gradientEndPoint: SIMD2<Float>
    public var gradientColorCount: Float
    /// Raw edge-antialiasing bits in x and the corner-curve exponent in y.
    public var edgeAntialiasingParameters: SIMD3<Float>
    /// Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
    /// Corresponds to CACornerMask corners for selective corner rounding.
    public var cornerRadii: SIMD4<Float>
    /// Replicator color applied to every stop after storage-buffer lookup.
    public var gradientColorMultiplier: SIMD4<Float>
    /// First stop index in the frame's read-only gradient storage buffer.
    public var gradientStopOffset: Float

    public init(
        mvpMatrix: Matrix4x4 = .identity,
        opacity: Float = 1.0,
        cornerRadius: Float = 0.0,
        layerSize: SIMD2<Float> = .zero,
        borderWidth: Float = 0.0,
        renderMode: Float = 0.0,
        gradientStartPoint: SIMD2<Float> = .zero,
        gradientEndPoint: SIMD2<Float> = SIMD2(0, 1),
        gradientColorCount: Float = 0,
        gradientColorMultiplier: SIMD4<Float> = SIMD4(repeating: 1),
        gradientStopOffset: Float = 0,
        edgeAntialiasingMask: Float = 0,
        cornerCurveExponent: Float = 2,
        cornerRadii: SIMD4<Float> = .zero
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
        self.edgeAntialiasingParameters = SIMD3(edgeAntialiasingMask, cornerCurveExponent, 0)
        self.cornerRadii = cornerRadii
        self.gradientColorMultiplier = gradientColorMultiplier
        self.gradientStopOffset = gradientStopOffset
    }
}

/// Uniform data for textured layer rendering.
public struct TexturedUniforms {
    public var mvpMatrix: Matrix4x4
    public var opacity: Float
    public var cornerRadius: Float
    public var layerSize: SIMD2<Float>
    /// Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
    public var cornerRadii: SIMD4<Float>
    public var samplingBias: Float
    /// Raw `CAEdgeAntialiasingMask` bits.
    public var edgeAntialiasingMask: Float
    public var cornerCurveExponent: Float
    public var padding2: Float = 0

    public init(
        mvpMatrix: Matrix4x4 = .identity,
        opacity: Float = 1.0,
        cornerRadius: Float = 0.0,
        layerSize: SIMD2<Float> = .zero,
        cornerRadii: SIMD4<Float> = .zero,
        samplingBias: Float = 0,
        edgeAntialiasingMask: Float = 0,
        cornerCurveExponent: Float = 2
    ) {
        self.mvpMatrix = mvpMatrix
        self.opacity = opacity
        self.cornerRadius = cornerRadius
        self.layerSize = layerSize
        self.cornerRadii = cornerRadii
        self.samplingBias = samplingBias
        self.edgeAntialiasingMask = edgeAntialiasingMask
        self.cornerCurveExponent = cornerCurveExponent
    }
}

/// Uniform data for premultiplied source/target transition interpolation.
struct TransitionFadeUniforms {
    var mvpMatrix: Matrix4x4
    var colorMultiplier: SIMD4<Float>
    var parameters: SIMD4<Float>

    init(
        mvpMatrix: Matrix4x4 = .identity,
        colorMultiplier: SIMD4<Float> = SIMD4(repeating: 1),
        opacity: Float = 1,
        progress: Float = 0
    ) {
        self.mvpMatrix = mvpMatrix
        self.colorMultiplier = colorMultiplier
        self.parameters = SIMD4(opacity, progress, 0, 0)
    }
}

/// Uniform data for single-pass filter compositing.
public struct FilterCompositeUniforms {
    public var opacity: Float
    public var filterType: Float
    public var parameter0: Float
    public var parameter1: Float
    public var colorMultiplier: SIMD4<Float>

    public init(
        opacity: Float = 1.0,
        filterType: Float = 0.0,
        parameter0: Float = 0.0,
        parameter1: Float = 0.0,
        colorMultiplier: SIMD4<Float> = SIMD4(1, 1, 1, 1)
    ) {
        self.opacity = opacity
        self.filterType = filterType
        self.parameter0 = parameter0
        self.parameter1 = parameter1
        self.colorMultiplier = colorMultiplier
    }
}

#endif
