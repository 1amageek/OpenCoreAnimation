#if arch(wasm32)
import Foundation

// MARK: - WASM Renderer Types

/// A structure representing a vertex for layer rendering (WASM version).
internal struct CARendererVertex {
    internal var position: SIMD2<Float>
    internal var texCoord: SIMD2<Float>
    internal var color: SIMD4<Float>

    internal init(position: SIMD2<Float>, texCoord: SIMD2<Float>, color: SIMD4<Float>) {
        self.position = position
        self.texCoord = texCoord
        self.color = color
    }
}

/// Uniform data passed to shaders for each layer (WASM version).
internal struct CARendererUniforms {
    internal var mvpMatrix: Matrix4x4
    internal var opacity: Float
    internal var cornerRadius: Float
    internal var layerSize: SIMD2<Float>
    internal var borderWidth: Float
    internal var renderMode: Float  // 0 = fill, 1 = border, 2 = axial, 3 = radial, 4 = conic
    internal var gradientStartPoint: SIMD2<Float>
    internal var gradientEndPoint: SIMD2<Float>
    internal var gradientColorCount: Float
    /// Raw edge-antialiasing bits in x and the corner-curve exponent in y.
    internal var edgeAntialiasingParameters: SIMD3<Float>
    /// Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
    /// Corresponds to CACornerMask corners for selective corner rounding.
    internal var cornerRadii: SIMD4<Float>
    /// Replicator color applied to every stop after storage-buffer lookup.
    internal var gradientColorMultiplier: SIMD4<Float>
    /// First stop index in the frame's read-only gradient storage buffer.
    internal var gradientStopOffset: Float

    internal init(
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
internal struct TexturedUniforms {
    internal var mvpMatrix: Matrix4x4
    internal var opacity: Float
    internal var cornerRadius: Float
    internal var layerSize: SIMD2<Float>
    /// Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
    internal var cornerRadii: SIMD4<Float>
    internal var samplingBias: Float
    /// Raw `CAEdgeAntialiasingMask` bits.
    internal var edgeAntialiasingMask: Float
    internal var cornerCurveExponent: Float
    internal var padding2: Float = 0

    internal init(
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
internal struct FilterCompositeUniforms {
    internal var opacity: Float
    internal var filterType: Float
    internal var parameter0: Float
    internal var parameter1: Float
    internal var colorMultiplier: SIMD4<Float>

    internal init(
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
