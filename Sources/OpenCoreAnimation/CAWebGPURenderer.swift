#if arch(wasm32)
import JavaScriptKit
import SwiftWebGPU

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
}

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

// MARK: - WebGPU Renderer

/// A renderer that uses WebGPU to render layer trees in WASM/Web environments.
///
/// This is the primary renderer for OpenCoreAnimation in production.
public final class CAWebGPURenderer: CARenderer {

    // MARK: - Constants

    /// Maximum number of layers that can be rendered per frame.
    private static let maxLayers = 256

    /// Uniform buffer alignment requirement (WebGPU minimum is 256 bytes).
    private static let uniformAlignment: UInt64 = 256

    /// Size of aligned uniform data per layer.
    private static var alignedUniformSize: UInt64 {
        let baseSize = UInt64(MemoryLayout<CARendererUniforms>.stride)
        return ((baseSize + uniformAlignment - 1) / uniformAlignment) * uniformAlignment
    }

    // MARK: - Properties

    /// The WebGPU device.
    private var device: GPUDevice?

    /// The canvas context.
    private var context: GPUCanvasContext?

    /// The render pipeline.
    private var pipeline: GPURenderPipeline?

    /// The vertex buffer (large enough for all layers).
    private var vertexBuffer: GPUBuffer?

    /// The uniform buffer (large enough for all layers with alignment).
    private var uniformBuffer: GPUBuffer?

    /// The bind group layout for dynamic uniform access.
    private var bindGroupLayout: GPUBindGroupLayout?

    /// The bind group.
    private var bindGroup: GPUBindGroup?

    /// The depth texture.
    private var depthTexture: GPUTexture?

    /// The current size.
    public private(set) var size: CGSize = CGSize(width: 0, height: 0)

    /// The preferred texture format.
    private var preferredFormat: GPUTextureFormat = .bgra8unorm

    /// The canvas element (JavaScript object).
    private let canvas: JSObject

    /// Current layer index during rendering.
    private var currentLayerIndex: Int = 0

    // MARK: - Texture Rendering

    /// The textured render pipeline.
    private var texturedPipeline: GPURenderPipeline?

    /// The bind group layout for textured rendering.
    private var texturedBindGroupLayout: GPUBindGroupLayout?

    /// The texture sampler.
    private var textureSampler: GPUSampler?

    /// Cache of GPU textures keyed by CGImage identity.
    private var textureCache: [ObjectIdentifier: GPUTexture] = [:]

    // MARK: - Shader Code

    private let shaderCode = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
        borderWidth: f32,
        renderMode: f32,  // 0 = fill, 1 = border, 2 = gradient
        gradientStartPoint: vec2<f32>,
        gradientEndPoint: vec2<f32>,
        gradientColorCount: f32,
        padding3: vec3<f32>,
        gradientColor0: vec4<f32>,
        gradientColor1: vec4<f32>,
        gradientColor2: vec4<f32>,
        gradientColor3: vec4<f32>,
        gradientColor4: vec4<f32>,
        gradientColor5: vec4<f32>,
        gradientColor6: vec4<f32>,
        gradientColor7: vec4<f32>,
        gradientLocations: vec4<f32>,
        gradientLocations2: vec4<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) texCoord: vec2<f32>,
        @location(2) color: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
        @location(1) color: vec4<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.texCoord = input.texCoord;
        output.color = input.color * uniforms.opacity;
        return output;
    }

    // Signed distance function for a rounded rectangle
    fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
        let q = abs(p) - halfSize + radius;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    }

    // Get gradient color at index
    fn getGradientColor(index: i32) -> vec4<f32> {
        switch (index) {
            case 0: { return uniforms.gradientColor0; }
            case 1: { return uniforms.gradientColor1; }
            case 2: { return uniforms.gradientColor2; }
            case 3: { return uniforms.gradientColor3; }
            case 4: { return uniforms.gradientColor4; }
            case 5: { return uniforms.gradientColor5; }
            case 6: { return uniforms.gradientColor6; }
            case 7: { return uniforms.gradientColor7; }
            default: { return vec4<f32>(0.0); }
        }
    }

    // Get gradient location at index
    fn getGradientLocation(index: i32) -> f32 {
        switch (index) {
            case 0: { return uniforms.gradientLocations.x; }
            case 1: { return uniforms.gradientLocations.y; }
            case 2: { return uniforms.gradientLocations.z; }
            case 3: { return uniforms.gradientLocations.w; }
            case 4: { return uniforms.gradientLocations2.x; }
            case 5: { return uniforms.gradientLocations2.y; }
            case 6: { return uniforms.gradientLocations2.z; }
            case 7: { return uniforms.gradientLocations2.w; }
            default: { return 0.0; }
        }
    }

    // Calculate gradient color at position t (0-1)
    fn sampleGradient(t: f32) -> vec4<f32> {
        let colorCount = i32(uniforms.gradientColorCount);
        if (colorCount <= 0) {
            return vec4<f32>(0.0);
        }
        if (colorCount == 1) {
            return getGradientColor(0);
        }

        let clampedT = clamp(t, 0.0, 1.0);

        // Find the two colors to interpolate between
        for (var i = 1; i < colorCount; i++) {
            let loc0 = getGradientLocation(i - 1);
            let loc1 = getGradientLocation(i);
            if (clampedT <= loc1) {
                let localT = (clampedT - loc0) / max(loc1 - loc0, 0.0001);
                let color0 = getGradientColor(i - 1);
                let color1 = getGradientColor(i);
                return mix(color0, color1, clamp(localT, 0.0, 1.0));
            }
        }

        // Return last color if past all stops
        return getGradientColor(colorCount - 1);
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        // Handle case where layer size is zero
        if (uniforms.layerSize.x <= 0.0 || uniforms.layerSize.y <= 0.0) {
            return input.color;
        }

        // Convert texCoord (0-1) to pixel coordinates centered at origin
        let pixelCoord = (input.texCoord - 0.5) * uniforms.layerSize;

        // Calculate half-size of the rectangle in pixels
        let halfSize = uniforms.layerSize * 0.5;

        // Clamp corner radius to half the smaller dimension
        let maxRadius = min(halfSize.x, halfSize.y);
        let radius = min(uniforms.cornerRadius, maxRadius);

        // Gradient rendering mode
        if (uniforms.renderMode > 1.5) {
            // Calculate gradient position
            let gradientDir = uniforms.gradientEndPoint - uniforms.gradientStartPoint;
            let gradientLen = length(gradientDir);
            var t: f32 = 0.0;
            if (gradientLen > 0.0001) {
                let normalizedDir = gradientDir / gradientLen;
                let relativePos = input.texCoord - uniforms.gradientStartPoint;
                t = dot(relativePos, normalizedDir) / gradientLen;
            }

            // Sample gradient
            var gradientColor = sampleGradient(t);
            gradientColor.a *= uniforms.opacity;

            // Apply corner radius if set
            if (radius > 0.0) {
                let dist = sdRoundedBox(pixelCoord, halfSize, radius);
                let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
                gradientColor.a *= alpha;
            }

            return gradientColor;
        }

        // Border rendering mode
        if (uniforms.renderMode > 0.5) {
            // Calculate outer and inner signed distances
            let outerDist = sdRoundedBox(pixelCoord, halfSize, radius);
            let innerHalfSize = halfSize - uniforms.borderWidth;
            let innerRadius = max(0.0, radius - uniforms.borderWidth);
            let innerDist = sdRoundedBox(pixelCoord, innerHalfSize, innerRadius);

            // Border is where we're inside outer but outside inner
            let outerAlpha = 1.0 - smoothstep(-1.0, 1.0, outerDist);
            let innerAlpha = 1.0 - smoothstep(-1.0, 1.0, innerDist);
            let borderAlpha = outerAlpha - innerAlpha;

            return vec4<f32>(input.color.rgb, input.color.a * borderAlpha);
        }

        // Fill rendering mode (default)
        if (uniforms.cornerRadius <= 0.0) {
            return input.color;
        }

        // Calculate signed distance for fill
        let dist = sdRoundedBox(pixelCoord, halfSize, radius);

        // Anti-aliased edge (smooth over 1 pixel)
        let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);

        return vec4<f32>(input.color.rgb, input.color.a * alpha);
    }
    """

    /// Shader code for textured rendering.
    private let texturedShaderCode = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var textureSampler: sampler;
    @group(0) @binding(2) var textureData: texture_2d<f32>;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) texCoord: vec2<f32>,
        @location(2) color: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
        @location(1) color: vec4<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.texCoord = input.texCoord;
        output.color = input.color * uniforms.opacity;
        return output;
    }

    // Signed distance function for a rounded rectangle
    fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
        let q = abs(p) - halfSize + radius;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        // Sample texture
        var texColor = textureSample(textureData, textureSampler, input.texCoord);

        // Apply opacity
        texColor.a *= uniforms.opacity;

        // Apply corner radius if set
        if (uniforms.cornerRadius > 0.0 && uniforms.layerSize.x > 0.0 && uniforms.layerSize.y > 0.0) {
            let pixelCoord = (input.texCoord - 0.5) * uniforms.layerSize;
            let halfSize = uniforms.layerSize * 0.5;
            let maxRadius = min(halfSize.x, halfSize.y);
            let radius = min(uniforms.cornerRadius, maxRadius);
            let dist = sdRoundedBox(pixelCoord, halfSize, radius);
            let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
            texColor.a *= alpha;
        }

        return texColor;
    }
    """

    // MARK: - Initialization

    /// Creates a new WebGPU renderer with the specified canvas element.
    ///
    /// - Parameter canvas: The JavaScript canvas element to render to.
    public init(canvas: JSObject) {
        self.canvas = canvas
    }

    // MARK: - CARenderer

    public func initialize() async throws {
        // Get GPU
        guard let gpu = GPU.shared else {
            throw CARendererError.deviceNotAvailable
        }

        // Request adapter
        guard let adapter = try await gpu.requestAdapter() else {
            throw CARendererError.deviceNotAvailable
        }

        // Request device
        let device = try await adapter.requestDevice()
        self.device = device

        // Get preferred format
        preferredFormat = gpu.preferredCanvasFormat

        // Configure canvas context
        let ctx = canvas.getContext!("webgpu")
        guard let ctxObject = ctx.object else {
            throw CARendererError.canvasNotConfigured
        }
        context = GPUCanvasContext(jsObject: ctxObject)

        context?.configure(GPUCanvasConfiguration(
            device: device,
            format: preferredFormat
        ))

        // Create shader module
        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: shaderCode
        ))

        // Create bind group layout with dynamic uniform buffer
        let layout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: .vertex,
                    buffer: GPUBufferBindingLayout(
                        type: .uniform,
                        hasDynamicOffset: true,
                        minBindingSize: UInt64(MemoryLayout<CARendererUniforms>.stride)
                    )
                )
            ]
        ))
        bindGroupLayout = layout

        // Create pipeline layout
        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [layout]
        ))

        // Create render pipeline with depth testing
        pipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plus,
                depthWriteEnabled: true,
                depthCompare: .lessEqual
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        // Create vertex buffer (large enough for all layers)
        let vertexBufferSize = UInt64(MemoryLayout<CARendererVertex>.stride * 6 * Self.maxLayers)
        vertexBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: vertexBufferSize,
            usage: [.vertex, .copyDst]
        ))

        // Create uniform buffer (with alignment for dynamic offsets)
        let uniformBufferSize = Self.alignedUniformSize * UInt64(Self.maxLayers)
        uniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: uniformBufferSize,
            usage: [.uniform, .copyDst]
        ))

        // Create bind group
        guard let uniformBuffer = uniformBuffer else {
            throw CARendererError.bufferCreationFailed
        }

        bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: layout,
            entries: [
                GPUBindGroupEntry(
                    binding: 0,
                    resource: .bufferBinding(GPUBufferBinding(
                        buffer: uniformBuffer,
                        size: UInt64(MemoryLayout<CARendererUniforms>.stride)
                    ))
                )
            ]
        ))

        // Get initial canvas size
        let width = canvas.width.number ?? 800
        let height = canvas.height.number ?? 600
        size = CGSize(width: width, height: height)

        // Create depth texture
        createDepthTexture(width: Int(width), height: Int(height))

        // Create textured pipeline
        try createTexturedPipeline(device: device)
    }

    /// Creates the textured render pipeline for layer.contents rendering.
    private func createTexturedPipeline(device: GPUDevice) throws {
        // Create shader module
        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: texturedShaderCode
        ))

        // Create sampler
        textureSampler = device.createSampler(descriptor: GPUSamplerDescriptor(
            magFilter: .linear,
            minFilter: .linear,
            mipmapFilter: .linear,
            addressModeU: .clampToEdge,
            addressModeV: .clampToEdge,
            addressModeW: .clampToEdge
        ))

        // Create bind group layout with uniform, sampler, and texture
        texturedBindGroupLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(
                        type: .uniform,
                        hasDynamicOffset: true,
                        minBindingSize: UInt64(MemoryLayout<TexturedUniforms>.stride)
                    )
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(
                        sampleType: .float,
                        viewDimension: .dimension2D
                    )
                )
            ]
        ))

        guard let texturedBindGroupLayout = texturedBindGroupLayout else {
            throw CARendererError.resourceCreationFailed
        }

        // Create pipeline layout
        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [texturedBindGroupLayout]
        ))

        // Create textured render pipeline
        texturedPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plus,
                depthWriteEnabled: true,
                depthCompare: .lessEqual
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))
    }

    /// Creates or recreates the depth texture for the given size.
    private func createDepthTexture(width: Int, height: Int) {
        guard let device = device, width > 0, height > 0 else { return }

        depthTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .depth24plus,
            usage: .renderAttachment
        ))
    }

    public func resize(width: Int, height: Int) {
        size = CGSize(width: width, height: height)

        // Update canvas size
        canvas.width = .number(Double(width))
        canvas.height = .number(Double(height))

        // Recreate depth texture
        createDepthTexture(width: width, height: height)
    }

    public func render(layer rootLayer: CALayer) {
        guard let device = device,
              let context = context,
              let pipeline = pipeline,
              bindGroup != nil,
              let depthTexture = depthTexture else { return }

        // Reset layer index for this frame
        currentLayerIndex = 0

        // Get current texture
        let currentTexture = context.getCurrentTexture()
        let textureView = currentTexture.createView()
        let depthTextureView = depthTexture.createView()

        // Create command encoder
        let encoder = device.createCommandEncoder()

        // Begin render pass with depth attachment
        let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: textureView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 1),
                    loadOp: .clear,
                    storeOp: .store
                )
            ],
            depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                view: depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: .clear,
                depthStoreOp: .store
            )
        ))

        renderPass.setPipeline(pipeline)

        // Create projection matrix
        let projectionMatrix = Matrix4x4.orthographic(
            left: 0,
            right: Float(size.width),
            bottom: Float(size.height),
            top: 0,
            near: -1000,
            far: 1000
        )

        // Render layer tree
        renderLayer(rootLayer, renderPass: renderPass, parentMatrix: projectionMatrix)

        renderPass.end()

        // Submit command buffer
        device.queue.submit([encoder.finish()])
    }

    public func invalidate() {
        vertexBuffer = nil
        uniformBuffer = nil
        bindGroup = nil
        bindGroupLayout = nil
        depthTexture = nil
        pipeline = nil
        texturedPipeline = nil
        texturedBindGroupLayout = nil
        textureSampler = nil
        textureCache.removeAll()
        context = nil
        device = nil
    }

    // MARK: - Private Methods

    /// Converts Swift data to a JavaScript Float32Array for WebGPU buffer writes.
    private func createFloat32Array<T>(from data: inout T) -> JSObject {
        let byteCount = MemoryLayout<T>.stride
        let floatCount = byteCount / 4
        let float32Array = JSObject.global.Float32Array.function!.new(floatCount)
        withUnsafeBytes(of: &data) { bytes in
            for i in 0..<floatCount {
                let value = bytes.load(fromByteOffset: i * 4, as: Float.self)
                float32Array[i] = .number(Double(value))
            }
        }
        return float32Array
    }

    /// Converts an array of vertices to a JavaScript Float32Array.
    private func createFloat32Array(from vertices: inout [CARendererVertex]) -> JSObject {
        let floatCount = vertices.count * (MemoryLayout<CARendererVertex>.stride / 4)
        let float32Array = JSObject.global.Float32Array.function!.new(floatCount)
        vertices.withUnsafeBytes { bytes in
            for i in 0..<floatCount {
                let value = bytes.load(fromByteOffset: i * 4, as: Float.self)
                float32Array[i] = .number(Double(value))
            }
        }
        return float32Array
    }

    private func renderLayer(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        guard let device = device,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        // Get the presentation layer for animated values, fall back to model layer
        // This is critical for animations to be visible - the presentation layer
        // reflects the current animated state of all properties
        let presentationLayer = layer.presentation() ?? layer

        // Skip hidden layers (using presentation layer values)
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }

        // Check layer limit
        guard currentLayerIndex < Self.maxLayers else { return }

        // Calculate model matrix using presentation layer values
        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Check if this is a text layer
        if let textLayer = presentationLayer as? CATextLayer, textLayer.string != nil {
            renderTextLayer(textLayer, device: device, renderPass: renderPass,
                           modelMatrix: modelMatrix, bindGroup: bindGroup)
        }
        // Check if this is a shape layer
        else if let shapeLayer = presentationLayer as? CAShapeLayer, shapeLayer.path != nil {
            renderShapeLayer(shapeLayer, device: device, renderPass: renderPass,
                            modelMatrix: modelMatrix, bindGroup: bindGroup)
        }
        // Check if this is a gradient layer
        else if let gradientLayer = presentationLayer as? CAGradientLayer,
           let colors = gradientLayer.colors, !colors.isEmpty {
            renderGradientLayer(gradientLayer, device: device, renderPass: renderPass,
                              modelMatrix: modelMatrix, bindGroup: bindGroup)
        }
        // Check if layer has contents (CGImage)
        else if let contents = presentationLayer.contents {
            renderContentsLayer(presentationLayer, contents: contents, device: device,
                               renderPass: renderPass, modelMatrix: modelMatrix)
        }
        // Render background color if set (and not a gradient layer, or gradient has no colors)
        else if presentationLayer.backgroundColor != nil {
            let layerIndex = currentLayerIndex
            currentLayerIndex += 1

            // Create scale matrix for layer bounds (column-major order)
            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
                SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let finalMatrix = modelMatrix * scaleMatrix

            // Update uniforms at the correct offset
            var uniforms = CARendererUniforms(
                mvpMatrix: finalMatrix,
                opacity: presentationLayer.opacity,
                cornerRadius: Float(presentationLayer.cornerRadius),
                layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height))
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(
                uniformBuffer,
                bufferOffset: uniformOffset,
                data: uniformData
            )

            // Create vertices using presentation layer color
            let color = presentationLayer.backgroundColorComponents
            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            ]

            // Write vertices at the correct offset
            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            // Set bind group with dynamic offset for this layer's uniforms
            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Render border if set
        if presentationLayer.borderWidth > 0 && presentationLayer.borderColor != nil {
            guard currentLayerIndex < Self.maxLayers else { return }

            let layerIndex = currentLayerIndex
            currentLayerIndex += 1

            // Create scale matrix for layer bounds (column-major order)
            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
                SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let finalMatrix = modelMatrix * scaleMatrix

            // Update uniforms with border mode (renderMode = 1)
            var uniforms = CARendererUniforms(
                mvpMatrix: finalMatrix,
                opacity: presentationLayer.opacity,
                cornerRadius: Float(presentationLayer.cornerRadius),
                layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height)),
                borderWidth: Float(presentationLayer.borderWidth),
                renderMode: 1.0  // Border mode
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(
                uniformBuffer,
                bufferOffset: uniformOffset,
                data: uniformData
            )

            // Create vertices using border color
            let color = presentationLayer.borderColorComponents
            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            ]

            // Write vertices at the correct offset
            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            // Set bind group with dynamic offset for this layer's uniforms
            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Render sublayers (use model layer hierarchy, but presentation layer's sublayerTransform)
        if let sublayers = layer.sublayers {
            var sublayerMatrix = modelMatrix
            if !CATransform3DIsIdentity(presentationLayer.sublayerTransform) {
                sublayerMatrix = sublayerMatrix * presentationLayer.sublayerTransform.matrix4x4
            }

            // Check if this is a replicator layer
            if let replicatorLayer = presentationLayer as? CAReplicatorLayer {
                renderReplicatorSublayers(
                    replicatorLayer: replicatorLayer,
                    sublayers: sublayers,
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix
                )
            } else {
                for sublayer in sublayers {
                    self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
                }
            }
        }
    }

    // MARK: - Replicator Layer Rendering

    /// Renders the sublayers of a CAReplicatorLayer with instance transformations.
    private func renderReplicatorSublayers(
        replicatorLayer: CAReplicatorLayer,
        sublayers: [CALayer],
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let instanceCount = max(1, replicatorLayer.instanceCount)
        let instanceTransform = replicatorLayer.instanceTransform

        // Get base instance color (defaults to white if not set)
        let baseColor: SIMD4<Float>
        if let color = replicatorLayer.instanceColor,
           let components = color.components, components.count >= 4 {
            baseColor = SIMD4<Float>(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
        } else {
            baseColor = SIMD4<Float>(1, 1, 1, 1)
        }

        // Color offsets per instance
        let redOffset = replicatorLayer.instanceRedOffset
        let greenOffset = replicatorLayer.instanceGreenOffset
        let blueOffset = replicatorLayer.instanceBlueOffset
        let alphaOffset = replicatorLayer.instanceAlphaOffset

        // Render each instance
        var cumulativeTransform = CATransform3DIdentity
        for instanceIndex in 0..<instanceCount {
            // Calculate color multiplier for this instance
            let colorMultiplier = SIMD4<Float>(
                clamp(baseColor.x + Float(instanceIndex) * redOffset, 0, 1),
                clamp(baseColor.y + Float(instanceIndex) * greenOffset, 0, 1),
                clamp(baseColor.z + Float(instanceIndex) * blueOffset, 0, 1),
                clamp(baseColor.w + Float(instanceIndex) * alphaOffset, 0, 1)
            )

            // Calculate instance matrix
            let instanceMatrix = parentMatrix * cumulativeTransform.matrix4x4

            // Render all sublayers with this instance's transform and color
            for sublayer in sublayers {
                renderLayerWithColorMultiplier(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: instanceMatrix,
                    colorMultiplier: colorMultiplier
                )
            }

            // Apply instance transform for next iteration
            cumulativeTransform = CATransform3DConcat(cumulativeTransform, instanceTransform)
        }
    }

    /// Renders a layer with a color multiplier applied (for replicator instances).
    private func renderLayerWithColorMultiplier(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4,
        colorMultiplier: SIMD4<Float>
    ) {
        guard let device = device,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        let presentationLayer = layer.presentation() ?? layer
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }
        guard currentLayerIndex < Self.maxLayers else { return }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // For now, render background color with the color multiplier applied
        if presentationLayer.backgroundColor != nil {
            let layerIndex = currentLayerIndex
            currentLayerIndex += 1

            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
                SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let finalMatrix = modelMatrix * scaleMatrix

            var uniforms = CARendererUniforms(
                mvpMatrix: finalMatrix,
                opacity: presentationLayer.opacity,
                cornerRadius: Float(presentationLayer.cornerRadius),
                layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height))
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(
                uniformBuffer,
                bufferOffset: uniformOffset,
                data: uniformData
            )

            // Apply color multiplier to the layer's background color
            var baseColor = presentationLayer.backgroundColorComponents
            baseColor.x *= colorMultiplier.x
            baseColor.y *= colorMultiplier.y
            baseColor.z *= colorMultiplier.z
            baseColor.w *= colorMultiplier.w

            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
            ]

            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Recursively render sublayers
        if let sublayers = layer.sublayers {
            var sublayerMatrix = modelMatrix
            if !CATransform3DIsIdentity(presentationLayer.sublayerTransform) {
                sublayerMatrix = sublayerMatrix * presentationLayer.sublayerTransform.matrix4x4
            }

            for sublayer in sublayers {
                renderLayerWithColorMultiplier(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix,
                    colorMultiplier: colorMultiplier
                )
            }
        }
    }

    /// Clamps a float value between min and max.
    private func clamp(_ value: Float, _ minVal: Float, _ maxVal: Float) -> Float {
        return min(max(value, minVal), maxVal)
    }

    // MARK: - Contents Layer Rendering (CGImage)

    /// Renders a layer with CGImage contents.
    private func renderContentsLayer(
        _ layer: CALayer,
        contents: CGImage,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let texturedPipeline = texturedPipeline,
              let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        guard currentLayerIndex < Self.maxLayers else { return }

        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        // Get or create GPU texture from CGImage
        let textureKey = ObjectIdentifier(contents)
        let gpuTexture: GPUTexture
        if let cached = textureCache[textureKey] {
            gpuTexture = cached
        } else {
            guard let newTexture = createGPUTexture(from: contents, device: device) else { return }
            textureCache[textureKey] = newTexture
            gpuTexture = newTexture
        }

        // Create scale matrix for layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(layer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(layer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Create uniforms for textured rendering
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: layer.opacity,
            cornerRadius: Float(layer.cornerRadius),
            layerSize: SIMD2<Float>(Float(layer.bounds.width), Float(layer.bounds.height))
        )

        // Write uniforms
        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        // Create vertices with white color (texture will provide color)
        let white = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
        ]

        // Write vertices
        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Create bind group with texture
        let texturedBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: texturedBindGroupLayout,
            entries: [
                GPUBindGroupEntry(
                    binding: 0,
                    resource: .bufferBinding(GPUBufferBinding(
                        buffer: uniformBuffer,
                        size: UInt64(MemoryLayout<TexturedUniforms>.stride)
                    ))
                ),
                GPUBindGroupEntry(
                    binding: 1,
                    resource: .sampler(textureSampler)
                ),
                GPUBindGroupEntry(
                    binding: 2,
                    resource: .textureView(gpuTexture.createView())
                )
            ]
        ))

        // Switch to textured pipeline and render
        renderPass.setPipeline(texturedPipeline)
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch back to non-textured pipeline for subsequent rendering
        if let pipeline = pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    /// Creates a GPU texture from a CGImage.
    private func createGPUTexture(from cgImage: CGImage, device: GPUDevice) -> GPUTexture? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0 && height > 0 else { return nil }

        // Get RGBA data
        guard let rgbaData = getRGBAData(from: cgImage) else { return nil }

        // Create texture
        let textureDescriptor = GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.textureBinding, .copyDst, .renderAttachment]
        )

        let texture = device.createTexture(descriptor: textureDescriptor)

        // Create JS Uint8Array from data
        let jsArray = createUint8Array(from: rgbaData)

        // Upload data
        device.queue.writeTexture(
            destination: GPUImageCopyTexture(texture: texture),
            data: jsArray,
            dataLayout: GPUImageDataLayout(
                offset: 0,
                bytesPerRow: UInt32(width * 4),
                rowsPerImage: UInt32(height)
            ),
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height))
        )

        return texture
    }

    /// Converts CGImage data to RGBA format.
    private func getRGBAData(from cgImage: CGImage) -> Data? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0 && height > 0 else { return nil }

        // If CGImage has data, try to use it directly
        guard let sourceData = cgImage.data else { return nil }

        let bytesPerPixel = cgImage.bitsPerPixel / 8
        let sourceBytesPerRow = cgImage.bytesPerRow
        let destBytesPerRow = width * 4
        let totalBytes = destBytesPerRow * height

        // If already RGBA8, use directly
        if bytesPerPixel == 4 && sourceBytesPerRow == destBytesPerRow {
            let isRGBA = cgImage.alphaInfo == .premultipliedLast ||
                        cgImage.alphaInfo == .last ||
                        cgImage.alphaInfo == .noneSkipLast
            if isRGBA {
                return sourceData
            }
        }

        // Need to convert
        var destData = Data(count: totalBytes)

        sourceData.withUnsafeBytes { sourceBuffer in
            destData.withUnsafeMutableBytes { destBuffer in
                guard let sourceBase = sourceBuffer.baseAddress,
                      let destBase = destBuffer.baseAddress else { return }

                for y in 0..<height {
                    for x in 0..<width {
                        let destOffset = y * destBytesPerRow + x * 4
                        let d = destBase.advanced(by: destOffset).assumingMemoryBound(to: UInt8.self)

                        if bytesPerPixel == 4 {
                            let sourceOffset = y * sourceBytesPerRow + x * 4
                            let s = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)

                            // Handle different alpha positions
                            switch cgImage.alphaInfo {
                            case .premultipliedFirst, .first:
                                // ARGB -> RGBA
                                d[0] = s[1]; d[1] = s[2]; d[2] = s[3]; d[3] = s[0]
                            case .noneSkipFirst:
                                // xRGB -> RGBA
                                d[0] = s[1]; d[1] = s[2]; d[2] = s[3]; d[3] = 255
                            case .noneSkipLast:
                                // RGBx -> RGBA
                                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255
                            default:
                                // Assume RGBA
                                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3]
                            }
                        } else if bytesPerPixel == 3 {
                            let sourceOffset = y * sourceBytesPerRow + x * 3
                            let s = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)
                            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255
                        } else if bytesPerPixel == 1 {
                            let sourceOffset = y * sourceBytesPerRow + x
                            let gray = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)[0]
                            d[0] = gray; d[1] = gray; d[2] = gray; d[3] = 255
                        }
                    }
                }
            }
        }

        return destData
    }

    /// Creates a JavaScript Uint8Array from Data.
    private func createUint8Array(from data: Data) -> JSObject {
        let uint8Array = JSObject.global.Uint8Array.function!.new(data.count)
        data.withUnsafeBytes { bytes in
            for i in 0..<data.count {
                uint8Array[i] = .number(Double(bytes.load(fromByteOffset: i, as: UInt8.self)))
            }
        }
        return uint8Array
    }

    /// Clears the texture cache for a specific CGImage.
    public func removeTexture(for cgImage: CGImage) {
        let key = ObjectIdentifier(cgImage)
        textureCache.removeValue(forKey: key)
    }

    /// Clears all cached textures.
    public func clearTextureCache() {
        textureCache.removeAll()
    }

    // MARK: - Text Layer Rendering

    /// Cache for text textures to avoid recreating them every frame.
    private var textTextureCache: [String: GPUTexture] = [:]

    /// Renders a CATextLayer using Canvas2D for text rasterization and texture-based rendering.
    ///
    /// This method uses an offscreen Canvas2D to render the text, creates a WebGPU texture
    /// from the canvas data, and displays it using the textured pipeline.
    private func renderTextLayer(
        _ textLayer: CATextLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let string = textLayer.string else { return }
        guard let texturedPipeline = texturedPipeline,
              let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let pipeline = pipeline else { return }

        let text: String
        if let str = string as? String {
            text = str
        } else {
            text = String(describing: string)
        }

        guard !text.isEmpty else { return }
        guard currentLayerIndex < Self.maxLayers else { return }

        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        let width = Int(textLayer.bounds.width)
        let height = Int(textLayer.bounds.height)

        guard width > 0 && height > 0 else { return }

        // Create cache key based on text content and properties
        let cacheKey = "\(text)_\(width)x\(height)_\(textLayer.fontSize)_\(textLayer.alignmentMode.rawValue)"

        // Check cache first
        let gpuTexture: GPUTexture
        if let cached = textTextureCache[cacheKey] {
            gpuTexture = cached
        } else {
            // Create offscreen canvas for text rendering
            let document = JSObject.global.document
            let offscreenCanvas = document.createElement("canvas")
            offscreenCanvas.width = .number(Double(width))
            offscreenCanvas.height = .number(Double(height))

        guard let ctx = offscreenCanvas.getContext("2d").object else { return }

        // Clear canvas with background color if set
        if let bgColor = textLayer.backgroundColor {
            let components = bgColor.components ?? [0, 0, 0, 1]
            let r = Int((components.count > 0 ? components[0] : 0) * 255)
            let g = Int((components.count > 1 ? components[1] : 0) * 255)
            let b = Int((components.count > 2 ? components[2] : 0) * 255)
            let a = components.count > 3 ? components[3] : 1.0
            ctx.fillStyle = .string("rgba(\(r),\(g),\(b),\(a))")
            _ = ctx.fillRect!(0, 0, width, height)
        } else {
            _ = ctx.clearRect!(0, 0, width, height)
        }

        // Set font
        let fontName: String
        if let font = textLayer.font as? String {
            fontName = font
        } else {
            fontName = "sans-serif"
        }
        ctx.font = .string("\(Int(textLayer.fontSize))px \(fontName)")

        // Set text color
        if let fgColor = textLayer.foregroundColor {
            let components = fgColor.components ?? [0, 0, 0, 1]
            let r = Int((components.count > 0 ? components[0] : 0) * 255)
            let g = Int((components.count > 1 ? components[1] : 0) * 255)
            let b = Int((components.count > 2 ? components[2] : 0) * 255)
            let a = components.count > 3 ? components[3] : 1.0
            ctx.fillStyle = .string("rgba(\(r),\(g),\(b),\(a))")
        } else {
            ctx.fillStyle = .string("rgba(255,255,255,1)")
        }

        // Set text alignment
        switch textLayer.alignmentMode {
        case .left:
            ctx.textAlign = .string("left")
        case .right:
            ctx.textAlign = .string("right")
        case .center:
            ctx.textAlign = .string("center")
        case .justified, .natural:
            ctx.textAlign = .string("start")
        default:
            ctx.textAlign = .string("start")
        }

        ctx.textBaseline = .string("top")

        // Calculate text position based on alignment
        let x: Double
        switch textLayer.alignmentMode {
        case .center:
            x = Double(width) / 2
        case .right:
            x = Double(width)
        default:
            x = 0
        }

        // Draw text (simple single-line for now)
        // For wrapped text, we would need to implement line breaking
        if textLayer.isWrapped {
            // Simple word wrapping
            drawWrappedText(ctx: ctx, text: text, x: x, y: 0,
                           maxWidth: Double(width), lineHeight: textLayer.fontSize * 1.2)
        } else {
            _ = ctx.fillText!(text, x, Double(textLayer.fontSize * 0.1))
        }

        // Get image data from canvas
        guard let imageData = ctx.getImageData!(0, 0, width, height).object else { return }
        guard let dataArray = imageData.data.object else { return }

        // Create WebGPU texture
        let textureDescriptor = GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.textureBinding, .copyDst, .renderAttachment]
        )

        let texture = device.createTexture(descriptor: textureDescriptor)

            // Copy image data to texture
            device.queue.writeTexture(
                destination: GPUImageCopyTexture(texture: texture),
                data: dataArray,
                dataLayout: GPUImageDataLayout(
                    offset: 0,
                    bytesPerRow: UInt32(width * 4),
                    rowsPerImage: UInt32(height)
                ),
                size: GPUExtent3D(width: UInt32(width), height: UInt32(height))
            )

            // Cache the texture
            textTextureCache[cacheKey] = texture
            gpuTexture = texture
        }

        // Create texture view
        let textureView = gpuTexture.createView()

        // Create bind group with the texture
        let texturedBindGroup = device.createBindGroup(
            descriptor: GPUBindGroupDescriptor(
                layout: texturedBindGroupLayout,
                entries: [
                    GPUBindGroupEntry(binding: 0, resource: .buffer(GPUBufferBinding(buffer: uniformBuffer))),
                    GPUBindGroupEntry(binding: 1, resource: .sampler(textureSampler)),
                    GPUBindGroupEntry(binding: 2, resource: .textureView(textureView))
                ]
            )
        )

        // Setup matrices
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(textLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(textLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Update uniforms
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: textLayer.opacity,
            cornerRadius: Float(textLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(textLayer.bounds.width), Float(textLayer.bounds.height))
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        // White vertex color for texture rendering (texture provides actual color)
        let white = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
        ]

        // Write vertices
        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Switch to textured pipeline and draw
        renderPass.setPipeline(texturedPipeline)
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch back to standard pipeline
        renderPass.setPipeline(pipeline)
    }

    /// Draws wrapped text on a Canvas2D context.
    private func drawWrappedText(ctx: JSObject, text: String, x: Double, y: Double,
                                 maxWidth: Double, lineHeight: CGFloat) {
        let words = text.split(separator: " ")
        var line = ""
        var currentY = y

        for word in words {
            let testLine = line.isEmpty ? String(word) : line + " " + String(word)
            let metrics = ctx.measureText!(testLine)
            let testWidth = metrics.width.number ?? 0

            if testWidth > maxWidth && !line.isEmpty {
                _ = ctx.fillText!(line, x, currentY)
                line = String(word)
                currentY += Double(lineHeight)
            } else {
                line = testLine
            }
        }

        if !line.isEmpty {
            _ = ctx.fillText!(line, x, currentY)
        }
    }

    // MARK: - Shape Layer Rendering

    /// Flattens a CGPath into an array of polylines (arrays of points).
    /// Each subpath becomes a separate polyline.
    private func flattenPath(_ path: CGPath, flatness: CGFloat = 0.5) -> [[CGPoint]] {
        var polylines: [[CGPoint]] = []
        var currentPolyline: [CGPoint] = []
        var currentPoint = CGPoint.zero
        var subpathStart = CGPoint.zero

        path.applyWithBlock { elementPtr in
            let element = elementPtr.pointee
            switch element.type {
            case .moveToPoint:
                if !currentPolyline.isEmpty {
                    polylines.append(currentPolyline)
                }
                currentPolyline = []
                let point = element.points![0]
                currentPolyline.append(point)
                currentPoint = point
                subpathStart = point

            case .addLineToPoint:
                let point = element.points![0]
                currentPolyline.append(point)
                currentPoint = point

            case .addQuadCurveToPoint:
                let control = element.points![0]
                let end = element.points![1]
                // Flatten quadratic bezier
                self.flattenQuadBezier(from: currentPoint, control: control, to: end,
                                       flatness: flatness, into: &currentPolyline)
                currentPoint = end

            case .addCurveToPoint:
                let control1 = element.points![0]
                let control2 = element.points![1]
                let end = element.points![2]
                // Flatten cubic bezier
                self.flattenCubicBezier(from: currentPoint, control1: control1,
                                        control2: control2, to: end,
                                        flatness: flatness, into: &currentPolyline)
                currentPoint = end

            case .closeSubpath:
                if !currentPolyline.isEmpty && currentPolyline.first != currentPolyline.last {
                    currentPolyline.append(subpathStart)
                }
                currentPoint = subpathStart

            @unknown default:
                break
            }
        }

        if !currentPolyline.isEmpty {
            polylines.append(currentPolyline)
        }

        return polylines
    }

    /// Flattens a quadratic bezier curve into line segments.
    private func flattenQuadBezier(from p0: CGPoint, control p1: CGPoint, to p2: CGPoint,
                                   flatness: CGFloat, into polyline: inout [CGPoint]) {
        // Simple recursive subdivision
        let midX = (p0.x + 2 * p1.x + p2.x) / 4
        let midY = (p0.y + 2 * p1.y + p2.y) / 4
        let dx = (p0.x + p2.x) / 2 - midX
        let dy = (p0.y + p2.y) / 2 - midY

        if dx * dx + dy * dy <= flatness * flatness {
            polyline.append(p2)
        } else {
            let p01 = CGPoint(x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2)
            let p12 = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
            let mid = CGPoint(x: (p01.x + p12.x) / 2, y: (p01.y + p12.y) / 2)
            flattenQuadBezier(from: p0, control: p01, to: mid, flatness: flatness, into: &polyline)
            flattenQuadBezier(from: mid, control: p12, to: p2, flatness: flatness, into: &polyline)
        }
    }

    /// Flattens a cubic bezier curve into line segments.
    private func flattenCubicBezier(from p0: CGPoint, control1 p1: CGPoint,
                                    control2 p2: CGPoint, to p3: CGPoint,
                                    flatness: CGFloat, into polyline: inout [CGPoint]) {
        // Check if curve is flat enough
        let dx1 = p1.x - p0.x
        let dy1 = p1.y - p0.y
        let dx2 = p2.x - p3.x
        let dy2 = p2.y - p3.y

        let d = sqrt(max(dx1 * dx1 + dy1 * dy1, dx2 * dx2 + dy2 * dy2))

        if d <= flatness {
            polyline.append(p3)
        } else {
            // Subdivide
            let p01 = CGPoint(x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2)
            let p12 = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
            let p23 = CGPoint(x: (p2.x + p3.x) / 2, y: (p2.y + p3.y) / 2)
            let p012 = CGPoint(x: (p01.x + p12.x) / 2, y: (p01.y + p12.y) / 2)
            let p123 = CGPoint(x: (p12.x + p23.x) / 2, y: (p12.y + p23.y) / 2)
            let mid = CGPoint(x: (p012.x + p123.x) / 2, y: (p012.y + p123.y) / 2)

            flattenCubicBezier(from: p0, control1: p01, control2: p012, to: mid,
                               flatness: flatness, into: &polyline)
            flattenCubicBezier(from: mid, control1: p123, control2: p23, to: p3,
                               flatness: flatness, into: &polyline)
        }
    }

    /// Triangulates a simple polygon using ear-clipping algorithm.
    /// Returns array of vertex indices forming triangles.
    private func triangulatePolygon(_ polygon: [CGPoint]) -> [Int] {
        guard polygon.count >= 3 else { return [] }

        // Remove duplicate last point if closed
        var points = polygon
        if let first = points.first, let last = points.last,
           first.x == last.x && first.y == last.y && points.count > 1 {
            points.removeLast()
        }

        guard points.count >= 3 else { return [] }

        var indices: [Int] = []
        var remaining = Array(0..<points.count)

        // Ensure polygon is counter-clockwise
        let area = signedArea(points)
        if area < 0 {
            remaining.reverse()
        }

        var safetyCounter = remaining.count * remaining.count

        while remaining.count > 3 && safetyCounter > 0 {
            safetyCounter -= 1
            var earFound = false

            for i in 0..<remaining.count {
                let prev = (i + remaining.count - 1) % remaining.count
                let next = (i + 1) % remaining.count

                let a = points[remaining[prev]]
                let b = points[remaining[i]]
                let c = points[remaining[next]]

                if isEar(a: a, b: b, c: c, polygon: points, remaining: remaining) {
                    indices.append(remaining[prev])
                    indices.append(remaining[i])
                    indices.append(remaining[next])
                    remaining.remove(at: i)
                    earFound = true
                    break
                }
            }

            if !earFound {
                // Fallback: just use first three vertices as triangle
                if remaining.count >= 3 {
                    indices.append(remaining[0])
                    indices.append(remaining[1])
                    indices.append(remaining[2])
                    remaining.remove(at: 1)
                }
            }
        }

        // Add final triangle
        if remaining.count == 3 {
            indices.append(remaining[0])
            indices.append(remaining[1])
            indices.append(remaining[2])
        }

        return indices
    }

    /// Calculates the signed area of a polygon.
    private func signedArea(_ polygon: [CGPoint]) -> CGFloat {
        var area: CGFloat = 0
        let n = polygon.count
        for i in 0..<n {
            let j = (i + 1) % n
            area += polygon[i].x * polygon[j].y
            area -= polygon[j].x * polygon[i].y
        }
        return area / 2
    }

    /// Checks if vertex b is an ear (can be clipped).
    private func isEar(a: CGPoint, b: CGPoint, c: CGPoint,
                       polygon: [CGPoint], remaining: [Int]) -> Bool {
        // Check if triangle is convex (counter-clockwise)
        let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if cross <= 0 {
            return false
        }

        // Check if any other point is inside the triangle
        for idx in remaining {
            let p = polygon[idx]
            if (p.x == a.x && p.y == a.y) ||
               (p.x == b.x && p.y == b.y) ||
               (p.x == c.x && p.y == c.y) {
                continue
            }
            if pointInTriangle(p: p, a: a, b: b, c: c) {
                return false
            }
        }

        return true
    }

    /// Checks if point p is inside triangle abc.
    private func pointInTriangle(p: CGPoint, a: CGPoint, b: CGPoint, c: CGPoint) -> Bool {
        let v0x = c.x - a.x
        let v0y = c.y - a.y
        let v1x = b.x - a.x
        let v1y = b.y - a.y
        let v2x = p.x - a.x
        let v2y = p.y - a.y

        let dot00 = v0x * v0x + v0y * v0y
        let dot01 = v0x * v1x + v0y * v1y
        let dot02 = v0x * v2x + v0y * v2y
        let dot11 = v1x * v1x + v1y * v1y
        let dot12 = v1x * v2x + v1y * v2y

        let invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        let u = (dot11 * dot02 - dot01 * dot12) * invDenom
        let v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= 0) && (v >= 0) && (u + v <= 1)
    }

    /// Generates stroke geometry for a polyline.
    /// Returns vertices forming triangle strips along the path.
    private func generateStrokeGeometry(polyline: [CGPoint], lineWidth: CGFloat,
                                        lineCap: CAShapeLayerLineCap,
                                        lineJoin: CAShapeLayerLineJoin) -> [CGPoint] {
        guard polyline.count >= 2 else { return [] }

        let halfWidth = lineWidth / 2
        var leftPoints: [CGPoint] = []
        var rightPoints: [CGPoint] = []

        for i in 0..<polyline.count {
            let curr = polyline[i]

            // Calculate tangent
            var tangent: CGPoint
            if i == 0 {
                let next = polyline[i + 1]
                tangent = CGPoint(x: next.x - curr.x, y: next.y - curr.y)
            } else if i == polyline.count - 1 {
                let prev = polyline[i - 1]
                tangent = CGPoint(x: curr.x - prev.x, y: curr.y - prev.y)
            } else {
                let prev = polyline[i - 1]
                let next = polyline[i + 1]
                tangent = CGPoint(x: next.x - prev.x, y: next.y - prev.y)
            }

            // Normalize tangent
            let len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y)
            if len > 0.0001 {
                tangent.x /= len
                tangent.y /= len
            }

            // Calculate normal (perpendicular)
            let normal = CGPoint(x: -tangent.y, y: tangent.x)

            // Offset points
            leftPoints.append(CGPoint(x: curr.x + normal.x * halfWidth,
                                      y: curr.y + normal.y * halfWidth))
            rightPoints.append(CGPoint(x: curr.x - normal.x * halfWidth,
                                       y: curr.y - normal.y * halfWidth))
        }

        // Build triangle strip as triangles
        var vertices: [CGPoint] = []
        for i in 0..<(leftPoints.count - 1) {
            // Triangle 1
            vertices.append(leftPoints[i])
            vertices.append(rightPoints[i])
            vertices.append(leftPoints[i + 1])

            // Triangle 2
            vertices.append(rightPoints[i])
            vertices.append(rightPoints[i + 1])
            vertices.append(leftPoints[i + 1])
        }

        // Add end caps
        if lineCap == .round {
            // Round cap at start
            let startCenter = polyline[0]
            let startTangent = CGPoint(x: polyline[1].x - polyline[0].x,
                                       y: polyline[1].y - polyline[0].y)
            addRoundCap(center: startCenter, tangent: startTangent, halfWidth: halfWidth,
                        isStart: true, into: &vertices)

            // Round cap at end
            let endCenter = polyline[polyline.count - 1]
            let endTangent = CGPoint(x: polyline[polyline.count - 1].x - polyline[polyline.count - 2].x,
                                     y: polyline[polyline.count - 1].y - polyline[polyline.count - 2].y)
            addRoundCap(center: endCenter, tangent: endTangent, halfWidth: halfWidth,
                        isStart: false, into: &vertices)
        } else if lineCap == .square {
            // Square caps extend by halfWidth
            let startCenter = polyline[0]
            let startDir = CGPoint(x: polyline[0].x - polyline[1].x,
                                   y: polyline[0].y - polyline[1].y)
            let startLen = sqrt(startDir.x * startDir.x + startDir.y * startDir.y)
            if startLen > 0 {
                let startNorm = CGPoint(x: startDir.x / startLen, y: startDir.y / startLen)
                let extendedStart = CGPoint(x: startCenter.x + startNorm.x * halfWidth,
                                            y: startCenter.y + startNorm.y * halfWidth)
                let perpStart = CGPoint(x: -startNorm.y, y: startNorm.x)

                vertices.append(leftPoints[0])
                vertices.append(CGPoint(x: extendedStart.x + perpStart.x * halfWidth,
                                        y: extendedStart.y + perpStart.y * halfWidth))
                vertices.append(rightPoints[0])

                vertices.append(rightPoints[0])
                vertices.append(CGPoint(x: extendedStart.x + perpStart.x * halfWidth,
                                        y: extendedStart.y + perpStart.y * halfWidth))
                vertices.append(CGPoint(x: extendedStart.x - perpStart.x * halfWidth,
                                        y: extendedStart.y - perpStart.y * halfWidth))
            }

            let endCenter = polyline[polyline.count - 1]
            let endDir = CGPoint(x: polyline[polyline.count - 1].x - polyline[polyline.count - 2].x,
                                 y: polyline[polyline.count - 1].y - polyline[polyline.count - 2].y)
            let endLen = sqrt(endDir.x * endDir.x + endDir.y * endDir.y)
            if endLen > 0 {
                let endNorm = CGPoint(x: endDir.x / endLen, y: endDir.y / endLen)
                let extendedEnd = CGPoint(x: endCenter.x + endNorm.x * halfWidth,
                                          y: endCenter.y + endNorm.y * halfWidth)
                let perpEnd = CGPoint(x: -endNorm.y, y: endNorm.x)

                vertices.append(leftPoints[leftPoints.count - 1])
                vertices.append(CGPoint(x: extendedEnd.x + perpEnd.x * halfWidth,
                                        y: extendedEnd.y + perpEnd.y * halfWidth))
                vertices.append(rightPoints[rightPoints.count - 1])

                vertices.append(rightPoints[rightPoints.count - 1])
                vertices.append(CGPoint(x: extendedEnd.x + perpEnd.x * halfWidth,
                                        y: extendedEnd.y + perpEnd.y * halfWidth))
                vertices.append(CGPoint(x: extendedEnd.x - perpEnd.x * halfWidth,
                                        y: extendedEnd.y - perpEnd.y * halfWidth))
            }
        }

        return vertices
    }

    /// Adds a round cap to the stroke geometry.
    private func addRoundCap(center: CGPoint, tangent: CGPoint, halfWidth: CGFloat,
                             isStart: Bool, into vertices: inout [CGPoint]) {
        let len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y)
        guard len > 0 else { return }

        let dir = CGPoint(x: tangent.x / len, y: tangent.y / len)
        let normal = CGPoint(x: -dir.y, y: dir.x)

        // Create semicircle with 8 segments
        let segments = 8
        let startAngle: CGFloat = isStart ? CGFloat.pi / 2 : -CGFloat.pi / 2
        let endAngle: CGFloat = isStart ? 3 * CGFloat.pi / 2 : CGFloat.pi / 2

        for i in 0..<segments {
            let angle1 = startAngle + CGFloat(i) * (endAngle - startAngle) / CGFloat(segments)
            let angle2 = startAngle + CGFloat(i + 1) * (endAngle - startAngle) / CGFloat(segments)

            let p1 = CGPoint(
                x: center.x + halfWidth * (cos(angle1) * normal.x - sin(angle1) * dir.x),
                y: center.y + halfWidth * (cos(angle1) * normal.y - sin(angle1) * dir.y)
            )
            let p2 = CGPoint(
                x: center.x + halfWidth * (cos(angle2) * normal.x - sin(angle2) * dir.x),
                y: center.y + halfWidth * (cos(angle2) * normal.y - sin(angle2) * dir.y)
            )

            vertices.append(center)
            vertices.append(p1)
            vertices.append(p2)
        }
    }

    /// Renders a CAShapeLayer with its path.
    private func renderShapeLayer(
        _ shapeLayer: CAShapeLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let path = shapeLayer.path else { return }
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        // Flatten the path
        let polylines = flattenPath(path)
        guard !polylines.isEmpty else { return }

        // Render fill if fillColor is set
        if let fillColor = shapeLayer.fillColor {
            for polyline in polylines {
                guard polyline.count >= 3 else { continue }

                // Triangulate the polygon
                let indices = triangulatePolygon(polyline)
                guard !indices.isEmpty else { continue }

                guard currentLayerIndex < Self.maxLayers else { return }
                let layerIndex = currentLayerIndex
                currentLayerIndex += 1

                // Create vertices from triangulation
                var vertices: [CARendererVertex] = []
                let colorComponents = cgColorToSIMD4(fillColor)

                for idx in indices {
                    let point = polyline[idx]
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: SIMD2(0, 0),  // Not used for solid color
                        color: colorComponents
                    ))
                }

                guard !vertices.isEmpty else { continue }

                // Update uniforms
                var uniforms = CARendererUniforms(
                    mvpMatrix: modelMatrix,
                    opacity: shapeLayer.opacity,
                    cornerRadius: 0,
                    layerSize: .zero  // No SDF-based corner radius for shapes
                )

                let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
                let uniformData = createFloat32Array(from: &uniforms)
                device.queue.writeBuffer(
                    uniformBuffer,
                    bufferOffset: uniformOffset,
                    data: uniformData
                )

                // Write vertices
                let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
                let vertexData = createFloat32Array(from: &vertices)
                device.queue.writeBuffer(
                    vertexBuffer,
                    bufferOffset: vertexOffset,
                    data: vertexData
                )

                // Draw
                renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
                renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
                renderPass.draw(vertexCount: UInt32(vertices.count))
            }
        }

        // Render stroke if strokeColor is set
        if let strokeColor = shapeLayer.strokeColor, shapeLayer.lineWidth > 0 {
            for polyline in polylines {
                guard polyline.count >= 2 else { continue }

                // Generate stroke geometry
                let strokeVertices = generateStrokeGeometry(
                    polyline: polyline,
                    lineWidth: shapeLayer.lineWidth,
                    lineCap: shapeLayer.lineCap,
                    lineJoin: shapeLayer.lineJoin
                )
                guard !strokeVertices.isEmpty else { continue }

                guard currentLayerIndex < Self.maxLayers else { return }
                let layerIndex = currentLayerIndex
                currentLayerIndex += 1

                // Create vertices
                var vertices: [CARendererVertex] = []
                let colorComponents = cgColorToSIMD4(strokeColor)

                for point in strokeVertices {
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: SIMD2(0, 0),
                        color: colorComponents
                    ))
                }

                // Update uniforms
                var uniforms = CARendererUniforms(
                    mvpMatrix: modelMatrix,
                    opacity: shapeLayer.opacity,
                    cornerRadius: 0,
                    layerSize: .zero
                )

                let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
                let uniformData = createFloat32Array(from: &uniforms)
                device.queue.writeBuffer(
                    uniformBuffer,
                    bufferOffset: uniformOffset,
                    data: uniformData
                )

                // Write vertices
                let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
                let vertexData = createFloat32Array(from: &vertices)
                device.queue.writeBuffer(
                    vertexBuffer,
                    bufferOffset: vertexOffset,
                    data: vertexData
                )

                // Draw
                renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
                renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
                renderPass.draw(vertexCount: UInt32(vertices.count))
            }
        }
    }

    /// Converts CGColor to SIMD4<Float>.
    private func cgColorToSIMD4(_ color: CGColor) -> SIMD4<Float> {
        guard let components = color.components, components.count >= 4 else {
            return SIMD4<Float>(0, 0, 0, 1)
        }
        return SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
    }

    // MARK: - Gradient Layer Rendering

    /// Renders a CAGradientLayer with its gradient colors.
    private func renderGradientLayer(
        _ gradientLayer: CAGradientLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let colors = gradientLayer.colors, !colors.isEmpty else { return }

        guard currentLayerIndex < Self.maxLayers else { return }

        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        // Create scale matrix for layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(gradientLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(gradientLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Prepare gradient uniforms
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: gradientLayer.opacity,
            cornerRadius: Float(gradientLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(gradientLayer.bounds.width), Float(gradientLayer.bounds.height)),
            borderWidth: 0,
            renderMode: 2.0,  // Gradient mode
            gradientStartPoint: SIMD2<Float>(Float(gradientLayer.startPoint.x), Float(gradientLayer.startPoint.y)),
            gradientEndPoint: SIMD2<Float>(Float(gradientLayer.endPoint.x), Float(gradientLayer.endPoint.y)),
            gradientColorCount: Float(min(colors.count, kMaxGradientStops))
        )

        // Extract gradient colors
        var gradientColors: [SIMD4<Float>] = []
        for colorAny in colors.prefix(kMaxGradientStops) {
            if let cgColor = colorAny as? CGColor,
               let components = cgColor.components,
               components.count >= 4 {
                gradientColors.append(SIMD4<Float>(
                    Float(components[0]),
                    Float(components[1]),
                    Float(components[2]),
                    Float(components[3])
                ))
            } else {
                gradientColors.append(SIMD4<Float>(0, 0, 0, 1))
            }
        }

        // Pad to 8 colors
        while gradientColors.count < kMaxGradientStops {
            gradientColors.append(.zero)
        }

        uniforms.gradientColors = (
            gradientColors[0], gradientColors[1], gradientColors[2], gradientColors[3],
            gradientColors[4], gradientColors[5], gradientColors[6], gradientColors[7]
        )

        // Extract or generate locations
        let colorCount = min(colors.count, kMaxGradientStops)
        var locations: [Float] = []
        if let providedLocations = gradientLayer.locations, !providedLocations.isEmpty {
            locations = providedLocations.prefix(kMaxGradientStops).map { Float($0) }
        } else {
            // Generate evenly spaced locations
            for i in 0..<colorCount {
                locations.append(Float(i) / Float(max(colorCount - 1, 1)))
            }
        }

        // Pad locations to 8
        while locations.count < kMaxGradientStops {
            locations.append(1.0)
        }

        uniforms.gradientLocations = SIMD4<Float>(locations[0], locations[1], locations[2], locations[3])
        uniforms.gradientLocations2 = SIMD4<Float>(locations[4], locations[5], locations[6], locations[7])

        // Write uniforms
        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        // Create vertices (color doesn't matter for gradient, shader uses gradient uniforms)
        let dummyColor = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: dummyColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: dummyColor),
        ]

        // Write vertices
        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Draw
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }
}

#endif
