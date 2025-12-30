#if arch(wasm32)
import Foundation

// MARK: - WebGPU Shader Code

/// A container for all WGSL shader code used by the WebGPU renderer.
///
/// ## Coordinate System: SpriteKit-Compatible (Y-up)
///
/// OpenCoreAnimation uses a SpriteKit-compatible coordinate system:
/// - **Origin**: Bottom-left corner (0, 0)
/// - **X-axis**: Positive X goes RIGHT
/// - **Y-axis**: Positive Y goes UP
///
/// Reference: https://developer.apple.com/documentation/spritekit/about-spritekit-coordinate-systems
///
/// > "A positive x coordinate goes to the right and a positive y coordinate goes up the screen."
///
/// ### Coordinate Transformations
///
/// The projection matrix transforms world coordinates to WebGPU NDC (Normalized Device Coordinates):
/// ```
/// World (Y-up)          NDC
/// Y=height  ───         Y=+1  ───
///           │             │
///           │     →       │
///           │             │
/// Y=0       ───         Y=-1  ───
/// (bottom)              (bottom)
/// ```
///
/// ### Texture Coordinate Handling
///
/// Textures store pixel data with row 0 at the TOP (standard image format).
/// For Y-up rendering, the V coordinate must be flipped:
/// ```
/// Screen (Y-up)         Texture
/// Y=height ───          V=0 ─── (image top)
///          │              │
///          │    flip V    │
///          │              │
/// Y=0      ───          V=1 ─── (image bottom)
/// ```
///
/// This applies to:
/// - Image rendering (CGImage contents)
/// - Text rendering (Canvas2D rendered text)
/// - 9-patch rendering (contentsCenter)
///
/// Solid color rendering does NOT flip V because texCoord is used for
/// position-based calculations (corner radius, gradients), not texture sampling.
public enum CAWebGPUShaders {

    // MARK: - Main Layer Shader

    /// Shader code for basic layer rendering with corner radius, border, and gradient support.
    public static let main = """
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
        // Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
        // Corresponds to: (bottom-left, bottom-right, top-left, top-right)
        cornerRadii: vec4<f32>,
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

    // Signed distance function for a rounded rectangle with uniform radius
    fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
        let q = abs(p) - halfSize + radius;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    }

    // Signed distance function for a rounded rectangle with per-corner radii
    // radii: (minXminY, maxXminY, minXmaxY, maxXmaxY) - corresponds to CACornerMask corners
    // In screen space with texCoord centered (Y-up coordinate system):
    //   - negative x, negative y -> minXminY (bottom-left)
    //   - positive x, negative y -> maxXminY (bottom-right)
    //   - negative x, positive y -> minXmaxY (top-left)
    //   - positive x, positive y -> maxXmaxY (top-right)
    fn sdRoundedBoxVariable(p: vec2<f32>, halfSize: vec2<f32>, radii: vec4<f32>) -> f32 {
        // Select the appropriate corner radius based on which quadrant we're in
        // radii.x = minXminY (x<0, y<0) - bottom-left
        // radii.y = maxXminY (x>0, y<0) - bottom-right
        // radii.z = minXmaxY (x<0, y>0) - top-left
        // radii.w = maxXmaxY (x>0, y>0) - top-right
        var r: f32;
        if (p.x >= 0.0) {
            if (p.y >= 0.0) {
                r = radii.w;  // maxXmaxY - top-right
            } else {
                r = radii.y;  // maxXminY - bottom-right
            }
        } else {
            if (p.y >= 0.0) {
                r = radii.z;  // minXmaxY - top-left
            } else {
                r = radii.x;  // minXminY - bottom-left
            }
        }

        // Clamp radius to half the smaller dimension
        let maxRadius = min(halfSize.x, halfSize.y);
        r = min(r, maxRadius);

        let q = abs(p) - halfSize + r;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
    }

    // Check if all corner radii are the same (for optimization)
    fn hasUniformCorners(radii: vec4<f32>) -> bool {
        return radii.x == radii.y && radii.y == radii.z && radii.z == radii.w;
    }

    // Get effective corner radii (uses cornerRadii if set, otherwise falls back to cornerRadius)
    fn getEffectiveRadii() -> vec4<f32> {
        // If cornerRadii are all zero, use the legacy cornerRadius for all corners
        if (uniforms.cornerRadii.x == 0.0 && uniforms.cornerRadii.y == 0.0 &&
            uniforms.cornerRadii.z == 0.0 && uniforms.cornerRadii.w == 0.0) {
            return vec4<f32>(uniforms.cornerRadius);
        }
        return uniforms.cornerRadii;
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

        // Get effective corner radii (per-corner or uniform)
        let radii = getEffectiveRadii();
        let hasAnyCornerRadius = radii.x > 0.0 || radii.y > 0.0 || radii.z > 0.0 || radii.w > 0.0;

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

            // Apply corner radius if set (using per-corner radii)
            if (hasAnyCornerRadius) {
                let dist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);
                let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
                gradientColor.a *= alpha;
            }

            return gradientColor;
        }

        // Border rendering mode
        if (uniforms.renderMode > 0.5) {
            // Calculate outer signed distance with per-corner radii
            let outerDist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);

            // Calculate inner radii (subtract border width from each corner)
            let innerRadii = vec4<f32>(
                max(0.0, radii.x - uniforms.borderWidth),
                max(0.0, radii.y - uniforms.borderWidth),
                max(0.0, radii.z - uniforms.borderWidth),
                max(0.0, radii.w - uniforms.borderWidth)
            );
            let innerHalfSize = halfSize - uniforms.borderWidth;
            let innerDist = sdRoundedBoxVariable(pixelCoord, innerHalfSize, innerRadii);

            // Border is where we're inside outer but outside inner
            let outerAlpha = 1.0 - smoothstep(-1.0, 1.0, outerDist);
            let innerAlpha = 1.0 - smoothstep(-1.0, 1.0, innerDist);
            let borderAlpha = outerAlpha - innerAlpha;

            return vec4<f32>(input.color.rgb, input.color.a * borderAlpha);
        }

        // Fill rendering mode (default)
        if (!hasAnyCornerRadius) {
            return input.color;
        }

        // Calculate signed distance for fill using per-corner radii
        let dist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);

        // Anti-aliased edge (smooth over 1 pixel)
        let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);

        return vec4<f32>(input.color.rgb, input.color.a * alpha);
    }
    """

    // MARK: - Textured Layer Shader

    /// Shader code for textured rendering.
    public static let textured = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
        // Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
        cornerRadii: vec4<f32>,
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

    // Signed distance function for a rounded rectangle with per-corner radii
    fn sdRoundedBoxVariable(p: vec2<f32>, halfSize: vec2<f32>, radii: vec4<f32>) -> f32 {
        var r: f32;
        if (p.x >= 0.0) {
            if (p.y >= 0.0) {
                r = radii.w;  // maxXmaxY - top-right
            } else {
                r = radii.y;  // maxXminY - bottom-right
            }
        } else {
            if (p.y >= 0.0) {
                r = radii.z;  // minXmaxY - top-left
            } else {
                r = radii.x;  // minXminY - bottom-left
            }
        }
        let maxRadius = min(halfSize.x, halfSize.y);
        r = min(r, maxRadius);
        let q = abs(p) - halfSize + r;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
    }

    // Get effective corner radii
    fn getEffectiveRadii() -> vec4<f32> {
        if (uniforms.cornerRadii.x == 0.0 && uniforms.cornerRadii.y == 0.0 &&
            uniforms.cornerRadii.z == 0.0 && uniforms.cornerRadii.w == 0.0) {
            return vec4<f32>(uniforms.cornerRadius);
        }
        return uniforms.cornerRadii;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        // Sample texture
        var texColor = textureSample(textureData, textureSampler, input.texCoord);

        // Apply opacity
        texColor.a *= uniforms.opacity;

        // Get effective corner radii
        let radii = getEffectiveRadii();
        let hasAnyCornerRadius = radii.x > 0.0 || radii.y > 0.0 || radii.z > 0.0 || radii.w > 0.0;

        // Apply corner radius if set
        if (hasAnyCornerRadius && uniforms.layerSize.x > 0.0 && uniforms.layerSize.y > 0.0) {
            let pixelCoord = (input.texCoord - 0.5) * uniforms.layerSize;
            let halfSize = uniforms.layerSize * 0.5;
            let dist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);
            let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
            texColor.a *= alpha;
        }

        return texColor;
    }
    """

    // MARK: - Shadow Mask Shader

    /// Shader code for shadow mask generation.
    public static let shadowMask = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
        // Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
        cornerRadii: vec4<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        // Full-screen quad from vertex index
        let positions = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
            vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
        );
        let pos = positions[vertexIndex];
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(pos, 0.0, 1.0);
        output.texCoord = pos;
        return output;
    }

    // Per-corner SDF
    fn sdRoundedBoxVariable(p: vec2<f32>, halfSize: vec2<f32>, radii: vec4<f32>) -> f32 {
        var r: f32;
        if (p.x >= 0.0) {
            if (p.y >= 0.0) { r = radii.w; } else { r = radii.y; }
        } else {
            if (p.y >= 0.0) { r = radii.z; } else { r = radii.x; }
        }
        r = min(r, min(halfSize.x, halfSize.y));
        let q = abs(p) - halfSize + r;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
    }

    fn getEffectiveRadii() -> vec4<f32> {
        if (uniforms.cornerRadii.x == 0.0 && uniforms.cornerRadii.y == 0.0 &&
            uniforms.cornerRadii.z == 0.0 && uniforms.cornerRadii.w == 0.0) {
            return vec4<f32>(uniforms.cornerRadius);
        }
        return uniforms.cornerRadii;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        if (uniforms.layerSize.x <= 0.0 || uniforms.layerSize.y <= 0.0) {
            return vec4<f32>(1.0);
        }
        let pixelCoord = (input.texCoord - 0.5) * uniforms.layerSize;
        let halfSize = uniforms.layerSize * 0.5;
        let radii = getEffectiveRadii();
        let dist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);
        let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
        return vec4<f32>(alpha, alpha, alpha, alpha);
    }
    """

    // MARK: - Blur Shaders

    /// Shader code for Gaussian blur (horizontal pass).
    public static let blurHorizontal = """
    struct BlurUniforms {
        texelSize: vec2<f32>,
        blurRadius: f32,
        padding: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: BlurUniforms;
    @group(0) @binding(1) var inputTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        let positions = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
            vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
        );
        let texCoords = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
            vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0)
        );
        var output: VertexOutput;
        output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
        output.texCoord = texCoords[vertexIndex];
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
        var result = textureSample(inputTexture, texSampler, input.texCoord) * weights[0];

        for (var i = 1; i < 5; i++) {
            let offset = vec2<f32>(f32(i) * uniforms.blurRadius * uniforms.texelSize.x, 0.0);
            result += textureSample(inputTexture, texSampler, input.texCoord + offset) * weights[i];
            result += textureSample(inputTexture, texSampler, input.texCoord - offset) * weights[i];
        }
        return result;
    }
    """

    /// Shader code for Gaussian blur (vertical pass).
    public static let blurVertical = """
    struct BlurUniforms {
        texelSize: vec2<f32>,
        blurRadius: f32,
        padding: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: BlurUniforms;
    @group(0) @binding(1) var inputTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        let positions = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
            vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
        );
        let texCoords = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
            vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0)
        );
        var output: VertexOutput;
        output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
        output.texCoord = texCoords[vertexIndex];
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
        var result = textureSample(inputTexture, texSampler, input.texCoord) * weights[0];

        for (var i = 1; i < 5; i++) {
            let offset = vec2<f32>(0.0, f32(i) * uniforms.blurRadius * uniforms.texelSize.y);
            result += textureSample(inputTexture, texSampler, input.texCoord + offset) * weights[i];
            result += textureSample(inputTexture, texSampler, input.texCoord - offset) * weights[i];
        }
        return result;
    }
    """

    // MARK: - Shadow Composite Shader

    /// Shader code for shadow compositing.
    public static let shadowComposite = """
    struct ShadowUniforms {
        mvpMatrix: mat4x4<f32>,
        shadowColor: vec4<f32>,
        shadowOffset: vec2<f32>,
        layerSize: vec2<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: ShadowUniforms;
    @group(0) @binding(1) var shadowTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) texCoord: vec2<f32>,
        @location(2) color: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.texCoord = input.texCoord;
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let shadowAlpha = textureSample(shadowTexture, texSampler, input.texCoord).r;
        return vec4<f32>(uniforms.shadowColor.rgb, uniforms.shadowColor.a * shadowAlpha);
    }
    """

    // MARK: - Masked Rendering Shader

    /// Shader code for masked rendering.
    public static let masked = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
        // Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
        cornerRadii: vec4<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var maskTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

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

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let maskAlpha = textureSample(maskTexture, texSampler, input.texCoord).a;
        return vec4<f32>(input.color.rgb, input.color.a * maskAlpha);
    }
    """

    // MARK: - Particle Shader

    /// Shader code for particle rendering.
    public static let particle = """
    struct ParticleUniforms {
        mvpMatrix: mat4x4<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: ParticleUniforms;
    @group(0) @binding(1) var particleTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct ParticleInstance {
        @location(0) position: vec3<f32>,
        @location(1) color: vec4<f32>,
        @location(2) scaleRotation: vec2<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
        @location(1) color: vec4<f32>,
    }

    @vertex
    fn vertexMain(
        @builtin(vertex_index) vertexIndex: u32,
        instance: ParticleInstance
    ) -> VertexOutput {
        let corners = array<vec2<f32>, 6>(
            vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5), vec2<f32>(-0.5, 0.5),
            vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.5)
        );
        var corner = corners[vertexIndex];

        // Apply rotation
        let cos_r = cos(instance.scaleRotation.y);
        let sin_r = sin(instance.scaleRotation.y);
        corner = vec2<f32>(
            corner.x * cos_r - corner.y * sin_r,
            corner.x * sin_r + corner.y * cos_r
        );

        // Apply scale
        corner *= instance.scaleRotation.x;

        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(instance.position + vec3<f32>(corner, 0.0), 1.0);
        output.texCoord = corners[vertexIndex] + vec2<f32>(0.5);
        output.color = instance.color;
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let texColor = textureSample(particleTexture, texSampler, input.texCoord);
        return texColor * input.color;
    }
    """
}

#endif
