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
    /// Full-screen depth reset used to isolate independent CATransformLayer groups.
    /// Color writes are disabled by the matching render pipeline.
    public static let depthClear = """
    @vertex
    fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
        let positions = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>(3.0, -1.0),
            vec2<f32>(-1.0, 3.0)
        );
        return vec4<f32>(positions[vertexIndex], 0.0, 1.0);
    }

    @fragment
    fn fragmentMain() -> @location(0) vec4<f32> {
        return vec4<f32>(0.0);
    }
    """


    // MARK: - Main Layer Shader

    /// Shader code for basic layer rendering with corner radius, border, and gradient support.
    public static let main = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
        borderWidth: f32,
        renderMode: f32,  // 0 = fill, 1 = border, 2 = axial, 3 = radial, 4 = conic
        gradientStartPoint: vec2<f32>,
        gradientEndPoint: vec2<f32>,
        gradientColorCount: f32,
        edgeAntialiasingParameters: vec3<f32>,
        // Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
        // Corresponds to: (bottom-left, bottom-right, top-left, top-right)
        cornerRadii: vec4<f32>,
        gradientColorMultiplier: vec4<f32>,
        gradientStopOffset: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct GradientStop {
        color: vec4<f32>,
        locationAndPadding: vec4<f32>,
    }

    @group(0) @binding(1) var<storage, read> gradientStops: array<GradientStop>;

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
        output.color = vec4<f32>(input.color.rgb, input.color.a * uniforms.opacity);
        return output;
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
        let outside = max(q, vec2<f32>(0.0));
        let exponent = max(uniforms.edgeAntialiasingParameters.y, 1.0);
        let curveLength = pow(
            pow(outside.x, exponent) + pow(outside.y, exponent),
            1.0 / exponent
        );
        return curveLength + min(max(q.x, q.y), 0.0) - r;
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

    // Calculates coverage for independently selectable layer edges. Coordinates
    // are layer-local (left/bottom = 0, right/top = 1), so derivatives preserve
    // a one-pixel transition under affine and perspective transforms.
    fn edgeCoverage(layerCoord: vec2<f32>) -> f32 {
        let mask = u32(max(uniforms.edgeAntialiasingParameters.x, 0.0));
        if (mask == 0u) {
            return 1.0;
        }
        let footprint = max(fwidth(layerCoord), vec2<f32>(0.000001));
        var coverage = 1.0;
        if ((mask & 1u) != 0u) {
            coverage *= smoothstep(0.0, footprint.x, layerCoord.x);
        }
        if ((mask & 2u) != 0u) {
            coverage *= smoothstep(0.0, footprint.x, 1.0 - layerCoord.x);
        }
        if ((mask & 4u) != 0u) {
            coverage *= smoothstep(0.0, footprint.y, layerCoord.y);
        }
        if ((mask & 8u) != 0u) {
            coverage *= smoothstep(0.0, footprint.y, 1.0 - layerCoord.y);
        }
        return coverage;
    }

    // Get gradient color at index
    fn getGradientColor(index: i32) -> vec4<f32> {
        let stopIndex = u32(uniforms.gradientStopOffset) + u32(index);
        return gradientStops[stopIndex].color * uniforms.gradientColorMultiplier;
    }

    // Get gradient location at index
    fn getGradientLocation(index: i32) -> f32 {
        let stopIndex = u32(uniforms.gradientStopOffset) + u32(index);
        return gradientStops[stopIndex].locationAndPadding.x;
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

    // Matches CAGradientLayer's unit-coordinate geometry. A negative value
    // identifies radial geometry that has no drawable two-dimensional extent.
    fn gradientParameter(position: vec2<f32>) -> f32 {
        let delta = uniforms.gradientEndPoint - uniforms.gradientStartPoint;
        let relative = position - uniforms.gradientStartPoint;

        if (uniforms.renderMode < 2.5) {
            let squaredLength = dot(delta, delta);
            if (squaredLength <= 0.0) {
                return 0.0;
            }
            return dot(relative, delta) / squaredLength;
        }

        if (uniforms.renderMode < 3.5) {
            let radius = abs(delta);
            if (radius.x <= 0.0 || radius.y <= 0.0) {
                return -1.0;
            }
            return length(relative / radius);
        }

        var directionAngle: f32 = 0.0;
        if (dot(delta, delta) > 0.0) {
            directionAngle = atan2(delta.y, delta.x);
        }
        if (dot(relative, relative) <= 0.0) {
            return 0.0;
        }
        let pointAngle = atan2(relative.y, relative.x);
        return fract((pointAngle - directionAngle) / 6.283185307179586 + 1.0);
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let layerEdgeCoverage = edgeCoverage(input.texCoord);
        // Handle case where layer size is zero
        if (uniforms.layerSize.x <= 0.0 || uniforms.layerSize.y <= 0.0) {
            if (input.color.a <= 0.0) {
                discard;
            }
            return vec4<f32>(input.color.rgb, input.color.a * layerEdgeCoverage);
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
            let t = gradientParameter(input.texCoord);
            if (t < 0.0) {
                discard;
            }

            // Sample gradient
            var gradientColor = sampleGradient(t);
            gradientColor.a *= uniforms.opacity * layerEdgeCoverage;

            // Apply corner radius if set (using per-corner radii)
            if (hasAnyCornerRadius) {
                let dist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);
                let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
                gradientColor.a *= alpha;
            }

            if (gradientColor.a <= 0.0) {
                discard;
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

            let borderColor = vec4<f32>(input.color.rgb, input.color.a * borderAlpha * layerEdgeCoverage);
            if (borderColor.a <= 0.0) {
                discard;
            }
            return borderColor;
        }

        // Fill rendering mode (default)
        if (!hasAnyCornerRadius) {
            if (input.color.a <= 0.0) {
                discard;
            }
            return vec4<f32>(input.color.rgb, input.color.a * layerEdgeCoverage);
        }

        // Calculate signed distance for fill using per-corner radii
        let dist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);

        // Anti-aliased edge (smooth over 1 pixel)
        let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);

        let fillColor = vec4<f32>(input.color.rgb, input.color.a * alpha * layerEdgeCoverage);
        if (fillColor.a <= 0.0) {
            discard;
        }
        return fillColor;
    }

    // Stencil clip fragment shader: discards fragments outside the rounded rectangle
    // so that the stencil buffer is only written within the rounded rect bounds.
    // Used by masksToBounds + cornerRadius clipping.
    @fragment
    fn stencilClipFragment(input: VertexOutput) -> @location(0) vec4<f32> {
        if (uniforms.layerSize.x <= 0.0 || uniforms.layerSize.y <= 0.0) {
            return vec4<f32>(1.0);
        }
        let pixelCoord = (input.texCoord - 0.5) * uniforms.layerSize;
        let halfSize = uniforms.layerSize * 0.5;
        let radii = getEffectiveRadii();
        let dist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);
        if (dist > 0.0) {
            discard;
        }
        return vec4<f32>(1.0);
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
        samplingBias: f32,
        edgeAntialiasingMask: f32,
        cornerCurveExponent: f32,
        padding2: f32,
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
        @location(2) layerCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.texCoord = input.texCoord;
        output.color = input.color;
        output.layerCoord = input.position;
        return output;
    }

    fn edgeCoverage(layerCoord: vec2<f32>) -> f32 {
        let mask = u32(max(uniforms.edgeAntialiasingMask, 0.0));
        if (mask == 0u) {
            return 1.0;
        }
        let footprint = max(fwidth(layerCoord), vec2<f32>(0.000001));
        var coverage = 1.0;
        if ((mask & 1u) != 0u) {
            coverage *= smoothstep(0.0, footprint.x, layerCoord.x);
        }
        if ((mask & 2u) != 0u) {
            coverage *= smoothstep(0.0, footprint.x, 1.0 - layerCoord.x);
        }
        if ((mask & 4u) != 0u) {
            coverage *= smoothstep(0.0, footprint.y, layerCoord.y);
        }
        if ((mask & 8u) != 0u) {
            coverage *= smoothstep(0.0, footprint.y, 1.0 - layerCoord.y);
        }
        return coverage;
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
        let outside = max(q, vec2<f32>(0.0));
        let exponent = max(uniforms.cornerCurveExponent, 1.0);
        let curveLength = pow(
            pow(outside.x, exponent) + pow(outside.y, exponent),
            1.0 / exponent
        );
        return curveLength + min(max(q.x, q.y), 0.0) - r;
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
        var texColor = textureSampleBias(
            textureData,
            textureSampler,
            input.texCoord,
            uniforms.samplingBias
        ) * input.color;

        // Apply opacity
        texColor.a *= uniforms.opacity * edgeCoverage(input.layerCoord);

        // Get effective corner radii
        let radii = getEffectiveRadii();
        let hasAnyCornerRadius = radii.x > 0.0 || radii.y > 0.0 || radii.z > 0.0 || radii.w > 0.0;

        // Apply corner radius if set
        if (hasAnyCornerRadius && uniforms.layerSize.x > 0.0 && uniforms.layerSize.y > 0.0) {
            let pixelCoord = (input.layerCoord - 0.5) * uniforms.layerSize;
            let halfSize = uniforms.layerSize * 0.5;
            let dist = sdRoundedBoxVariable(pixelCoord, halfSize, radii);
            let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
            texColor.a *= alpha;
        }

        if (texColor.a <= 0.0) {
            discard;
        }
        return texColor;
    }
    """

    /// Shader for compositing premultiplied-alpha offscreen captures.
    public static let premultipliedTextured = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
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
        output.color = input.color;
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let color = textureSample(textureData, textureSampler, input.texCoord)
            * input.color
            * uniforms.opacity;
        if (color.a <= 0.0) {
            discard;
        }
        return color;
    }
    """

    /// Interpolates two premultiplied layer captures without source-over feedback.
    public static let transitionFade = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        colorMultiplier: vec4<f32>,
        parameters: vec4<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var textureSampler: sampler;
    @group(0) @binding(2) var sourceTexture: texture_2d<f32>;
    @group(0) @binding(3) var targetTexture: texture_2d<f32>;

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
        let progress = clamp(uniforms.parameters.y, 0.0, 1.0);
        let opacity = clamp(uniforms.parameters.x, 0.0, 1.0);
        let source = textureSample(sourceTexture, textureSampler, input.texCoord);
        let targetColor = textureSample(targetTexture, textureSampler, input.texCoord);
        let interpolated = mix(source, targetColor, progress);
        let alphaScale = uniforms.colorMultiplier.a * opacity;
        let result = vec4<f32>(
            interpolated.rgb * uniforms.colorMultiplier.rgb * alphaScale,
            interpolated.a * alphaScale
        );
        if (result.a <= 0.0) {
            discard;
        }
        return result;
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
        borderWidth: f32,
        renderMode: f32,
        gradientStartPoint: vec2<f32>,
        gradientEndPoint: vec2<f32>,
        gradientColorCount: f32,
        edgeAntialiasingParameters: vec3<f32>,
        // Per-corner radii: (minXminY, maxXminY, minXmaxY, maxXmaxY)
        cornerRadii: vec4<f32>,
        gradientColorMultiplier: vec4<f32>,
        gradientStopOffset: f32,
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
        let outside = max(q, vec2<f32>(0.0));
        let exponent = max(uniforms.edgeAntialiasingParameters.y, 1.0);
        let curveLength = pow(
            pow(outside.x, exponent) + pow(outside.y, exponent),
            1.0 / exponent
        );
        return curveLength + min(max(q.x, q.y), 0.0) - r;
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

    /// Horizontal shadow blur that derives the silhouette from captured alpha.
    public static let shadowAlphaBlurHorizontal = """
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
        var alpha = textureSample(inputTexture, texSampler, input.texCoord).a * weights[0];

        for (var i = 1; i < 5; i++) {
            let offset = vec2<f32>(f32(i) * uniforms.blurRadius * uniforms.texelSize.x, 0.0);
            alpha += textureSample(inputTexture, texSampler, input.texCoord + offset).a * weights[i];
            alpha += textureSample(inputTexture, texSampler, input.texCoord - offset).a * weights[i];
        }
        return vec4<f32>(alpha);
    }
    """

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
        let safeSize = max(uniforms.layerSize, vec2<f32>(1.0));
        let shadowOffsetUV = vec2<f32>(
            uniforms.shadowOffset.x / safeSize.x,
            uniforms.shadowOffset.y / safeSize.y
        );
        // Texture V points down while layer-space Y points up.
        let shadowCoordinate = input.texCoord + vec2<f32>(-shadowOffsetUV.x, shadowOffsetUV.y);
        let lowerCoverage = step(vec2<f32>(0.0), shadowCoordinate);
        let upperCoverage = step(shadowCoordinate, vec2<f32>(1.0));
        let inBounds = lowerCoverage.x * lowerCoverage.y * upperCoverage.x * upperCoverage.y;
        let shadowAlpha = textureSample(shadowTexture, texSampler, shadowCoordinate).r * inBounds;
        return vec4<f32>(uniforms.shadowColor.rgb, uniforms.shadowColor.a * shadowAlpha);
    }
    """

    /// Combines a layer-local blurred alpha mask behind premultiplied content.
    public static let rasterizedShadowComposite = """
    struct RasterShadowUniforms {
        shadowColor: vec4<f32>,
        shadowOffsetUV: vec2<f32>,
        padding: vec2<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: RasterShadowUniforms;
    @group(0) @binding(1) var contentTexture: texture_2d<f32>;
    @group(0) @binding(2) var shadowTexture: texture_2d<f32>;
    @group(0) @binding(3) var textureSampler: sampler;

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
        let content = textureSample(contentTexture, textureSampler, input.texCoord);
        let shadowCoordinate = input.texCoord - uniforms.shadowOffsetUV;
        let shadowMask = textureSample(shadowTexture, textureSampler, shadowCoordinate).r;
        let shadowAlpha = uniforms.shadowColor.a * shadowMask;
        let shadow = vec4<f32>(uniforms.shadowColor.rgb * shadowAlpha, shadowAlpha);
        return content + shadow * (1.0 - content.a);
    }
    """

    // MARK: - Filter Composite Shader

    /// Shader code for filter compositing and color adjustments.
    public static let filterComposite = """
    struct FilterUniforms {
        opacity: f32,
        filterType: f32,
        parameter0: f32,
        parameter1: f32,
        colorMultiplier: vec4<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: FilterUniforms;
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

    fn applyBrightness(color: vec3<f32>, amount: f32) -> vec3<f32> {
        return clamp(color + vec3<f32>(amount), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    fn applyContrast(color: vec3<f32>, amount: f32) -> vec3<f32> {
        return clamp((color - vec3<f32>(0.5)) * amount + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    fn applySaturation(color: vec3<f32>, amount: f32) -> vec3<f32> {
        let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
        let grayscale = vec3<f32>(luminance);
        return clamp(mix(grayscale, color, amount), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    fn applyColorInvert(color: vec3<f32>) -> vec3<f32> {
        return vec3<f32>(1.0) - color;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        var color = textureSample(inputTexture, texSampler, input.texCoord);
        let capturedAlpha = color.a;
        var straightRGB = vec3<f32>(0.0);
        if (capturedAlpha > 0.00001) {
            straightRGB = color.rgb / capturedAlpha;
        }

        if (uniforms.filterType == 5.0) {
            return vec4<f32>(straightRGB, capturedAlpha);
        } else if (uniforms.filterType == 6.0) {
            return vec4<f32>(color.rgb * capturedAlpha, capturedAlpha);
        }

        if (uniforms.filterType == 1.0) {
            straightRGB = applyBrightness(straightRGB, uniforms.parameter0);
        } else if (uniforms.filterType == 2.0) {
            straightRGB = applyContrast(straightRGB, uniforms.parameter0);
        } else if (uniforms.filterType == 3.0) {
            straightRGB = applySaturation(straightRGB, uniforms.parameter0);
        } else if (uniforms.filterType == 4.0) {
            straightRGB = applyColorInvert(straightRGB);
        }

        color.rgb = straightRGB * capturedAlpha;
        color *= vec4<f32>(
            uniforms.colorMultiplier.rgb * uniforms.colorMultiplier.a,
            uniforms.colorMultiplier.a
        );
        color *= uniforms.opacity;
        return color;
    }
    """

    /// Replaces only a transformed layer plane with a viewport-sized composition result.
    ///
    /// The geometry carries the layer's real model-view-projection transform, so WebGPU
    /// performs the same perspective interpolation and depth test as every other plane in
    /// a CATransformLayer. Fragment coordinates select the corresponding pixel from the
    /// full-viewport composition texture.
    public static let transformedComposition = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        viewportSize: vec2<f32>,
        cornerRadii: vec4<f32>,
        samplingBias: f32,
        padding0: f32,
        padding1: f32,
        padding2: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var compositionTexture: texture_2d<f32>;
    @group(0) @binding(2) var textureSampler: sampler;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) texCoord: vec2<f32>,
        @location(2) color: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let safeViewport = max(uniforms.viewportSize, vec2<f32>(1.0));
        let viewportCoordinate = input.position.xy / safeViewport;
        return textureSample(compositionTexture, textureSampler, viewportCoordinate);
    }
    """

    /// Reprojects a viewport-sized composition while rendering into a local capture.
    public static let capturedComposition = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        viewportSize: vec2<f32>,
        cornerRadii: vec4<f32>,
        samplingBias: f32,
        padding0: f32,
        padding1: f32,
        padding2: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var compositionTexture: texture_2d<f32>;
    @group(0) @binding(2) var textureSampler: sampler;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) viewportNumerator: vec2<f32>,
        @location(2) homogeneousCoordinate: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) @interpolate(linear) viewportNumerator: vec2<f32>,
        @location(1) @interpolate(linear) homogeneousW: f32,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.viewportNumerator = input.viewportNumerator;
        output.homogeneousW = input.homogeneousCoordinate.x;
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        if (abs(input.homogeneousW) <= 0.000001) {
            discard;
        }
        let viewportCoordinate = input.viewportNumerator / input.homogeneousW;
        return textureSample(compositionTexture, textureSampler, viewportCoordinate);
    }
    """

    /// Combines a filtered backdrop with its unmodified source through a layer-shape mask.
    public static let backdropFilterMix = """
    @group(0) @binding(0) var originalBackdrop: texture_2d<f32>;
    @group(0) @binding(1) var filteredBackdrop: texture_2d<f32>;
    @group(0) @binding(2) var layerMask: texture_2d<f32>;
    @group(0) @binding(3) var texSampler: sampler;

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
        let original = textureSample(originalBackdrop, texSampler, input.texCoord);
        let filtered = textureSample(filteredBackdrop, texSampler, input.texCoord);
        let coverage = clamp(textureSample(layerMask, texSampler, input.texCoord).a, 0.0, 1.0);
        return mix(original, filtered, coverage);
    }
    """

    /// Intersects coverage masks or applies one to a premultiplied source texture.
    public static let compositionMaskOperation = """
    @group(0) @binding(0) var firstTexture: texture_2d<f32>;
    @group(0) @binding(1) var secondTexture: texture_2d<f32>;
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
    fn intersectMasks(input: VertexOutput) -> @location(0) vec4<f32> {
        let firstCoverage = textureSample(firstTexture, texSampler, input.texCoord).a;
        let secondCoverage = textureSample(secondTexture, texSampler, input.texCoord).a;
        let coverage = clamp(firstCoverage * secondCoverage, 0.0, 1.0);
        return vec4<f32>(coverage);
    }

    @fragment
    fn applyMask(input: VertexOutput) -> @location(0) vec4<f32> {
        let color = textureSample(firstTexture, texSampler, input.texCoord);
        let coverage = clamp(textureSample(secondTexture, texSampler, input.texCoord).a, 0.0, 1.0);
        return color * coverage;
    }
    """

}

#endif
