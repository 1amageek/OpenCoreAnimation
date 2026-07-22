# OpenCoreAnimation

A Swift library providing CoreAnimation (QuartzCore) API compatibility for WebAssembly, powered by WebGPU.

## Overview

OpenCoreAnimation enables CoreAnimation-style code to run in the browser via WebAssembly. Full compatibility is the target; current completion is established by tests of specific API and renderer paths.

## Verified Status

| Evidence | Result |
|---|---|
| Native package | 550 tests passed |
| Browser | 3 checks passed, including float16 extended-dynamic-range color and `CGImage` output with SDR restoration, straight-alpha image normalization and alpha-correct mipmaps with typed malformed/non-finite-storage rejection, contents geometry, rasterized/tiled pixels, frozen transition pairs, filters, multiple shadows, shape fill-rule holes, trimmed/dashed strokes, axial/radial/conic and 12-stop gradients, depth-preserving emitters/replicators, and replicated background/border/shape/image/gradient pixels read back from WebGPU |
| Layer defaults | `CALayer.defaultValue(forKey:)` returns QuartzCore-compatible typed defaults for geometry, contents, appearance, rasterization, and timing keys instead of treating every key as unknown. Shape, gradient, replicator, emitter, text, tiled, and scroll layers override their specialized defaults while inheriting base values. Fresh layers now use opaque-black borders, enabled edge antialiasing, infinite layer duration, and Helvetica text to match QuartzCore; native tests compare stored, zero/unknown, inherited, and instance defaults, while browser readback verifies default edge coverage and black-border rendering |
| Layer archiving | `CALayer.shouldArchiveValue(forKey:)` compares each supported base property with its QuartzCore archive default instead of returning a fixed success value. Shape, gradient, replicator, emitter, text, tiled, and scroll layers own their specialized decisions and defer unknown keys through the class hierarchy. Fresh and changed values are cross-checked against QuartzCore, including collection, delegate, timing, derived, and unknown-key behavior |
| Display invalidation | `CALayer.needsDisplay(forKey:)` preserves the QuartzCore base and non-text `false` contract while `CATextLayer` identifies its ten text/style/scale redraw keys. Text mutations set the public display-invalid state only when a stored value changes, and copy/presentation initialization transfers backing values without manufacturing redraw work |
| Explicit renderer | `CARenderer` is a QuartzCore-compatible class rather than the former backend protocol. `beginFrame`, automatic and explicit update regions, supplied media-time presentation evaluation, `nextFrameTime`, `render`, and `endFrame` execute over the same internal WebGPU/Metal backend contract used by the animation engine. Automatic regions include overflowing descendants plus both old and new extents so moved or removed content cannot leave stale pixels; active animations and effects conservatively invalidate the destination bounds. Native tests compare lifecycle decisions with QuartzCore, verify future/active/paused scheduling and descendant removal, and read an actual submitted Metal texture pixel; the release WASM build includes the canvas-backed initializer and Core Video timestamp stand-ins |
| Unified invalidation | Geometry, contents, shadows, filters, timing, masks, and hierarchy changes share `_dirtyMask` and `_subtreeDirtyCount`. The obsolete shadow/filter subtree counters and their hierarchy-wide propagation were removed because the renderer no longer consumed them and model-only counts could not represent animated presentation effects. Existing dirty-propagation tests cover clean-to-dirty idempotence, reparenting, ordering, masks, presentation isolation, and commit clearing |
| Native engine verification | The native `CAAnimationEngine` uses the real offscreen `CAMetalRenderer`, lazily creates a destination from the root bounds, submits a command buffer, records synchronous failures explicitly, and clears dirty state only after submission. The former no-op native renderer was removed; a native test reads the submitted green pixel from the engine-owned Metal texture |
| Animation and emitter-cell defaults | `CAAnimation`, every concrete animation subclass, and `CAEmitterCell` expose QuartzCore-compatible defaults and `shouldArchiveValue(forKey:)` decisions for persistent state. Runtime-only frame-rate hints and spring initial velocity remain intentionally unarchived, while unknown keys fail closed. `CAEmitterCell` includes `style`, defaults to white, enabled, and infinite duration, and distinguishes positive infinity from invalid non-finite timing. Native tests cover typed values, subclass inheritance, every archive key, canonical/unknown keys, style storage, indefinite emission, infinite repeats, and typed timing failure; browser emitter diagnostics exercise the default infinite duration with zero spawn failures |
| Shape fills | `CAShapeLayer` tessellates all path contours as one fill, preserving `.nonZero` winding and `.evenOdd` parity for nested, overlapping, coincident, curved, open, and self-intersecting subpaths. Unknown rules and non-finite paths fail through renderer diagnostics instead of producing fallback geometry. Shape draws select their own solid/stencil/depth pipeline so output does not depend on the preceding sibling type. Native geometry tests and browser pixel readback cover both fill rules, holes, and submitted draw/vertex counts |
| Shape strokes | `strokeStart` and `strokeEnd` trim against total length across all subpaths, then the shared OpenCoreGraphics geometry path applies `lineDashPattern`, phase continuity, line caps, line joins, and miter limits before the outline enters the same WebGPU tessellator as fills. Raw animation values remain unclamped in the model and clamp only at rendering. Invalid geometry, dash patterns, and unknown styles report explicit failures. Native tests cover trim, multi-subpath ranges, dash/phase, cap styles, and failure contracts; browser pixels prove trimmed alternating dash segments reach WebGPU |
| Gradients | `CAGradientLayer.type` selects axial projection, unit-coordinate elliptical radial distance, or conic angular progression around `startPoint` with the `startPoint`→`endPoint` ray as zero. Colors and locations are validated as one configuration; unknown types, non-finite geometry, non-color values, and invalid location sequences are rejected through renderer diagnostics instead of being replaced. Gradient stops use dynamically growing, triple-buffered read-only GPU storage rather than a fixed uniform array. Native tests cover geometry, unbounded valid stop lists, and every input-failure class; browser pixels verify all three modes, a 12-stop gradient that crosses the former eight-stop boundary, and an unsupported type through WebGPU readback |
| Corner curves | `.circular` and `.continuous` use distinct calibrated corner geometry in the native path and matching Lp signed-distance fields in WebGPU. The curve exponent is carried through solid fills, borders, gradients, textured contents, rounded stencil clips, content masks, composition clips, and shadow cache identity. Unknown raw values are rejected through renderer diagnostics instead of falling back to circular corners or dropping a requested mask. Native boundary tests distinguish both shapes; browser readback verifies solid, texture, gradient, `masksToBounds`, and `mask` pixels plus the explicit failure path |
| Boolean and discrete animations | `hidden`, `masksToBounds`, `doubleSided`, and `shouldRasterize` participate in basic and keyframe presentation evaluation. Discrete keyframes select the latest reached value, single-value sequences apply directly, scalar position components animate independently, and values hold at the final key time. Browser pixels verify visibility, clipping, backface rendering, and explicit rasterization capture |
| Contents animations | `contents` uses QuartzCore-compatible midpoint selection for basic and interpolated keyframe segments, while discrete keyframes hold the latest reached image. Presentation copies use backing storage so evaluation does not enqueue model transactions. Browser readback verifies red, green, and blue `CGImage` selection at the boundary times |
| Contents geometry | Single-quad and nine-slice image rendering validate image, bounds, crop, scale, and gravity through one typed configuration. `contentsRect` controls both UVs and the logical size used by gravity before `contentsCenter` subdivides the selected image; `contentsScale` converts fixed source pixels to points. Nine-slice is restricted to resizing gravity modes, unknown gravity and invalid geometry increment explicit renderer failures, and pipeline selection cannot fall through to the previous draw state. Native tests cover crop/center/scale/gravity decisions while browser readback verifies cropped blue/magenta/yellow slices, centered non-sliced contents, and invalid-center rejection |
| Rasterization scale animations | `rasterizationScale` mutations resolve layer actions and basic/keyframe presentation values drive the explicit offscreen capture dimensions. Browser diagnostics verify a 40-point layer animated from scale 1 to 2 captures at 60×60 pixels at half progress |
| Shadow path keyframes | `shadowPath` accepts single-value and discrete keyframes and applies compatible linear/cubic path morphs to the presentation layer. Browser shadow pixels verify that discrete and cubic path selection reaches the explicit-path tessellation renderer |
| Transform component animations | `transform.rotation[.x/.y/.z]`, `transform.scale[.x/.y/.z]`, and `transform.translation[.x/.y/.z]` read and replace decomposed transform components instead of multiplying the model matrix. Basic and keyframe animations share linear, discrete, cubic, additive, and cumulative evaluation. Browser pixels verify that translation, scale, and rotation presentation transforms reach WebGPU vertex rendering |
| Basic aggregate animations | `CABasicAnimation` applies the complete from/to/by endpoint contract to rectangles, RGBA colors, gradient color/location arrays, and full layer, sublayer, and replicator transforms. `toValue + byValue` uses verified inverse transform resolution; singular transforms and incompatible arrays leave the presentation value unchanged instead of being reinterpreted or partially truncated. Additive colors start from transparent black, explicit color and gradient-array endpoints work without model values, and presentation evaluation writes backing storage without registering model transactions. Native tests cover every endpoint family and failure path; browser pixels verify bounds expansion, color addition, and full-transform translation reach WebGPU rendering |
| Additive keyframes | Linear, paced, discrete, cubic, and single-value keyframes share one typed interpolation-to-application path. Scalar, geometry, RGBA, gradient-array, full-transform, and specialized-layer values add to presentation state instead of replacing it; `CFTimeInterval` and `CGFloat` scalar inputs normalize at the API boundary. Cumulative gradient colors and locations carry complete terminal arrays across repeats, while incompatible array lengths or elements leave the complete presentation array unchanged. Native tests cover base, shape, text, emitter, gradient, and replicator layers; browser pixels verify additive position, bounds, and color reach WebGPU rendering |
| Animation graphs | Root and nested `CAAnimationGroup` trees are evaluated in graph-wide non-additive and additive passes, so child-array or top-level dictionary order cannot overwrite additive contributions. Group timing functions remap child basic time and compose with each child's own timing function. Grouped transitions receive a recursively captured source tree and evaluate in the group's repeating basic time. Lifecycle callbacks belong to the group attached to the layer; nested groups and child animations are evaluated without emitting partial start-only delegate events. Native tests cover direct, nested, and root-crossing ordering, hierarchical pacing, grouped transition state, natural completion, explicit removal, retention, and transaction completion; browser pixels exercise a deliberately additive-first mixed group through presentation and WebGPU rendering |
| Constraint layout | `CAConstraintLayoutManager` solves each independent system of sibling and superlayer equations simultaneously, preserving unconstrained geometry and isolating inconsistent components instead of depending on sublayer or constraint order. Bounds, constraint, layout-manager, and hierarchy mutations invalidate layout, notify the manager once per clean-to-dirty transition, and converge repeated invalidations before `CAAnimationEngine` renders parent-to-child. Native tests cover nonzero bounds origins, coupled edge sizing, reversed sibling chains, missing sources, conflicts, notification/reentry, and automatic render-time layout; browser pixels verify the solved sibling frames reach WebGPU rendering without a manual layout call |
| Value-function animations | `CAValueFunction` rejects unknown names and validates input arity instead of producing an identity fallback. Scalar functions and the three-component `.scale` / `.translate` functions participate in basic and keyframe linear, discrete, cubic, paced, additive, and cumulative evaluation. Non-additive animations replace the model transform, while additive animations concatenate with it. Native tests cover integer and floating-point inputs, and browser pixels verify aggregate keyframe translation reaches WebGPU rendering |
| Edge antialiasing | `allowsEdgeAntialiasing` and each `CAEdgeAntialiasingMask` bit drive derivative-based WebGPU coverage in layer-local coordinates. Solid, border, gradient, shape, image, text, nine-slice, and exterior tiled edges share the contract; internal tile seams remain untouched and captured layers are not antialiased again during final composition. Browser readback verifies disabled, left-only, right-only, runtime mask mutation, and textured-content pixels |
| Filters and transitions | Sibling and nested `CAFilter` chains use per-layer WebGPU resources with browser pixel and cleanup evidence. Built-in fades interpolate frozen premultiplied source/target RGBA in one fragment pass, preserving translucent and transparent pixels under stencil and true-3D depth. All 7 executable OpenCoreImage transition pipelines also have filter-specific browser pixel evidence, while unsupported transition filter objects, unknown built-in transition types, and unknown directional subtypes are rejected without target-image or `.fromLeft` fallback |
| Shadows | Every visible shadow owns an independent mask, blur target, and uniform set; a nil `shadowPath` derives its silhouette from the rendered subtree alpha, while an explicit path uses the same complete-path non-zero tessellation as shape fills. Silhouettes are captured at the content position and `shadowOffset` is applied once during display for raw, filtered, masked, and explicit-path inputs. Rendered mask-tree alpha, including filtered descendants, active transitions, and partial coverage, shapes the shadow; detached mask mutations invalidate the cached silhouette. Browser evidence covers transparent image pixels, sublayer-only content, empty content, multiple shadows, inherited opacity, animated `shadowOpacity` from a zero model value, ancestor transform invalidation, mask transition and mutation, empty and holed `shadowPath` values, and resource cleanup |
| Emitters | Particle simulation state, fractional birth accumulation, random state, and cleanup are isolated per model `CAEmitterLayer`. All documented shape/mode combinations and uniform 3D emission cones honor emitter geometry, latitude, longitude, and range. Every render mode is active, including z-sorted back-to-front rendering, source-additive compositing, and stencil-aware masked particles. `preservesDepth` selects direct particle depth writes inside 3D hierarchies; the default path captures particles as one plane, and emitter-containing rasterization captures are refreshed rather than freezing simulation behind a clean dirty mask. `CGImage` cells snapshot `contentsRect`, `contentsScale`, tint, magnification/minification filters, and mip bias at birth; nil contents remain simulated but invisible, while unsupported content is rejected. Nested child cells honor their media-timing window, emit from the parent's moving position, orient relative to its current direction, inherit its animated color and scale, and support later generations. Browser evidence covers these child-cell semantics plus image cropping/scaling, linear versus trilinear pixels, ordering, additive pixel readback, depth-path selection, concurrent low-rate emitters, rectangle-outline and sphere-surface geometry, orthogonal 3D velocities, and independent removal |
| Replicators | Instances traverse the normal layer renderer, with cumulative transforms, color multiplication/offsets, nested inherited state, `instanceDelay` animation evaluation, and a true zero-instance result. With `preservesDepth`, replicated descendants share a depth group, draw far-to-near for translucent blending, and use per-pixel depth tests instead of flattening in instance order. Filter, shadow, and rasterization resources are isolated by model layer plus nested replicator path. Browser pixels cover background, border, shape, image, gradient, delayed opacity/color, all three offscreen paths, and the visible flat-versus-depth occlusion difference |
| Group opacity | Translucent layers with `allowsGroupOpacity` capture their complete subtree and apply opacity once during premultiplied-alpha composite; disabling the property preserves per-component opacity. Browser pixels verify overlapping opaque and translucent children in both modes |
| Layer filters | `CAFilter` stages and executable `CIFilter` objects run in declared mixed order against the captured subtree. A typed execution plan validates parameter names, numeric types, finite values, and documented ranges before dispatch, and materializes sepia/vignette defaults explicitly at the Core Image boundary. Explicit straight/premultiplied-alpha conversions preserve translucent edges there. Invalid configurations and incompatible multi-input filters are rejected through renderer diagnostics and never silently render an unfiltered layer |
| Compositing filters | Executable two-input `CIFilter` operations receive a draw-order-accurate backdrop captured immediately before the layer. Directly inherited opacity and replicator color scale only the premultiplied source before that operation; the backdrop is not faded. Group-opacity and filtered ancestors create transparent local scopes, apply their opacity/filter once after child composition, and then rejoin their parent backdrop. Replicator instances own distinct backdrop resources and preserve instance transform, color offsets, and ordering. Nested composition is evaluated deepest-first; each parent source is recaptured after its children complete, preventing global backdrop leakage and double composition. Ancestor transform/effect flattening and explicit `shouldRasterize` captures are deferred until their descendant compositions exist, then recaptured without stale backdrop-cache reuse. Local captures reproject viewport composition coordinates through homogeneous interpolation, while true-3D display planes sample framebuffer coordinates and retain depth writes. Consecutive operations feed the preceding composite into the next backdrop, later siblings remain on top, and the cumulative snapshot replaces rather than re-blends the framebuffer. Transformed, rounded, nested `masksToBounds` shapes and rendered `CALayer.mask` trees are rasterized independently and intersected as full-viewport coverage masks for both source composition and backdrop filtering. Detached mask trees own transparent backdrop roots, recursively resolve nested `compositingFilter` and `backgroundFilters` targets before the main tree, and retain all sibling-context resources until command submission. Mask-root `filters` execute mixed `CAFilter`/`CIFilter` stages with explicit alpha conversion before becoming coverage. `backgroundFilters` execute the same mixed stages against the layer bounds when clipped and the parent/full backdrop extent when unclipped. Failed or unprepared paths never fall through to unprocessed source-over. Browser pixels cover multiply, screen, source-in mask alpha, mask-local backdrop blur, direct/group source opacity, translucent backdrop replacement, mixed background/mask-filter stages, nested rounded and content-mask clipping, replicated color/transform, nested filtered and rasterized scopes, projective reprojection, ordered chaining, and resource eviction |
| Display link | Fresh timing values match QuartzCore's zero state. After delivery, `duration` estimates the maximum physical refresh cadence while `targetTimestamp - timestamp` reports the selected callback interval, including factor-based 30 fps delivery on a faster display. Non-finite frame-rate hints cannot create invalid timers, pause/resume is terminally separated from mode registration, and independent modes retain or stop delivery correctly. Native run-loop callbacks and browser rAF verify monotonic timing, throttling, pause/resume, removal, invalidation, and target release |
| Animation frame-rate hints | `CAAnimation.preferredFrameRateRange` is preserved by defensive animation copies and arbitrated across active animations, nested groups, and the complete layer tree. Future and completed animations do not affect the current request; the highest-demand active range is submitted to `CADisplayLink`, while the engine baseline is restored when no explicit hint is active |
| Dynamic range | `CALayer.ToneMapMode`, `CALayer.DynamicRange`, `toneMapMode`, `preferredDynamicRange`, and `contentsHeadroom` match current QuartzCore names, raw values, defaults, and copy/presentation behavior. The WebGPU renderer uses an `rgba16float` canvas, validates the complete visible layer tree, switches canvas tone mapping between standard and extended modes, and reports typed failures for invalid policy, invalid headroom, or unavailable explicit HDR output. Float, HDR-tagged, and extended-space `CGImage` contents are converted to straight-alpha extended-linear RGBA16Float without the former RGBA8 quantization; SDR images stay RGBA8. Both formats receive alpha-correct mip chains and format-aware cache accounting. Browser GPU readback proves that `(2.0, 0.5, 0.25, 1.0)` survives both color and image texture paths, an invalid `0.5` headroom rejects the frame, and removing HDR content restores standard output. Rasterization budgets account for the eight-byte pixel format |
| Delegate drawing | Ordinary `CALayer` instances consume `setNeedsDisplay()`, run `display(_:)` before `layerWillDraw(_:)` / `draw(_:in:)`, rasterize the draw callback into a `contentsScale`-aware software backing store, and feed that image through the normal WebGPU contents, filter, shadow, and rasterization paths. Rectangular invalidations union until display, preserve pixels outside the clipped update region, clear stale pixels inside it, and retain invalidations raised reentrantly by `display(_:)`. Full redraw replaces the backing store, detached layers release it, and invalid/non-finite extents are rejected through renderer diagnostics. Browser pixels verify partial preservation, full redraw, and both Y-up and `isGeometryFlipped` context orientation |
| Text layout | `CATextLayer` shares one Canvas-measured layout path between sizing and rendering. Width wrapping preserves Latin separators without inventing spaces between CJK characters, explicit LF/CRLF/CR paragraph breaks render even when `isWrapped` is false, oversized tokens remain grapheme-safe, and `.justified` distributes Latin words or CJK characters while leaving each paragraph's final line unchanged. `.start`, `.middle`, and `.end` truncation retain extended grapheme clusters; unknown modes do not silently fall back. Browser pixels verify truncation placement, wrapped overflow, mode-change cache invalidation, explicit multiline rendering, and first-line justification |
| Cubic keyframes | `.cubic` and `.cubicPaced` use Kochanek-Bartels tangents with per-control-point tension, continuity, and bias defaults. First and last segment control points are extrapolated from adjacent differences, matching QuartzCore endpoint slopes instead of halving them through duplicated endpoints. Scalar, point, size, rectangle, color, transform, gradient-array, and compatible-path values use the same cubic contract without linear fallback; runtime tests assert interior and endpoint presentation values plus parameter-sensitive paced arc lengths, and browser pixels verify the endpoint position reaches WebGPU rendering |
| Spring animations | `CASpringAnimation` exposes the current QuartzCore perceptual API (`init(perceptualDuration:bounce:)`, `perceptualDuration`, `bounce`, and `allowsOverdamping`) and derives its physical coefficients from the same damping-ratio mapping. The response evaluator distinguishes critical and explicitly enabled overdamped motion without rewriting stored coefficients. Settling estimates replace the former four-time-constant approximation and are checked against measured QuartzCore underdamped, critical, initial-velocity, and infinite-damping boundaries |
| Geometry flipping | `isGeometryFlipped` reflects descendant geometry around the layer bounds while leaving the layer's own contents plane unchanged. Coordinate conversion, inverse hit testing, ordinary traversal, transform/depth groups, filters, masks, shadows, and rasterization use the same flipped parent matrix. Native tests match QuartzCore values for nonzero bounds origins, arbitrary anchors, and rotation; browser readback verifies that asymmetric child layers exchange vertical positions after a runtime flip |
| Hit testing | Normal layer trees traverse stable `zPosition` order. Coordinate conversion and picking compose each hierarchy's full 4×4 transform, including parent `sublayerTransform`, then project and invert the resulting plane homography so perspective and out-of-plane rotations remain aligned with rendered pixels. A `nil` conversion endpoint uses the receiver's superlayer coordinate space for points and all four rectangle corners, including anchor, transform, and geometry-flip effects. Singular projections return non-finite coordinates instead of silently using an unrelated affine result. `CATransformLayer` rejects 2D hit testing with `nil`, because its true-3D hierarchy has no single 2D coordinate space; callers must provide scene-specific 3D picking |
| Transform layers | `CATransformLayer` preserves `zPosition` and `anchorPointZ`, orders blended children by projected center depth, and enables per-pixel WebGPU depth writes/tests only within the true-3D hierarchy. Independent transform groups reset depth without changing color; transparent texels discard before depth write. A normal `CALayer` child subtree is captured with transparent clear and composited as one plane, while a nested `CATransformLayer` keeps the shared depth space. Capture extents recursively union transformed out-of-bounds descendants unless an ancestor clips with `masksToBounds`; oversized logical extents reduce resolution to WebGPU limits instead of cropping geometry. Layer-sized effect captures preserve rendered mask trees, partial mask alpha, mask-descendant filters, group opacity, root filters, and nested filters without falling back to viewport-sized composites. Premultiplied capture composition applies mask alpha once, and detached mask-tree revisions invalidate explicit raster caches after mutation. Shadow captures expand beyond layer bounds for offsets and blur, derive silhouettes from subtree alpha or `shadowPath`, and composite behind content on the same 3D plane. Browser readback verifies intersecting planes, transparent cutouts, group isolation, flattening, nested 3D, overflow clipping, effects, masks, shadows, mutation, and cache reuse |
| Remaining boundary | Core Image transition types without executable WGSL remain unavailable. Complete QuartzCore parity is not claimed |

```swift
#if canImport(QuartzCore)
import QuartzCore      // Native platforms (iOS, macOS)
#else
import OpenCoreAnimation  // WASM/Web
#endif

// Same API works in both environments
let layer = CALayer()
layer.frame = CGRect(x: 0, y: 0, width: 100, height: 100)
layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
layer.cornerRadius = 10
```

## Features

### Layers

| Layer Type | Description |
|------------|-------------|
| `CALayer` | Base layer rendering for opacity, transform, shadow, border, mask, and filters |
| `CAShapeLayer` | Vector path rendering with strokes and complete-path non-zero/even-odd fills |
| `CAGradientLayer` | Axial, elliptical radial, and conic gradients with validated stops |
| `CATextLayer` | Text rendering with font and alignment |
| `CAReplicatorLayer` | Instance replication with cumulative transforms, color offsets, and animation delay |
| `CAScrollLayer` | Scrollable content with bounds.origin-based scrolling |
| `CATransformLayer` | 3D transform container (does not flatten sublayers) |
| `CATiledLayer` | Tiled content rendering with level-of-detail support |
| `CAEmitterLayer` | Particle systems with physics simulation |

### Animations

| Animation Type | Description |
|----------------|-------------|
| `CABasicAnimation` | Simple from/to/by animations |
| `CAKeyframeAnimation` | Multi-keyframe animations with path support |
| `CASpringAnimation` | Physics-based spring animations |
| `CAAnimationGroup` | Animation composition |
| `CATransition` | Layer transitions (fade, push, moveIn, reveal) |
| `CAValueFunction` | Transform animations from scalar values |

### Animation Features

- **Timing Functions**: Linear, ease-in, ease-out, ease-in-ease-out, custom cubic bezier
- **Path Animations**: Animate position along CGPath with auto-rotation
- **Value Functions**: Animate transforms using scalar values (rotateX/Y/Z, scaleX/Y/Z, translateX/Y/Z)
- **Spring Physics**: Configurable mass, stiffness, damping, and initial velocity

### Layer Properties

- **Visual**: backgroundColor, borderColor, borderWidth, cornerRadius, opacity, isHidden
- **Shadow**: shadowColor, shadowOpacity, shadowOffset, shadowRadius
- **Transform**: transform, sublayerTransform, anchorPoint, anchorPointZ, zPosition
- **Content**: contents (CGImage), contentsGravity, contentsRect, contentsCenter, contentsScale
- **Masking**: mask, masksToBounds, cornerRadius with maskedCorners
- **Filters**: CAFilter for blur and color effects

### Filters (CAFilter)

```swift
// Apply blur filter to layer
layer.filters = [CAFilter.blur(radius: 10)]

// Color adjustments
layer.filters = [
    CAFilter.brightness(0.2),
    CAFilter.contrast(1.5),
    CAFilter.saturation(0.8)
```

Supported filter types:
- `gaussianBlur` - Gaussian blur effect
- `brightness` - Brightness adjustment (-1 to 1)
- `contrast` - Contrast adjustment (0 to 4)
- `saturation` - Saturation adjustment (0 to 2)
- `colorInvert` - Color inversion
- `sepiaTone` - Sepia tone effect
- `vignette` - Vignette effect

### Core Components

- `CATransaction` - Implicit animation batching with nested transaction support
- `CADisplayLink` - Frame-synchronized callbacks using requestAnimationFrame
- `CATransform3D` - Full 3D transformation matrix support
- `CAMediaTiming` - Animation timing protocol

## Installation

Add OpenCoreAnimation to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/OpenCoreAnimation.git", branch: "main")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["OpenCoreAnimation"]
)
```

### Dependencies

- [OpenCoreGraphics](https://github.com/1amageek/OpenCoreGraphics) - CoreGraphics types for WASM
- [swift-webgpu](https://github.com/1amageek/swift-webgpu) - WebGPU bindings (WASM only)

## Usage

### Basic Layer Setup

```swift
import OpenCoreAnimation

// Create a layer hierarchy
let rootLayer = CALayer()
rootLayer.bounds = CGRect(x: 0, y: 0, width: 800, height: 600)
rootLayer.position = CGPoint(x: 400, y: 300)
rootLayer.backgroundColor = CGColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1.0)

// Add a sublayer
let boxLayer = CALayer()
boxLayer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
boxLayer.position = CGPoint(x: 150, y: 150)
boxLayer.backgroundColor = CGColor(red: 0.2, green: 0.6, blue: 1.0, alpha: 1.0)
boxLayer.cornerRadius = 10
boxLayer.shadowOpacity = 0.5
boxLayer.shadowRadius = 10
boxLayer.shadowOffset = CGSize(width: 5, height: 5)
rootLayer.addSublayer(boxLayer)
```

### Image Content with Gravity

```swift
let imageLayer = CALayer()
imageLayer.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)
imageLayer.contents = myCGImage

// Content positioning
imageLayer.contentsGravity = .resizeAspectFill
imageLayer.contentsRect = CGRect(x: 0, y: 0, width: 1, height: 1)  // Full image
imageLayer.contentsScale = 2.0  // Retina

// 9-patch scaling for stretchable images
imageLayer.contentsCenter = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
```

### Animations

```swift
// Basic animation
let positionAnim = CABasicAnimation(keyPath: "position")
positionAnim.fromValue = CGPoint(x: 100, y: 100)
positionAnim.toValue = CGPoint(x: 300, y: 300)
positionAnim.duration = 1.0
positionAnim.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
layer.add(positionAnim, forKey: "move")

// Spring animation
let springAnim = CASpringAnimation(keyPath: "transform.scale")
springAnim.fromValue = 1.0
springAnim.toValue = 1.5
springAnim.damping = 10
springAnim.stiffness = 100
springAnim.mass = 1
layer.add(springAnim, forKey: "spring")

// Keyframe animation with values
let keyframeAnim = CAKeyframeAnimation(keyPath: "position")
keyframeAnim.values = [
    CGPoint(x: 0, y: 0),
    CGPoint(x: 100, y: 50),
    CGPoint(x: 200, y: 0)
]
keyframeAnim.keyTimes = [0, 0.5, 1.0]
keyframeAnim.duration = 2.0
layer.add(keyframeAnim, forKey: "path")

// Path-based animation with auto-rotation
let pathAnim = CAKeyframeAnimation(keyPath: "position")
let path = CGMutablePath()
path.move(to: CGPoint(x: 50, y: 50))
path.addCurve(to: CGPoint(x: 350, y: 350),
              control1: CGPoint(x: 200, y: 50),
              control2: CGPoint(x: 200, y: 350))
pathAnim.path = path
pathAnim.rotationMode = .rotateAuto  // Rotate along path tangent
pathAnim.duration = 3.0
layer.add(pathAnim, forKey: "followPath")

// Value function animation (rotate around Y axis)
let rotateAnim = CABasicAnimation(keyPath: "transform")
rotateAnim.valueFunction = CAValueFunction(name: .rotateY)
rotateAnim.fromValue = 0
rotateAnim.toValue = CGFloat.pi * 2
rotateAnim.duration = 2.0
layer.add(rotateAnim, forKey: "rotate")

// Transition
let transition = CATransition()
transition.type = .push
transition.subtype = .fromRight
transition.duration = 0.5
layer.add(transition, forKey: "transition")
```

### Shape Layers

```swift
let shapeLayer = CAShapeLayer()
shapeLayer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)

// Create a path
let path = CGMutablePath()
path.addEllipse(in: CGRect(x: 0, y: 0, width: 100, height: 100))

shapeLayer.path = path
shapeLayer.fillColor = CGColor(red: 1, green: 0.5, blue: 0, alpha: 1)
shapeLayer.strokeColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
shapeLayer.lineWidth = 2
shapeLayer.strokeStart = 0
shapeLayer.strokeEnd = 1  // Animate this for drawing effect
```

### Gradient Layers

```swift
let gradientLayer = CAGradientLayer()
gradientLayer.bounds = CGRect(x: 0, y: 0, width: 150, height: 150)
gradientLayer.colors = [
    CGColor(red: 1.0, green: 0.4, blue: 0.4, alpha: 1.0),
    CGColor(red: 0.4, green: 1.0, blue: 0.4, alpha: 1.0)
]
gradientLayer.startPoint = CGPoint(x: 0, y: 0)
gradientLayer.endPoint = CGPoint(x: 1, y: 1)
gradientLayer.locations = [0.0, 1.0]
```

### Replicator Layers

```swift
let replicator = CAReplicatorLayer()
replicator.instanceCount = 10
replicator.instanceDelay = 0.1
replicator.instanceTransform = CATransform3DMakeRotation(.pi / 5, 0, 0, 1)
replicator.instanceRedOffset = -0.1
replicator.instanceGreenOffset = -0.1
replicator.instanceBlueOffset = 0.0
replicator.instanceAlphaOffset = -0.1

let dot = CALayer()
dot.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
dot.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
dot.cornerRadius = 10
replicator.addSublayer(dot)
```

### WebGPU Rendering (WASM)

```swift
import JavaScriptKit
import OpenCoreAnimation

@main
struct MyApp {
    static func main() async throws {
        let document = JSObject.global.document
        let canvas = document.createElement("canvas")
        canvas.width = 800
        canvas.height = 600
        _ = document.body.appendChild(canvas)

        // Create layer hierarchy
        let rootLayer = CALayer()
        rootLayer.bounds = CGRect(x: 0, y: 0, width: 800, height: 600)
        rootLayer.position = CGPoint(x: 400, y: 300)
        // ... configure layers ...

        // Initialize and start animation engine
        let engine = CAAnimationEngine.shared
        try await engine.setCanvas(canvas.object!)
        engine.rootLayer = rootLayer
        engine.start()
    }
}
```

## Building

### Native (for testing)

```bash
swift build
perl -e 'alarm 30; exec @ARGV' -- \
  xcodebuild test -scheme OpenCoreAnimation -destination 'platform=macOS' \
  -only-testing:OpenCoreAnimationTests
```

### WASM

```bash
swiftly run swift build --swift-sdk swift-6.3.1-RELEASE_wasm
cd Tests/e2e && npm test
```

## Platform Strategy

| Platform | Rendering | Timing | Usage |
|----------|-----------|--------|-------|
| WASM/Web | WebGPU | `requestAnimationFrame` | **Production** |
| macOS/iOS | Metal offscreen verification | `Timer` | Testing only |

On native Apple platforms, use Apple's QuartzCore directly for production. OpenCoreAnimation's native implementations are for testing purposes only.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OpenCoreAnimation API                       │
│ (CALayer, CAAnimation, CADisplayLink, CARenderer - API)     │
├─────────────────────────────────────────────────────────────┤
│                  WebGPU Rendering Layer                      │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ CAWebGPURenderer│  │ CAAnimationEngine│                  │
│  │ (Layer drawing) │  │ (Timing/frames)  │                  │
│  └─────────────────┘  └──────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                     swift-webgpu                             │
│              (Type-safe WebGPU bindings)                     │
├─────────────────────────────────────────────────────────────┤
│                     JavaScriptKit                            │
│              (Swift-to-JavaScript bridge)                    │
├─────────────────────────────────────────────────────────────┤
│                   Browser WebGPU API                         │
└─────────────────────────────────────────────────────────────┘
```

## Rendering Features

The WebGPU renderer provides full support for:

- **Layer Hierarchy**: Proper sublayer rendering with transform inheritance
- **3D Transforms**: Full CATransform3D support with perspective
- **Shadows**: Gaussian blur shadows with configurable radius, offset, and color
- **Masks**: Layer masking via the `mask` property
- **Clipping**: `masksToBounds` and corner radius clipping
- **Edge Antialiasing**: Independently selectable left, right, bottom, and top edge coverage
- **Image Content**: CGImage rendering with contentsGravity, contentsRect, and contentsCenter
- **9-Patch Scaling**: Stretchable images via contentsCenter
- **Filters**: Blur and color adjustment filters
- **Blend Modes**: Standard alpha blending
- **Depth Testing**: Proper z-ordering for 3D scenes

## Examples

See the [Examples](Examples/) directory for complete working demos:

- **BasicAnimation** - Layer hierarchy, animations, gradients, and shapes

## Requirements

- Swift 6.0+
- For WASM: Browser with WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly)

## License

MIT License

## References

- [Core Animation Documentation](https://developer.apple.com/documentation/quartzcore)
- [Core Animation Programming Guide](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CoreAnimation_guide/Introduction/Introduction.html)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
