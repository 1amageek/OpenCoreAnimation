# OpenCoreAnimation

A Swift library providing CoreAnimation (QuartzCore) API compatibility for WebAssembly, powered by WebGPU.

## Overview

OpenCoreAnimation enables CoreAnimation-style code to run in the browser via WebAssembly. Full compatibility is the target; current completion is established by tests of specific API and renderer paths.

## Verified Status

| Evidence | Result |
|---|---|
| Native package | 365 tests passed |
| Browser | 3 checks passed, including rasterized/tiled pixels, frozen transition pairs, filters, multiple shadows, and replicated background/border/shape/image/gradient pixels read back from WebGPU |
| Filters | Sibling and nested `CAFilter` chains use per-layer WebGPU resources with browser pixel and cleanup evidence; all 7 executable OpenCoreImage transition pipelines also have filter-specific browser pixel evidence, while unsupported transition filter objects, unknown built-in transition types, and unknown directional subtypes are rejected without target-image or `.fromLeft` fallback |
| Shadows | Every visible shadow owns an independent mask, blur target, and uniform set; a nil `shadowPath` derives its silhouette from the rendered subtree alpha, while an explicit path uses direct tessellation. Silhouettes are captured at the content position and `shadowOffset` is applied once during display for raw, filtered, masked, and explicit-path inputs. Rendered mask-tree alpha, including filtered descendants, active transitions, and partial coverage, shapes the shadow; detached mask mutations invalidate the cached silhouette. Browser evidence covers transparent image pixels, sublayer-only content, empty content, multiple shadows, inherited opacity, animated `shadowOpacity` from a zero model value, ancestor transform invalidation, mask transition and mutation, empty `shadowPath`, and resource cleanup |
| Emitters | Particle simulation state, fractional birth accumulation, random state, and cleanup are isolated per model `CAEmitterLayer`. All documented shape/mode combinations and uniform 3D emission cones honor emitter geometry, latitude, longitude, and range. Every render mode is active, including z-sorted back-to-front rendering, source-additive compositing, and stencil-aware masked particles. `CGImage` cells snapshot `contentsRect`, `contentsScale`, tint, magnification/minification filters, and mip bias at birth; nil contents remain simulated but invisible, while unsupported content is rejected. Nested child cells honor their media-timing window, emit from the parent's moving position, orient relative to its current direction, inherit its animated color and scale, and support later generations. Browser evidence covers these child-cell semantics plus image cropping/scaling, linear versus trilinear pixels, ordering, additive pixel readback, concurrent low-rate emitters, rectangle-outline and sphere-surface geometry, orthogonal 3D velocities, and independent removal |
| Replicators | Instances traverse the normal layer renderer, with cumulative transforms, color multiplication/offsets, nested inherited state, `instanceDelay` animation evaluation, and a true zero-instance result. Filter, shadow, and rasterization resources are isolated by model layer plus nested replicator path. Browser pixels cover background, border, shape, image, gradient, delayed opacity/color, and all three offscreen paths |
| Group opacity | Translucent layers with `allowsGroupOpacity` capture their complete subtree and apply opacity once during premultiplied-alpha composite; disabling the property preserves per-component opacity. Browser pixels verify overlapping opaque and translucent children in both modes |
| Layer filters | `CAFilter` stages and executable `CIFilter` objects run in declared mixed order against the captured subtree. Explicit straight/premultiplied-alpha conversions preserve translucent edges at the Core Image boundary. Incompatible multi-input filters are rejected through renderer diagnostics and never silently render an unfiltered layer |
| Compositing filters | Executable two-input `CIFilter` operations receive a draw-order-accurate backdrop captured immediately before the layer. Directly inherited opacity and replicator color scale only the premultiplied source before that operation; the backdrop is not faded. Group-opacity and filtered ancestors create transparent local scopes, apply their opacity/filter once after child composition, and then rejoin their parent backdrop. Replicator instances own distinct backdrop resources and preserve instance transform, color offsets, and ordering. Nested composition is evaluated deepest-first; each parent source is recaptured after its children complete, preventing global backdrop leakage and double composition. Ancestor transform/effect flattening and explicit `shouldRasterize` captures are deferred until their descendant compositions exist, then recaptured without stale backdrop-cache reuse. Local captures reproject viewport composition coordinates through homogeneous interpolation, while true-3D display planes sample framebuffer coordinates and retain depth writes. Consecutive operations feed the preceding composite into the next backdrop, later siblings remain on top, and the cumulative snapshot replaces rather than re-blends the framebuffer. Transformed, rounded, nested `masksToBounds` shapes and rendered `CALayer.mask` trees are rasterized independently and intersected as full-viewport coverage masks for both source composition and backdrop filtering. Mask-root `filters` execute mixed `CAFilter`/`CIFilter` stages with explicit alpha conversion before becoming coverage. `backgroundFilters` execute the same mixed stages against the layer bounds when clipped and the parent/full backdrop extent when unclipped. Failed paths never fall through to unprocessed source-over. Browser pixels cover multiply, screen, direct/group source opacity, translucent backdrop replacement, mixed background/mask-filter stages, nested rounded and content-mask clipping, replicated color/transform, nested filtered and rasterized scopes, projective reprojection, ordered chaining, and resource eviction |
| Display link | `duration` tracks display refresh cadence independently from preferred callback frequency; run-loop mode registrations are removed independently and verified through browser rAF delivery |
| Cubic keyframes | `.cubic` and `.cubicPaced` use Kochanek-Bartels tangents with per-control-point tension, continuity, and bias defaults. Scalar, point, size, rectangle, color, transform, gradient-array, and compatible-path values use the same cubic contract without linear fallback; runtime tests assert evaluated presentation values and parameter-sensitive paced arc lengths |
| Hit testing | Normal layer trees traverse stable `zPosition` order. Coordinate conversion and picking compose each hierarchy's full 4×4 transform, including parent `sublayerTransform`, then project and invert the resulting plane homography so perspective and out-of-plane rotations remain aligned with rendered pixels. Singular projections return non-finite coordinates instead of silently using an unrelated affine result. `CATransformLayer` rejects 2D hit testing with `nil`, because its true-3D hierarchy has no single 2D coordinate space; callers must provide scene-specific 3D picking |
| Transform layers | `CATransformLayer` preserves `zPosition` and `anchorPointZ`, orders blended children by projected center depth, and enables per-pixel WebGPU depth writes/tests only within the true-3D hierarchy. Independent transform groups reset depth without changing color; transparent texels discard before depth write. A normal `CALayer` child subtree is captured with transparent clear and composited as one plane, while a nested `CATransformLayer` keeps the shared depth space. Capture extents recursively union transformed out-of-bounds descendants unless an ancestor clips with `masksToBounds`; oversized logical extents reduce resolution to WebGPU limits instead of cropping geometry. Layer-sized effect captures preserve rendered mask trees, partial mask alpha, mask-descendant filters, group opacity, root filters, and nested filters without falling back to viewport-sized composites. Premultiplied capture composition applies mask alpha once, and detached mask-tree revisions invalidate explicit raster caches after mutation. Shadow captures expand beyond layer bounds for offsets and blur, derive silhouettes from subtree alpha or `shadowPath`, and composite behind content on the same 3D plane. Browser readback verifies intersecting planes, transparent cutouts, group isolation, flattening, nested 3D, overflow clipping, effects, masks, shadows, mutation, and cache reuse |
| Remaining boundary | Core Image transition types without executable WGSL remain unavailable. Compositing and background-filter effects inside the mask layer tree are explicitly rejected pending recursive backdrop propagation; complete QuartzCore parity is not claimed |

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
| `CAShapeLayer` | Vector shape rendering with paths, stroke, and fill |
| `CAGradientLayer` | Linear and radial gradients |
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
| macOS/iOS | Metal (stub) | `Timer` | Testing only |

On native Apple platforms, use Apple's QuartzCore directly for production. OpenCoreAnimation's native implementations are for testing purposes only.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OpenCoreAnimation API                       │
│     (CALayer, CAAnimation, CADisplayLink - QuartzCore API)   │
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
