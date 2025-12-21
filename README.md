# OpenCoreAnimation

A Swift library providing CoreAnimation (QuartzCore) API compatibility for WebAssembly, powered by WebGPU.

## Overview

OpenCoreAnimation enables you to write CoreAnimation code that runs in the browser via WebAssembly. The API is designed to be **100% compatible** with Apple's CoreAnimation framework, allowing existing code to work without modification.

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
| `CALayer` | Base layer with full property support (opacity, transform, shadow, border, mask, filters) |
| `CAShapeLayer` | Vector shape rendering with paths, stroke, and fill |
| `CAGradientLayer` | Linear and radial gradients |
| `CATextLayer` | Text rendering with font and alignment |
| `CAReplicatorLayer` | Instance replication with transforms and color offsets |
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
]
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
    .package(url: "https://github.com/aspect-analytics/OpenCoreAnimation.git", branch: "main")
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

- [OpenCoreGraphics](https://github.com/aspect-analytics/OpenCoreGraphics) - CoreGraphics types for WASM
- [swift-webgpu](https://github.com/aspect-analytics/swift-webgpu) - WebGPU bindings (WASM only)

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
swift test
```

### WASM

```bash
swift build --triple wasm32-unknown-wasi
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
