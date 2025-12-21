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

- `CALayer` - Base layer class with full property support
- `CAShapeLayer` - Vector shape rendering with paths
- `CAGradientLayer` - Linear and radial gradients
- `CATextLayer` - Text rendering
- `CAReplicatorLayer` - Instance replication with transforms
- `CAScrollLayer` - Scrollable content
- `CATransformLayer` - 3D transform container
- `CATiledLayer` - Tiled content rendering
- `CAEmitterLayer` - Particle systems

### Animations

- `CABasicAnimation` - Simple from/to animations
- `CAKeyframeAnimation` - Multi-keyframe animations
- `CASpringAnimation` - Physics-based spring animations
- `CAAnimationGroup` - Animation composition
- `CATransition` - Layer transitions
- `CAMediaTimingFunction` - Easing curves

### Core Components

- `CATransaction` - Implicit animation batching
- `CADisplayLink` - Frame-synchronized callbacks
- `CATransform3D` - 3D transformation matrices
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
rootLayer.addSublayer(boxLayer)
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

// Keyframe animation
let keyframeAnim = CAKeyframeAnimation(keyPath: "position")
keyframeAnim.values = [
    CGPoint(x: 0, y: 0),
    CGPoint(x: 100, y: 50),
    CGPoint(x: 200, y: 0)
]
keyframeAnim.keyTimes = [0, 0.5, 1.0]
keyframeAnim.duration = 2.0
layer.add(keyframeAnim, forKey: "path")
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

        // Initialize WebGPU renderer
        let renderer = try await CAWebGPURenderer(canvas: canvas.object!)

        // Create layer hierarchy
        let rootLayer = CALayer()
        // ... configure layers ...

        // Start animation engine
        let engine = CAAnimationEngine.shared
        engine.rootLayer = rootLayer
        engine.renderer = renderer
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
| macOS/iOS | Metal | `Timer` | Testing only |

On native Apple platforms, use Apple's QuartzCore directly for production. OpenCoreAnimation's native implementations are for testing purposes.

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
