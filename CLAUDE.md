# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenCoreAnimation is a Swift library that provides **full API compatibility with Apple's CoreAnimation (QuartzCore) framework** for WebAssembly (WASM) environments.

### Core Principle: Full Compatibility

**The API must be 100% compatible with CoreAnimation.** This means:
- Identical type names, method signatures, and property names
- Same behavior and semantics as CoreAnimation
- Code written for CoreAnimation should compile and work without modification when using OpenCoreAnimation

### How `canImport` Works

Users of this library will write code like:

```swift
#if canImport(QuartzCore)
import QuartzCore
#else
import OpenCoreAnimation
#endif

let layer = CALayer()
layer.frame = CGRect(x: 0, y: 0, width: 100, height: 100)
```

- **When QuartzCore is available** (iOS, macOS, etc.): Users import QuartzCore directly
- **When QuartzCore is NOT available** (WASM): Users import OpenCoreAnimation

## Build Commands

```bash
swift build              # Build the package
swift test               # Run all tests
swift test --filter <TestName>  # Run a specific test
swift build --triple wasm32-unknown-wasi  # Build for WASM
```

## Platform Strategy

**OpenCoreAnimation primarily targets WASM/Web environments**, but includes native implementations for testing purposes.

### Production Use

- **WASM**: Uses WebGPU via swift-webgpu, JavaScriptKit for browser APIs
- **Native platforms (iOS, macOS)**: Users should import Apple's QuartzCore directly

### Native Test Support

To enable `swift test` on macOS/Linux, the library includes fallback implementations:

| Component | WASM Implementation | Native (Test) Implementation |
|-----------|---------------------|------------------------------|
| Timing | `performance.now()` | `ProcessInfo.systemUptime` |
| Display Link | `requestAnimationFrame` | `Timer` |
| Transactions | `setTimeout` | `DispatchQueue.main.async` |
| Renderer | `CAWebGPURenderer` | `CAMetalRenderer` |
| Graphics | `OpenCoreGraphics` | `CoreGraphics` (re-exported) |

These native implementations are **for testing only** and should not be used in production.

### Conditional Compilation

Use `#if arch(wasm32)` to distinguish between WASM and native platforms:

```swift
#if arch(wasm32)
import JavaScriptKit
// WASM-specific implementation
#else
import Foundation
// Native fallback for testing
#endif
```

For graphics types, `OpenCoreAnimation.swift` re-exports the appropriate module:

```swift
#if canImport(CoreGraphics)
@_exported import CoreGraphics      // Native: use Apple's CoreGraphics
#else
@_exported import OpenCoreGraphics  // WASM: use OpenCoreGraphics
#endif
```

### WASM-Specific Features

For WASM-specific features like timing and display links, use JavaScriptKit directly:

```swift
import OpenCoreGraphics
import JavaScriptKit

// Use JavaScript APIs for timing
let performance = JSObject.global.performance
let timestamp = performance.now().number ?? 0

// Use requestAnimationFrame for display sync
let callback = JSClosure { ... }
_ = JSObject.global.requestAnimationFrame!(callback)
```

## Testing

Uses Swift Testing framework (not XCTest):

```swift
import Testing
import OpenCoreGraphics
@testable import OpenCoreAnimation

@Test func testCALayerFrame() {
    let layer = CALayer()
    layer.frame = CGRect(x: 10, y: 20, width: 100, height: 200)
    #expect(layer.frame == CGRect(x: 10, y: 20, width: 100, height: 200))
}
```

## WebGPU Rendering Backend

### Overview

OpenCoreAnimation **primarily targets WASM/Web environments** and uses **WebGPU** as its GPU rendering backend via [swift-webgpu](https://github.com/1amageek/swift-webgpu). This provides hardware-accelerated layer rendering comparable to Metal on Apple platforms.

**Key point**: On native platforms (iOS, macOS), users should import Apple's QuartzCore directly for production use. OpenCoreAnimation includes native fallback implementations (Metal renderer, Foundation-based timing) for **testing purposes only**.

### Dependency: swift-webgpu

```swift
// Package.swift dependency
.package(url: "https://github.com/1amageek/swift-webgpu.git", branch: "main")

// Target dependency
.target(
    name: "OpenCoreAnimation",
    dependencies: [
        "OpenCoreGraphics",
        .product(name: "WebGPU", package: "swift-webgpu")
    ]
)
```

swift-webgpu provides:
- Type-safe Swift bindings for WebGPU API
- JavaScriptKit-based interop for WASM
- GPUDevice, GPUBuffer, GPUTexture, GPURenderPipeline types

### JavaScriptKit Fundamentals

swift-webgpu uses [JavaScriptKit](https://github.com/aspect-analytics/aspect-labs-swift-javascriptkit) for Swift-to-JavaScript interop. Understanding these patterns is essential for WASM-compatible implementations.

#### Core Types

```swift
import JavaScriptKit

// JSObject - Represents any JavaScript object
let global = JSObject.global  // Access to window/globalThis

// JSValue - Any JavaScript value (string, number, object, etc.)
let value: JSValue = .number(42)
let stringValue: JSValue = .string("hello")

// Type conversion
let jsNumber = JSValue.number(3.14)
let swiftDouble = jsNumber.number!  // Convert to Swift Double
```

#### Accessing Browser APIs

```swift
import JavaScriptKit

// Access global objects
let navigator = JSObject.global.navigator
let document = JSObject.global.document
let performance = JSObject.global.performance
let window = JSObject.global

// Call JavaScript methods
let timestamp = performance.now()  // Returns JSValue
let currentTime = timestamp.number! / 1000.0  // Convert to seconds

// Access nested properties
let gpu = navigator.gpu  // navigator.gpu for WebGPU
```

#### Async Operations with JSPromise

```swift
import JavaScriptKit

// JavaScript promises are wrapped as JSPromise
func initializeGPU() async throws -> GPUDevice {
    let gpu = JSObject.global.navigator.gpu
    let adapterPromise = gpu.requestAdapter()
    let adapter = try await JSPromise(adapterPromise.object!)!.value

    let devicePromise = adapter.requestDevice()
    let device = try await JSPromise(devicePromise.object!)!.value

    return GPUDevice(jsObject: device.object!)
}
```

#### Closures as JavaScript Callbacks

```swift
import JavaScriptKit

// Create JavaScript-callable closure
let callback = JSClosure { arguments in
    let timestamp = arguments[0].number!
    // Handle callback
    return .undefined
}

// Use with requestAnimationFrame
_ = JSObject.global.requestAnimationFrame!(callback)

// IMPORTANT: Keep reference to closure to prevent deallocation
// Store in instance variable or class property
```

### WASM-Compatible Implementation Patterns

**CRITICAL**: WASM environments do NOT have Foundation, Dispatch, or Darwin APIs. Use JavaScriptKit for all platform-specific functionality.

#### CACurrentMediaTime → performance.now()

```swift
import JavaScriptKit

/// Returns the current absolute time in seconds (WASM implementation)
public func CACurrentMediaTime() -> CFTimeInterval {
    // performance.now() returns milliseconds
    let performance = JSObject.global.performance
    let milliseconds = performance.now().number!
    return milliseconds / 1000.0
}
```

#### CADisplayLink → requestAnimationFrame

```swift
import JavaScriptKit

open class CADisplayLink {
    private var animationFrameCallback: JSClosure?
    private var animationFrameId: Int32 = 0
    private var isRunning = false

    private func startAnimationLoop() {
        // Create JavaScript callback for requestAnimationFrame
        animationFrameCallback = JSClosure { [weak self] arguments in
            guard let self = self, self.isRunning, !self.isPaused else {
                return .undefined
            }

            // Update timestamps
            let timestamp = arguments[0].number! / 1000.0  // Convert to seconds
            self.timestamp = timestamp
            self.targetTimestamp = timestamp + self.duration

            // Call target's displayLinkDidFire
            if let target = self.target as? CADisplayLinkTarget {
                target.displayLinkDidFire(self)
            }

            // Request next frame
            if self.isRunning && !self.isPaused {
                self.requestNextFrame()
            }

            return .undefined
        }

        requestNextFrame()
    }

    private func requestNextFrame() {
        guard let callback = animationFrameCallback else { return }
        let result = JSObject.global.requestAnimationFrame!(callback)
        animationFrameId = Int32(result.number!)
    }

    private func stopAnimationLoop() {
        if animationFrameId != 0 {
            _ = JSObject.global.cancelAnimationFrame!(animationFrameId)
            animationFrameId = 0
        }
        animationFrameCallback = nil
    }
}
```

#### CATransaction → No Locks Needed

**WASM is single-threaded.** No locks or synchronization primitives are needed:

```swift
open class CATransaction {
    // WASM is single-threaded - simple stack-based implementation
    private static var transactionStack: [TransactionState] = []

    open class func begin() {
        transactionStack.append(TransactionState())
    }

    open class func commit() {
        guard !transactionStack.isEmpty else { return }
        let state = transactionStack.removeLast()
        // Apply animations from this transaction
    }

    // No lock() or unlock() needed - WASM is single-threaded
    open class func lock() {
        // No-op in WASM
    }

    open class func unlock() {
        // No-op in WASM
    }
}
```

#### setTimeout / setInterval Patterns

```swift
import JavaScriptKit

// One-time delayed execution
func setTimeout(milliseconds: Int, callback: @escaping () -> Void) -> Int32 {
    let jsClosure = JSClosure { _ in
        callback()
        return .undefined
    }
    let result = JSObject.global.setTimeout!(jsClosure, milliseconds)
    return Int32(result.number!)
}

// Repeating execution
func setInterval(milliseconds: Int, callback: @escaping () -> Void) -> Int32 {
    let jsClosure = JSClosure { _ in
        callback()
        return .undefined
    }
    let result = JSObject.global.setInterval!(jsClosure, milliseconds)
    return Int32(result.number!)
}

// Cancel timers
func clearTimeout(_ id: Int32) {
    _ = JSObject.global.clearTimeout!(id)
}

func clearInterval(_ id: Int32) {
    _ = JSObject.global.clearInterval!(id)
}
```

### WebGPU Usage Patterns

#### Device Initialization

```swift
import WebGPU
import JavaScriptKit

class GPUContextManager {
    var device: GPUDevice?
    var context: GPUCanvasContext?

    func initialize(canvas: JSObject) async throws {
        // Request adapter
        let gpu = JSObject.global.navigator.gpu
        guard let adapterPromise = gpu.requestAdapter().object else {
            throw GPUError.adapterNotAvailable
        }
        let adapter = try await JSPromise(adapterPromise)!.value

        // Request device
        guard let devicePromise = adapter.requestDevice().object else {
            throw GPUError.deviceNotAvailable
        }
        let jsDevice = try await JSPromise(devicePromise)!.value
        device = GPUDevice(jsObject: jsDevice.object!)

        // Configure canvas context
        let ctx = canvas.getContext!("webgpu")
        context = GPUCanvasContext(jsObject: ctx.object!)

        let format = gpu.getPreferredCanvasFormat!().string!
        context?.configure(GPUCanvasConfiguration(
            device: device!,
            format: GPUTextureFormat(rawValue: format)!
        ))
    }
}
```

#### Render Pipeline Creation

```swift
let pipelineDescriptor = GPURenderPipelineDescriptor(
    vertex: GPUVertexState(
        module: shaderModule,
        entryPoint: "vertexMain"
    ),
    fragment: GPUFragmentState(
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [
            GPUColorTargetState(format: .bgra8unorm)
        ]
    ),
    primitive: GPUPrimitiveState(
        topology: .triangleList
    )
)

let pipeline = device.createRenderPipeline(pipelineDescriptor)
```

#### Render Pass Execution

```swift
func render() {
    guard let device = device,
          let context = context else { return }

    let encoder = device.createCommandEncoder()

    let renderPass = encoder.beginRenderPass(GPURenderPassDescriptor(
        colorAttachments: [
            GPURenderPassColorAttachment(
                view: context.getCurrentTexture().createView(),
                loadOp: .clear,
                storeOp: .store,
                clearValue: GPUColor(r: 0, g: 0, b: 0, a: 1)
            )
        ]
    ))

    renderPass.setPipeline(pipeline)
    renderPass.draw(3)  // Draw 3 vertices
    renderPass.end()

    device.queue.submit([encoder.finish()])
}
```

### Import Strategy (Platform Conditional Compilation)

Use `#if arch(wasm32)` to distinguish between WASM and native platforms:

```
┌──────────────────────────────────────────────────────────────────┐
│                      OpenCoreAnimation                           │
├──────────────────────────────────────────────────────────────────┤
│  Graphics Types (OpenCoreAnimation.swift)                        │
│  ├── #if canImport(CoreGraphics)                                 │
│  │   └── @_exported import CoreGraphics  (Native)                │
│  └── #else                                                       │
│      └── @_exported import OpenCoreGraphics  (WASM)              │
├──────────────────────────────────────────────────────────────────┤
│  #if arch(wasm32)                                                │
│  │   import JavaScriptKit                                        │
│  │   import SwiftWebGPU                                          │
│  │   ├── CAWebGPURenderer (WebGPU rendering)                     │
│  │   ├── performance.now() (timing)                              │
│  │   └── requestAnimationFrame (display link)                    │
│  │                                                               │
│  #else (Apple/Linux - for testing)                               │
│  │   #if canImport(Metal)                                        │
│  │   │   import Metal                                            │
│  │   │   └── CAMetalRenderer (Metal rendering)                   │
│  │   #endif                                                      │
│  │   └── Foundation (timing via ProcessInfo)                     │
│  #endif                                                          │
└──────────────────────────────────────────────────────────────────┘
```

#### Condition Summary

| Condition | Purpose |
|-----------|---------|
| `#if canImport(CoreGraphics)` | Graphics types re-export (Native vs OpenCoreGraphics) |
| `#if arch(wasm32)` | WASM platform (WebGPU, JavaScriptKit) |
| `#if canImport(Metal)` | Metal renderer (Apple platforms for testing) |
| `#if canImport(simd)` | SIMD helper types (Apple platforms) |

#### Example: CAMediaTiming.swift

```swift
import OpenCoreGraphics

public typealias CFTimeInterval = Double

#if arch(wasm32)
import JavaScriptKit

public func CACurrentMediaTime() -> CFTimeInterval {
    let performance = JSObject.global.performance
    return performance.now().number! / 1000.0
}
#else
import Foundation

public func CACurrentMediaTime() -> CFTimeInterval {
    return ProcessInfo.processInfo.systemUptime
}
#endif
```

#### Example: Renderer Selection

```swift
// CARenderer.swift - Protocol
public protocol CARenderer: AnyObject {
    init() async throws
    func render(rootLayer: CALayer)
    func resize(width: Int, height: Int)
    func invalidate()
}

// CAWebGPURenderer.swift - WASM
#if arch(wasm32)
import SwiftWebGPU

public final class CAWebGPURenderer: CARenderer {
    // WebGPU implementation
}
#endif

// CAMetalRenderer.swift - Apple platforms (testing)
#if canImport(Metal)
import Metal

public final class CAMetalRenderer: CARenderer {
    // Metal implementation
}
#endif
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OpenCoreAnimation API                       │
│     (CALayer, CAAnimation, CADisplayLink - QuartzCore API)   │
├─────────────────────────────────────────────────────────────┤
│                  WebGPU Rendering Layer                      │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ GPUContextManager│  │ LayerRenderer    │                  │
│  │ (Device init)   │  │ (Layer drawing)  │                  │
│  └─────────────────┘  └──────────────────┘                  │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ GPUTexturePool  │  │ AnimationScheduler│                  │
│  │ (Memory mgmt)   │  │ (Timing/frames)  │                  │
│  └─────────────────┘  └──────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                     swift-webgpu                             │
│         (SwiftWebGPU - Type-safe WebGPU bindings)            │
├─────────────────────────────────────────────────────────────┤
│                     JavaScriptKit                            │
│              (Swift-to-JavaScript bridge)                    │
├─────────────────────────────────────────────────────────────┤
│                   Browser WebGPU API                         │
│                    (navigator.gpu)                           │
└─────────────────────────────────────────────────────────────┘
```

### User-Facing Platform Strategy

OpenCoreAnimation **primarily targets WASM/Web environments**. Native implementations exist only for testing.

Users select between QuartzCore and OpenCoreAnimation at the import level:

```swift
// User's application code
#if canImport(QuartzCore)
import QuartzCore  // Native platforms - uses Metal
#else
import OpenCoreAnimation  // WASM/Web - uses WebGPU
#endif

// Same API works in both environments
let layer = CALayer()
layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
```

## Implementation Policy

- **Do NOT implement deprecated APIs** - Only implement current, non-deprecated CoreAnimation APIs
- Always refer to Apple's official CoreAnimation documentation to ensure API signatures match exactly
- Focus on APIs meaningful for WASM environments

### Excluded APIs (Not Implemented)

The following CoreAnimation APIs are **not implemented** and will not be added:

| API | Status |
|-----|--------|
| `CAOpenGLLayer` | Deleted (legacy OpenGL) |
| `CAEAGLLayer` | Deleted (legacy OpenGL ES for iOS) |
| `CAMetalLayer` | Not implemented (Apple Metal specific) |
| `CAMetalDrawable` | Not implemented (Apple Metal specific) |
| `CAEDRMetadata` | Not implemented (Apple HDR specific) |

OpenCoreAnimation targets WASM with WebGPU. Platform-specific GPU APIs (Metal, OpenGL) are out of scope.

### API Compatibility Notes

#### CADisplayLink

CADisplayLink maintains full API signature compatibility with Apple's CoreAnimation. The `add(to:forMode:)` method uses `RunLoop` and `RunLoop.Mode` types on both platforms.

**CoreAnimation (Native):**
```swift
let displayLink = CADisplayLink(target: self, selector: #selector(update))
displayLink.add(to: .main, forMode: .common)

@objc func update(_ displayLink: CADisplayLink) {
    // Called via Objective-C runtime
}
```

**OpenCoreAnimation (WASM):**
```swift
// Selector is a placeholder type (no Objective-C runtime on WASM)
let displayLink = CADisplayLink(target: self, selector: Selector(""))
displayLink.add(to: .main, forMode: .common)  // Same API!

// Target must conform to CADisplayLinkDelegate
extension MyClass: CADisplayLinkDelegate {
    func displayLinkDidFire(_ displayLink: CADisplayLink) {
        // Called via protocol method
    }
}
```

**Key Differences:**
- **Selector**: On WASM, `Selector` is a stub type. Use `CADisplayLinkDelegate` for callbacks instead of `@objc` methods.
- **RunLoop.Mode**: On WASM, `RunLoop.Mode` values (`.common`, `.default`) are accepted but ignored. JavaScript's `requestAnimationFrame` always fires when the browser is ready to paint, which is equivalent to `.common` mode behavior.
- **RunLoop**: A stub `RunLoop` type is provided for API compatibility. On WASM, there is no actual run loop; `requestAnimationFrame` is used internally.

**Migration:** When porting code to WASM, the only change required is replacing `@objc` selector-based callbacks with `CADisplayLinkDelegate` conformance. The `add(to:forMode:)` call remains identical.

## OpenCoreGraphics Reference Architecture

OpenCoreAnimation's WebGPU rendering should follow the patterns established in [OpenCoreGraphics](../OpenCoreGraphics). This section documents key architectural decisions that apply to CALayer rendering.

### Project Structure Pattern

OpenCoreGraphics separates API layer from rendering implementation:

```
Sources/OpenCoreGraphics/
├── Graphics/                    # CoreGraphics-compatible API layer
│   ├── CGContext.swift          # Drawing context (command recorder)
│   ├── CGPath.swift             # Path primitives
│   └── CGContextRendererDelegate.swift  # Renderer protocol
└── Rendering/WebGPU/            # WASM-specific rendering
    ├── CGWebGPUContextRenderer.swift    # Main renderer
    ├── PathTessellator.swift            # Path → triangles
    ├── Shaders.swift                    # WGSL shaders
    └── Internal/
        ├── BufferPool.swift             # Ring buffer pool
        ├── TextureManager.swift         # LRU texture cache
        └── GeometryCache.swift          # Tessellation cache
```

**Apply to OpenCoreAnimation:**
```
Sources/OpenCoreAnimation/
├── Layers/                      # CALayer API (CoreAnimation-compatible)
├── Animation/                   # CAAnimation API
└── Rendering/WebGPU/            # WASM rendering
    ├── CAWebGPURenderer.swift
    ├── LayerTessellator.swift   # Layer bounds → quads
    └── Internal/
        ├── BufferPool.swift
        └── TextureManager.swift
```

### Renderer Delegate Pattern (Internal Architecture Switching)

OpenCoreGraphics uses a **protocol-based internal delegate** for rendering. The delegate is:

- **Internal**: Not exposed to external users - implementation is selected automatically based on platform
- **Non-weak (Strong reference)**: The context owns the renderer; no retain cycle exists
- **Non-optional (Conceptually)**: On target platforms (WASM), a renderer is always required

#### Key Design Principle

The delegate is **not externally provided** - it is configured internally based on the target architecture at initialization time. This means:

1. Users never set or access the renderer delegate directly
2. The correct implementation is selected via `#if arch(wasm32)` at compile time
3. The delegate has the same lifetime as the context that owns it

#### OpenCoreGraphics Implementation

```swift
// CGContextRendererDelegate.swift - Internal protocol
internal protocol CGContextRendererDelegate: AnyObject, Sendable {
    func fill(path: CGPath, color: CGColor, alpha: CGFloat, blendMode: CGBlendMode, rule: CGPathFillRule)
    func stroke(path: CGPath, color: CGColor, lineWidth: CGFloat, ...)
    func draw(image: CGImage, in rect: CGRect, ...)
    // ... other drawing operations
}

// CGContext.swift - Internal delegate ownership
public class CGContext {
    /// Strong reference - CGContext owns the renderer on WASM.
    /// The renderer is created internally and should live as long as the context.
    internal var rendererDelegate: CGContextRendererDelegate?

    public init?(width: Int, height: Int, ...) {
        // Architecture-based internal initialization
        #if arch(wasm32)
        let renderer = CGWebGPUContextRenderer(width: width, height: height)
        renderer.setup()
        self.rendererDelegate = renderer  // Internal, not user-provided
        #endif
    }

    public func fillPath(using rule: CGPathFillRule) {
        // Delegate call - always available on WASM
        rendererDelegate?.fill(path: transformedPath, color: currentState.fillColor, ...)
    }
}
```

#### Why Non-Weak and Non-Optional

| Aspect | Reason |
|--------|--------|
| **Non-weak** | Context owns renderer; no back-reference that would cause retain cycle |
| **Non-optional** | Rendering cannot proceed without a renderer on target platform (WASM) |
| **Internal** | Implementation detail - users interact only with CGContext/CALayer API |
| **Architecture-based** | `#if arch(wasm32)` selects WebGPU; native uses Metal for testing |

#### Apply to OpenCoreAnimation

CARenderer should follow the same pattern:

```swift
// CARenderer.swift - Protocol
public protocol CARenderer: AnyObject {
    func render(layer: CALayer)
    func resize(width: Int, height: Int)
    func invalidate()
}

// CAAnimationEngine.swift or similar - Internal delegate ownership
public final class CAAnimationEngine {
    /// Internal renderer - not weak, not optional on WASM.
    /// Selected based on architecture at initialization.
    internal var renderer: CARenderer

    init() {
        #if arch(wasm32)
        self.renderer = CAWebGPURenderer()
        #else
        self.renderer = CAMetalRenderer()  // For testing
        #endif
    }
}
```

> **Note**: The current implementation uses `var renderer: CARenderer?` (optional) for flexibility during initialization. The principle is that once initialized on WASM, it should never be nil during normal operation.

### Path Tessellation (CPU-side)

OpenCoreGraphics converts vector paths to GPU triangles on CPU:

1. **Bézier Flattening**: Recursive subdivision until flatness threshold
2. **Polygon Triangulation**: Ear-clipping algorithm for fills
3. **Stroke Mesh Generation**: Quads along path with caps/joins

```swift
// PathTessellator.swift pattern
public struct PathTessellator {
    public func tessellateFill(path: CGPath, transform: CGAffineTransform) -> [CGWebGPUVertex] {
        let flattenedPoints = flattenPath(path, transform: transform)
        let triangles = EarClipping.triangulate(polygon: flattenedPoints)
        return triangles.map { CGWebGPUVertex(position: $0, color: fillColor) }
    }
}
```

**For CALayer**: Layers are typically rectangular, so tessellation is simpler (2 triangles per layer). Complex shapes use CAShapeLayer which delegates to CGContext.

### Vertex Data Format

OpenCoreGraphics uses interleaved vertex data:

```swift
public struct CGWebGPUVertex: Sendable {
    public var x: Float        // NDC position X
    public var y: Float        // NDC position Y
    public var r: Float        // Color R (0-1)
    public var g: Float        // Color G (0-1)
    public var b: Float        // Color B (0-1)
    public var a: Float        // Color A (0-1)

    public static var stride: UInt64 { 24 }  // 6 floats × 4 bytes
}

// Batch conversion for GPU upload
extension Array where Element == CGWebGPUVertex {
    public func toFloatArray() -> [Float] {
        var data: [Float] = []
        data.reserveCapacity(count * 6)
        for v in self {
            data.append(contentsOf: [v.x, v.y, v.r, v.g, v.b, v.a])
        }
        return data
    }
}
```

### Coordinate Transformation

CoreGraphics uses bottom-left origin (Y-up), WebGPU uses NDC (center origin, -1 to 1):

```swift
// Transform CG coords to NDC
func toNDC(x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat) -> (Float, Float) {
    let ndcX = Float(2.0 * x / width - 1.0)
    let ndcY = Float(2.0 * y / height - 1.0)
    return (ndcX, ndcY)
}
```

### GPU Memory Management

OpenCoreGraphics implements several caching strategies:

#### BufferPool (Ring Buffer)

Prevents GPU/CPU synchronization stalls:

```swift
public final class BufferPool: @unchecked Sendable {
    private let device: GPUDevice
    private var buffers: [GPUBuffer] = []
    private var currentIndex: Int = 0
    private let frameCount: Int = 3  // Triple buffering

    public func allocate(size: Int) -> GPUBuffer {
        // Return buffer for current frame, cycle to next
        let buffer = buffers[currentIndex]
        currentIndex = (currentIndex + 1) % frameCount
        return buffer
    }
}
```

#### TextureManager (LRU Cache)

Caches CGImage → GPUTexture conversions:

```swift
public final class TextureManager: @unchecked Sendable {
    private var cache: [ObjectIdentifier: CacheEntry] = [:]
    private let maxMemoryBytes: Int = 256 * 1024 * 1024  // 256 MB

    public func texture(for image: CGImage) -> GPUTexture {
        let id = ObjectIdentifier(image)
        if let cached = cache[id] {
            cached.accessTime = CACurrentMediaTime()
            return cached.texture
        }
        // Create new texture, evict LRU if needed
        return createTexture(from: image)
    }
}
```

#### GeometryCache

Caches tessellated vertex data for repeated paths:

```swift
public final class GeometryCache: @unchecked Sendable {
    private var cache: [PathKey: CachedGeometry] = [:]
    private let maxEntries: Int = 500

    public func geometry(for path: CGPath, transform: CGAffineTransform) -> [CGWebGPUVertex]? {
        let key = PathKey(path: path, transform: transform)
        return cache[key]?.vertices
    }
}
```

### WGSL Shader Patterns

OpenCoreGraphics defines shaders as Swift string constants:

```swift
// Shaders.swift
public enum CGWebGPUShaders {
    public static let simple2D = """
    struct VertexInput {
        @location(0) position: vec2f,
        @location(1) color: vec4f,
    }

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec4f,
    }

    @vertex
    fn vs_main(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = vec4f(input.position, 0.0, 1.0);
        output.color = input.color;
        return output;
    }

    @fragment
    fn fs_main(input: VertexOutput) -> @location(0) vec4f {
        return input.color;
    }
    """

    public static let linearGradient = """
    // Gradient direction uniform
    @group(0) @binding(0) var<uniform> gradientParams: GradientParams;

    @fragment
    fn fs_main(input: VertexOutput) -> @location(0) vec4f {
        let t = dot(input.position.xy, gradientParams.direction);
        return mix(gradientParams.startColor, gradientParams.endColor, t);
    }
    """
}
```

### Clipping Implementation

Uses stencil buffer for complex clip paths:

```swift
// 1. Render clip path to stencil buffer (increment)
// 2. Set stencil test (only render where stencil > 0)
// 3. Render content
// 4. Clear stencil for next clip

let stencilState = GPUDepthStencilState(
    format: .depth24plusStencil8,
    depthWriteEnabled: false,
    stencilFront: GPUStencilFaceState(
        compare: .equal,
        passOp: .keep
    )
)
```

### Shadow Rendering

Uses offscreen rendering + blur:

```swift
// 1. Render shape to shadow mask texture
// 2. Apply Gaussian blur (horizontal + vertical passes)
// 3. Composite shadow with offset to main target

func renderShadow(layer: CALayer, offset: CGSize, blur: CGFloat, color: CGColor) {
    // Render to shadowMaskTexture
    renderToTexture(shadowMaskTexture) {
        renderLayerShape(layer)
    }

    // Apply blur passes
    applyGaussianBlur(texture: shadowMaskTexture, radius: blur)

    // Composite with offset
    compositeTexture(shadowMaskTexture, offset: offset, color: color)
}
```

### Rendering Pipeline Flow

```
CALayer API
    ↓
CAWebGPURenderer.render(rootLayer:)
    ↓
Layer tree traversal (respecting zPosition, sublayers)
    ↓
For each layer:
    ├── Apply transform (position, anchorPoint, transform)
    ├── Clip to bounds if masksToBounds
    ├── Render shadow if shadowOpacity > 0
    ├── Render backgroundColor (2 triangles)
    ├── Render contents (CGImage texture)
    ├── Render border if borderWidth > 0
    └── Render sublayers recursively
    ↓
BufferPool.allocate() → GPUBuffer write
    ↓
GPURenderPass execution
    ↓
queue.submit([commandBuffer])
```

### Key Design Principles

1. **API/Rendering Separation**: Keep CoreAnimation API layer separate from WebGPU implementation
2. **CPU Tessellation**: Convert shapes to triangles on CPU, send vertex buffers to GPU
3. **Efficient Memory**: Use ring buffers, LRU caches to minimize allocations
4. **Lazy Compilation**: Create pipelines on-demand, cache for reuse
5. **Stateless Commands**: Pass all state to shaders via uniforms/vertex data
6. **Sendable Types**: All GPU resources marked `@unchecked Sendable` (WASM is single-threaded)

## Future Work (TODO)

### Android / Vulkan Support

Currently, OpenCoreAnimation supports:
- **WASM** (Production): WebGPU via swift-webgpu
- **Apple platforms** (Testing only): Metal renderer, Foundation-based timing

Future support could include:
- **Android**: Vulkan renderer (CAVulkanRenderer)
- **Linux**: Vulkan or OpenGL fallback

This would require:
1. Platform detection for Android (`#if os(Android)`)
2. Vulkan bindings for Swift
3. CAVulkanRenderer implementation following CARenderer protocol

## Reference

- [Core Animation Documentation](https://developer.apple.com/documentation/quartzcore)
- [Core Animation Programming Guide](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CoreAnimation_guide/Introduction/Introduction.html)
- [swift-webgpu](https://github.com/1amageek/swift-webgpu)