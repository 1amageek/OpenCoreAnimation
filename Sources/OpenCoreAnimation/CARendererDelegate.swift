//
//  CARendererDelegate.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics

// MARK: - Renderer Delegate Protocol

/// Internal protocol for rendering backends that execute CALayer rendering operations.
///
/// This protocol enables pluggable rendering backends (such as WebGPU for WASM)
/// to receive layer trees and render them to their respective targets.
///
/// ## Architecture-Based Selection
///
/// The renderer delegate is configured internally based on the target architecture:
/// - **WASM**: `CAWebGPURenderer` using WebGPU
/// - **Native (Apple)**: `CAMetalRenderer` using Metal
/// - **Native (Other)**: `CAMetalRendererDelegate` stub for testing
///
/// ## Key Design Principles
///
/// - **Internal**: Not exposed to external users
/// - **Non-weak**: The owner (CAAnimationEngine) owns the renderer
/// - **Non-optional**: Rendering cannot proceed without a renderer
///
/// ## Implementation Example
///
/// ```swift
/// final class MyRenderer: CARendererDelegate {
///     var size: CGSize = .zero
///
///     func initialize() async throws {
///         // Set up GPU resources
///     }
///
///     func render(layer: CALayer) {
///         // Traverse and render layer tree
///     }
///
///     func resize(width: Int, height: Int) {
///         size = CGSize(width: width, height: height)
///     }
///
///     func invalidate() {
///         // Release GPU resources
///     }
/// }
/// ```
internal protocol CARendererDelegate: AnyObject, Sendable {

    // MARK: - Properties

    /// The size of the render target in pixels.
    var size: CGSize { get }

    // MARK: - Lifecycle

    /// Initializes the renderer asynchronously.
    ///
    /// This method sets up the GPU device, creates render pipelines,
    /// and prepares all resources needed for rendering.
    ///
    /// - Throws: `CARendererError` if initialization fails.
    func initialize() async throws

    /// Releases all GPU resources.
    ///
    /// Call this method when the renderer is no longer needed.
    func invalidate()

    // MARK: - Rendering

    /// Renders the layer tree starting from the root layer.
    ///
    /// This method traverses the layer hierarchy and renders each layer
    /// according to its properties (transform, opacity, background color, etc.).
    ///
    /// - Parameter rootLayer: The root layer of the tree to render.
    func render(layer rootLayer: CALayer)

    /// Resizes the render target.
    ///
    /// Call this method when the canvas/view size changes.
    ///
    /// - Parameters:
    ///   - width: The new width in pixels.
    ///   - height: The new height in pixels.
    func resize(width: Int, height: Int)
}

// MARK: - Factory

/// Factory for creating renderer delegates based on architecture.
///
/// This enum provides architecture-specific renderer creation without
/// exposing implementation details to the caller.
internal enum CARendererDelegateFactory {

    /// Creates the appropriate renderer delegate for the current architecture.
    ///
    /// ## Platform-Specific Behavior
    ///
    /// - **WASM**: Returns `CAWebGPURenderer` (async, requires canvas)
    /// - **Native**: Returns `CAMetalRendererDelegate` stub (sync, for testing)
    ///
    /// Native platforms use a synchronous stub renderer because:
    /// 1. Production apps on Apple platforms should use QuartzCore directly
    /// 2. This library's native builds are for testing only
    ///
    /// - Parameter canvas: HTML canvas element (WASM only).
    /// - Returns: A renderer delegate appropriate for the current platform.
    /// - Throws: `CARendererError` if renderer creation fails (WASM only).
    #if arch(wasm32)
    static func createRenderer(canvas: JavaScriptKit.JSObject) async throws -> CARendererDelegate {
        let renderer = CAWebGPURenderer(canvas: canvas)
        try await renderer.initialize()
        return renderer
    }
    #else
    static func createRenderer() -> CARendererDelegate {
        // Native: use stub renderer for testing
        // Production apps on Apple platforms should use QuartzCore directly
        return CAMetalRendererDelegate()
    }
    #endif
}

// MARK: - Native Testing Implementation

#if !arch(wasm32)

/// Metal-based renderer delegate for native testing.
///
/// This is a minimal implementation for testing on macOS/iOS.
/// Production use on Apple platforms should use QuartzCore directly.
internal final class CAMetalRendererDelegate: CARendererDelegate, @unchecked Sendable {

    var size: CGSize = .zero

    func initialize() async throws {
        // No-op for testing
    }

    func invalidate() {
        // No-op for testing
    }

    func render(layer rootLayer: CALayer) {
        // No-op for testing - actual rendering would use Metal
    }

    func resize(width: Int, height: Int) {
        size = CGSize(width: width, height: height)
    }
}

#endif
