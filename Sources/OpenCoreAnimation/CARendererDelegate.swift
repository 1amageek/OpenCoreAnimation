//
//  CARendererDelegate.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
#if arch(wasm32)
import JavaScriptKit
#endif

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
@MainActor internal protocol CARendererDelegate: AnyObject {

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
    /// - **Native**: Returns a lazily configured offscreen `CAMetalRenderer`
    ///
    /// - Parameter canvas: HTML canvas element (WASM only).
    /// - Returns: A renderer delegate appropriate for the current platform.
    /// - Throws: `CARendererError` if renderer creation fails (WASM only).
    #if arch(wasm32)
    @MainActor static func createRenderer(canvas: JavaScriptKit.JSObject) async throws -> CARendererDelegate {
        let renderer = CAWebGPURenderer(canvas: canvas)
        try await renderer.initialize()
        return renderer
    }
    #else
    @MainActor static func createRenderer() -> CARendererDelegate {
        CAMetalRenderer()
    }
    #endif
}
