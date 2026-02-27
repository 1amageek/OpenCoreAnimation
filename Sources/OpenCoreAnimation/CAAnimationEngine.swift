#if arch(wasm32)
import JavaScriptKit
#endif

/// The animation engine orchestrates the animation loop, connecting
/// display link timing to layer presentation updates and rendering.
///
/// This class provides the core animation loop that:
/// 1. Processes animation completions for all layers in the tree
/// 2. Updates presentation layers with current animated values
/// 3. Triggers rendering of the layer tree
///
/// ## Usage
///
/// ```swift
/// let engine = CAAnimationEngine.shared
/// engine.rootLayer = myRootLayer
/// engine.start()
/// ```
///
/// The engine automatically handles the display refresh cycle and ensures
/// animations are properly updated each frame.
///
/// ## Renderer Delegate Pattern
///
/// The renderer is configured automatically based on the target architecture:
/// - **WASM**: Uses `CAWebGPURenderer` (configured via `setCanvas(_:)`)
/// - **Native**: Uses `CAMetalRendererDelegate` stub (for testing)
///
/// The renderer delegate is:
/// - **Internal**: Not exposed to external users
/// - **Non-weak**: The engine owns the renderer
/// - **Auto-configured**: Selected based on architecture at initialization
public final class CAAnimationEngine: CADisplayLinkDelegate {

    // MARK: - Singleton

    /// The shared animation engine instance.
    ///
    /// Use this shared instance for the main application animation loop.
    public static nonisolated(unsafe) let shared = CAAnimationEngine()

    // MARK: - Properties

    /// The root layer of the layer tree to animate.
    ///
    /// Set this to the root layer of your layer hierarchy. The engine will
    /// recursively process animations for all sublayers.
    public weak var rootLayer: CALayer?

    /// Internal renderer delegate - configured automatically based on architecture.
    ///
    /// - WASM: `CAWebGPURenderer`
    /// - Native: `CAMetalRendererDelegate` stub (for testing)
    ///
    /// This is a strong reference because the engine owns the renderer.
    /// The renderer is created automatically during initialization.
    internal var rendererDelegate: CARendererDelegate?

    /// Public accessor for the renderer (read-only).
    ///
    /// This provides access to the renderer for advanced use cases
    /// while maintaining internal ownership.
    public var renderer: CARenderer? {
        return rendererDelegate as? CARenderer
    }

    /// The display link driving the animation loop.
    private var displayLink: CADisplayLink?

    /// Whether the engine is currently running.
    public private(set) var isRunning: Bool = false

    /// The preferred frame rate for animations.
    public var preferredFrameRate: Float = 60 {
        didSet {
            displayLink?.preferredFrameRateRange = CAFrameRateRange(
                minimum: preferredFrameRate,
                maximum: preferredFrameRate,
                preferred: preferredFrameRate
            )
        }
    }

    // MARK: - Initialization

    /// Creates a new animation engine.
    ///
    /// For most use cases, use the shared instance instead of creating new engines.
    /// The renderer is configured automatically based on the target architecture.
    public init() {
        #if !arch(wasm32)
        // Native: auto-configure renderer for testing
        rendererDelegate = CARendererDelegateFactory.createRenderer()
        #endif
    }

    deinit {
        stop()
    }

    #if arch(wasm32)
    // MARK: - WASM Configuration

    /// Sets the canvas element for WebGPU rendering.
    ///
    /// This method must be called before starting the animation loop on WASM.
    /// It initializes the WebGPU renderer with the provided canvas element.
    ///
    /// - Parameter canvas: The HTML canvas element to render to.
    /// - Throws: `CARendererError` if renderer initialization fails.
    public func setCanvas(_ canvas: JavaScriptKit.JSObject) async throws {
        rendererDelegate = try await CARendererDelegateFactory.createRenderer(canvas: canvas)
    }
    #endif

    // MARK: - Control

    /// Starts the animation loop.
    ///
    /// Call this method to begin processing animations. The engine will
    /// automatically call the renderer each frame to update the display.
    public func start() {
        guard !isRunning else { return }

        isRunning = true
        displayLink = CADisplayLink(target: self, selector: Selector("displayLinkDidFire"))
        displayLink?.preferredFrameRateRange = CAFrameRateRange(
            minimum: preferredFrameRate,
            maximum: preferredFrameRate,
            preferred: preferredFrameRate
        )
        displayLink?.add(to: .main, forMode: .common)
    }

    /// Stops the animation loop.
    ///
    /// Call this method to stop processing animations. The display will
    /// remain in its current state until the loop is restarted.
    public func stop() {
        guard isRunning else { return }

        isRunning = false
        displayLink?.invalidate()
        displayLink = nil
    }

    /// Pauses the animation loop without invalidating the display link.
    public func pause() {
        displayLink?.isPaused = true
    }

    /// Resumes a paused animation loop.
    public func resume() {
        displayLink?.isPaused = false
    }

    // MARK: - CADisplayLinkDelegate

    /// Called by the display link on each frame refresh.
    ///
    /// This method:
    /// 1. Renders the layer tree using presentation layer values
    /// 2. Processes animation completions for all layers
    ///
    /// Rendering must happen before completion processing so that
    /// `.forwards` fill mode animations render their final frame
    /// before being removed.
    public func displayLinkDidFire(_ displayLink: CADisplayLink) {
        // Render the layer tree first using internal delegate
        if let rootLayer = rootLayer, let delegate = rendererDelegate {
            delegate.render(layer: rootLayer)
        }

        // Process animation completions after rendering
        processAnimationsRecursively(rootLayer)
    }

    // MARK: - Private Methods

    /// Recursively processes animations for all layers in the tree.
    ///
    /// - Parameter layer: The layer to process, along with all its sublayers.
    private func processAnimationsRecursively(_ layer: CALayer?) {
        guard let layer = layer else { return }

        // Process this layer's animations
        layer.processAnimationCompletions()

        // Process sublayers
        if let sublayers = layer.sublayers {
            for sublayer in sublayers {
                processAnimationsRecursively(sublayer)
            }
        }
    }

    // MARK: - Manual Rendering

    /// Manually triggers a single render frame.
    ///
    /// Use this method when you need to render without the animation loop running,
    /// for example, after making changes that should be immediately visible.
    public func renderFrame() {
        if let rootLayer = rootLayer, let delegate = rendererDelegate {
            delegate.render(layer: rootLayer)
        }
        processAnimationsRecursively(rootLayer)
    }
}
