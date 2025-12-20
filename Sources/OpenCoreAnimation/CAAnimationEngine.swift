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
/// engine.renderer = myRenderer
/// engine.start()
/// ```
///
/// The engine automatically handles the display refresh cycle and ensures
/// animations are properly updated each frame.
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

    /// The renderer to use for drawing.
    ///
    /// Set this to an appropriate renderer for your platform:
    /// - `CAWebGPURenderer` for WASM/Web
    /// - `CAMetalRenderer` for macOS/iOS
    public var renderer: CARenderer?

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
    public init() {}

    deinit {
        stop()
    }

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
    /// 1. Processes animation completions for all layers
    /// 2. Renders the layer tree using presentation layer values
    public func displayLinkDidFire(_ displayLink: CADisplayLink) {
        // Process animation completions for all layers in the tree
        processAnimationsRecursively(rootLayer)

        // Render the layer tree
        if let rootLayer = rootLayer, let renderer = renderer {
            renderer.render(layer: rootLayer)
        }
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
        processAnimationsRecursively(rootLayer)
        if let rootLayer = rootLayer, let renderer = renderer {
            renderer.render(layer: rootLayer)
        }
    }
}
