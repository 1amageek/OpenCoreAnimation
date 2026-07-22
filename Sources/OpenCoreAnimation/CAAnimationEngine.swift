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
@MainActor public final class CAAnimationEngine: CADisplayLinkDelegate {

    // MARK: - Singleton

    /// The shared animation engine instance.
    ///
    /// Use this shared instance for the main application animation loop.
    public static let shared = CAAnimationEngine()

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

    /// The range currently submitted to the display scheduler.
    internal var displayLinkFrameRateRange: CAFrameRateRange? {
        displayLink?.preferredFrameRateRange
    }

    /// Whether the engine is currently running.
    public private(set) var isRunning: Bool = false

    /// The preferred frame rate for animations.
    public var preferredFrameRate: Float = 60 {
        didSet {
            updateDisplayLinkFrameRate(at: CACurrentMediaTime())
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

    #if arch(wasm32)
    // MARK: - WASM Configuration

    /// Sets the canvas element for WebGPU rendering.
    ///
    /// This method must be called before starting the animation loop on WASM.
    /// It initializes the WebGPU renderer with the provided canvas element.
    ///
    /// - Parameter canvas: The HTML canvas element to render to.
    /// - Throws: `CARendererError` if renderer initialization fails.
    @MainActor public func setCanvas(_ canvas: JavaScriptKit.JSObject) async throws {
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
        updateDisplayLinkFrameRate(at: CACurrentMediaTime())
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
        updateDisplayLinkFrameRate(at: CACurrentMediaTime())

        // Render the layer tree first using internal delegate
        if let rootLayer = rootLayer, let delegate = rendererDelegate {
            layoutRecursively(rootLayer)
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

    /// Resolves the highest-demand timing hints among animations active in the tree.
    ///
    /// The browser ultimately controls its physical refresh rate. This arbitration
    /// only selects the callback range requested from `CADisplayLink`.
    internal func resolvedFrameRateRange(at mediaTime: CFTimeInterval) -> CAFrameRateRange {
        var accumulator = FrameRateRangeAccumulator()
        collectActiveFrameRateRanges(
            from: rootLayer,
            mediaTime: mediaTime,
            into: &accumulator
        )

        guard accumulator.hasExplicitRange else {
            return CAFrameRateRange(
                minimum: preferredFrameRate,
                maximum: preferredFrameRate,
                preferred: preferredFrameRate
            )
        }

        return accumulator.resolvedRange
    }

    /// Incrementally combines animation hints without allocating on each frame.
    private struct FrameRateRangeAccumulator {
        private(set) var hasExplicitRange = false
        private var minimum: Float = 0
        private var maximum: Float = 0
        private var preferred: Float?

        mutating func include(_ range: CAFrameRateRange) {
            hasExplicitRange = true
            minimum = max(minimum, range.minimum)
            maximum = max(maximum, range.maximum)
            if let candidate = range.preferred, candidate > 0 {
                preferred = max(preferred ?? 0, candidate)
            }
        }

        var resolvedRange: CAFrameRateRange {
            let normalizedMaximum = max(minimum, max(maximum, preferred ?? 0))
            let normalizedPreferred = preferred.map {
                min(normalizedMaximum, max(minimum, $0))
            }
            return CAFrameRateRange(
                minimum: minimum,
                maximum: normalizedMaximum,
                preferred: normalizedPreferred
            )
        }
    }

    private func collectActiveFrameRateRanges(
        from layer: CALayer?,
        mediaTime: CFTimeInterval,
        into accumulator: inout FrameRateRangeAccumulator
    ) {
        guard let layer else { return }
        let localTime = layer.convertTime(mediaTime, from: nil)

        layer.forEachAttachedAnimation { animation in
            collectActiveFrameRateRanges(
                from: animation,
                parentTime: localTime,
                inheritedDuration: nil,
                into: &accumulator
            )
        }

        for sublayer in layer.sublayers ?? [] {
            collectActiveFrameRateRanges(
                from: sublayer,
                mediaTime: mediaTime,
                into: &accumulator
            )
        }
    }

    private func collectActiveFrameRateRanges(
        from animation: CAAnimation,
        parentTime: CFTimeInterval,
        inheritedDuration: CFTimeInterval?,
        into accumulator: inout FrameRateRangeAccumulator
    ) {
        guard !animation.isFinished else { return }
        let duration = animation.duration > 0
            ? animation.duration
            : inheritedDuration ?? animation.effectiveBaseDuration
        let timing = CAMediaTimingEvaluator.evaluate(
            animation,
            parentTime: parentTime,
            duration: duration
        )
        guard timing.phase == .active else { return }

        if animation.preferredFrameRateRange != .default {
            accumulator.include(animation.preferredFrameRateRange)
        }
        if let group = animation as? CAAnimationGroup {
            let childTime: CFTimeInterval
            if let timingFunction = group.timingFunction {
                let easedProgress = timingFunction.evaluate(at: Float(timing.progress))
                childTime = CFTimeInterval(max(0, min(1, easedProgress))) * duration
            } else {
                childTime = timing.basicTime
            }
            for child in group.animations ?? [] {
                collectActiveFrameRateRanges(
                    from: child,
                    parentTime: childTime,
                    inheritedDuration: duration,
                    into: &accumulator
                )
            }
        }
    }

    private func updateDisplayLinkFrameRate(at mediaTime: CFTimeInterval) {
        let range = resolvedFrameRateRange(at: mediaTime)
        if displayLink?.preferredFrameRateRange != range {
            displayLink?.preferredFrameRateRange = range
        }
    }

    /// Resolves pending layout from parent to child before presentation values
    /// are captured by the renderer.
    private func layoutRecursively(_ layer: CALayer) {
        layer.layoutIfNeeded()
        for sublayer in layer.sublayers ?? [] {
            layoutRecursively(sublayer)
        }
    }

    // MARK: - Manual Rendering

    /// Manually triggers a single render frame.
    ///
    /// Use this method when you need to render without the animation loop running,
    /// for example, after making changes that should be immediately visible.
    public func renderFrame() {
        if let rootLayer = rootLayer, let delegate = rendererDelegate {
            layoutRecursively(rootLayer)
            delegate.render(layer: rootLayer)
        }
        processAnimationsRecursively(rootLayer)
    }
}
