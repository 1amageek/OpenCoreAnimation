#if arch(wasm32)
import JavaScriptKit

/// A placeholder type for Selector on WASM.
///
/// On WASM, there is no Objective-C runtime, so Selector cannot function as it does on Apple platforms.
/// This type exists for API compatibility. The actual callback mechanism uses `CADisplayLinkDelegate`.
public struct Selector: Hashable, ExpressibleByStringLiteral, Sendable {
    public var description: String

    public init(_ string: String) {
        self.description = string
    }

    public init(stringLiteral value: String) {
        self.description = value
    }
}

/// A timer object that allows your application to synchronize its drawing to the refresh rate of the display.
///
/// This WASM implementation uses JavaScript's `requestAnimationFrame` API to synchronize with
/// the browser's display refresh rate.
///
/// ## Usage
///
/// On WASM, your target object must conform to `CADisplayLinkDelegate`:
///
/// ```swift
/// class MyAnimator: CADisplayLinkDelegate {
///     lazy var displayLink = CADisplayLink(target: self, selector: Selector(""))
///
///     func start() {
///         displayLink.add(to: AnyObject.self, forMode: AnyObject.self)
///     }
///
///     func displayLinkDidFire(_ displayLink: CADisplayLink) {
///         // Update animation here
///     }
/// }
/// ```
open class CADisplayLink: @unchecked Sendable {

    // MARK: - Properties

    /// The time interval between screen refresh updates.
    open var duration: CFTimeInterval {
        if preferredFrameRateRange.preferred > 0 {
            return 1.0 / CFTimeInterval(preferredFrameRateRange.preferred)
        }
        return 1.0 / 60.0
    }

    /// The time value associated with the last frame that was displayed.
    open private(set) var timestamp: CFTimeInterval = 0

    /// The time value associated with the next frame that was displayed.
    open private(set) var targetTimestamp: CFTimeInterval = 0

    /// A Boolean value that indicates whether the system suspends the display link's notifications to the target.
    open var isPaused: Bool = false {
        didSet {
            if isPaused {
                stopAnimationLoop()
            } else if isRunning {
                startAnimationLoop()
            }
        }
    }

    /// The preferred frame rate for the display link callback.
    open var preferredFrameRateRange: CAFrameRateRange = CAFrameRateRange()

    /// The preferred frame rate in frames per second.
    open var preferredFramesPerSecond: Int = 0 {
        didSet {
            if preferredFramesPerSecond > 0 {
                preferredFrameRateRange = CAFrameRateRange(
                    minimum: Float(preferredFramesPerSecond),
                    maximum: Float(preferredFramesPerSecond),
                    preferred: Float(preferredFramesPerSecond)
                )
            }
        }
    }

    // MARK: - Private Properties

    private weak var target: AnyObject?
    private var selector: Selector
    private var isRunning: Bool = false
    private var animationFrameCallback: JSClosure?
    private var animationFrameId: Int32 = 0

    /// The timestamp of the last frame that was dispatched to the delegate.
    private var lastDispatchedTimestamp: CFTimeInterval = 0

    /// The minimum time interval between frame dispatches based on preferred frame rate.
    private var minimumFrameInterval: CFTimeInterval {
        if preferredFrameRateRange.maximum > 0 {
            return 1.0 / CFTimeInterval(preferredFrameRateRange.maximum)
        } else if preferredFrameRateRange.preferred > 0 {
            return 1.0 / CFTimeInterval(preferredFrameRateRange.preferred)
        } else if preferredFramesPerSecond > 0 {
            return 1.0 / CFTimeInterval(preferredFramesPerSecond)
        }
        return 0 // No throttling
    }

    // MARK: - Initialization

    /// Creates a display link with the target and selector you specify.
    ///
    /// - Parameters:
    ///   - target: An object that conforms to `CADisplayLinkDelegate`. The delegate method will be called on each frame.
    ///   - sel: A selector. On WASM, this parameter is ignored; use `CADisplayLinkDelegate` instead.
    public init(target: Any, selector sel: Selector) {
        self.target = target as AnyObject
        self.selector = sel
    }

    deinit {
        invalidate()
    }

    // MARK: - Scheduling

    /// Registers the display link with a run loop.
    ///
    /// On WASM, the run loop parameters are ignored. This method starts the `requestAnimationFrame` loop.
    ///
    /// - Parameters:
    ///   - runloop: Ignored on WASM.
    ///   - mode: Ignored on WASM.
    open func add(to runloop: AnyObject, forMode mode: AnyObject) {
        guard !isRunning else { return }
        isRunning = true
        if !isPaused {
            startAnimationLoop()
        }
    }

    /// Removes the display link from all run loop modes.
    ///
    /// This stops the `requestAnimationFrame` loop and releases resources.
    open func invalidate() {
        isRunning = false
        stopAnimationLoop()
    }

    /// Removes the display link from the run loop for the given mode.
    ///
    /// On WASM, this is equivalent to calling `invalidate()`.
    ///
    /// - Parameters:
    ///   - runloop: Ignored on WASM.
    ///   - mode: Ignored on WASM.
    open func remove(from runloop: AnyObject, forMode mode: AnyObject) {
        invalidate()
    }

    // MARK: - Private Methods

    private func startAnimationLoop() {
        stopAnimationLoop()

        animationFrameCallback = JSClosure { [weak self] arguments in
            guard let self = self else { return .undefined }
            guard self.isRunning && !self.isPaused else { return .undefined }

            // Update timestamps (convert from milliseconds to seconds)
            let timestampMs = arguments[0].number ?? 0
            let currentTimestamp = timestampMs / 1000.0
            self.timestamp = currentTimestamp
            self.targetTimestamp = currentTimestamp + self.duration

            // Check if we should dispatch this frame based on frame rate throttling
            let minInterval = self.minimumFrameInterval
            let shouldDispatch: Bool

            if minInterval > 0 {
                // Frame rate throttling is enabled
                let timeSinceLastDispatch = currentTimestamp - self.lastDispatchedTimestamp
                shouldDispatch = timeSinceLastDispatch >= minInterval
            } else {
                // No throttling, dispatch every frame
                shouldDispatch = true
            }

            if shouldDispatch {
                self.lastDispatchedTimestamp = currentTimestamp

                // Call the delegate method
                if let delegate = self.target as? CADisplayLinkDelegate {
                    delegate.displayLinkDidFire(self)
                }
            }

            // Request next frame if still running (always request to maintain accurate timing)
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
        animationFrameId = Int32(result.number ?? 0)
    }

    private func stopAnimationLoop() {
        if animationFrameId != 0 {
            _ = JSObject.global.cancelAnimationFrame!(animationFrameId)
            animationFrameId = 0
        }
        // Release the closure to prevent memory leaks
        animationFrameCallback?.release()
        animationFrameCallback = nil
    }
}

#endif
