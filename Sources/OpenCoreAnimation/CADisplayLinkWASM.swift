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
///         displayLink.add(to: .main, forMode: .common)
///     }
///
///     func displayLinkDidFire(_ displayLink: CADisplayLink) {
///         // Update animation here
///     }
/// }
/// ```
@MainActor open class CADisplayLink {

    // MARK: - Properties

    /// The time interval between screen refresh updates.
    open private(set) var duration: CFTimeInterval = 0

    /// The time value associated with the last frame that was displayed.
    open private(set) var timestamp: CFTimeInterval = 0

    /// The time value associated with the next frame that was displayed.
    open private(set) var targetTimestamp: CFTimeInterval = 0

    /// The number of browser scheduling failures observed by this display link.
    @_spi(RendererDiagnostics)
    public private(set) var schedulingFailureCount = 0

    /// The most recent browser scheduling failure, cleared when a new loop starts.
    @_spi(RendererDiagnostics)
    public private(set) var lastSchedulingFailure: CADisplayLinkSchedulingFailure?

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
    ///
    /// Unlike the native implementation, we do not stop/start the rAF loop
    /// when this changes — the next browser frame picks up the new throttling
    /// interval automatically. We simply reset the last-dispatched timestamp
    /// so the new interval applies cleanly.
    open var preferredFrameRateRange: CAFrameRateRange = CAFrameRateRange() {
        didSet {
            if isRunning {
                lastDispatchedTimestamp = 0
                hasDispatchedFrame = false
            }
        }
    }

    // MARK: - Private Properties

    private struct Registration: Hashable {
        let runLoopID: ObjectIdentifier
        let mode: RunLoop.Mode
    }

    private var target: AnyObject?
    private var selector: Selector
    private var registrations: [Registration] = []
    private var isInvalidated = false
    private var pendingFrameCallback: JSOneshotClosure?
    private var animationFrameId: UInt32?
    private var animationLoopGeneration: UInt64 = 0
    private var uncancellableFrameCallbacks: [UInt64: JSOneshotClosure] = [:]

    private var isRunning: Bool {
        !isInvalidated && !registrations.isEmpty
    }

    internal var _registrationCount: Int { registrations.count }
    internal var _hasTarget: Bool { target != nil }
    internal var _isInvalidated: Bool { isInvalidated }

    /// The timestamp of the last frame that was dispatched to the delegate.
    private var lastDispatchedTimestamp: CFTimeInterval = 0

    /// Distinguishes a real first frame from a browser timestamp of exactly zero.
    private var hasDispatchedFrame = false

    /// Browser refresh timestamp used to measure the physical rAF cadence.
    private var previousRefreshTimestamp: CFTimeInterval?

    /// Sliding samples estimate the fastest current physical refresh interval.
    /// Taking the minimum prevents a delayed callback from redefining the
    /// display's maximum refresh rate, while the bounded window permits a
    /// runtime refresh-rate change to converge.
    private var refreshIntervalSamples: [CFTimeInterval] = []

    private let nominalRefreshInterval: CFTimeInterval = 1.0 / 60.0

    /// The minimum time interval between frame dispatches based on preferred frame rate.
    private var minimumFrameInterval: CFTimeInterval {
        if let frameRate = preferredFrameRateRange.effectiveFrameRate {
            return 1.0 / CFTimeInterval(frameRate)
        }
        return 0
    }

    /// Resolves the requested callback interval to a whole number of physical
    /// refreshes, matching display-link factor selection.
    private var callbackInterval: CFTimeInterval {
        let refreshInterval = duration > 0 ? duration : nominalRefreshInterval
        let requestedInterval = minimumFrameInterval
        guard requestedInterval > refreshInterval else { return refreshInterval }
        let refreshCount = max(1, Int((requestedInterval / refreshInterval).rounded()))
        return refreshInterval * CFTimeInterval(refreshCount)
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

    // MARK: - Scheduling

    /// Registers the display link with a run loop.
    ///
    /// Browser scheduling uses one event loop, but registrations remain distinct so
    /// removing one mode does not invalidate registrations in other modes.
    ///
    /// - Parameters:
    ///   - runloop: Identifies the browser run-loop registration.
    ///   - mode: Identifies an independent registration lifetime.
    open func add(to runloop: RunLoop, forMode mode: RunLoop.Mode) {
        guard !isInvalidated else { return }
        let wasRunning = isRunning
        let registration = Registration(
            runLoopID: ObjectIdentifier(runloop),
            mode: mode
        )
        if !registrations.contains(registration) {
            registrations.append(registration)
        }
        if !wasRunning && isRunning && !isPaused {
            startAnimationLoop()
        }
    }

    /// Removes the display link from all run loop modes.
    ///
    /// This stops the `requestAnimationFrame` loop and releases resources.
    open func invalidate() {
        guard !isInvalidated else { return }
        isInvalidated = true
        registrations.removeAll(keepingCapacity: false)
        stopAnimationLoop()
        target = nil
    }

    /// Removes the display link from the run loop for the given mode.
    ///
    /// - Parameters:
    ///   - runloop: Identifies the browser run-loop registration.
    ///   - mode: Identifies the registration to remove.
    open func remove(from runloop: RunLoop, forMode mode: RunLoop.Mode) {
        guard !isInvalidated else { return }
        let registration = Registration(
            runLoopID: ObjectIdentifier(runloop),
            mode: mode
        )
        registrations.removeAll { $0 == registration }
        if registrations.isEmpty {
            stopAnimationLoop()
        }
    }

    // MARK: - Private Methods

    private func startAnimationLoop() {
        stopAnimationLoop()
        lastSchedulingFailure = nil
        requestNextFrame()
    }

    private func handleAnimationFrame(
        arguments: [JSValue],
        generation: UInt64
    ) {
        uncancellableFrameCallbacks.removeValue(forKey: generation)

        guard generation == animationLoopGeneration else {
            return
        }
        pendingFrameCallback = nil
        animationFrameId = nil
        guard lastSchedulingFailure == nil else { return }

        switch CADisplayLinkBrowserValueValidator.timestamp(
            milliseconds: arguments.first?.number
        ) {
        case .success(let timestamp):
            handleAnimationFrame(timestamp: timestamp)
        case .failure(let failure):
            recordSchedulingFailure(failure)
        }
    }

    private func handleAnimationFrame(timestamp currentTimestamp: CFTimeInterval) {
        guard isRunning && !isPaused else { return }

        if let previousRefreshTimestamp {
            let measuredDuration = currentTimestamp - previousRefreshTimestamp
            if measuredDuration.isFinite, measuredDuration > 0 {
                refreshIntervalSamples.append(measuredDuration)
                if refreshIntervalSamples.count > 8 {
                    refreshIntervalSamples.removeFirst(
                        refreshIntervalSamples.count - 8
                    )
                }
                if let fastestInterval = refreshIntervalSamples.min() {
                    duration = fastestInterval
                }
            }
        } else {
            duration = nominalRefreshInterval
        }
        previousRefreshTimestamp = currentTimestamp

        let resolvedCallbackInterval = callbackInterval
        let tolerance = min(duration * 0.25, resolvedCallbackInterval * 0.1)
        let shouldDispatch = !hasDispatchedFrame
            || currentTimestamp - lastDispatchedTimestamp
                >= resolvedCallbackInterval - tolerance

        if shouldDispatch {
            hasDispatchedFrame = true
            lastDispatchedTimestamp = currentTimestamp
            timestamp = currentTimestamp
            targetTimestamp = currentTimestamp + resolvedCallbackInterval
            if let delegate = target as? CADisplayLinkDelegate {
                delegate.displayLinkDidFire(self)
            }
        }

        if isRunning && !isPaused {
            requestNextFrame()
        }
    }

    private func requestNextFrame() {
        guard pendingFrameCallback == nil else { return }
        guard let requestAnimationFrame = JSObject.global.requestAnimationFrame.function else {
            recordSchedulingFailure(.requestAnimationFrameUnavailable)
            return
        }

        let generation = animationLoopGeneration
        let callback = JSOneshotClosure { [weak self] arguments in
            MainActor.assumeIsolated {
                self?.handleAnimationFrame(arguments: arguments, generation: generation)
            }
            return .undefined
        }
        pendingFrameCallback = callback
        let result = requestAnimationFrame(callback)

        switch CADisplayLinkBrowserValueValidator.requestIdentifier(result.number) {
        case .success(let identifier):
            animationFrameId = identifier
        case .failure(let failure):
            pendingFrameCallback = nil
            uncancellableFrameCallbacks[generation] = callback
            recordSchedulingFailure(failure)
        }
    }

    private func stopAnimationLoop() {
        let stoppedGeneration = animationLoopGeneration
        animationLoopGeneration &+= 1

        if let animationFrameId, let callback = pendingFrameCallback {
            if let cancelAnimationFrame = JSObject.global.cancelAnimationFrame.function {
                _ = cancelAnimationFrame(animationFrameId)
                callback.release()
            } else {
                uncancellableFrameCallbacks[stoppedGeneration] = callback
                recordSchedulingFailure(
                    .cancelAnimationFrameUnavailable(identifier: animationFrameId)
                )
            }
        }
        animationFrameId = nil
        pendingFrameCallback = nil
        previousRefreshTimestamp = nil
        refreshIntervalSamples.removeAll(keepingCapacity: true)
        lastDispatchedTimestamp = 0
        hasDispatchedFrame = false
    }

    private func recordSchedulingFailure(_ failure: CADisplayLinkSchedulingFailure) {
        schedulingFailureCount += 1
        lastSchedulingFailure = failure
    }
}

#endif
