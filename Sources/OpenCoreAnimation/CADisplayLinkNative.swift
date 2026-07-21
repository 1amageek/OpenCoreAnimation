#if !arch(wasm32)

import Foundation

/// A placeholder type for Selector on native platforms (for testing).
///
/// This library uses `CADisplayLinkDelegate` protocol instead of Objective-C selectors.
/// This type exists for API compatibility with the WASM version and CoreAnimation.
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
/// This native implementation uses `Timer` for testing purposes.
/// On Apple platforms in production, use `QuartzCore.CADisplayLink` directly.
@MainActor open class CADisplayLink {

    // MARK: - Properties

    /// The time interval between screen refresh updates.
    open private(set) var duration: CFTimeInterval = 1.0 / 60.0

    /// The time value associated with the last frame that was displayed.
    open private(set) var timestamp: CFTimeInterval = 0

    /// The time value associated with the next frame that was displayed.
    open private(set) var targetTimestamp: CFTimeInterval = 0

    /// A Boolean value that indicates whether the system suspends the display link's notifications to the target.
    open var isPaused: Bool = false {
        didSet {
            if isPaused {
                stopTimer()
            } else if isRunning {
                startTimer()
            }
        }
    }

    /// The preferred frame rate for the display link callback.
    open var preferredFrameRateRange: CAFrameRateRange = CAFrameRateRange() {
        didSet {
            // Restart timer with new frame rate if running
            if isRunning && !isPaused {
                startTimer()
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
    private var registrations: [Registration: RunLoop] = [:]
    private var isInvalidated = false
    private var timer: Timer?

    private var isRunning: Bool {
        !isInvalidated && !registrations.isEmpty
    }

    internal var _registrationCount: Int { registrations.count }
    internal var _hasTarget: Bool { target != nil }
    internal var _isInvalidated: Bool { isInvalidated }

    private var callbackInterval: CFTimeInterval {
        if let frameRate = preferredFrameRateRange.effectiveFrameRate {
            return 1.0 / CFTimeInterval(frameRate)
        }
        return duration
    }

    // MARK: - Initialization

    /// Creates a display link with the target and selector you specify.
    ///
    /// - Parameters:
    ///   - target: The object to which the selector message is sent.
    ///   - sel: The selector to call on the target.
    public init(target: Any, selector sel: Selector) {
        self.target = target as AnyObject
        self.selector = sel
    }

    isolated deinit {
        timer?.invalidate()
    }

    // MARK: - Scheduling

    /// Registers the display link with a run loop.
    ///
    /// - Parameters:
    ///   - runloop: The run loop to add the display link to.
    ///   - mode: The run loop mode.
    open func add(to runloop: RunLoop, forMode mode: RunLoop.Mode) {
        guard !isInvalidated else { return }
        let registration = Registration(
            runLoopID: ObjectIdentifier(runloop),
            mode: mode
        )
        guard registrations[registration] == nil else { return }
        registrations[registration] = runloop
        if !isPaused {
            startTimer()
        }
    }

    /// Removes the display link from all run loop modes.
    open func invalidate() {
        guard !isInvalidated else { return }
        isInvalidated = true
        registrations.removeAll(keepingCapacity: false)
        stopTimer()
        target = nil
    }

    /// Removes the display link from the run loop for the given mode.
    ///
    /// - Parameters:
    ///   - runloop: The run loop to remove from.
    ///   - mode: The run loop mode to remove.
    open func remove(from runloop: RunLoop, forMode mode: RunLoop.Mode) {
        guard !isInvalidated else { return }
        registrations.removeValue(forKey: Registration(
            runLoopID: ObjectIdentifier(runloop),
            mode: mode
        ))
        if registrations.isEmpty || isPaused {
            stopTimer()
        } else {
            startTimer()
        }
    }

    // MARK: - Private Methods

    private func startTimer() {
        stopTimer()

        guard isRunning else { return }
        let interval = callbackInterval
        let newTimer = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                guard self.isRunning && !self.isPaused else { return }

                self.timestamp = CACurrentMediaTime()
                self.targetTimestamp = self.timestamp + self.duration

                if let delegate = self.target as? CADisplayLinkDelegate {
                    delegate.displayLinkDidFire(self)
                }
            }
        }
        timer = newTimer

        for (registration, runLoop) in registrations {
            runLoop.add(newTimer, forMode: registration.mode)
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

#endif
