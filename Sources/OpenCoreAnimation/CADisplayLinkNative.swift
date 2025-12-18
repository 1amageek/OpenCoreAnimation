#if !arch(wasm32)

import Foundation

/// A timer object that allows your application to synchronize its drawing to the refresh rate of the display.
///
/// This native implementation uses `Timer` for testing purposes.
/// On Apple platforms in production, use `QuartzCore.CADisplayLink` directly.
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
                stopTimer()
            } else if isRunning {
                startTimer()
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
    private var timer: Timer?

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

    deinit {
        invalidate()
    }

    // MARK: - Scheduling

    /// Registers the display link with a run loop.
    ///
    /// - Parameters:
    ///   - runloop: The run loop to add the display link to.
    ///   - mode: The run loop mode.
    open func add(to runloop: AnyObject, forMode mode: AnyObject) {
        guard !isRunning else { return }
        isRunning = true
        if !isPaused {
            startTimer()
        }
    }

    /// Removes the display link from all run loop modes.
    open func invalidate() {
        isRunning = false
        stopTimer()
    }

    /// Removes the display link from the run loop for the given mode.
    ///
    /// - Parameters:
    ///   - runloop: The run loop to remove from.
    ///   - mode: The run loop mode to remove.
    open func remove(from runloop: AnyObject, forMode mode: AnyObject) {
        invalidate()
    }

    // MARK: - Private Methods

    private func startTimer() {
        stopTimer()

        let interval = duration
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            guard self.isRunning && !self.isPaused else { return }

            self.timestamp = CACurrentMediaTime()
            self.targetTimestamp = self.timestamp + self.duration

            if let delegate = self.target as? CADisplayLinkDelegate {
                delegate.displayLinkDidFire(self)
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

#endif
