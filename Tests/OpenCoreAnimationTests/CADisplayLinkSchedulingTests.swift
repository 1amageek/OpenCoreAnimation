import Testing
@testable import OpenCoreAnimation

@Suite("CADisplayLink scheduling")
@MainActor
struct CADisplayLinkSchedulingTests {
    private final class NoopTarget: CADisplayLinkDelegate {
        func displayLinkDidFire(_ displayLink: CADisplayLink) {}
    }

    @Test("Frame-rate range defaults to native refresh delivery")
    func defaultRangeIsZeroed() {
        let target = NoopTarget()
        let link = CADisplayLink(target: target, selector: Selector(""))

        #expect(link.preferredFrameRateRange == .default)
        #expect(CAFrameRateRange.default.minimum == 0)
        #expect(CAFrameRateRange.default.maximum == 0)
        #expect(CAFrameRateRange.default.preferred == nil)
    }

    @Test("Duration remains the display refresh interval")
    func preferredRangeDoesNotRewriteDuration() {
        let target = NoopTarget()
        let link = CADisplayLink(target: target, selector: Selector(""))
        let refreshDuration = link.duration

        link.preferredFrameRateRange = CAFrameRateRange(
            minimum: 30,
            maximum: 120,
            preferred: 30
        )

        #expect(link.duration == refreshDuration)
    }

    @Test("Frame-rate resolution rejects non-finite rates and respects bounds")
    func effectiveFrameRate() {
        #expect(CAFrameRateRange(
            minimum: 30,
            maximum: 60,
            preferred: 120
        ).effectiveFrameRate == 60)
        #expect(CAFrameRateRange(
            minimum: 30,
            maximum: 60,
            preferred: 10
        ).effectiveFrameRate == 30)
        #expect(CAFrameRateRange(
            minimum: .nan,
            maximum: 60,
            preferred: .infinity
        ).effectiveFrameRate == 60)
        #expect(CAFrameRateRange(
            minimum: -.infinity,
            maximum: .nan,
            preferred: -.infinity
        ).effectiveFrameRate == nil)
    }

    @Test("Mode registrations are removed independently")
    func modeRegistrationsAreIndependent() {
        let target = NoopTarget()
        let link = CADisplayLink(target: target, selector: Selector(""))

        link.add(to: .main, forMode: .default)
        link.add(to: .main, forMode: .common)
        link.add(to: .main, forMode: .common)
        #expect(link._registrationCount == 2)

        link.remove(from: .main, forMode: .default)
        #expect(link._registrationCount == 1)

        link.remove(
            from: .main,
            forMode: RunLoop.Mode(rawValue: "unregistered-mode")
        )
        #expect(link._registrationCount == 1)

        link.remove(from: .main, forMode: .common)
        #expect(link._registrationCount == 0)

        link.add(to: .main, forMode: .common)
        #expect(link._registrationCount == 1)
        link.invalidate()
    }

    @Test("Invalidate clears registrations and releases the target")
    func invalidateIsTerminalAndReleasesTarget() {
        weak var weakTarget: NoopTarget?
        let link: CADisplayLink
        do {
            let target = NoopTarget()
            weakTarget = target
            link = CADisplayLink(target: target, selector: Selector(""))
        }
        link.add(to: .main, forMode: .common)

        #expect(weakTarget != nil)
        #expect(link._hasTarget)

        link.invalidate()
        #expect(link._registrationCount == 0)
        #expect(link._isInvalidated)
        #expect(!link._hasTarget)
        #expect(weakTarget == nil)

        link.add(to: .main, forMode: .common)
        #expect(link._registrationCount == 0)
    }

    #if !arch(wasm32)
    @Test("Native callbacks expose maximum and selected frame intervals")
    func nativeCallbackTiming() {
        final class Target: CADisplayLinkDelegate {
            var samples: [(timestamp: CFTimeInterval, target: CFTimeInterval, duration: CFTimeInterval)] = []

            func displayLinkDidFire(_ displayLink: CADisplayLink) {
                samples.append((
                    displayLink.timestamp,
                    displayLink.targetTimestamp,
                    displayLink.duration
                ))
            }
        }

        let target = Target()
        let link = CADisplayLink(target: target, selector: Selector(""))
        #expect(link.timestamp == 0)
        #expect(link.targetTimestamp == 0)
        #expect(link.duration == 0)

        link.preferredFrameRateRange = CAFrameRateRange(
            minimum: 30,
            maximum: 60,
            preferred: 30
        )
        link.add(to: .main, forMode: .default)
        runMainLoop(for: 0.12)
        link.invalidate()

        #expect(target.samples.count >= 2)
        for sample in target.samples {
            #expect(sample.timestamp > 0)
            #expect(abs(sample.duration - 1.0 / 60.0) < 0.000_001)
            #expect(abs((sample.target - sample.timestamp) - 1.0 / 30.0) < 0.000_001)
        }
        for pair in zip(target.samples, target.samples.dropFirst()) {
            #expect(pair.1.timestamp > pair.0.timestamp)
        }
    }

    @Test("Pause, resume, and independent modes preserve callback lifetime")
    func nativePauseResumeAndModes() {
        final class Target: CADisplayLinkDelegate {
            var callbackCount = 0
            func displayLinkDidFire(_ displayLink: CADisplayLink) {
                callbackCount += 1
            }
        }

        let target = Target()
        let link = CADisplayLink(target: target, selector: Selector(""))
        link.add(to: .main, forMode: .default)
        link.add(to: .main, forMode: .common)
        runMainLoop(for: 0.06)
        let runningCount = target.callbackCount
        #expect(runningCount > 0)

        link.isPaused = true
        runMainLoop(for: 0.05)
        #expect(target.callbackCount == runningCount)

        link.isPaused = false
        runMainLoop(for: 0.06)
        let resumedCount = target.callbackCount
        #expect(resumedCount > runningCount)

        link.remove(from: .main, forMode: .default)
        runMainLoop(for: 0.05)
        let retainedModeCount = target.callbackCount
        #expect(retainedModeCount > resumedCount)

        link.remove(from: .main, forMode: .common)
        runMainLoop(for: 0.05)
        #expect(target.callbackCount == retainedModeCount)
        link.invalidate()
    }

    private func runMainLoop(for interval: CFTimeInterval) {
        let limit = Date(timeIntervalSinceNow: interval)
        while Date() < limit {
            _ = RunLoop.main.run(
                mode: .default,
                before: min(limit, Date(timeIntervalSinceNow: 0.01))
            )
        }
    }
    #endif
}
