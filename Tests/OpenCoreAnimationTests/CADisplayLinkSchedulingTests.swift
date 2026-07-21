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
}
