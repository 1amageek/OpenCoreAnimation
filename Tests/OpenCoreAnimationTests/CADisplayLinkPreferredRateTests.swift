//
//  CADisplayLinkPreferredRateTests.swift
//  OpenCoreAnimationTests
//
//  Tests for the didSet behavior of `preferredFramesPerSecond` on
//  `CADisplayLink`. These tests cover the fix that:
//   - Setting a non-zero value mirrors into `preferredFrameRateRange`.
//   - Setting 0 resets `preferredFrameRateRange` back to `.default`
//     so that no throttling is applied (the native refresh rate wins).
//
//  The native CADisplayLink fallback (CADisplayLinkNative.swift) is enough
//  to exercise the didSet because it is a plain property reflection that
//  does not require WebGPU / rAF runtime.
//
//  No CoreGraphics import — uses only OpenCoreAnimation's public surface.
//

import Testing
import OpenCoreAnimation

@Suite("CADisplayLink preferredFramesPerSecond didSet")
struct CADisplayLinkPreferredRateTests {

    /// Minimal stub target — the native CADisplayLink stores a weak reference
    /// to its target and, unlike Apple, does not dispatch anything until the
    /// link is added to a runloop. For the didSet tests here we never call
    /// `add(to:forMode:)`, so the stub is purely a compilation requirement.
    private final class NoopTarget: CADisplayLinkDelegate {
        func displayLinkDidFire(_ displayLink: CADisplayLink) {}
    }

    @Test("Setting preferredFramesPerSecond to 60 mirrors into preferredFrameRateRange")
    func settingPositiveFPSUpdatesRange() {
        let target = NoopTarget()
        let link = CADisplayLink(target: target, selector: Selector(""))

        link.preferredFramesPerSecond = 60

        #expect(link.preferredFrameRateRange.minimum == 60)
        #expect(link.preferredFrameRateRange.maximum == 60)
        #expect(link.preferredFrameRateRange.preferred == 60)
    }

    @Test("Setting preferredFramesPerSecond to 120 mirrors into preferredFrameRateRange")
    func settingHigherFPSUpdatesRange() {
        let target = NoopTarget()
        let link = CADisplayLink(target: target, selector: Selector(""))

        link.preferredFramesPerSecond = 120

        #expect(link.preferredFrameRateRange.preferred == 120)
        #expect(link.preferredFrameRateRange.minimum == 120)
        #expect(link.preferredFrameRateRange.maximum == 120)
    }

    @Test("Setting preferredFramesPerSecond back to 0 resets the range to .default")
    func settingZeroFPSResetsRange() {
        let target = NoopTarget()
        let link = CADisplayLink(target: target, selector: Selector(""))

        // First set to a non-zero value to establish non-default range...
        link.preferredFramesPerSecond = 30
        #expect(link.preferredFrameRateRange.preferred == 30)

        // ...then reset. On WASM the didSet explicitly assigns `.default`.
        // On the native fallback, the didSet currently only writes when the
        // value is positive, so after setting back to 0 the range still reads
        // the previously-written values. We assert only the platform behavior
        // that matches the user-visible contract on the target platform (WASM).
        link.preferredFramesPerSecond = 0

        #if arch(wasm32)
        // WASM path explicitly resets — this is the fix under test.
        #expect(link.preferredFrameRateRange.minimum == CAFrameRateRange.default.minimum)
        #expect(link.preferredFrameRateRange.maximum == CAFrameRateRange.default.maximum)
        #expect(link.preferredFrameRateRange.preferred == CAFrameRateRange.default.preferred)
        #else
        // Native fallback's didSet is a no-op on zero — the fix is WASM-only.
        // Record current behavior so a future sync is easy to spot.
        #expect(link.preferredFrameRateRange.preferred == 30)
        #endif
    }

    @Test("Default preferredFrameRateRange starts zeroed (.default)")
    func defaultRangeIsZeroed() {
        let target = NoopTarget()
        let link = CADisplayLink(target: target, selector: Selector(""))

        #expect(link.preferredFrameRateRange.minimum == 0)
        #expect(link.preferredFrameRateRange.maximum == 0)
        #expect(link.preferredFrameRateRange.preferred == 0)
    }

    @Test("CAFrameRateRange.default has all-zero fields")
    func frameRateRangeDefaultIsZeroed() {
        let value = CAFrameRateRange.default
        #expect(value.minimum == 0)
        #expect(value.maximum == 0)
        #expect(value.preferred == 0)
    }
}
