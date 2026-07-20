// Performance test infrastructure shared by every Performance/* suite.
//
// All Performance/* suites mutate the global `CALayer._currentFrameToken`
// and other process-wide counters. Swift Testing parallelizes independent
// suites by default, so every performance suite is nested under the single
// serialized parent below. Suite-local serialization is insufficient because
// sibling suites can otherwise reset the same globals concurrently.

import Testing
@testable import OpenCoreAnimation

@Suite(.serialized)
struct PerformanceTests {}

/// Reset every piece of global state Performance/* tests touch.
/// Called from each suite's `init()` so tests start from a known baseline.
func resetPerformanceTestState() {
    CALayer._currentFrameToken = 0
}

extension CALayer {
    /// Test-only escape hatch: clear dirty state without going through the
    /// renderer. Used by Phase 1 tests to drive clean→dirty→clean cycles.
    func _testClearDirty() {
        recursivelyClearDirtyAfterCommit()
    }
}
