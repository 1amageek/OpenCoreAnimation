// Performance test infrastructure shared by every Performance/* suite.
//
// All Performance/* suites mutate the global `CALayer._currentFrameToken`
// and other process-wide counters. swift-testing parallelises by default,
// so each suite must be `@Suite(.serialized)` AND must reset the globals
// in `init()`. See PERFORMANCE_DESIGN.md §10.
//
// If you add a new Performance suite, run the whole tag with
// `swift test --filter Performance --no-parallel` (or `xcodebuild test
// -only-testing:OpenCoreAnimationTests -parallel-testing-enabled NO`).

import Testing
@testable import OpenCoreAnimation

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
