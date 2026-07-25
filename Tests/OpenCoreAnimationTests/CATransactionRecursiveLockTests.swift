import Foundation
import Synchronization
import Testing
@testable import OpenCoreAnimation

@Suite("CATransaction recursive locking", .serialized)
struct CATransactionRecursiveLockTests {
    private struct ContentionState: Sendable {
        var didStartContending = false
        var didAcquire = false
        var didFinish = false
    }

    @Test("The owner can acquire recursively and a contender waits for final unlock")
    func recursiveAcquisitionAndContention() throws {
        let state = Mutex(ContentionState())

        CATransaction.lock()
        CATransaction.lock()

        Thread.detachNewThread {
            state.withLock { $0.didStartContending = true }
            CATransaction.lock()
            state.withLock { $0.didAcquire = true }
            CATransaction.unlock()
            state.withLock { $0.didFinish = true }
        }

        try waitUntil {
            state.withLock { $0.didStartContending }
        }
        #expect(state.withLock { !$0.didAcquire })

        CATransaction.unlock()
        #expect(state.withLock { !$0.didAcquire })

        CATransaction.unlock()
        try waitUntil {
            state.withLock { $0.didFinish }
        }
        #expect(state.withLock { $0.didAcquire })
    }

    private func waitUntil(
        _ condition: @escaping @Sendable () -> Bool
    ) throws {
        let deadline = ContinuousClock.now + .seconds(2)
        while !condition() {
            guard ContinuousClock.now < deadline else {
                throw LockTestError.timedOut
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }
}

private enum LockTestError: Error {
    case timedOut
}
