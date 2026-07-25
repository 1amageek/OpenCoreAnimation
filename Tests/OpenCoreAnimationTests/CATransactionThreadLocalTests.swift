import Foundation
import Synchronization
import Testing
@testable import OpenCoreAnimation

extension CATransactionTestSuites {
@Suite("Thread-local ownership")
struct ThreadLocalOwnership {
    private struct IsolationState: Sendable {
        var readyCount = 0
        var firstDuration: CFTimeInterval?
        var secondDuration: CFTimeInterval?
        var finishedCount = 0
        var coordinationFailureCount = 0
    }

    private struct RunLoopState: Sendable {
        var owner: ObjectIdentifier?
        var completionThread: ObjectIdentifier?
        var completionCount = 0
        var lifecycleIdentifier: UInt64?
        var finished = false
    }

    private struct ReleaseState: Sendable {
        var lifecycleIdentifier: UInt64?
        var threadFinished = false
    }

    @Test("Concurrent threads retain independent transaction values")
    func concurrentThreadsRetainIndependentValues() throws {
        let state = Mutex(IsolationState())

        Thread.detachNewThread {
            CATransaction.begin()
            CATransaction.setAnimationDuration(0.125)
            state.withLock { $0.readyCount += 1 }
            if !Self.waitForBothThreads(state) {
                state.withLock { $0.coordinationFailureCount += 1 }
            }
            let duration = CATransaction.animationDuration()
            CATransaction.commit()
            state.withLock {
                $0.firstDuration = duration
                $0.finishedCount += 1
            }
        }
        Thread.detachNewThread {
            CATransaction.begin()
            CATransaction.setAnimationDuration(0.875)
            state.withLock { $0.readyCount += 1 }
            if !Self.waitForBothThreads(state) {
                state.withLock { $0.coordinationFailureCount += 1 }
            }
            let duration = CATransaction.animationDuration()
            CATransaction.commit()
            state.withLock {
                $0.secondDuration = duration
                $0.finishedCount += 1
            }
        }

        try waitUntil {
            state.withLock { $0.finishedCount == 2 }
        }
        let result = state.withLock { $0 }
        #expect(result.coordinationFailureCount == 0)
        #expect(result.firstDuration == 0.125)
        #expect(result.secondDuration == 0.875)
    }

    @Test("Implicit commit executes on its owning background thread")
    func implicitCommitExecutesOnOwningThread() throws {
        let state = Mutex(RunLoopState())

        Thread.detachNewThread {
            let owner = ObjectIdentifier(Thread.current)
            state.withLock { $0.owner = owner }
            CATransaction.setCompletionBlock {
                state.withLock {
                    $0.completionThread = ObjectIdentifier(Thread.current)
                    $0.completionCount += 1
                }
            }
            CATransaction.setAnimationDuration(0.375)
            let lifecycleIdentifier =
                CATransaction.transactionStackLifecycleIdentifierForTesting()
            state.withLock {
                $0.lifecycleIdentifier = lifecycleIdentifier
            }
            RunLoop.current.run(
                until: Date(timeIntervalSinceNow: 0.25)
            )
            state.withLock { $0.finished = true }
        }

        try waitUntil {
            state.withLock { $0.finished }
        }
        let result = state.withLock { $0 }
        #expect(result.completionCount == 1)
        #expect(result.completionThread == result.owner)
        let lifecycleIdentifier = try #require(result.lifecycleIdentifier)
        #expect(
            CATransaction.transactionStackReleaseCountForTesting(
                lifecycleIdentifier: lifecycleIdentifier
            ) == 1
        )
    }

    @Test("Thread exit releases an uncommitted TLS stack exactly once")
    func threadExitReleasesStackExactlyOnce() throws {
        let state = Mutex(ReleaseState())

        Thread.detachNewThread {
            CATransaction.begin()
            let identifier =
                CATransaction.transactionStackLifecycleIdentifierForTesting()
            state.withLock {
                $0.lifecycleIdentifier = identifier
                $0.threadFinished = true
            }
        }

        try waitUntil {
            state.withLock { $0.threadFinished }
        }
        let identifier = try #require(
            state.withLock { $0.lifecycleIdentifier }
        )
        try waitUntil {
            CATransaction.transactionStackReleaseCountForTesting(
                lifecycleIdentifier: identifier
            ) == 1
        }
        Thread.sleep(forTimeInterval: 0.01)
        #expect(
            CATransaction.transactionStackReleaseCountForTesting(
                lifecycleIdentifier: identifier
            ) == 1
        )
    }

    private func waitUntil(
        _ condition: @escaping @Sendable () -> Bool
    ) throws {
        let deadline = ContinuousClock.now + .seconds(2)
        while !condition() {
            guard ContinuousClock.now < deadline else {
                throw ThreadLocalTestError.timedOut
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    private static func waitForBothThreads(
        _ state: borrowing Mutex<IsolationState>
    ) -> Bool {
        let deadline = ContinuousClock.now + .seconds(2)
        while !state.withLock({ $0.readyCount == 2 }) {
            guard ContinuousClock.now < deadline else { return false }
            Thread.sleep(forTimeInterval: 0.001)
        }
        return true
    }
}
}

private enum ThreadLocalTestError: Error {
    case timedOut
}
