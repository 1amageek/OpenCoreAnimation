import Foundation
import Synchronization

/// A cross-platform recursive lock used by the split `CATransaction.lock()` /
/// `unlock()` API.
///
/// `Mutex.withLock` remains the synchronization primitive for owner metadata.
/// The unit-valued gate is held across the public API boundary because the
/// Core Animation contract deliberately separates acquisition and release.
internal final class CATransactionRecursiveLock: Sendable {
    private struct State: Sendable {
        var owner: ObjectIdentifier?
        var recursionDepth = 0
    }

    private let gate = Mutex<Void>(())
    private let state = Mutex(State())

    internal func lock() {
        let caller = ObjectIdentifier(Thread.current)
        let isRecursiveAcquisition = state.withLock { state in
            guard state.owner == caller else { return false }
            state.recursionDepth += 1
            return true
        }
        guard !isRecursiveAcquisition else { return }

        gate._unsafeLock()
        state.withLock { state in
            precondition(
                state.owner == nil && state.recursionDepth == 0,
                "Transaction lock ownership must be empty after gate acquisition"
            )
            state.owner = caller
            state.recursionDepth = 1
        }
    }

    internal func unlock() {
        let caller = ObjectIdentifier(Thread.current)
        let releasesGate = state.withLock { state in
            precondition(
                state.owner == caller && state.recursionDepth > 0,
                "CATransaction.unlock() must be called by the lock owner"
            )
            state.recursionDepth -= 1
            guard state.recursionDepth == 0 else { return false }
            state.owner = nil
            return true
        }
        if releasesGate {
            gate._unsafeUnlock()
        }
    }
}
