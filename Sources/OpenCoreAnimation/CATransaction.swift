
#if arch(wasm32)
import JavaScriptKit
#else
import Foundation
#endif

/// Transaction state storage.
///
/// WASM is single-threaded, so no locks are needed.
private final class CATransactionState: @unchecked Sendable {
    static let shared = CATransactionState()

    var animationDuration: CFTimeInterval = 0.25
    var disableActions: Bool = false
    var animationTimingFunction: CAMediaTimingFunction?
    var completionBlock: (() -> Void)?
    var transactionDepth: Int = 0

    /// Whether the current transaction is implicit (auto-created)
    var isImplicitTransaction: Bool = false

    /// Whether an implicit commit has been scheduled
    var implicitCommitScheduled: Bool = false

    /// Pending layer changes to be flushed
    var pendingChanges: [CATransactionChange] = []

    private init() {}
}

/// Represents a pending change in a transaction.
private struct CATransactionChange {
    let layer: CALayer
    let keyPath: String
    let oldValue: Any?
    let newValue: Any?
}

/// A mechanism for grouping multiple layer-tree operations into atomic updates to the render tree.
public class CATransaction {

    /// Begin a new transaction for the current thread.
    public class func begin() {
        CATransactionState.shared.transactionDepth += 1
    }

    /// Commit all changes made during the current transaction.
    public class func commit() {
        let state = CATransactionState.shared
        guard state.transactionDepth > 0 else { return }
        state.transactionDepth -= 1

        if state.transactionDepth == 0 {
            // Process pending changes
            let changes = state.pendingChanges
            state.pendingChanges.removeAll()

            // Apply any pending implicit animations
            for change in changes {
                applyChange(change)
            }

            // Execute completion block
            let completion = state.completionBlock
            state.completionBlock = nil

            // Reset transaction state for next transaction
            state.animationDuration = 0.25
            state.disableActions = false
            state.animationTimingFunction = nil
            state.isImplicitTransaction = false
            state.implicitCommitScheduled = false

            completion?()
        }
    }

    /// Commit all changes made during the current transaction while acquiring the appropriate locks.
    ///
    /// This method commits any extant implicit transaction and
    /// flushes any pending drawing to the screen.
    public class func flush() {
        let state = CATransactionState.shared

        // If there's an implicit transaction, commit it
        while state.transactionDepth > 0 {
            state.transactionDepth -= 1
        }

        // Process all pending changes immediately
        let changes = state.pendingChanges
        state.pendingChanges.removeAll()

        for change in changes {
            applyChange(change)
        }

        // Execute completion block
        let completion = state.completionBlock
        state.completionBlock = nil

        // Reset state
        state.animationDuration = 0.25
        state.disableActions = false
        state.animationTimingFunction = nil

        completion?()
    }

    /// Internal method to apply a change and trigger implicit animations.
    private class func applyChange(_ change: CATransactionChange) {
        let state = CATransactionState.shared

        // Skip if actions are disabled
        guard !state.disableActions else { return }

        // Get the action for this property change
        guard let action = change.layer.action(forKey: change.keyPath) else { return }

        // Run the action with the change context
        var arguments: [AnyHashable: Any] = [:]
        if let oldValue = change.oldValue {
            arguments["previousValue"] = oldValue
        }
        if let newValue = change.newValue {
            arguments["newValue"] = newValue
        }

        action.run(forKey: change.keyPath, object: change.layer, arguments: arguments)
    }

    /// Internal method to register a pending change.
    /// If no explicit transaction is active, creates an implicit transaction.
    internal class func registerChange(layer: CALayer, keyPath: String, oldValue: Any?, newValue: Any?) {
        let state = CATransactionState.shared

        // If actions are disabled, don't register the change
        if state.disableActions {
            return
        }

        // Create an implicit transaction if none exists
        if state.transactionDepth == 0 {
            beginImplicit()
        }

        state.pendingChanges.append(CATransactionChange(
            layer: layer,
            keyPath: keyPath,
            oldValue: oldValue,
            newValue: newValue
        ))
    }

    /// Begins an implicit transaction.
    /// Implicit transactions are automatically committed at the end of the current run loop iteration.
    private class func beginImplicit() {
        let state = CATransactionState.shared
        state.transactionDepth += 1
        state.isImplicitTransaction = true

        // Schedule implicit commit for the end of the run loop
        scheduleImplicitCommit()
    }

    /// Schedules the implicit transaction to be committed.
    private class func scheduleImplicitCommit() {
        let state = CATransactionState.shared

        // Don't schedule if already scheduled
        guard !state.implicitCommitScheduled else { return }
        state.implicitCommitScheduled = true

        #if arch(wasm32)
        // In WASM, use setTimeout to schedule commit at end of current event loop
        let callback = JSClosure { _ in
            CATransaction.commitImplicit()
            return .undefined
        }
        _ = JSObject.global.setTimeout!(callback, 0)
        #else
        // For native platforms, use DispatchQueue
        DispatchQueue.main.async {
            CATransaction.commitImplicit()
        }
        #endif
    }

    /// Commits the implicit transaction.
    private class func commitImplicit() {
        let state = CATransactionState.shared

        // Reset scheduled flag
        state.implicitCommitScheduled = false

        guard state.isImplicitTransaction && state.transactionDepth > 0 else { return }

        state.isImplicitTransaction = false
        commit()
    }

    // MARK: - Animation Duration

    /// Returns the animation duration used by all animations within the transaction group.
    public class func animationDuration() -> CFTimeInterval {
        return CATransactionState.shared.animationDuration
    }

    /// Sets the animation duration used by all animations within the transaction group.
    public class func setAnimationDuration(_ dur: CFTimeInterval) {
        CATransactionState.shared.animationDuration = dur
    }

    // MARK: - Animation Timing Function

    /// Returns the timing function used for all animations within the transaction group.
    public class func animationTimingFunction() -> CAMediaTimingFunction? {
        return CATransactionState.shared.animationTimingFunction
    }

    /// Sets the timing function used for all animations within the transaction group.
    public class func setAnimationTimingFunction(_ function: CAMediaTimingFunction?) {
        CATransactionState.shared.animationTimingFunction = function
    }

    // MARK: - Disable Actions

    /// Returns whether actions triggered as a result of property changes made within the transaction group are suppressed.
    public class func disableActions() -> Bool {
        return CATransactionState.shared.disableActions
    }

    /// Sets whether actions triggered as a result of property changes made within the transaction group are suppressed.
    public class func setDisableActions(_ flag: Bool) {
        CATransactionState.shared.disableActions = flag
    }

    // MARK: - Completion Block

    /// Returns the completion block associated with the transaction group.
    public class func completionBlock() -> (() -> Void)? {
        return CATransactionState.shared.completionBlock
    }

    /// Sets the completion block associated with the transaction group.
    public class func setCompletionBlock(_ block: (() -> Void)?) {
        CATransactionState.shared.completionBlock = block
    }

    // MARK: - Lock Management

    /// Attempts to acquire a recursive spin-lock lock, ensuring that returned
    /// layer values are valid until unlocked.
    ///
    /// Note: WASM is single-threaded, so this is a no-op.
    public class func lock() {
        // No-op in WASM
    }

    /// Relinquishes a previously acquired transaction lock.
    ///
    /// Note: WASM is single-threaded, so this is a no-op.
    public class func unlock() {
        // No-op in WASM
    }

    // MARK: - Value Access

    /// Returns the value for a given transaction property key.
    public class func value(forKey key: String) -> Any? {
        let state = CATransactionState.shared
        switch key {
        case "animationDuration":
            return state.animationDuration
        case "disableActions":
            return state.disableActions
        case "animationTimingFunction":
            return state.animationTimingFunction
        case "completionBlock":
            return state.completionBlock
        default:
            return nil
        }
    }

    /// Sets the value for a given transaction property key.
    public class func setValue(_ anObject: Any?, forKey key: String) {
        let state = CATransactionState.shared
        switch key {
        case "animationDuration":
            if let duration = anObject as? CFTimeInterval {
                state.animationDuration = duration
            }
        case "disableActions":
            if let flag = anObject as? Bool {
                state.disableActions = flag
            }
        case "animationTimingFunction":
            state.animationTimingFunction = anObject as? CAMediaTimingFunction
        case "completionBlock":
            state.completionBlock = anObject as? (() -> Void)
        default:
            break
        }
    }
}
