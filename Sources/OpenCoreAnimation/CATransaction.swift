
import Foundation
#if arch(wasm32)
import JavaScriptKit
#endif

/// Represents the state of a single transaction level.
///
/// Each `begin()` creates a new `CATransactionLevel` that is pushed onto the stack.
/// Properties set within this transaction level are stored here and restored on `commit()`.
private struct CATransactionLevel {
    var animationDuration: CFTimeInterval = 0.25
    var disableActions: Bool = false
    var animationTimingFunction: CAMediaTimingFunction?
    var completionBlock: (() -> Void)?

    /// Whether this transaction level is implicit (auto-created)
    var isImplicitTransaction: Bool = false

    /// Pending layer changes for this transaction level.
    /// Key is "layerObjectID:keyPath" to enable coalescing.
    var pendingChanges: [String: CATransactionChange] = [:]
}

/// Thread-local transaction stack storage.
///
/// Each thread has its own transaction stack, following CoreAnimation's behavior.
/// WASM is single-threaded, so only one stack exists in practice.
private final class CATransactionStack: @unchecked Sendable {
    /// Stack of transaction levels. The last element is the current (innermost) transaction.
    var levels: [CATransactionLevel] = []

    /// Whether an implicit commit has been scheduled for this thread
    var implicitCommitScheduled: Bool = false

    #if arch(wasm32)
    /// The JSClosure used for scheduling implicit commits.
    /// Retained here to prevent premature garbage collection.
    var implicitCommitClosure: JSClosure?
    #endif

    init() {}

    deinit {
        #if arch(wasm32)
        // Release the closure when the stack is deallocated
        implicitCommitClosure?.release()
        #endif
    }
}

#if arch(wasm32)
/// WASM is single-threaded, so we use a single global stack.
nonisolated(unsafe) private var _wasmTransactionStack = CATransactionStack()

private func getCurrentTransactionStack() -> CATransactionStack {
    return _wasmTransactionStack
}
#else
/// Key for thread-local storage of transaction stack.
private let transactionStackKey = "CATransactionStack"

private func getCurrentTransactionStack() -> CATransactionStack {
    let threadDict = Thread.current.threadDictionary
    if let stack = threadDict[transactionStackKey] as? CATransactionStack {
        return stack
    }
    let newStack = CATransactionStack()
    threadDict[transactionStackKey] = newStack
    return newStack
}
#endif

/// Represents a pending change in a transaction.
///
/// Transaction settings are captured at registration time, following CoreAnimation behavior.
/// This ensures that the animation uses the settings that were in effect when the property
/// was changed, not the settings at commit time.
private struct CATransactionChange {
    let layer: CALayer
    let keyPath: String
    let oldValue: Any?
    let newValue: Any?

    /// The animation duration captured at registration time.
    let capturedDuration: CFTimeInterval

    /// The timing function captured at registration time.
    let capturedTimingFunction: CAMediaTimingFunction?

    /// Whether actions were disabled at registration time.
    let capturedDisableActions: Bool
}

/// A mechanism for grouping multiple layer-tree operations into atomic updates to the render tree.
public class CATransaction {

    /// Begin a new transaction for the current thread.
    ///
    /// Nested transactions are supported. Each `begin()` creates a new transaction level
    /// with its own set of properties (duration, timing function, etc.).
    public class func begin() {
        let stack = getCurrentTransactionStack()

        // Inherit properties from parent transaction if exists
        var newLevel = CATransactionLevel()
        if let currentLevel = stack.levels.last {
            newLevel.animationDuration = currentLevel.animationDuration
            newLevel.disableActions = currentLevel.disableActions
            newLevel.animationTimingFunction = currentLevel.animationTimingFunction
            // Note: completionBlock is NOT inherited - each level has its own
        }

        stack.levels.append(newLevel)
    }

    /// Commit all changes made during the current transaction.
    ///
    /// Following CoreAnimation behavior:
    /// - Nested transactions merge their changes to the outer transaction
    /// - Only the outermost transaction actually applies the animations
    /// - Each change carries its captured settings (duration, timingFunction, disableActions)
    ///   from when the property was changed
    ///
    /// From Apple documentation:
    /// "Only after you commit the changes for the outermost transaction does
    /// Core Animation begin the associated animations."
    public class func commit() {
        let stack = getCurrentTransactionStack()
        guard !stack.levels.isEmpty else { return }

        // Pop the current level
        let level = stack.levels.removeLast()

        if stack.levels.isEmpty {
            // This was the outermost transaction - apply all changes now
            // Process changes one at a time because applyChange() might trigger
            // new property changes (via custom CAAction implementations)
            var remainingChanges = level.pendingChanges
            while !remainingChanges.isEmpty {
                guard let (key, change) = remainingChanges.first else { break }
                remainingChanges.removeValue(forKey: key)
                applyChange(change)
            }

            // Reset the implicit commit flag
            stack.implicitCommitScheduled = false
        } else {
            // This is a nested transaction - merge changes to the outer transaction
            // The outer level is now at stack.levels.count - 1
            let outerIndex = stack.levels.count - 1

            for (key, change) in level.pendingChanges {
                // If outer already has a change for this key, preserve outer's oldValue
                // (the very first oldValue in the chain)
                if let existingChange = stack.levels[outerIndex].pendingChanges[key] {
                    // Keep outer's oldValue but use inner's newValue and captured settings
                    stack.levels[outerIndex].pendingChanges[key] = CATransactionChange(
                        layer: change.layer,
                        keyPath: change.keyPath,
                        oldValue: existingChange.oldValue,
                        newValue: change.newValue,
                        capturedDuration: change.capturedDuration,
                        capturedTimingFunction: change.capturedTimingFunction,
                        capturedDisableActions: change.capturedDisableActions
                    )
                } else {
                    // No existing change - just copy it to outer
                    stack.levels[outerIndex].pendingChanges[key] = change
                }
            }
        }

        // Execute this level's completion block
        level.completionBlock?()
    }

    /// Commit all changes made during the current transaction while acquiring the appropriate locks.
    ///
    /// This method commits any extant implicit transaction and
    /// flushes any pending drawing to the screen.
    public class func flush() {
        let stack = getCurrentTransactionStack()

        // Commit all transaction levels from innermost to outermost
        while !stack.levels.isEmpty {
            commit()
        }

        stack.implicitCommitScheduled = false
    }

    /// Internal method to apply a change and trigger implicit animations.
    ///
    /// Uses the captured settings from the change, not the current transaction settings.
    /// This ensures animations use the settings that were in effect when the property
    /// was changed, regardless of when the outermost transaction commits.
    private class func applyChange(_ change: CATransactionChange) {
        // Skip if actions were disabled when the change was registered
        guard !change.capturedDisableActions else { return }

        // Get the action for this property change
        guard let action = change.layer.action(forKey: change.keyPath) else { return }

        // Run the action with the change context, including captured settings
        var arguments: [AnyHashable: Any] = [:]
        if let oldValue = change.oldValue {
            arguments["previousValue"] = oldValue
        }
        if let newValue = change.newValue {
            arguments["newValue"] = newValue
        }

        // Pass captured transaction settings to the action
        arguments["animationDuration"] = change.capturedDuration
        if let timingFunction = change.capturedTimingFunction {
            arguments["animationTimingFunction"] = timingFunction
        }

        action.run(forKey: change.keyPath, object: change.layer, arguments: arguments)
    }

    /// Internal method to register a pending change.
    /// If no explicit transaction is active, creates an implicit transaction.
    ///
    /// Changes to the same layer+keyPath within a transaction are coalesced,
    /// keeping only the most recent change (with the original oldValue).
    ///
    /// Transaction settings (duration, timingFunction, disableActions) are captured
    /// at registration time and stored with the change, following CoreAnimation behavior.
    internal class func registerChange(layer: CALayer, keyPath: String, oldValue: Any?, newValue: Any?) {
        let stack = getCurrentTransactionStack()

        // Check if actions are disabled in current transaction
        if let currentLevel = stack.levels.last, currentLevel.disableActions {
            return
        }

        // Create an implicit transaction if none exists
        if stack.levels.isEmpty {
            beginImplicit()
        }

        guard let currentLevel = stack.levels.last else { return }
        let levelIndex = stack.levels.count - 1

        // Capture current transaction settings
        let capturedDuration = currentLevel.animationDuration
        let capturedTimingFunction = currentLevel.animationTimingFunction
        let capturedDisableActions = currentLevel.disableActions

        // Create a unique key for coalescing: use layer's ObjectIdentifier and keyPath
        let layerID = ObjectIdentifier(layer)
        let changeKey = "\(layerID):\(keyPath)"

        // If there's already a change for this layer+keyPath, preserve the original oldValue
        let effectiveOldValue: Any?
        if let existingChange = currentLevel.pendingChanges[changeKey] {
            effectiveOldValue = existingChange.oldValue
        } else {
            effectiveOldValue = oldValue
        }

        // Update the pending change with the new value and captured settings
        stack.levels[levelIndex].pendingChanges[changeKey] = CATransactionChange(
            layer: layer,
            keyPath: keyPath,
            oldValue: effectiveOldValue,
            newValue: newValue,
            capturedDuration: capturedDuration,
            capturedTimingFunction: capturedTimingFunction,
            capturedDisableActions: capturedDisableActions
        )
    }

    /// Begins an implicit transaction.
    /// Implicit transactions are automatically committed at the end of the current run loop iteration.
    private class func beginImplicit() {
        let stack = getCurrentTransactionStack()

        var newLevel = CATransactionLevel()
        newLevel.isImplicitTransaction = true
        stack.levels.append(newLevel)

        // Schedule implicit commit for the end of the run loop
        scheduleImplicitCommit()
    }

    /// Schedules the implicit transaction to be committed.
    private class func scheduleImplicitCommit() {
        let stack = getCurrentTransactionStack()

        // Don't schedule if already scheduled
        guard !stack.implicitCommitScheduled else { return }
        stack.implicitCommitScheduled = true

        #if arch(wasm32)
        // Release any previously held closure
        stack.implicitCommitClosure?.release()

        // In WASM, use setTimeout to schedule commit at end of current event loop
        let callback = JSClosure { _ in
            CATransaction.commitImplicit()
            return .undefined
        }

        // Retain the closure to prevent garbage collection before callback fires
        stack.implicitCommitClosure = callback
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
        let stack = getCurrentTransactionStack()

        // Reset scheduled flag
        stack.implicitCommitScheduled = false

        #if arch(wasm32)
        // Release the closure now that the callback has fired
        // This prevents memory leaks from holding onto completed closures
        stack.implicitCommitClosure?.release()
        stack.implicitCommitClosure = nil
        #endif

        // Commit all implicit transaction levels
        while let level = stack.levels.last, level.isImplicitTransaction {
            commit()
        }
    }

    // MARK: - Animation Duration

    /// Returns the animation duration used by all animations within the transaction group.
    public class func animationDuration() -> CFTimeInterval {
        let stack = getCurrentTransactionStack()
        return stack.levels.last?.animationDuration ?? 0.25
    }

    /// Sets the animation duration used by all animations within the transaction group.
    ///
    /// If no transaction is active, an implicit transaction is automatically created.
    public class func setAnimationDuration(_ dur: CFTimeInterval) {
        let stack = getCurrentTransactionStack()
        if stack.levels.isEmpty {
            beginImplicit()
        }
        stack.levels[stack.levels.count - 1].animationDuration = dur
    }

    // MARK: - Animation Timing Function

    /// Returns the timing function used for all animations within the transaction group.
    public class func animationTimingFunction() -> CAMediaTimingFunction? {
        let stack = getCurrentTransactionStack()
        return stack.levels.last?.animationTimingFunction
    }

    /// Sets the timing function used for all animations within the transaction group.
    ///
    /// If no transaction is active, an implicit transaction is automatically created.
    public class func setAnimationTimingFunction(_ function: CAMediaTimingFunction?) {
        let stack = getCurrentTransactionStack()
        if stack.levels.isEmpty {
            beginImplicit()
        }
        stack.levels[stack.levels.count - 1].animationTimingFunction = function
    }

    // MARK: - Disable Actions

    /// Returns whether actions triggered as a result of property changes made within the transaction group are suppressed.
    public class func disableActions() -> Bool {
        let stack = getCurrentTransactionStack()
        return stack.levels.last?.disableActions ?? false
    }

    /// Sets whether actions triggered as a result of property changes made within the transaction group are suppressed.
    ///
    /// If no transaction is active, an implicit transaction is automatically created.
    public class func setDisableActions(_ flag: Bool) {
        let stack = getCurrentTransactionStack()
        if stack.levels.isEmpty {
            beginImplicit()
        }
        stack.levels[stack.levels.count - 1].disableActions = flag
    }

    // MARK: - Completion Block

    /// Returns the completion block associated with the transaction group.
    public class func completionBlock() -> (() -> Void)? {
        let stack = getCurrentTransactionStack()
        return stack.levels.last?.completionBlock
    }

    /// Sets the completion block associated with the transaction group.
    ///
    /// If no transaction is active, an implicit transaction is automatically created.
    public class func setCompletionBlock(_ block: (() -> Void)?) {
        let stack = getCurrentTransactionStack()
        if stack.levels.isEmpty {
            beginImplicit()
        }
        stack.levels[stack.levels.count - 1].completionBlock = block
    }

    // MARK: - Lock Management

    /// Attempts to acquire a recursive spin-lock lock, ensuring that returned
    /// layer values are valid until unlocked.
    ///
    /// Note: In WASM (single-threaded) this is a no-op.
    /// On native platforms, this could be implemented with a recursive lock if needed.
    public class func lock() {
        // No-op - WASM is single-threaded and for native testing we don't need locks
    }

    /// Relinquishes a previously acquired transaction lock.
    ///
    /// Note: In WASM (single-threaded) this is a no-op.
    public class func unlock() {
        // No-op - WASM is single-threaded and for native testing we don't need locks
    }

    // MARK: - Value Access

    /// Returns the value for a given transaction property key.
    public class func value(forKey key: String) -> Any? {
        let stack = getCurrentTransactionStack()
        guard let level = stack.levels.last else { return nil }
        switch key {
        case "animationDuration":
            return level.animationDuration
        case "disableActions":
            return level.disableActions
        case "animationTimingFunction":
            return level.animationTimingFunction
        case "completionBlock":
            return level.completionBlock
        default:
            return nil
        }
    }

    /// Sets the value for a given transaction property key.
    ///
    /// If no transaction is active, an implicit transaction is automatically created.
    public class func setValue(_ anObject: Any?, forKey key: String) {
        let stack = getCurrentTransactionStack()
        if stack.levels.isEmpty {
            beginImplicit()
        }
        switch key {
        case "animationDuration":
            if let duration = anObject as? CFTimeInterval {
                stack.levels[stack.levels.count - 1].animationDuration = duration
            }
        case "disableActions":
            if let flag = anObject as? Bool {
                stack.levels[stack.levels.count - 1].disableActions = flag
            }
        case "animationTimingFunction":
            stack.levels[stack.levels.count - 1].animationTimingFunction = anObject as? CAMediaTimingFunction
        case "completionBlock":
            stack.levels[stack.levels.count - 1].completionBlock = anObject as? (() -> Void)
        default:
            break
        }
    }
}
