
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

    /// Explicit animations added while this transaction level is active.
    var pendingAnimations: [CAAnimation] = []

    /// Layers whose committed model state must reach a renderer submission.
    ///
    /// This is independent of `pendingChanges`: mutations with disabled
    /// actions, hierarchy edits, masks, and display invalidations still
    /// participate in the transaction's render-completion contract.
    var mutatedLayers: [ObjectIdentifier: CALayer] = [:]

    /// Completion coordinators created by nested transaction levels.
    var deferredCompletionCoordinators: [CATransactionCompletionCoordinator] = []

    /// Render-submission obligations created by nested transaction levels.
    var deferredRenderCommits: [CATransactionRenderCommit] = []
}

/// Tracks the animations associated with one transaction completion block.
internal final class CATransactionCompletionCoordinator {
    private let block: () -> Void
    private var remainingAnimationCount = 0
    private var remainingRenderSubmissionCount = 0
    private var registeredRenderRoots: Set<ObjectIdentifier> = []
    private var isSealed = false
    private var didComplete = false

    internal init(block: @escaping () -> Void) {
        self.block = block
    }

    internal func registerAnimation() {
        guard !isSealed, !didComplete else { return }
        remainingAnimationCount += 1
    }

    internal func animationCompleted() {
        guard !didComplete, remainingAnimationCount > 0 else { return }
        remainingAnimationCount -= 1
        completeIfReady()
    }

    @discardableResult
    internal func registerRenderSubmission(for root: CALayer) -> Bool {
        guard !isSealed, !didComplete else { return false }
        guard registeredRenderRoots.insert(ObjectIdentifier(root)).inserted else {
            return false
        }
        remainingRenderSubmissionCount += 1
        return true
    }

    internal func renderSubmitted() {
        guard !didComplete, remainingRenderSubmissionCount > 0 else { return }
        remainingRenderSubmissionCount -= 1
        completeIfReady()
    }

    internal func seal() {
        guard !isSealed else { return }
        isSealed = true
        completeIfReady()
    }

    private func completeIfReady() {
        guard isSealed,
              remainingAnimationCount == 0,
              remainingRenderSubmissionCount == 0,
              !didComplete else {
            return
        }
        didComplete = true
        block()
    }
}

/// Associates one transaction completion with the exact model roots mutated
/// inside that transaction level.
private struct CATransactionRenderCommit {
    let coordinator: CATransactionCompletionCoordinator
    let layers: [CALayer]
}

/// Thread-local transaction stack storage.
///
/// Each thread has its own transaction stack, following CoreAnimation's behavior.
/// WASM is single-threaded, so only one stack exists in practice.
private final class CATransactionStack {
    /// Stack of transaction levels. The last element is the current (innermost) transaction.
    var levels: [CATransactionLevel] = []

    /// Whether an implicit commit has been scheduled for this thread
    var implicitCommitScheduled: Bool = false

    /// Number of browser scheduling failures observed by this transaction stack.
    var implicitCommitSchedulingFailureCount = 0

    /// Most recent browser scheduling failure, cleared by a valid new schedule.
    var lastImplicitCommitSchedulingFailure: CATransactionSchedulingFailure?

    /// Separates cancelled requests from the currently active scheduling generation.
    var implicitCommitGeneration: UInt64 = 0

    /// Completion coordinators associated with the change currently being applied.
    var applyingCompletionCoordinators: [CATransactionCompletionCoordinator] = []

    /// Prevents mutations produced by a custom action during commit from
    /// opening a second implicit transaction.
    var isApplyingChange = false

    #if arch(wasm32)
    /// The one-shot callback and validated browser handle for the active commit request.
    var implicitCommitClosure: JSOneshotClosure?
    var implicitCommitTimerIdentifier: Double?

    /// Retains callbacks that the browser accepted without returning a safely cancellable handle.
    var uncancellableImplicitCommitClosures: [UInt64: JSOneshotClosure] = [:]
    #endif

    init() {}

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

    /// Transaction completion blocks waiting for this change's animation.
    let completionCoordinators: [CATransactionCompletionCoordinator]
}

private extension CATransactionChange {
    func addingCompletionCoordinator(
        _ coordinator: CATransactionCompletionCoordinator
    ) -> CATransactionChange {
        CATransactionChange(
            layer: layer,
            keyPath: keyPath,
            oldValue: oldValue,
            newValue: newValue,
            capturedDuration: capturedDuration,
            capturedTimingFunction: capturedTimingFunction,
            capturedDisableActions: capturedDisableActions,
            completionCoordinators: completionCoordinators.merging([coordinator])
        )
    }
}

private extension Array where Element == CATransactionCompletionCoordinator {
    func merging(_ other: [CATransactionCompletionCoordinator]) -> Self {
        var result = self
        for coordinator in other where !result.contains(where: { $0 === coordinator }) {
            result.append(coordinator)
        }
        return result
    }
}

/// A mechanism for grouping multiple layer-tree operations into atomic updates to the render tree.
public class CATransaction {
    #if !arch(wasm32)
    private static let transactionLock = NSRecursiveLock()
    #endif

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
        var level = stack.levels.removeLast()
        let ownCoordinator = level.completionBlock.map(CATransactionCompletionCoordinator.init)
        var renderCommits = level.deferredRenderCommits
        if let ownCoordinator, !level.mutatedLayers.isEmpty {
            renderCommits.append(CATransactionRenderCommit(
                coordinator: ownCoordinator,
                layers: Array(level.mutatedLayers.values)
            ))
        }

        if let ownCoordinator {
            for animation in level.pendingAnimations {
                animation.attachCompletionCoordinator(ownCoordinator)
            }
            level.pendingChanges = level.pendingChanges.mapValues {
                $0.addingCompletionCoordinator(ownCoordinator)
            }
        }

        var coordinatorsToSeal = level.deferredCompletionCoordinators
        if let ownCoordinator {
            coordinatorsToSeal.append(ownCoordinator)
        }

        if stack.levels.isEmpty {
            // This was the outermost transaction - apply all changes now
            cancelImplicitCommitSchedule(stack)

            // Process changes one at a time because applyChange() might trigger
            // new property changes (via custom CAAction implementations)
            var remainingChanges = level.pendingChanges
            while !remainingChanges.isEmpty {
                guard let (key, change) = remainingChanges.first else { break }
                remainingChanges.removeValue(forKey: key)
                stack.applyingCompletionCoordinators = change.completionCoordinators
                stack.isApplyingChange = true
                applyChange(change)
                stack.isApplyingChange = false
                stack.applyingCompletionCoordinators = []
            }

            for renderCommit in renderCommits {
                enqueueRenderCommit(renderCommit)
            }
            for coordinator in coordinatorsToSeal {
                coordinator.seal()
            }
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
                        capturedDisableActions: change.capturedDisableActions,
                        completionCoordinators: existingChange.completionCoordinators.merging(
                            change.completionCoordinators
                        )
                    )
                } else {
                    // No existing change - just copy it to outer
                    stack.levels[outerIndex].pendingChanges[key] = change
                }
            }
            stack.levels[outerIndex].pendingAnimations.append(contentsOf: level.pendingAnimations)
            stack.levels[outerIndex].mutatedLayers.merge(level.mutatedLayers) {
                current, _ in current
            }
            stack.levels[outerIndex].deferredCompletionCoordinators.append(contentsOf: coordinatorsToSeal)
            stack.levels[outerIndex].deferredRenderCommits.append(contentsOf: renderCommits)
            scheduleImplicitCommit()
        }
    }

    /// Queues one completion coordinator on every distinct render-tree root
    /// affected by the committed transaction.
    private class func enqueueRenderCommit(_ renderCommit: CATransactionRenderCommit) {
        var roots: [ObjectIdentifier: CALayer] = [:]
        for layer in renderCommit.layers {
            let root = layer.transactionRenderRoot
            roots[ObjectIdentifier(root)] = root
        }
        for root in roots.values {
            if renderCommit.coordinator.registerRenderSubmission(for: root) {
                root.enqueueTransactionCompletionAfterRender(renderCommit.coordinator)
            }
        }
    }

    /// Commit all changes made during the current transaction while acquiring the appropriate locks.
    ///
    /// This method commits any extant implicit transaction and
    /// flushes any pending drawing to the screen.
    public class func flush() {
        let stack = getCurrentTransactionStack()

        cancelImplicitCommitSchedule(stack)

        // Commit all transaction levels from innermost to outermost
        while !stack.levels.isEmpty {
            commit()
        }

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
        // Presentation layers are render-time snapshots. Updating their
        // backing values must never enqueue a model-tree transaction.
        guard !layer._isPresentationLayer else { return }

        let stack = getCurrentTransactionStack()

        // Create an implicit transaction if none exists
        if stack.levels.isEmpty {
            beginImplicit()
        }

        guard let currentLevel = stack.levels.last else { return }
        let levelIndex = stack.levels.count - 1
        stack.levels[levelIndex].mutatedLayers[ObjectIdentifier(layer)] = layer

        // Capture current transaction settings
        let capturedDuration = currentLevel.animationDuration
        let capturedTimingFunction = currentLevel.animationTimingFunction
        let capturedDisableActions = currentLevel.disableActions

        // Create a unique key for coalescing: use layer's ObjectIdentifier and keyPath
        let layerID = ObjectIdentifier(layer)
        let changeKey = "\(layerID):\(keyPath)"

        // If there's already a change for this layer+keyPath, preserve the original oldValue
        let existingChange = currentLevel.pendingChanges[changeKey]
        let effectiveOldValue: Any?
        if let existingChange {
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
            capturedDisableActions: capturedDisableActions,
            completionCoordinators: existingChange?.completionCoordinators ?? []
        )
        scheduleImplicitCommit()
    }

    /// Records a model mutation even when it does not resolve a layer action.
    ///
    /// `markDirty(_:)` is the common entry point for hierarchy, mask, display,
    /// and ordinary property changes, so tracking here prevents completion
    /// blocks from running before those changes have reached a GPU submission.
    internal class func registerMutation(layer: CALayer) {
        guard !layer._isPresentationLayer else { return }
        let stack = getCurrentTransactionStack()

        // Changes produced while applying an already-committing action belong
        // to the transaction currently being drained. Associate custom-action
        // mutations with that transaction without opening a second implicit
        // transaction.
        if stack.isApplyingChange {
            let root = layer.transactionRenderRoot
            for coordinator in stack.applyingCompletionCoordinators {
                if coordinator.registerRenderSubmission(for: root) {
                    root.enqueueTransactionCompletionAfterRender(coordinator)
                }
            }
            return
        }

        // Generic dirty marks supplement an existing transaction. Ordinary
        // animatable setters call `registerChange`, which owns implicit
        // transaction creation. A hierarchy/mask/display mutation made with
        // no transaction has no completion coordinator to track and must not
        // open an unrelated transaction scope.
        guard !stack.levels.isEmpty else { return }
        let levelIndex = stack.levels.count - 1
        stack.levels[levelIndex].mutatedLayers[ObjectIdentifier(layer)] = layer
        scheduleImplicitCommit()
    }

    /// Associates an animation with the transaction that added it.
    internal class func registerAnimation(_ animation: CAAnimation) {
        let stack = getCurrentTransactionStack()
        if !stack.applyingCompletionCoordinators.isEmpty {
            for coordinator in stack.applyingCompletionCoordinators {
                animation.attachCompletionCoordinator(coordinator)
            }
            return
        }

        guard !stack.levels.isEmpty else { return }
        stack.levels[stack.levels.count - 1].pendingAnimations.append(animation)
    }

    /// Begins an implicit transaction.
    /// Implicit transactions are automatically committed at the end of the current run loop iteration.
    private class func beginImplicit() {
        let stack = getCurrentTransactionStack()

        var newLevel = CATransactionLevel()
        newLevel.isImplicitTransaction = true
        stack.levels.append(newLevel)
    }

    /// Schedules the implicit transaction to be committed.
    private class func scheduleImplicitCommit() {
        let stack = getCurrentTransactionStack()

        // Explicit-only transaction stacks are committed by their caller.
        guard stack.levels.contains(where: \.isImplicitTransaction) else { return }

        // Don't schedule if already scheduled
        guard !stack.implicitCommitScheduled else { return }
        #if arch(wasm32)
        guard let setTimeout = JSObject.global.setTimeout.function else {
            recordImplicitCommitSchedulingFailure(.setTimeoutUnavailable, on: stack)
            return
        }

        stack.implicitCommitGeneration &+= 1
        let generation = stack.implicitCommitGeneration
        let callback = JSOneshotClosure { _ in
            CATransaction.handleImplicitCommitCallback(generation: generation)
            return .undefined
        }
        let result = setTimeout(this: JSObject.global, callback, 0)

        switch CATransactionBrowserTimerIdentifierValidator.identifier(result.number) {
        case .success(let identifier):
            stack.implicitCommitClosure = callback
            stack.implicitCommitTimerIdentifier = identifier
            stack.implicitCommitScheduled = true
            stack.lastImplicitCommitSchedulingFailure = nil
        case .failure(let failure):
            // The host may still invoke the callback even though it did not return
            // a safely cancellable handle, so retain it until delivery.
            stack.uncancellableImplicitCommitClosures[generation] = callback
            recordImplicitCommitSchedulingFailure(failure, on: stack)
        }
        #else
        stack.implicitCommitGeneration &+= 1
        let generation = stack.implicitCommitGeneration
        stack.implicitCommitScheduled = true
        stack.lastImplicitCommitSchedulingFailure = nil
        // Schedule on the main actor so implicit transactions commit after
        // the current synchronous mutation batch without using GCD.
        Task { @MainActor in
            CATransaction.handleImplicitCommitCallback(generation: generation)
        }
        #endif
    }

    /// Commits the implicit transaction.
    private class func commitImplicit() {
        let stack = getCurrentTransactionStack()

        // Commit all implicit transaction levels
        while let level = stack.levels.last, level.isImplicitTransaction {
            commit()
        }
    }

    private class func handleImplicitCommitCallback(generation: UInt64) {
        let stack = getCurrentTransactionStack()
        #if arch(wasm32)
        stack.uncancellableImplicitCommitClosures.removeValue(forKey: generation)
        #endif

        guard generation == stack.implicitCommitGeneration,
              stack.implicitCommitScheduled else {
            return
        }

        stack.implicitCommitScheduled = false
        #if arch(wasm32)
        stack.implicitCommitTimerIdentifier = nil
        stack.implicitCommitClosure = nil
        #endif
        commitImplicit()
    }

    private class func cancelImplicitCommitSchedule(_ stack: CATransactionStack) {
        guard stack.implicitCommitScheduled else { return }

        let stoppedGeneration = stack.implicitCommitGeneration
        stack.implicitCommitGeneration &+= 1

        #if arch(wasm32)
        if let identifier = stack.implicitCommitTimerIdentifier,
           let callback = stack.implicitCommitClosure {
            if let clearTimeout = JSObject.global.clearTimeout.function {
                _ = clearTimeout(this: JSObject.global, identifier)
                callback.release()
            } else {
                stack.uncancellableImplicitCommitClosures[stoppedGeneration] = callback
                recordImplicitCommitSchedulingFailure(
                    .clearTimeoutUnavailable(identifier: identifier),
                    on: stack
                )
            }
        }
        stack.implicitCommitTimerIdentifier = nil
        stack.implicitCommitClosure = nil
        #endif

        stack.implicitCommitScheduled = false
    }

    private class func recordImplicitCommitSchedulingFailure(
        _ failure: CATransactionSchedulingFailure,
        on stack: CATransactionStack
    ) {
        stack.implicitCommitSchedulingFailureCount += 1
        stack.lastImplicitCommitSchedulingFailure = failure
    }

    /// Number of browser implicit-commit scheduling failures on the current transaction stack.
    @_spi(RendererDiagnostics)
    public static var implicitCommitSchedulingFailureCount: Int {
        getCurrentTransactionStack().implicitCommitSchedulingFailureCount
    }

    /// Most recent browser implicit-commit scheduling failure on the current transaction stack.
    @_spi(RendererDiagnostics)
    public static var lastImplicitCommitSchedulingFailure: CATransactionSchedulingFailure? {
        getCurrentTransactionStack().lastImplicitCommitSchedulingFailure
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
        scheduleImplicitCommit()
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
        scheduleImplicitCommit()
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
        scheduleImplicitCommit()
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
        scheduleImplicitCommit()
    }

    // MARK: - Lock Management

    /// Attempts to acquire a recursive spin-lock lock, ensuring that returned
    /// layer values are valid until unlocked.
    ///
    /// On WASM this is a no-op because the JavaScript host is single-threaded.
    /// Native callers receive the recursive exclusion required by nested
    /// Core Animation transaction operations.
    public class func lock() {
        #if !arch(wasm32)
        transactionLock.lock()
        #endif
    }

    /// Relinquishes a previously acquired transaction lock.
    ///
    /// On WASM this is a no-op because the JavaScript host is single-threaded.
    public class func unlock() {
        #if !arch(wasm32)
        transactionLock.unlock()
        #endif
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
        scheduleImplicitCommit()
    }
}
