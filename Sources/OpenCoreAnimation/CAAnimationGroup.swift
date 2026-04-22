// CAAnimationGroup.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework


/// An object that allows multiple animations to be grouped and run concurrently.
open class CAAnimationGroup: CAAnimation {

    /// An array of CAAnimation objects to be evaluated concurrently.
    open var animations: [CAAnimation]?

    public required init() {
        super.init()
    }

    public required init(animation: CAAnimation) {
        super.init(animation: animation)
        if let source = animation as? CAAnimationGroup {
            // Deep-copy nested animations so mutation of originals does not
            // propagate into the grouped copy.
            self.animations = source.animations?.map { $0.copy() }
        }
    }
}
