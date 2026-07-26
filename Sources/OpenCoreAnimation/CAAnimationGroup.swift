// CAAnimationGroup.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework


/// An object that allows multiple animations to be grouped and run concurrently.
open class CAAnimationGroup: CAAnimation {

    /// An array of CAAnimation objects to be evaluated concurrently.
    open var animations: [CAAnimation]? {
        didSet {
            oldValue?.forEach { $0.detachFromLayer() }
            attachmentDidChange(attachmentReference)
            notifyAttachedLayerOfMutation()
        }
    }

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

    internal override func attachmentDidChange(
        _ reference: CAAnimationLayerReference?
    ) {
        for animation in animations ?? [] {
            if let reference {
                animation.attach(using: reference)
            } else {
                animation.detachFromLayer()
            }
        }
    }

    open override func shouldArchiveValue(forKey key: String) -> Bool {
        switch key {
        case "animations":
            return animations != nil
        default:
            return super.shouldArchiveValue(forKey: key)
        }
    }
}
