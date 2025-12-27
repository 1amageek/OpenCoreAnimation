
/// The abstract superclass for animations in Core Animation.
///
/// CAAnimation provides basic support for the CAMediaTiming and CAAction protocols.
/// You do not create instances of CAAnimation; create instances of one of the concrete subclasses
/// such as CABasicAnimation or CAKeyframeAnimation.
open class CAAnimation: CAMediaTiming, CAAction {

    // MARK: - Initialization

    /// Creates a new animation object.
    public init() {}

    // MARK: - CAMediaTiming

    /// Specifies the begin time of the receiver in relation to its parent object, if applicable.
    open var beginTime: CFTimeInterval = 0

    /// Specifies an additional time offset in active local time.
    open var timeOffset: CFTimeInterval = 0

    /// Determines the number of times the animation will repeat.
    open var repeatCount: Float = 0

    /// Determines how many seconds the animation will repeat for.
    open var repeatDuration: CFTimeInterval = 0

    /// Specifies the basic duration of the animation, in seconds.
    open var duration: CFTimeInterval = 0

    /// Specifies how time is mapped to receiver's time space from the parent time space.
    open var speed: Float = 1

    /// Determines if the receiver plays in the reverse upon completion.
    open var autoreverses: Bool = false

    /// Determines if the receiver's presentation is frozen or removed once its active duration has completed.
    open var fillMode: CAMediaTimingFillMode = .removed

    // MARK: - Animation Properties

    /// The timing function defining the pacing of the animation.
    open var timingFunction: CAMediaTimingFunction?

    /// The animation's delegate object.
    open weak var delegate: (any CAAnimationDelegate)?

    /// Determines if the animation is removed from the target layer's animations upon completion.
    open var isRemovedOnCompletion: Bool = true

    // MARK: - Internal State

    /// The time at which the animation was added to the layer.
    internal var addedTime: CFTimeInterval = 0

    /// Whether the animation has completed.
    internal var isFinished: Bool = false

    /// The layer this animation is attached to (weak reference).
    internal weak var attachedLayer: CALayer?

    /// The key used when this animation was added to the layer.
    internal var animationKey: String?

    /// Calculates the total duration including repeats and autoreverses.
    internal var totalDuration: CFTimeInterval {
        // Get base duration - subclasses may override effectiveBaseDuration
        let baseDuration = effectiveBaseDuration
        var total = baseDuration

        // Note: If both repeatDuration and repeatCount are specified,
        // the behavior is undefined according to Apple docs.
        // We prioritize repeatDuration when set.
        if repeatDuration > 0 {
            total = repeatDuration
        } else if repeatCount > 0 {
            total *= CFTimeInterval(repeatCount)
        }

        if autoreverses {
            total *= 2
        }

        return total
    }

    /// The effective base duration for the animation.
    /// Subclasses can override this to provide custom duration logic (e.g., CASpringAnimation).
    internal var effectiveBaseDuration: CFTimeInterval {
        return duration > 0 ? duration : 0.25
    }

    /// Marks the animation as finished and notifies the delegate.
    internal func markFinished(completed: Bool) {
        guard !isFinished else { return }
        isFinished = true
        delegate?.animationDidStop(self, finished: completed)
    }

    // MARK: - CAAction

    /// Called to trigger the action specified by the identifier.
    ///
    /// When an implicit animation is triggered by changing a layer property, this method
    /// is called to run the animation. The implementation adds the animation to the
    /// target layer with the event as the key.
    ///
    /// - Parameters:
    ///   - event: The action identifier (typically the property key path being animated).
    ///   - anObject: The layer on which the action should run.
    ///   - dict: A dictionary of additional parameters. May contain:
    ///     - `"previousValue"`: The value before the change.
    ///     - `"newValue"`: The new value after the change.
    ///     - `"animationDuration"`: The animation duration captured at registration time.
    ///     - `"animationTimingFunction"`: The timing function captured at registration time.
    open func run(forKey event: String, object anObject: Any, arguments dict: [AnyHashable: Any]?) {
        guard let layer = anObject as? CALayer else { return }

        // For property animations, set up from/to values if not already set
        if let propertyAnimation = self as? CAPropertyAnimation {
            // Set the keyPath if not already set
            if propertyAnimation.keyPath == nil {
                propertyAnimation.keyPath = event
            }

            // For basic animations, set up from/to values from the arguments
            if let basicAnimation = propertyAnimation as? CABasicAnimation {
                if let previousValue = dict?["previousValue"], basicAnimation.fromValue == nil {
                    basicAnimation.fromValue = previousValue
                }
                if let newValue = dict?["newValue"], basicAnimation.toValue == nil {
                    basicAnimation.toValue = newValue
                }
            }
        }

        // Set duration from captured settings (preferred) or fall back to transaction
        // The captured duration ensures we use the settings from when the property was changed,
        // not from when the animation is applied (transaction may have been popped by then)
        if duration == 0 {
            if let capturedDuration = dict?["animationDuration"] as? CFTimeInterval {
                duration = capturedDuration
            } else {
                duration = CATransaction.animationDuration()
            }
        }

        // Set timing function from captured settings or transaction
        if timingFunction == nil {
            if let capturedTiming = dict?["animationTimingFunction"] as? CAMediaTimingFunction {
                timingFunction = capturedTiming
            } else {
                timingFunction = CATransaction.animationTimingFunction()
            }
        }

        // Add the animation to the layer
        // Note: delegate?.animationDidStart is called inside layer.add()
        layer.add(self, forKey: event)
    }

    // MARK: - Class Methods

    /// Returns the default value of the property with the specified key.
    ///
    /// - Parameter key: The key of the property.
    /// - Returns: The default value for the property, or `nil` if no default is defined.
    open class func defaultValue(forKey key: String) -> Any? {
        switch key {
        case "beginTime":
            return CFTimeInterval(0)
        case "timeOffset":
            return CFTimeInterval(0)
        case "repeatCount":
            return Float(0)
        case "repeatDuration":
            return CFTimeInterval(0)
        case "duration":
            return CFTimeInterval(0)
        case "speed":
            return Float(1)
        case "autoreverses":
            return false
        case "fillMode":
            return CAMediaTimingFillMode.removed
        case "isRemovedOnCompletion":
            return true
        default:
            return nil
        }
    }
}
