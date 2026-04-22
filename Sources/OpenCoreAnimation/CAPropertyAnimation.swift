
/// An abstract subclass for creating animations that manipulate the value of layer properties.
open class CAPropertyAnimation: CAAnimation {

    /// The key path of the property to be animated.
    open var keyPath: String?

    /// Creates an animation with the specified key path.
    public convenience init(keyPath: String?) {
        self.init()
        self.keyPath = keyPath
    }

    public required init() {
        super.init()
    }

    public required init(animation: CAAnimation) {
        super.init(animation: animation)
        if let source = animation as? CAPropertyAnimation {
            self.keyPath = source.keyPath
            self.isAdditive = source.isAdditive
            self.isCumulative = source.isCumulative
            self.valueFunction = source.valueFunction
        }
    }

    /// Determines if the value specified by the animation is added to the current render tree value
    /// to produce the new render tree value.
    open var isAdditive: Bool = false

    /// Determines if the value of the property is the value at the end of the previous repeat cycle,
    /// plus the value of the current repeat cycle.
    open var isCumulative: Bool = false

    /// An optional value function that is applied to interpolated values.
    open var valueFunction: CAValueFunction?
}
