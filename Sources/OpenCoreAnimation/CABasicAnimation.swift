
/// An object that provides basic, single-keyframe animation capabilities for a layer property.
open class CABasicAnimation: CAPropertyAnimation {

    /// The value at the start of the animation.
    open var fromValue: Any?

    /// The value at the end of the animation.
    open var toValue: Any?

    /// The value at which the animation will interpolate between.
    open var byValue: Any?

    public required init() {
        super.init()
    }

    public required init(animation: CAAnimation) {
        super.init(animation: animation)
        if let source = animation as? CABasicAnimation {
            self.fromValue = source.fromValue
            self.toValue = source.toValue
            self.byValue = source.byValue
        }
    }

    open override func shouldArchiveValue(forKey key: String) -> Bool {
        switch key {
        case "fromValue":
            return fromValue != nil
        case "toValue":
            return toValue != nil
        case "byValue":
            return byValue != nil
        default:
            return super.shouldArchiveValue(forKey: key)
        }
    }
}
