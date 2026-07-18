import Testing
@testable import OpenCoreAnimation

func setStoredAnimationAddedTime(
    _ addedTime: CFTimeInterval,
    on layer: CALayer,
    forKey key: String
) {
    guard let animation = layer.animation(forKey: key) else {
        Issue.record("Expected stored animation for key \(key).")
        return
    }
    animation.addedTime = addedTime
}
