import Testing
@testable import OpenCoreAnimation

func setStoredAnimationBeginTime(
    _ beginTime: CFTimeInterval,
    on layer: CALayer,
    forKey key: String
) {
    guard let animation = layer.animation(forKey: key) else {
        Issue.record("Expected stored animation for key \(key).")
        return
    }
    animation.beginTime = beginTime
}
