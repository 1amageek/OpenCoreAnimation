import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CAAnimationGroup lifecycle")
struct CAAnimationGroupLifecycleTests {
    private final class AnimationDelegate: CAAnimationDelegate {
        private(set) var startCount = 0
        private(set) var stopEvents: [Bool] = []

        func animationDidStart(_ anim: CAAnimation) {
            startCount += 1
        }

        func animationDidStop(_ anim: CAAnimation, finished flag: Bool) {
            stopEvents.append(flag)
        }
    }

    @Test("Only the attached group receives lifecycle callbacks")
    func onlyAttachedGroupReceivesLifecycleCallbacks() {
        let layer = CALayer()
        let childDelegate = AnimationDelegate()
        let nestedGroupDelegate = AnimationDelegate()
        let rootGroupDelegate = AnimationDelegate()

        let child = CABasicAnimation(keyPath: "opacity")
        child.fromValue = Float(0)
        child.toValue = Float(1)
        child.duration = 1
        child.delegate = childDelegate

        let nestedGroup = CAAnimationGroup()
        nestedGroup.animations = [child]
        nestedGroup.duration = 1
        nestedGroup.delegate = nestedGroupDelegate

        let rootGroup = CAAnimationGroup()
        rootGroup.animations = [nestedGroup]
        rootGroup.duration = 1
        rootGroup.delegate = rootGroupDelegate

        layer.add(rootGroup, forKey: "group")
        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "group")
        _ = layer.presentation()

        #expect(rootGroupDelegate.startCount == 1)
        #expect(childDelegate.startCount == 0)
        #expect(nestedGroupDelegate.startCount == 0)

        setStoredAnimationBeginTime(CACurrentMediaTime() - 2, on: layer, forKey: "group")
        layer.processAnimationCompletions()

        #expect(rootGroupDelegate.stopEvents == [true])
        #expect(childDelegate.stopEvents.isEmpty)
        #expect(nestedGroupDelegate.stopEvents.isEmpty)
    }

    @Test("An empty attached group starts when its active interval begins")
    func emptyGroupStartsAtActiveInterval() {
        let layer = CALayer()
        let delegate = AnimationDelegate()
        let group = CAAnimationGroup()
        group.animations = []
        group.duration = 1
        group.delegate = delegate

        layer.add(group, forKey: "group")
        setStoredAnimationBeginTime(CACurrentMediaTime() - 0.5, on: layer, forKey: "group")
        _ = layer.presentation()

        #expect(delegate.startCount == 1)
        #expect(delegate.stopEvents.isEmpty)

        setStoredAnimationBeginTime(CACurrentMediaTime() - 2, on: layer, forKey: "group")
        layer.processAnimationCompletions()

        #expect(delegate.startCount == 1)
        #expect(delegate.stopEvents == [true])
    }

    @Test("Explicit group removal stops only the attached group")
    func explicitRemovalStopsOnlyAttachedGroup() {
        let layer = CALayer()
        let childDelegate = AnimationDelegate()
        let groupDelegate = AnimationDelegate()

        let child = CABasicAnimation(keyPath: "opacity")
        child.duration = 10
        child.delegate = childDelegate

        let group = CAAnimationGroup()
        group.animations = [child]
        group.duration = 10
        group.delegate = groupDelegate

        layer.add(group, forKey: "group")
        layer.removeAnimation(forKey: "group")

        #expect(groupDelegate.startCount == 0)
        #expect(groupDelegate.stopEvents == [false])
        #expect(childDelegate.startCount == 0)
        #expect(childDelegate.stopEvents.isEmpty)
    }

    @Test("A retained completed group notifies its delegate exactly once")
    func retainedCompletedGroupNotifiesOnce() {
        let layer = CALayer()
        let delegate = AnimationDelegate()
        let group = CAAnimationGroup()
        group.animations = []
        group.duration = 1
        group.isRemovedOnCompletion = false
        group.delegate = delegate

        layer.add(group, forKey: "group")
        setStoredAnimationBeginTime(CACurrentMediaTime() - 2, on: layer, forKey: "group")
        layer.processAnimationCompletions()
        layer.processAnimationCompletions()

        #expect(layer.animation(forKey: "group") != nil)
        #expect(delegate.startCount == 1)
        #expect(delegate.stopEvents == [true])
    }
}
