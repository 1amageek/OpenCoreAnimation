import Testing
@testable import OpenCoreAnimation

@Suite("Animation archive decisions")
struct CAAnimationArchivingTests {
    private final class Delegate: CAAnimationDelegate {}

    @Test("Base animation archives only persistent nondefault state")
    func baseAnimation() {
        let defaultKeys = [
            "beginTime", "timeOffset", "repeatCount", "repeatDuration",
            "duration", "speed", "autoreverses", "fillMode",
            "timingFunction", "delegate", "removedOnCompletion",
            "preferredFrameRateRange", "futureKey",
        ]
        let animation = CAAnimation()
        for key in defaultKeys {
            #expect(!animation.shouldArchiveValue(forKey: key), "fresh \(key)")
        }

        let delegate = Delegate()
        animation.beginTime = 1
        animation.timeOffset = 1
        animation.repeatCount = 2
        animation.repeatDuration = 2
        animation.duration = 2
        animation.speed = 2
        animation.autoreverses = true
        animation.fillMode = .forwards
        animation.timingFunction = CAMediaTimingFunction(name: .easeIn)
        animation.delegate = delegate
        animation.isRemovedOnCompletion = false
        animation.preferredFrameRateRange = CAFrameRateRange(
            minimum: 24,
            maximum: 60,
            preferred: 30
        )

        for key in defaultKeys.dropLast(2) {
            #expect(animation.shouldArchiveValue(forKey: key), "changed \(key)")
        }
        #expect(!animation.shouldArchiveValue(forKey: "preferredFrameRateRange"))
        #expect(!animation.shouldArchiveValue(forKey: "futureKey"))
    }

    @Test("Concrete animation subclasses archive their persistent state")
    func animationSubclasses() {
        let property = CAPropertyAnimation()
        for key in ["keyPath", "additive", "cumulative", "valueFunction"] {
            #expect(!property.shouldArchiveValue(forKey: key), "fresh property \(key)")
        }
        property.keyPath = "opacity"
        property.isAdditive = true
        property.isCumulative = true
        property.valueFunction = CAValueFunction(name: .rotateZ)
        for key in ["keyPath", "additive", "cumulative", "valueFunction"] {
            #expect(property.shouldArchiveValue(forKey: key), "changed property \(key)")
        }

        let basic = CABasicAnimation()
        basic.fromValue = 0
        basic.toValue = 1
        basic.byValue = 2
        for key in ["fromValue", "toValue", "byValue"] {
            #expect(basic.shouldArchiveValue(forKey: key), "changed basic \(key)")
        }

        let path = CGMutablePath()
        path.addRect(CGRect(x: 0, y: 0, width: 2, height: 3))
        let keyframe = CAKeyframeAnimation()
        keyframe.values = [0, 1]
        keyframe.path = path
        keyframe.keyTimes = [0, 1]
        keyframe.timingFunctions = [CAMediaTimingFunction(name: .linear)]
        keyframe.calculationMode = .paced
        keyframe.tensionValues = [1]
        keyframe.continuityValues = [1]
        keyframe.biasValues = [1]
        keyframe.rotationMode = .rotateAuto
        for key in [
            "values", "path", "keyTimes", "timingFunctions", "calculationMode",
            "tensionValues", "continuityValues", "biasValues", "rotationMode",
        ] {
            #expect(keyframe.shouldArchiveValue(forKey: key), "changed keyframe \(key)")
        }

        let group = CAAnimationGroup()
        #expect(!group.shouldArchiveValue(forKey: "animations"))
        group.animations = []
        #expect(group.shouldArchiveValue(forKey: "animations"))

        let transition = CATransition()
        transition.type = .push
        transition.subtype = .fromLeft
        transition.startProgress = 0.25
        transition.endProgress = 0.75
        transition.filter = "filter"
        for key in ["type", "subtype", "startProgress", "endProgress", "filter"] {
            #expect(transition.shouldArchiveValue(forKey: key), "changed transition \(key)")
        }

        let spring = CASpringAnimation()
        spring.mass = 2
        spring.stiffness = 200
        spring.damping = 20
        spring.initialVelocity = 2
        spring.allowsOverdamping = true
        for key in ["mass", "stiffness", "damping", "allowsOverdamping"] {
            #expect(spring.shouldArchiveValue(forKey: key), "changed spring \(key)")
        }
        #expect(!spring.shouldArchiveValue(forKey: "initialVelocity"))
    }

    @Test("Emitter cell archives every changed persistent property")
    func emitterCell() {
        let keys = [
            "contents", "contentsRect", "contentsScale", "magnificationFilter",
            "minificationFilter", "minificationFilterBias", "birthRate", "lifetime",
            "lifetimeRange", "emitterCells", "color", "redRange", "greenRange",
            "blueRange", "alphaRange", "redSpeed", "greenSpeed", "blueSpeed",
            "alphaSpeed", "velocity", "velocityRange", "xAcceleration",
            "yAcceleration", "zAcceleration", "scale", "scaleRange", "scaleSpeed",
            "spin", "spinRange", "emissionLatitude", "emissionLongitude",
            "emissionRange", "name", "enabled", "style", "beginTime",
            "timeOffset", "repeatCount", "repeatDuration", "duration", "speed",
            "autoreverses", "fillMode",
        ]
        let cell = CAEmitterCell()
        for key in keys {
            #expect(!cell.shouldArchiveValue(forKey: key), "fresh \(key)")
        }

        cell.contents = "contents"
        cell.contentsRect = CGRect(x: 1, y: 2, width: 3, height: 4)
        cell.contentsScale = 2
        cell.magnificationFilter = CALayerContentsFilter.nearest.rawValue
        cell.minificationFilter = CALayerContentsFilter.nearest.rawValue
        cell.minificationFilterBias = 1
        cell.birthRate = 1
        cell.lifetime = 1
        cell.lifetimeRange = 1
        cell.emitterCells = []
        cell.color = CGColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 0.5)
        cell.redRange = 1
        cell.greenRange = 1
        cell.blueRange = 1
        cell.alphaRange = 1
        cell.redSpeed = 1
        cell.greenSpeed = 1
        cell.blueSpeed = 1
        cell.alphaSpeed = 1
        cell.velocity = 1
        cell.velocityRange = 1
        cell.xAcceleration = 1
        cell.yAcceleration = 1
        cell.zAcceleration = 1
        cell.scale = 2
        cell.scaleRange = 1
        cell.scaleSpeed = 1
        cell.spin = 1
        cell.spinRange = 1
        cell.emissionLatitude = 1
        cell.emissionLongitude = 1
        cell.emissionRange = 1
        cell.name = "cell"
        cell.isEnabled = false
        cell.style = ["key": "value"]
        cell.beginTime = 1
        cell.timeOffset = 1
        cell.repeatCount = 2
        cell.repeatDuration = 2
        cell.duration = 2
        cell.speed = 2
        cell.autoreverses = true
        cell.fillMode = .forwards

        for key in keys {
            #expect(cell.shouldArchiveValue(forKey: key), "changed \(key)")
        }
        #expect(!cell.shouldArchiveValue(forKey: "futureKey"))
    }

    @Test("Automatic contents format is distinct from the layer default")
    func automaticContentsFormat() {
        #expect(CALayerContentsFormat.automatic.rawValue == "Automatic")
        #expect(CALayer().contentsFormat == .RGBA8Uint)
        #expect(CALayerContentsFormat.automatic != .RGBA8Uint)
    }
}
