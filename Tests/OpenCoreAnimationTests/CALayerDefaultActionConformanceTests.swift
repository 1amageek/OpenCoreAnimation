import Testing
import OpenCoreAnimation

@Suite("CALayer default action conformance")
struct CALayerDefaultActionConformanceTests {
    @Test("CALayer does not synthesize default actions")
    func caLayerDefaultActionsAreNil() {
        #expect(CALayer.defaultAction(forKey: "opacity") == nil)
        #expect(CALayer.defaultAction(forKey: "bounds") == nil)
        #expect(CALayer.defaultAction(forKey: "backgroundColor") == nil)
    }

    @Test("Layer subclasses inherit nil default actions")
    func subclassDefaultActionsAreNil() {
        #expect(CAShapeLayer.defaultAction(forKey: "strokeEnd") == nil)
        #expect(CAGradientLayer.defaultAction(forKey: "colors") == nil)
        #expect(CAReplicatorLayer.defaultAction(forKey: "instanceDelay") == nil)
        #expect(CAEmitterLayer.defaultAction(forKey: "birthRate") == nil)
        #expect(CATextLayer.defaultAction(forKey: "fontSize") == nil)
    }

    @Test("A custom action remains available")
    func customActionIsResolved() {
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        layer.actions = ["opacity": animation]

        let resolved = layer.action(forKey: "opacity") as? CABasicAnimation
        #expect(resolved === animation)
    }
}
