import Testing
@testable import OpenCoreAnimation

@Suite("CALayer hit-testing order and coordinates")
struct CALayerHitTestingTests {
    @Test("Hit testing follows visual zPosition instead of insertion order")
    func hitTestingUsesZPosition() {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        root.anchorPoint = .zero
        root.position = .zero

        let front = CALayer()
        front.frame = root.bounds
        front.zPosition = 10
        root.addSublayer(front)

        let back = CALayer()
        back.frame = root.bounds
        back.zPosition = -10
        root.addSublayer(back)

        #expect(root.hitTest(CGPoint(x: 50, y: 50)) === front)
    }

    @Test("Equal zPosition preserves frontmost insertion order")
    func hitTestingUsesStableInsertionOrder() {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        root.anchorPoint = .zero
        root.position = .zero

        let first = CALayer()
        first.frame = root.bounds
        root.addSublayer(first)

        let second = CALayer()
        second.frame = root.bounds
        root.addSublayer(second)

        #expect(root.hitTest(CGPoint(x: 50, y: 50)) === second)
    }

    @Test("Transform layer converts the input point exactly once")
    func transformLayerConvertsPointOnce() {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 300, height: 300)
        root.anchorPoint = .zero
        root.position = .zero

        let transformLayer = CATransformLayer()
        transformLayer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        transformLayer.position = CGPoint(x: 150, y: 150)
        root.addSublayer(transformLayer)

        let child = CALayer()
        child.frame = transformLayer.bounds
        transformLayer.addSublayer(child)

        #expect(transformLayer.hitTest(CGPoint(x: 150, y: 150)) === child)
        #expect(transformLayer.hitTest(CGPoint(x: 250, y: 250)) == nil)
    }

    @Test("Transform layer hit testing follows child zPosition")
    func transformLayerHitTestingUsesZPosition() {
        let transformLayer = CATransformLayer()
        transformLayer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        transformLayer.anchorPoint = .zero
        transformLayer.position = .zero

        let front = CALayer()
        front.frame = transformLayer.bounds
        front.zPosition = 5
        transformLayer.addSublayer(front)

        let back = CALayer()
        back.frame = transformLayer.bounds
        back.zPosition = -5
        transformLayer.addSublayer(back)

        #expect(transformLayer.hitTest(CGPoint(x: 50, y: 50)) === front)
    }
}
