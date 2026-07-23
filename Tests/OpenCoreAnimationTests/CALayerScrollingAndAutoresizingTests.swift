import Testing
@testable import OpenCoreAnimation

@Suite("CALayer scrolling and autoresizing semantics")
struct CALayerScrollingAndAutoresizingTests {
    @Test("CAScrollLayer point scrolling changes bounds origin without content clamping")
    func pointScrolling() {
        let scrollLayer = CAScrollLayer()
        scrollLayer.bounds = CGRect(x: 5, y: 7, width: 100, height: 100)

        scrollLayer.scroll(to: CGPoint(x: 50, y: 60))

        #expect(scrollLayer.bounds.origin == CGPoint(x: 50, y: 60))
    }

    @Test("CAScrollLayer rectangle scrolling performs the minimum movement")
    func rectangleScrolling() {
        let scrollLayer = CAScrollLayer()
        scrollLayer.bounds = CGRect(x: 5, y: 7, width: 100, height: 100)

        scrollLayer.scroll(to: CGRect(x: 120, y: 130, width: 20, height: 30))

        #expect(scrollLayer.bounds.origin == CGPoint(x: 40, y: 60))
    }

    @Test("CAScrollLayer point scrolling respects every mode")
    func pointScrollingModes() {
        let expectations: [(CAScrollLayerScrollMode, CGPoint)] = [
            (.none, CGPoint(x: 10, y: 20)),
            (.horizontally, CGPoint(x: 30, y: 20)),
            (.vertically, CGPoint(x: 10, y: 50)),
            (.both, CGPoint(x: 30, y: 50)),
            (
                CAScrollLayerScrollMode(rawValue: "future"),
                CGPoint(x: 10, y: 20)
            ),
        ]

        for (mode, expectedOrigin) in expectations {
            let scrollLayer = CAScrollLayer()
            scrollLayer.bounds = CGRect(x: 10, y: 20, width: 100, height: 80)
            scrollLayer.scrollMode = mode

            scrollLayer.scroll(to: CGPoint(x: 30, y: 50))

            #expect(scrollLayer.bounds.origin == expectedOrigin)
        }
    }

    @Test("CAScrollLayer rectangle scrolling respects every mode")
    func rectangleScrollingModes() {
        let expectations: [(CAScrollLayerScrollMode, CGPoint)] = [
            (.none, CGPoint(x: 10, y: 20)),
            (.horizontally, CGPoint(x: 20, y: 20)),
            (.vertically, CGPoint(x: 10, y: 35)),
            (.both, CGPoint(x: 20, y: 35)),
            (
                CAScrollLayerScrollMode(rawValue: "future"),
                CGPoint(x: 10, y: 20)
            ),
        ]

        for (mode, expectedOrigin) in expectations {
            let scrollLayer = CAScrollLayer()
            scrollLayer.bounds = CGRect(x: 10, y: 20, width: 100, height: 80)
            scrollLayer.scrollMode = mode

            scrollLayer.scroll(
                to: CGRect(x: 90, y: 75, width: 30, height: 40)
            )

            #expect(scrollLayer.bounds.origin == expectedOrigin)
        }
    }

    @Test("CALayer scrolling targets the closest ancestor scroll layer")
    func ancestorScrolling() {
        let outerScrollLayer = CAScrollLayer()
        outerScrollLayer.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)

        let innerScrollLayer = CAScrollLayer()
        innerScrollLayer.anchorPoint = .zero
        innerScrollLayer.frame = CGRect(x: 20, y: 30, width: 100, height: 100)
        outerScrollLayer.addSublayer(innerScrollLayer)

        let content = CALayer()
        content.anchorPoint = .zero
        content.frame = CGRect(x: 100, y: 100, width: 50, height: 50)
        innerScrollLayer.addSublayer(content)

        content.scroll(CGPoint(x: 10, y: 20))

        #expect(innerScrollLayer.bounds.origin == CGPoint(x: 110, y: 120))
        #expect(outerScrollLayer.bounds.origin == .zero)
        #expect(content.visibleRect == CGRect(x: 10, y: 20, width: 40, height: 30))
    }

    @Test("Autoresizing distributes size changes across flexible frame segments")
    func flexibleSegments() {
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)

        let child = CALayer()
        child.anchorPoint = .zero
        child.frame = CGRect(x: 10, y: 15, width: 20, height: 25)
        child.autoresizingMask = [
            .layerMinXMargin,
            .layerWidthSizable,
            .layerMaxXMargin,
            .layerHeightSizable
        ]
        parent.addSublayer(child)

        parent.bounds.size = CGSize(width: 200, height: 160)

        #expect(child.frame == CGRect(x: 43, y: 15, width: 54, height: 85))
    }

    @Test("An empty autoresizing mask preserves a fractional frame")
    func emptyMask() {
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)

        let child = CALayer()
        child.anchorPoint = .zero
        child.frame = CGRect(x: 10.2, y: 15.3, width: 20.4, height: 25.5)
        parent.addSublayer(child)

        parent.bounds.size = CGSize(width: 101.2, height: 100)

        #expect(child.frame == CGRect(x: 10.2, y: 15.3, width: 20.4, height: 25.5))
    }
}
