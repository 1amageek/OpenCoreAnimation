import Testing
@testable import OpenCoreAnimation

@Suite("CAConstraintLayoutManager Behavior Tests")
struct CAConstraintLayoutManagerBehaviorTests {
    @Test("Solves position and size constraints against nonzero superlayer bounds")
    func solvesSuperlayerGeometry() {
        let container = CALayer()
        container.bounds = CGRect(x: 20, y: 30, width: 200, height: 100)
        container.layoutManager = CAConstraintLayoutManager()

        let child = CALayer()
        child.bounds = CGRect(x: 7, y: 9, width: 20, height: 30)
        child.anchorPoint = CGPoint(x: 0.25, y: 0.75)
        child.constraints = [
            constraint(.minX, relativeTo: "superlayer", .minX, offset: 10),
            constraint(.width, relativeTo: "superlayer", .width, scale: 0.5),
            constraint(.minY, relativeTo: "superlayer", .minY, offset: 5),
            constraint(.height, relativeTo: "superlayer", .height, scale: 0.5),
        ]
        container.addSublayer(child)

        layout(container)

        #expect(child.bounds == CGRect(x: 7, y: 9, width: 100, height: 50))
        #expect(child.position == CGPoint(x: 55, y: 72.5))
        #expect(child.frame == CGRect(x: 30, y: 35, width: 100, height: 50))
    }

    @Test("Derives size from coupled edge constraints")
    func derivesSizeFromEdges() {
        let container = makeContainer(width: 200, height: 100)
        let child = CALayer()
        child.frame = CGRect(x: 50, y: 12, width: 20, height: 30)
        child.constraints = [
            constraint(.minX, relativeTo: "superlayer", .minX, offset: 10),
            constraint(.maxX, relativeTo: "superlayer", .maxX, offset: -20),
        ]
        container.addSublayer(child)

        layout(container)

        #expect(child.frame == CGRect(x: 10, y: 12, width: 170, height: 30))
    }

    @Test("Solves sibling chains independently of sublayer order", arguments: [false, true])
    func solvesSiblingChain(reverseOrder: Bool) {
        let container = makeContainer(width: 200, height: 100)
        let first = namedLayer("first", frame: CGRect(x: 80, y: 40, width: 20, height: 20))
        let second = namedLayer("second", frame: CGRect(x: 3, y: 4, width: 30, height: 20))
        let third = namedLayer("third", frame: CGRect(x: 6, y: 7, width: 40, height: 20))

        first.constraints = [
            constraint(.minX, relativeTo: "superlayer", .minX, offset: 10),
            constraint(.minY, relativeTo: "superlayer", .minY, offset: 8),
        ]
        second.constraints = [
            constraint(.minX, relativeTo: "first", .maxX, offset: 5),
            constraint(.minY, relativeTo: "first", .minY),
        ]
        third.constraints = [
            constraint(.minX, relativeTo: "second", .maxX, offset: 7),
            constraint(.minY, relativeTo: "second", .minY),
        ]

        let layers = reverseOrder ? [third, second, first] : [first, second, third]
        layers.forEach(container.addSublayer)

        layout(container)

        #expect(first.frame == CGRect(x: 10, y: 8, width: 20, height: 20))
        #expect(second.frame == CGRect(x: 35, y: 8, width: 30, height: 20))
        #expect(third.frame == CGRect(x: 72, y: 8, width: 40, height: 20))
    }

    @Test("Conflicting equations do not block independent geometry components")
    func isolatesConflictingComponents() {
        let container = makeContainer(width: 200, height: 100)
        let conflicted = namedLayer(
            "conflicted",
            frame: CGRect(x: 50, y: 5, width: 20, height: 20)
        )
        conflicted.constraints = [
            constraint(.minX, relativeTo: "superlayer", .minX, offset: 10),
            constraint(.minX, relativeTo: "superlayer", .minX, offset: 20),
            constraint(.minY, relativeTo: "superlayer", .minY, offset: 15),
        ]

        let valid = namedLayer("valid", frame: CGRect(x: 1, y: 2, width: 30, height: 30))
        valid.constraints = [
            constraint(.minX, relativeTo: "superlayer", .minX, offset: 25),
        ]
        container.addSublayer(conflicted)
        container.addSublayer(valid)

        layout(container)

        #expect(conflicted.frame == CGRect(x: 50, y: 15, width: 20, height: 20))
        #expect(valid.frame == CGRect(x: 25, y: 2, width: 30, height: 30))
    }

    @Test("Ignores unresolved sources while applying valid constraints")
    func ignoresUnresolvedSources() {
        let container = makeContainer(width: 200, height: 100)
        let child = CALayer()
        child.frame = CGRect(x: 40, y: 6, width: 20, height: 30)
        child.constraints = [
            constraint(.minX, relativeTo: "missing", .maxX, offset: 99),
            constraint(.minY, relativeTo: "superlayer", .minY, offset: 12),
        ]
        container.addSublayer(child)

        layout(container)

        #expect(child.frame == CGRect(x: 40, y: 12, width: 20, height: 30))
    }

    @Test("Returns the layer bounds size as its preferred size")
    func returnsPreferredSize() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 10, y: 20, width: 80, height: 45)

        #expect(CAConstraintLayoutManager().preferredSize(of: layer) == CGSize(width: 80, height: 45))
    }

    @Test("Invalidates layout when its inputs change")
    func invalidatesLayoutFromInputs() {
        let container = CALayer()
        #expect(!container.needsLayout())

        container.layoutManager = CAConstraintLayoutManager()
        #expect(container.needsLayout())
        container.layoutIfNeeded()
        #expect(!container.needsLayout())

        let child = CALayer()
        child.frame = CGRect(x: 30, y: 20, width: 10, height: 10)
        container.addSublayer(child)
        #expect(container.needsLayout())
        container.layoutIfNeeded()

        child.constraints = [
            constraint(.minX, relativeTo: "superlayer", .minX, offset: 5),
        ]
        #expect(container.needsLayout())
        container.layoutIfNeeded()
        #expect(child.frame.minX == 5)

        container.bounds = CGRect(x: 10, y: 0, width: 100, height: 100)
        #expect(container.needsLayout())
        container.layoutIfNeeded()
        #expect(child.frame.minX == 15)

        child.removeFromSuperlayer()
        #expect(container.needsLayout())
    }

    @MainActor
    @Test("Animation engine resolves pending layout before rendering")
    func animationEngineResolvesLayout() {
        let engine = CAAnimationEngine()
        let container = makeContainer(width: 200, height: 100)
        let child = CALayer()
        child.frame = CGRect(x: 80, y: 30, width: 20, height: 20)
        child.constraints = [
            constraint(.minX, relativeTo: "superlayer", .minX, offset: 10),
            constraint(.minY, relativeTo: "superlayer", .minY, offset: 12),
        ]
        container.addSublayer(child)
        engine.rootLayer = container

        engine.renderFrame()

        #expect(child.frame == CGRect(x: 10, y: 12, width: 20, height: 20))
        #expect(!container.needsLayout())
    }

    @Test("Layout repeats when invalidated during an active pass")
    func repeatsInvalidatedLayoutPass() {
        let manager = InvalidatingLayoutManager()
        let layer = CALayer()
        layer.layoutManager = manager

        layer.layoutIfNeeded()

        #expect(manager.layoutCount == 2)
        #expect(manager.invalidationCount == 2)
        #expect(!layer.needsLayout())
    }

    private func makeContainer(width: CGFloat, height: CGFloat) -> CALayer {
        let container = CALayer()
        container.bounds = CGRect(x: 0, y: 0, width: width, height: height)
        container.layoutManager = CAConstraintLayoutManager()
        return container
    }

    private func namedLayer(_ name: String, frame: CGRect) -> CALayer {
        let layer = CALayer()
        layer.name = name
        layer.frame = frame
        return layer
    }

    private func layout(_ container: CALayer) {
        container.setNeedsLayout()
        container.layoutIfNeeded()
    }

    private func constraint(
        _ attribute: CAConstraintAttribute,
        relativeTo sourceName: String,
        _ sourceAttribute: CAConstraintAttribute,
        scale: CGFloat = 1,
        offset: CGFloat = 0
    ) -> CAConstraint {
        CAConstraint(
            attribute: attribute,
            relativeTo: sourceName,
            attribute: sourceAttribute,
            scale: scale,
            offset: offset
        )
    }

    private final class InvalidatingLayoutManager: CALayoutManager {
        private(set) var layoutCount = 0
        private(set) var invalidationCount = 0

        func invalidateLayout(of layer: CALayer) {
            invalidationCount += 1
            layer.setNeedsLayout()
        }

        func layoutSublayers(of layer: CALayer) {
            layoutCount += 1
            if layoutCount == 1 {
                layer.setNeedsLayout()
            }
        }

        func preferredSize(of layer: CALayer) -> CGSize {
            layer.bounds.size
        }
    }
}
