import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CALayer coordinate conversion")
struct CALayerCoordinateConversionTests {
    @Test("Nil point conversion uses the superlayer coordinate space")
    func nilPointConversionUsesSuperlayerCoordinateSpace() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 5, y: 7, width: 100, height: 80)
        layer.anchorPoint = CGPoint(x: 0.25, y: 0.75)
        layer.position = CGPoint(x: 80, y: 90)
        layer.transform = CATransform3DMakeRotation(0.3, 0, 0, 1)
        layer.isGeometryFlipped = true

        let localPoint = CGPoint(x: 11, y: 13)
        let superlayerPoint = layer.convert(localPoint, to: nil)
        let convertedFromSuperlayer = layer.convert(localPoint, from: nil)

        #expect(abs(superlayerPoint.x - 57.7113238134) < 0.0001)
        #expect(abs(superlayerPoint.y - 97.7598269212) < 0.0001)
        #expect(abs(convertedFromSuperlayer.x + 58.6732736626) < 0.0001)
        #expect(abs(convertedFromSuperlayer.y - 80.1700154030) < 0.0001)
        let roundTrip = layer.convert(superlayerPoint, from: nil)
        #expect(abs(roundTrip.x - localPoint.x) < 0.0001)
        #expect(abs(roundTrip.y - localPoint.y) < 0.0001)
    }

    @Test("Nil rectangle conversion projects every corner")
    func nilRectangleConversionProjectsEveryCorner() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 40, height: 20)
        layer.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        layer.position = CGPoint(x: 50, y: 60)
        layer.transform = CATransform3DMakeRotation(.pi / 2, 0, 0, 1)

        let converted = layer.convert(layer.bounds, to: nil)
        #expect(abs(converted.origin.x - 40) < 0.0001)
        #expect(abs(converted.origin.y - 40) < 0.0001)
        #expect(abs(converted.width - 20) < 0.0001)
        #expect(abs(converted.height - 40) < 0.0001)

        let restored = layer.convert(converted, from: nil)
        #expect(abs(restored.origin.x) < 0.0001)
        #expect(abs(restored.origin.y) < 0.0001)
        #expect(abs(restored.width - 40) < 0.0001)
        #expect(abs(restored.height - 20) < 0.0001)
    }

    @Test("Geometry flipping matches Core Animation coordinate conversion")
    func geometryFlippingMatchesCoreAnimationCoordinateConversion() {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)
        root.anchorPoint = .zero
        root.position = .zero

        let parent = CALayer()
        parent.bounds = CGRect(x: 5, y: 7, width: 100, height: 80)
        parent.anchorPoint = .zero
        parent.position = CGPoint(x: 20, y: 30)
        parent.isGeometryFlipped = true
        root.addSublayer(parent)

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        child.anchorPoint = .zero
        child.position = CGPoint(x: 15, y: 17)
        parent.addSublayer(child)

        let childOrigin = child.convert(CGPoint.zero, to: root)
        let childTop = child.convert(CGPoint(x: 0, y: 10), to: root)
        let parentBoundsOrigin = parent.convert(parent.bounds.origin, to: root)

        #expect(childOrigin == CGPoint(x: 30, y: 100))
        #expect(childTop == CGPoint(x: 30, y: 90))
        #expect(parentBoundsOrigin == CGPoint(x: 20, y: 110))
        #expect(child.convert(childOrigin, from: root) == CGPoint.zero)
        #expect(root.hitTest(CGPoint(x: 35, y: 95)) === child)
    }

    @Test("Geometry flipping preserves anchor-relative transforms")
    func geometryFlippingPreservesAnchorRelativeTransforms() {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 300, height: 300)
        root.anchorPoint = .zero
        root.position = .zero

        let layer = CALayer()
        layer.bounds = CGRect(x: 5, y: 7, width: 100, height: 80)
        layer.anchorPoint = CGPoint(x: 0.25, y: 0.75)
        layer.position = CGPoint(x: 80, y: 90)
        layer.transform = CATransform3DMakeRotation(0.3, 0, 0, 1)
        layer.isGeometryFlipped = true
        root.addSublayer(layer)

        let boundsOrigin = layer.convert(CGPoint(x: 5, y: 7), to: root)
        let lowerRight = layer.convert(CGPoint(x: 105, y: 7), to: root)
        let upperLeft = layer.convert(CGPoint(x: 5, y: 87), to: root)

        #expect(abs(boundsOrigin.x - 50.2061836386) < 0.0001)
        #expect(abs(boundsOrigin.y - 101.718724616) < 0.0001)
        #expect(abs(lowerRight.x - 145.739832551) < 0.0001)
        #expect(abs(lowerRight.y - 131.270745282) < 0.0001)
        #expect(abs(upperLeft.x - 73.8478001715) < 0.0001)
        #expect(abs(upperLeft.y - 25.2918054859) < 0.0001)
    }

    @Test("Sublayer transforms affect conversion and hit testing")
    func sublayerTransformAffectsConversionAndHitTesting() {
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)
        parent.anchorPoint = .zero
        parent.position = .zero
        parent.sublayerTransform = CATransform3DMakeScale(2, 2, 1)

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
        child.anchorPoint = .zero
        child.position = CGPoint(x: 20, y: 20)
        parent.addSublayer(child)

        let parentPoint = child.convert(CGPoint(x: 5, y: 5), to: parent)
        #expect(abs(parentPoint.x - 50) < 0.0001)
        #expect(abs(parentPoint.y - 50) < 0.0001)

        let childPoint = child.convert(parentPoint, from: parent)
        #expect(abs(childPoint.x - 5) < 0.0001)
        #expect(abs(childPoint.y - 5) < 0.0001)
        #expect(parent.hitTest(parentPoint) === child)
        #expect(parent.hitTest(CGPoint(x: 25, y: 25)) === parent)
    }

    @Test("Projective transforms divide homogeneous coordinates")
    func projectiveTransformDividesHomogeneousCoordinates() {
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        parent.anchorPoint = .zero
        parent.position = .zero

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
        child.anchorPoint = .zero
        child.position = .zero
        var transform = CATransform3DIdentity
        transform.m14 = 0.05
        child.transform = transform
        parent.addSublayer(child)

        let projected = child.convert(CGPoint(x: 10, y: 0), to: parent)
        #expect(abs(projected.x - (10 / 1.5)) < 0.0001)
        #expect(abs(projected.y) < 0.0001)

        let restored = child.convert(projected, from: parent)
        #expect(abs(restored.x - 10) < 0.0001)
        #expect(abs(restored.y) < 0.0001)
    }

    @Test("3D planes invert through their projected homography")
    func threeDimensionalPlaneUsesProjectedInverse() {
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        parent.anchorPoint = .zero
        parent.position = .zero
        var perspective = CATransform3DIdentity
        perspective.m34 = -1 / 100
        parent.sublayerTransform = perspective

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
        child.anchorPoint = .zero
        child.position = .zero
        child.transform = CATransform3DMakeRotation(CGFloat.pi / 4, 0, 1, 0)
        parent.addSublayer(child)

        let localPoint = CGPoint(x: 10, y: 10)
        let projected = child.convert(localPoint, to: parent)
        let cosine = cos(CGFloat.pi / 4)
        let sine = sin(CGFloat.pi / 4)
        let homogeneousW = 1 + localPoint.x * sine / 100
        #expect(abs(projected.x - localPoint.x * cosine / homogeneousW) < 0.0001)
        #expect(abs(projected.y - localPoint.y / homogeneousW) < 0.0001)

        let restored = child.convert(projected, from: parent)
        #expect(abs(restored.x - localPoint.x) < 0.0001)
        #expect(abs(restored.y - localPoint.y) < 0.0001)
        #expect(parent.hitTest(projected) === child)
    }

    @Test("Singular projections do not produce false hits")
    func singularProjectionDoesNotProduceFalseHits() {
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        parent.anchorPoint = .zero
        parent.position = .zero

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
        child.anchorPoint = .zero
        child.position = CGPoint(x: 20, y: 20)
        child.transform = CATransform3DMakeScale(0, 1, 1)
        parent.addSublayer(child)

        let converted = child.convert(CGPoint(x: 20, y: 20), from: parent)
        #expect(!converted.x.isFinite)
        #expect(!converted.y.isFinite)
        #expect(parent.hitTest(CGPoint(x: 20, y: 20)) === parent)
    }
}
