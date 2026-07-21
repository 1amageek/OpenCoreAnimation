import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CALayer coordinate conversion")
struct CALayerCoordinateConversionTests {
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
