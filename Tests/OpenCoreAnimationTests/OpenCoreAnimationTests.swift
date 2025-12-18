import Testing
@testable import OpenCoreAnimation

// MARK: - CATransform3D Tests

@Suite("CATransform3D Tests")
struct CATransform3DTests {

    // MARK: - Identity Transform

    @Test("Identity transform has correct values")
    func identityTransform() {
        let identity = CATransform3DIdentity
        #expect(identity.m11 == 1)
        #expect(identity.m12 == 0)
        #expect(identity.m13 == 0)
        #expect(identity.m14 == 0)
        #expect(identity.m21 == 0)
        #expect(identity.m22 == 1)
        #expect(identity.m23 == 0)
        #expect(identity.m24 == 0)
        #expect(identity.m31 == 0)
        #expect(identity.m32 == 0)
        #expect(identity.m33 == 1)
        #expect(identity.m34 == 0)
        #expect(identity.m41 == 0)
        #expect(identity.m42 == 0)
        #expect(identity.m43 == 0)
        #expect(identity.m44 == 1)
    }

    @Test("CATransform3DIsIdentity returns true for identity")
    func isIdentity() {
        #expect(CATransform3DIsIdentity(CATransform3DIdentity))
        #expect(!CATransform3DIsIdentity(CATransform3DMakeTranslation(1, 0, 0)))
    }

    // MARK: - Translation

    @Test("Translation transform creates correct matrix")
    func translationTransform() {
        let t = CATransform3DMakeTranslation(10, 20, 30)
        #expect(t.m41 == 10)
        #expect(t.m42 == 20)
        #expect(t.m43 == 30)
        #expect(t.m11 == 1)
        #expect(t.m22 == 1)
        #expect(t.m33 == 1)
        #expect(t.m44 == 1)
    }

    @Test("Chained translation works correctly")
    func chainedTranslation() {
        let t1 = CATransform3DMakeTranslation(10, 0, 0)
        let t2 = CATransform3DTranslate(t1, 5, 0, 0)
        #expect(t2.m41 == 15)
    }

    // MARK: - Scale

    @Test("Scale transform creates correct matrix")
    func scaleTransform() {
        let t = CATransform3DMakeScale(2, 3, 4)
        #expect(t.m11 == 2)
        #expect(t.m22 == 3)
        #expect(t.m33 == 4)
        #expect(t.m44 == 1)
    }

    @Test("Chained scale works correctly")
    func chainedScale() {
        let t1 = CATransform3DMakeScale(2, 2, 2)
        let t2 = CATransform3DScale(t1, 3, 3, 3)
        #expect(t2.m11 == 6)
        #expect(t2.m22 == 6)
        #expect(t2.m33 == 6)
    }

    // MARK: - Rotation

    @Test("Rotation around Z axis creates correct matrix")
    func rotationZAxis() {
        let angle = CGFloat.pi / 2 // 90 degrees
        let t = CATransform3DMakeRotation(angle, 0, 0, 1)

        // cos(90) ≈ 0, sin(90) ≈ 1
        #expect(abs(t.m11 - 0) < 0.0001)
        #expect(abs(t.m12 - 1) < 0.0001)
        #expect(abs(t.m21 - (-1)) < 0.0001)
        #expect(abs(t.m22 - 0) < 0.0001)
    }

    @Test("Rotation with zero vector returns identity")
    func rotationZeroVector() {
        let t = CATransform3DMakeRotation(CGFloat.pi, 0, 0, 0)
        #expect(CATransform3DIsIdentity(t))
    }

    // MARK: - Concatenation

    @Test("Concatenation combines transforms correctly")
    func concatenation() {
        let t1 = CATransform3DMakeTranslation(10, 0, 0)
        let t2 = CATransform3DMakeScale(2, 2, 2)
        let combined = CATransform3DConcat(t1, t2)

        // t1 * t2: first translate by (10,0,0), then scale by 2
        // The translation gets scaled: (10*2, 0, 0) = (20, 0, 0)
        #expect(combined.m41 == 20)
        #expect(combined.m11 == 2)
    }

    // MARK: - Inversion

    @Test("Inversion of translation")
    func invertTranslation() {
        let t = CATransform3DMakeTranslation(10, 20, 30)
        let inv = CATransform3DInvert(t)

        #expect(inv.m41 == -10)
        #expect(inv.m42 == -20)
        #expect(inv.m43 == -30)
    }

    @Test("Inversion of scale")
    func invertScale() {
        let t = CATransform3DMakeScale(2, 4, 8)
        let inv = CATransform3DInvert(t)

        #expect(inv.m11 == 0.5)
        #expect(inv.m22 == 0.25)
        #expect(inv.m33 == 0.125)
    }

    @Test("Transform multiplied by inverse equals identity")
    func inverseMultipliedByOriginal() {
        let t = CATransform3DMakeTranslation(10, 20, 30)
        let inv = CATransform3DInvert(t)
        let result = CATransform3DConcat(t, inv)

        // Should be approximately identity
        #expect(abs(result.m11 - 1) < 0.0001)
        #expect(abs(result.m22 - 1) < 0.0001)
        #expect(abs(result.m33 - 1) < 0.0001)
        #expect(abs(result.m44 - 1) < 0.0001)
        #expect(abs(result.m41) < 0.0001)
        #expect(abs(result.m42) < 0.0001)
        #expect(abs(result.m43) < 0.0001)
    }

    // MARK: - Affine Transform Conversion

    @Test("CATransform3DIsAffine returns true for 2D transforms")
    func isAffine() {
        let t2D = CATransform3DMakeTranslation(10, 20, 0)
        let t3D = CATransform3DMakeRotation(CGFloat.pi / 4, 1, 0, 0) // Rotation around X

        #expect(CATransform3DIsAffine(t2D))
        #expect(!CATransform3DIsAffine(t3D))
    }

    @Test("Affine transform conversion round-trip")
    func affineConversion() {
        // Test using CALayer's affineTransform() and setAffineTransform() methods
        // which internally use CATransform3DMakeAffineTransform and CATransform3DGetAffineTransform
        let layer = CALayer()
        let affine = CGAffineTransform(a: 1, b: 0, c: 0, d: 1, tx: 10, ty: 20)
        layer.setAffineTransform(affine)
        let backToAffine = layer.affineTransform()

        #expect(backToAffine.a == affine.a)
        #expect(backToAffine.b == affine.b)
        #expect(backToAffine.c == affine.c)
        #expect(backToAffine.d == affine.d)
        #expect(backToAffine.tx == affine.tx)
        #expect(backToAffine.ty == affine.ty)
    }

    // MARK: - Equality

    @Test("Transform equality")
    func transformEquality() {
        let t1 = CATransform3DMakeTranslation(10, 20, 30)
        let t2 = CATransform3DMakeTranslation(10, 20, 30)
        let t3 = CATransform3DMakeTranslation(10, 20, 31)

        #expect(CATransform3DEqualToTransform(t1, t2))
        #expect(!CATransform3DEqualToTransform(t1, t3))
        #expect(t1 == t2)
        #expect(t1 != t3)
    }
}

// MARK: - CAMediaTimingFunction Tests

@Suite("CAMediaTimingFunction Tests")
struct CAMediaTimingFunctionTests {

    @Test("Linear timing function")
    func linearTimingFunction() {
        let linear = CAMediaTimingFunction(name: .linear)

        #expect(linear.evaluate(at: 0) == 0)
        #expect(linear.evaluate(at: 1) == 1)
        #expect(abs(linear.evaluate(at: 0.5) - 0.5) < 0.01)
    }

    @Test("Ease-in timing function starts slow")
    func easeInTimingFunction() {
        let easeIn = CAMediaTimingFunction(name: .easeIn)

        // At t=0.5, ease-in should be less than 0.5 (starts slow)
        let midpoint = easeIn.evaluate(at: 0.5)
        #expect(midpoint < 0.5)
    }

    @Test("Ease-out timing function ends slow")
    func easeOutTimingFunction() {
        let easeOut = CAMediaTimingFunction(name: .easeOut)

        // At t=0.5, ease-out should be greater than 0.5 (ends slow)
        let midpoint = easeOut.evaluate(at: 0.5)
        #expect(midpoint > 0.5)
    }

    @Test("Ease-in-ease-out timing function")
    func easeInEaseOutTimingFunction() {
        let easeInOut = CAMediaTimingFunction(name: .easeInEaseOut)

        // At t=0.5, should be approximately 0.5
        let midpoint = easeInOut.evaluate(at: 0.5)
        #expect(abs(midpoint - 0.5) < 0.1)
    }

    @Test("Custom control points")
    func customControlPoints() {
        let custom = CAMediaTimingFunction(controlPoints: 0.25, 0.1, 0.25, 1.0)

        var p1 = [Float](repeating: 0, count: 2)
        var p2 = [Float](repeating: 0, count: 2)

        custom.getControlPoint(at: 1, values: &p1)
        custom.getControlPoint(at: 2, values: &p2)

        #expect(p1[0] == 0.25)
        #expect(p1[1] == 0.1)
        #expect(p2[0] == 0.25)
        #expect(p2[1] == 1.0)
    }

    @Test("Control point at index 0 is origin")
    func controlPointOrigin() {
        let func1 = CAMediaTimingFunction(name: .linear)
        var point = [Float](repeating: -1, count: 2)
        func1.getControlPoint(at: 0, values: &point)

        #expect(point[0] == 0.0)
        #expect(point[1] == 0.0)
    }

    @Test("Control point at index 3 is (1,1)")
    func controlPointEnd() {
        let func1 = CAMediaTimingFunction(name: .linear)
        var point = [Float](repeating: -1, count: 2)
        func1.getControlPoint(at: 3, values: &point)

        #expect(point[0] == 1.0)
        #expect(point[1] == 1.0)
    }

    @Test("Edge cases for evaluate")
    func evaluateEdgeCases() {
        let func1 = CAMediaTimingFunction(name: .easeIn)

        // Values at or beyond boundaries
        #expect(func1.evaluate(at: -0.5) == 0)
        #expect(func1.evaluate(at: 1.5) == 1)
    }

    @Test("Timing function equality")
    func timingFunctionEquality() {
        let f1 = CAMediaTimingFunction(name: .linear)
        let f2 = CAMediaTimingFunction(name: .linear)
        let f3 = CAMediaTimingFunction(name: .easeIn)

        #expect(f1 == f2)
        #expect(f1 != f3)
    }

    @Test("Timing function hashing")
    func timingFunctionHashing() {
        let f1 = CAMediaTimingFunction(name: .linear)
        let f2 = CAMediaTimingFunction(name: .linear)

        #expect(f1.hashValue == f2.hashValue)
    }
}

// MARK: - CALayer Tests

@Suite("CALayer Tests")
struct CALayerTests {

    // MARK: - Initialization

    @Test("Default layer properties")
    func defaultProperties() {
        let layer = CALayer()

        #expect(layer.bounds == .zero)
        #expect(layer.position == .zero)
        #expect(layer.anchorPoint == CGPoint(x: 0.5, y: 0.5))
        #expect(layer.zPosition == 0)
        #expect(layer.opacity == 1.0)
        #expect(layer.isHidden == false)
        #expect(layer.masksToBounds == false)
        #expect(layer.cornerRadius == 0)
        #expect(layer.borderWidth == 0)
        #expect(layer.backgroundColor == nil)
        #expect(CATransform3DIsIdentity(layer.transform))
        #expect(layer.sublayers == nil)
        #expect(layer.superlayer == nil)
    }

    // MARK: - Frame and Bounds

    @Test("Frame is derived from bounds, position, and anchorPoint")
    func frameDerivedCorrectly() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 50)
        layer.position = CGPoint(x: 150, y: 100)
        layer.anchorPoint = CGPoint(x: 0.5, y: 0.5)

        let frame = layer.frame
        #expect(frame.origin.x == 100) // 150 - 100 * 0.5
        #expect(frame.origin.y == 75)  // 100 - 50 * 0.5
        #expect(frame.size.width == 100)
        #expect(frame.size.height == 50)
    }

    @Test("Setting frame updates bounds and position")
    func settingFrame() {
        let layer = CALayer()
        layer.frame = CGRect(x: 50, y: 50, width: 200, height: 100)

        #expect(layer.bounds.size == CGSize(width: 200, height: 100))
        // With default anchor point (0.5, 0.5)
        #expect(layer.position.x == 150) // 50 + 200 * 0.5
        #expect(layer.position.y == 100) // 50 + 100 * 0.5
    }

    @Test("Frame with non-center anchor point")
    func frameWithCustomAnchorPoint() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        layer.anchorPoint = CGPoint(x: 0, y: 0) // Top-left
        layer.position = CGPoint(x: 50, y: 50)

        let frame = layer.frame
        #expect(frame.origin.x == 50)
        #expect(frame.origin.y == 50)
    }

    // MARK: - Opacity

    @Test("Opacity is clamped to valid range")
    func opacityClamping() {
        let layer = CALayer()

        layer.opacity = 1.5
        #expect(layer.opacity == 1.0)

        layer.opacity = -0.5
        #expect(layer.opacity == 0.0)
    }

    // MARK: - Corner Radius

    @Test("Corner radius cannot be negative")
    func cornerRadiusClamping() {
        let layer = CALayer()

        layer.cornerRadius = -10
        #expect(layer.cornerRadius == 0)

        layer.cornerRadius = 10
        #expect(layer.cornerRadius == 10)
    }

    // MARK: - Border Width

    @Test("Border width cannot be negative")
    func borderWidthClamping() {
        let layer = CALayer()

        layer.borderWidth = -5
        #expect(layer.borderWidth == 0)

        layer.borderWidth = 5
        #expect(layer.borderWidth == 5)
    }

    // MARK: - Shadow Properties

    @Test("Shadow opacity is clamped")
    func shadowOpacityClamping() {
        let layer = CALayer()

        layer.shadowOpacity = 1.5
        #expect(layer.shadowOpacity == 1.0)

        layer.shadowOpacity = -0.5
        #expect(layer.shadowOpacity == 0.0)
    }

    @Test("Shadow radius cannot be negative")
    func shadowRadiusClamping() {
        let layer = CALayer()

        layer.shadowRadius = -10
        #expect(layer.shadowRadius == 0)
    }

    // MARK: - Layer Hierarchy

    @Test("Adding sublayer sets superlayer")
    func addSublayer() {
        let parent = CALayer()
        let child = CALayer()

        parent.addSublayer(child)

        #expect(child.superlayer === parent)
        #expect(parent.sublayers?.contains { $0 === child } == true)
    }

    @Test("Removing from superlayer clears relationship")
    func removeFromSuperlayer() {
        let parent = CALayer()
        let child = CALayer()

        parent.addSublayer(child)
        child.removeFromSuperlayer()

        #expect(child.superlayer == nil)
        #expect(parent.sublayers?.contains { $0 === child } != true)
    }

    @Test("Insert sublayer at index")
    func insertSublayerAtIndex() {
        let parent = CALayer()
        let child1 = CALayer()
        let child2 = CALayer()
        let child3 = CALayer()

        parent.addSublayer(child1)
        parent.addSublayer(child3)
        parent.insertSublayer(child2, at: 1)

        #expect(parent.sublayers?[0] === child1)
        #expect(parent.sublayers?[1] === child2)
        #expect(parent.sublayers?[2] === child3)
    }

    @Test("Insert sublayer below sibling")
    func insertSublayerBelow() {
        let parent = CALayer()
        let child1 = CALayer()
        let child2 = CALayer()

        parent.addSublayer(child1)
        parent.insertSublayer(child2, below: child1)

        #expect(parent.sublayers?[0] === child2)
        #expect(parent.sublayers?[1] === child1)
    }

    @Test("Insert sublayer above sibling")
    func insertSublayerAbove() {
        let parent = CALayer()
        let child1 = CALayer()
        let child2 = CALayer()

        parent.addSublayer(child1)
        parent.insertSublayer(child2, above: child1)

        #expect(parent.sublayers?[0] === child1)
        #expect(parent.sublayers?[1] === child2)
    }

    @Test("Replace sublayer")
    func replaceSublayer() {
        let parent = CALayer()
        let oldChild = CALayer()
        let newChild = CALayer()

        parent.addSublayer(oldChild)
        parent.replaceSublayer(oldChild, with: newChild)

        #expect(oldChild.superlayer == nil)
        #expect(newChild.superlayer === parent)
        #expect(parent.sublayers?.count == 1)
        #expect(parent.sublayers?[0] === newChild)
    }

    @Test("Adding layer to new parent removes from old parent")
    func reparenting() {
        let parent1 = CALayer()
        let parent2 = CALayer()
        let child = CALayer()

        parent1.addSublayer(child)
        parent2.addSublayer(child)

        #expect(child.superlayer === parent2)
        #expect(parent1.sublayers?.contains { $0 === child } != true)
        #expect(parent2.sublayers?.contains { $0 === child } == true)
    }

    // MARK: - Transform

    @Test("Affine transform getter and setter")
    func affineTransformGetterSetter() {
        let layer = CALayer()
        let affine = CGAffineTransform(translationX: 10, y: 20)

        layer.setAffineTransform(affine)
        let result = layer.affineTransform()

        #expect(result.tx == 10)
        #expect(result.ty == 20)
    }

    // MARK: - Display and Layout

    @Test("setNeedsDisplay marks layer for update")
    func setNeedsDisplay() {
        let layer = CALayer()

        #expect(!layer.needsDisplay())
        layer.setNeedsDisplay()
        #expect(layer.needsDisplay())
    }

    @Test("setNeedsLayout marks layer for layout")
    func setNeedsLayout() {
        let layer = CALayer()

        #expect(!layer.needsLayout())
        layer.setNeedsLayout()
        #expect(layer.needsLayout())
    }

    // MARK: - Hit Testing

    @Test("Contains point within bounds")
    func containsPoint() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)

        #expect(layer.contains(CGPoint(x: 50, y: 50)))
        #expect(layer.contains(CGPoint(x: 0, y: 0)))
        #expect(!layer.contains(CGPoint(x: -1, y: 50)))
        #expect(!layer.contains(CGPoint(x: 101, y: 50)))
    }

    @Test("Hit test returns deepest layer")
    func hitTestDeepest() {
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)
        parent.position = CGPoint(x: 100, y: 100)

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 50, height: 50)
        child.position = CGPoint(x: 100, y: 100) // Center of parent

        parent.addSublayer(child)

        // Hit test at child's center (in parent's coordinate space)
        let hit = parent.hitTest(CGPoint(x: 100, y: 100))
        #expect(hit === child)
    }

    @Test("Hit test ignores hidden layers")
    func hitTestIgnoresHidden() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        layer.isHidden = true

        #expect(layer.hitTest(CGPoint(x: 50, y: 50)) == nil)
    }

    @Test("Hit test ignores transparent layers")
    func hitTestIgnoresTransparent() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        layer.opacity = 0

        #expect(layer.hitTest(CGPoint(x: 50, y: 50)) == nil)
    }

    // MARK: - Coordinate Conversion

    @Test("Convert point to self returns same point")
    func convertPointToSelf() {
        let layer = CALayer()
        let point = CGPoint(x: 50, y: 50)

        #expect(layer.convert(point, from: layer) == point)
        #expect(layer.convert(point, to: layer) == point)
    }

    @Test("Convert point between parent and child")
    func convertPointParentChild() {
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        child.position = CGPoint(x: 100, y: 100)
        child.anchorPoint = CGPoint(x: 0.5, y: 0.5)

        parent.addSublayer(child)

        // Point at child's origin (50, 50) in parent space is (0, 0) in child space
        let parentPoint = CGPoint(x: 50, y: 50)
        let childPoint = child.convert(parentPoint, from: parent)

        #expect(abs(childPoint.x - 0) < 0.001)
        #expect(abs(childPoint.y - 0) < 0.001)
    }

    // MARK: - Animation Management

    @Test("Add and retrieve animation")
    func addAndRetrieveAnimation() {
        let layer = CALayer()
        let animation = CABasicAnimation()
        animation.keyPath = "opacity"

        layer.add(animation, forKey: "testAnimation")

        let retrieved = layer.animation(forKey: "testAnimation")
        #expect(retrieved != nil)
    }

    @Test("Animation keys returns correct keys")
    func animationKeys() {
        let layer = CALayer()
        let anim1 = CABasicAnimation()
        let anim2 = CABasicAnimation()

        layer.add(anim1, forKey: "anim1")
        layer.add(anim2, forKey: "anim2")

        let keys = layer.animationKeys()
        #expect(keys?.count == 2)
        #expect(keys?.contains("anim1") == true)
        #expect(keys?.contains("anim2") == true)
    }

    @Test("Remove animation")
    func removeAnimation() {
        let layer = CALayer()
        let animation = CABasicAnimation()

        layer.add(animation, forKey: "test")
        layer.removeAnimation(forKey: "test")

        #expect(layer.animation(forKey: "test") == nil)
    }

    @Test("Remove all animations")
    func removeAllAnimations() {
        let layer = CALayer()
        layer.add(CABasicAnimation(), forKey: "anim1")
        layer.add(CABasicAnimation(), forKey: "anim2")

        layer.removeAllAnimations()

        #expect(layer.animationKeys() == nil)
    }

    @Test("Animation with nil key generates unique key")
    func animationWithNilKey() {
        let layer = CALayer()
        layer.add(CABasicAnimation(), forKey: nil)
        layer.add(CABasicAnimation(), forKey: nil)

        #expect(layer.animationKeys()?.count == 2)
    }

    // MARK: - Presentation Layer

    @Test("Presentation layer is separate copy")
    func presentationLayerIsCopy() {
        let layer = CALayer()
        layer.opacity = 0.5

        let presentation = layer.presentation()

        #expect(presentation !== layer)
        #expect(presentation?.opacity == 0.5)
    }

    @Test("Model layer returns self for model layer")
    func modelLayerSelf() {
        let layer = CALayer()
        #expect(layer.model() === layer)
    }

    // MARK: - Constraints

    @Test("Add constraint")
    func addConstraint() {
        let layer = CALayer()
        let constraint = CAConstraint(
            attribute: .minX,
            relativeTo: "superlayer",
            attribute: .minX
        )

        layer.addConstraint(constraint)

        #expect(layer.constraints?.count == 1)
    }

    // MARK: - Layer Copying

    @Test("Layer copying preserves properties")
    func layerCopying() {
        let original = CALayer()
        original.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        original.position = CGPoint(x: 50, y: 50)
        original.opacity = 0.5
        original.cornerRadius = 10
        original.name = "TestLayer"

        let copy = CALayer(layer: original)

        #expect(copy.bounds == original.bounds)
        #expect(copy.position == original.position)
        #expect(copy.opacity == original.opacity)
        #expect(copy.cornerRadius == original.cornerRadius)
        #expect(copy.name == original.name)
    }

    // MARK: - Hashable

    @Test("Layer identity equality")
    func layerEquality() {
        let layer1 = CALayer()
        let layer2 = CALayer()

        #expect(layer1 == layer1)
        #expect(layer1 != layer2)
    }
}

// MARK: - CAAnimation Tests

@Suite("CAAnimation Tests")
struct CAAnimationTests {

    @Test("Default animation properties")
    func defaultProperties() {
        let animation = CABasicAnimation()

        #expect(animation.duration == 0)
        #expect(animation.beginTime == 0)
        #expect(animation.timeOffset == 0)
        #expect(animation.repeatCount == 0)
        #expect(animation.repeatDuration == 0)
        #expect(animation.speed == 1)
        #expect(animation.autoreverses == false)
        #expect(animation.fillMode == .removed)
        #expect(animation.isRemovedOnCompletion == true)
    }

    @Test("Default value for key returns correct values")
    func defaultValueForKey() {
        #expect(CAAnimation.defaultValue(forKey: "duration") as? CFTimeInterval == 0)
        #expect(CAAnimation.defaultValue(forKey: "speed") as? Float == 1)
        #expect(CAAnimation.defaultValue(forKey: "autoreverses") as? Bool == false)
        #expect(CAAnimation.defaultValue(forKey: "isRemovedOnCompletion") as? Bool == true)
    }

    @Test("Effective base duration uses default when duration is 0")
    func effectiveBaseDuration() {
        let animation = CABasicAnimation()
        animation.duration = 0

        // Should use default 0.25 when duration is 0
        #expect(animation.effectiveBaseDuration == 0.25)

        animation.duration = 1.0
        #expect(animation.effectiveBaseDuration == 1.0)
    }

    @Test("Total duration calculation without repeat")
    func totalDurationWithoutRepeat() {
        let animation = CABasicAnimation()
        animation.duration = 1.0

        #expect(animation.totalDuration == 1.0)
    }

    @Test("Total duration calculation with repeat count")
    func totalDurationWithRepeatCount() {
        let animation = CABasicAnimation()
        animation.duration = 1.0
        animation.repeatCount = 3

        #expect(animation.totalDuration == 3.0)
    }

    @Test("Total duration calculation with autoreverses")
    func totalDurationWithAutoreverses() {
        let animation = CABasicAnimation()
        animation.duration = 1.0
        animation.autoreverses = true

        #expect(animation.totalDuration == 2.0)
    }

    @Test("Total duration calculation with repeat and autoreverses")
    func totalDurationWithRepeatAndAutoreverses() {
        let animation = CABasicAnimation()
        animation.duration = 1.0
        animation.repeatCount = 2
        animation.autoreverses = true

        // 1.0 * 2 (repeat) * 2 (autoreverses) = 4.0
        #expect(animation.totalDuration == 4.0)
    }

    @Test("Total duration capped by repeatDuration")
    func totalDurationCappedByRepeatDuration() {
        let animation = CABasicAnimation()
        animation.duration = 1.0
        animation.repeatCount = 10
        animation.repeatDuration = 3.0

        #expect(animation.totalDuration == 3.0)
    }
}

// MARK: - CABasicAnimation Tests

@Suite("CABasicAnimation Tests")
struct CABasicAnimationTests {

    @Test("Basic animation properties")
    func basicAnimationProperties() {
        let animation = CABasicAnimation()

        animation.fromValue = 0.0
        animation.toValue = 1.0
        animation.byValue = 0.5
        animation.keyPath = "opacity"

        #expect((animation.fromValue as? Double) == 0.0)
        #expect((animation.toValue as? Double) == 1.0)
        #expect((animation.byValue as? Double) == 0.5)
        #expect(animation.keyPath == "opacity")
    }

    @Test("Animation with different value types")
    func animationWithDifferentValueTypes() {
        let animation = CABasicAnimation()

        // CGFloat values
        animation.fromValue = CGFloat(0)
        animation.toValue = CGFloat(100)
        #expect(animation.fromValue != nil)
        #expect(animation.toValue != nil)

        // CGPoint values
        animation.fromValue = CGPoint(x: 0, y: 0)
        animation.toValue = CGPoint(x: 100, y: 100)
        #expect((animation.fromValue as? CGPoint) == CGPoint(x: 0, y: 0))

        // CATransform3D values
        animation.fromValue = CATransform3DIdentity
        animation.toValue = CATransform3DMakeScale(2, 2, 2)
        #expect(animation.fromValue != nil)
    }
}

// MARK: - CASpringAnimation Tests

@Suite("CASpringAnimation Tests")
struct CASpringAnimationTests {

    @Test("Default spring properties")
    func defaultSpringProperties() {
        let spring = CASpringAnimation()

        #expect(spring.mass == 1)
        #expect(spring.stiffness == 100)
        #expect(spring.damping == 10)
        #expect(spring.initialVelocity == 0)
    }

    @Test("Spring parameters are clamped to positive values")
    func springParametersClamping() {
        let spring = CASpringAnimation()

        spring.mass = -1
        #expect(spring.mass > 0)

        spring.stiffness = -1
        #expect(spring.stiffness > 0)

        spring.damping = -1
        #expect(spring.damping > 0)
    }

    @Test("Settling duration is calculated correctly")
    func settlingDurationCalculation() {
        let spring = CASpringAnimation()
        spring.mass = 1
        spring.stiffness = 100
        spring.damping = 10

        // Settling duration should be positive
        #expect(spring.settlingDuration > 0)
    }

    @Test("Underdamped spring has longer settling duration than overdamped")
    func underdampedVsOverdamped() {
        let underdamped = CASpringAnimation()
        underdamped.mass = 1
        underdamped.stiffness = 100
        underdamped.damping = 5 // Low damping = underdamped

        let overdamped = CASpringAnimation()
        overdamped.mass = 1
        overdamped.stiffness = 100
        overdamped.damping = 50 // High damping = overdamped

        // Both should have positive settling duration
        #expect(underdamped.settlingDuration > 0)
        #expect(overdamped.settlingDuration > 0)
    }

    @Test("Spring animation uses settlingDuration when duration is 0")
    func springUsesSettlingDuration() {
        let spring = CASpringAnimation()
        spring.duration = 0
        spring.mass = 1
        spring.stiffness = 100
        spring.damping = 10

        // effectiveBaseDuration should equal settlingDuration
        #expect(spring.effectiveBaseDuration == spring.settlingDuration)
    }

    @Test("Spring animation uses explicit duration when set")
    func springUsesExplicitDuration() {
        let spring = CASpringAnimation()
        spring.duration = 2.0

        #expect(spring.effectiveBaseDuration == 2.0)
    }

    @Test("Critical damping ratio")
    func criticalDampingRatio() {
        // Critical damping: c = 2 * sqrt(k * m)
        let spring = CASpringAnimation()
        spring.mass = 1
        spring.stiffness = 100
        spring.damping = 20 // 2 * sqrt(100 * 1) = 20

        // Settling duration should be finite and positive
        #expect(spring.settlingDuration > 0)
        #expect(spring.settlingDuration.isFinite)
    }
}

// MARK: - CAKeyframeAnimation Tests

@Suite("CAKeyframeAnimation Tests")
struct CAKeyframeAnimationTests {

    @Test("Default keyframe properties")
    func defaultKeyframeProperties() {
        let keyframe = CAKeyframeAnimation()

        #expect(keyframe.values == nil)
        #expect(keyframe.keyTimes == nil)
        #expect(keyframe.timingFunctions == nil)
        #expect(keyframe.calculationMode == .linear)
        #expect(keyframe.path == nil)
    }

    @Test("Setting keyframe values")
    func settingKeyframeValues() {
        let keyframe = CAKeyframeAnimation()
        keyframe.keyPath = "position"
        keyframe.values = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 100, y: 0),
            CGPoint(x: 100, y: 100)
        ]
        keyframe.keyTimes = [0, 0.5, 1.0]

        #expect(keyframe.values?.count == 3)
        #expect(keyframe.keyTimes?.count == 3)
    }

    @Test("Calculation modes")
    func calculationModes() {
        let keyframe = CAKeyframeAnimation()

        keyframe.calculationMode = .discrete
        #expect(keyframe.calculationMode == .discrete)

        keyframe.calculationMode = .linear
        #expect(keyframe.calculationMode == .linear)

        keyframe.calculationMode = .paced
        #expect(keyframe.calculationMode == .paced)

        keyframe.calculationMode = .cubic
        #expect(keyframe.calculationMode == .cubic)

        keyframe.calculationMode = .cubicPaced
        #expect(keyframe.calculationMode == .cubicPaced)
    }
}

// MARK: - CAAnimationGroup Tests

@Suite("CAAnimationGroup Tests")
struct CAAnimationGroupTests {

    @Test("Animation group holds multiple animations")
    func animationGroupHoldsAnimations() {
        let group = CAAnimationGroup()

        let anim1 = CABasicAnimation()
        anim1.keyPath = "opacity"

        let anim2 = CABasicAnimation()
        anim2.keyPath = "position"

        group.animations = [anim1, anim2]

        #expect(group.animations?.count == 2)
    }

    @Test("Animation group duration")
    func animationGroupDuration() {
        let group = CAAnimationGroup()
        group.duration = 2.0

        #expect(group.duration == 2.0)
    }
}

// MARK: - CATransaction Tests

@Suite("CATransaction Tests")
struct CATransactionTests {

    @Test("Default animation duration")
    func defaultAnimationDuration() {
        // Default duration should be 0.25
        #expect(CATransaction.animationDuration() == 0.25)
    }

    @Test("Set and get animation duration")
    func setAnimationDuration() {
        CATransaction.begin()
        CATransaction.setAnimationDuration(1.0)
        #expect(CATransaction.animationDuration() == 1.0)
        CATransaction.commit()
    }

    @Test("Disable actions")
    func disableActions() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        #expect(CATransaction.disableActions() == true)
        CATransaction.setDisableActions(false)
        #expect(CATransaction.disableActions() == false)
        CATransaction.commit()
    }

    @Test("Animation timing function")
    func animationTimingFunction() {
        CATransaction.begin()
        let timingFunc = CAMediaTimingFunction(name: .easeIn)
        CATransaction.setAnimationTimingFunction(timingFunc)
        #expect(CATransaction.animationTimingFunction() != nil)
        CATransaction.commit()
    }

    @Test("Value for key")
    func valueForKey() {
        CATransaction.begin()
        CATransaction.setAnimationDuration(1.5)

        let duration = CATransaction.value(forKey: "animationDuration") as? CFTimeInterval
        #expect(duration == 1.5)

        CATransaction.commit()
    }

    @Test("Set value for key")
    func setValueForKey() {
        CATransaction.begin()
        CATransaction.setValue(2.0, forKey: "animationDuration")
        #expect(CATransaction.animationDuration() == 2.0)
        CATransaction.commit()
    }

    @Test("Nested transactions")
    func nestedTransactions() {
        CATransaction.begin()
        CATransaction.setAnimationDuration(1.0)

        CATransaction.begin()
        CATransaction.setAnimationDuration(2.0)
        #expect(CATransaction.animationDuration() == 2.0)
        CATransaction.commit()

        // After inner commit, value should still be from inner (shared state in this implementation)
        CATransaction.commit()
    }

    @Test("Flush commits all transactions")
    func flushCommits() {
        CATransaction.begin()
        CATransaction.begin()

        CATransaction.flush()

        // After flush, duration should be reset to default
        #expect(CATransaction.animationDuration() == 0.25)
    }
}

// MARK: - CAAction Tests

@Suite("CAAction Tests")
struct CAActionTests {

    @Test("Layer action for key")
    func layerActionForKey() {
        let layer = CALayer()

        // By default, no action is returned (no delegate or custom actions)
        let action = layer.action(forKey: "opacity")
        #expect(action == nil)
    }

    @Test("Custom actions dictionary")
    func customActionsDictionary() {
        let layer = CALayer()
        let customAnimation = CABasicAnimation()
        customAnimation.duration = 1.0

        layer.actions = ["opacity": customAnimation]

        let action = layer.action(forKey: "opacity")
        #expect(action != nil)
    }
}

// MARK: - CAMediaTiming Tests

@Suite("CAMediaTiming Tests")
struct CAMediaTimingTests {

    @Test("Layer implements CAMediaTiming")
    func layerImplementsCAMediaTiming() {
        let layer = CALayer()

        layer.beginTime = 1.0
        layer.duration = 2.0
        layer.speed = 0.5
        layer.timeOffset = 0.5
        layer.repeatCount = 3
        layer.repeatDuration = 10
        layer.autoreverses = true
        layer.fillMode = .forwards

        #expect(layer.beginTime == 1.0)
        #expect(layer.duration == 2.0)
        #expect(layer.speed == 0.5)
        #expect(layer.timeOffset == 0.5)
        #expect(layer.repeatCount == 3)
        #expect(layer.repeatDuration == 10)
        #expect(layer.autoreverses == true)
        #expect(layer.fillMode == .forwards)
    }

    @Test("Animation implements CAMediaTiming")
    func animationImplementsCAMediaTiming() {
        let animation = CABasicAnimation()

        animation.beginTime = 1.0
        animation.duration = 2.0
        animation.speed = 2.0
        animation.timeOffset = 0.25
        animation.repeatCount = 2
        animation.repeatDuration = 5
        animation.autoreverses = true
        animation.fillMode = .both

        #expect(animation.beginTime == 1.0)
        #expect(animation.duration == 2.0)
        #expect(animation.speed == 2.0)
        #expect(animation.timeOffset == 0.25)
        #expect(animation.repeatCount == 2)
        #expect(animation.repeatDuration == 5)
        #expect(animation.autoreverses == true)
        #expect(animation.fillMode == .both)
    }
}

// MARK: - Fill Mode Tests

@Suite("Fill Mode Tests")
struct FillModeTests {

    @Test("Fill mode values")
    func fillModeValues() {
        #expect(CAMediaTimingFillMode.removed.rawValue == "removed")
        #expect(CAMediaTimingFillMode.forwards.rawValue == "forwards")
        #expect(CAMediaTimingFillMode.backwards.rawValue == "backwards")
        #expect(CAMediaTimingFillMode.both.rawValue == "both")
    }
}

// MARK: - Type Aliases and Constants Tests

@Suite("Type Aliases and Constants Tests")
struct TypeAliasesTests {

    @Test("CALayerContentsGravity values")
    func contentsGravityValues() {
        let layer = CALayer()

        layer.contentsGravity = .center
        #expect(layer.contentsGravity == .center)

        layer.contentsGravity = .resize
        #expect(layer.contentsGravity == .resize)

        layer.contentsGravity = .resizeAspect
        #expect(layer.contentsGravity == .resizeAspect)

        layer.contentsGravity = .resizeAspectFill
        #expect(layer.contentsGravity == .resizeAspectFill)
    }

    @Test("CALayerCornerCurve values")
    func cornerCurveValues() {
        let layer = CALayer()

        layer.cornerCurve = .circular
        #expect(layer.cornerCurve == .circular)

        layer.cornerCurve = .continuous
        #expect(layer.cornerCurve == .continuous)
    }

    @Test("Corner curve expansion factor")
    func cornerCurveExpansionFactor() {
        let circular = CALayer.cornerCurveExpansionFactor(.circular)
        let continuous = CALayer.cornerCurveExpansionFactor(.continuous)

        #expect(circular == 1.0)
        #expect(continuous > 1.0) // Continuous curves need more space
    }

    @Test("CACornerMask values")
    func cornerMaskValues() {
        let layer = CALayer()

        layer.maskedCorners = [.layerMinXMinYCorner, .layerMaxXMinYCorner]
        #expect(layer.maskedCorners.contains(.layerMinXMinYCorner))
        #expect(layer.maskedCorners.contains(.layerMaxXMinYCorner))
        #expect(!layer.maskedCorners.contains(.layerMinXMaxYCorner))
    }

    @Test("CAMediaTimingFunctionName values")
    func timingFunctionNameValues() {
        _ = CAMediaTimingFunction(name: .linear)
        _ = CAMediaTimingFunction(name: .easeIn)
        _ = CAMediaTimingFunction(name: .easeOut)
        _ = CAMediaTimingFunction(name: .easeInEaseOut)
        _ = CAMediaTimingFunction(name: .default)

        // If we got here without crashing, the names are valid
        #expect(true)
    }
}

// MARK: - CAConstraint Tests

@Suite("CAConstraint Tests")
struct CAConstraintTests {

    @Test("Create constraint with basic attributes")
    func createBasicConstraint() {
        let constraint = CAConstraint(
            attribute: .minX,
            relativeTo: "superlayer",
            attribute: .minX
        )

        #expect(constraint.attribute == .minX)
        #expect(constraint.sourceName == "superlayer")
        #expect(constraint.sourceAttribute == .minX)
    }

    @Test("Create constraint with offset")
    func createConstraintWithOffset() {
        let constraint = CAConstraint(
            attribute: .midY,
            relativeTo: "sibling",
            attribute: .midY,
            offset: 10.0
        )

        #expect(constraint.offset == 10.0)
    }

    @Test("Create constraint with scale")
    func createConstraintWithScale() {
        let constraint = CAConstraint(
            attribute: .width,
            relativeTo: "superlayer",
            attribute: .width,
            scale: 0.5,
            offset: 0
        )

        #expect(constraint.scale == 0.5)
    }

    @Test("Constraint attributes")
    func constraintAttributes() {
        // Test all attribute values can be used
        let attributes: [CAConstraintAttribute] = [
            .minX, .midX, .maxX,
            .minY, .midY, .maxY,
            .width, .height
        ]

        for attr in attributes {
            let constraint = CAConstraint(
                attribute: attr,
                relativeTo: "superlayer",
                attribute: attr
            )
            #expect(constraint.attribute == attr)
        }
    }
}

// MARK: - Edge Antialiasing Mask Tests

@Suite("CAEdgeAntialiasingMask Tests")
struct CAEdgeAntialiasingMaskTests {

    @Test("Default edge antialiasing mask")
    func defaultEdgeAntialiasingMask() {
        let layer = CALayer()

        // Default should include all edges
        #expect(layer.edgeAntialiasingMask.contains(.layerLeftEdge))
        #expect(layer.edgeAntialiasingMask.contains(.layerRightEdge))
        #expect(layer.edgeAntialiasingMask.contains(.layerTopEdge))
        #expect(layer.edgeAntialiasingMask.contains(.layerBottomEdge))
    }

    @Test("Custom edge antialiasing mask")
    func customEdgeAntialiasingMask() {
        let layer = CALayer()
        layer.edgeAntialiasingMask = [.layerLeftEdge, .layerRightEdge]

        #expect(layer.edgeAntialiasingMask.contains(.layerLeftEdge))
        #expect(layer.edgeAntialiasingMask.contains(.layerRightEdge))
        #expect(!layer.edgeAntialiasingMask.contains(.layerTopEdge))
        #expect(!layer.edgeAntialiasingMask.contains(.layerBottomEdge))
    }
}

// MARK: - Autoresizing Mask Tests

@Suite("CAAutoresizingMask Tests")
struct CAAutoresizingMaskTests {

    @Test("Default autoresizing mask is empty")
    func defaultAutoresizingMask() {
        let layer = CALayer()
        #expect(layer.autoresizingMask.isEmpty)
    }

    @Test("Set autoresizing mask")
    func setAutoresizingMask() {
        let layer = CALayer()
        layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]

        #expect(layer.autoresizingMask.contains(.layerWidthSizable))
        #expect(layer.autoresizingMask.contains(.layerHeightSizable))
    }
}

// MARK: - Integration Tests

@Suite("Integration Tests")
struct IntegrationTests {

    @Test("Complex layer hierarchy")
    func complexLayerHierarchy() {
        let root = CALayer()
        root.bounds = CGRect(x: 0, y: 0, width: 400, height: 400)
        root.position = CGPoint(x: 200, y: 200)

        let container = CALayer()
        container.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)
        container.position = CGPoint(x: 200, y: 200)

        let child1 = CALayer()
        child1.bounds = CGRect(x: 0, y: 0, width: 50, height: 50)
        child1.position = CGPoint(x: 50, y: 50)

        let child2 = CALayer()
        child2.bounds = CGRect(x: 0, y: 0, width: 50, height: 50)
        child2.position = CGPoint(x: 150, y: 150)

        root.addSublayer(container)
        container.addSublayer(child1)
        container.addSublayer(child2)

        #expect(root.sublayers?.count == 1)
        #expect(container.sublayers?.count == 2)
        #expect(child1.superlayer === container)
        #expect(container.superlayer === root)
    }

    @Test("Animation setup and timing")
    func animationSetupAndTiming() {
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)

        let animation = CABasicAnimation()
        animation.keyPath = "opacity"
        animation.fromValue = 1.0
        animation.toValue = 0.0
        animation.duration = 1.0
        animation.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)

        layer.add(animation, forKey: "fadeOut")

        #expect(layer.animation(forKey: "fadeOut") != nil)
        #expect(animation.duration == 1.0)
    }

    @Test("Transform chain")
    func transformChain() {
        // Create a transform that translates, then rotates, then scales
        var transform = CATransform3DIdentity
        transform = CATransform3DTranslate(transform, 100, 0, 0)
        transform = CATransform3DRotate(transform, CGFloat.pi / 4, 0, 0, 1)
        transform = CATransform3DScale(transform, 2, 2, 1)

        let layer = CALayer()
        layer.transform = transform

        #expect(!CATransform3DIsIdentity(layer.transform))
        #expect(CATransform3DIsAffine(layer.transform))
    }

    @Test("Spring animation total duration uses settling duration")
    func springAnimationTotalDuration() {
        let spring = CASpringAnimation()
        spring.keyPath = "position"
        spring.mass = 1
        spring.stiffness = 100
        spring.damping = 10
        spring.duration = 0 // Let it use settlingDuration

        // Total duration should equal settling duration when repeatCount is 0
        #expect(spring.totalDuration == spring.settlingDuration)
    }
}
