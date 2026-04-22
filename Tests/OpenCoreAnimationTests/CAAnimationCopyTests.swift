//
//  CAAnimationCopyTests.swift
//  OpenCoreAnimationTests
//
//  Tests for CAAnimation.copy() polymorphism and CALayer.add(_:forKey:) defensive copy.
//  These tests cover the fix that ensures:
//   - copy() preserves the concrete subclass identity
//   - subclass-specific properties are duplicated
//   - CALayer.add(_:forKey:) stores a copy, so mutating the original does not
//     leak into the layer's stored animation
//
//  This file intentionally does NOT import CoreGraphics, because doing so
//  re-triggers the pre-existing CGSize.== ambiguity pulled in via
//  CAMetalRenderer + MetalKit. The umbrella module already re-exports
//  OpenCoreGraphics, so CGFloat / CGRect / CGPoint are in scope via
//  OpenCoreAnimation alone.
//

import Testing
import OpenCoreAnimation

@Suite("CAAnimation.copy polymorphism")
struct CAAnimationCopyPolymorphismTests {

    @Test("CABasicAnimation.copy() returns a CABasicAnimation")
    func basicAnimationCopyPreservesType() {
        let original = CABasicAnimation()
        let copied = original.copy()
        #expect(type(of: copied) == CABasicAnimation.self)
    }

    @Test("CAKeyframeAnimation.copy() returns a CAKeyframeAnimation")
    func keyframeAnimationCopyPreservesType() {
        let original = CAKeyframeAnimation()
        let copied = original.copy()
        #expect(type(of: copied) == CAKeyframeAnimation.self)
    }

    @Test("CASpringAnimation.copy() returns a CASpringAnimation")
    func springAnimationCopyPreservesType() {
        let original = CASpringAnimation()
        let copied = original.copy()
        #expect(type(of: copied) == CASpringAnimation.self)
    }

    @Test("CATransition.copy() returns a CATransition")
    func transitionCopyPreservesType() {
        let original = CATransition()
        let copied = original.copy()
        #expect(type(of: copied) == CATransition.self)
    }

    @Test("CAAnimationGroup.copy() returns a CAAnimationGroup")
    func animationGroupCopyPreservesType() {
        let original = CAAnimationGroup()
        let copied = original.copy()
        #expect(type(of: copied) == CAAnimationGroup.self)
    }

    @Test("CAPropertyAnimation.copy() returns a CAPropertyAnimation")
    func propertyAnimationCopyPreservesType() {
        let original = CAPropertyAnimation()
        let copied = original.copy()
        #expect(type(of: copied) == CAPropertyAnimation.self)
    }
}

@Suite("CAAnimation.copy carries subclass state")
struct CAAnimationCopyStateTests {

    @Test("CABasicAnimation copies fromValue / toValue / byValue")
    func basicAnimationSubclassStateCopied() throws {
        let original = CABasicAnimation(keyPath: "opacity")
        original.fromValue = 0.25 as Double
        original.toValue = 0.75 as Double
        original.byValue = 0.5 as Double

        let copied = original.copy()

        // Mutate original — copy must be unaffected.
        original.fromValue = 99.0 as Double
        original.toValue = 99.0 as Double
        original.byValue = 99.0 as Double

        let copiedFrom = try #require(copied.fromValue as? Double)
        let copiedTo = try #require(copied.toValue as? Double)
        let copiedBy = try #require(copied.byValue as? Double)

        #expect(copiedFrom == 0.25)
        #expect(copiedTo == 0.75)
        #expect(copiedBy == 0.5)
    }

    @Test("CAPropertyAnimation keyPath / isAdditive / isCumulative are copied")
    func propertyAnimationStateCopied() {
        let original = CABasicAnimation(keyPath: "position")
        original.isAdditive = true
        original.isCumulative = true

        let copied = original.copy()

        original.keyPath = "bounds"
        original.isAdditive = false
        original.isCumulative = false

        #expect(copied.keyPath == "position")
        #expect(copied.isAdditive == true)
        #expect(copied.isCumulative == true)
    }

    @Test("CASpringAnimation copies spring parameters")
    func springAnimationStateCopied() {
        let original = CASpringAnimation()
        original.mass = 2
        original.stiffness = 200
        original.damping = 20
        original.initialVelocity = 5

        let copied = original.copy()

        original.mass = 1
        original.stiffness = 100
        original.damping = 10
        original.initialVelocity = 0

        #expect(copied.mass == 2)
        #expect(copied.stiffness == 200)
        #expect(copied.damping == 20)
        #expect(copied.initialVelocity == 5)
    }

    @Test("CAAnimationGroup deep-copies nested animations")
    func animationGroupDeepCopiesChildren() throws {
        let child = CABasicAnimation(keyPath: "opacity")
        child.fromValue = 0.0 as Double
        child.toValue = 1.0 as Double

        let group = CAAnimationGroup()
        group.animations = [child]

        let copied = group.copy()

        // Mutate the original child — the copied group's child must be
        // unaffected, which requires that copy() deep-copies the array.
        child.toValue = 9.0 as Double

        let copiedChildren = try #require(copied.animations)
        #expect(copiedChildren.count == 1)
        let copiedChild = try #require(copiedChildren.first as? CABasicAnimation)
        let copiedTo = try #require(copiedChild.toValue as? Double)
        #expect(copiedTo == 1.0)
    }

    @Test("CATransition.copy() preserves CATransition type even with default state")
    func transitionCopyKeepsSubtype() {
        let original = CATransition()
        let copied = original.copy()
        // The critical assertion for the recent fix is identity preservation;
        // specific CATransition properties vary by implementation.
        #expect(type(of: copied) == CATransition.self)
        #expect(copied !== original)
    }
}

@Suite("CALayer.add(_:forKey:) stores a defensive copy")
struct CALayerAddAnimationDefensiveCopyTests {

    @Test("Mutating the original after add() does not change the stored animation")
    func addStoresCopyNotReference() throws {
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.duration = 0.5
        animation.fromValue = 0.0 as Double
        animation.toValue = 1.0 as Double

        layer.add(animation, forKey: "fade")

        // Mutate the original after it's been added. If add() stored the
        // reference directly, these mutations would leak into the layer.
        animation.duration = 999
        animation.fromValue = 0.9 as Double
        animation.toValue = 0.1 as Double

        let stored = try #require(layer.animation(forKey: "fade") as? CABasicAnimation)
        #expect(stored.duration == 0.5)
        let storedFrom = try #require(stored.fromValue as? Double)
        let storedTo = try #require(stored.toValue as? Double)
        #expect(storedFrom == 0.0)
        #expect(storedTo == 1.0)
    }

    @Test("Stored animation is not the same instance as the one passed in")
    func addStoresDistinctInstance() throws {
        let layer = CALayer()
        let animation = CABasicAnimation(keyPath: "opacity")

        layer.add(animation, forKey: "fade")

        let stored = try #require(layer.animation(forKey: "fade"))
        #expect(stored !== animation)
    }

    @Test("Stored animation preserves concrete subclass after defensive copy")
    func addPreservesSubclassThroughCopy() throws {
        let layer = CALayer()
        let spring = CASpringAnimation()
        spring.keyPath = "position.x"
        spring.stiffness = 250
        layer.add(spring, forKey: "springy")

        let stored = try #require(layer.animation(forKey: "springy"))
        #expect(type(of: stored) == CASpringAnimation.self)
        let storedSpring = try #require(stored as? CASpringAnimation)
        #expect(storedSpring.stiffness == 250)
    }
}
