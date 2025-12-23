//
//  CALayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics


/// An object that manages image-based content and allows you to perform animations on that content.
///
/// Layers are often used to provide the backing store for views but can also be used without a view
/// to display content. A layer's main job is to manage the visual content that you provide but the
/// layer itself has visual attributes that can be set, such as a background color, border, and shadow.
open class CALayer: CAMediaTiming, Hashable {

    // MARK: - Initialization

    /// Returns an initialized CALayer object.
    public required init() {}

    /// Override to copy or initialize custom fields of the specified layer.
    ///
    /// - Parameter layer: The layer from which custom fields should be copied.
    public required init(layer: Any) {
        if let otherLayer = layer as? CALayer {
            // Copy geometry properties
            self._bounds = otherLayer._bounds
            self._position = otherLayer._position
            self._anchorPoint = otherLayer._anchorPoint
            self._zPosition = otherLayer._zPosition
            self._anchorPointZ = otherLayer._anchorPointZ
            self._transform = otherLayer._transform
            self._sublayerTransform = otherLayer._sublayerTransform
            self._contentsScale = otherLayer._contentsScale

            // Copy appearance properties
            self._opacity = otherLayer._opacity
            self._isHidden = otherLayer._isHidden
            self._masksToBounds = otherLayer._masksToBounds
            self._isDoubleSided = otherLayer._isDoubleSided
            self._cornerRadius = otherLayer._cornerRadius
            self.maskedCorners = otherLayer.maskedCorners
            self.cornerCurve = otherLayer.cornerCurve
            self._borderWidth = otherLayer._borderWidth
            self._borderColor = otherLayer._borderColor
            self._backgroundColor = otherLayer._backgroundColor
            self._shadowOpacity = otherLayer._shadowOpacity
            self._shadowRadius = otherLayer._shadowRadius
            self._shadowOffset = otherLayer._shadowOffset
            self._shadowColor = otherLayer._shadowColor
            self._shadowPath = otherLayer._shadowPath

            // Copy content properties
            self.contents = otherLayer.contents
            self.contentsRect = otherLayer.contentsRect
            self.contentsCenter = otherLayer.contentsCenter
            self.contentsGravity = otherLayer.contentsGravity
            self.contentsFormat = otherLayer.contentsFormat

            // Copy rendering properties
            self.isOpaque = otherLayer.isOpaque
            self.isGeometryFlipped = otherLayer.isGeometryFlipped
            self.drawsAsynchronously = otherLayer.drawsAsynchronously
            self.shouldRasterize = otherLayer.shouldRasterize
            self.rasterizationScale = otherLayer.rasterizationScale
            self.allowsEdgeAntialiasing = otherLayer.allowsEdgeAntialiasing
            self.allowsGroupOpacity = otherLayer.allowsGroupOpacity
            self.edgeAntialiasingMask = otherLayer.edgeAntialiasingMask

            // Copy filter properties
            self.filters = otherLayer.filters
            self.compositingFilter = otherLayer.compositingFilter
            self.backgroundFilters = otherLayer.backgroundFilters
            self.minificationFilter = otherLayer.minificationFilter
            self.minificationFilterBias = otherLayer.minificationFilterBias
            self.magnificationFilter = otherLayer.magnificationFilter

            // Copy layout properties
            self.autoresizingMask = otherLayer.autoresizingMask
            self.needsDisplayOnBoundsChange = otherLayer.needsDisplayOnBoundsChange
            self.constraints = otherLayer.constraints

            // Copy timing properties
            self.beginTime = otherLayer.beginTime
            self.timeOffset = otherLayer.timeOffset
            self.repeatCount = otherLayer.repeatCount
            self.repeatDuration = otherLayer.repeatDuration
            self.duration = otherLayer.duration
            self.speed = otherLayer.speed
            self.autoreverses = otherLayer.autoreverses
            self.fillMode = otherLayer.fillMode

            // Copy identification
            self._name = otherLayer._name
            self.style = otherLayer.style

            // Note: We intentionally do NOT copy:
            // - delegate (weak reference, not owned)
            // - sublayers (hierarchy is not copied)
            // - superlayer (hierarchy relationship)
            // - mask (would create ownership issues)
            // - layoutManager (typically shared)
            // - actions (typically defined at class level)
            // - animations (presentation layer specific)
        }
    }

    // MARK: - Accessing Related Layer Objects

    /// The presentation layer associated with this layer during animations.
    private var _presentationLayer: CALayer?

    /// Whether this layer is a presentation layer.
    private var _isPresentation: Bool = false

    /// The model layer if this is a presentation layer.
    private weak var _modelLayer: CALayer?

    /// Returns a copy of the presentation layer object that represents the state of the layer
    /// as it currently appears onscreen.
    ///
    /// The presentation layer reflects the current state of any active animations. If there are
    /// no animations running, the presentation layer's values match the model layer's values.
    ///
    /// - Returns: A copy of the presentation layer, or `nil` if this layer hasn't been committed
    ///   to the render tree.
    open func presentation() -> Self? {
        // If we're already a presentation layer, return self
        if _isPresentation {
            return self
        }

        // Create or update presentation layer
        if _presentationLayer == nil {
            _presentationLayer = createPresentationLayer()
        }

        // Update presentation layer with current animated values
        updatePresentationLayer()

        return _presentationLayer as? Self
    }

    /// Creates a new presentation layer as a copy of this layer.
    private func createPresentationLayer() -> CALayer {
        let presentationClass = type(of: self)
        let presentation = presentationClass.init(layer: self)
        presentation._isPresentation = true
        presentation._modelLayer = self
        return presentation
    }

    /// Returns a presentation layer copy with animations evaluated at a specific time offset.
    ///
    /// This is used by CAReplicatorLayer to implement instanceDelay, where each
    /// replicated instance has its animations delayed by a fixed amount.
    ///
    /// - Parameter timeOffset: The time offset to subtract from the current time when evaluating animations.
    ///                         Positive values make the animation appear earlier (as if more time has passed).
    /// - Returns: A new presentation layer with animations evaluated at the offset time.
    public func presentationAtTimeOffset(_ timeOffset: CFTimeInterval) -> Self {
        // Create a new presentation layer copy
        let presentationClass = type(of: self)
        let presentation = presentationClass.init(layer: self)
        presentation._isPresentation = true
        presentation._modelLayer = self

        // Update with animations at the offset time
        let evaluationTime = CACurrentMediaTime() - timeOffset
        updatePresentationLayer(presentation, at: evaluationTime)

        return presentation
    }

    /// Updates the presentation layer with current animated values.
    private func updatePresentationLayer() {
        guard let presentation = _presentationLayer else { return }

        let currentTime = CACurrentMediaTime()
        updatePresentationLayer(presentation, at: currentTime)
    }

    /// Updates a presentation layer with animated values at a specific time.
    private func updatePresentationLayer(_ presentation: CALayer, at currentTime: CFTimeInterval) {

        // Copy current property values
        presentation._bounds = _bounds
        presentation._position = _position
        presentation._anchorPoint = _anchorPoint
        presentation._zPosition = _zPosition
        presentation._transform = _transform
        presentation._opacity = _opacity
        presentation._isHidden = _isHidden
        presentation._backgroundColor = _backgroundColor
        presentation._borderColor = _borderColor
        presentation._borderWidth = _borderWidth
        presentation._cornerRadius = _cornerRadius
        presentation._shadowColor = _shadowColor
        presentation._shadowOpacity = _shadowOpacity
        presentation._shadowOffset = _shadowOffset
        presentation._shadowRadius = _shadowRadius

        // Copy CAShapeLayer properties if applicable
        if let shapePresentation = presentation as? CAShapeLayer,
           let shapeSelf = self as? CAShapeLayer {
            shapePresentation._path = shapeSelf._path
            shapePresentation._fillColor = shapeSelf._fillColor
            shapePresentation._strokeColor = shapeSelf._strokeColor
            shapePresentation._strokeStart = shapeSelf._strokeStart
            shapePresentation._strokeEnd = shapeSelf._strokeEnd
            shapePresentation._lineWidth = shapeSelf._lineWidth
            shapePresentation._lineDashPhase = shapeSelf._lineDashPhase
            shapePresentation._miterLimit = shapeSelf._miterLimit
        }

        // Copy CATextLayer properties if applicable
        if let textPresentation = presentation as? CATextLayer,
           let textSelf = self as? CATextLayer {
            textPresentation.string = textSelf.string
            textPresentation.font = textSelf.font
            textPresentation._fontSize = textSelf._fontSize
            textPresentation._foregroundColor = textSelf._foregroundColor
            textPresentation.isWrapped = textSelf.isWrapped
            textPresentation.truncationMode = textSelf.truncationMode
            textPresentation.alignmentMode = textSelf.alignmentMode
            textPresentation.allowsFontSubpixelQuantization = textSelf.allowsFontSubpixelQuantization
        }

        // Copy CAGradientLayer properties if applicable
        if let gradientPresentation = presentation as? CAGradientLayer,
           let gradientSelf = self as? CAGradientLayer {
            gradientPresentation._colors = gradientSelf._colors
            gradientPresentation._locations = gradientSelf._locations
            gradientPresentation._startPoint = gradientSelf._startPoint
            gradientPresentation._endPoint = gradientSelf._endPoint
        }

        // Apply active animations
        for (_, animation) in _animations {
            applyAnimation(animation, to: presentation, at: currentTime)
        }
    }

    /// Applies an animation to the presentation layer at the given time.
    private func applyAnimation(_ animation: CAAnimation, to layer: CALayer, at time: CFTimeInterval) {
        // Handle animation groups
        if let animationGroup = animation as? CAAnimationGroup {
            applyAnimationGroup(animationGroup, to: layer, at: time)
            return
        }

        // Handle transitions
        if let transition = animation as? CATransition {
            applyTransition(transition, to: layer, at: time)
            return
        }

        guard let propertyAnimation = animation as? CAPropertyAnimation,
              let keyPath = propertyAnimation.keyPath else { return }

        // Calculate animation progress consistently with processAnimationCompletions
        // elapsed = currentTime - addedTime - beginTime
        let elapsed = time - animation.addedTime - animation.beginTime

        // Get the effective duration for a single cycle
        let singleCycleDuration: CFTimeInterval
        if let springAnimation = animation as? CASpringAnimation {
            // For spring animations, use settlingDuration if duration is not explicitly set
            singleCycleDuration = animation.duration > 0 ? animation.duration : springAnimation.settlingDuration
        } else {
            singleCycleDuration = animation.duration > 0 ? animation.duration : CATransaction.animationDuration()
        }

        guard singleCycleDuration > 0 else { return }

        // Handle fillMode for animations that haven't started yet
        if elapsed < 0 {
            switch animation.fillMode {
            case .backwards, .both:
                // Show the initial state before animation starts
                applyAnimationValue(propertyAnimation, to: layer, keyPath: keyPath, progress: 0)
            default:
                // Don't apply any animation value
                return
            }
            return
        }

        var progress = elapsed / singleCycleDuration

        // Handle repeat count and autoreverses
        if animation.repeatCount > 0 || animation.autoreverses {
            let totalCycles = animation.autoreverses ? 2.0 : 1.0
            let effectiveRepeatCount = max(1, CFTimeInterval(animation.repeatCount))
            let totalProgress = effectiveRepeatCount * totalCycles

            if progress >= totalProgress {
                // Animation has completed all cycles
                switch animation.fillMode {
                case .forwards, .both:
                    // Keep final state - determine if we ended forward or reversed
                    if animation.autoreverses && Int(effectiveRepeatCount * totalCycles) % 2 == 0 {
                        progress = 0 // Ended on reverse, back at start
                    } else {
                        progress = 1 // Ended on forward
                    }
                default:
                    // Animation completed, don't apply any value
                    return
                }
            } else {
                // Get progress within current cycle
                let cycleProgress = progress.truncatingRemainder(dividingBy: totalCycles)

                if animation.autoreverses {
                    // In autoreverse mode: 0-1 is forward, 1-2 is reverse
                    if cycleProgress < 1 {
                        progress = cycleProgress
                    } else {
                        // Reverse: map 1-2 to 1-0
                        progress = 2 - cycleProgress
                    }
                } else {
                    progress = cycleProgress
                }
            }
        } else if progress >= 1 {
            // No repeat, animation completed
            switch animation.fillMode {
            case .forwards, .both:
                progress = 1
            default:
                return
            }
        }

        // Apply spring physics or timing function
        if let springAnimation = animation as? CASpringAnimation {
            // For spring animations, use physics-based interpolation
            // Calculate the actual elapsed time for spring physics
            let springTime = progress * singleCycleDuration
            progress = CFTimeInterval(springAnimation.springValue(at: springTime))
            // Spring animations can overshoot, so don't clamp to 0-1
        } else if let timingFunction = animation.timingFunction {
            // Apply timing function for non-spring animations
            progress = CFTimeInterval(timingFunction.evaluate(at: Float(progress)))
            progress = max(0, min(1, progress))
        } else {
            progress = max(0, min(1, progress))
        }

        // Interpolate and apply value based on animation type
        applyAnimationValue(propertyAnimation, to: layer, keyPath: keyPath, progress: progress)
    }

    /// Applies a transition animation to the presentation layer.
    ///
    /// Transitions provide animated effects when layer content changes.
    /// Unlike property animations, transitions affect the overall appearance
    /// of the layer during the transition period.
    private func applyTransition(_ transition: CATransition, to layer: CALayer, at time: CFTimeInterval) {
        // Calculate transition progress
        let elapsed = time - transition.addedTime - transition.beginTime
        let duration = transition.duration > 0 ? transition.duration : CATransaction.animationDuration()

        guard duration > 0 else { return }

        // Handle fillMode for transitions that haven't started yet
        if elapsed < 0 {
            switch transition.fillMode {
            case .backwards, .both:
                applyTransitionEffect(transition, to: layer, progress: 0)
            default:
                return
            }
            return
        }

        // Calculate base progress (0-1)
        var progress = elapsed / duration

        // Handle completed transitions
        if progress >= 1 {
            switch transition.fillMode {
            case .forwards, .both:
                progress = 1
            default:
                return
            }
        }

        // Apply timing function if available
        if let timingFunction = transition.timingFunction {
            progress = CFTimeInterval(timingFunction.evaluate(at: Float(progress)))
        }

        // Apply start/end progress range
        let startProgress = CFTimeInterval(transition.startProgress)
        let endProgress = CFTimeInterval(transition.endProgress)
        let adjustedProgress = startProgress + progress * (endProgress - startProgress)

        // Apply the transition effect
        applyTransitionEffect(transition, to: layer, progress: adjustedProgress)
    }

    /// Applies the visual effect of a transition at a given progress.
    private func applyTransitionEffect(_ transition: CATransition, to layer: CALayer, progress: CFTimeInterval) {
        let clampedProgress = max(0, min(1, progress))

        switch transition.type {
        case .fade:
            // Fade: animate opacity from 0 to 1
            layer._opacity = Float(clampedProgress)

        case .push:
            // Push: new content pushes old content out
            // The direction determines which way the content moves
            let offset = calculateTransitionOffset(
                transition: transition,
                layerBounds: layer.bounds,
                progress: clampedProgress,
                isPush: true
            )
            layer._position = CGPoint(
                x: _position.x + offset.x,
                y: _position.y + offset.y
            )

        case .moveIn:
            // MoveIn: new content slides in over old content
            // Similar to push but old content stays in place
            let offset = calculateTransitionOffset(
                transition: transition,
                layerBounds: layer.bounds,
                progress: clampedProgress,
                isPush: false
            )
            layer._position = CGPoint(
                x: _position.x + offset.x,
                y: _position.y + offset.y
            )
            // Also fade in slightly for smoother effect
            layer._opacity = Float(clampedProgress)

        case .reveal:
            // Reveal: content is gradually revealed from a direction
            // This would typically use clipping, but we'll simulate with opacity/position
            layer._opacity = Float(clampedProgress)
            let offset = calculateTransitionOffset(
                transition: transition,
                layerBounds: layer.bounds,
                progress: clampedProgress,
                isPush: false
            )
            // Move content in opposite direction during reveal
            layer._position = CGPoint(
                x: _position.x - offset.x * 0.3,  // Subtle movement
                y: _position.y - offset.y * 0.3
            )

        default:
            // Default to fade for unknown transition types
            layer._opacity = Float(clampedProgress)
        }
    }

    /// Calculates the position offset for push/moveIn/reveal transitions.
    private func calculateTransitionOffset(
        transition: CATransition,
        layerBounds: CGRect,
        progress: CFTimeInterval,
        isPush: Bool
    ) -> CGPoint {
        // Calculate the offset based on subtype (direction)
        let width = layerBounds.width
        let height = layerBounds.height

        // For push, content moves from off-screen to on-screen
        // Progress 0 = fully off-screen, Progress 1 = fully on-screen
        let offsetMultiplier = CGFloat(1 - progress)

        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0

        switch transition.subtype {
        case .fromLeft:
            offsetX = -width * offsetMultiplier
        case .fromRight:
            offsetX = width * offsetMultiplier
        case .fromTop:
            offsetY = -height * offsetMultiplier
        case .fromBottom:
            offsetY = height * offsetMultiplier
        default:
            // Default to fromLeft if no subtype specified
            offsetX = -width * offsetMultiplier
        }

        return CGPoint(x: offsetX, y: offsetY)
    }

    /// Applies an animation group to the presentation layer.
    private func applyAnimationGroup(_ group: CAAnimationGroup, to layer: CALayer, at time: CFTimeInterval) {
        guard let animations = group.animations else { return }

        for animation in animations {
            // Create a context with inherited timing properties
            // Note: We don't modify the animation object directly to avoid side effects
            let effectiveDuration = animation.duration > 0 ? animation.duration : group.duration
            let effectiveAddedTime = animation.addedTime > 0 ? animation.addedTime : group.addedTime

            // Apply each child animation with effective timing
            applyAnimationWithContext(animation, to: layer, at: time,
                                      effectiveDuration: effectiveDuration,
                                      effectiveAddedTime: effectiveAddedTime)
        }
    }

    /// Applies an animation with explicit timing context (used by animation groups).
    private func applyAnimationWithContext(_ animation: CAAnimation, to layer: CALayer, at time: CFTimeInterval,
                                           effectiveDuration: CFTimeInterval, effectiveAddedTime: CFTimeInterval) {
        // Handle nested animation groups
        if let animationGroup = animation as? CAAnimationGroup {
            applyAnimationGroup(animationGroup, to: layer, at: time)
            return
        }

        guard let propertyAnimation = animation as? CAPropertyAnimation,
              let keyPath = propertyAnimation.keyPath else { return }

        // Calculate animation progress with effective timing
        let elapsed = time - effectiveAddedTime - animation.beginTime

        // Get the effective duration for a single cycle
        let singleCycleDuration: CFTimeInterval
        if let springAnimation = animation as? CASpringAnimation {
            singleCycleDuration = effectiveDuration > 0 ? effectiveDuration : springAnimation.settlingDuration
        } else {
            singleCycleDuration = effectiveDuration > 0 ? effectiveDuration : CATransaction.animationDuration()
        }

        guard singleCycleDuration > 0 else { return }

        // Handle fillMode for animations that haven't started yet
        if elapsed < 0 {
            switch animation.fillMode {
            case .backwards, .both:
                applyAnimationValue(propertyAnimation, to: layer, keyPath: keyPath, progress: 0)
            default:
                return
            }
            return
        }

        var progress = elapsed / singleCycleDuration

        // Handle repeat count and autoreverses (simplified for group context)
        if progress >= 1 {
            switch animation.fillMode {
            case .forwards, .both:
                progress = 1
            default:
                return
            }
        }

        // Apply spring physics or timing function
        if let springAnimation = animation as? CASpringAnimation {
            let springTime = progress * singleCycleDuration
            progress = CFTimeInterval(springAnimation.springValue(at: springTime))
        } else if let timingFunction = animation.timingFunction {
            progress = CFTimeInterval(timingFunction.evaluate(at: Float(progress)))
            progress = max(0, min(1, progress))
        } else {
            progress = max(0, min(1, progress))
        }

        applyAnimationValue(propertyAnimation, to: layer, keyPath: keyPath, progress: progress)
    }

    /// Applies an animation value to a layer property.
    private func applyAnimationValue(_ animation: CAPropertyAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        if let keyframeAnimation = animation as? CAKeyframeAnimation {
            applyKeyframeAnimation(keyframeAnimation, to: layer, keyPath: keyPath, progress: progress)
        } else if let basicAnimation = animation as? CABasicAnimation {
            applyBasicAnimation(basicAnimation, to: layer, keyPath: keyPath, progress: progress)
        }
    }

    /// Applies a basic animation to a layer property.
    private func applyBasicAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        // Check for valueFunction - used to convert scalar values to transforms
        if let valueFunction = animation.valueFunction {
            applyValueFunctionAnimation(animation, valueFunction: valueFunction, to: layer, progress: progress)
            return
        }

        // Handle common animatable properties
        applyFloatAnimation(animation, to: layer, keyPath: keyPath, progress: progress)
        applyPointAnimation(animation, to: layer, keyPath: keyPath, progress: progress)
        applyRectAnimation(animation, to: layer, keyPath: keyPath, progress: progress)
        applyTransformAnimation(animation, to: layer, keyPath: keyPath, progress: progress)
        applyColorAnimation(animation, to: layer, keyPath: keyPath, progress: progress)
        applyArrayAnimation(animation, to: layer, keyPath: keyPath, progress: progress)
    }

    /// Applies an animation using a value function to convert scalar values to transforms.
    ///
    /// Value functions allow animating transform properties using a single scalar value
    /// instead of specifying full CATransform3D values.
    private func applyValueFunctionAnimation(
        _ animation: CABasicAnimation,
        valueFunction: CAValueFunction,
        to layer: CALayer,
        progress: CFTimeInterval
    ) {
        // Get scalar values from the animation
        let fromValue: CGFloat
        if let from = animation.fromValue as? CGFloat {
            fromValue = from
        } else if let from = animation.fromValue as? Double {
            fromValue = CGFloat(from)
        } else if let from = animation.fromValue as? Float {
            fromValue = CGFloat(from)
        } else if let from = animation.fromValue as? Int {
            fromValue = CGFloat(from)
        } else {
            // Default starting value depends on the function type
            fromValue = defaultValueForFunction(valueFunction)
        }

        let toValue: CGFloat
        if let to = animation.toValue as? CGFloat {
            toValue = to
        } else if let to = animation.toValue as? Double {
            toValue = CGFloat(to)
        } else if let to = animation.toValue as? Float {
            toValue = CGFloat(to)
        } else if let to = animation.toValue as? Int {
            toValue = CGFloat(to)
        } else {
            // Default ending value depends on the function type
            toValue = defaultValueForFunction(valueFunction)
        }

        // Use the value function to interpolate and get the transform
        let transform = valueFunction.interpolate(from: fromValue, to: toValue, progress: CGFloat(progress))

        // Apply the transform to the layer
        // If the layer already has a transform, concatenate with it
        let baseTransform = _transform
        if CATransform3DIsIdentity(baseTransform) {
            layer._transform = transform
        } else {
            layer._transform = CATransform3DConcat(baseTransform, transform)
        }
    }

    /// Returns the default value for a given value function type.
    ///
    /// For rotation functions, 0 is the natural starting value.
    /// For scale functions, 1 is the natural starting value.
    /// For translation functions, 0 is the natural starting value.
    private func defaultValueForFunction(_ valueFunction: CAValueFunction) -> CGFloat {
        switch valueFunction.name {
        case .scale, .scaleX, .scaleY, .scaleZ:
            return 1.0  // Scale defaults to 1 (no scaling)
        case .rotateX, .rotateY, .rotateZ, .translateX, .translateY, .translateZ:
            return 0.0  // Rotation and translation default to 0
        default:
            return 0.0
        }
    }

    private func applyFloatAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        switch keyPath {
        case "opacity":
            guard let from = (animation.fromValue as? Float) ?? _opacity as Float?,
                  let to = (animation.toValue as? Float) ?? _opacity as Float? else { return }
            layer._opacity = from + Float(progress) * (to - from)
        case "cornerRadius":
            guard let from = (animation.fromValue as? CGFloat) ?? _cornerRadius as CGFloat?,
                  let to = (animation.toValue as? CGFloat) ?? _cornerRadius as CGFloat? else { return }
            layer._cornerRadius = from + CGFloat(progress) * (to - from)
        case "borderWidth":
            guard let from = (animation.fromValue as? CGFloat) ?? _borderWidth as CGFloat?,
                  let to = (animation.toValue as? CGFloat) ?? _borderWidth as CGFloat? else { return }
            layer._borderWidth = from + CGFloat(progress) * (to - from)
        case "shadowRadius":
            guard let from = (animation.fromValue as? CGFloat) ?? _shadowRadius as CGFloat?,
                  let to = (animation.toValue as? CGFloat) ?? _shadowRadius as CGFloat? else { return }
            layer._shadowRadius = from + CGFloat(progress) * (to - from)
        case "shadowOpacity":
            guard let from = (animation.fromValue as? Float) ?? _shadowOpacity as Float?,
                  let to = (animation.toValue as? Float) ?? _shadowOpacity as Float? else { return }
            layer._shadowOpacity = from + Float(progress) * (to - from)
        case "zPosition":
            guard let from = (animation.fromValue as? CGFloat) ?? _zPosition as CGFloat?,
                  let to = (animation.toValue as? CGFloat) ?? _zPosition as CGFloat? else { return }
            layer._zPosition = from + CGFloat(progress) * (to - from)

        // CAShapeLayer properties
        case "strokeStart":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            let from = (animation.fromValue as? CGFloat) ?? modelShapeLayer._strokeStart
            let to = (animation.toValue as? CGFloat) ?? modelShapeLayer._strokeStart
            shapeLayer._strokeStart = from + CGFloat(progress) * (to - from)
        case "strokeEnd":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            let from = (animation.fromValue as? CGFloat) ?? modelShapeLayer._strokeEnd
            let to = (animation.toValue as? CGFloat) ?? modelShapeLayer._strokeEnd
            shapeLayer._strokeEnd = from + CGFloat(progress) * (to - from)
        case "lineWidth":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            let from = (animation.fromValue as? CGFloat) ?? modelShapeLayer._lineWidth
            let to = (animation.toValue as? CGFloat) ?? modelShapeLayer._lineWidth
            shapeLayer._lineWidth = from + CGFloat(progress) * (to - from)
        case "lineDashPhase":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            let from = (animation.fromValue as? CGFloat) ?? modelShapeLayer._lineDashPhase
            let to = (animation.toValue as? CGFloat) ?? modelShapeLayer._lineDashPhase
            shapeLayer._lineDashPhase = from + CGFloat(progress) * (to - from)
        case "miterLimit":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            let from = (animation.fromValue as? CGFloat) ?? modelShapeLayer._miterLimit
            let to = (animation.toValue as? CGFloat) ?? modelShapeLayer._miterLimit
            shapeLayer._miterLimit = from + CGFloat(progress) * (to - from)

        default:
            break
        }
    }

    private func applyPointAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        switch keyPath {
        case "position":
            guard let from = (animation.fromValue as? CGPoint) ?? _position as CGPoint?,
                  let to = (animation.toValue as? CGPoint) ?? _position as CGPoint? else { return }
            layer._position = CGPoint(
                x: from.x + CGFloat(progress) * (to.x - from.x),
                y: from.y + CGFloat(progress) * (to.y - from.y)
            )
        case "position.x":
            guard let from = (animation.fromValue as? CGFloat) ?? _position.x as CGFloat?,
                  let to = (animation.toValue as? CGFloat) ?? _position.x as CGFloat? else { return }
            layer._position.x = from + CGFloat(progress) * (to - from)
        case "position.y":
            guard let from = (animation.fromValue as? CGFloat) ?? _position.y as CGFloat?,
                  let to = (animation.toValue as? CGFloat) ?? _position.y as CGFloat? else { return }
            layer._position.y = from + CGFloat(progress) * (to - from)
        case "anchorPoint":
            guard let from = (animation.fromValue as? CGPoint) ?? _anchorPoint as CGPoint?,
                  let to = (animation.toValue as? CGPoint) ?? _anchorPoint as CGPoint? else { return }
            layer._anchorPoint = CGPoint(
                x: from.x + CGFloat(progress) * (to.x - from.x),
                y: from.y + CGFloat(progress) * (to.y - from.y)
            )
        case "shadowOffset":
            guard let from = (animation.fromValue as? CGSize) ?? _shadowOffset as CGSize?,
                  let to = (animation.toValue as? CGSize) ?? _shadowOffset as CGSize? else { return }
            layer._shadowOffset = CGSize(
                width: from.width + CGFloat(progress) * (to.width - from.width),
                height: from.height + CGFloat(progress) * (to.height - from.height)
            )

        // CAGradientLayer properties
        case "startPoint":
            guard let gradientLayer = layer as? CAGradientLayer,
                  let modelGradientLayer = self as? CAGradientLayer else { return }
            let from = (animation.fromValue as? CGPoint) ?? modelGradientLayer._startPoint
            let to = (animation.toValue as? CGPoint) ?? modelGradientLayer._startPoint
            gradientLayer._startPoint = CGPoint(
                x: from.x + CGFloat(progress) * (to.x - from.x),
                y: from.y + CGFloat(progress) * (to.y - from.y)
            )
        case "endPoint":
            guard let gradientLayer = layer as? CAGradientLayer,
                  let modelGradientLayer = self as? CAGradientLayer else { return }
            let from = (animation.fromValue as? CGPoint) ?? modelGradientLayer._endPoint
            let to = (animation.toValue as? CGPoint) ?? modelGradientLayer._endPoint
            gradientLayer._endPoint = CGPoint(
                x: from.x + CGFloat(progress) * (to.x - from.x),
                y: from.y + CGFloat(progress) * (to.y - from.y)
            )

        default:
            break
        }
    }

    private func applyRectAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        switch keyPath {
        case "bounds":
            guard let from = (animation.fromValue as? CGRect) ?? _bounds as CGRect?,
                  let to = (animation.toValue as? CGRect) ?? _bounds as CGRect? else { return }
            layer._bounds = CGRect(
                x: from.origin.x + CGFloat(progress) * (to.origin.x - from.origin.x),
                y: from.origin.y + CGFloat(progress) * (to.origin.y - from.origin.y),
                width: from.size.width + CGFloat(progress) * (to.size.width - from.size.width),
                height: from.size.height + CGFloat(progress) * (to.size.height - from.size.height)
            )
        case "bounds.size":
            guard let from = (animation.fromValue as? CGSize) ?? _bounds.size as CGSize?,
                  let to = (animation.toValue as? CGSize) ?? _bounds.size as CGSize? else { return }
            layer._bounds.size = CGSize(
                width: from.width + CGFloat(progress) * (to.width - from.width),
                height: from.height + CGFloat(progress) * (to.height - from.height)
            )
        default:
            break
        }
    }

    private func applyTransformAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        // Get the model layer's base transform (self is the model layer)
        let baseTransform = _transform

        switch keyPath {
        case "transform":
            // Full transform animation: interpolate between from and to values
            guard let from = (animation.fromValue as? CATransform3D) ?? _transform as CATransform3D?,
                  let to = (animation.toValue as? CATransform3D) ?? _transform as CATransform3D? else { return }
            layer._transform = interpolateTransform(from: from, to: to, progress: CGFloat(progress))

        case "transform.scale":
            // Scale animation: apply scale to the base transform
            let from = (animation.fromValue as? CGFloat) ?? 1.0
            let to = (animation.toValue as? CGFloat) ?? 1.0
            let scale = from + CGFloat(progress) * (to - from)
            // Apply scale to base transform instead of replacing it
            layer._transform = CATransform3DScale(baseTransform, scale, scale, scale)

        case "transform.scale.x":
            // Scale X: modify only the X scale component
            let from = (animation.fromValue as? CGFloat) ?? 1.0
            let to = (animation.toValue as? CGFloat) ?? 1.0
            let scaleX = from + CGFloat(progress) * (to - from)
            // Apply X scale to base transform
            layer._transform = CATransform3DScale(baseTransform, scaleX, 1, 1)

        case "transform.scale.y":
            // Scale Y: modify only the Y scale component
            let from = (animation.fromValue as? CGFloat) ?? 1.0
            let to = (animation.toValue as? CGFloat) ?? 1.0
            let scaleY = from + CGFloat(progress) * (to - from)
            // Apply Y scale to base transform
            layer._transform = CATransform3DScale(baseTransform, 1, scaleY, 1)

        case "transform.scale.z":
            // Scale Z: modify only the Z scale component
            let from = (animation.fromValue as? CGFloat) ?? 1.0
            let to = (animation.toValue as? CGFloat) ?? 1.0
            let scaleZ = from + CGFloat(progress) * (to - from)
            // Apply Z scale to base transform
            layer._transform = CATransform3DScale(baseTransform, 1, 1, scaleZ)

        case "transform.rotation", "transform.rotation.z":
            // Rotation around Z axis: apply rotation to base transform
            let from = (animation.fromValue as? CGFloat) ?? 0
            let to = (animation.toValue as? CGFloat) ?? 0
            let angle = from + CGFloat(progress) * (to - from)
            // Apply rotation to base transform instead of replacing it
            layer._transform = CATransform3DRotate(baseTransform, angle, 0, 0, 1)

        case "transform.rotation.x":
            // Rotation around X axis
            let from = (animation.fromValue as? CGFloat) ?? 0
            let to = (animation.toValue as? CGFloat) ?? 0
            let angle = from + CGFloat(progress) * (to - from)
            layer._transform = CATransform3DRotate(baseTransform, angle, 1, 0, 0)

        case "transform.rotation.y":
            // Rotation around Y axis
            let from = (animation.fromValue as? CGFloat) ?? 0
            let to = (animation.toValue as? CGFloat) ?? 0
            let angle = from + CGFloat(progress) * (to - from)
            layer._transform = CATransform3DRotate(baseTransform, angle, 0, 1, 0)

        case "transform.translation":
            // Translation animation: apply translation to base transform
            guard let from = (animation.fromValue as? CGSize) ?? CGSize.zero as CGSize?,
                  let to = (animation.toValue as? CGSize) ?? CGSize.zero as CGSize? else { return }
            let tx = from.width + CGFloat(progress) * (to.width - from.width)
            let ty = from.height + CGFloat(progress) * (to.height - from.height)
            layer._transform = CATransform3DTranslate(baseTransform, tx, ty, 0)

        case "transform.translation.x":
            // Translation X: apply X translation to base transform
            let from = (animation.fromValue as? CGFloat) ?? 0
            let to = (animation.toValue as? CGFloat) ?? 0
            let tx = from + CGFloat(progress) * (to - from)
            layer._transform = CATransform3DTranslate(baseTransform, tx, 0, 0)

        case "transform.translation.y":
            // Translation Y: apply Y translation to base transform
            let from = (animation.fromValue as? CGFloat) ?? 0
            let to = (animation.toValue as? CGFloat) ?? 0
            let ty = from + CGFloat(progress) * (to - from)
            layer._transform = CATransform3DTranslate(baseTransform, 0, ty, 0)

        case "transform.translation.z":
            // Translation Z: apply Z translation to base transform
            let from = (animation.fromValue as? CGFloat) ?? 0
            let to = (animation.toValue as? CGFloat) ?? 0
            let tz = from + CGFloat(progress) * (to - from)
            layer._transform = CATransform3DTranslate(baseTransform, 0, 0, tz)

        default:
            break
        }
    }

    /// Interpolates between two transforms.
    private func interpolateTransform(from: CATransform3D, to: CATransform3D, progress: CGFloat) -> CATransform3D {
        return CATransform3D(
            m11: from.m11 + progress * (to.m11 - from.m11),
            m12: from.m12 + progress * (to.m12 - from.m12),
            m13: from.m13 + progress * (to.m13 - from.m13),
            m14: from.m14 + progress * (to.m14 - from.m14),
            m21: from.m21 + progress * (to.m21 - from.m21),
            m22: from.m22 + progress * (to.m22 - from.m22),
            m23: from.m23 + progress * (to.m23 - from.m23),
            m24: from.m24 + progress * (to.m24 - from.m24),
            m31: from.m31 + progress * (to.m31 - from.m31),
            m32: from.m32 + progress * (to.m32 - from.m32),
            m33: from.m33 + progress * (to.m33 - from.m33),
            m34: from.m34 + progress * (to.m34 - from.m34),
            m41: from.m41 + progress * (to.m41 - from.m41),
            m42: from.m42 + progress * (to.m42 - from.m42),
            m43: from.m43 + progress * (to.m43 - from.m43),
            m44: from.m44 + progress * (to.m44 - from.m44)
        )
    }

    private func applyColorAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        switch keyPath {
        case "backgroundColor":
            guard let from = extractColor(animation.fromValue) ?? _backgroundColor,
                  let to = extractColor(animation.toValue) ?? _backgroundColor else { return }
            layer._backgroundColor = interpolateColor(from: from, to: to, progress: CGFloat(progress))
        case "borderColor":
            guard let from = extractColor(animation.fromValue) ?? _borderColor,
                  let to = extractColor(animation.toValue) ?? _borderColor else { return }
            layer._borderColor = interpolateColor(from: from, to: to, progress: CGFloat(progress))
        case "shadowColor":
            guard let from = extractColor(animation.fromValue) ?? _shadowColor,
                  let to = extractColor(animation.toValue) ?? _shadowColor else { return }
            layer._shadowColor = interpolateColor(from: from, to: to, progress: CGFloat(progress))

        // CAShapeLayer color properties
        case "fillColor":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let from = extractColor(animation.fromValue) ?? modelShapeLayer._fillColor,
                  let to = extractColor(animation.toValue) ?? modelShapeLayer._fillColor else { return }
            shapeLayer._fillColor = interpolateColor(from: from, to: to, progress: CGFloat(progress))
        case "strokeColor":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let from = extractColor(animation.fromValue) ?? modelShapeLayer._strokeColor,
                  let to = extractColor(animation.toValue) ?? modelShapeLayer._strokeColor else { return }
            shapeLayer._strokeColor = interpolateColor(from: from, to: to, progress: CGFloat(progress))

        default:
            break
        }
    }

    /// Safely extracts a CGColor from an Any value.
    private func extractColor(_ value: Any?) -> CGColor? {
        return value as? CGColor
    }

    /// Safely extracts a CGPath from an Any value.
    private func extractPath(_ value: Any?) -> CGPath? {
        if let path = value as? CGPath {
            return path
        }
        if let mutablePath = value as? CGMutablePath {
            return mutablePath
        }
        return nil
    }

    /// Interpolates between two colors in RGBA color space.
    private func interpolateColor(from: CGColor, to: CGColor, progress: CGFloat) -> CGColor {
        // Convert both colors to RGBA format, handling different component counts
        let (fromR, fromG, fromB, fromA) = extractRGBA(from: from)
        let (toR, toG, toB, toA) = extractRGBA(from: to)

        let r = fromR + progress * (toR - fromR)
        let g = fromG + progress * (toG - fromG)
        let b = fromB + progress * (toB - fromB)
        let a = fromA + progress * (toA - fromA)

        return CGColor(red: r, green: g, blue: b, alpha: a)
    }

    /// Extracts RGBA components from a CGColor, handling different color space formats.
    ///
    /// - 1 component: gray (alpha = 1)
    /// - 2 components: [gray, alpha]
    /// - 3 components: [R, G, B] (alpha = 1)
    /// - 4 components: [R, G, B, A]
    private func extractRGBA(from color: CGColor) -> (r: CGFloat, g: CGFloat, b: CGFloat, a: CGFloat) {
        let components = color.components ?? [0, 0, 0, 1]
        let count = components.count

        switch count {
        case 1:
            // Gray only, assume alpha = 1
            let gray = components[0]
            return (gray, gray, gray, 1.0)
        case 2:
            // Grayscale: [gray, alpha]
            let gray = components[0]
            let alpha = components[1]
            return (gray, gray, gray, alpha)
        case 3:
            // RGB without alpha
            return (components[0], components[1], components[2], 1.0)
        case 4:
            // RGBA
            return (components[0], components[1], components[2], components[3])
        default:
            // Fallback
            return (0, 0, 0, 1)
        }
    }

    private func applyArrayAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        switch keyPath {
        // CAGradientLayer array properties
        case "colors":
            guard let gradientLayer = layer as? CAGradientLayer,
                  let modelGradientLayer = self as? CAGradientLayer else { return }
            guard let fromColors = (animation.fromValue as? [Any]) ?? modelGradientLayer._colors,
                  let toColors = (animation.toValue as? [Any]) ?? modelGradientLayer._colors else { return }

            // Interpolate each color in the array
            let count = min(fromColors.count, toColors.count)
            var interpolatedColors: [Any] = []
            for i in 0..<count {
                if let fromColor = extractColor(fromColors[i]),
                   let toColor = extractColor(toColors[i]) {
                    interpolatedColors.append(interpolateColor(from: fromColor, to: toColor, progress: CGFloat(progress)))
                } else {
                    // Can't interpolate, use from value
                    interpolatedColors.append(fromColors[i])
                }
            }
            gradientLayer._colors = interpolatedColors

        case "locations":
            guard let gradientLayer = layer as? CAGradientLayer,
                  let modelGradientLayer = self as? CAGradientLayer else { return }
            guard let fromLocations = (animation.fromValue as? [Float]) ?? modelGradientLayer._locations,
                  let toLocations = (animation.toValue as? [Float]) ?? modelGradientLayer._locations else { return }

            // Interpolate each location in the array
            let count = min(fromLocations.count, toLocations.count)
            var interpolatedLocations: [Float] = []
            for i in 0..<count {
                let from = fromLocations[i]
                let to = toLocations[i]
                interpolatedLocations.append(from + Float(progress) * (to - from))
            }
            gradientLayer._locations = interpolatedLocations

        default:
            break
        }
    }

    // MARK: - Path Sampling for Keyframe Animation

    /// Samples a CGPath at regular intervals and returns an array of points.
    ///
    /// The path is flattened (curves converted to line segments) and then points
    /// are extracted at regular intervals based on arc length.
    private func samplePathPoints(_ path: CGPath, numPoints: Int = 100) -> [CGPoint] {
        var points: [CGPoint] = []
        var currentPoint = CGPoint.zero
        var subpathStart = CGPoint.zero

        // First, flatten the path into line segments
        path.applyWithBlock { elementPtr in
            let element = elementPtr.pointee
            guard let elementPoints = element.points else { return }
            switch element.type {
            case .moveToPoint:
                let point = elementPoints[0]
                points.append(point)
                currentPoint = point
                subpathStart = point

            case .addLineToPoint:
                let point = elementPoints[0]
                points.append(point)
                currentPoint = point

            case .addQuadCurveToPoint:
                let control = elementPoints[0]
                let end = elementPoints[1]
                // Flatten quad bezier into line segments
                for i in 1...10 {
                    let t = CGFloat(i) / 10.0
                    let tt = t * t
                    let u = 1 - t
                    let uu = u * u
                    let x = uu * currentPoint.x + 2 * u * t * control.x + tt * end.x
                    let y = uu * currentPoint.y + 2 * u * t * control.y + tt * end.y
                    points.append(CGPoint(x: x, y: y))
                }
                currentPoint = end

            case .addCurveToPoint:
                let control1 = elementPoints[0]
                let control2 = elementPoints[1]
                let end = elementPoints[2]
                // Flatten cubic bezier into line segments
                for i in 1...10 {
                    let t = CGFloat(i) / 10.0
                    let tt = t * t
                    let ttt = tt * t
                    let u = 1 - t
                    let uu = u * u
                    let uuu = uu * u
                    let x = uuu * currentPoint.x + 3 * uu * t * control1.x + 3 * u * tt * control2.x + ttt * end.x
                    let y = uuu * currentPoint.y + 3 * uu * t * control1.y + 3 * u * tt * control2.y + ttt * end.y
                    points.append(CGPoint(x: x, y: y))
                }
                currentPoint = end

            case .closeSubpath:
                if currentPoint != subpathStart {
                    points.append(subpathStart)
                }
                currentPoint = subpathStart

            @unknown default:
                break
            }
        }

        return points
    }

    /// Samples a point and optional tangent on a path at a given normalized progress (0-1).
    ///
    /// Uses arc-length parameterization for uniform motion along the path.
    private func samplePathAtProgress(_ path: CGPath, progress: CGFloat) -> (point: CGPoint, tangent: CGFloat)? {
        let points = samplePathPoints(path)
        guard points.count >= 2 else { return nil }

        // Calculate cumulative arc lengths
        var arcLengths: [CGFloat] = [0]
        for i in 1..<points.count {
            let dx = points[i].x - points[i - 1].x
            let dy = points[i].y - points[i - 1].y
            let segmentLength = sqrt(dx * dx + dy * dy)
            arcLengths.append(arcLengths.last! + segmentLength)
        }

        let totalLength = arcLengths.last!
        guard totalLength > 0 else { return (point: points[0], tangent: 0) }

        // Find the target arc length
        let targetLength = totalLength * min(max(progress, 0), 1)

        // Find the segment containing this arc length
        var segmentIndex = 0
        for i in 1..<arcLengths.count {
            if arcLengths[i] >= targetLength {
                segmentIndex = i - 1
                break
            }
        }

        // Interpolate within the segment
        let segmentStart = arcLengths[segmentIndex]
        let segmentEnd = arcLengths[segmentIndex + 1]
        let segmentProgress: CGFloat
        if segmentEnd > segmentStart {
            segmentProgress = (targetLength - segmentStart) / (segmentEnd - segmentStart)
        } else {
            segmentProgress = 0
        }

        let p0 = points[segmentIndex]
        let p1 = points[segmentIndex + 1]

        let point = CGPoint(
            x: p0.x + segmentProgress * (p1.x - p0.x),
            y: p0.y + segmentProgress * (p1.y - p0.y)
        )

        // Calculate tangent angle
        let dx = p1.x - p0.x
        let dy = p1.y - p0.y
        let tangent = CGFloat(atan2(dy, dx))

        return (point: point, tangent: tangent)
    }

    /// Applies a path-based keyframe animation to a layer.
    ///
    /// The path is used for position animation, and optionally for rotation
    /// if rotationMode is set.
    private func applyPathKeyframeAnimation(_ animation: CAKeyframeAnimation, path: CGPath, to layer: CALayer, progress: CFTimeInterval) {
        guard let sample = samplePathAtProgress(path, progress: CGFloat(progress)) else { return }

        // Apply position
        layer._position = sample.point

        // Apply rotation if rotationMode is set
        if let rotationMode = animation.rotationMode {
            switch rotationMode {
            case .rotateAuto:
                // Rotate to match path tangent
                layer._transform = CATransform3DMakeRotation(sample.tangent, 0, 0, 1)
            case .rotateAutoReverse:
                // Rotate to match path tangent + 180 degrees
                layer._transform = CATransform3DMakeRotation(sample.tangent + .pi, 0, 0, 1)
            default:
                break
            }
        }
    }

    /// Applies a keyframe animation to a layer property.
    private func applyKeyframeAnimation(_ animation: CAKeyframeAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        // Check if path is available for position animation
        if let path = animation.path, keyPath == "position" {
            applyPathKeyframeAnimation(animation, path: path, to: layer, progress: progress)
            return
        }

        guard let values = animation.values, values.count > 1 else { return }

        // For paced modes, remap progress based on arc length
        let effectiveProgress: CFTimeInterval
        var effectiveKeyTimes: [Float]

        switch animation.calculationMode {
        case .paced, .cubicPaced:
            // Calculate arc-length parameterized progress
            let (remappedProgress, pacedKeyTimes) = calculatePacedProgress(progress: Float(progress), values: values, cubic: animation.calculationMode == .cubicPaced)
            effectiveProgress = CFTimeInterval(remappedProgress)
            effectiveKeyTimes = pacedKeyTimes
        default:
            effectiveProgress = progress
            effectiveKeyTimes = animation.keyTimes ?? defaultKeyTimes(for: values.count)
        }

        // Determine which keyframe segment we're in
        let segmentIndex = findSegmentIndex(for: Float(effectiveProgress), in: effectiveKeyTimes)
        let startIndex = segmentIndex
        let endIndex = min(segmentIndex + 1, values.count - 1)

        // Get local progress within the segment
        let startTime = effectiveKeyTimes[startIndex]
        let endTime = effectiveKeyTimes[endIndex]
        let segmentProgress: Float
        if endTime > startTime {
            segmentProgress = (Float(effectiveProgress) - startTime) / (endTime - startTime)
        } else {
            segmentProgress = 1.0
        }

        // Apply timing function for this segment if available (not used for paced modes)
        let adjustedProgress: Float
        if animation.calculationMode != .paced && animation.calculationMode != .cubicPaced,
           let timingFunctions = animation.timingFunctions,
           startIndex < timingFunctions.count {
            adjustedProgress = timingFunctions[startIndex].evaluate(at: segmentProgress)
        } else {
            adjustedProgress = segmentProgress
        }

        // Interpolate between keyframe values based on calculation mode
        let fromValue = values[startIndex]
        let toValue = values[endIndex]

        switch animation.calculationMode {
        case .discrete:
            // No interpolation, use the from value
            applyKeyframeValue(fromValue, layer: layer, keyPath: keyPath)
        case .linear, .paced:
            // Linear interpolation between values
            interpolateKeyframeValue(from: fromValue, to: toValue, progress: CFTimeInterval(adjustedProgress), layer: layer, keyPath: keyPath)
        case .cubic, .cubicPaced:
            // Catmull-Rom spline interpolation
            // Get the 4 control points: P0, P1, P2, P3
            let p0Index = max(0, startIndex - 1)
            let p3Index = min(values.count - 1, endIndex + 1)

            let p0 = values[p0Index]
            let p1 = values[startIndex]
            let p2 = values[endIndex]
            let p3 = values[p3Index]

            interpolateCubicKeyframeValue(p0: p0, p1: p1, p2: p2, p3: p3, t: CGFloat(adjustedProgress), layer: layer, keyPath: keyPath)
        default:
            interpolateKeyframeValue(from: fromValue, to: toValue, progress: CFTimeInterval(adjustedProgress), layer: layer, keyPath: keyPath)
        }
    }

    /// Calculates arc-length parameterized progress for paced animation modes.
    ///
    /// Returns the remapped progress and the paced key times array.
    private func calculatePacedProgress(progress: Float, values: [Any], cubic: Bool) -> (Float, [Float]) {
        guard values.count > 1 else { return (progress, [0]) }

        // Calculate distances between consecutive keyframes
        var distances: [CGFloat] = []
        for i in 0..<(values.count - 1) {
            let distance: CGFloat
            if cubic {
                // For cubic paced, estimate arc length of Catmull-Rom spline segment
                let p0Index = max(0, i - 1)
                let p3Index = min(values.count - 1, i + 2)
                distance = estimateCubicArcLength(
                    p0: values[p0Index],
                    p1: values[i],
                    p2: values[i + 1],
                    p3: values[p3Index]
                )
            } else {
                distance = calculateDistance(from: values[i], to: values[i + 1])
            }
            distances.append(max(distance, 0.0001)) // Avoid zero distance
        }

        // Calculate cumulative distances
        var cumulativeDistances: [CGFloat] = [0]
        for distance in distances {
            cumulativeDistances.append(cumulativeDistances.last! + distance)
        }
        let totalDistance = cumulativeDistances.last!

        // Calculate paced key times (normalized by total distance)
        var pacedKeyTimes: [Float] = []
        for cumDist in cumulativeDistances {
            pacedKeyTimes.append(Float(cumDist / totalDistance))
        }

        return (progress, pacedKeyTimes)
    }

    /// Calculates the distance between two animation values.
    private func calculateDistance(from: Any, to: Any) -> CGFloat {
        // CGPoint distance (Euclidean)
        if let fromPoint = from as? CGPoint, let toPoint = to as? CGPoint {
            let dx = toPoint.x - fromPoint.x
            let dy = toPoint.y - fromPoint.y
            return sqrt(dx * dx + dy * dy)
        }

        // CGFloat distance (absolute)
        if let fromFloat = from as? CGFloat, let toFloat = to as? CGFloat {
            return abs(toFloat - fromFloat)
        }

        // Float distance (absolute)
        if let fromFloat = from as? Float, let toFloat = to as? Float {
            return CGFloat(abs(toFloat - fromFloat))
        }

        // CGRect distance (sum of component distances)
        if let fromRect = from as? CGRect, let toRect = to as? CGRect {
            return abs(toRect.origin.x - fromRect.origin.x) +
                   abs(toRect.origin.y - fromRect.origin.y) +
                   abs(toRect.size.width - fromRect.size.width) +
                   abs(toRect.size.height - fromRect.size.height)
        }

        // CGColor distance (RGBA Euclidean)
        if let fromColor = extractColor(from), let toColor = extractColor(to) {
            let fromComponents = fromColor.components ?? [0, 0, 0, 1]
            let toComponents = toColor.components ?? [0, 0, 0, 1]
            var sum: CGFloat = 0
            for i in 0..<min(fromComponents.count, toComponents.count) {
                let diff = toComponents[i] - fromComponents[i]
                sum += diff * diff
            }
            return sqrt(sum)
        }

        // Default: return 1 for equal spacing
        return 1.0
    }

    /// Estimates the arc length of a Catmull-Rom spline segment using numerical integration.
    private func estimateCubicArcLength(p0: Any, p1: Any, p2: Any, p3: Any) -> CGFloat {
        // Use numerical integration with Simpson's rule
        // Sample the curve at multiple points and sum segment lengths
        let numSamples = 10
        var arcLength: CGFloat = 0

        guard let prev = interpolateForDistance(p0: p0, p1: p1, p2: p2, p3: p3, t: 0) else {
            // Fallback to linear distance
            return calculateDistance(from: p1, to: p2)
        }

        var previousPoint = prev

        for i in 1...numSamples {
            let t = CGFloat(i) / CGFloat(numSamples)
            guard let currentPoint = interpolateForDistance(p0: p0, p1: p1, p2: p2, p3: p3, t: t) else {
                return calculateDistance(from: p1, to: p2)
            }
            arcLength += calculatePointDistance(from: previousPoint, to: currentPoint)
            previousPoint = currentPoint
        }

        return arcLength
    }

    /// Interpolates a point on the Catmull-Rom spline for distance calculation.
    private func interpolateForDistance(p0: Any, p1: Any, p2: Any, p3: Any, t: CGFloat) -> (x: CGFloat, y: CGFloat)? {
        // Only support CGPoint for now
        if let v0 = p0 as? CGPoint, let v1 = p1 as? CGPoint, let v2 = p2 as? CGPoint, let v3 = p3 as? CGPoint {
            let x = catmullRom(v0.x, v1.x, v2.x, v3.x, t)
            let y = catmullRom(v0.y, v1.y, v2.y, v3.y, t)
            return (x, y)
        }
        return nil
    }

    /// Calculates Euclidean distance between two 2D points.
    private func calculatePointDistance(from: (x: CGFloat, y: CGFloat), to: (x: CGFloat, y: CGFloat)) -> CGFloat {
        let dx = to.x - from.x
        let dy = to.y - from.y
        return sqrt(dx * dx + dy * dy)
    }

    /// Generates default key times evenly distributed.
    private func defaultKeyTimes(for count: Int) -> [Float] {
        guard count > 1 else { return [0] }
        return (0..<count).map { Float($0) / Float(count - 1) }
    }

    /// Finds the segment index for a given progress value.
    private func findSegmentIndex(for progress: Float, in keyTimes: [Float]) -> Int {
        for i in 0..<(keyTimes.count - 1) {
            if progress >= keyTimes[i] && progress < keyTimes[i + 1] {
                return i
            }
        }
        return max(0, keyTimes.count - 2)
    }

    /// Applies a single keyframe value without interpolation.
    private func applyKeyframeValue(_ value: Any, layer: CALayer, keyPath: String) {
        switch keyPath {
        case "opacity":
            if let v = value as? Float { layer._opacity = v }
        case "position":
            if let v = value as? CGPoint { layer._position = v }
        case "bounds":
            if let v = value as? CGRect { layer._bounds = v }
        case "cornerRadius":
            if let v = value as? CGFloat { layer._cornerRadius = v }
        case "borderWidth":
            if let v = value as? CGFloat { layer._borderWidth = v }
        case "zPosition":
            if let v = value as? CGFloat { layer._zPosition = v }
        case "transform":
            if let v = value as? CATransform3D { layer._transform = v }

        // Color properties
        case "backgroundColor":
            if let v = extractColor(value) { layer._backgroundColor = v }
        case "borderColor":
            if let v = extractColor(value) { layer._borderColor = v }
        case "shadowColor":
            if let v = extractColor(value) { layer._shadowColor = v }

        // CAShapeLayer properties
        case "strokeStart":
            if let shapeLayer = layer as? CAShapeLayer, let v = value as? CGFloat { shapeLayer._strokeStart = v }
        case "strokeEnd":
            if let shapeLayer = layer as? CAShapeLayer, let v = value as? CGFloat { shapeLayer._strokeEnd = v }
        case "lineWidth":
            if let shapeLayer = layer as? CAShapeLayer, let v = value as? CGFloat { shapeLayer._lineWidth = v }
        case "lineDashPhase":
            if let shapeLayer = layer as? CAShapeLayer, let v = value as? CGFloat { shapeLayer._lineDashPhase = v }
        case "miterLimit":
            if let shapeLayer = layer as? CAShapeLayer, let v = value as? CGFloat { shapeLayer._miterLimit = v }
        case "fillColor":
            if let shapeLayer = layer as? CAShapeLayer, let v = extractColor(value) { shapeLayer._fillColor = v }
        case "strokeColor":
            if let shapeLayer = layer as? CAShapeLayer, let v = extractColor(value) { shapeLayer._strokeColor = v }
        case "path":
            if let shapeLayer = layer as? CAShapeLayer, let v = extractPath(value) { shapeLayer._path = v }

        // CAGradientLayer properties
        case "startPoint":
            if let gradientLayer = layer as? CAGradientLayer, let v = value as? CGPoint { gradientLayer._startPoint = v }
        case "endPoint":
            if let gradientLayer = layer as? CAGradientLayer, let v = value as? CGPoint { gradientLayer._endPoint = v }
        case "colors":
            if let gradientLayer = layer as? CAGradientLayer, let v = value as? [Any] { gradientLayer._colors = v }
        case "locations":
            if let gradientLayer = layer as? CAGradientLayer, let v = value as? [Float] { gradientLayer._locations = v }

        default:
            break
        }
    }

    /// Interpolates between two keyframe values.
    private func interpolateKeyframeValue(from fromValue: Any, to toValue: Any, progress: CFTimeInterval, layer: CALayer, keyPath: String) {
        switch keyPath {
        case "opacity":
            if let f = fromValue as? Float, let t = toValue as? Float {
                layer._opacity = f + Float(progress) * (t - f)
            }
        case "position":
            if let f = fromValue as? CGPoint, let t = toValue as? CGPoint {
                layer._position = CGPoint(
                    x: f.x + CGFloat(progress) * (t.x - f.x),
                    y: f.y + CGFloat(progress) * (t.y - f.y)
                )
            }
        case "bounds":
            if let f = fromValue as? CGRect, let t = toValue as? CGRect {
                layer._bounds = CGRect(
                    x: f.origin.x + CGFloat(progress) * (t.origin.x - f.origin.x),
                    y: f.origin.y + CGFloat(progress) * (t.origin.y - f.origin.y),
                    width: f.size.width + CGFloat(progress) * (t.size.width - f.size.width),
                    height: f.size.height + CGFloat(progress) * (t.size.height - f.size.height)
                )
            }
        case "cornerRadius", "borderWidth", "shadowRadius", "zPosition":
            if let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                let value = f + CGFloat(progress) * (t - f)
                switch keyPath {
                case "cornerRadius": layer._cornerRadius = value
                case "borderWidth": layer._borderWidth = value
                case "shadowRadius": layer._shadowRadius = value
                case "zPosition": layer._zPosition = value
                default: break
                }
            }
        case "transform":
            if let f = fromValue as? CATransform3D, let t = toValue as? CATransform3D {
                layer._transform = interpolateTransform(from: f, to: t, progress: CGFloat(progress))
            }

        // Color properties
        case "backgroundColor":
            if let f = extractColor(fromValue), let t = extractColor(toValue) {
                layer._backgroundColor = interpolateColor(from: f, to: t, progress: CGFloat(progress))
            }
        case "borderColor":
            if let f = extractColor(fromValue), let t = extractColor(toValue) {
                layer._borderColor = interpolateColor(from: f, to: t, progress: CGFloat(progress))
            }
        case "shadowColor":
            if let f = extractColor(fromValue), let t = extractColor(toValue) {
                layer._shadowColor = interpolateColor(from: f, to: t, progress: CGFloat(progress))
            }

        // CAShapeLayer float properties
        case "strokeStart":
            if let shapeLayer = layer as? CAShapeLayer,
               let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                shapeLayer._strokeStart = f + CGFloat(progress) * (t - f)
            }
        case "strokeEnd":
            if let shapeLayer = layer as? CAShapeLayer,
               let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                shapeLayer._strokeEnd = f + CGFloat(progress) * (t - f)
            }
        case "lineWidth":
            if let shapeLayer = layer as? CAShapeLayer,
               let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                shapeLayer._lineWidth = f + CGFloat(progress) * (t - f)
            }
        case "lineDashPhase":
            if let shapeLayer = layer as? CAShapeLayer,
               let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                shapeLayer._lineDashPhase = f + CGFloat(progress) * (t - f)
            }
        case "miterLimit":
            if let shapeLayer = layer as? CAShapeLayer,
               let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                shapeLayer._miterLimit = f + CGFloat(progress) * (t - f)
            }

        // CAShapeLayer color properties
        case "fillColor":
            if let shapeLayer = layer as? CAShapeLayer,
               let f = extractColor(fromValue), let t = extractColor(toValue) {
                shapeLayer._fillColor = interpolateColor(from: f, to: t, progress: CGFloat(progress))
            }
        case "strokeColor":
            if let shapeLayer = layer as? CAShapeLayer,
               let f = extractColor(fromValue), let t = extractColor(toValue) {
                shapeLayer._strokeColor = interpolateColor(from: f, to: t, progress: CGFloat(progress))
            }

        // CAGradientLayer properties
        case "startPoint":
            if let gradientLayer = layer as? CAGradientLayer,
               let f = fromValue as? CGPoint, let t = toValue as? CGPoint {
                gradientLayer._startPoint = CGPoint(
                    x: f.x + CGFloat(progress) * (t.x - f.x),
                    y: f.y + CGFloat(progress) * (t.y - f.y)
                )
            }
        case "endPoint":
            if let gradientLayer = layer as? CAGradientLayer,
               let f = fromValue as? CGPoint, let t = toValue as? CGPoint {
                gradientLayer._endPoint = CGPoint(
                    x: f.x + CGFloat(progress) * (t.x - f.x),
                    y: f.y + CGFloat(progress) * (t.y - f.y)
                )
            }
        case "colors":
            if let gradientLayer = layer as? CAGradientLayer,
               let fromColors = fromValue as? [Any], let toColors = toValue as? [Any] {
                let count = min(fromColors.count, toColors.count)
                var interpolatedColors: [Any] = []
                for i in 0..<count {
                    if let fromColor = extractColor(fromColors[i]),
                       let toColor = extractColor(toColors[i]) {
                        interpolatedColors.append(interpolateColor(from: fromColor, to: toColor, progress: CGFloat(progress)))
                    } else {
                        interpolatedColors.append(fromColors[i])
                    }
                }
                gradientLayer._colors = interpolatedColors
            }
        case "locations":
            if let gradientLayer = layer as? CAGradientLayer,
               let fromLocations = fromValue as? [Float], let toLocations = toValue as? [Float] {
                let count = min(fromLocations.count, toLocations.count)
                var interpolatedLocations: [Float] = []
                for i in 0..<count {
                    interpolatedLocations.append(fromLocations[i] + Float(progress) * (toLocations[i] - fromLocations[i]))
                }
                gradientLayer._locations = interpolatedLocations
            }

        default:
            break
        }
    }

    /// Catmull-Rom spline interpolation between 4 control points.
    ///
    /// The Catmull-Rom spline passes through P1 and P2, using P0 and P3 to determine
    /// the tangent at those points. The parameter `t` goes from 0 (at P1) to 1 (at P2).
    ///
    /// Formula:
    /// ```
    /// p(t) = 0.5 * ((2*P1) + (-P0 + P2)*t + (2*P0 - 5*P1 + 4*P2 - P3)*t² + (-P0 + 3*P1 - 3*P2 + P3)*t³)
    /// ```
    private func interpolateCubicKeyframeValue(p0: Any, p1: Any, p2: Any, p3: Any, t: CGFloat, layer: CALayer, keyPath: String) {
        switch keyPath {
        case "opacity":
            if let v0 = p0 as? Float, let v1 = p1 as? Float, let v2 = p2 as? Float, let v3 = p3 as? Float {
                layer._opacity = catmullRom(v0, v1, v2, v3, Float(t))
            }
        case "position":
            if let v0 = p0 as? CGPoint, let v1 = p1 as? CGPoint, let v2 = p2 as? CGPoint, let v3 = p3 as? CGPoint {
                layer._position = CGPoint(
                    x: catmullRom(v0.x, v1.x, v2.x, v3.x, t),
                    y: catmullRom(v0.y, v1.y, v2.y, v3.y, t)
                )
            }
        case "bounds":
            if let v0 = p0 as? CGRect, let v1 = p1 as? CGRect, let v2 = p2 as? CGRect, let v3 = p3 as? CGRect {
                layer._bounds = CGRect(
                    x: catmullRom(v0.origin.x, v1.origin.x, v2.origin.x, v3.origin.x, t),
                    y: catmullRom(v0.origin.y, v1.origin.y, v2.origin.y, v3.origin.y, t),
                    width: catmullRom(v0.size.width, v1.size.width, v2.size.width, v3.size.width, t),
                    height: catmullRom(v0.size.height, v1.size.height, v2.size.height, v3.size.height, t)
                )
            }
        case "cornerRadius", "borderWidth", "shadowRadius", "zPosition":
            if let v0 = p0 as? CGFloat, let v1 = p1 as? CGFloat, let v2 = p2 as? CGFloat, let v3 = p3 as? CGFloat {
                let value = catmullRom(v0, v1, v2, v3, t)
                switch keyPath {
                case "cornerRadius": layer._cornerRadius = value
                case "borderWidth": layer._borderWidth = value
                case "shadowRadius": layer._shadowRadius = value
                case "zPosition": layer._zPosition = value
                default: break
                }
            }
        case "transform":
            if let v0 = p0 as? CATransform3D, let v1 = p1 as? CATransform3D, let v2 = p2 as? CATransform3D, let v3 = p3 as? CATransform3D {
                layer._transform = catmullRomTransform(v0, v1, v2, v3, t)
            }
        default:
            // Fallback to linear interpolation for unsupported types
            interpolateKeyframeValue(from: p1, to: p2, progress: CFTimeInterval(t), layer: layer, keyPath: keyPath)
        }
    }

    /// Catmull-Rom spline interpolation for a single CGFloat value.
    private func catmullRom(_ p0: CGFloat, _ p1: CGFloat, _ p2: CGFloat, _ p3: CGFloat, _ t: CGFloat) -> CGFloat {
        let t2 = t * t
        let t3 = t2 * t

        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
    }

    /// Catmull-Rom spline interpolation for a single Float value.
    private func catmullRom(_ p0: Float, _ p1: Float, _ p2: Float, _ p3: Float, _ t: Float) -> Float {
        let t2 = t * t
        let t3 = t2 * t

        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
    }

    /// Catmull-Rom spline interpolation for CATransform3D.
    private func catmullRomTransform(_ p0: CATransform3D, _ p1: CATransform3D, _ p2: CATransform3D, _ p3: CATransform3D, _ t: CGFloat) -> CATransform3D {
        return CATransform3D(
            m11: catmullRom(p0.m11, p1.m11, p2.m11, p3.m11, t),
            m12: catmullRom(p0.m12, p1.m12, p2.m12, p3.m12, t),
            m13: catmullRom(p0.m13, p1.m13, p2.m13, p3.m13, t),
            m14: catmullRom(p0.m14, p1.m14, p2.m14, p3.m14, t),
            m21: catmullRom(p0.m21, p1.m21, p2.m21, p3.m21, t),
            m22: catmullRom(p0.m22, p1.m22, p2.m22, p3.m22, t),
            m23: catmullRom(p0.m23, p1.m23, p2.m23, p3.m23, t),
            m24: catmullRom(p0.m24, p1.m24, p2.m24, p3.m24, t),
            m31: catmullRom(p0.m31, p1.m31, p2.m31, p3.m31, t),
            m32: catmullRom(p0.m32, p1.m32, p2.m32, p3.m32, t),
            m33: catmullRom(p0.m33, p1.m33, p2.m33, p3.m33, t),
            m34: catmullRom(p0.m34, p1.m34, p2.m34, p3.m34, t),
            m41: catmullRom(p0.m41, p1.m41, p2.m41, p3.m41, t),
            m42: catmullRom(p0.m42, p1.m42, p2.m42, p3.m42, t),
            m43: catmullRom(p0.m43, p1.m43, p2.m43, p3.m43, t),
            m44: catmullRom(p0.m44, p1.m44, p2.m44, p3.m44, t)
        )
    }

    /// Returns the model layer object associated with the receiver, if any.
    ///
    /// - Returns: The model layer if this is a presentation layer, otherwise `self`.
    open func model() -> Self {
        if let model = _modelLayer as? Self {
            return model
        }
        return self
    }

    // MARK: - Accessing the Delegate

    /// The layer's delegate object.
    open weak var delegate: (any CALayerDelegate)?

    // MARK: - Providing the Layer's Content

    /// An object that provides the contents of the layer. Animatable.
    open var contents: CGImage?

    /// The rectangle, in the unit coordinate space, that defines the portion of the layer's
    /// contents that should be used. Animatable.
    open var contentsRect: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)

    /// The rectangle that defines how the layer contents are scaled if the layer's contents
    /// are resized. Animatable.
    open var contentsCenter: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)

    /// Reloads the content of this layer.
    open func display() {
        delegate?.display(self)
    }

    /// Draws the layer's content using the specified graphics context.
    open func draw(in ctx: CGContext) {
        delegate?.draw(self, in: ctx)
    }

    // MARK: - Modifying the Layer's Appearance

    /// A constant that specifies how the layer's contents are positioned or scaled within its bounds.
    open var contentsGravity: CALayerContentsGravity = .resize

    private var _opacity: Float = 1.0
    /// The opacity of the receiver. Animatable.
    open var opacity: Float {
        get { return _opacity }
        set {
            let oldValue = _opacity
            let clampedValue = max(0, min(1, newValue))
            _opacity = clampedValue
            CATransaction.registerChange(layer: self, keyPath: "opacity", oldValue: oldValue, newValue: clampedValue)
        }
    }

    private var _isHidden: Bool = false
    /// A Boolean indicating whether the layer is displayed. Animatable.
    open var isHidden: Bool {
        get { return _isHidden }
        set {
            let oldValue = _isHidden
            _isHidden = newValue
            CATransaction.registerChange(layer: self, keyPath: "isHidden", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _masksToBounds: Bool = false
    /// A Boolean indicating whether sublayers are clipped to the layer's bounds. Animatable.
    open var masksToBounds: Bool {
        get { return _masksToBounds }
        set {
            let oldValue = _masksToBounds
            _masksToBounds = newValue
            CATransaction.registerChange(layer: self, keyPath: "masksToBounds", oldValue: oldValue, newValue: newValue)
        }
    }

    /// An optional layer whose alpha channel is used to mask the layer's content.
    open var mask: CALayer?

    private var _isDoubleSided: Bool = true
    /// A Boolean indicating whether the layer displays its content when facing away from the viewer. Animatable.
    open var isDoubleSided: Bool {
        get { return _isDoubleSided }
        set {
            let oldValue = _isDoubleSided
            _isDoubleSided = newValue
            CATransaction.registerChange(layer: self, keyPath: "isDoubleSided", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _cornerRadius: CGFloat = 0
    /// The radius to use when drawing rounded corners for the layer's background. Animatable.
    open var cornerRadius: CGFloat {
        get { return _cornerRadius }
        set {
            let oldValue = _cornerRadius
            let clampedValue = max(0, newValue)
            _cornerRadius = clampedValue
            CATransaction.registerChange(layer: self, keyPath: "cornerRadius", oldValue: oldValue, newValue: clampedValue)
        }
    }

    /// A bitmask defining which of the four corners receives the masking.
    open var maskedCorners: CACornerMask = [.layerMinXMinYCorner, .layerMaxXMinYCorner, .layerMinXMaxYCorner, .layerMaxXMaxYCorner]

    /// The curve to use when drawing the rounded corners.
    open var cornerCurve: CALayerCornerCurve = .circular

    private var _borderWidth: CGFloat = 0
    /// The width of the layer's border. Animatable.
    open var borderWidth: CGFloat {
        get { return _borderWidth }
        set {
            let oldValue = _borderWidth
            let clampedValue = max(0, newValue)
            _borderWidth = clampedValue
            CATransaction.registerChange(layer: self, keyPath: "borderWidth", oldValue: oldValue, newValue: clampedValue)
        }
    }

    private var _borderColor: CGColor?
    /// The color of the layer's border. Animatable.
    open var borderColor: CGColor? {
        get { return _borderColor }
        set {
            let oldValue = _borderColor
            _borderColor = newValue
            CATransaction.registerChange(layer: self, keyPath: "borderColor", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _backgroundColor: CGColor?
    /// The background color of the receiver. Animatable.
    open var backgroundColor: CGColor? {
        get { return _backgroundColor }
        set {
            let oldValue = _backgroundColor
            _backgroundColor = newValue
            CATransaction.registerChange(layer: self, keyPath: "backgroundColor", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _shadowOpacity: Float = 0
    /// The opacity of the layer's shadow. Animatable.
    open var shadowOpacity: Float {
        get { return _shadowOpacity }
        set {
            let oldValue = _shadowOpacity
            let clampedValue = max(0, min(1, newValue))
            _shadowOpacity = clampedValue
            CATransaction.registerChange(layer: self, keyPath: "shadowOpacity", oldValue: oldValue, newValue: clampedValue)
        }
    }

    private var _shadowRadius: CGFloat = 3
    /// The blur radius (in points) used to render the layer's shadow. Animatable.
    open var shadowRadius: CGFloat {
        get { return _shadowRadius }
        set {
            let oldValue = _shadowRadius
            let clampedValue = max(0, newValue)
            _shadowRadius = clampedValue
            CATransaction.registerChange(layer: self, keyPath: "shadowRadius", oldValue: oldValue, newValue: clampedValue)
        }
    }

    private var _shadowOffset: CGSize = CGSize(width: 0, height: -3)
    /// The offset (in points) of the layer's shadow. Animatable.
    open var shadowOffset: CGSize {
        get { return _shadowOffset }
        set {
            let oldValue = _shadowOffset
            _shadowOffset = newValue
            CATransaction.registerChange(layer: self, keyPath: "shadowOffset", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _shadowColor: CGColor?
    /// The color of the layer's shadow. Animatable.
    open var shadowColor: CGColor? {
        get { return _shadowColor }
        set {
            let oldValue = _shadowColor
            _shadowColor = newValue
            CATransaction.registerChange(layer: self, keyPath: "shadowColor", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _shadowPath: CGPath?
    /// The shape of the layer's shadow. Animatable.
    open var shadowPath: CGPath? {
        get { return _shadowPath }
        set {
            let oldValue = _shadowPath
            _shadowPath = newValue
            CATransaction.registerChange(layer: self, keyPath: "shadowPath", oldValue: oldValue, newValue: newValue)
        }
    }

    /// An optional dictionary used to store property values that aren't explicitly defined by the layer.
    open var style: [AnyHashable: Any]?

    /// A Boolean indicating whether the layer is allowed to perform edge antialiasing.
    open var allowsEdgeAntialiasing: Bool = false

    /// A Boolean indicating whether the layer is allowed to composite itself as a group separate from its parent.
    open var allowsGroupOpacity: Bool = true

    // MARK: - Layer Filters

    /// An array of Core Image filters to apply to the contents of the layer and its sublayers. Animatable.
    open var filters: [Any]?

    /// A CoreImage filter used to composite the layer and the content behind it. Animatable.
    open var compositingFilter: Any?

    /// An array of Core Image filters to apply to the content immediately behind the layer. Animatable.
    open var backgroundFilters: [Any]?

    /// The filter used when reducing the size of the content.
    open var minificationFilter: CALayerContentsFilter = .linear

    /// The bias factor used by the minification filter to determine the levels of detail.
    open var minificationFilterBias: Float = 0

    /// The filter used when increasing the size of the content.
    open var magnificationFilter: CALayerContentsFilter = .linear

    // MARK: - Configuring the Layer's Rendering Behavior

    /// A Boolean value indicating whether the layer contains completely opaque content.
    open var isOpaque: Bool = false

    /// A bitmask defining how the edges of the receiver are rasterized.
    open var edgeAntialiasingMask: CAEdgeAntialiasingMask = [.layerLeftEdge, .layerRightEdge, .layerBottomEdge, .layerTopEdge]

    /// Returns a Boolean indicating whether the layer content is implicitly flipped when rendered.
    open func contentsAreFlipped() -> Bool {
        return false
    }

    /// A Boolean that indicates whether the geometry of the layer and its sublayers is flipped vertically.
    open var isGeometryFlipped: Bool = false

    /// A Boolean indicating whether drawing commands are deferred and processed asynchronously in a background thread.
    open var drawsAsynchronously: Bool = false

    /// A Boolean that indicates whether the layer is rendered as a bitmap before compositing. Animatable.
    open var shouldRasterize: Bool = false

    /// The scale at which to rasterize content, relative to the coordinate space of the layer. Animatable.
    open var rasterizationScale: CGFloat = 1.0

    /// A hint for the desired storage format of the layer contents.
    open var contentsFormat: CALayerContentsFormat = .RGBA8Uint

    /// Renders the layer and its sublayers into the specified context.
    ///
    /// This method renders the layer's contents, including its visual appearance
    /// (background color, border, shadow, etc.) and any sublayers.
    ///
    /// - Parameter ctx: The graphics context in which to render.
    open func render(in ctx: CGContext) {
        // Save the graphics state
        ctx.saveGState()
        defer { ctx.restoreGState() }

        // Skip hidden layers
        guard !isHidden && opacity > 0 else { return }

        // Apply transform
        if !CATransform3DIsIdentity(_transform) {
            let affine = CATransform3DGetAffineTransform(_transform)
            ctx.concatenate(affine)
        }

        // Apply opacity
        ctx.setAlpha(CGFloat(opacity))

        // Set up clipping if masksToBounds is true
        if masksToBounds {
            let clipPath = CGPath(roundedRect: bounds, cornerWidth: cornerRadius, cornerHeight: cornerRadius, transform: nil)
            ctx.addPath(clipPath)
            ctx.clip()
        }

        // Draw shadow (before content)
        if shadowOpacity > 0, let shadowColor = shadowColor {
            ctx.setShadow(
                offset: shadowOffset,
                blur: shadowRadius,
                color: shadowColor
            )
        }

        // Draw background color
        if let bgColor = backgroundColor {
            ctx.setFillColor(bgColor)
            if cornerRadius > 0 {
                let path = CGPath(roundedRect: bounds, cornerWidth: cornerRadius, cornerHeight: cornerRadius, transform: nil)
                ctx.addPath(path)
                ctx.fillPath()
            } else {
                ctx.fill(bounds)
            }
        }

        // Draw contents
        if let contents = contents {
            drawContents(contents, in: ctx)
        }

        // Let delegate draw if needed
        delegate?.draw(self, in: ctx)

        // Draw border
        if borderWidth > 0, let borderColor = borderColor {
            ctx.setStrokeColor(borderColor)
            ctx.setLineWidth(borderWidth)
            if cornerRadius > 0 {
                let path = CGPath(roundedRect: bounds.insetBy(dx: borderWidth / 2, dy: borderWidth / 2),
                                  cornerWidth: cornerRadius, cornerHeight: cornerRadius, transform: nil)
                ctx.addPath(path)
                ctx.strokePath()
            } else {
                ctx.stroke(bounds.insetBy(dx: borderWidth / 2, dy: borderWidth / 2))
            }
        }

        // Render sublayers
        if let sublayers = sublayers {
            for sublayer in sublayers {
                ctx.saveGState()

                // Translate to sublayer position
                let sublayerOrigin = CGPoint(
                    x: sublayer.position.x - sublayer.bounds.width * sublayer.anchorPoint.x,
                    y: sublayer.position.y - sublayer.bounds.height * sublayer.anchorPoint.y
                )
                ctx.translateBy(x: sublayerOrigin.x, y: sublayerOrigin.y)

                // Apply sublayer transform
                if !CATransform3DIsIdentity(sublayerTransform) {
                    let affine = CATransform3DGetAffineTransform(sublayerTransform)
                    ctx.concatenate(affine)
                }

                sublayer.render(in: ctx)
                ctx.restoreGState()
            }
        }
    }

    /// Draws the contents image into the context.
    private func drawContents(_ image: CGImage, in ctx: CGContext) {
        let destRect = calculateContentsRect(for: image)
        ctx.draw(image, in: destRect)
    }

    /// Calculates the destination rectangle for drawing contents based on contentsGravity.
    private func calculateContentsRect(for image: CGImage) -> CGRect {
        let imageSize = CGSize(width: CGFloat(image.width), height: CGFloat(image.height))
        let boundsSize = bounds.size

        switch contentsGravity {
        case .center:
            return CGRect(
                x: (boundsSize.width - imageSize.width) / 2,
                y: (boundsSize.height - imageSize.height) / 2,
                width: imageSize.width,
                height: imageSize.height
            )
        case .resize:
            return bounds
        case .resizeAspect:
            let scale = min(boundsSize.width / imageSize.width, boundsSize.height / imageSize.height)
            let scaledSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
            return CGRect(
                x: (boundsSize.width - scaledSize.width) / 2,
                y: (boundsSize.height - scaledSize.height) / 2,
                width: scaledSize.width,
                height: scaledSize.height
            )
        case .resizeAspectFill:
            let scale = max(boundsSize.width / imageSize.width, boundsSize.height / imageSize.height)
            let scaledSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
            return CGRect(
                x: (boundsSize.width - scaledSize.width) / 2,
                y: (boundsSize.height - scaledSize.height) / 2,
                width: scaledSize.width,
                height: scaledSize.height
            )
        case .top:
            return CGRect(x: (boundsSize.width - imageSize.width) / 2, y: 0, width: imageSize.width, height: imageSize.height)
        case .bottom:
            return CGRect(x: (boundsSize.width - imageSize.width) / 2, y: boundsSize.height - imageSize.height, width: imageSize.width, height: imageSize.height)
        case .left:
            return CGRect(x: 0, y: (boundsSize.height - imageSize.height) / 2, width: imageSize.width, height: imageSize.height)
        case .right:
            return CGRect(x: boundsSize.width - imageSize.width, y: (boundsSize.height - imageSize.height) / 2, width: imageSize.width, height: imageSize.height)
        case .topLeft:
            return CGRect(origin: .zero, size: imageSize)
        case .topRight:
            return CGRect(x: boundsSize.width - imageSize.width, y: 0, width: imageSize.width, height: imageSize.height)
        case .bottomLeft:
            return CGRect(x: 0, y: boundsSize.height - imageSize.height, width: imageSize.width, height: imageSize.height)
        case .bottomRight:
            return CGRect(x: boundsSize.width - imageSize.width, y: boundsSize.height - imageSize.height, width: imageSize.width, height: imageSize.height)
        default:
            return bounds
        }
    }

    // MARK: - Modifying the Layer Geometry

    /// The layer's frame rectangle.
    ///
    /// The frame is a derived property computed from `bounds`, `position`, `anchorPoint`, and `transform`.
    /// When the transform is not the identity, the frame represents the smallest rectangle that
    /// completely contains the transformed layer.
    open var frame: CGRect {
        get {
            // If transform is identity, compute simple frame
            if CATransform3DIsIdentity(_transform) {
                let width = _bounds.size.width
                let height = _bounds.size.height
                let x = _position.x - width * _anchorPoint.x
                let y = _position.y - height * _anchorPoint.y
                return CGRect(x: x, y: y, width: width, height: height)
            } else {
                // For non-identity transforms, compute the bounding box of transformed corners
                let width = _bounds.size.width
                let height = _bounds.size.height

                // Calculate the four corners relative to anchor point
                let corners = [
                    CGPoint(x: -width * _anchorPoint.x, y: -height * _anchorPoint.y),
                    CGPoint(x: width * (1 - _anchorPoint.x), y: -height * _anchorPoint.y),
                    CGPoint(x: -width * _anchorPoint.x, y: height * (1 - _anchorPoint.y)),
                    CGPoint(x: width * (1 - _anchorPoint.x), y: height * (1 - _anchorPoint.y))
                ]

                // Transform corners and find bounding box
                var minX = CGFloat.infinity
                var minY = CGFloat.infinity
                var maxX = -CGFloat.infinity
                var maxY = -CGFloat.infinity

                for corner in corners {
                    // Apply 3D transform (simplified 2D projection)
                    let tx = corner.x * _transform.m11 + corner.y * _transform.m21 + _transform.m41
                    let ty = corner.x * _transform.m12 + corner.y * _transform.m22 + _transform.m42

                    let transformedX = _position.x + tx
                    let transformedY = _position.y + ty

                    minX = min(minX, transformedX)
                    minY = min(minY, transformedY)
                    maxX = max(maxX, transformedX)
                    maxY = max(maxY, transformedY)
                }

                return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
            }
        }
        set {
            // Setting frame updates bounds.size and position
            // This assumes identity transform; for non-identity transforms, behavior is undefined
            _bounds.size = newValue.size
            _position = CGPoint(
                x: newValue.origin.x + newValue.size.width * _anchorPoint.x,
                y: newValue.origin.y + newValue.size.height * _anchorPoint.y
            )
        }
    }

    private var _bounds: CGRect = .zero
    /// The layer's bounds rectangle. Animatable.
    open var bounds: CGRect {
        get { return _bounds }
        set {
            let oldValue = _bounds
            _bounds = newValue
            CATransaction.registerChange(layer: self, keyPath: "bounds", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _position: CGPoint = .zero
    /// The layer's position in its superlayer's coordinate space. Animatable.
    open var position: CGPoint {
        get { return _position }
        set {
            let oldValue = _position
            _position = newValue
            CATransaction.registerChange(layer: self, keyPath: "position", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _zPosition: CGFloat = 0
    /// The layer's position on the z axis. Animatable.
    open var zPosition: CGFloat {
        get { return _zPosition }
        set {
            let oldValue = _zPosition
            _zPosition = newValue
            CATransaction.registerChange(layer: self, keyPath: "zPosition", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _anchorPointZ: CGFloat = 0
    /// The anchor point for the layer's position along the z axis. Animatable.
    open var anchorPointZ: CGFloat {
        get { return _anchorPointZ }
        set {
            let oldValue = _anchorPointZ
            _anchorPointZ = newValue
            CATransaction.registerChange(layer: self, keyPath: "anchorPointZ", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _anchorPoint: CGPoint = CGPoint(x: 0.5, y: 0.5)
    /// Defines the anchor point of the layer's bounds rectangle. Animatable.
    open var anchorPoint: CGPoint {
        get { return _anchorPoint }
        set {
            let oldValue = _anchorPoint
            _anchorPoint = newValue
            CATransaction.registerChange(layer: self, keyPath: "anchorPoint", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _contentsScale: CGFloat = 1.0
    /// The scale factor applied to the layer. Animatable.
    open var contentsScale: CGFloat {
        get { return _contentsScale }
        set {
            let oldValue = _contentsScale
            let clampedValue = max(0, newValue)
            _contentsScale = clampedValue
            CATransaction.registerChange(layer: self, keyPath: "contentsScale", oldValue: oldValue, newValue: clampedValue)
        }
    }

    // MARK: - Managing the Layer's Transform

    private var _transform: CATransform3D = CATransform3DIdentity
    /// The transform applied to the layer's contents. Animatable.
    open var transform: CATransform3D {
        get { return _transform }
        set {
            let oldValue = _transform
            _transform = newValue
            CATransaction.registerChange(layer: self, keyPath: "transform", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _sublayerTransform: CATransform3D = CATransform3DIdentity
    /// Specifies the transform to apply to sublayers when rendering. Animatable.
    open var sublayerTransform: CATransform3D {
        get { return _sublayerTransform }
        set {
            let oldValue = _sublayerTransform
            _sublayerTransform = newValue
            CATransaction.registerChange(layer: self, keyPath: "sublayerTransform", oldValue: oldValue, newValue: newValue)
        }
    }

    /// Returns an affine version of the layer's transform.
    open func affineTransform() -> CGAffineTransform {
        return CATransform3DGetAffineTransform(_transform)
    }

    /// Sets the layer's transform to the specified affine transform.
    open func setAffineTransform(_ m: CGAffineTransform) {
        _transform = CATransform3DMakeAffineTransform(m)
    }

    // MARK: - Managing the Layer Hierarchy

    private var _sublayers: [CALayer]?
    /// An array containing the layer's sublayers.
    open var sublayers: [CALayer]? {
        get { return _sublayers }
        set {
            // Remove old sublayers
            _sublayers?.forEach { $0._superlayer = nil }
            // Set new sublayers
            _sublayers = newValue
            _sublayers?.forEach { $0._superlayer = self }
        }
    }

    private weak var _superlayer: CALayer?
    /// The superlayer of the layer.
    open var superlayer: CALayer? {
        return _superlayer
    }

    /// Appends the layer to the layer's list of sublayers.
    open func addSublayer(_ layer: CALayer) {
        layer.removeFromSuperlayer()
        if _sublayers == nil {
            _sublayers = []
        }
        _sublayers?.append(layer)
        layer._superlayer = self
    }

    /// Detaches the layer from its parent layer.
    open func removeFromSuperlayer() {
        guard let superlayer = _superlayer else { return }
        superlayer._sublayers?.removeAll { $0 === self }
        _superlayer = nil
    }

    /// Inserts the specified layer into the receiver's list of sublayers at the specified index.
    open func insertSublayer(_ layer: CALayer, at idx: UInt32) {
        layer.removeFromSuperlayer()
        if _sublayers == nil {
            _sublayers = []
        }
        let index = min(Int(idx), _sublayers?.count ?? 0)
        _sublayers?.insert(layer, at: index)
        layer._superlayer = self
    }

    /// Inserts the specified sublayer below a different sublayer that already belongs to the receiver.
    open func insertSublayer(_ layer: CALayer, below sibling: CALayer?) {
        layer.removeFromSuperlayer()
        if _sublayers == nil {
            _sublayers = []
        }
        if let sibling = sibling, let index = _sublayers?.firstIndex(where: { $0 === sibling }) {
            _sublayers?.insert(layer, at: index)
        } else {
            _sublayers?.insert(layer, at: 0)
        }
        layer._superlayer = self
    }

    /// Inserts the specified sublayer above a different sublayer that already belongs to the receiver.
    open func insertSublayer(_ layer: CALayer, above sibling: CALayer?) {
        layer.removeFromSuperlayer()
        if _sublayers == nil {
            _sublayers = []
        }
        if let sibling = sibling, let index = _sublayers?.firstIndex(where: { $0 === sibling }) {
            _sublayers?.insert(layer, at: index + 1)
        } else {
            _sublayers?.append(layer)
        }
        layer._superlayer = self
    }

    /// Replaces the specified sublayer with a different layer object.
    open func replaceSublayer(_ oldLayer: CALayer, with newLayer: CALayer) {
        guard let index = _sublayers?.firstIndex(where: { $0 === oldLayer }) else { return }
        newLayer.removeFromSuperlayer()
        oldLayer._superlayer = nil
        _sublayers?[index] = newLayer
        newLayer._superlayer = self
    }

    // MARK: - Updating Layer Display

    private var _needsDisplay: Bool = false

    /// Marks the layer's contents as needing to be updated.
    open func setNeedsDisplay() {
        _needsDisplay = true
    }

    /// Marks the region within the specified rectangle as needing to be updated.
    open func setNeedsDisplay(_ r: CGRect) {
        _needsDisplay = true
    }

    /// A Boolean indicating whether the layer contents must be updated when its bounds rectangle changes.
    open var needsDisplayOnBoundsChange: Bool = false

    /// Initiates the update process for a layer if it is currently marked as needing an update.
    open func displayIfNeeded() {
        if _needsDisplay {
            display()
            _needsDisplay = false
        }
    }

    /// Returns a Boolean indicating whether the layer has been marked as needing an update.
    open func needsDisplay() -> Bool {
        return _needsDisplay
    }

    /// Returns a Boolean indicating whether changes to the specified key require the layer to be redisplayed.
    open class func needsDisplay(forKey key: String) -> Bool {
        return false
    }

    // MARK: - Layer Animations

    private var _animations: [String: CAAnimation] = [:]

    private var _animationKeyCounter: Int = 0

    /// Add the specified animation object to the layer's render tree.
    ///
    /// If an animation with the same key already exists, it is replaced.
    /// The replaced animation's delegate receives `animationDidStop(_:finished:)`
    /// with `finished: false` before the new animation starts.
    open func add(_ anim: CAAnimation, forKey key: String?) {
        let animKey: String
        if let key = key {
            animKey = key
        } else {
            _animationKeyCounter += 1
            animKey = "animation_\(_animationKeyCounter)"
        }

        // If there's an existing animation with this key, stop it properly
        if let existingAnimation = _animations[animKey] {
            if !existingAnimation.isFinished {
                existingAnimation.markFinished(completed: false)
            }
        }

        // Set up animation internal state
        anim.addedTime = CACurrentMediaTime()
        anim.isFinished = false
        anim.attachedLayer = self
        anim.animationKey = animKey

        _animations[animKey] = anim

        // Notify delegate that animation started
        anim.delegate?.animationDidStart(anim)
    }

    /// Returns the animation object with the specified identifier.
    open func animation(forKey key: String) -> CAAnimation? {
        return _animations[key]
    }

    /// Remove all animations attached to the layer.
    open func removeAllAnimations() {
        // Notify delegates that animations were stopped
        for (_, animation) in _animations {
            if !animation.isFinished {
                animation.markFinished(completed: false)
            }
        }
        _animations.removeAll()
    }

    /// Remove the animation object with the specified key.
    open func removeAnimation(forKey key: String) {
        if let animation = _animations[key] {
            // Notify delegate that animation was stopped (not completed naturally)
            if !animation.isFinished {
                animation.markFinished(completed: false)
            }
        }
        _animations.removeValue(forKey: key)
    }

    /// Returns an array of strings that identify the animations currently attached to the layer.
    open func animationKeys() -> [String]? {
        return _animations.isEmpty ? nil : Array(_animations.keys)
    }

    /// Checks for completed animations and removes them if needed.
    ///
    /// This method is called by `CAAnimationEngine` during the display refresh cycle
    /// to process animations that have naturally completed.
    ///
    /// You typically don't need to call this method directly. Instead, use
    /// `CAAnimationEngine.shared` to manage the animation loop.
    public func processAnimationCompletions() {
        let currentTime = CACurrentMediaTime()
        var keysToRemove: [String] = []

        for (key, animation) in _animations {
            guard !animation.isFinished else {
                // Already finished, check if it should be removed
                if animation.isRemovedOnCompletion {
                    keysToRemove.append(key)
                }
                continue
            }

            let elapsed = currentTime - animation.addedTime - animation.beginTime
            let totalDuration = animation.totalDuration

            if elapsed >= totalDuration {
                // Animation has completed
                animation.markFinished(completed: true)

                if animation.isRemovedOnCompletion {
                    keysToRemove.append(key)
                }
            }
        }

        // Remove completed animations
        for key in keysToRemove {
            _animations.removeValue(forKey: key)
        }
    }

    // MARK: - Managing Layer Resizing and Layout

    /// The object responsible for laying out the layer's sublayers.
    open var layoutManager: (any CALayoutManager)?

    private var _needsLayout: Bool = false

    /// Invalidates the layer's layout and marks it as needing an update.
    open func setNeedsLayout() {
        _needsLayout = true
    }

    /// Tells the layer to update its layout.
    open func layoutSublayers() {
        layoutManager?.layoutSublayers(of: self)
        delegate?.layoutSublayers(of: self)
    }

    /// Recalculate the receiver's layout, if required.
    open func layoutIfNeeded() {
        if _needsLayout {
            layoutSublayers()
            _needsLayout = false
        }
    }

    /// Returns a Boolean indicating whether the layer has been marked as needing a layout update.
    open func needsLayout() -> Bool {
        return _needsLayout
    }

    /// A bitmask defining how the layer is resized when the bounds of its superlayer changes.
    open var autoresizingMask: CAAutoresizingMask = []

    /// Informs the receiver that the size of its superlayer changed.
    open func resize(withOldSuperlayerSize size: CGSize) {
        // Default implementation
    }

    /// Informs the receiver's sublayers that the receiver's size has changed.
    open func resizeSublayers(withOldSize size: CGSize) {
        sublayers?.forEach { $0.resize(withOldSuperlayerSize: size) }
    }

    /// Returns the preferred size of the layer in the coordinate space of its superlayer.
    open func preferredFrameSize() -> CGSize {
        return layoutManager?.preferredSize(of: self) ?? bounds.size
    }

    // MARK: - Managing Layer Constraints

    /// The constraints used to position this layer relative to sibling layers or its superlayer.
    ///
    /// Constraints define geometric relationships between this layer and other layers in the same
    /// sibling group. The constraint's `sourceName` property identifies the reference layer by name,
    /// or "superlayer" to reference the parent layer.
    open var constraints: [CAConstraint]?

    /// Adds the specified constraint to the layer.
    ///
    /// - Parameter c: The constraint to add. The constraint defines how this layer is positioned
    ///   relative to a sibling layer or its superlayer.
    open func addConstraint(_ c: CAConstraint) {
        if constraints == nil {
            constraints = []
        }
        constraints?.append(c)
    }

    // MARK: - Getting the Layer's Actions

    /// Returns the action object assigned to the specified key.
    open func action(forKey event: String) -> (any CAAction)? {
        // First check the delegate
        if let action = delegate?.action(for: self, forKey: event) {
            return action
        }
        // Then check the actions dictionary
        if let action = actions?[event] {
            return action
        }
        // Finally check the class default
        return Self.defaultAction(forKey: event)
    }

    /// A dictionary containing layer actions.
    open var actions: [String: any CAAction]?

    /// The list of animatable property keys.
    ///
    /// These properties support implicit animations when changed outside of a
    /// `CATransaction.setDisableActions(true)` block.
    private static let animatableKeys: Set<String> = [
        "opacity",
        "bounds",
        "bounds.origin",
        "bounds.size",
        "position",
        "position.x",
        "position.y",
        "zPosition",
        "anchorPoint",
        "anchorPointZ",
        "transform",
        "transform.scale",
        "transform.scale.x",
        "transform.scale.y",
        "transform.scale.z",
        "transform.rotation",
        "transform.rotation.x",
        "transform.rotation.y",
        "transform.rotation.z",
        "transform.translation",
        "transform.translation.x",
        "transform.translation.y",
        "transform.translation.z",
        "sublayerTransform",
        "cornerRadius",
        "borderWidth",
        "borderColor",
        "backgroundColor",
        "shadowOpacity",
        "shadowRadius",
        "shadowOffset",
        "shadowColor",
        "shadowPath",
        "contentsRect",
        "contentsCenter",
        "contentsScale",
        "isHidden",
        "masksToBounds",
        "isDoubleSided",
        "shouldRasterize",
        "rasterizationScale"
    ]

    /// Returns the default action for the current class.
    ///
    /// For animatable properties, this returns a `CABasicAnimation` configured
    /// with the current transaction's duration and timing function.
    open class func defaultAction(forKey event: String) -> (any CAAction)? {
        guard animatableKeys.contains(event) else { return nil }

        let animation = CABasicAnimation(keyPath: event)
        animation.duration = CATransaction.animationDuration()
        if let timingFunction = CATransaction.animationTimingFunction() {
            animation.timingFunction = timingFunction
        }
        return animation
    }

    // MARK: - Mapping Between Coordinate and Time Spaces

    /// Converts a point from this layer's local coordinates to its superlayer's coordinates.
    private func convertPointToSuperlayer(_ p: CGPoint) -> CGPoint {
        // First, offset by bounds origin (scroll offset)
        var point = CGPoint(x: p.x - _bounds.origin.x, y: p.y - _bounds.origin.y)

        // Then offset by anchor point (convert to anchor-relative coordinates)
        point.x -= _bounds.size.width * _anchorPoint.x
        point.y -= _bounds.size.height * _anchorPoint.y

        // Apply transform
        if !CATransform3DIsIdentity(_transform) {
            let tx = point.x * _transform.m11 + point.y * _transform.m21 + _transform.m41
            let ty = point.x * _transform.m12 + point.y * _transform.m22 + _transform.m42
            point = CGPoint(x: tx, y: ty)
        }

        // Translate to superlayer coordinates using position
        point.x += _position.x
        point.y += _position.y

        return point
    }

    /// Converts a point from this layer's superlayer's coordinates to local coordinates.
    private func convertPointFromSuperlayer(_ p: CGPoint) -> CGPoint {
        // Translate from superlayer coordinates using position
        var point = CGPoint(x: p.x - _position.x, y: p.y - _position.y)

        // Apply inverse transform
        if !CATransform3DIsIdentity(_transform) {
            let inverted = CATransform3DInvert(_transform)
            let tx = point.x * inverted.m11 + point.y * inverted.m21 + inverted.m41
            let ty = point.x * inverted.m12 + point.y * inverted.m22 + inverted.m42
            point = CGPoint(x: tx, y: ty)
        }

        // Offset by anchor point (convert from anchor-relative to bounds-relative)
        point.x += _bounds.size.width * _anchorPoint.x
        point.y += _bounds.size.height * _anchorPoint.y

        // Add bounds origin (scroll offset)
        point.x += _bounds.origin.x
        point.y += _bounds.origin.y

        return point
    }

    /// Returns the chain of layers from this layer up to the root (or until reaching the specified ancestor).
    private func ancestorChain(upTo ancestor: CALayer? = nil) -> [CALayer] {
        var chain: [CALayer] = [self]
        var current: CALayer? = _superlayer
        while let layer = current {
            if layer === ancestor { break }
            chain.append(layer)
            current = layer._superlayer
        }
        return chain
    }

    /// Converts the point from the specified layer's coordinate system to the receiver's coordinate system.
    open func convert(_ p: CGPoint, from l: CALayer?) -> CGPoint {
        guard let sourceLayer = l else { return p }
        if sourceLayer === self { return p }

        // Convert from source layer up to common ancestor (or root)
        var point = p
        var current: CALayer? = sourceLayer
        while let layer = current, layer !== self {
            point = layer.convertPointToSuperlayer(point)
            current = layer._superlayer
        }

        // If we found self in the chain, we're done
        if current === self {
            return point
        }

        // Otherwise, we need to convert down from root to self
        // First, get the chain from self to root
        let selfChain = ancestorChain()

        // Find common ancestor
        var sourceAncestors = Set<ObjectIdentifier>()
        current = sourceLayer
        while let layer = current {
            sourceAncestors.insert(ObjectIdentifier(layer))
            current = layer._superlayer
        }

        var commonAncestorIndex = selfChain.count - 1
        for (index, layer) in selfChain.enumerated() {
            if sourceAncestors.contains(ObjectIdentifier(layer)) {
                commonAncestorIndex = index
                break
            }
        }

        // Convert from source to common ancestor
        point = p
        current = sourceLayer
        while let layer = current {
            if selfChain.indices.contains(commonAncestorIndex) && layer === selfChain[commonAncestorIndex] {
                break
            }
            point = layer.convertPointToSuperlayer(point)
            current = layer._superlayer
        }

        // Convert from common ancestor down to self
        if commonAncestorIndex > 0 {
            for i in stride(from: commonAncestorIndex - 1, through: 0, by: -1) {
                point = selfChain[i].convertPointFromSuperlayer(point)
            }
        }

        return point
    }

    /// Converts the point from the receiver's coordinate system to the specified layer's coordinate system.
    open func convert(_ p: CGPoint, to l: CALayer?) -> CGPoint {
        guard let targetLayer = l else { return p }
        if targetLayer === self { return p }

        // Use the inverse operation
        return targetLayer.convert(p, from: self)
    }

    /// Converts the rectangle from the specified layer's coordinate system to the receiver's coordinate system.
    open func convert(_ r: CGRect, from l: CALayer?) -> CGRect {
        guard let sourceLayer = l else { return r }
        if sourceLayer === self { return r }

        // For rectangles with transforms, we need to convert all four corners
        // and compute the bounding box
        let topLeft = convert(CGPoint(x: r.minX, y: r.minY), from: l)
        let topRight = convert(CGPoint(x: r.maxX, y: r.minY), from: l)
        let bottomLeft = convert(CGPoint(x: r.minX, y: r.maxY), from: l)
        let bottomRight = convert(CGPoint(x: r.maxX, y: r.maxY), from: l)

        let minX = min(topLeft.x, topRight.x, bottomLeft.x, bottomRight.x)
        let maxX = max(topLeft.x, topRight.x, bottomLeft.x, bottomRight.x)
        let minY = min(topLeft.y, topRight.y, bottomLeft.y, bottomRight.y)
        let maxY = max(topLeft.y, topRight.y, bottomLeft.y, bottomRight.y)

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    /// Converts the rectangle from the receiver's coordinate system to the specified layer's coordinate system.
    open func convert(_ r: CGRect, to l: CALayer?) -> CGRect {
        guard let targetLayer = l else { return r }
        if targetLayer === self { return r }

        // For rectangles with transforms, we need to convert all four corners
        // and compute the bounding box
        let topLeft = convert(CGPoint(x: r.minX, y: r.minY), to: l)
        let topRight = convert(CGPoint(x: r.maxX, y: r.minY), to: l)
        let bottomLeft = convert(CGPoint(x: r.minX, y: r.maxY), to: l)
        let bottomRight = convert(CGPoint(x: r.maxX, y: r.maxY), to: l)

        let minX = min(topLeft.x, topRight.x, bottomLeft.x, bottomRight.x)
        let maxX = max(topLeft.x, topRight.x, bottomLeft.x, bottomRight.x)
        let minY = min(topLeft.y, topRight.y, bottomLeft.y, bottomRight.y)
        let maxY = max(topLeft.y, topRight.y, bottomLeft.y, bottomRight.y)

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    /// Converts the time interval from the specified layer's time space to the receiver's time space.
    open func convertTime(_ t: CFTimeInterval, from l: CALayer?) -> CFTimeInterval {
        guard let sourceLayer = l else { return t }
        if sourceLayer === self { return t }

        // Convert time considering speed and timeOffset along the layer chain
        var time = t
        var current: CALayer? = sourceLayer

        // Convert from source up to root
        while let layer = current {
            if layer === self { break }
            // Apply layer's time mapping: parentTime = (localTime - timeOffset) * speed + beginTime
            time = (time - layer.timeOffset) * CFTimeInterval(layer.speed) + layer.beginTime
            current = layer._superlayer
        }

        if current === self {
            return time
        }

        // Convert from root down to self (inverse time mapping)
        let selfChain = ancestorChain()
        for layer in selfChain.reversed() {
            if layer === self { continue }
            // Inverse: localTime = (parentTime - beginTime) / speed + timeOffset
            if layer.speed != 0 {
                time = (time - layer.beginTime) / CFTimeInterval(layer.speed) + layer.timeOffset
            }
        }

        return time
    }

    /// Converts the time interval from the receiver's time space to the specified layer's time space.
    open func convertTime(_ t: CFTimeInterval, to l: CALayer?) -> CFTimeInterval {
        guard let targetLayer = l else { return t }
        if targetLayer === self { return t }

        return targetLayer.convertTime(t, from: self)
    }

    // MARK: - Hit Testing

    /// Returns the farthest descendant of the receiver in the layer hierarchy (including itself)
    /// that contains the specified point.
    ///
    /// - Parameter p: A point in the coordinate system of the receiver's superlayer.
    /// - Returns: The layer that contains the point, or nil if the point lies outside the receiver's bounds.
    open func hitTest(_ p: CGPoint) -> CALayer? {
        guard !isHidden && opacity > 0 else { return nil }

        // Convert from superlayer coordinates to local coordinates
        let localPoint = convertPointFromSuperlayer(p)
        guard contains(localPoint) else { return nil }

        // Check sublayers in reverse order (front to back)
        // localPoint is in self's coordinate system, which is the sublayer's superlayer coordinate system
        if let sublayers = sublayers?.reversed() {
            for sublayer in sublayers {
                if let hit = sublayer.hitTest(localPoint) {
                    return hit
                }
            }
        }

        return self
    }

    /// Returns whether the receiver contains a specified point.
    open func contains(_ p: CGPoint) -> Bool {
        return bounds.contains(p)
    }

    // MARK: - Scrolling

    /// The visible region of the layer in its own coordinate space.
    open var visibleRect: CGRect {
        return bounds
    }

    /// Initiates a scroll in the layer's closest ancestor scroll layer so that the specified point
    /// lies at the origin of the scroll layer.
    open func scroll(_ p: CGPoint) {
        // Default implementation
    }

    /// Initiates a scroll in the layer's closest ancestor scroll layer so that the specified rectangle
    /// becomes visible.
    open func scrollRectToVisible(_ r: CGRect) {
        // Default implementation
    }

    // MARK: - Identifying the Layer

    private var _name: String?
    /// The name of the receiver.
    open var name: String? {
        get { return _name }
        set { _name = newValue }
    }

    // MARK: - Key-Value Coding Extensions

    /// Returns a Boolean indicating whether the value of the specified key should be archived.
    open func shouldArchiveValue(forKey key: String) -> Bool {
        return true
    }

    /// Specifies the default value associated with the specified key.
    open class func defaultValue(forKey key: String) -> Any? {
        return nil
    }

    // MARK: - Corner Curve

    /// Returns the expansion factor required when using continuous corner curves.
    open class func cornerCurveExpansionFactor(_ curve: CALayerCornerCurve) -> CGFloat {
        switch curve {
        case .continuous:
            return 1.528665
        default:
            return 1.0
        }
    }

    // MARK: - CAMediaTiming

    /// Specifies the begin time of the receiver in relation to its parent object, if applicable.
    open var beginTime: CFTimeInterval = 0

    /// Specifies an additional time offset in active local time.
    open var timeOffset: CFTimeInterval = 0

    /// Determines the number of times the animation will repeat.
    open var repeatCount: Float = 0

    /// Determines how many seconds the animation will repeat for.
    open var repeatDuration: CFTimeInterval = 0

    /// Specifies the basic duration of the animation, in seconds.
    open var duration: CFTimeInterval = 0

    /// Specifies how time is mapped to receiver's time space from the parent time space.
    open var speed: Float = 1

    /// Determines if the receiver plays in the reverse upon completion.
    open var autoreverses: Bool = false

    /// Determines if the receiver's presentation is frozen or removed once its active duration has completed.
    open var fillMode: CAMediaTimingFillMode = .removed

    // MARK: - Hashable

    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }

    public static func == (lhs: CALayer, rhs: CALayer) -> Bool {
        return lhs === rhs
    }
}
