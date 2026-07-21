//
//  CALayer.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation


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

            // Copy content properties — use backing storage to bypass
            // markDirty during init (we reset dirty state below).
            self.contents = otherLayer.contents
            self._contentsRect = otherLayer._contentsRect
            self.contentsCenter = otherLayer.contentsCenter
            self.contentsGravity = otherLayer.contentsGravity
            self.contentsFormat = otherLayer.contentsFormat

            // Copy rendering properties — backing storage as above.
            self._isOpaque = otherLayer._isOpaque
            self.isGeometryFlipped = otherLayer.isGeometryFlipped
            self.drawsAsynchronously = otherLayer.drawsAsynchronously
            self._shouldRasterize = otherLayer._shouldRasterize
            self._rasterizationScale = otherLayer._rasterizationScale
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

            // Seed subtree counters from copied state. Shadow fields above are
            // assigned via direct backing-store writes (to mirror the original
            // values without re-clamping); `filters` goes through its setter
            // and updates `_subtreeFilterCount` as a side effect, but the
            // shadow contribution must be seeded explicitly.
            self._subtreeShadowCount = self.selfShadowContribution
        }
    }

    // MARK: - Accessing Related Layer Objects

    /// The presentation layer associated with this layer during animations.
    private var _presentationLayer: CALayer?

    /// Whether this layer is a presentation layer.
    private var _isPresentation: Bool = false

    /// The model layer if this is a presentation layer.
    private weak var _modelLayer: CALayer?

    /// Set only on a presentation layer while a built-in CATransition is active.
    internal var _transitionRenderState: CATransitionRenderState?

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

        // R2.1 cache hit (PERFORMANCE_DESIGN.md §4.1): repeat calls within
        // a single frame must return the same instance. The token is reset
        // to 0 by `markDirty(_:)` so any presentation-affecting mutation
        // forces a recompute on the next call.
        if _presentationCacheIsValid,
           _presentationCacheToken == Self._currentFrameToken,
           let cached = _presentationLayer {
            return cached as? Self
        }

        // Create or update presentation layer
        if _presentationLayer == nil {
            _presentationLayer = createPresentationLayer()
        }

        // Update presentation layer with current animated values
        updatePresentationLayer()
        _presentationCacheToken = Self._currentFrameToken
        _presentationCacheIsValid = true

        return _presentationLayer as? Self
    }

    /// Renderer-internal presentation lookup. Implements the R2.2 self-return
    /// fast path (PERFORMANCE_DESIGN.md §4.1 / §4.2): when no animation is
    /// live AND no presentation-affecting bit is dirty, the model values
    /// **are** the presentation values, so the renderer can read this layer
    /// directly and skip the per-frame allocation.
    ///
    /// Public callers (user code) must keep using `presentation()` to
    /// preserve Apple's documented "distinct copy" semantics — this fast
    /// path is invisible at the public surface.
    internal func _renderTimePresentation() -> CALayer {
        if _isPresentation { return self }

        let allFinished = _animations.allSatisfy { $0.value.isFinished }
        if allFinished
            && _dirtyMask.isDisjoint(with: DirtyFlags.presentationAffecting) {
            return self
        }

        return presentation() ?? self
    }

    /// Returns the receiver's `_sublayers` sorted by `zPosition` (then by
    /// original insertion index for stability). Cached per-frame and
    /// invalidated by `.sublayerHierarchy` / `.sublayerOrdering` so the
    /// renderer skips the O(N log N) sort on clean trees
    /// (PERFORMANCE_DESIGN.md §4.3 / R2.3).
    internal func sortedSublayers() -> [CALayer] {
        guard let subs = _sublayers, !subs.isEmpty else { return [] }
        if _dirtyMask.isDisjoint(with: [.sublayerHierarchy, .sublayerOrdering]),
           _sortedSublayersToken == Self._currentFrameToken,
           _sortedSublayers.count == subs.count {
            return _sortedSublayers
        }
        let sorted = subs.enumerated().sorted { a, b in
            let zA = a.element._renderTimePresentation().zPosition
            let zB = b.element._renderTimePresentation().zPosition
            if zA != zB { return zA < zB }
            return a.offset < b.offset
        }.map(\.element)
        _sortedSublayers = sorted
        _sortedSublayersToken = Self._currentFrameToken
        return sorted
    }

    /// Creates a new presentation layer as a copy of this layer.
    private func createPresentationLayer() -> CALayer {
        let presentationClass = type(of: self)
        let presentation = presentationClass.init(layer: self)
        presentation._isPresentation = true
        presentation._modelLayer = self
        // Presentation layers are read-only consumers and must not
        // contribute to any ancestor's dirty count (PERFORMANCE_DESIGN.md
        // §3.5). init(layer:) seeded the standard initial state; reset it.
        presentation._dirtyMask = []
        presentation._subtreeDirtyCount = 0
        presentation._presentationCacheToken = 0
        presentation._presentationCacheIsValid = false
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
    internal func presentationAtTimeOffset(_ timeOffset: CFTimeInterval) -> Self {
        // Create a new presentation layer copy
        let presentationClass = type(of: self)
        let presentation = presentationClass.init(layer: self)
        presentation._isPresentation = true
        presentation._modelLayer = self
        // See createPresentationLayer() — presentation layers don't
        // contribute to dirty propagation.
        presentation._dirtyMask = []
        presentation._subtreeDirtyCount = 0
        presentation._presentationCacheToken = 0
        presentation._presentationCacheIsValid = false

        // Update with animations at the offset time
        let evaluationTime = CACurrentMediaTime() - timeOffset
        updatePresentationLayer(presentation, at: evaluationTime)

        return presentation
    }

    /// Updates the presentation layer with current animated values.
    private func updatePresentationLayer() {
        guard let presentation = _presentationLayer else { return }

        updatePresentationLayer(presentation, at: CACurrentMediaTime())
    }

    /// Updates a presentation layer with animated values at a specific time.
    private func updatePresentationLayer(_ presentation: CALayer, at currentTime: CFTimeInterval) {

        presentation._transitionRenderState = nil

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
        presentation._anchorPointZ = _anchorPointZ
        presentation._sublayerTransform = _sublayerTransform
        presentation._contentsScale = _contentsScale
        presentation._masksToBounds = _masksToBounds
        presentation._isDoubleSided = _isDoubleSided
        presentation.maskedCorners = maskedCorners
        presentation.cornerCurve = cornerCurve
        presentation._shadowPath = _shadowPath

        // Copy contents-related properties (critical for texture animation)
        presentation.contents = contents
        presentation._contentsRect = _contentsRect
        presentation.contentsCenter = contentsCenter
        presentation.contentsGravity = contentsGravity
        presentation.contentsFormat = contentsFormat

        // Copy render configuration that may change after the presentation
        // object was first allocated.
        presentation.mask = mask
        presentation._isOpaque = _isOpaque
        presentation.isGeometryFlipped = isGeometryFlipped
        presentation.drawsAsynchronously = drawsAsynchronously
        presentation._shouldRasterize = _shouldRasterize
        presentation._rasterizationScale = _rasterizationScale
        presentation.allowsEdgeAntialiasing = allowsEdgeAntialiasing
        presentation.allowsGroupOpacity = allowsGroupOpacity
        presentation.edgeAntialiasingMask = edgeAntialiasingMask
        presentation.filters = filters
        presentation.compositingFilter = compositingFilter
        presentation.backgroundFilters = backgroundFilters
        presentation.minificationFilter = minificationFilter
        presentation.minificationFilterBias = minificationFilterBias
        presentation.magnificationFilter = magnificationFilter

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
            shapePresentation.fillRule = shapeSelf.fillRule
            shapePresentation.lineCap = shapeSelf.lineCap
            shapePresentation.lineDashPattern = shapeSelf.lineDashPattern
            shapePresentation.lineJoin = shapeSelf.lineJoin
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
            gradientPresentation.type = gradientSelf.type
        }

        if let emitterPresentation = presentation as? CAEmitterLayer,
           let emitterSelf = self as? CAEmitterLayer {
            emitterPresentation.emitterCells = emitterSelf.emitterCells
            emitterPresentation._emitterPosition = emitterSelf._emitterPosition
            emitterPresentation._emitterZPosition = emitterSelf._emitterZPosition
            emitterPresentation._emitterSize = emitterSelf._emitterSize
            emitterPresentation._emitterDepth = emitterSelf._emitterDepth
            emitterPresentation.emitterShape = emitterSelf.emitterShape
            emitterPresentation.emitterMode = emitterSelf.emitterMode
            emitterPresentation.renderMode = emitterSelf.renderMode
            emitterPresentation.preservesDepth = emitterSelf.preservesDepth
            emitterPresentation._birthRate = emitterSelf._birthRate
            emitterPresentation._lifetime = emitterSelf._lifetime
            emitterPresentation._velocity = emitterSelf._velocity
            emitterPresentation._scale = emitterSelf._scale
            emitterPresentation._spin = emitterSelf._spin
            emitterPresentation.seed = emitterSelf.seed
        }

        if let replicatorPresentation = presentation as? CAReplicatorLayer,
           let replicatorSelf = self as? CAReplicatorLayer {
            replicatorPresentation.instanceCount = replicatorSelf.instanceCount
            replicatorPresentation.preservesDepth = replicatorSelf.preservesDepth
            replicatorPresentation._instanceDelay = replicatorSelf._instanceDelay
            replicatorPresentation._instanceTransform = replicatorSelf._instanceTransform
            replicatorPresentation._instanceColor = replicatorSelf._instanceColor
            replicatorPresentation._instanceRedOffset = replicatorSelf._instanceRedOffset
            replicatorPresentation._instanceGreenOffset = replicatorSelf._instanceGreenOffset
            replicatorPresentation._instanceBlueOffset = replicatorSelf._instanceBlueOffset
            replicatorPresentation._instanceAlphaOffset = replicatorSelf._instanceAlphaOffset
        }

        if let tiledPresentation = presentation as? CATiledLayer,
           let tiledSelf = self as? CATiledLayer {
            tiledPresentation.levelsOfDetail = tiledSelf.levelsOfDetail
            tiledPresentation.levelsOfDetailBias = tiledSelf.levelsOfDetailBias
            tiledPresentation.tileSize = tiledSelf.tileSize
        }

        if let scrollPresentation = presentation as? CAScrollLayer,
           let scrollSelf = self as? CAScrollLayer {
            scrollPresentation.scrollMode = scrollSelf.scrollMode
        }

        // Animation begin times live in the layer's local time space.
        let layerLocalTime = convertTime(currentTime, from: nil)

        // Apply active animations: non-additive first, then additive.
        // Additive animations accumulate deltas on top of the current value.
        var additiveAnimations: [(String, CAAnimation)] = []
        for (key, animation) in _animations {
            if let propAnim = animation as? CAPropertyAnimation, propAnim.isAdditive {
                additiveAnimations.append((key, animation))
            } else {
                applyAnimation(animation, to: presentation, at: layerLocalTime)
            }
        }
        for (_, animation) in additiveAnimations {
            applyAnimation(animation, to: presentation, at: layerLocalTime)
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

        let singleCycleDuration: CFTimeInterval
        if let springAnimation = animation as? CASpringAnimation {
            // For spring animations, use settlingDuration if duration is not explicitly set
            singleCycleDuration = animation.duration > 0 ? animation.duration : springAnimation.settlingDuration
        } else {
            singleCycleDuration = animation.duration > 0 ? animation.duration : CATransaction.animationDuration()
        }

        let timing = CAMediaTimingEvaluator.evaluate(
            animation,
            parentTime: time,
            duration: singleCycleDuration
        )
        if timing.phase != .before {
            animation.markStarted()
        }
        guard timing.applies(fillMode: animation.fillMode) else { return }
        let progress = timing.progress

        // Apply spring physics or timing function
        var adjustedProgress = progress
        if let springAnimation = animation as? CASpringAnimation {
            // For spring animations, use physics-based interpolation
            // Calculate the actual elapsed time for spring physics
            let springTime = adjustedProgress * singleCycleDuration
            adjustedProgress = CFTimeInterval(springAnimation.springValue(at: springTime))
            // Spring animations can overshoot, so don't clamp to 0-1
        } else if let timingFunction = animation.timingFunction {
            // Apply timing function for non-spring animations
            adjustedProgress = CFTimeInterval(timingFunction.evaluate(at: Float(adjustedProgress)))
            adjustedProgress = max(0, min(1, adjustedProgress))
        } else {
            adjustedProgress = max(0, min(1, adjustedProgress))
        }

        // Interpolate and apply value based on animation type
        applyAnimationValue(
            propertyAnimation,
            to: layer,
            keyPath: keyPath,
            progress: adjustedProgress,
            completedCycles: timing.completedCycles
        )
    }

    /// Applies a transition animation to the presentation layer.
    ///
    /// Transitions provide animated effects when layer content changes.
    /// Unlike property animations, transitions affect the overall appearance
    /// of the layer during the transition period.
    private func applyTransition(_ transition: CATransition, to layer: CALayer, at time: CFTimeInterval) {
        // Calculate transition progress
        let duration = transition.duration > 0 ? transition.duration : CATransaction.animationDuration()
        let timing = CAMediaTimingEvaluator.evaluate(transition, parentTime: time, duration: duration)
        if timing.phase != .before {
            transition.markStarted()
        }
        guard timing.applies(fillMode: transition.fillMode) else { return }
        let progress = timing.progress

        // Apply timing function if available
        var adjustedProgress = progress
        if let timingFunction = transition.timingFunction {
            adjustedProgress = CFTimeInterval(timingFunction.evaluate(at: Float(adjustedProgress)))
        }

        // Apply start/end progress range
        let startProgress = CFTimeInterval(transition.startProgress)
        let endProgress = CFTimeInterval(transition.endProgress)
        let rangedProgress = startProgress + adjustedProgress * (endProgress - startProgress)

        guard let sourceLayer = transition.sourceLayerSnapshot else { return }
        layer._transitionRenderState = CATransitionRenderState(
            sourceLayer: sourceLayer,
            type: transition.type,
            subtype: transition.subtype,
            filter: transition.filter,
            progress: max(0, min(1, rangedProgress))
        )
    }

    /// Applies an animation group to the presentation layer.
    private func applyAnimationGroup(_ group: CAAnimationGroup, to layer: CALayer, at time: CFTimeInterval) {
        guard let animations = group.animations else { return }

        let groupBaseDuration = group.duration > 0 ? group.duration : CATransaction.animationDuration()
        let timing = CAMediaTimingEvaluator.evaluate(group, parentTime: time, duration: groupBaseDuration)
        if timing.phase != .before {
            group.markStarted()
        }
        guard timing.applies(fillMode: group.fillMode) else { return }

        // Children are evaluated in the group's repeating basic time space.
        // The group's duration clips longer children without scaling them.
        for animation in animations {
            let effectiveDuration = animation.duration > 0 ? animation.duration : groupBaseDuration
            applyAnimationWithContext(
                animation,
                to: layer,
                at: timing.basicTime,
                effectiveDuration: effectiveDuration
            )
        }
    }

    /// Applies an animation with explicit timing context (used by animation groups).
    private func applyAnimationWithContext(
        _ animation: CAAnimation,
        to layer: CALayer,
        at time: CFTimeInterval,
        effectiveDuration: CFTimeInterval
    ) {
        // Handle nested animation groups
        if let animationGroup = animation as? CAAnimationGroup {
            applyAnimationGroup(animationGroup, to: layer, at: time)
            return
        }

        guard let propertyAnimation = animation as? CAPropertyAnimation,
              let keyPath = propertyAnimation.keyPath else { return }

        // Get the effective duration for a single cycle
        let singleCycleDuration: CFTimeInterval
        if let springAnimation = animation as? CASpringAnimation {
            singleCycleDuration = effectiveDuration > 0 ? effectiveDuration : springAnimation.settlingDuration
        } else {
            singleCycleDuration = effectiveDuration > 0 ? effectiveDuration : CATransaction.animationDuration()
        }

        guard singleCycleDuration > 0 else { return }

        let timing = CAMediaTimingEvaluator.evaluate(
            animation,
            parentTime: time,
            duration: singleCycleDuration
        )
        if timing.phase != .before {
            animation.markStarted()
        }
        guard timing.applies(fillMode: animation.fillMode) else { return }
        var progress = timing.progress

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

        applyAnimationValue(
            propertyAnimation,
            to: layer,
            keyPath: keyPath,
            progress: progress,
            completedCycles: timing.completedCycles
        )
    }

    /// Applies an animation value to a layer property.
    private func applyAnimationValue(
        _ animation: CAPropertyAnimation,
        to layer: CALayer,
        keyPath: String,
        progress: CFTimeInterval,
        completedCycles: Int
    ) {
        if let keyframeAnimation = animation as? CAKeyframeAnimation {
            applyKeyframeAnimation(keyframeAnimation, to: layer, keyPath: keyPath, progress: progress)
        } else if let basicAnimation = animation as? CABasicAnimation {
            applyBasicAnimation(basicAnimation, to: layer, keyPath: keyPath, progress: progress)
        }

        guard animation.isCumulative, completedCycles > 0 else { return }
        let terminalValue: Any?
        if let basicAnimation = animation as? CABasicAnimation {
            terminalValue = cumulativeTerminalValue(for: basicAnimation, keyPath: keyPath)
        } else if let keyframeAnimation = animation as? CAKeyframeAnimation,
                  let values = keyframeAnimation.values, !values.isEmpty {
            terminalValue = keyframeAnimation.autoreverses ? values.first : values.last
        } else {
            terminalValue = nil
        }
        if let terminalValue {
            applyCumulativeContribution(
                terminalValue,
                cycles: completedCycles,
                to: layer,
                keyPath: keyPath
            )
        }
    }

    /// Applies a basic animation to a layer property.
    private func applyBasicAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        // Check for valueFunction - used to convert scalar values to transforms
        if let valueFunction = animation.valueFunction {
            applyValueFunctionAnimation(animation, valueFunction: valueFunction, to: layer, progress: progress)
            return
        }

        // Dispatch to the correct type-specific method based on keyPath
        switch keyPath {
        // Float/CGFloat properties
        case "opacity", "cornerRadius", "borderWidth", "shadowRadius", "shadowOpacity",
             "zPosition", "anchorPointZ", "contentsScale", "rasterizationScale",
             "strokeStart", "strokeEnd", "lineWidth", "lineDashPhase", "miterLimit",
             "fontSize", "emitterZPosition", "emitterDepth", "birthRate", "lifetime",
             "velocity", "scale", "spin", "instanceDelay", "instanceRedOffset",
             "instanceGreenOffset", "instanceBlueOffset", "instanceAlphaOffset":
            applyFloatAnimation(animation, to: layer, keyPath: keyPath, progress: progress)

        // Point/Size properties
        case "position", "position.x", "position.y", "anchorPoint", "bounds.origin",
             "shadowOffset", "startPoint", "endPoint", "emitterPosition", "emitterSize":
            applyPointAnimation(animation, to: layer, keyPath: keyPath, progress: progress)

        // Rect properties
        case "bounds", "bounds.size", "contentsRect", "contentsCenter":
            applyRectAnimation(animation, to: layer, keyPath: keyPath, progress: progress)

        // Transform properties
        case _ where keyPath == "transform" || keyPath.hasPrefix("transform.") || keyPath == "sublayerTransform" || keyPath == "instanceTransform":
            applyTransformAnimation(animation, to: layer, keyPath: keyPath, progress: progress)

        // Color properties
        case "backgroundColor", "borderColor", "shadowColor", "fillColor", "strokeColor",
             "foregroundColor", "instanceColor":
            applyColorAnimation(animation, to: layer, keyPath: keyPath, progress: progress)

        case "path", "shadowPath":
            applyPathAnimation(animation, to: layer, keyPath: keyPath, progress: progress)

        case "hidden", "isHidden", "masksToBounds", "doubleSided", "isDoubleSided", "shouldRasterize":
            applyBooleanAnimation(animation, to: layer, keyPath: keyPath, progress: progress)

        // Array properties (locations, colors, etc.)
        default:
            applyArrayAnimation(animation, to: layer, keyPath: keyPath, progress: progress)
        }
    }

    /// Core Animation represents Boolean animation endpoints numerically. The
    /// presentation value is false only when the interpolated scalar is exactly
    /// zero, matching QuartzCore's asymmetric false→true / true→false behavior.
    private func applyBooleanAnimation(
        _ animation: CABasicAnimation,
        to layer: CALayer,
        keyPath: String,
        progress: CFTimeInterval
    ) {
        guard let current = booleanAnimationValue(for: keyPath) else { return }
        let from = (animation.fromValue as? Bool) ?? current
        let to = (animation.toValue as? Bool) ?? current
        guard animation.fromValue != nil || animation.toValue != nil else { return }
        let fromScalar: CFTimeInterval = from ? 1 : 0
        let toScalar: CFTimeInterval = to ? 1 : 0
        applyBooleanAnimationValue(
            fromScalar + progress * (toScalar - fromScalar) != 0,
            to: layer,
            keyPath: keyPath
        )
    }

    private func booleanAnimationValue(for keyPath: String) -> Bool? {
        switch keyPath {
        case "hidden", "isHidden": return _isHidden
        case "masksToBounds": return _masksToBounds
        case "doubleSided", "isDoubleSided": return _isDoubleSided
        case "shouldRasterize": return _shouldRasterize
        default: return nil
        }
    }

    private func applyBooleanAnimationValue(
        _ value: Bool,
        to layer: CALayer,
        keyPath: String
    ) {
        switch keyPath {
        case "hidden", "isHidden": layer._isHidden = value
        case "masksToBounds": layer._masksToBounds = value
        case "doubleSided", "isDoubleSided": layer._isDoubleSided = value
        case "shouldRasterize": layer._shouldRasterize = value
        default: break
        }
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
        // Read from presentation layer so value function animations compose
        // with other transform animations applied earlier in the same frame.
        let baseTransform = layer._transform
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

    // MARK: - byValue Resolution Helpers

    /// Resolves from/to values for a CGFloat animation, considering `byValue`.
    ///
    /// Resolution rules:
    /// - `from` + `to`: use as-is
    /// - `from` + `by`: to = from + by
    /// - `to` + `by`: from = to - by
    /// - `by` only: from = currentValue, to = currentValue + by
    /// - `from` only: to = currentValue
    /// - `to` only: from = currentValue
    private func resolveFromTo(_ animation: CABasicAnimation, currentValue: CGFloat) -> (from: CGFloat, to: CGFloat)? {
        let fromVal = (animation.fromValue as? CGFloat) ?? (animation.fromValue as? Double).map { CGFloat($0) } ?? (animation.fromValue as? Float).map { CGFloat($0) }
        let toVal = (animation.toValue as? CGFloat) ?? (animation.toValue as? Double).map { CGFloat($0) } ?? (animation.toValue as? Float).map { CGFloat($0) }
        let byVal = (animation.byValue as? CGFloat) ?? (animation.byValue as? Double).map { CGFloat($0) } ?? (animation.byValue as? Float).map { CGFloat($0) }

        if let from = fromVal, let to = toVal {
            return (from, to)
        } else if let from = fromVal, let by = byVal {
            return (from, from + by)
        } else if let to = toVal, let by = byVal {
            return (to - by, to)
        } else if let by = byVal {
            return (currentValue, currentValue + by)
        } else if let from = fromVal {
            return (from, currentValue)
        } else if let to = toVal {
            return (currentValue, to)
        }
        return nil
    }

    /// Resolves from/to Float values for animations, considering `byValue`.
    private func resolveFromToFloat(_ animation: CABasicAnimation, currentValue: Float) -> (from: Float, to: Float)? {
        let fromVal = (animation.fromValue as? Float) ?? (animation.fromValue as? Double).map { Float($0) }
        let toVal = (animation.toValue as? Float) ?? (animation.toValue as? Double).map { Float($0) }
        let byVal = (animation.byValue as? Float) ?? (animation.byValue as? Double).map { Float($0) }

        if let from = fromVal, let to = toVal {
            return (from, to)
        } else if let from = fromVal, let by = byVal {
            return (from, from + by)
        } else if let to = toVal, let by = byVal {
            return (to - by, to)
        } else if let by = byVal {
            return (currentValue, currentValue + by)
        } else if let from = fromVal {
            return (from, currentValue)
        } else if let to = toVal {
            return (currentValue, to)
        }
        return nil
    }

    /// Resolves from/to CGPoint values for animations, considering `byValue`.
    private func resolveFromToPoint(_ animation: CABasicAnimation, currentValue: CGPoint) -> (from: CGPoint, to: CGPoint)? {
        let fromVal = animation.fromValue as? CGPoint
        let toVal = animation.toValue as? CGPoint
        let byVal = animation.byValue as? CGPoint

        if let from = fromVal, let to = toVal {
            return (from, to)
        } else if let from = fromVal, let by = byVal {
            return (from, CGPoint(x: from.x + by.x, y: from.y + by.y))
        } else if let to = toVal, let by = byVal {
            return (CGPoint(x: to.x - by.x, y: to.y - by.y), to)
        } else if let by = byVal {
            return (currentValue, CGPoint(x: currentValue.x + by.x, y: currentValue.y + by.y))
        } else if let from = fromVal {
            return (from, currentValue)
        } else if let to = toVal {
            return (currentValue, to)
        }
        return nil
    }

    /// Resolves from/to CGSize values for animations, considering `byValue`.
    private func resolveFromToSize(_ animation: CABasicAnimation, currentValue: CGSize) -> (from: CGSize, to: CGSize)? {
        let fromVal = animation.fromValue as? CGSize
        let toVal = animation.toValue as? CGSize
        let byVal = animation.byValue as? CGSize

        if let from = fromVal, let to = toVal {
            return (from, to)
        } else if let from = fromVal, let by = byVal {
            return (from, CGSize(width: from.width + by.width, height: from.height + by.height))
        } else if let to = toVal, let by = byVal {
            return (CGSize(width: to.width - by.width, height: to.height - by.height), to)
        } else if let by = byVal {
            return (currentValue, CGSize(width: currentValue.width + by.width, height: currentValue.height + by.height))
        } else if let from = fromVal {
            return (from, currentValue)
        } else if let to = toVal {
            return (currentValue, to)
        }
        return nil
    }

    /// Resolves the value contributed by one completed repeat cycle.
    private func cumulativeTerminalValue(
        for animation: CABasicAnimation,
        keyPath: String
    ) -> Any? {
        let currentValue = animation.isAdditive
            ? zeroAnimationValue(matching: animation.fromValue ?? animation.toValue ?? animation.byValue)
            : currentAnimationValue(for: keyPath)

        if animation.autoreverses {
            if let fromValue = animation.fromValue { return fromValue }
            if let toValue = animation.toValue, let byValue = animation.byValue {
                return combineAnimationValues(toValue, byValue, subtract: true)
            }
            return currentValue
        }

        if let toValue = animation.toValue { return toValue }
        if let fromValue = animation.fromValue, let byValue = animation.byValue {
            return combineAnimationValues(fromValue, byValue, subtract: false)
        }
        if let byValue = animation.byValue, let currentValue {
            return combineAnimationValues(currentValue, byValue, subtract: false)
        }
        return currentValue
    }

    private func zeroAnimationValue(matching value: Any?) -> Any? {
        switch value {
        case is Float: return Float(0)
        case is CGFloat, is Double: return CGFloat(0)
        case is CGPoint: return CGPoint.zero
        case is CGSize: return CGSize.zero
        case is CGRect: return CGRect.zero
        case is CGColor: return CGColor(red: 0, green: 0, blue: 0, alpha: 0)
        case is CATransform3D: return CATransform3DIdentity
        case let values as [CGFloat]: return Array(repeating: CGFloat(0), count: values.count)
        default: return nil
        }
    }

    private func combineAnimationValues(_ lhs: Any, _ rhs: Any, subtract: Bool) -> Any? {
        let sign: CGFloat = subtract ? -1 : 1
        if let lhs = lhs as? Float, let rhs = rhs as? Float {
            return lhs + Float(sign) * rhs
        }
        if let lhs = lhs as? CGFloat, let rhs = rhs as? CGFloat {
            return lhs + sign * rhs
        }
        if let lhs = lhs as? CGPoint, let rhs = rhs as? CGPoint {
            return CGPoint(x: lhs.x + sign * rhs.x, y: lhs.y + sign * rhs.y)
        }
        if let lhs = lhs as? CGSize, let rhs = rhs as? CGSize {
            return CGSize(width: lhs.width + sign * rhs.width, height: lhs.height + sign * rhs.height)
        }
        if let lhs = lhs as? CGRect, let rhs = rhs as? CGRect {
            return CGRect(
                x: lhs.origin.x + sign * rhs.origin.x,
                y: lhs.origin.y + sign * rhs.origin.y,
                width: lhs.size.width + sign * rhs.size.width,
                height: lhs.size.height + sign * rhs.size.height
            )
        }
        if let lhs = lhs as? CGColor, let rhs = rhs as? CGColor {
            let (lr, lg, lb, la) = extractRGBA(from: lhs)
            let (rr, rg, rb, ra) = extractRGBA(from: rhs)
            return CGColor(
                red: lr + sign * rr,
                green: lg + sign * rg,
                blue: lb + sign * rb,
                alpha: la + sign * ra
            )
        }
        if let lhs = lhs as? [CGFloat], let rhs = rhs as? [CGFloat], lhs.count == rhs.count {
            return zip(lhs, rhs).map { $0 + sign * $1 }
        }
        if let lhs = lhs as? CATransform3D, let rhs = rhs as? CATransform3D, !subtract {
            return CATransform3DConcat(lhs, rhs)
        }
        return nil
    }

    private func currentAnimationValue(for keyPath: String) -> Any? {
        switch keyPath {
        case "opacity": return _opacity
        case "hidden", "isHidden": return _isHidden
        case "masksToBounds": return _masksToBounds
        case "doubleSided", "isDoubleSided": return _isDoubleSided
        case "shouldRasterize": return _shouldRasterize
        case "bounds": return _bounds
        case "bounds.origin": return _bounds.origin
        case "bounds.size": return _bounds.size
        case "position": return _position
        case "position.x": return _position.x
        case "position.y": return _position.y
        case "zPosition": return _zPosition
        case "anchorPoint": return _anchorPoint
        case "anchorPointZ": return _anchorPointZ
        case "transform": return _transform
        case "sublayerTransform": return _sublayerTransform
        case "backgroundColor": return _backgroundColor
        case "cornerRadius": return _cornerRadius
        case "borderWidth": return _borderWidth
        case "borderColor": return _borderColor
        case "shadowColor": return _shadowColor
        case "shadowOpacity": return _shadowOpacity
        case "shadowOffset": return _shadowOffset
        case "shadowRadius": return _shadowRadius
        case "contentsRect": return _contentsRect
        case "contentsCenter": return contentsCenter
        case "contentsScale": return _contentsScale
        case "rasterizationScale": return _rasterizationScale
        case "strokeStart": return (self as? CAShapeLayer)?._strokeStart
        case "strokeEnd": return (self as? CAShapeLayer)?._strokeEnd
        case "lineWidth": return (self as? CAShapeLayer)?._lineWidth
        case "lineDashPhase": return (self as? CAShapeLayer)?._lineDashPhase
        case "miterLimit": return (self as? CAShapeLayer)?._miterLimit
        case "fillColor": return (self as? CAShapeLayer)?._fillColor
        case "strokeColor": return (self as? CAShapeLayer)?._strokeColor
        case "startPoint": return (self as? CAGradientLayer)?._startPoint
        case "endPoint": return (self as? CAGradientLayer)?._endPoint
        case "locations": return (self as? CAGradientLayer)?._locations
        case "fontSize": return (self as? CATextLayer)?._fontSize
        case "foregroundColor": return (self as? CATextLayer)?._foregroundColor
        case "emitterPosition": return (self as? CAEmitterLayer)?._emitterPosition
        case "emitterSize": return (self as? CAEmitterLayer)?._emitterSize
        case "emitterZPosition": return (self as? CAEmitterLayer)?._emitterZPosition
        case "emitterDepth": return (self as? CAEmitterLayer)?._emitterDepth
        case "birthRate": return (self as? CAEmitterLayer)?._birthRate
        case "lifetime": return (self as? CAEmitterLayer)?._lifetime
        case "velocity": return (self as? CAEmitterLayer)?._velocity
        case "scale": return (self as? CAEmitterLayer)?._scale
        case "spin": return (self as? CAEmitterLayer)?._spin
        case "instanceDelay": return (self as? CAReplicatorLayer)?._instanceDelay
        case "instanceTransform": return (self as? CAReplicatorLayer)?._instanceTransform
        case "instanceColor": return (self as? CAReplicatorLayer)?._instanceColor
        case "instanceRedOffset": return (self as? CAReplicatorLayer)?._instanceRedOffset
        case "instanceGreenOffset": return (self as? CAReplicatorLayer)?._instanceGreenOffset
        case "instanceBlueOffset": return (self as? CAReplicatorLayer)?._instanceBlueOffset
        case "instanceAlphaOffset": return (self as? CAReplicatorLayer)?._instanceAlphaOffset
        default: return nil
        }
    }

    private func applyCumulativeContribution(
        _ contribution: Any,
        cycles: Int,
        to layer: CALayer,
        keyPath: String
    ) {
        let count = CGFloat(cycles)
        let floatCount = Float(cycles)

        switch keyPath {
        case "opacity": if let value = contribution as? Float { layer._opacity += value * floatCount }
        case "position.x": if let value = contribution as? CGFloat { layer._position.x += value * count }
        case "position.y": if let value = contribution as? CGFloat { layer._position.y += value * count }
        case "position": if let value = contribution as? CGPoint {
            layer._position.x += value.x * count
            layer._position.y += value.y * count
        }
        case "bounds": if let value = contribution as? CGRect {
            layer._bounds.origin.x += value.origin.x * count
            layer._bounds.origin.y += value.origin.y * count
            layer._bounds.size.width += value.size.width * count
            layer._bounds.size.height += value.size.height * count
        }
        case "bounds.origin": if let value = contribution as? CGPoint {
            layer._bounds.origin.x += value.x * count
            layer._bounds.origin.y += value.y * count
        }
        case "bounds.size": if let value = contribution as? CGSize {
            layer._bounds.size.width += value.width * count
            layer._bounds.size.height += value.height * count
        }
        case "anchorPoint": if let value = contribution as? CGPoint {
            layer._anchorPoint.x += value.x * count
            layer._anchorPoint.y += value.y * count
        }
        case "shadowOffset": if let value = contribution as? CGSize {
            layer._shadowOffset.width += value.width * count
            layer._shadowOffset.height += value.height * count
        }
        case "cornerRadius", "borderWidth", "shadowRadius", "zPosition", "anchorPointZ", "contentsScale", "rasterizationScale":
            guard let value = contribution as? CGFloat else { return }
            switch keyPath {
            case "cornerRadius": layer._cornerRadius += value * count
            case "borderWidth": layer._borderWidth += value * count
            case "shadowRadius": layer._shadowRadius += value * count
            case "zPosition": layer._zPosition += value * count
            case "anchorPointZ": layer._anchorPointZ += value * count
            case "contentsScale": layer._contentsScale += value * count
            default: layer._rasterizationScale += value * count
            }
        case "shadowOpacity": if let value = contribution as? Float { layer._shadowOpacity += value * floatCount }
        case "transform": if let value = contribution as? CATransform3D {
            for _ in 0..<cycles { layer._transform = CATransform3DConcat(layer._transform, value) }
        }
        case "sublayerTransform": if let value = contribution as? CATransform3D {
            for _ in 0..<cycles { layer._sublayerTransform = CATransform3DConcat(layer._sublayerTransform, value) }
        }
        case "backgroundColor": addCumulativeColor(contribution, cycles: count, to: &layer._backgroundColor)
        case "borderColor": addCumulativeColor(contribution, cycles: count, to: &layer._borderColor)
        case "shadowColor": addCumulativeColor(contribution, cycles: count, to: &layer._shadowColor)
        default:
            applySpecializedCumulativeContribution(contribution, cycles: cycles, to: layer, keyPath: keyPath)
        }
    }

    private func addCumulativeColor(_ contribution: Any, cycles: CGFloat, to color: inout CGColor?) {
        guard let contribution = contribution as? CGColor else { return }
        let (r, g, b, a) = extractRGBA(from: contribution)
        let (br, bg, bb, ba) = color.map(extractRGBA) ?? (0, 0, 0, 0)
        color = CGColor(red: br + r * cycles, green: bg + g * cycles, blue: bb + b * cycles, alpha: ba + a * cycles)
    }

    private func applySpecializedCumulativeContribution(
        _ contribution: Any,
        cycles: Int,
        to layer: CALayer,
        keyPath: String
    ) {
        let count = CGFloat(cycles)
        let floatCount = Float(cycles)
        switch keyPath {
        case "strokeStart", "strokeEnd", "lineWidth", "lineDashPhase", "miterLimit":
            guard let shape = layer as? CAShapeLayer, let value = contribution as? CGFloat else { return }
            switch keyPath {
            case "strokeStart": shape._strokeStart += value * count
            case "strokeEnd": shape._strokeEnd += value * count
            case "lineWidth": shape._lineWidth += value * count
            case "lineDashPhase": shape._lineDashPhase += value * count
            default: shape._miterLimit += value * count
            }
        case "fillColor": if let shape = layer as? CAShapeLayer { addCumulativeColor(contribution, cycles: count, to: &shape._fillColor) }
        case "strokeColor": if let shape = layer as? CAShapeLayer { addCumulativeColor(contribution, cycles: count, to: &shape._strokeColor) }
        case "fontSize": if let text = layer as? CATextLayer, let value = contribution as? CGFloat { text._fontSize += value * count }
        case "foregroundColor": if let text = layer as? CATextLayer { addCumulativeColor(contribution, cycles: count, to: &text._foregroundColor) }
        case "startPoint", "endPoint":
            guard let gradient = layer as? CAGradientLayer, let value = contribution as? CGPoint else { return }
            if keyPath == "startPoint" {
                gradient._startPoint.x += value.x * count; gradient._startPoint.y += value.y * count
            } else {
                gradient._endPoint.x += value.x * count; gradient._endPoint.y += value.y * count
            }
        case "locations":
            guard let gradient = layer as? CAGradientLayer,
                  let value = contribution as? [CGFloat],
                  let locations = gradient._locations,
                  locations.count == value.count else { return }
            gradient._locations = zip(locations, value).map { $0 + $1 * count }
        case "emitterPosition": if let emitter = layer as? CAEmitterLayer, let value = contribution as? CGPoint {
            emitter._emitterPosition.x += value.x * count; emitter._emitterPosition.y += value.y * count
        }
        case "emitterSize": if let emitter = layer as? CAEmitterLayer, let value = contribution as? CGSize {
            emitter._emitterSize.width += value.width * count; emitter._emitterSize.height += value.height * count
        }
        case "emitterZPosition", "emitterDepth":
            guard let emitter = layer as? CAEmitterLayer, let value = contribution as? CGFloat else { return }
            if keyPath == "emitterZPosition" { emitter._emitterZPosition += value * count }
            else { emitter._emitterDepth += value * count }
        case "birthRate", "lifetime", "velocity", "scale", "spin":
            guard let emitter = layer as? CAEmitterLayer, let value = contribution as? Float else { return }
            switch keyPath {
            case "birthRate": emitter._birthRate += value * floatCount
            case "lifetime": emitter._lifetime += value * floatCount
            case "velocity": emitter._velocity += value * floatCount
            case "scale": emitter._scale += value * floatCount
            default: emitter._spin += value * floatCount
            }
        case "instanceDelay": if let replicator = layer as? CAReplicatorLayer, let value = contribution as? CGFloat { replicator._instanceDelay += value * count }
        case "instanceTransform": if let replicator = layer as? CAReplicatorLayer, let value = contribution as? CATransform3D {
            for _ in 0..<cycles { replicator._instanceTransform = CATransform3DConcat(replicator._instanceTransform, value) }
        }
        case "instanceColor": if let replicator = layer as? CAReplicatorLayer { addCumulativeColor(contribution, cycles: count, to: &replicator._instanceColor) }
        case "instanceRedOffset", "instanceGreenOffset", "instanceBlueOffset", "instanceAlphaOffset":
            guard let replicator = layer as? CAReplicatorLayer, let value = contribution as? Float else { return }
            switch keyPath {
            case "instanceRedOffset": replicator._instanceRedOffset += value * floatCount
            case "instanceGreenOffset": replicator._instanceGreenOffset += value * floatCount
            case "instanceBlueOffset": replicator._instanceBlueOffset += value * floatCount
            default: replicator._instanceAlphaOffset += value * floatCount
            }
        default: break
        }
    }

    private func applyFloatAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        // When isAdditive, pass zero as currentValue to resolveFromTo* so that
        // implicit from/to values don't include the model value (which would be double-counted).
        let additive = animation.isAdditive
        switch keyPath {
        case "opacity":
            guard let resolved = resolveFromToFloat(animation, currentValue: additive ? 0 : _opacity) else { return }
            let value = resolved.from + Float(progress) * (resolved.to - resolved.from)
            layer._opacity = additive ? layer._opacity + value : value
        case "cornerRadius":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : _cornerRadius) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer._cornerRadius = additive ? layer._cornerRadius + value : value
        case "borderWidth":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : _borderWidth) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer._borderWidth = additive ? layer._borderWidth + value : value
        case "shadowRadius":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : _shadowRadius) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer._shadowRadius = additive ? layer._shadowRadius + value : value
        case "shadowOpacity":
            guard let resolved = resolveFromToFloat(animation, currentValue: additive ? 0 : _shadowOpacity) else { return }
            let value = resolved.from + Float(progress) * (resolved.to - resolved.from)
            layer._shadowOpacity = additive ? layer._shadowOpacity + value : value
        case "zPosition":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : _zPosition) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer._zPosition = additive ? layer._zPosition + value : value
        case "anchorPointZ":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : _anchorPointZ) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer._anchorPointZ = additive ? layer._anchorPointZ + value : value
        case "contentsScale":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : _contentsScale) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer._contentsScale = additive ? layer._contentsScale + value : value
        case "rasterizationScale":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : rasterizationScale) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer.rasterizationScale = additive ? layer.rasterizationScale + value : value

        // CAShapeLayer properties
        case "strokeStart":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : modelShapeLayer._strokeStart) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            shapeLayer._strokeStart = additive ? shapeLayer._strokeStart + value : value
        case "strokeEnd":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : modelShapeLayer._strokeEnd) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            shapeLayer._strokeEnd = additive ? shapeLayer._strokeEnd + value : value
        case "lineWidth":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : modelShapeLayer._lineWidth) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            shapeLayer._lineWidth = additive ? shapeLayer._lineWidth + value : value
        case "lineDashPhase":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : modelShapeLayer._lineDashPhase) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            shapeLayer._lineDashPhase = additive ? shapeLayer._lineDashPhase + value : value
        case "miterLimit":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : modelShapeLayer._miterLimit) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            shapeLayer._miterLimit = additive ? shapeLayer._miterLimit + value : value

        case "fontSize":
            guard let textLayer = layer as? CATextLayer,
                  let modelTextLayer = self as? CATextLayer,
                  let resolved = resolveFromTo(animation, currentValue: additive ? 0 : modelTextLayer._fontSize) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            textLayer._fontSize = additive ? textLayer._fontSize + value : value

        case "emitterZPosition", "emitterDepth":
            guard let emitterLayer = layer as? CAEmitterLayer,
                  let modelEmitterLayer = self as? CAEmitterLayer else { return }
            let current = keyPath == "emitterZPosition" ? modelEmitterLayer._emitterZPosition : modelEmitterLayer._emitterDepth
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : current) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            if keyPath == "emitterZPosition" {
                emitterLayer._emitterZPosition = additive ? emitterLayer._emitterZPosition + value : value
            } else {
                emitterLayer._emitterDepth = additive ? emitterLayer._emitterDepth + value : value
            }

        case "birthRate", "lifetime", "velocity", "scale", "spin":
            guard let emitterLayer = layer as? CAEmitterLayer,
                  let modelEmitterLayer = self as? CAEmitterLayer else { return }
            let current: Float
            switch keyPath {
            case "birthRate": current = modelEmitterLayer._birthRate
            case "lifetime": current = modelEmitterLayer._lifetime
            case "velocity": current = modelEmitterLayer._velocity
            case "scale": current = modelEmitterLayer._scale
            default: current = modelEmitterLayer._spin
            }
            guard let resolved = resolveFromToFloat(animation, currentValue: additive ? 0 : current) else { return }
            let value = resolved.from + Float(progress) * (resolved.to - resolved.from)
            switch keyPath {
            case "birthRate": emitterLayer._birthRate = additive ? emitterLayer._birthRate + value : value
            case "lifetime": emitterLayer._lifetime = additive ? emitterLayer._lifetime + value : value
            case "velocity": emitterLayer._velocity = additive ? emitterLayer._velocity + value : value
            case "scale": emitterLayer._scale = additive ? emitterLayer._scale + value : value
            default: emitterLayer._spin = additive ? emitterLayer._spin + value : value
            }

        case "instanceDelay":
            guard let replicatorLayer = layer as? CAReplicatorLayer,
                  let modelReplicatorLayer = self as? CAReplicatorLayer,
                  let resolved = resolveFromTo(animation, currentValue: additive ? 0 : CGFloat(modelReplicatorLayer._instanceDelay)) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            replicatorLayer._instanceDelay = additive ? replicatorLayer._instanceDelay + CFTimeInterval(value) : CFTimeInterval(value)

        case "instanceRedOffset", "instanceGreenOffset", "instanceBlueOffset", "instanceAlphaOffset":
            guard let replicatorLayer = layer as? CAReplicatorLayer,
                  let modelReplicatorLayer = self as? CAReplicatorLayer else { return }
            let current: Float
            switch keyPath {
            case "instanceRedOffset": current = modelReplicatorLayer._instanceRedOffset
            case "instanceGreenOffset": current = modelReplicatorLayer._instanceGreenOffset
            case "instanceBlueOffset": current = modelReplicatorLayer._instanceBlueOffset
            default: current = modelReplicatorLayer._instanceAlphaOffset
            }
            guard let resolved = resolveFromToFloat(animation, currentValue: additive ? 0 : current) else { return }
            let value = resolved.from + Float(progress) * (resolved.to - resolved.from)
            switch keyPath {
            case "instanceRedOffset": replicatorLayer._instanceRedOffset = additive ? replicatorLayer._instanceRedOffset + value : value
            case "instanceGreenOffset": replicatorLayer._instanceGreenOffset = additive ? replicatorLayer._instanceGreenOffset + value : value
            case "instanceBlueOffset": replicatorLayer._instanceBlueOffset = additive ? replicatorLayer._instanceBlueOffset + value : value
            default: replicatorLayer._instanceAlphaOffset = additive ? replicatorLayer._instanceAlphaOffset + value : value
            }

        default:
            break
        }
    }

    private func applyPointAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        let additive = animation.isAdditive
        switch keyPath {
        case "position":
            guard let resolved = resolveFromToPoint(animation, currentValue: additive ? .zero : _position) else { return }
            let value = CGPoint(
                x: resolved.from.x + CGFloat(progress) * (resolved.to.x - resolved.from.x),
                y: resolved.from.y + CGFloat(progress) * (resolved.to.y - resolved.from.y)
            )
            if additive {
                layer._position.x += value.x
                layer._position.y += value.y
            } else {
                layer._position = value
            }
        case "position.x":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : _position.x) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer._position.x = additive ? layer._position.x + value : value
        case "position.y":
            guard let resolved = resolveFromTo(animation, currentValue: additive ? 0 : _position.y) else { return }
            let value = resolved.from + CGFloat(progress) * (resolved.to - resolved.from)
            layer._position.y = additive ? layer._position.y + value : value
        case "anchorPoint":
            guard let resolved = resolveFromToPoint(animation, currentValue: additive ? .zero : _anchorPoint) else { return }
            let value = CGPoint(
                x: resolved.from.x + CGFloat(progress) * (resolved.to.x - resolved.from.x),
                y: resolved.from.y + CGFloat(progress) * (resolved.to.y - resolved.from.y)
            )
            if additive {
                layer._anchorPoint.x += value.x
                layer._anchorPoint.y += value.y
            } else {
                layer._anchorPoint = value
            }
        case "bounds.origin":
            guard let resolved = resolveFromToPoint(animation, currentValue: additive ? .zero : _bounds.origin) else { return }
            let value = CGPoint(
                x: resolved.from.x + CGFloat(progress) * (resolved.to.x - resolved.from.x),
                y: resolved.from.y + CGFloat(progress) * (resolved.to.y - resolved.from.y)
            )
            if additive {
                layer._bounds.origin.x += value.x
                layer._bounds.origin.y += value.y
            } else {
                layer._bounds.origin = value
            }
        case "shadowOffset":
            guard let resolved = resolveFromToSize(animation, currentValue: additive ? .zero : _shadowOffset) else { return }
            let value = CGSize(
                width: resolved.from.width + CGFloat(progress) * (resolved.to.width - resolved.from.width),
                height: resolved.from.height + CGFloat(progress) * (resolved.to.height - resolved.from.height)
            )
            if additive {
                layer._shadowOffset.width += value.width
                layer._shadowOffset.height += value.height
            } else {
                layer._shadowOffset = value
            }

        // CAGradientLayer properties
        case "startPoint":
            guard let gradientLayer = layer as? CAGradientLayer,
                  let modelGradientLayer = self as? CAGradientLayer else { return }
            guard let resolved = resolveFromToPoint(animation, currentValue: additive ? .zero : modelGradientLayer._startPoint) else { return }
            let value = CGPoint(
                x: resolved.from.x + CGFloat(progress) * (resolved.to.x - resolved.from.x),
                y: resolved.from.y + CGFloat(progress) * (resolved.to.y - resolved.from.y)
            )
            if additive {
                gradientLayer._startPoint.x += value.x
                gradientLayer._startPoint.y += value.y
            } else {
                gradientLayer._startPoint = value
            }
        case "endPoint":
            guard let gradientLayer = layer as? CAGradientLayer,
                  let modelGradientLayer = self as? CAGradientLayer else { return }
            guard let resolved = resolveFromToPoint(animation, currentValue: additive ? .zero : modelGradientLayer._endPoint) else { return }
            let value = CGPoint(
                x: resolved.from.x + CGFloat(progress) * (resolved.to.x - resolved.from.x),
                y: resolved.from.y + CGFloat(progress) * (resolved.to.y - resolved.from.y)
            )
            if additive {
                gradientLayer._endPoint.x += value.x
                gradientLayer._endPoint.y += value.y
            } else {
                gradientLayer._endPoint = value
            }

        case "emitterPosition":
            guard let emitterLayer = layer as? CAEmitterLayer,
                  let modelEmitterLayer = self as? CAEmitterLayer,
                  let resolved = resolveFromToPoint(animation, currentValue: additive ? .zero : modelEmitterLayer._emitterPosition) else { return }
            let value = CGPoint(
                x: resolved.from.x + CGFloat(progress) * (resolved.to.x - resolved.from.x),
                y: resolved.from.y + CGFloat(progress) * (resolved.to.y - resolved.from.y)
            )
            emitterLayer._emitterPosition = additive
                ? CGPoint(x: emitterLayer._emitterPosition.x + value.x, y: emitterLayer._emitterPosition.y + value.y)
                : value

        case "emitterSize":
            guard let emitterLayer = layer as? CAEmitterLayer,
                  let modelEmitterLayer = self as? CAEmitterLayer,
                  let resolved = resolveFromToSize(animation, currentValue: additive ? .zero : modelEmitterLayer._emitterSize) else { return }
            let value = CGSize(
                width: resolved.from.width + CGFloat(progress) * (resolved.to.width - resolved.from.width),
                height: resolved.from.height + CGFloat(progress) * (resolved.to.height - resolved.from.height)
            )
            emitterLayer._emitterSize = additive
                ? CGSize(width: emitterLayer._emitterSize.width + value.width, height: emitterLayer._emitterSize.height + value.height)
                : value

        default:
            break
        }
    }

    private func applyRectAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        let additive = animation.isAdditive
        switch keyPath {
        case "bounds":
            let fromVal = animation.fromValue as? CGRect
            let toVal = animation.toValue as? CGRect
            let fallback = additive ? CGRect.zero : _bounds
            let from = fromVal ?? fallback
            let to = toVal ?? fallback
            let value = CGRect(
                x: from.origin.x + CGFloat(progress) * (to.origin.x - from.origin.x),
                y: from.origin.y + CGFloat(progress) * (to.origin.y - from.origin.y),
                width: from.size.width + CGFloat(progress) * (to.size.width - from.size.width),
                height: from.size.height + CGFloat(progress) * (to.size.height - from.size.height)
            )
            if additive {
                layer._bounds.origin.x += value.origin.x
                layer._bounds.origin.y += value.origin.y
                layer._bounds.size.width += value.size.width
                layer._bounds.size.height += value.size.height
            } else {
                layer._bounds = value
            }
        case "bounds.size":
            guard let resolved = resolveFromToSize(animation, currentValue: additive ? .zero : _bounds.size) else { return }
            let value = CGSize(
                width: resolved.from.width + CGFloat(progress) * (resolved.to.width - resolved.from.width),
                height: resolved.from.height + CGFloat(progress) * (resolved.to.height - resolved.from.height)
            )
            if additive {
                layer._bounds.size.width += value.width
                layer._bounds.size.height += value.height
            } else {
                layer._bounds.size = value
            }
        case "contentsRect":
            let fromVal = animation.fromValue as? CGRect
            let toVal = animation.toValue as? CGRect
            let crFallback = additive ? CGRect.zero : contentsRect
            let from = fromVal ?? crFallback
            let to = toVal ?? crFallback
            let value = CGRect(
                x: from.origin.x + CGFloat(progress) * (to.origin.x - from.origin.x),
                y: from.origin.y + CGFloat(progress) * (to.origin.y - from.origin.y),
                width: from.size.width + CGFloat(progress) * (to.size.width - from.size.width),
                height: from.size.height + CGFloat(progress) * (to.size.height - from.size.height)
            )
            if additive {
                let cur = layer.contentsRect
                layer.contentsRect = CGRect(
                    x: cur.origin.x + value.origin.x,
                    y: cur.origin.y + value.origin.y,
                    width: cur.size.width + value.size.width,
                    height: cur.size.height + value.size.height
                )
            } else {
                layer.contentsRect = value
            }
        case "contentsCenter":
            let fromVal = animation.fromValue as? CGRect
            let toVal = animation.toValue as? CGRect
            let ccFallback = additive ? CGRect.zero : contentsCenter
            let from = fromVal ?? ccFallback
            let to = toVal ?? ccFallback
            let value = CGRect(
                x: from.origin.x + CGFloat(progress) * (to.origin.x - from.origin.x),
                y: from.origin.y + CGFloat(progress) * (to.origin.y - from.origin.y),
                width: from.size.width + CGFloat(progress) * (to.size.width - from.size.width),
                height: from.size.height + CGFloat(progress) * (to.size.height - from.size.height)
            )
            if additive {
                let cur = layer.contentsCenter
                layer.contentsCenter = CGRect(
                    x: cur.origin.x + value.origin.x,
                    y: cur.origin.y + value.origin.y,
                    width: cur.size.width + value.size.width,
                    height: cur.size.height + value.size.height
                )
            } else {
                layer.contentsCenter = value
            }
        default:
            break
        }
    }

    private func applyTransformAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        // Read from the presentation layer's accumulated transform so that
        // multiple transform.* animations compose (e.g., scale + rotation).
        let baseTransform = layer._transform

        switch keyPath {
        case "transform":
            // Full transform animation: interpolate between from and to values
            guard let from = (animation.fromValue as? CATransform3D) ?? _transform as CATransform3D?,
                  let to = (animation.toValue as? CATransform3D) ?? _transform as CATransform3D? else { return }
            let value = interpolateTransform(from: from, to: to, progress: CGFloat(progress))
            if animation.isAdditive {
                layer._transform = CATransform3DConcat(value, baseTransform)
            } else {
                layer._transform = value
            }

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

        case "sublayerTransform":
            // Full sublayerTransform animation: interpolate between from and to values
            guard let from = (animation.fromValue as? CATransform3D) ?? _sublayerTransform as CATransform3D?,
                  let to = (animation.toValue as? CATransform3D) ?? _sublayerTransform as CATransform3D? else { return }
            let value = interpolateTransform(from: from, to: to, progress: CGFloat(progress))
            if animation.isAdditive {
                layer._sublayerTransform = CATransform3DConcat(value, layer._sublayerTransform)
            } else {
                layer._sublayerTransform = value
            }

        case "instanceTransform":
            guard let replicatorLayer = layer as? CAReplicatorLayer,
                  let modelReplicatorLayer = self as? CAReplicatorLayer else { return }
            let from = (animation.fromValue as? CATransform3D) ?? modelReplicatorLayer._instanceTransform
            let to = (animation.toValue as? CATransform3D) ?? modelReplicatorLayer._instanceTransform
            let value = interpolateTransform(from: from, to: to, progress: CGFloat(progress))
            replicatorLayer._instanceTransform = animation.isAdditive
                ? CATransform3DConcat(value, replicatorLayer._instanceTransform)
                : value

        default:
            break
        }
    }

    /// Interpolates between two transforms by decomposing each into translation,
    /// rotation (as a quaternion), scale, skew, and perspective components,
    /// interpolating each independently, and recomposing.
    ///
    /// Naive element-wise interpolation of the 16 matrix entries produces
    /// invalid intermediate matrices whenever a rotation is involved. The
    /// actual work lives in `CATransform3DInterpolation`.
    private func interpolateTransform(from: CATransform3D, to: CATransform3D, progress: CGFloat) -> CATransform3D {
        return CATransform3DInterpolation.interpolate(from: from, to: to, progress: progress)
    }

    private func applyColorAnimation(_ animation: CABasicAnimation, to layer: CALayer, keyPath: String, progress: CFTimeInterval) {
        let additive = animation.isAdditive
        switch keyPath {
        case "backgroundColor":
            guard let from = extractColor(animation.fromValue) ?? _backgroundColor,
                  let to = extractColor(animation.toValue) ?? _backgroundColor else { return }
            let value = interpolateColor(from: from, to: to, progress: CGFloat(progress))
            layer._backgroundColor = additive ? addColor(value, to: layer._backgroundColor) : value
        case "borderColor":
            guard let from = extractColor(animation.fromValue) ?? _borderColor,
                  let to = extractColor(animation.toValue) ?? _borderColor else { return }
            let value = interpolateColor(from: from, to: to, progress: CGFloat(progress))
            layer._borderColor = additive ? addColor(value, to: layer._borderColor) : value
        case "shadowColor":
            guard let from = extractColor(animation.fromValue) ?? _shadowColor,
                  let to = extractColor(animation.toValue) ?? _shadowColor else { return }
            let value = interpolateColor(from: from, to: to, progress: CGFloat(progress))
            layer._shadowColor = additive ? addColor(value, to: layer._shadowColor) : value

        // CAShapeLayer color properties
        case "fillColor":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let from = extractColor(animation.fromValue) ?? modelShapeLayer._fillColor,
                  let to = extractColor(animation.toValue) ?? modelShapeLayer._fillColor else { return }
            let value = interpolateColor(from: from, to: to, progress: CGFloat(progress))
            shapeLayer._fillColor = additive ? addColor(value, to: shapeLayer._fillColor) : value
        case "strokeColor":
            guard let shapeLayer = layer as? CAShapeLayer,
                  let modelShapeLayer = self as? CAShapeLayer else { return }
            guard let from = extractColor(animation.fromValue) ?? modelShapeLayer._strokeColor,
                  let to = extractColor(animation.toValue) ?? modelShapeLayer._strokeColor else { return }
            let value = interpolateColor(from: from, to: to, progress: CGFloat(progress))
            shapeLayer._strokeColor = additive ? addColor(value, to: shapeLayer._strokeColor) : value

        case "foregroundColor":
            guard let textLayer = layer as? CATextLayer,
                  let modelTextLayer = self as? CATextLayer,
                  let from = extractColor(animation.fromValue) ?? modelTextLayer._foregroundColor,
                  let to = extractColor(animation.toValue) ?? modelTextLayer._foregroundColor else { return }
            let value = interpolateColor(from: from, to: to, progress: CGFloat(progress))
            textLayer._foregroundColor = additive ? addColor(value, to: textLayer._foregroundColor) : value

        case "instanceColor":
            guard let replicatorLayer = layer as? CAReplicatorLayer,
                  let modelReplicatorLayer = self as? CAReplicatorLayer,
                  let from = extractColor(animation.fromValue) ?? modelReplicatorLayer._instanceColor,
                  let to = extractColor(animation.toValue) ?? modelReplicatorLayer._instanceColor else { return }
            let value = interpolateColor(from: from, to: to, progress: CGFloat(progress))
            replicatorLayer._instanceColor = additive ? addColor(value, to: replicatorLayer._instanceColor) : value

        default:
            break
        }
    }

    /// Safely extracts a CGColor from an Any value.
    private func extractColor(_ value: Any?) -> CGColor? {
        return value as? CGColor
    }

    /// Adds RGBA components of two colors (for additive animation).
    private func addColor(_ value: CGColor, to base: CGColor?) -> CGColor {
        guard let base = base else { return value }
        let (vR, vG, vB, vA) = extractRGBA(from: value)
        let (bR, bG, bB, bA) = extractRGBA(from: base)
        return CGColor(red: bR + vR, green: bG + vG, blue: bB + vB, alpha: bA + vA)
    }

    private struct PathElementSample {
        let type: CGPathElementType
        let points: [CGPoint]
    }

    private func applyPathAnimation(
        _ animation: CABasicAnimation,
        to layer: CALayer,
        keyPath: String,
        progress: CFTimeInterval
    ) {
        let modelPath = keyPath == "path" ? (self as? CAShapeLayer)?._path : _shadowPath
        guard let from = extractPath(animation.fromValue) ?? modelPath,
              let to = extractPath(animation.toValue) ?? modelPath else { return }

        let value = interpolatePath(from: from, to: to, progress: CGFloat(progress))
        if keyPath == "path" {
            (layer as? CAShapeLayer)?._path = value
        } else {
            layer._shadowPath = value
        }
    }

    /// Interpolates paths with matching element topology. Incompatible paths
    /// change discretely because their control points cannot be paired.
    private func interpolatePath(from: CGPath, to: CGPath, progress: CGFloat) -> CGPath {
        let fromElements = pathElements(from)
        let toElements = pathElements(to)
        guard fromElements.count == toElements.count else {
            return progress < 1 ? from : to
        }

        let output = CGMutablePath()
        for index in fromElements.indices {
            let lhs = fromElements[index]
            let rhs = toElements[index]
            guard lhs.type == rhs.type, lhs.points.count == rhs.points.count else {
                return progress < 1 ? from : to
            }
            let points = zip(lhs.points, rhs.points).map { start, end in
                CGPoint(
                    x: start.x + progress * (end.x - start.x),
                    y: start.y + progress * (end.y - start.y)
                )
            }
            switch lhs.type {
            case .moveToPoint:
                output.move(to: points[0])
            case .addLineToPoint:
                output.addLine(to: points[0])
            case .addQuadCurveToPoint:
                output.addQuadCurve(to: points[1], control: points[0])
            case .addCurveToPoint:
                output.addCurve(to: points[2], control1: points[0], control2: points[1])
            case .closeSubpath:
                output.closeSubpath()
            @unknown default:
                return progress < 1 ? from : to
            }
        }
        return output
    }

    private func pathElements(_ path: CGPath) -> [PathElementSample] {
        var result: [PathElementSample] = []
        path.applyWithBlock { elementPointer in
            let element = elementPointer.pointee
            let pointCount: Int
            switch element.type {
            case .moveToPoint, .addLineToPoint:
                pointCount = 1
            case .addQuadCurveToPoint:
                pointCount = 2
            case .addCurveToPoint:
                pointCount = 3
            case .closeSubpath:
                pointCount = 0
            @unknown default:
                pointCount = 0
            }
            let points: [CGPoint]
            if pointCount > 0, let elementPoints = element.points {
                points = (0..<pointCount).map { elementPoints[$0] }
            } else {
                points = []
            }
            result.append(PathElementSample(type: element.type, points: points))
        }
        return result
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
            guard let fromLocations = (animation.fromValue as? [CGFloat]) ?? modelGradientLayer._locations,
                  let toLocations = (animation.toValue as? [CGFloat]) ?? modelGradientLayer._locations else { return }

            // Interpolate each location in the array
            let count = min(fromLocations.count, toLocations.count)
            var interpolatedLocations: [CGFloat] = []
            for i in 0..<count {
                let from = fromLocations[i]
                let to = toLocations[i]
                interpolatedLocations.append(from + CGFloat(progress) * (to - from))
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

        guard let values = animation.values, !values.isEmpty else { return }
        if values.count == 1 {
            applyKeyframeValue(values[0], layer: layer, keyPath: keyPath)
            return
        }

        // For paced modes, remap progress based on arc length
        let effectiveProgress: CFTimeInterval
        var effectiveKeyTimes: [CGFloat]

        switch animation.calculationMode {
        case .paced, .cubicPaced:
            // Calculate arc-length parameterized progress
            let (remappedProgress, pacedKeyTimes) = calculatePacedProgress(
                progress: CGFloat(progress),
                values: values,
                animation: animation
            )
            effectiveProgress = CFTimeInterval(remappedProgress)
            effectiveKeyTimes = pacedKeyTimes
        default:
            effectiveProgress = progress
            effectiveKeyTimes = animation.keyTimes ?? defaultKeyTimes(for: values.count)
        }

        guard effectiveKeyTimes.count == values.count,
              effectiveKeyTimes.allSatisfy(\.isFinite),
              zip(effectiveKeyTimes, effectiveKeyTimes.dropFirst()).allSatisfy({ $0 <= $1 }) else {
            return
        }

        let clampedProgress = CGFloat(effectiveProgress)
        if clampedProgress <= effectiveKeyTimes[0] {
            applyKeyframeValue(values[0], layer: layer, keyPath: keyPath)
            return
        }
        if clampedProgress >= effectiveKeyTimes[values.count - 1] {
            applyKeyframeValue(values[values.count - 1], layer: layer, keyPath: keyPath)
            return
        }
        if animation.calculationMode == .discrete {
            let valueIndex = effectiveKeyTimes.lastIndex(where: { $0 <= clampedProgress }) ?? 0
            applyKeyframeValue(values[valueIndex], layer: layer, keyPath: keyPath)
            return
        }

        // Determine which keyframe segment we're in
        let segmentIndex = findSegmentIndex(for: clampedProgress, in: effectiveKeyTimes)
        let startIndex = segmentIndex
        let endIndex = min(segmentIndex + 1, values.count - 1)

        // Get local progress within the segment
        let startTime = effectiveKeyTimes[startIndex]
        let endTime = effectiveKeyTimes[endIndex]
        let segmentProgress: Float
        if endTime > startTime {
            segmentProgress = Float((CGFloat(effectiveProgress) - startTime) / (endTime - startTime))
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

            interpolateCubicKeyframeValue(
                p0: p0,
                p1: p1,
                p2: p2,
                p3: p3,
                t: CGFloat(adjustedProgress),
                startParameters: cubicParameters(for: animation, at: startIndex),
                endParameters: cubicParameters(for: animation, at: endIndex),
                layer: layer,
                keyPath: keyPath
            )
        default:
            interpolateKeyframeValue(from: fromValue, to: toValue, progress: CFTimeInterval(adjustedProgress), layer: layer, keyPath: keyPath)
        }
    }

    /// Calculates arc-length parameterized progress for paced animation modes.
    ///
    /// Returns the remapped progress and the paced key times array.
    private func calculatePacedProgress(
        progress: CGFloat,
        values: [Any],
        animation: CAKeyframeAnimation
    ) -> (CGFloat, [CGFloat]) {
        guard values.count > 1 else { return (progress, [0]) }

        // Calculate distances between consecutive keyframes
        var distances: [CGFloat] = []
        for i in 0..<(values.count - 1) {
            let distance: CGFloat
            if animation.calculationMode == .cubicPaced {
                // For cubic paced, estimate the Kochanek-Bartels segment length.
                let p0Index = max(0, i - 1)
                let p3Index = min(values.count - 1, i + 2)
                distance = estimateCubicArcLength(
                    p0: values[p0Index],
                    p1: values[i],
                    p2: values[i + 1],
                    p3: values[p3Index],
                    startParameters: cubicParameters(for: animation, at: i),
                    endParameters: cubicParameters(for: animation, at: i + 1)
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
        var pacedKeyTimes: [CGFloat] = []
        for cumDist in cumulativeDistances {
            pacedKeyTimes.append(cumDist / totalDistance)
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

        if let fromBool = from as? Bool, let toBool = to as? Bool {
            return fromBool == toBool ? 0 : 1
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

    /// Estimates the arc length of a Kochanek-Bartels spline segment using numerical integration.
    private func estimateCubicArcLength(
        p0: Any,
        p1: Any,
        p2: Any,
        p3: Any,
        startParameters: CubicParameters,
        endParameters: CubicParameters
    ) -> CGFloat {
        // Sample the curve at multiple points and sum the chord lengths.
        let numSamples = 10
        var arcLength: CGFloat = 0

        guard let prev = interpolateForDistance(
            p0: p0,
            p1: p1,
            p2: p2,
            p3: p3,
            t: 0,
            startParameters: startParameters,
            endParameters: endParameters
        ) else {
            // Fallback to linear distance
            return calculateDistance(from: p1, to: p2)
        }

        var previousPoint = prev

        for i in 1...numSamples {
            let t = CGFloat(i) / CGFloat(numSamples)
            guard let currentPoint = interpolateForDistance(
                p0: p0,
                p1: p1,
                p2: p2,
                p3: p3,
                t: t,
                startParameters: startParameters,
                endParameters: endParameters
            ) else {
                return calculateDistance(from: p1, to: p2)
            }
            arcLength += calculatePointDistance(from: previousPoint, to: currentPoint)
            previousPoint = currentPoint
        }

        return arcLength
    }

    /// Interpolates a point on the Kochanek-Bartels spline for distance calculation.
    private func interpolateForDistance(
        p0: Any,
        p1: Any,
        p2: Any,
        p3: Any,
        t: CGFloat,
        startParameters: CubicParameters,
        endParameters: CubicParameters
    ) -> (x: CGFloat, y: CGFloat)? {
        if let v0 = p0 as? CGPoint, let v1 = p1 as? CGPoint, let v2 = p2 as? CGPoint, let v3 = p3 as? CGPoint {
            let x = cubicInterpolate(
                v0.x, v1.x, v2.x, v3.x, t,
                startParameters: startParameters,
                endParameters: endParameters
            )
            let y = cubicInterpolate(
                v0.y, v1.y, v2.y, v3.y, t,
                startParameters: startParameters,
                endParameters: endParameters
            )
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
    private func defaultKeyTimes(for count: Int) -> [CGFloat] {
        guard count > 1 else { return [0] }
        return (0..<count).map { CGFloat($0) / CGFloat(count - 1) }
    }

    /// Finds the segment index for a given progress value.
    private func findSegmentIndex(for progress: CGFloat, in keyTimes: [CGFloat]) -> Int {
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
        case "position.x":
            if let v = value as? CGFloat { layer._position.x = v }
        case "position.y":
            if let v = value as? CGFloat { layer._position.y = v }
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
        case "hidden", "isHidden", "masksToBounds", "doubleSided", "isDoubleSided", "shouldRasterize":
            if let v = value as? Bool {
                applyBooleanAnimationValue(v, to: layer, keyPath: keyPath)
            }

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

        // Additional CGFloat properties
        case "anchorPointZ":
            if let v = value as? CGFloat { layer._anchorPointZ = v }
        case "contentsScale":
            if let v = value as? CGFloat { layer._contentsScale = v }
        case "rasterizationScale":
            if let v = value as? CGFloat { layer.rasterizationScale = v }

        // Additional Float properties
        case "shadowOpacity":
            if let v = value as? Float { layer._shadowOpacity = v }

        // Additional CGFloat properties (via shadowRadius already handled above)
        case "shadowRadius":
            if let v = value as? CGFloat { layer._shadowRadius = v }

        // CGPoint properties
        case "bounds.origin":
            if let v = value as? CGPoint { layer._bounds.origin = v }
        case "anchorPoint":
            if let v = value as? CGPoint { layer._anchorPoint = v }

        // CGSize properties
        case "shadowOffset":
            if let v = value as? CGSize { layer._shadowOffset = v }
        case "bounds.size":
            if let v = value as? CGSize { layer._bounds.size = v }

        // CGRect properties
        case "contentsRect":
            if let v = value as? CGRect { layer.contentsRect = v }
        case "contentsCenter":
            if let v = value as? CGRect { layer.contentsCenter = v }

        // CATransform3D properties
        case "sublayerTransform":
            if let v = value as? CATransform3D { layer._sublayerTransform = v }

        // CAGradientLayer properties
        case "startPoint":
            if let gradientLayer = layer as? CAGradientLayer, let v = value as? CGPoint { gradientLayer._startPoint = v }
        case "endPoint":
            if let gradientLayer = layer as? CAGradientLayer, let v = value as? CGPoint { gradientLayer._endPoint = v }
        case "colors":
            if let gradientLayer = layer as? CAGradientLayer, let v = value as? [Any] { gradientLayer._colors = v }
        case "locations":
            if let gradientLayer = layer as? CAGradientLayer, let v = value as? [CGFloat] { gradientLayer._locations = v }

        // CATextLayer properties
        case "fontSize":
            if let textLayer = layer as? CATextLayer, let v = value as? CGFloat { textLayer._fontSize = v }
        case "foregroundColor":
            if let textLayer = layer as? CATextLayer, let v = extractColor(value) { textLayer._foregroundColor = v }

        // CAEmitterLayer properties
        case "emitterPosition":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? CGPoint { emitterLayer._emitterPosition = v }
        case "emitterSize":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? CGSize { emitterLayer._emitterSize = v }
        case "emitterZPosition":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? CGFloat { emitterLayer._emitterZPosition = v }
        case "emitterDepth":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? CGFloat { emitterLayer._emitterDepth = v }
        case "birthRate":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? Float { emitterLayer._birthRate = v }
        case "lifetime":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? Float { emitterLayer._lifetime = v }
        case "velocity":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? Float { emitterLayer._velocity = v }
        case "scale":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? Float { emitterLayer._scale = v }
        case "spin":
            if let emitterLayer = layer as? CAEmitterLayer, let v = value as? Float { emitterLayer._spin = v }

        // CAReplicatorLayer properties
        case "instanceDelay":
            if let replicatorLayer = layer as? CAReplicatorLayer, let v = value as? CFTimeInterval { replicatorLayer._instanceDelay = v }
        case "instanceTransform":
            if let replicatorLayer = layer as? CAReplicatorLayer, let v = value as? CATransform3D { replicatorLayer._instanceTransform = v }
        case "instanceColor":
            if let replicatorLayer = layer as? CAReplicatorLayer, let v = extractColor(value) { replicatorLayer._instanceColor = v }
        case "instanceRedOffset":
            if let replicatorLayer = layer as? CAReplicatorLayer, let v = value as? Float { replicatorLayer._instanceRedOffset = v }
        case "instanceGreenOffset":
            if let replicatorLayer = layer as? CAReplicatorLayer, let v = value as? Float { replicatorLayer._instanceGreenOffset = v }
        case "instanceBlueOffset":
            if let replicatorLayer = layer as? CAReplicatorLayer, let v = value as? Float { replicatorLayer._instanceBlueOffset = v }
        case "instanceAlphaOffset":
            if let replicatorLayer = layer as? CAReplicatorLayer, let v = value as? Float { replicatorLayer._instanceAlphaOffset = v }

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
        case "position.x", "position.y":
            if let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                let value = f + CGFloat(progress) * (t - f)
                if keyPath == "position.x" { layer._position.x = value }
                else { layer._position.y = value }
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
        case "cornerRadius", "borderWidth", "shadowRadius", "zPosition",
             "anchorPointZ", "contentsScale", "rasterizationScale":
            if let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                let value = f + CGFloat(progress) * (t - f)
                switch keyPath {
                case "cornerRadius": layer._cornerRadius = value
                case "borderWidth": layer._borderWidth = value
                case "shadowRadius": layer._shadowRadius = value
                case "zPosition": layer._zPosition = value
                case "anchorPointZ": layer._anchorPointZ = value
                case "contentsScale": layer._contentsScale = value
                case "rasterizationScale": layer.rasterizationScale = value
                default: break
                }
            }
        case "shadowOpacity":
            if let f = fromValue as? Float, let t = toValue as? Float {
                layer._shadowOpacity = f + Float(progress) * (t - f)
            }
        case "hidden", "isHidden", "masksToBounds", "doubleSided", "isDoubleSided", "shouldRasterize":
            if let f = fromValue as? Bool, let t = toValue as? Bool {
                let fromScalar: CFTimeInterval = f ? 1 : 0
                let toScalar: CFTimeInterval = t ? 1 : 0
                applyBooleanAnimationValue(
                    fromScalar + progress * (toScalar - fromScalar) != 0,
                    to: layer,
                    keyPath: keyPath
                )
            }
        case "anchorPoint":
            if let f = fromValue as? CGPoint, let t = toValue as? CGPoint {
                layer._anchorPoint = CGPoint(
                    x: f.x + CGFloat(progress) * (t.x - f.x),
                    y: f.y + CGFloat(progress) * (t.y - f.y)
                )
            }
        case "bounds.origin":
            if let f = fromValue as? CGPoint, let t = toValue as? CGPoint {
                layer._bounds.origin = CGPoint(
                    x: f.x + CGFloat(progress) * (t.x - f.x),
                    y: f.y + CGFloat(progress) * (t.y - f.y)
                )
            }
        case "shadowOffset":
            if let f = fromValue as? CGSize, let t = toValue as? CGSize {
                layer._shadowOffset = CGSize(
                    width: f.width + CGFloat(progress) * (t.width - f.width),
                    height: f.height + CGFloat(progress) * (t.height - f.height)
                )
            }
        case "bounds.size":
            if let f = fromValue as? CGSize, let t = toValue as? CGSize {
                layer._bounds.size = CGSize(
                    width: f.width + CGFloat(progress) * (t.width - f.width),
                    height: f.height + CGFloat(progress) * (t.height - f.height)
                )
            }
        case "contentsRect", "contentsCenter":
            if let f = fromValue as? CGRect, let t = toValue as? CGRect {
                let value = CGRect(
                    x: f.origin.x + CGFloat(progress) * (t.origin.x - f.origin.x),
                    y: f.origin.y + CGFloat(progress) * (t.origin.y - f.origin.y),
                    width: f.size.width + CGFloat(progress) * (t.size.width - f.size.width),
                    height: f.size.height + CGFloat(progress) * (t.size.height - f.size.height)
                )
                switch keyPath {
                case "contentsRect": layer.contentsRect = value
                case "contentsCenter": layer.contentsCenter = value
                default: break
                }
            }
        case "transform":
            if let f = fromValue as? CATransform3D, let t = toValue as? CATransform3D {
                layer._transform = interpolateTransform(from: f, to: t, progress: CGFloat(progress))
            }
        case "sublayerTransform":
            if let f = fromValue as? CATransform3D, let t = toValue as? CATransform3D {
                layer._sublayerTransform = interpolateTransform(from: f, to: t, progress: CGFloat(progress))
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
               let fromLocations = fromValue as? [CGFloat], let toLocations = toValue as? [CGFloat] {
                let count = min(fromLocations.count, toLocations.count)
                var interpolatedLocations: [CGFloat] = []
                for i in 0..<count {
                    interpolatedLocations.append(fromLocations[i] + CGFloat(progress) * (toLocations[i] - fromLocations[i]))
                }
                gradientLayer._locations = interpolatedLocations
            }

        // CATextLayer properties
        case "fontSize":
            if let textLayer = layer as? CATextLayer,
               let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                textLayer._fontSize = f + CGFloat(progress) * (t - f)
            }
        case "foregroundColor":
            if let textLayer = layer as? CATextLayer,
               let f = extractColor(fromValue), let t = extractColor(toValue) {
                textLayer._foregroundColor = interpolateColor(from: f, to: t, progress: CGFloat(progress))
            }

        // CAEmitterLayer properties
        case "emitterPosition":
            if let emitterLayer = layer as? CAEmitterLayer,
               let f = fromValue as? CGPoint, let t = toValue as? CGPoint {
                emitterLayer._emitterPosition = CGPoint(
                    x: f.x + CGFloat(progress) * (t.x - f.x),
                    y: f.y + CGFloat(progress) * (t.y - f.y)
                )
            }
        case "emitterSize":
            if let emitterLayer = layer as? CAEmitterLayer,
               let f = fromValue as? CGSize, let t = toValue as? CGSize {
                emitterLayer._emitterSize = CGSize(
                    width: f.width + CGFloat(progress) * (t.width - f.width),
                    height: f.height + CGFloat(progress) * (t.height - f.height)
                )
            }
        case "emitterZPosition", "emitterDepth":
            if let emitterLayer = layer as? CAEmitterLayer,
               let f = fromValue as? CGFloat, let t = toValue as? CGFloat {
                let value = f + CGFloat(progress) * (t - f)
                if keyPath == "emitterZPosition" { emitterLayer._emitterZPosition = value }
                else { emitterLayer._emitterDepth = value }
            }
        case "birthRate", "lifetime", "velocity", "scale", "spin":
            if let emitterLayer = layer as? CAEmitterLayer,
               let f = fromValue as? Float, let t = toValue as? Float {
                let value = f + Float(progress) * (t - f)
                switch keyPath {
                case "birthRate": emitterLayer._birthRate = value
                case "lifetime": emitterLayer._lifetime = value
                case "velocity": emitterLayer._velocity = value
                case "scale": emitterLayer._scale = value
                default: emitterLayer._spin = value
                }
            }

        // CAReplicatorLayer properties
        case "instanceDelay":
            if let replicatorLayer = layer as? CAReplicatorLayer,
               let f = fromValue as? CFTimeInterval, let t = toValue as? CFTimeInterval {
                replicatorLayer._instanceDelay = f + progress * (t - f)
            }
        case "instanceTransform":
            if let replicatorLayer = layer as? CAReplicatorLayer,
               let f = fromValue as? CATransform3D, let t = toValue as? CATransform3D {
                replicatorLayer._instanceTransform = interpolateTransform(from: f, to: t, progress: CGFloat(progress))
            }
        case "instanceColor":
            if let replicatorLayer = layer as? CAReplicatorLayer,
               let f = extractColor(fromValue), let t = extractColor(toValue) {
                replicatorLayer._instanceColor = interpolateColor(from: f, to: t, progress: CGFloat(progress))
            }
        case "instanceRedOffset", "instanceGreenOffset", "instanceBlueOffset", "instanceAlphaOffset":
            if let replicatorLayer = layer as? CAReplicatorLayer,
               let f = fromValue as? Float, let t = toValue as? Float {
                let value = f + Float(progress) * (t - f)
                switch keyPath {
                case "instanceRedOffset": replicatorLayer._instanceRedOffset = value
                case "instanceGreenOffset": replicatorLayer._instanceGreenOffset = value
                case "instanceBlueOffset": replicatorLayer._instanceBlueOffset = value
                default: replicatorLayer._instanceAlphaOffset = value
                }
            }

        case "path":
            if let shapeLayer = layer as? CAShapeLayer,
               let f = extractPath(fromValue), let t = extractPath(toValue) {
                shapeLayer._path = interpolatePath(from: f, to: t, progress: CGFloat(progress))
            }
        case "shadowPath":
            if let f = extractPath(fromValue), let t = extractPath(toValue) {
                layer._shadowPath = interpolatePath(from: f, to: t, progress: CGFloat(progress))
            }

        default:
            break
        }
    }

    private struct CubicParameters {
        let tension: CGFloat
        let continuity: CGFloat
        let bias: CGFloat
    }

    private func cubicParameters(
        for animation: CAKeyframeAnimation,
        at index: Int
    ) -> CubicParameters {
        func value(_ values: [CGFloat]?) -> CGFloat {
            guard let values, values.indices.contains(index) else { return 0 }
            return values[index]
        }
        return CubicParameters(
            tension: value(animation.tensionValues),
            continuity: value(animation.continuityValues),
            bias: value(animation.biasValues)
        )
    }

    /// Applies Kochanek-Bartels cubic interpolation without falling back to linear
    /// interpolation when a value family is not cubic-interpolable.
    private func interpolateCubicKeyframeValue(
        p0: Any,
        p1: Any,
        p2: Any,
        p3: Any,
        t: CGFloat,
        startParameters: CubicParameters,
        endParameters: CubicParameters,
        layer: CALayer,
        keyPath: String
    ) {
        guard let value = cubicValue(
            p0: p0,
            p1: p1,
            p2: p2,
            p3: p3,
            t: t,
            startParameters: startParameters,
            endParameters: endParameters
        ) else {
            return
        }
        applyKeyframeValue(value, layer: layer, keyPath: keyPath)
    }

    private func cubicValue(
        p0: Any,
        p1: Any,
        p2: Any,
        p3: Any,
        t: CGFloat,
        startParameters: CubicParameters,
        endParameters: CubicParameters
    ) -> Any? {
        func scalar(_ v0: CGFloat, _ v1: CGFloat, _ v2: CGFloat, _ v3: CGFloat) -> CGFloat {
            cubicInterpolate(
                v0, v1, v2, v3, t,
                startParameters: startParameters,
                endParameters: endParameters
            )
        }

        if let v0 = p0 as? CGFloat,
           let v1 = p1 as? CGFloat,
           let v2 = p2 as? CGFloat,
           let v3 = p3 as? CGFloat {
            return scalar(v0, v1, v2, v3)
        }
        if let v0 = p0 as? Float,
           let v1 = p1 as? Float,
           let v2 = p2 as? Float,
           let v3 = p3 as? Float {
            return Float(scalar(CGFloat(v0), CGFloat(v1), CGFloat(v2), CGFloat(v3)))
        }
        if let v0 = p0 as? Bool,
           let v1 = p1 as? Bool,
           let v2 = p2 as? Bool,
           let v3 = p3 as? Bool {
            return scalar(
                v0 ? 1 : 0,
                v1 ? 1 : 0,
                v2 ? 1 : 0,
                v3 ? 1 : 0
            ) != 0
        }
        if let v0 = p0 as? CGPoint,
           let v1 = p1 as? CGPoint,
           let v2 = p2 as? CGPoint,
           let v3 = p3 as? CGPoint {
            return CGPoint(
                x: scalar(v0.x, v1.x, v2.x, v3.x),
                y: scalar(v0.y, v1.y, v2.y, v3.y)
            )
        }
        if let v0 = p0 as? CGSize,
           let v1 = p1 as? CGSize,
           let v2 = p2 as? CGSize,
           let v3 = p3 as? CGSize {
            return CGSize(
                width: scalar(v0.width, v1.width, v2.width, v3.width),
                height: scalar(v0.height, v1.height, v2.height, v3.height)
            )
        }
        if let v0 = p0 as? CGRect,
           let v1 = p1 as? CGRect,
           let v2 = p2 as? CGRect,
           let v3 = p3 as? CGRect {
            return CGRect(
                x: scalar(v0.origin.x, v1.origin.x, v2.origin.x, v3.origin.x),
                y: scalar(v0.origin.y, v1.origin.y, v2.origin.y, v3.origin.y),
                width: scalar(v0.size.width, v1.size.width, v2.size.width, v3.size.width),
                height: scalar(v0.size.height, v1.size.height, v2.size.height, v3.size.height)
            )
        }
        if let v0 = p0 as? CATransform3D,
           let v1 = p1 as? CATransform3D,
           let v2 = p2 as? CATransform3D,
           let v3 = p3 as? CATransform3D {
            return cubicTransform(
                v0, v1, v2, v3, t,
                startParameters: startParameters,
                endParameters: endParameters
            )
        }
        if let v0 = extractColor(p0),
           let v1 = extractColor(p1),
           let v2 = extractColor(p2),
           let v3 = extractColor(p3) {
            let c0 = extractRGBA(from: v0)
            let c1 = extractRGBA(from: v1)
            let c2 = extractRGBA(from: v2)
            let c3 = extractRGBA(from: v3)
            return CGColor(
                red: scalar(c0.r, c1.r, c2.r, c3.r),
                green: scalar(c0.g, c1.g, c2.g, c3.g),
                blue: scalar(c0.b, c1.b, c2.b, c3.b),
                alpha: scalar(c0.a, c1.a, c2.a, c3.a)
            )
        }
        if let v0 = p0 as? [CGFloat],
           let v1 = p1 as? [CGFloat],
           let v2 = p2 as? [CGFloat],
           let v3 = p3 as? [CGFloat],
           v0.count == v1.count,
           v1.count == v2.count,
           v2.count == v3.count {
            return v0.indices.map { index in
                scalar(v0[index], v1[index], v2[index], v3[index])
            }
        }
        if let v0 = p0 as? [Any],
           let v1 = p1 as? [Any],
           let v2 = p2 as? [Any],
           let v3 = p3 as? [Any],
           v0.count == v1.count,
           v1.count == v2.count,
           v2.count == v3.count {
            var colors: [Any] = []
            colors.reserveCapacity(v0.count)
            for index in v0.indices {
                guard let color = cubicValue(
                    p0: v0[index],
                    p1: v1[index],
                    p2: v2[index],
                    p3: v3[index],
                    t: t,
                    startParameters: startParameters,
                    endParameters: endParameters
                ) as? CGColor else {
                    return nil
                }
                colors.append(color)
            }
            return colors
        }
        if let v0 = extractPath(p0),
           let v1 = extractPath(p1),
           let v2 = extractPath(p2),
           let v3 = extractPath(p3) {
            return cubicPath(
                v0, v1, v2, v3, t,
                startParameters: startParameters,
                endParameters: endParameters
            )
        }
        return nil
    }

    private func cubicInterpolate(
        _ p0: CGFloat,
        _ p1: CGFloat,
        _ p2: CGFloat,
        _ p3: CGFloat,
        _ t: CGFloat,
        startParameters: CubicParameters,
        endParameters: CubicParameters
    ) -> CGFloat {
        let startScale = 0.5 * (1 - startParameters.tension)
        let outgoing = startScale * (
            (1 + startParameters.continuity) * (1 + startParameters.bias) * (p1 - p0)
                + (1 - startParameters.continuity) * (1 - startParameters.bias) * (p2 - p1)
        )
        let endScale = 0.5 * (1 - endParameters.tension)
        let incoming = endScale * (
            (1 - endParameters.continuity) * (1 + endParameters.bias) * (p2 - p1)
                + (1 + endParameters.continuity) * (1 - endParameters.bias) * (p3 - p2)
        )
        let t2 = t * t
        let t3 = t2 * t
        let h00 = 2 * t3 - 3 * t2 + 1
        let h10 = t3 - 2 * t2 + t
        let h01 = -2 * t3 + 3 * t2
        let h11 = t3 - t2
        return h00 * p1 + h10 * outgoing + h01 * p2 + h11 * incoming
    }

    private func cubicTransform(
        _ p0: CATransform3D,
        _ p1: CATransform3D,
        _ p2: CATransform3D,
        _ p3: CATransform3D,
        _ t: CGFloat,
        startParameters: CubicParameters,
        endParameters: CubicParameters
    ) -> CATransform3D {
        func scalar(_ v0: CGFloat, _ v1: CGFloat, _ v2: CGFloat, _ v3: CGFloat) -> CGFloat {
            cubicInterpolate(
                v0, v1, v2, v3, t,
                startParameters: startParameters,
                endParameters: endParameters
            )
        }
        return CATransform3D(
            m11: scalar(p0.m11, p1.m11, p2.m11, p3.m11),
            m12: scalar(p0.m12, p1.m12, p2.m12, p3.m12),
            m13: scalar(p0.m13, p1.m13, p2.m13, p3.m13),
            m14: scalar(p0.m14, p1.m14, p2.m14, p3.m14),
            m21: scalar(p0.m21, p1.m21, p2.m21, p3.m21),
            m22: scalar(p0.m22, p1.m22, p2.m22, p3.m22),
            m23: scalar(p0.m23, p1.m23, p2.m23, p3.m23),
            m24: scalar(p0.m24, p1.m24, p2.m24, p3.m24),
            m31: scalar(p0.m31, p1.m31, p2.m31, p3.m31),
            m32: scalar(p0.m32, p1.m32, p2.m32, p3.m32),
            m33: scalar(p0.m33, p1.m33, p2.m33, p3.m33),
            m34: scalar(p0.m34, p1.m34, p2.m34, p3.m34),
            m41: scalar(p0.m41, p1.m41, p2.m41, p3.m41),
            m42: scalar(p0.m42, p1.m42, p2.m42, p3.m42),
            m43: scalar(p0.m43, p1.m43, p2.m43, p3.m43),
            m44: scalar(p0.m44, p1.m44, p2.m44, p3.m44)
        )
    }

    private func cubicPath(
        _ p0: CGPath,
        _ p1: CGPath,
        _ p2: CGPath,
        _ p3: CGPath,
        _ t: CGFloat,
        startParameters: CubicParameters,
        endParameters: CubicParameters
    ) -> CGPath? {
        let elements0 = pathElements(p0)
        let elements1 = pathElements(p1)
        let elements2 = pathElements(p2)
        let elements3 = pathElements(p3)
        guard elements0.count == elements1.count,
              elements1.count == elements2.count,
              elements2.count == elements3.count else {
            return nil
        }

        let output = CGMutablePath()
        for index in elements0.indices {
            let e0 = elements0[index]
            let e1 = elements1[index]
            let e2 = elements2[index]
            let e3 = elements3[index]
            guard e0.type == e1.type,
                  e1.type == e2.type,
                  e2.type == e3.type,
                  e0.points.count == e1.points.count,
                  e1.points.count == e2.points.count,
                  e2.points.count == e3.points.count else {
                return nil
            }
            let points = e0.points.indices.map { pointIndex in
                CGPoint(
                    x: cubicInterpolate(
                        e0.points[pointIndex].x,
                        e1.points[pointIndex].x,
                        e2.points[pointIndex].x,
                        e3.points[pointIndex].x,
                        t,
                        startParameters: startParameters,
                        endParameters: endParameters
                    ),
                    y: cubicInterpolate(
                        e0.points[pointIndex].y,
                        e1.points[pointIndex].y,
                        e2.points[pointIndex].y,
                        e3.points[pointIndex].y,
                        t,
                        startParameters: startParameters,
                        endParameters: endParameters
                    )
                )
            }
            switch e1.type {
            case .moveToPoint:
                output.move(to: points[0])
            case .addLineToPoint:
                output.addLine(to: points[0])
            case .addQuadCurveToPoint:
                output.addQuadCurve(to: points[1], control: points[0])
            case .addCurveToPoint:
                output.addCurve(to: points[2], control1: points[0], control2: points[1])
            case .closeSubpath:
                output.closeSubpath()
            @unknown default:
                return nil
            }
        }
        return output
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
    open var contents: Any? {
        didSet {
            markDirty(.contents)
            if Self.needsDisplay(forKey: "contents") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "contents", oldValue: oldValue, newValue: contents)
        }
    }

    private var _contentsRect: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)
    /// The rectangle, in the unit coordinate space, that defines the portion of the layer's
    /// contents that should be used. Animatable.
    open var contentsRect: CGRect {
        get { return _contentsRect }
        set {
            let oldValue = _contentsRect
            guard oldValue != newValue else { return }
            _contentsRect = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "contentsRect") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "contentsRect", oldValue: oldValue, newValue: newValue)
        }
    }

    /// The rectangle that defines how the layer contents are scaled if the layer's contents
    /// are resized. Animatable.
    open var contentsCenter: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1) {
        didSet {
            guard oldValue != contentsCenter else { return }
            markDirty(.contents)
            if Self.needsDisplay(forKey: "contentsCenter") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "contentsCenter", oldValue: oldValue, newValue: contentsCenter)
        }
    }

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
    open var contentsGravity: CALayerContentsGravity = .resize {
        didSet {
            guard oldValue != contentsGravity else { return }
            markDirty(.contents)
            if Self.needsDisplay(forKey: "contentsGravity") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "contentsGravity", oldValue: oldValue, newValue: contentsGravity)
        }
    }

    private var _opacity: Float = 1.0
    /// The opacity of the receiver. Animatable.
    open var opacity: Float {
        get { return _opacity }
        set {
            let oldValue = _opacity
            guard oldValue != newValue else { return }
            _opacity = newValue
            markDirty(.appearance)
            if Self.needsDisplay(forKey: "opacity") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "opacity", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _isHidden: Bool = false
    /// A Boolean indicating whether the layer is displayed. Animatable.
    open var isHidden: Bool {
        get { return _isHidden }
        set {
            let oldValue = _isHidden
            guard oldValue != newValue else { return }
            _isHidden = newValue
            markDirty(.appearance)
            if Self.needsDisplay(forKey: "isHidden") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "isHidden", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _masksToBounds: Bool = false
    /// A Boolean indicating whether sublayers are clipped to the layer's bounds. Animatable.
    open var masksToBounds: Bool {
        get { return _masksToBounds }
        set {
            let oldValue = _masksToBounds
            guard oldValue != newValue else { return }
            _masksToBounds = newValue
            markDirty(.appearance)
            CATransaction.registerChange(layer: self, keyPath: "masksToBounds", oldValue: oldValue, newValue: newValue)
        }
    }

    /// An optional layer whose alpha channel is used to mask the layer's content.
    open var mask: CALayer? {
        didSet {
            guard oldValue !== mask else { return }
            markDirty(.mask)
        }
    }

    private var _isDoubleSided: Bool = true
    /// A Boolean indicating whether the layer displays its content when facing away from the viewer. Animatable.
    open var isDoubleSided: Bool {
        get { return _isDoubleSided }
        set {
            let oldValue = _isDoubleSided
            guard oldValue != newValue else { return }
            _isDoubleSided = newValue
            markDirty(.appearance)
            CATransaction.registerChange(layer: self, keyPath: "isDoubleSided", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _cornerRadius: CGFloat = 0
    /// The radius to use when drawing rounded corners for the layer's background. Animatable.
    open var cornerRadius: CGFloat {
        get { return _cornerRadius }
        set {
            let oldValue = _cornerRadius
            guard oldValue != newValue else { return }
            _cornerRadius = newValue
            markDirty(.appearance)
            if Self.needsDisplay(forKey: "cornerRadius") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "cornerRadius", oldValue: oldValue, newValue: newValue)
        }
    }

    /// A bitmask defining which of the four corners receives the masking.
    open var maskedCorners: CACornerMask = [.layerMinXMinYCorner, .layerMaxXMinYCorner, .layerMinXMaxYCorner, .layerMaxXMaxYCorner] {
        didSet {
            guard oldValue != maskedCorners else { return }
            markDirty(.appearance)
        }
    }

    /// The curve to use when drawing the rounded corners.
    open var cornerCurve: CALayerCornerCurve = .circular {
        didSet {
            guard oldValue != cornerCurve else { return }
            markDirty(.appearance)
        }
    }

    private var _borderWidth: CGFloat = 0
    /// The width of the layer's border. Animatable.
    open var borderWidth: CGFloat {
        get { return _borderWidth }
        set {
            let oldValue = _borderWidth
            guard oldValue != newValue else { return }
            _borderWidth = newValue
            markDirty(.appearance)
            CATransaction.registerChange(layer: self, keyPath: "borderWidth", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _borderColor: CGColor?
    /// The color of the layer's border. Animatable.
    open var borderColor: CGColor? {
        get { return _borderColor }
        set {
            let oldValue = _borderColor
            _borderColor = newValue
            markDirty(.appearance)
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
            markDirty(.appearance)
            CATransaction.registerChange(layer: self, keyPath: "backgroundColor", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _shadowOpacity: Float = 0
    /// The opacity of the layer's shadow. Animatable.
    open var shadowOpacity: Float {
        get { return _shadowOpacity }
        set {
            let oldValue = _shadowOpacity
            guard oldValue != newValue else { return }
            let oldContribution = selfShadowContribution
            _shadowOpacity = newValue
            CALayer.propagateShadowDelta(selfShadowContribution - oldContribution, startingAt: self)
            markDirty(.shadow)
            if Self.needsDisplay(forKey: "shadowOpacity") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "shadowOpacity", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _shadowRadius: CGFloat = 3
    /// The blur radius (in points) used to render the layer's shadow. Animatable.
    open var shadowRadius: CGFloat {
        get { return _shadowRadius }
        set {
            let oldValue = _shadowRadius
            guard oldValue != newValue else { return }
            _shadowRadius = newValue
            markDirty(.shadow)
            if Self.needsDisplay(forKey: "shadowRadius") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "shadowRadius", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _shadowOffset: CGSize = CGSize(width: 0, height: -3)
    /// The offset (in points) of the layer's shadow. Animatable.
    open var shadowOffset: CGSize {
        get { return _shadowOffset }
        set {
            let oldValue = _shadowOffset
            guard oldValue != newValue else { return }
            _shadowOffset = newValue
            markDirty(.shadow)
            CATransaction.registerChange(layer: self, keyPath: "shadowOffset", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _shadowColor: CGColor? = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
    /// The color of the layer's shadow. Animatable.
    open var shadowColor: CGColor? {
        get { return _shadowColor }
        set {
            let oldValue = _shadowColor
            let oldContribution = selfShadowContribution
            _shadowColor = newValue
            CALayer.propagateShadowDelta(selfShadowContribution - oldContribution, startingAt: self)
            markDirty(.shadow)
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
            markDirty(.shadow)
            CATransaction.registerChange(layer: self, keyPath: "shadowPath", oldValue: oldValue, newValue: newValue)
        }
    }

    /// An optional dictionary used to store property values that aren't explicitly defined by the layer.
    open var style: [AnyHashable: Any]?

    /// A Boolean indicating whether the layer is allowed to perform edge antialiasing.
    open var allowsEdgeAntialiasing: Bool = false {
        didSet {
            guard oldValue != allowsEdgeAntialiasing else { return }
            markDirty(.rasterization)
        }
    }

    /// A Boolean indicating whether the layer is allowed to composite itself as a group separate from its parent.
    open var allowsGroupOpacity: Bool = true {
        didSet {
            guard oldValue != allowsGroupOpacity else { return }
            markDirty(.appearance)
        }
    }

    // MARK: - Layer Filters

    fileprivate var _filters: [Any]?
    /// An array of Core Image filters to apply to the contents of the layer and its sublayers. Animatable.
    open var filters: [Any]? {
        get { return _filters }
        set {
            let oldContribution = selfFilterContribution
            _filters = newValue
            CALayer.propagateFilterDelta(selfFilterContribution - oldContribution, startingAt: self)
            markDirty(.filters)
        }
    }

    /// A CoreImage filter used to composite the layer and the content behind it. Animatable.
    open var compositingFilter: Any? {
        didSet { markDirty(.filters) }
    }

    /// An array of Core Image filters to apply to the content immediately behind the layer. Animatable.
    open var backgroundFilters: [Any]? {
        didSet { markDirty(.filters) }
    }

    /// The filter used when reducing the size of the content.
    open var minificationFilter: CALayerContentsFilter = .linear {
        didSet {
            guard oldValue != minificationFilter else { return }
            markDirty(.contents)
        }
    }

    /// The bias factor used by the minification filter to determine the levels of detail.
    open var minificationFilterBias: Float = 0 {
        didSet {
            guard oldValue != minificationFilterBias else { return }
            markDirty(.contents)
        }
    }

    /// The filter used when increasing the size of the content.
    open var magnificationFilter: CALayerContentsFilter = .linear {
        didSet {
            guard oldValue != magnificationFilter else { return }
            markDirty(.contents)
        }
    }

    // MARK: - Configuring the Layer's Rendering Behavior

    private var _isOpaque: Bool = false
    /// A Boolean value indicating whether the layer contains completely opaque content.
    open var isOpaque: Bool {
        get { return _isOpaque }
        set {
            guard _isOpaque != newValue else { return }
            _isOpaque = newValue
            markDirty(.rasterization)
            if Self.needsDisplay(forKey: "isOpaque") { setNeedsDisplay() }
        }
    }

    /// A bitmask defining how the edges of the receiver are rasterized.
    open var edgeAntialiasingMask: CAEdgeAntialiasingMask = [.layerLeftEdge, .layerRightEdge, .layerBottomEdge, .layerTopEdge] {
        didSet {
            guard oldValue != edgeAntialiasingMask else { return }
            markDirty(.rasterization)
        }
    }

    /// Returns a Boolean indicating whether the layer content is implicitly flipped when rendered.
    open func contentsAreFlipped() -> Bool {
        isGeometryFlipped
    }

    /// A Boolean that indicates whether the geometry of the layer and its sublayers is flipped vertically.
    open var isGeometryFlipped: Bool = false {
        didSet {
            guard oldValue != isGeometryFlipped else { return }
            markDirty(.geometry)
        }
    }

    /// A Boolean indicating whether drawing commands are deferred and processed asynchronously in a background thread.
    open var drawsAsynchronously: Bool = false {
        didSet {
            guard oldValue != drawsAsynchronously else { return }
            markDirty(.contentsRedraw)
        }
    }

    private var _shouldRasterize: Bool = false
    /// A Boolean that indicates whether the layer is rendered as a bitmap before compositing. Animatable.
    open var shouldRasterize: Bool {
        get { return _shouldRasterize }
        set {
            let oldValue = _shouldRasterize
            guard oldValue != newValue else { return }
            _shouldRasterize = newValue
            markDirty(.rasterization)
            CATransaction.registerChange(
                layer: self,
                keyPath: "shouldRasterize",
                oldValue: oldValue,
                newValue: newValue
            )
        }
    }

    private var _rasterizationScale: CGFloat = 1.0
    /// The scale at which to rasterize content, relative to the coordinate space of the layer. Animatable.
    open var rasterizationScale: CGFloat {
        get { return _rasterizationScale }
        set {
            guard _rasterizationScale != newValue else { return }
            _rasterizationScale = newValue
            markDirty(.rasterization)
        }
    }

    /// A hint for the desired storage format of the layer contents.
    open var contentsFormat: CALayerContentsFormat = .RGBA8Uint {
        didSet {
            guard oldValue != contentsFormat else { return }
            markDirty(.contents)
        }
    }

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
            let clipPath = layerShapePath(in: bounds)
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

        let maskLayer = mask
        if maskLayer != nil {
            ctx.beginTransparencyLayer(auxiliaryInfo: nil)
        }

        // Draw background color
        if let bgColor = backgroundColor {
            ctx.setFillColor(bgColor)
            if cornerRadius > 0 {
                let path = layerShapePath(in: bounds)
                ctx.addPath(path)
                ctx.fillPath()
            } else {
                ctx.fill(bounds)
            }
        }

        // Draw contents
        if let contents = contents as? CGImage {
            drawContents(contents, in: ctx)
        }

        // Let delegate draw if needed
        delegate?.draw(self, in: ctx)

        // Draw border
        if borderWidth > 0, let borderColor = borderColor {
            ctx.setStrokeColor(borderColor)
            ctx.setLineWidth(borderWidth)
            if cornerRadius > 0 {
                let path = layerShapePath(in: bounds.insetBy(dx: borderWidth / 2, dy: borderWidth / 2))
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

        if let maskLayer {
            applyMask(maskLayer, in: ctx)
            ctx.endTransparencyLayer()
        }
    }

    private func layerShapePath(in rect: CGRect) -> CGPath {
        guard rect.width > 0, rect.height > 0 else {
            return CGPath(rect: .zero, transform: nil)
        }

        let radius = min(cornerRadius, min(rect.width, rect.height) / 2)
        guard radius > 0 else {
            return CGPath(rect: rect, transform: nil)
        }

        let topLeft = maskedCorners.contains(.layerMinXMinYCorner) ? radius : 0
        let topRight = maskedCorners.contains(.layerMaxXMinYCorner) ? radius : 0
        let bottomLeft = maskedCorners.contains(.layerMinXMaxYCorner) ? radius : 0
        let bottomRight = maskedCorners.contains(.layerMaxXMaxYCorner) ? radius : 0

        if topLeft == radius, topRight == radius, bottomLeft == radius, bottomRight == radius {
            return CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil)
        }

        if topLeft == 0, topRight == 0, bottomLeft == 0, bottomRight == 0 {
            return CGPath(rect: rect, transform: nil)
        }

        let path = CGMutablePath()
        let kappa = CGFloat(0.5522847498)

        path.move(to: CGPoint(x: rect.minX + topLeft, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX - topRight, y: rect.minY))
        if topRight > 0 {
            let offset = topRight * kappa
            path.addCurve(
                to: CGPoint(x: rect.maxX, y: rect.minY + topRight),
                control1: CGPoint(x: rect.maxX - topRight + offset, y: rect.minY),
                control2: CGPoint(x: rect.maxX, y: rect.minY + topRight - offset)
            )
        } else {
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        }

        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - bottomRight))
        if bottomRight > 0 {
            let offset = bottomRight * kappa
            path.addCurve(
                to: CGPoint(x: rect.maxX - bottomRight, y: rect.maxY),
                control1: CGPoint(x: rect.maxX, y: rect.maxY - bottomRight + offset),
                control2: CGPoint(x: rect.maxX - bottomRight + offset, y: rect.maxY)
            )
        } else {
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        }

        path.addLine(to: CGPoint(x: rect.minX + bottomLeft, y: rect.maxY))
        if bottomLeft > 0 {
            let offset = bottomLeft * kappa
            path.addCurve(
                to: CGPoint(x: rect.minX, y: rect.maxY - bottomLeft),
                control1: CGPoint(x: rect.minX + bottomLeft - offset, y: rect.maxY),
                control2: CGPoint(x: rect.minX, y: rect.maxY - bottomLeft + offset)
            )
        } else {
            path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        }

        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY + topLeft))
        if topLeft > 0 {
            let offset = topLeft * kappa
            path.addCurve(
                to: CGPoint(x: rect.minX + topLeft, y: rect.minY),
                control1: CGPoint(x: rect.minX, y: rect.minY + topLeft - offset),
                control2: CGPoint(x: rect.minX + topLeft - offset, y: rect.minY)
            )
        } else {
            path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
        }

        path.closeSubpath()
        return path
    }

    private func applyMask(_ maskLayer: CALayer, in ctx: CGContext) {
        ctx.saveGState()
        defer { ctx.restoreGState() }

        let maskOrigin = CGPoint(
            x: maskLayer.position.x - maskLayer.bounds.width * maskLayer.anchorPoint.x,
            y: maskLayer.position.y - maskLayer.bounds.height * maskLayer.anchorPoint.y
        )
        ctx.translateBy(x: maskOrigin.x, y: maskOrigin.y)
        ctx.setBlendMode(.destinationIn)
        maskLayer.render(in: ctx)
    }

    /// Draws the contents image into the context.
    private func drawContents(_ image: CGImage, in ctx: CGContext) {
        let sourceImage = applyContentsRect(to: image)
        if shouldUseNineSliceScaling, contentsCenter != CGRect(x: 0, y: 0, width: 1, height: 1) {
            drawNineSliceContents(sourceImage, in: ctx)
        } else {
            let destRect = calculateContentsRect(for: sourceImage)
            ctx.draw(sourceImage, in: destRect)
        }
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

    private var shouldUseNineSliceScaling: Bool {
        switch contentsGravity {
        case .resize, .resizeAspect, .resizeAspectFill:
            return true
        default:
            return false
        }
    }

    private func applyContentsRect(to image: CGImage) -> CGImage {
        let unitRect = CGRect(x: 0, y: 0, width: 1, height: 1)
        guard contentsRect != unitRect else { return image }

        let minX = Int((contentsRect.minX * CGFloat(image.width)).rounded(.down))
        let minY = Int((contentsRect.minY * CGFloat(image.height)).rounded(.down))
        let maxX = Int((contentsRect.maxX * CGFloat(image.width)).rounded(.up))
        let maxY = Int((contentsRect.maxY * CGFloat(image.height)).rounded(.up))
        let cropRect = CGRect(
            x: CGFloat(minX),
            y: CGFloat(minY),
            width: CGFloat(maxX - minX),
            height: CGFloat(maxY - minY)
        )

        return image.cropping(to: cropRect) ?? image
    }

    private func drawNineSliceContents(_ image: CGImage, in ctx: CGContext) {
        let destinationRect = calculateContentsRect(for: image)
        let centerRect = resolvedContentsCenter(in: image)

        let leftSourceWidth = centerRect.minX
        let centerSourceWidth = centerRect.width
        let rightSourceWidth = CGFloat(image.width) - centerRect.maxX
        let topSourceHeight = centerRect.minY
        let centerSourceHeight = centerRect.height
        let bottomSourceHeight = CGFloat(image.height) - centerRect.maxY

        var leftDestinationWidth = leftSourceWidth
        var rightDestinationWidth = rightSourceWidth
        let fixedHorizontalWidth = leftDestinationWidth + rightDestinationWidth
        if fixedHorizontalWidth > destinationRect.width, fixedHorizontalWidth > 0 {
            let scale = destinationRect.width / fixedHorizontalWidth
            leftDestinationWidth *= scale
            rightDestinationWidth *= scale
        }
        let centerDestinationWidth = max(0, destinationRect.width - leftDestinationWidth - rightDestinationWidth)

        var topDestinationHeight = topSourceHeight
        var bottomDestinationHeight = bottomSourceHeight
        let fixedVerticalHeight = topDestinationHeight + bottomDestinationHeight
        if fixedVerticalHeight > destinationRect.height, fixedVerticalHeight > 0 {
            let scale = destinationRect.height / fixedVerticalHeight
            topDestinationHeight *= scale
            bottomDestinationHeight *= scale
        }
        let centerDestinationHeight = max(0, destinationRect.height - topDestinationHeight - bottomDestinationHeight)

        let sourceColumns = [
            (origin: CGFloat(0), size: leftSourceWidth),
            (origin: centerRect.minX, size: centerSourceWidth),
            (origin: centerRect.maxX, size: rightSourceWidth)
        ]
        let sourceRows = [
            (origin: CGFloat(0), size: topSourceHeight),
            (origin: centerRect.minY, size: centerSourceHeight),
            (origin: centerRect.maxY, size: bottomSourceHeight)
        ]
        let destinationColumns = [
            (origin: destinationRect.minX, size: leftDestinationWidth),
            (origin: destinationRect.minX + leftDestinationWidth, size: centerDestinationWidth),
            (origin: destinationRect.maxX - rightDestinationWidth, size: rightDestinationWidth)
        ]
        let destinationRows = [
            (origin: destinationRect.minY, size: topDestinationHeight),
            (origin: destinationRect.minY + topDestinationHeight, size: centerDestinationHeight),
            (origin: destinationRect.maxY - bottomDestinationHeight, size: bottomDestinationHeight)
        ]

        for row in 0..<3 {
            for column in 0..<3 {
                let sourceRect = CGRect(
                    x: sourceColumns[column].origin,
                    y: sourceRows[row].origin,
                    width: sourceColumns[column].size,
                    height: sourceRows[row].size
                )
                let destinationRect = CGRect(
                    x: destinationColumns[column].origin,
                    y: destinationRows[row].origin,
                    width: destinationColumns[column].size,
                    height: destinationRows[row].size
                )

                guard sourceRect.width > 0,
                      sourceRect.height > 0,
                      destinationRect.width > 0,
                      destinationRect.height > 0,
                      let sliceImage = image.cropping(to: sourceRect) else {
                    continue
                }

                ctx.draw(sliceImage, in: destinationRect)
            }
        }
    }

    private func resolvedContentsCenter(in image: CGImage) -> CGRect {
        let minX = Int((contentsCenter.minX * CGFloat(image.width)).rounded(.down))
        let minY = Int((contentsCenter.minY * CGFloat(image.height)).rounded(.down))
        let maxX = Int((contentsCenter.maxX * CGFloat(image.width)).rounded(.up))
        let maxY = Int((contentsCenter.maxY * CGFloat(image.height)).rounded(.up))

        let clampedMinX = max(0, min(image.width, minX))
        let clampedMinY = max(0, min(image.height, minY))
        let clampedMaxX = max(clampedMinX, min(image.width, maxX))
        let clampedMaxY = max(clampedMinY, min(image.height, maxY))

        return CGRect(
            x: CGFloat(clampedMinX),
            y: CGFloat(clampedMinY),
            width: CGFloat(clampedMaxX - clampedMinX),
            height: CGFloat(clampedMaxY - clampedMinY)
        )
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
            // Setting frame updates bounds.size and position through their public setters
            // so that transactions are properly registered for implicit animations.
            // This assumes identity transform; for non-identity transforms, behavior is undefined.
            bounds = CGRect(origin: bounds.origin, size: newValue.size)
            position = CGPoint(
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
            guard oldValue != newValue else { return }
            _bounds = newValue
            if oldValue.size != newValue.size {
                resizeSublayers(withOldSize: oldValue.size)
            }
            markDirty(.geometry)
            if needsDisplayOnBoundsChange {
                setNeedsDisplay()
            } else if Self.needsDisplay(forKey: "bounds") {
                setNeedsDisplay()
            }
            CATransaction.registerChange(layer: self, keyPath: "bounds", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _position: CGPoint = .zero
    /// The layer's position in its superlayer's coordinate space. Animatable.
    open var position: CGPoint {
        get { return _position }
        set {
            let oldValue = _position
            guard oldValue != newValue else { return }
            _position = newValue
            markDirty(.geometry)
            CATransaction.registerChange(layer: self, keyPath: "position", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _zPosition: CGFloat = 0
    /// The layer's position on the z axis. Animatable.
    open var zPosition: CGFloat {
        get { return _zPosition }
        set {
            let oldValue = _zPosition
            guard oldValue != newValue else { return }
            _zPosition = newValue
            markDirty(.geometry)
            // The painter's-algorithm sort lives on the parent — mark the
            // parent's sublayer-ordering bit so its sortedSublayers cache
            // (Phase 2) invalidates.
            _superlayer?.markDirty(.sublayerOrdering)
            if Self.needsDisplay(forKey: "zPosition") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "zPosition", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _anchorPointZ: CGFloat = 0
    /// The anchor point for the layer's position along the z axis. Animatable.
    open var anchorPointZ: CGFloat {
        get { return _anchorPointZ }
        set {
            let oldValue = _anchorPointZ
            guard oldValue != newValue else { return }
            _anchorPointZ = newValue
            markDirty(.geometry)
            CATransaction.registerChange(layer: self, keyPath: "anchorPointZ", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _anchorPoint: CGPoint = CGPoint(x: 0.5, y: 0.5)
    /// Defines the anchor point of the layer's bounds rectangle. Animatable.
    open var anchorPoint: CGPoint {
        get { return _anchorPoint }
        set {
            let oldValue = _anchorPoint
            guard oldValue != newValue else { return }
            _anchorPoint = newValue
            markDirty(.geometry)
            CATransaction.registerChange(layer: self, keyPath: "anchorPoint", oldValue: oldValue, newValue: newValue)
        }
    }

    private var _contentsScale: CGFloat = 1.0
    /// The scale factor applied to the layer. Animatable.
    open var contentsScale: CGFloat {
        get { return _contentsScale }
        set {
            let oldValue = _contentsScale
            guard oldValue != newValue else { return }
            _contentsScale = newValue
            markDirty(.contents)
            if Self.needsDisplay(forKey: "contentsScale") { setNeedsDisplay() }
            CATransaction.registerChange(layer: self, keyPath: "contentsScale", oldValue: oldValue, newValue: newValue)
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
            markDirty(.geometry)
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
            markDirty(.geometry)
            CATransaction.registerChange(layer: self, keyPath: "sublayerTransform", oldValue: oldValue, newValue: newValue)
        }
    }

    /// Returns an affine version of the layer's transform.
    open func affineTransform() -> CGAffineTransform {
        return CATransform3DGetAffineTransform(_transform)
    }

    /// Sets the layer's transform to the specified affine transform.
    open func setAffineTransform(_ m: CGAffineTransform) {
        transform = CATransform3DMakeAffineTransform(m)
    }

    // MARK: - Managing the Layer Hierarchy

    private var _sublayers: [CALayer]?
    /// An array containing the layer's sublayers.
    open var sublayers: [CALayer]? {
        get { return _sublayers }
        set {
            // Remove old sublayers (propagate -counts up from self before detaching)
            if let old = _sublayers {
                var shadowDelta = 0
                var filterDelta = 0
                var dirtyDelta = 0
                for child in old {
                    shadowDelta += child._subtreeShadowCount
                    filterDelta += child._subtreeFilterCount
                    dirtyDelta += child._subtreeDirtyCount
                    child._superlayer = nil
                }
                CALayer.propagateShadowDelta(-shadowDelta, startingAt: self)
                CALayer.propagateFilterDelta(-filterDelta, startingAt: self)
                CALayer.propagateDirtyDeltaPublic(-dirtyDelta, startingAt: self)
            }
            newValue?.forEach { child in
                if child._superlayer !== self {
                    child.removeFromSuperlayer()
                }
            }
            // Set new sublayers (propagate +counts up from self after attaching)
            _sublayers = newValue
            if let new = newValue {
                var shadowDelta = 0
                var filterDelta = 0
                var dirtyDelta = 0
                for child in new {
                    child._superlayer = self
                    shadowDelta += child._subtreeShadowCount
                    filterDelta += child._subtreeFilterCount
                    dirtyDelta += child._subtreeDirtyCount
                }
                CALayer.propagateShadowDelta(shadowDelta, startingAt: self)
                CALayer.propagateFilterDelta(filterDelta, startingAt: self)
                CALayer.propagateDirtyDeltaPublic(dirtyDelta, startingAt: self)
            }
            markDirty(.sublayerHierarchy)
        }
    }

    private weak var _superlayer: CALayer?
    /// The superlayer of the layer.
    open var superlayer: CALayer? {
        return _superlayer
    }

    // MARK: - Subtree Render-Effect Counters
    //
    // `_subtreeShadowCount` / `_subtreeFilterCount` track how many descendants
    // (including self) have a shadow or supported filter contribution. The
    // renderer uses these counters to skip the recursive `findFirstShadowLayer`
    // / `findFirstFilteredLayer` walks when no descendant could possibly need
    // pre-rendering.
    //
    // Limitation: counters reflect MODEL state only. Animations that drive
    // `shadowOpacity` from a model value of 0 (so the model contribution is 0
    // but the presentation value is > 0 mid-animation) are not detected.
    // SpriteKit-style users typically set static shadow values, so the model
    // count is accurate for them.

    internal fileprivate(set) var _subtreeShadowCount: Int = 0
    internal fileprivate(set) var _subtreeFilterCount: Int = 0

    // MARK: - Phase 1 dirty propagation (PERFORMANCE_DESIGN.md §3)
    //
    // `_dirtyMask` records which categories of state changed on THIS layer
    // since the last `recursivelyClearDirtyAfterCommit()`. `_subtreeDirtyCount`
    // tracks how many descendants (incl. self) have a non-empty mask, so the
    // renderer can early-return at any depth.
    //
    // The fields live here (not in CALayer+Dirty.swift) because Swift does
    // not allow stored properties in extensions on classes.

    internal var _dirtyMask: DirtyFlags = CALayer._initialDirtyMask
    internal var _subtreeDirtyCount: Int = 1
    /// Monotonically identifies model-state changes even after dirty bits are
    /// cleared. The renderer uses this for detached dependency trees such as
    /// `mask`, whose mutations do not propagate through `superlayer`.
    internal var _contentRevision: UInt64 = 0
    internal var _presentationCacheToken: UInt64 = 0
    internal var _presentationCacheIsValid: Bool = false

    // Phase 2 sublayer ordering cache (PERFORMANCE_DESIGN.md §4.3 / R2.3).
    // `_sortedSublayers` holds the painter's-algorithm sort of `_sublayers`
    // for the frame identified by `_sortedSublayersToken`. Invalidated by
    // `.sublayerHierarchy` (membership change) or `.sublayerOrdering`
    // (any child's `zPosition` change), and by frame-token mismatch.
    internal var _sortedSublayersToken: UInt64 = 0
    internal var _sortedSublayers: [CALayer] = []

    /// Bridge accessor — the extension in CALayer+Dirty.swift cannot see
    /// `_isPresentation` directly (it is `private`).
    internal var _isPresentationLayer: Bool { _isPresentation }

    /// Bridge accessor — the extension in CALayer+Dirty.swift cannot see
    /// `_superlayer` directly (it is `private weak`).
    internal var _superlayerForDirty: CALayer? { _superlayer }

    /// Bridge accessor — the extension in CALayer+Dirty.swift cannot see
    /// `_sublayers` directly (it is `private`).
    internal var _sublayersForDirty: [CALayer]? { _sublayers }

    /// Detached mask trees are not represented by `_subtreeDirtyCount` on the
    /// masked layer, but their dirty state must still be cleared after commit.
    internal var _maskForDirty: CALayer? { mask }

    /// Test-only accessor for the `_needsDisplay` boolean axis (B7).
    internal var _needsDisplayForTest: Bool { _needsDisplay }

    fileprivate var selfShadowContribution: Int {
        (_shadowOpacity > 0 && _shadowColor != nil) ? 1 : 0
    }

    fileprivate var selfFilterContribution: Int {
        (_filters?.isEmpty == false) ? 1 : 0
    }

    fileprivate static func propagateShadowDelta(_ delta: Int, startingAt layer: CALayer?) {
        guard delta != 0 else { return }
        var node = layer
        while let n = node {
            n._subtreeShadowCount += delta
            node = n._superlayer
        }
    }

    fileprivate static func propagateFilterDelta(_ delta: Int, startingAt layer: CALayer?) {
        guard delta != 0 else { return }
        var node = layer
        while let n = node {
            n._subtreeFilterCount += delta
            node = n._superlayer
        }
    }

    /// Appends the layer to the layer's list of sublayers.
    open func addSublayer(_ layer: CALayer) {
        layer.removeFromSuperlayer()
        if _sublayers == nil {
            _sublayers = []
        }
        _sublayers?.append(layer)
        layer._superlayer = self
        CALayer.propagateShadowDelta(layer._subtreeShadowCount, startingAt: self)
        CALayer.propagateFilterDelta(layer._subtreeFilterCount, startingAt: self)
        // Phase 1: parent receives the moving subtree's dirty contribution
        // and a fresh `.sublayerHierarchy` bit. Mirrors propagateShadowDelta.
        CALayer.propagateDirtyDeltaPublic(+layer._subtreeDirtyCount, startingAt: self)
        markDirty(.sublayerHierarchy)
    }

    /// Detaches the layer from its parent layer.
    open func removeFromSuperlayer() {
        guard let superlayer = _superlayer else { return }
        CALayer.propagateShadowDelta(-_subtreeShadowCount, startingAt: superlayer)
        CALayer.propagateFilterDelta(-_subtreeFilterCount, startingAt: superlayer)
        // Phase 1: subtract the leaving subtree's dirty contribution from
        // the OLD ancestor chain BEFORE clearing _superlayer. The parent
        // also receives `.sublayerHierarchy` (its child set just changed).
        CALayer.propagateDirtyDeltaPublic(-_subtreeDirtyCount, startingAt: superlayer)
        superlayer.markDirty(.sublayerHierarchy)
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
        CALayer.propagateShadowDelta(layer._subtreeShadowCount, startingAt: self)
        CALayer.propagateFilterDelta(layer._subtreeFilterCount, startingAt: self)
        CALayer.propagateDirtyDeltaPublic(+layer._subtreeDirtyCount, startingAt: self)
        markDirty(.sublayerHierarchy)
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
        CALayer.propagateShadowDelta(layer._subtreeShadowCount, startingAt: self)
        CALayer.propagateFilterDelta(layer._subtreeFilterCount, startingAt: self)
        CALayer.propagateDirtyDeltaPublic(+layer._subtreeDirtyCount, startingAt: self)
        markDirty(.sublayerHierarchy)
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
        CALayer.propagateShadowDelta(layer._subtreeShadowCount, startingAt: self)
        CALayer.propagateFilterDelta(layer._subtreeFilterCount, startingAt: self)
        CALayer.propagateDirtyDeltaPublic(+layer._subtreeDirtyCount, startingAt: self)
        markDirty(.sublayerHierarchy)
    }

    /// Replaces the specified sublayer with a different layer object.
    open func replaceSublayer(_ oldLayer: CALayer, with newLayer: CALayer) {
        guard let index = _sublayers?.firstIndex(where: { $0 === oldLayer }) else { return }
        newLayer.removeFromSuperlayer()
        CALayer.propagateShadowDelta(-oldLayer._subtreeShadowCount, startingAt: self)
        CALayer.propagateFilterDelta(-oldLayer._subtreeFilterCount, startingAt: self)
        CALayer.propagateDirtyDeltaPublic(-oldLayer._subtreeDirtyCount, startingAt: self)
        oldLayer._superlayer = nil
        _sublayers?[index] = newLayer
        newLayer._superlayer = self
        CALayer.propagateShadowDelta(newLayer._subtreeShadowCount, startingAt: self)
        CALayer.propagateFilterDelta(newLayer._subtreeFilterCount, startingAt: self)
        CALayer.propagateDirtyDeltaPublic(+newLayer._subtreeDirtyCount, startingAt: self)
        markDirty(.sublayerHierarchy)
    }

    // MARK: - Updating Layer Display

    private var _needsDisplay: Bool = false

    /// Marks the layer's contents as needing to be updated.
    open func setNeedsDisplay() {
        _needsDisplay = true
        markDirty(.contentsRedraw)
    }

    /// Marks the region within the specified rectangle as needing to be updated.
    open func setNeedsDisplay(_ r: CGRect) {
        _needsDisplay = true
        markDirty(.contentsRedraw)
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
    ///
    /// Matching Apple's documented behavior, the animation is copied on add —
    /// mutating the original after insertion does not affect the in-flight
    /// animation, and the same template may safely be reused.
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

        // Copy per Apple's contract: "the animation is copied".
        let copied = anim.copy()

        // Resolve defaults at insertion time. Core Animation stores a concrete
        // begin time and duration on the animation copy returned by
        // animation(forKey:), rather than consulting a later transaction.
        let layerLocalTime = convertTime(CACurrentMediaTime(), from: nil)
        if copied.beginTime == 0 {
            copied.beginTime = layerLocalTime
        }
        let defaultDuration = CATransaction.animationDuration()
        prepareAnimationForAddition(copied, inheritedDuration: defaultDuration)

        // Set up animation internal state on the copy.
        copied.isFinished = false
        copied.hasStarted = false
        copied.attachedLayer = self
        copied.animationKey = animKey

        if let transition = copied as? CATransition {
            transition.sourceLayerSnapshot = makeTransitionSnapshot()
        }

        _animations[animKey] = copied
        CATransaction.registerAnimation(copied)
        markDirty(.animations)

    }

    /// Captures the renderable model tree without animations or ownership links.
    private func makeTransitionSnapshot() -> CALayer {
        let snapshot = type(of: self).init(layer: self)
        if let mask {
            snapshot.mask = mask.makeTransitionSnapshot()
        }
        for sublayer in _sublayers ?? [] {
            snapshot.addSublayer(sublayer.makeTransitionSnapshot())
        }
        snapshot.recursivelyClearDirtyAfterCommit()
        return snapshot
    }

    /// Resolves inherited durations throughout an animation group.
    /// Group children whose duration is zero use their group's duration, and
    /// nested groups repeat that rule recursively.
    private func prepareAnimationForAddition(
        _ animation: CAAnimation,
        inheritedDuration: CFTimeInterval
    ) {
        if animation.duration == 0 {
            animation.duration = inheritedDuration
        }
        guard let group = animation as? CAAnimationGroup else { return }
        group.animations?.forEach {
            prepareAnimationForAddition($0, inheritedDuration: group.duration)
        }
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
        markDirty(.animations)
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
        markDirty(.animations)
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
        let currentTime = convertTime(CACurrentMediaTime(), from: nil)
        var keysToRemove: [String] = []

        for (key, animation) in _animations {
            guard !animation.isFinished else {
                // Already finished, check if it should be removed
                if animation.isRemovedOnCompletion {
                    keysToRemove.append(key)
                }
                continue
            }

            let duration = animation.duration > 0 ? animation.duration : animation.effectiveBaseDuration
            let timing = CAMediaTimingEvaluator.evaluate(
                animation,
                parentTime: currentTime,
                duration: duration
            )

            if timing.phase == .after {
                // Animation has completed
                animation.markStarted()
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
        if !keysToRemove.isEmpty {
            markDirty(.animations)
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
        guard let superlayer, !autoresizingMask.isEmpty else { return }

        var resizedFrame = frame
        let widthDelta = superlayer.bounds.width - size.width
        let heightDelta = superlayer.bounds.height - size.height

        let horizontalFlexibleCount = [
            CAAutoresizingMask.layerMinXMargin,
            .layerWidthSizable,
            .layerMaxXMargin
        ].reduce(into: 0) { count, option in
            if autoresizingMask.contains(option) { count += 1 }
        }
        if horizontalFlexibleCount > 0 {
            let share = widthDelta / CGFloat(horizontalFlexibleCount)
            if autoresizingMask.contains(.layerMinXMargin) {
                resizedFrame.origin.x += share
            }
            if autoresizingMask.contains(.layerWidthSizable) {
                resizedFrame.size.width += share
            }
        }

        let verticalFlexibleCount = [
            CAAutoresizingMask.layerMinYMargin,
            .layerHeightSizable,
            .layerMaxYMargin
        ].reduce(into: 0) { count, option in
            if autoresizingMask.contains(option) { count += 1 }
        }
        if verticalFlexibleCount > 0 {
            let share = heightDelta / CGFloat(verticalFlexibleCount)
            if autoresizingMask.contains(.layerMinYMargin) {
                resizedFrame.origin.y += share
            }
            if autoresizingMask.contains(.layerHeightSizable) {
                resizedFrame.size.height += share
            }
        }

        frame = resizedFrame.integral
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

    /// Returns the default action for the current class.
    open class func defaultAction(forKey event: String) -> (any CAAction)? {
        nil
    }

    // MARK: - Mapping Between Coordinate and Time Spaces

    private func localToSuperlayerTransform() -> CATransform3D {
        var result = CATransform3DMakeTranslation(
            -_bounds.origin.x - _bounds.size.width * _anchorPoint.x,
            -_bounds.origin.y - _bounds.size.height * _anchorPoint.y,
            -_anchorPointZ
        )
        result = CATransform3DConcat(result, _transform)
        result = CATransform3DConcat(
            result,
            CATransform3DMakeTranslation(_position.x, _position.y, _zPosition)
        )

        if let superlayer = _superlayer {
            result = CATransform3DConcat(
                result,
                CATransform3DMakeTranslation(
                    -superlayer._bounds.origin.x,
                    -superlayer._bounds.origin.y,
                    0
                )
            )
            result = CATransform3DConcat(result, superlayer._sublayerTransform)
            result = CATransform3DConcat(
                result,
                CATransform3DMakeTranslation(
                    superlayer._bounds.origin.x,
                    superlayer._bounds.origin.y,
                    0
                )
            )
        }
        return result
    }

    private struct PlaneProjectiveTransform {
        let m11: CGFloat
        let m12: CGFloat
        let m13: CGFloat
        let m21: CGFloat
        let m22: CGFloat
        let m23: CGFloat
        let m31: CGFloat
        let m32: CGFloat
        let m33: CGFloat

        init(_ transform: CATransform3D) {
            m11 = transform.m11
            m12 = transform.m12
            m13 = transform.m14
            m21 = transform.m21
            m22 = transform.m22
            m23 = transform.m24
            m31 = transform.m41
            m32 = transform.m42
            m33 = transform.m44
        }

        init(
            m11: CGFloat, m12: CGFloat, m13: CGFloat,
            m21: CGFloat, m22: CGFloat, m23: CGFloat,
            m31: CGFloat, m32: CGFloat, m33: CGFloat
        ) {
            self.m11 = m11
            self.m12 = m12
            self.m13 = m13
            self.m21 = m21
            self.m22 = m22
            self.m23 = m23
            self.m31 = m31
            self.m32 = m32
            self.m33 = m33
        }

        func concatenating(_ other: Self) -> Self {
            Self(
                m11: m11 * other.m11 + m12 * other.m21 + m13 * other.m31,
                m12: m11 * other.m12 + m12 * other.m22 + m13 * other.m32,
                m13: m11 * other.m13 + m12 * other.m23 + m13 * other.m33,
                m21: m21 * other.m11 + m22 * other.m21 + m23 * other.m31,
                m22: m21 * other.m12 + m22 * other.m22 + m23 * other.m32,
                m23: m21 * other.m13 + m22 * other.m23 + m23 * other.m33,
                m31: m31 * other.m11 + m32 * other.m21 + m33 * other.m31,
                m32: m31 * other.m12 + m32 * other.m22 + m33 * other.m32,
                m33: m31 * other.m13 + m32 * other.m23 + m33 * other.m33
            )
        }

        func inverted() -> Self? {
            let determinant = m11 * (m22 * m33 - m23 * m32)
                - m12 * (m21 * m33 - m23 * m31)
                + m13 * (m21 * m32 - m22 * m31)
            guard determinant.isFinite,
                  abs(determinant) > 0.000001 else {
                return nil
            }
            let reciprocal = 1 / determinant
            return Self(
                m11: reciprocal * (m22 * m33 - m23 * m32),
                m12: reciprocal * (m13 * m32 - m12 * m33),
                m13: reciprocal * (m12 * m23 - m13 * m22),
                m21: reciprocal * (m23 * m31 - m21 * m33),
                m22: reciprocal * (m11 * m33 - m13 * m31),
                m23: reciprocal * (m13 * m21 - m11 * m23),
                m31: reciprocal * (m21 * m32 - m22 * m31),
                m32: reciprocal * (m12 * m31 - m11 * m32),
                m33: reciprocal * (m11 * m22 - m12 * m21)
            )
        }

        func project(_ point: CGPoint) -> CGPoint {
            let x = point.x * m11 + point.y * m21 + m31
            let y = point.x * m12 + point.y * m22 + m32
            let w = point.x * m13 + point.y * m23 + m33
            guard x.isFinite,
                  y.isFinite,
                  w.isFinite,
                  abs(w) > 0.000001 else {
                return CGPoint(x: CGFloat.nan, y: CGFloat.nan)
            }
            return CGPoint(x: x / w, y: y / w)
        }
    }

    /// Converts a point from this layer's superlayer's coordinates to local coordinates.
    private func convertPointFromSuperlayer(_ p: CGPoint) -> CGPoint {
        guard let inverse = PlaneProjectiveTransform(
            localToSuperlayerTransform()
        ).inverted() else {
            return CGPoint(x: CGFloat.nan, y: CGFloat.nan)
        }
        return inverse.project(p)
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

        var sourceAncestors = Set<ObjectIdentifier>()
        var current: CALayer? = sourceLayer
        while let layer = current {
            sourceAncestors.insert(ObjectIdentifier(layer))
            current = layer._superlayer
        }
        let commonAncestor = ancestorChain().first {
            sourceAncestors.contains(ObjectIdentifier($0))
        }

        func transform(
            from layer: CALayer,
            to ancestor: CALayer?
        ) -> CATransform3D {
            var result = CATransform3DIdentity
            var current: CALayer? = layer
            while let node = current, node !== ancestor {
                result = CATransform3DConcat(result, node.localToSuperlayerTransform())
                current = node._superlayer
            }
            return result
        }

        let sourceToCommon = transform(from: sourceLayer, to: commonAncestor)
        let receiverToCommon = transform(from: self, to: commonAncestor)
        let sourceProjection = PlaneProjectiveTransform(sourceToCommon)
        guard let receiverProjectionInverse = PlaneProjectiveTransform(
            receiverToCommon
        ).inverted() else {
            return CGPoint(x: CGFloat.nan, y: CGFloat.nan)
        }
        return sourceProjection
            .concatenating(receiverProjectionInverse)
            .project(p)
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
        guard let sourceLayer = l else {
            return convertGlobalTimeToLocal(t)
        }
        if sourceLayer === self { return t }

        let selfChain = ancestorChain()
        var sourceAncestors = Set<ObjectIdentifier>()
        var current: CALayer? = sourceLayer
        while let layer = current {
            sourceAncestors.insert(ObjectIdentifier(layer))
            current = layer._superlayer
        }

        let commonAncestor = selfChain.first { sourceAncestors.contains(ObjectIdentifier($0)) }

        var time = t
        current = sourceLayer

        while let layer = current, layer !== commonAncestor {
            let speed = CFTimeInterval(layer.speed)
            if speed != 0 {
                time = (time - layer.timeOffset) / speed + layer.beginTime
            } else {
                time = 0
            }
            current = layer._superlayer
        }

        guard let commonAncestor else {
            return convertGlobalTimeToLocal(time)
        }

        if commonAncestor === self {
            return time
        }

        guard let commonIndex = selfChain.firstIndex(where: { $0 === commonAncestor }) else {
            return convertGlobalTimeToLocal(time)
        }

        for layer in selfChain[..<commonIndex].reversed() {
            if time < layer.beginTime {
                return 0
            }
            time = (time - layer.beginTime) * CFTimeInterval(layer.speed) + layer.timeOffset
        }

        return time
    }

    /// Converts the time interval from the receiver's time space to the specified layer's time space.
    open func convertTime(_ t: CFTimeInterval, to l: CALayer?) -> CFTimeInterval {
        guard let targetLayer = l else {
            return convertLocalTimeToGlobal(t)
        }
        if targetLayer === self { return t }

        return targetLayer.convertTime(t, from: self)
    }

    private func convertLocalTimeToGlobal(_ t: CFTimeInterval) -> CFTimeInterval {
        var time = t
        var current: CALayer? = self

        while let layer = current {
            let speed = CFTimeInterval(layer.speed)
            if speed != 0 {
                time = (time - layer.timeOffset) / speed + layer.beginTime
            } else {
                time = 0
            }
            current = layer._superlayer
        }

        return time
    }

    private func convertGlobalTimeToLocal(_ t: CFTimeInterval) -> CFTimeInterval {
        var time = t
        for layer in ancestorChain().reversed() {
            if time < layer.beginTime {
                return 0
            }
            time = (time - layer.beginTime) * CFTimeInterval(layer.speed) + layer.timeOffset
        }
        return time
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

        // Renderer order is zPosition first and insertion order second. Walk the
        // exact reverse order so input reaches the visually frontmost subtree.
        for sublayer in sortedSublayers().reversed() {
            if let hit = sublayer.hitTest(localPoint) {
                return hit
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
        var ancestor = _superlayer
        while let layer = ancestor {
            if let scrollLayer = layer as? CAScrollLayer {
                return bounds.intersection(convert(scrollLayer.bounds, from: scrollLayer))
            }
            ancestor = layer._superlayer
        }
        return bounds
    }

    /// Initiates a scroll in the layer's closest ancestor scroll layer so that the specified point
    /// lies at the origin of the scroll layer.
    open func scroll(_ p: CGPoint) {
        var ancestor = _superlayer
        while let layer = ancestor {
            if let scrollLayer = layer as? CAScrollLayer {
                scrollLayer.scroll(to: convert(p, to: scrollLayer))
                return
            }
            ancestor = layer._superlayer
        }
    }

    /// Initiates a scroll in the layer's closest ancestor scroll layer so that the specified rectangle
    /// becomes visible.
    open func scrollRectToVisible(_ r: CGRect) {
        var ancestor = _superlayer
        while let layer = ancestor {
            if let scrollLayer = layer as? CAScrollLayer {
                scrollLayer.scroll(to: convert(r, to: scrollLayer))
                return
            }
            ancestor = layer._superlayer
        }
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
