// CARenderer.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework

import Foundation
#if arch(wasm32)
import JavaScriptKit
#endif

/// Errors that can occur during renderer operations.
public enum CARendererError: Error, Equatable, Sendable {
    /// The GPU/graphics device is not available.
    case deviceNotAvailable
    /// Failed to create the render pipeline.
    case pipelineCreationFailed
    /// Failed to create shader module.
    case shaderCompilationFailed(String)
    /// Failed to create buffer.
    case bufferCreationFailed
    /// Failed to create texture.
    case textureCreationFailed
    /// Failed to create GPU resource.
    case resourceCreationFailed
    /// The canvas/view is not configured.
    case canvasNotConfigured
    /// The requested render target cannot be represented by the renderer.
    case invalidRenderTarget(CARenderTargetConfigurationError)
    /// General rendering error.
    case renderingFailed(String)
}

/// Supplies a deterministic media time while a renderer evaluates a frame.
internal enum CARenderTimeContext {
    @TaskLocal internal static var mediaTime: CFTimeInterval?

    internal static var currentMediaTime: CFTimeInterval {
        mediaTime ?? CACurrentMediaTime()
    }
}

/// Renders a layer tree into an explicit destination through a frame lifecycle.
@MainActor public final class CARenderer {
    /// The root layer of the layer tree rendered by the receiver.
    public var layer: CALayer?

    /// The destination region used to clip automatically discovered updates.
    public var bounds: CGRect = CARenderer.nullRect

    internal let backend: any CARendererDelegate
    private var frameTime: CFTimeInterval = 0
    private var hasFrameTime = false
    private var frameIsOpen = false
    private var updateRegion: CGRect = CARenderer.nullRect
    private var previousRenderedExtent: CGRect = CARenderer.nullRect

    internal init(backend: any CARendererDelegate) {
        self.backend = backend
    }

    #if arch(wasm32)
    /// Creates and initializes a WebGPU-backed layer renderer for a browser canvas.
    public convenience init(canvas: JavaScriptKit.JSObject) async throws {
        let webGPUBackend = CAWebGPURenderer(canvas: canvas)
        try await webGPUBackend.initialize()
        self.init(backend: webGPUBackend)
    }

    /// Resizes the WebGPU destination while preserving the logical renderer bounds.
    public func resize(width: Int, height: Int) {
        backend.resize(width: width, height: height)
    }
    #endif

    internal func beginFrame(atTime time: CFTimeInterval) {
        frameTime = time
        hasFrameTime = true
        frameIsOpen = true
        updateRegion = automaticUpdateRegion(at: time)
    }

    #if !canImport(CoreVideo)
    /// Begins evaluation of a frame at the supplied layer media time.
    public func beginFrame(
        atTime time: CFTimeInterval,
        timeStamp: UnsafeMutablePointer<CVTimeStamp>?
    ) {
        beginFrame(atTime: time)
    }
    #endif

    /// Adds a rectangle to the current frame's update region.
    public func addUpdate(_ rect: CGRect) {
        guard frameIsOpen, !Self.isNull(rect) else { return }
        updateRegion = Self.isNull(updateRegion) ? rect : Self.union(updateRegion, rect)
    }

    /// Returns the region containing the pixels updated by the current frame.
    public func updateBounds() -> CGRect {
        updateRegion
    }

    /// Renders the current update region through the configured GPU backend.
    public func render() {
        guard frameIsOpen, !Self.isNull(updateRegion), let layer else { return }
        layoutRecursively(layer)
        CARenderTimeContext.$mediaTime.withValue(frameTime) {
            backend.render(layer: layer)
            processAnimationCompletionsRecursively(layer)
        }
        previousRenderedExtent = layerTreeExtent(layer)
    }

    /// Returns the media time at which the next animation update is required.
    public func nextFrameTime() -> CFTimeInterval {
        guard hasFrameTime, let layer else { return .infinity }
        return nextAnimationTime(in: layer, at: frameTime)
    }

    /// Releases state associated with the current frame.
    public func endFrame() {
        frameIsOpen = false
        updateRegion = Self.nullRect
    }

    private func automaticUpdateRegion(at mediaTime: CFTimeInterval) -> CGRect {
        guard let layer, !Self.isNull(bounds) else { return Self.nullRect }
        let scheduling = animationSchedule(in: layer, at: mediaTime)
        guard layer._subtreeDirtyCount > 0 || scheduling.hasActiveAnimation else {
            return Self.nullRect
        }
        if scheduling.hasActiveAnimation || layerTreeHasUnboundedEffects(layer) {
            return bounds
        }
        let currentExtent = layerTreeExtent(layer)
        let changedExtent = Self.isNull(previousRenderedExtent)
            ? currentExtent
            : Self.union(previousRenderedExtent, currentExtent)
        return Self.intersection(changedExtent, bounds)
    }

    private func layerTreeExtent(_ rootLayer: CALayer) -> CGRect {
        let ownExtent = rootLayer.convert(rootLayer.bounds, to: nil)
        guard !rootLayer.masksToBounds else { return ownExtent }

        var extent = ownExtent
        for child in rootLayer.sublayers ?? [] {
            let childExtent = layerTreeExtent(child)
            if !Self.isNull(childExtent) {
                extent = Self.isNull(extent) ? childExtent : Self.union(extent, childExtent)
            }
        }
        return extent
    }

    private func layerTreeHasUnboundedEffects(_ rootLayer: CALayer) -> Bool {
        if rootLayer.shadowOpacity > 0
            || !(rootLayer.filters?.isEmpty ?? true)
            || rootLayer.compositingFilter != nil
            || !(rootLayer.backgroundFilters?.isEmpty ?? true) {
            return true
        }
        return (rootLayer.sublayers ?? []).contains { layerTreeHasUnboundedEffects($0) }
    }

    private func nextAnimationTime(
        in layer: CALayer,
        at mediaTime: CFTimeInterval
    ) -> CFTimeInterval {
        animationSchedule(in: layer, at: mediaTime).nextTime
    }

    private func animationSchedule(
        in rootLayer: CALayer,
        at mediaTime: CFTimeInterval
    ) -> (hasActiveAnimation: Bool, nextTime: CFTimeInterval) {
        var active = false
        var nextTime = CFTimeInterval.infinity

        func visit(_ layer: CALayer) {
            let parentTime = layer.convertTime(mediaTime, from: nil)
            var futureBeginTimes: [CFTimeInterval] = []
            layer.forEachAttachedAnimation { animation in
                guard !animation.isFinished else { return }
                let duration = animation.duration > 0
                    ? animation.duration
                    : animation.effectiveBaseDuration
                let timing = CAMediaTimingEvaluator.evaluate(
                    animation,
                    parentTime: parentTime,
                    duration: duration
                )
                switch timing.phase {
                case .active:
                    if animation.speed != 0 {
                        active = true
                        nextTime = min(nextTime, mediaTime)
                    }
                case .before:
                    futureBeginTimes.append(animation.beginTime)
                case .after:
                    break
                }
            }
            for beginTime in futureBeginTimes {
                let globalBeginTime = layer.convertTime(beginTime, to: nil)
                if globalBeginTime.isFinite, globalBeginTime >= mediaTime {
                    nextTime = min(nextTime, globalBeginTime)
                }
            }
            for child in layer.sublayers ?? [] {
                visit(child)
            }
        }

        visit(rootLayer)
        return (active, nextTime)
    }

    private func layoutRecursively(_ layer: CALayer) {
        layer.layoutIfNeeded()
        for child in layer.sublayers ?? [] {
            layoutRecursively(child)
        }
    }

    private func processAnimationCompletionsRecursively(_ layer: CALayer) {
        layer.processAnimationCompletions()
        for child in layer.sublayers ?? [] {
            processAnimationCompletionsRecursively(child)
        }
    }

    private static var nullRect: CGRect {
        CGRect(x: CGFloat.infinity, y: CGFloat.infinity, width: 0, height: 0)
    }

    private static func isNull(_ rect: CGRect) -> Bool {
        rect.origin.x == .infinity && rect.origin.y == .infinity
    }

    private static func union(_ lhs: CGRect, _ rhs: CGRect) -> CGRect {
        let lhsMinX = min(lhs.origin.x, lhs.origin.x + lhs.size.width)
        let lhsMaxX = max(lhs.origin.x, lhs.origin.x + lhs.size.width)
        let lhsMinY = min(lhs.origin.y, lhs.origin.y + lhs.size.height)
        let lhsMaxY = max(lhs.origin.y, lhs.origin.y + lhs.size.height)
        let rhsMinX = min(rhs.origin.x, rhs.origin.x + rhs.size.width)
        let rhsMaxX = max(rhs.origin.x, rhs.origin.x + rhs.size.width)
        let rhsMinY = min(rhs.origin.y, rhs.origin.y + rhs.size.height)
        let rhsMaxY = max(rhs.origin.y, rhs.origin.y + rhs.size.height)
        let minX = min(lhsMinX, rhsMinX)
        let maxX = max(lhsMaxX, rhsMaxX)
        let minY = min(lhsMinY, rhsMinY)
        let maxY = max(lhsMaxY, rhsMaxY)
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private static func intersection(_ lhs: CGRect, _ rhs: CGRect) -> CGRect {
        let minX = max(
            min(lhs.origin.x, lhs.origin.x + lhs.size.width),
            min(rhs.origin.x, rhs.origin.x + rhs.size.width)
        )
        let maxX = min(
            max(lhs.origin.x, lhs.origin.x + lhs.size.width),
            max(rhs.origin.x, rhs.origin.x + rhs.size.width)
        )
        let minY = max(
            min(lhs.origin.y, lhs.origin.y + lhs.size.height),
            min(rhs.origin.y, rhs.origin.y + rhs.size.height)
        )
        let maxY = min(
            max(lhs.origin.y, lhs.origin.y + lhs.size.height),
            max(rhs.origin.y, rhs.origin.y + rhs.size.height)
        )
        guard maxX > minX, maxY > minY else { return nullRect }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}

// MARK: - SIMD-dependent types and helpers (Apple platforms only)

#if canImport(simd)
import simd

/// A structure representing a vertex for the native Metal layer backend.
internal struct CAMetalRendererVertex {
    /// Position in normalized device coordinates.
    internal var position: SIMD2<Float>
    /// Texture coordinates (0-1 range).
    internal var texCoord: SIMD2<Float>
    /// Vertex color (RGBA).
    internal var color: SIMD4<Float>

    internal init(position: SIMD2<Float>, texCoord: SIMD2<Float>, color: SIMD4<Float>) {
        self.position = position
        self.texCoord = texCoord
        self.color = color
    }
}

/// Uniform data passed to the native Metal shaders for each layer.
internal struct CAMetalRendererUniforms {
    /// Model-view-projection matrix.
    internal var mvpMatrix: simd_float4x4
    /// Layer opacity (0-1).
    internal var opacity: Float
    /// Corner radius in pixels.
    internal var cornerRadius: Float
    /// Padding for alignment.
    internal var padding: SIMD2<Float>

    internal init(
        mvpMatrix: simd_float4x4 = matrix_identity_float4x4,
        opacity: Float = 1.0,
        cornerRadius: Float = 0.0
    ) {
        self.mvpMatrix = mvpMatrix
        self.opacity = opacity
        self.cornerRadius = cornerRadius
        self.padding = .zero
    }
}

// MARK: - Helper Extensions

extension CALayer {
    /// Converts the layer's background color to SIMD4<Float>.
    internal var backgroundColorComponents: SIMD4<Float> {
        guard let color = backgroundColor,
              let components = color.components,
              components.count >= 4 else {
            return SIMD4<Float>(0, 0, 0, 0)
        }
        return SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
    }

    /// Calculates the model matrix for this layer.
    internal func modelMatrix(parentMatrix: simd_float4x4 = matrix_identity_float4x4) -> simd_float4x4 {
        // Start with translation to position
        var matrix = parentMatrix

        // Translate to layer position
        let translation = simd_float4x4(translation: SIMD3<Float>(
            Float(position.x),
            Float(position.y),
            Float(zPosition)
        ))
        matrix = matrix * translation

        // Apply layer transform if not identity
        if !CATransform3DIsIdentity(transform) {
            let layerTransform = transform.simdMatrix
            matrix = matrix * layerTransform
        }

        // Translate by anchor point offset
        let anchorOffset = simd_float4x4(translation: SIMD3<Float>(
            Float(-bounds.width * anchorPoint.x),
            Float(-bounds.height * anchorPoint.y),
            Float(-anchorPointZ)
        ))
        matrix = matrix * anchorOffset

        return matrix
    }

    /// Calculates the parent matrix for sublayer positioning.
    ///
    /// This includes the layer's sublayerTransform and bounds.origin offset.
    /// The bounds.origin offset ensures that sublayers are correctly positioned when the
    /// layer's bounds origin is non-zero (e.g., for CAScrollLayer scrolling).
    ///
    /// - Parameter modelMatrix: The layer's model matrix
    /// - Returns: The matrix to use as parentMatrix for sublayer rendering
    internal func sublayerMatrix(modelMatrix: simd_float4x4) -> simd_float4x4 {
        var result = modelMatrix

        // Apply sublayerTransform if not identity
        if !CATransform3DIsIdentity(sublayerTransform) {
            result = result * sublayerTransform.simdMatrix
        }

        // Apply bounds.origin offset
        // In CoreAnimation, bounds.origin defines where the coordinate system origin is
        // within the layer. A sublayer at position (0,0) with parent's bounds.origin = (50, 50)
        // should appear at (-50, -50) relative to the parent's visible top-left.
        // This is the scrolling behavior used by CAScrollLayer.
        if bounds.origin.x != 0 || bounds.origin.y != 0 {
            let boundsOriginOffset = simd_float4x4(translation: SIMD3<Float>(
                Float(-bounds.origin.x),
                Float(-bounds.origin.y),
                0
            ))
            result = result * boundsOriginOffset
        }

        return result
    }
}

extension CATransform3D {
    /// Converts CATransform3D to simd_float4x4.
    internal var simdMatrix: simd_float4x4 {
        // CATransform3D uses row-vector semantics (translation is m41/m42/m43).
        // Transpose its rows into SIMD columns so matrix * columnVector is equivalent.
        return simd_float4x4(columns: (
            SIMD4<Float>(Float(m11), Float(m12), Float(m13), Float(m14)),
            SIMD4<Float>(Float(m21), Float(m22), Float(m23), Float(m24)),
            SIMD4<Float>(Float(m31), Float(m32), Float(m33), Float(m34)),
            SIMD4<Float>(Float(m41), Float(m42), Float(m43), Float(m44))
        ))
    }
}

extension simd_float4x4 {
    /// Creates a translation matrix.
    internal init(translation: SIMD3<Float>) {
        self = matrix_identity_float4x4
        self.columns.3 = SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    }

    /// Creates an orthographic projection matrix.
    internal static func orthographic(
        left: Float,
        right: Float,
        bottom: Float,
        top: Float,
        near: Float,
        far: Float
    ) -> simd_float4x4 {
        let width = right - left
        let height = top - bottom
        let depth = far - near

        // simd_float4x4 is column-major, so we construct from columns.
        // Metal uses [0,1] depth range convention (not [-1,1] like OpenGL),
        // so Z column uses 1/depth and translation uses -near/depth.
        return simd_float4x4(columns: (
            SIMD4<Float>(2 / width, 0, 0, 0),
            SIMD4<Float>(0, 2 / height, 0, 0),
            SIMD4<Float>(0, 0, 1 / depth, 0),
            SIMD4<Float>(-(right + left) / width, -(top + bottom) / height, -near / depth, 1)
        ))
    }
}

#endif
