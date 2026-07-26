#if arch(wasm32)
import Foundation

// MARK: - Particle Data Structure

/// Represents a single particle in the emitter system.
internal struct EmitterParticle {
    internal var birthSequence: UInt64 = 0
    internal var generation: Int = 0
    internal var emitterCells: [CAEmitterCellSnapshot] = []
    internal var contents: CAEmitterCellSnapshot.Image?
    internal var position: SIMD3<Float> = .zero
    internal var previousPosition: SIMD3<Float> = .zero
    internal var velocity: SIMD3<Float> = .zero
    internal var emissionDirection: SIMD3<Float> = SIMD3(0, 0, 1)
    internal var acceleration: SIMD3<Float> = .zero
    internal var color: SIMD4<Float> = SIMD4(1, 1, 1, 1)
    internal var previousColor: SIMD4<Float> = SIMD4(1, 1, 1, 1)
    internal var colorSpeed: SIMD4<Float> = .zero
    internal var scale: Float = 1.0
    internal var previousScale: Float = 1.0
    internal var scaleSpeed: Float = 0.0
    internal var rotation: Float = 0.0
    internal var rotationSpeed: Float = 0.0
    internal var lifetime: Float = 0.0
    internal var previousLifetime: Float = 0.0
    internal var maxLifetime: Float = 1.0
    internal var isAlive: Bool = false

    internal init() {}

    /// Updates the particle state for the given time delta.
    internal mutating func update(deltaTime: Float) {
        guard isAlive else { return }

        previousPosition = position
        previousColor = color
        previousScale = scale
        previousLifetime = lifetime

        let step = max(0, min(deltaTime, lifetime))

        // Update position
        velocity += acceleration * step
        position += velocity * step
        let velocityLength = sqrt(
            velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
        )
        if velocityLength > 0 {
            emissionDirection = velocity / velocityLength
        }

        // Update color
        color += colorSpeed * step
        color = SIMD4(
            max(0, min(1, color.x)),
            max(0, min(1, color.y)),
            max(0, min(1, color.z)),
            max(0, min(1, color.w))
        )

        // Update scale
        scale += scaleSpeed * step
        scale = max(0, scale)

        // Update rotation
        rotation += rotationSpeed * step

        lifetime -= step
        if lifetime <= 0 {
            lifetime = 0
            isAlive = false
        }
    }
}

/// Blur uniform data.
public struct BlurUniforms {
    public var texelSize: SIMD2<Float>
    public var blurRadius: Float
    public var padding: Float = 0

    public init(texelSize: SIMD2<Float>, blurRadius: Float) {
        self.texelSize = texelSize
        self.blurRadius = blurRadius
    }
}

/// Shadow uniform data.
public struct ShadowUniforms {
    public var mvpMatrix: Matrix4x4
    public var shadowColor: SIMD4<Float>
    public var shadowOffset: SIMD2<Float>
    public var layerSize: SIMD2<Float>

    public init(
        mvpMatrix: Matrix4x4 = .identity,
        shadowColor: SIMD4<Float> = SIMD4(0, 0, 0, 1),
        shadowOffset: SIMD2<Float> = .zero,
        layerSize: SIMD2<Float> = .zero
    ) {
        self.mvpMatrix = mvpMatrix
        self.shadowColor = shadowColor
        self.shadowOffset = shadowOffset
        self.layerSize = layerSize
    }
}

#endif
