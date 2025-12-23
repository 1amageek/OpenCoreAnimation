#if arch(wasm32)
import Foundation

// MARK: - Particle Data Structure

/// Represents a single particle in the emitter system.
public struct EmitterParticle {
    public var position: SIMD3<Float> = .zero
    public var velocity: SIMD3<Float> = .zero
    public var acceleration: SIMD3<Float> = .zero
    public var color: SIMD4<Float> = SIMD4(1, 1, 1, 1)
    public var colorSpeed: SIMD4<Float> = .zero
    public var scale: Float = 1.0
    public var scaleSpeed: Float = 0.0
    public var rotation: Float = 0.0
    public var rotationSpeed: Float = 0.0
    public var lifetime: Float = 0.0
    public var maxLifetime: Float = 1.0
    public var isAlive: Bool = false

    public init() {}

    /// Updates the particle state for the given time delta.
    public mutating func update(deltaTime: Float) {
        guard isAlive else { return }

        lifetime -= deltaTime
        if lifetime <= 0 {
            isAlive = false
            return
        }

        // Update position
        velocity += acceleration * deltaTime
        position += velocity * deltaTime

        // Update color
        color += colorSpeed * deltaTime
        color = SIMD4(
            max(0, min(1, color.x)),
            max(0, min(1, color.y)),
            max(0, min(1, color.z)),
            max(0, min(1, color.w))
        )

        // Update scale
        scale += scaleSpeed * deltaTime
        scale = max(0, scale)

        // Update rotation
        rotation += rotationSpeed * deltaTime
    }
}

/// GPU-compatible particle instance data.
public struct ParticleInstanceData {
    public var position: SIMD3<Float>
    public var color: SIMD4<Float>
    public var scaleRotation: SIMD2<Float>

    public init(from particle: EmitterParticle) {
        self.position = particle.position
        self.color = particle.color
        self.scaleRotation = SIMD2(particle.scale, particle.rotation)
    }

    public static var stride: UInt64 {
        return UInt64(MemoryLayout<ParticleInstanceData>.stride)
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
