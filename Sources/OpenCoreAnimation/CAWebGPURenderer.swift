#if arch(wasm32)
import JavaScriptKit
import SwiftWebGPU

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

// MARK: - WASM Matrix Types (simd replacement)

/// A 4x4 matrix of Float values for WASM environments.
/// This replaces simd_float4x4 which is not available on WASM.
public struct Matrix4x4 {
    public var columns: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)

    public init(columns: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)) {
        self.columns = columns
    }

    /// Identity matrix
    public static var identity: Matrix4x4 {
        Matrix4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }

    /// Creates a translation matrix.
    public init(translation: SIMD3<Float>) {
        self = .identity
        self.columns.3 = SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    }

    /// Creates an orthographic projection matrix for WebGPU (depth range 0 to 1).
    ///
    /// WebGPU uses a depth range of [0, 1] in clip space, unlike OpenGL which uses [-1, 1].
    /// This matrix maps:
    /// - X: [left, right] → [-1, 1]
    /// - Y: [bottom, top] → [-1, 1]
    /// - Z: [near, far] → [0, 1]
    public static func orthographic(
        left: Float,
        right: Float,
        bottom: Float,
        top: Float,
        near: Float,
        far: Float
    ) -> Matrix4x4 {
        let width = right - left
        let height = top - bottom
        let depth = far - near

        // WebGPU depth range [0, 1]:
        // z_ndc = (z_eye - near) / (far - near)
        //       = z_eye / depth - near / depth
        return Matrix4x4(columns: (
            SIMD4<Float>(2 / width, 0, 0, 0),
            SIMD4<Float>(0, 2 / height, 0, 0),
            SIMD4<Float>(0, 0, 1 / depth, 0),
            SIMD4<Float>(-(right + left) / width, -(top + bottom) / height, -near / depth, 1)
        ))
    }

    /// Matrix multiplication
    public static func * (lhs: Matrix4x4, rhs: Matrix4x4) -> Matrix4x4 {
        var result = Matrix4x4.identity

        for i in 0..<4 {
            let col = getColumn(rhs, i)
            let x = lhs.columns.0 * col.x
            let y = lhs.columns.1 * col.y
            let z = lhs.columns.2 * col.z
            let w = lhs.columns.3 * col.w
            setColumn(&result, i, x + y + z + w)
        }

        return result
    }

    private static func getColumn(_ m: Matrix4x4, _ i: Int) -> SIMD4<Float> {
        switch i {
        case 0: return m.columns.0
        case 1: return m.columns.1
        case 2: return m.columns.2
        case 3: return m.columns.3
        default: return .zero
        }
    }

    private static func setColumn(_ m: inout Matrix4x4, _ i: Int, _ v: SIMD4<Float>) {
        switch i {
        case 0: m.columns.0 = v
        case 1: m.columns.1 = v
        case 2: m.columns.2 = v
        case 3: m.columns.3 = v
        default: break
        }
    }
}

// MARK: - WASM Renderer Types

/// A structure representing a vertex for layer rendering (WASM version).
public struct CARendererVertex {
    public var position: SIMD2<Float>
    public var texCoord: SIMD2<Float>
    public var color: SIMD4<Float>

    public init(position: SIMD2<Float>, texCoord: SIMD2<Float>, color: SIMD4<Float>) {
        self.position = position
        self.texCoord = texCoord
        self.color = color
    }
}

/// Maximum number of gradient color stops supported.
public let kMaxGradientStops: Int = 8

/// Uniform data passed to shaders for each layer (WASM version).
public struct CARendererUniforms {
    public var mvpMatrix: Matrix4x4
    public var opacity: Float
    public var cornerRadius: Float
    public var layerSize: SIMD2<Float>
    public var borderWidth: Float
    public var renderMode: Float  // 0 = fill, 1 = border, 2 = gradient
    public var gradientStartPoint: SIMD2<Float>
    public var gradientEndPoint: SIMD2<Float>
    public var gradientColorCount: Float
    public var padding3: SIMD3<Float>
    // Gradient color stops: each is vec4 color + vec4 (location, 0, 0, 0) = 8 bytes per stop
    // For simplicity, we'll pack 8 colors and 8 locations separately
    public var gradientColors: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>,
                                SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    public var gradientLocations: SIMD4<Float>  // First 4 locations
    public var gradientLocations2: SIMD4<Float> // Next 4 locations

    public init(
        mvpMatrix: Matrix4x4 = .identity,
        opacity: Float = 1.0,
        cornerRadius: Float = 0.0,
        layerSize: SIMD2<Float> = .zero,
        borderWidth: Float = 0.0,
        renderMode: Float = 0.0,
        gradientStartPoint: SIMD2<Float> = .zero,
        gradientEndPoint: SIMD2<Float> = SIMD2(0, 1),
        gradientColorCount: Float = 0
    ) {
        self.mvpMatrix = mvpMatrix
        self.opacity = opacity
        self.cornerRadius = cornerRadius
        self.layerSize = layerSize
        self.borderWidth = borderWidth
        self.renderMode = renderMode
        self.gradientStartPoint = gradientStartPoint
        self.gradientEndPoint = gradientEndPoint
        self.gradientColorCount = gradientColorCount
        self.padding3 = .zero
        self.gradientColors = (.zero, .zero, .zero, .zero, .zero, .zero, .zero, .zero)
        self.gradientLocations = .zero
        self.gradientLocations2 = .zero
    }
}

/// Uniform data for textured layer rendering.
public struct TexturedUniforms {
    public var mvpMatrix: Matrix4x4
    public var opacity: Float
    public var cornerRadius: Float
    public var layerSize: SIMD2<Float>

    public init(
        mvpMatrix: Matrix4x4 = .identity,
        opacity: Float = 1.0,
        cornerRadius: Float = 0.0,
        layerSize: SIMD2<Float> = .zero
    ) {
        self.mvpMatrix = mvpMatrix
        self.opacity = opacity
        self.cornerRadius = cornerRadius
        self.layerSize = layerSize
    }
}

// MARK: - WASM Helper Extensions

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

    /// Converts the layer's border color to SIMD4<Float>.
    internal var borderColorComponents: SIMD4<Float> {
        guard let color = borderColor,
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
    ///
    /// The z-coordinate is negated to match CoreAnimation's convention where
    /// higher zPosition values appear "in front" (closer to the viewer).
    /// With WebGPU's depth range [0, 1] and lessEqual comparison:
    /// - Lower z in eye space → lower clip z → passes depth test → in front
    /// - So we negate zPosition: higher zPosition → lower z_eye → in front
    internal func modelMatrix(parentMatrix: Matrix4x4 = .identity) -> Matrix4x4 {
        var matrix = parentMatrix

        // Negate zPosition so higher values appear in front (CoreAnimation convention)
        let translation = Matrix4x4(translation: SIMD3<Float>(
            Float(position.x),
            Float(position.y),
            Float(-zPosition)  // Negated for correct z-ordering
        ))
        matrix = matrix * translation

        if !CATransform3DIsIdentity(transform) {
            let layerTransform = transform.matrix4x4
            matrix = matrix * layerTransform
        }

        // Negate anchorPointZ to match the z-coordinate convention
        let anchorOffset = Matrix4x4(translation: SIMD3<Float>(
            Float(-bounds.width * anchorPoint.x),
            Float(-bounds.height * anchorPoint.y),
            Float(anchorPointZ)  // Negated (double negation with the minus sign)
        ))
        matrix = matrix * anchorOffset

        return matrix
    }
}

extension CATransform3D {
    /// Converts CATransform3D to Matrix4x4.
    internal var matrix4x4: Matrix4x4 {
        return Matrix4x4(columns: (
            SIMD4<Float>(Float(m11), Float(m21), Float(m31), Float(m41)),
            SIMD4<Float>(Float(m12), Float(m22), Float(m32), Float(m42)),
            SIMD4<Float>(Float(m13), Float(m23), Float(m33), Float(m43)),
            SIMD4<Float>(Float(m14), Float(m24), Float(m34), Float(m44))
        ))
    }
}

// MARK: - Buffer Pool (Triple Buffering)

/// A pool of GPU buffers for triple buffering.
///
/// Triple buffering prevents GPU stalls by allowing the CPU to write to one buffer
/// while the GPU is reading from another. This class manages a ring buffer of GPU buffers
/// that cycle each frame.
///
/// ## Usage
///
/// ```swift
/// let pool = GPUBufferPool(device: device, bufferSize: 1024, usage: [.vertex, .copyDst], bufferCount: 3)
///
/// // Each frame
/// let buffer = pool.currentBuffer
/// pool.advanceFrame()
/// ```
public final class GPUBufferPool {

    // MARK: - Properties

    /// The GPU buffers in the pool.
    private var buffers: [GPUBuffer]

    /// The current buffer index.
    private var currentIndex: Int = 0

    /// Number of buffers in the pool.
    public let bufferCount: Int

    /// Size of each buffer in bytes.
    public let bufferSize: UInt64

    /// The current frame number (incremented each advanceFrame call).
    public private(set) var frameNumber: UInt64 = 0

    // MARK: - Initialization

    /// Creates a new buffer pool with the specified configuration.
    ///
    /// - Parameters:
    ///   - device: The GPU device to create buffers on.
    ///   - bufferSize: The size of each buffer in bytes.
    ///   - usage: The usage flags for the buffers.
    ///   - bufferCount: The number of buffers in the pool (default: 3 for triple buffering).
    public init(device: GPUDevice, bufferSize: UInt64, usage: GPUBufferUsage, bufferCount: Int = 3) {
        self.bufferSize = bufferSize
        self.bufferCount = bufferCount
        self.buffers = []

        for _ in 0..<bufferCount {
            let buffer = device.createBuffer(descriptor: GPUBufferDescriptor(
                size: bufferSize,
                usage: usage
            ))
            buffers.append(buffer)
        }
    }

    // MARK: - Public Methods

    /// The current buffer to use for this frame.
    public var currentBuffer: GPUBuffer {
        return buffers[currentIndex]
    }

    /// The buffer at the specified offset from the current buffer.
    ///
    /// - Parameter offset: Offset from current buffer (0 = current, 1 = next, etc.)
    /// - Returns: The buffer at the specified offset, wrapping around if necessary.
    public func buffer(at offset: Int) -> GPUBuffer {
        let index = (currentIndex + offset) % bufferCount
        return buffers[index]
    }

    /// Advances to the next buffer in the pool.
    ///
    /// Call this at the end of each frame to cycle to the next buffer.
    public func advanceFrame() {
        currentIndex = (currentIndex + 1) % bufferCount
        frameNumber += 1
    }

    /// Resets the pool to its initial state.
    public func reset() {
        currentIndex = 0
        frameNumber = 0
    }

    /// Destroys all buffers in the pool.
    public func invalidate() {
        buffers.removeAll()
        currentIndex = 0
        frameNumber = 0
    }
}

/// A pool of uniform buffers with bind group management.
///
/// This specialized pool manages uniform buffers along with their bind groups,
/// which is the common pattern for per-frame uniform data in WebGPU.
public final class UniformBufferPool {

    // MARK: - Properties

    /// The buffer pool for uniform data.
    private let bufferPool: GPUBufferPool

    /// The bind groups for each buffer in the pool.
    private var bindGroups: [GPUBindGroup]

    /// The bind group layout used for creating bind groups.
    public let bindGroupLayout: GPUBindGroupLayout

    // MARK: - Initialization

    /// Creates a new uniform buffer pool.
    ///
    /// - Parameters:
    ///   - device: The GPU device.
    ///   - bufferSize: The size of each uniform buffer in bytes.
    ///   - bindGroupLayout: The layout for creating bind groups.
    ///   - bindingSize: The size of the buffer binding (may be smaller than bufferSize for dynamic offsets).
    ///   - bufferCount: Number of buffers in the pool (default: 3).
    public init(
        device: GPUDevice,
        bufferSize: UInt64,
        bindGroupLayout: GPUBindGroupLayout,
        bindingSize: UInt64,
        bufferCount: Int = 3
    ) {
        self.bindGroupLayout = bindGroupLayout
        self.bufferPool = GPUBufferPool(
            device: device,
            bufferSize: bufferSize,
            usage: [.uniform, .copyDst],
            bufferCount: bufferCount
        )
        self.bindGroups = []

        // Create bind groups for each buffer
        for i in 0..<bufferCount {
            let buffer = bufferPool.buffer(at: i)
            let bindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                layout: bindGroupLayout,
                entries: [
                    GPUBindGroupEntry(
                        binding: 0,
                        resource: .bufferBinding(GPUBufferBinding(
                            buffer: buffer,
                            size: bindingSize
                        ))
                    )
                ]
            ))
            bindGroups.append(bindGroup)
        }
    }

    // MARK: - Public Methods

    /// The current uniform buffer for this frame.
    public var currentBuffer: GPUBuffer {
        return bufferPool.currentBuffer
    }

    /// The current bind group for this frame.
    public var currentBindGroup: GPUBindGroup {
        return bindGroups[currentIndex]
    }

    /// Current buffer index.
    private var currentIndex: Int {
        return Int(bufferPool.frameNumber % UInt64(bufferPool.bufferCount))
    }

    /// Advances to the next buffer/bind group pair.
    public func advanceFrame() {
        bufferPool.advanceFrame()
    }

    /// Resets the pool to its initial state.
    public func reset() {
        bufferPool.reset()
    }

    /// Destroys all resources in the pool.
    public func invalidate() {
        bufferPool.invalidate()
        bindGroups.removeAll()
    }
}

/// A pool of vertex buffers for efficient vertex data management.
public final class VertexBufferPool {

    // MARK: - Properties

    /// The buffer pool for vertex data.
    private let bufferPool: GPUBufferPool

    // MARK: - Initialization

    /// Creates a new vertex buffer pool.
    ///
    /// - Parameters:
    ///   - device: The GPU device.
    ///   - bufferSize: The size of each vertex buffer in bytes.
    ///   - bufferCount: Number of buffers in the pool (default: 3).
    public init(device: GPUDevice, bufferSize: UInt64, bufferCount: Int = 3) {
        self.bufferPool = GPUBufferPool(
            device: device,
            bufferSize: bufferSize,
            usage: [.vertex, .copyDst],
            bufferCount: bufferCount
        )
    }

    // MARK: - Public Methods

    /// The current vertex buffer for this frame.
    public var currentBuffer: GPUBuffer {
        return bufferPool.currentBuffer
    }

    /// Advances to the next vertex buffer.
    public func advanceFrame() {
        bufferPool.advanceFrame()
    }

    /// Resets the pool to its initial state.
    public func reset() {
        bufferPool.reset()
    }

    /// Destroys all buffers in the pool.
    public func invalidate() {
        bufferPool.invalidate()
    }
}

// MARK: - Texture Manager (LRU Cache)

/// A texture cache entry with access tracking for LRU eviction.
private struct TextureCacheEntry {
    let texture: GPUTexture
    let width: Int
    let height: Int
    var lastAccessFrame: UInt64
    var accessCount: UInt64

    /// Approximate memory usage in bytes (4 bytes per pixel for RGBA8).
    var memorySize: UInt64 {
        return UInt64(width * height * 4)
    }
}

/// A texture manager with LRU (Least Recently Used) cache eviction.
///
/// This class manages GPU textures with automatic memory management.
/// When the cache exceeds its capacity, the least recently used textures
/// are evicted to make room for new ones.
///
/// ## Usage
///
/// ```swift
/// let manager = GPUTextureManager(device: device, maxTextures: 256, maxMemory: 256 * 1024 * 1024)
///
/// // Get or create a texture
/// let texture = manager.getOrCreateTexture(for: imageId) { device in
///     return createTextureFromImage(image, device: device)
/// }
///
/// // At end of frame, update frame counter
/// manager.advanceFrame()
/// ```
public final class GPUTextureManager {

    // MARK: - Properties

    /// The GPU device for creating textures.
    private weak var device: GPUDevice?

    /// Cache of textures keyed by ObjectIdentifier.
    private var cache: [ObjectIdentifier: TextureCacheEntry] = [:]

    /// Current frame number for LRU tracking.
    private var currentFrame: UInt64 = 0

    /// Maximum number of textures in the cache.
    public let maxTextures: Int

    /// Maximum total memory in bytes for cached textures.
    public let maxMemoryBytes: UInt64

    /// Current number of textures in the cache.
    public var textureCount: Int {
        return cache.count
    }

    /// Current total memory usage in bytes.
    public private(set) var currentMemoryBytes: UInt64 = 0

    /// Number of cache hits since creation.
    public private(set) var cacheHits: UInt64 = 0

    /// Number of cache misses since creation.
    public private(set) var cacheMisses: UInt64 = 0

    /// Cache hit rate (0.0 to 1.0).
    public var hitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0.0
    }

    // MARK: - Initialization

    /// Creates a new texture manager.
    ///
    /// - Parameters:
    ///   - device: The GPU device for creating textures.
    ///   - maxTextures: Maximum number of textures to cache (default: 256).
    ///   - maxMemoryBytes: Maximum memory in bytes (default: 256MB).
    public init(device: GPUDevice, maxTextures: Int = 256, maxMemoryBytes: UInt64 = 256 * 1024 * 1024) {
        self.device = device
        self.maxTextures = maxTextures
        self.maxMemoryBytes = maxMemoryBytes
    }

    // MARK: - Public Methods

    /// Gets a cached texture or creates a new one using the provided factory.
    ///
    /// - Parameters:
    ///   - key: The unique identifier for the texture (typically ObjectIdentifier of source image).
    ///   - width: Width of the texture (used for memory tracking).
    ///   - height: Height of the texture (used for memory tracking).
    ///   - factory: A closure that creates the texture if not cached.
    /// - Returns: The cached or newly created texture.
    public func getOrCreateTexture(
        for key: ObjectIdentifier,
        width: Int,
        height: Int,
        factory: () -> GPUTexture?
    ) -> GPUTexture? {
        // Check cache first
        if var entry = cache[key] {
            // Update access tracking
            entry.lastAccessFrame = currentFrame
            entry.accessCount += 1
            cache[key] = entry
            cacheHits += 1
            return entry.texture
        }

        // Cache miss - create new texture
        cacheMisses += 1

        guard let texture = factory() else {
            return nil
        }

        let memorySize = UInt64(width * height * 4)

        // Evict if necessary
        evictIfNeeded(forNewMemory: memorySize)

        // Add to cache
        let entry = TextureCacheEntry(
            texture: texture,
            width: width,
            height: height,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        cache[key] = entry
        currentMemoryBytes += memorySize

        return texture
    }

    /// Gets a cached texture if it exists.
    ///
    /// - Parameter key: The unique identifier for the texture.
    /// - Returns: The cached texture, or nil if not found.
    public func getCachedTexture(for key: ObjectIdentifier) -> GPUTexture? {
        guard var entry = cache[key] else {
            return nil
        }

        // Update access tracking
        entry.lastAccessFrame = currentFrame
        entry.accessCount += 1
        cache[key] = entry
        cacheHits += 1

        return entry.texture
    }

    /// Manually adds a texture to the cache.
    ///
    /// - Parameters:
    ///   - texture: The texture to cache.
    ///   - key: The unique identifier for the texture.
    ///   - width: Width of the texture.
    ///   - height: Height of the texture.
    public func cacheTexture(_ texture: GPUTexture, for key: ObjectIdentifier, width: Int, height: Int) {
        // Remove existing entry if present
        if let existing = cache[key] {
            currentMemoryBytes -= existing.memorySize
        }

        let memorySize = UInt64(width * height * 4)
        evictIfNeeded(forNewMemory: memorySize)

        let entry = TextureCacheEntry(
            texture: texture,
            width: width,
            height: height,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        cache[key] = entry
        currentMemoryBytes += memorySize
    }

    /// Removes a specific texture from the cache.
    ///
    /// - Parameter key: The unique identifier of the texture to remove.
    public func removeTexture(for key: ObjectIdentifier) {
        if let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.memorySize
        }
    }

    /// Advances the frame counter for LRU tracking.
    ///
    /// Call this at the end of each frame.
    public func advanceFrame() {
        currentFrame += 1
    }

    /// Clears all cached textures.
    public func clearAll() {
        cache.removeAll()
        currentMemoryBytes = 0
    }

    /// Invalidates the texture manager.
    public func invalidate() {
        clearAll()
        device = nil
    }

    /// Evicts textures that haven't been used for the specified number of frames.
    ///
    /// - Parameter frameThreshold: Number of frames after which unused textures are evicted.
    public func evictStale(olderThan frameThreshold: UInt64) {
        let cutoffFrame = currentFrame > frameThreshold ? currentFrame - frameThreshold : 0

        var keysToRemove: [ObjectIdentifier] = []
        for (key, entry) in cache {
            if entry.lastAccessFrame < cutoffFrame {
                keysToRemove.append(key)
            }
        }

        for key in keysToRemove {
            if let entry = cache.removeValue(forKey: key) {
                currentMemoryBytes -= entry.memorySize
            }
        }
    }

    // MARK: - Private Methods

    /// Evicts least recently used textures if the cache is over capacity.
    private func evictIfNeeded(forNewMemory newMemory: UInt64) {
        // Check if we need to evict based on count
        while cache.count >= maxTextures {
            evictLeastRecentlyUsed()
        }

        // Check if we need to evict based on memory
        while currentMemoryBytes + newMemory > maxMemoryBytes && !cache.isEmpty {
            evictLeastRecentlyUsed()
        }
    }

    /// Evicts the single least recently used texture.
    private func evictLeastRecentlyUsed() {
        guard !cache.isEmpty else { return }

        // Find the entry with the oldest last access frame
        var oldestKey: ObjectIdentifier?
        var oldestFrame: UInt64 = .max

        for (key, entry) in cache {
            if entry.lastAccessFrame < oldestFrame {
                oldestFrame = entry.lastAccessFrame
                oldestKey = key
            }
        }

        if let key = oldestKey, let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.memorySize
        }
    }
}

// MARK: - Geometry Cache (Path Tessellation)

/// A key for caching tessellated geometry.
///
/// This key uniquely identifies a tessellated path based on:
/// - The path itself (via hash)
/// - Rendering parameters (line width, cap, join, fill rule)
/// - Stroke/fill mode
public struct GeometryCacheKey: Hashable {
    /// Hash of the path data.
    public let pathHash: Int

    /// Whether this is for stroke (true) or fill (false).
    public let isStroke: Bool

    /// Line width for strokes.
    public let lineWidth: CGFloat

    /// Line cap style for strokes.
    public let lineCap: CAShapeLayerLineCap

    /// Line join style for strokes.
    public let lineJoin: CAShapeLayerLineJoin

    /// Miter limit for miter joins.
    public let miterLimit: CGFloat

    /// Fill rule for fills.
    public let fillRule: CAShapeLayerFillRule

    /// Stroke start (0.0 to 1.0).
    public let strokeStart: CGFloat

    /// Stroke end (0.0 to 1.0).
    public let strokeEnd: CGFloat

    public init(
        pathHash: Int,
        isStroke: Bool,
        lineWidth: CGFloat = 1.0,
        lineCap: CAShapeLayerLineCap = .butt,
        lineJoin: CAShapeLayerLineJoin = .miter,
        miterLimit: CGFloat = 10.0,
        fillRule: CAShapeLayerFillRule = .nonZero,
        strokeStart: CGFloat = 0.0,
        strokeEnd: CGFloat = 1.0
    ) {
        self.pathHash = pathHash
        self.isStroke = isStroke
        self.lineWidth = lineWidth
        self.lineCap = lineCap
        self.lineJoin = lineJoin
        self.miterLimit = miterLimit
        self.fillRule = fillRule
        self.strokeStart = strokeStart
        self.strokeEnd = strokeEnd
    }
}

/// Cached tessellated geometry data.
public struct TessellatedGeometry {
    /// Triangle vertices for rendering.
    public let vertices: [CGPoint]

    /// Number of triangles (vertices.count / 3).
    public var triangleCount: Int {
        return vertices.count / 3
    }

    /// Approximate memory usage in bytes.
    public var memorySize: Int {
        return vertices.count * MemoryLayout<CGPoint>.stride
    }

    public init(vertices: [CGPoint]) {
        self.vertices = vertices
    }
}

/// A cache entry with access tracking for LRU eviction.
private struct GeometryCacheEntry {
    let geometry: TessellatedGeometry
    var lastAccessFrame: UInt64
    var accessCount: UInt64
}

/// A cache for tessellated path geometry with LRU eviction.
///
/// Path tessellation (converting bezier curves to triangles) is computationally
/// expensive. This cache stores tessellated geometry so paths that haven't changed
/// don't need to be re-tessellated every frame.
///
/// ## Usage
///
/// ```swift
/// let cache = GeometryCache(maxEntries: 256, maxMemoryBytes: 64 * 1024 * 1024)
///
/// // Create a cache key for a shape layer's path
/// let key = GeometryCacheKey(
///     pathHash: path.hashValue,
///     isStroke: true,
///     lineWidth: shapeLayer.lineWidth,
///     lineCap: shapeLayer.lineCap,
///     lineJoin: shapeLayer.lineJoin
/// )
///
/// // Get cached geometry or compute it
/// if let cached = cache.getGeometry(for: key) {
///     // Use cached geometry
/// } else {
///     let geometry = tessellate(path)
///     cache.cacheGeometry(geometry, for: key)
/// }
///
/// // At end of frame
/// cache.advanceFrame()
/// ```
public final class GeometryCache {

    // MARK: - Properties

    /// Cache of tessellated geometry keyed by GeometryCacheKey.
    private var cache: [GeometryCacheKey: GeometryCacheEntry] = [:]

    /// Current frame number for LRU tracking.
    private var currentFrame: UInt64 = 0

    /// Maximum number of entries in the cache.
    public let maxEntries: Int

    /// Maximum total memory in bytes for cached geometry.
    public let maxMemoryBytes: Int

    /// Current number of entries in the cache.
    public var entryCount: Int {
        return cache.count
    }

    /// Current total memory usage in bytes.
    public private(set) var currentMemoryBytes: Int = 0

    /// Number of cache hits since creation.
    public private(set) var cacheHits: UInt64 = 0

    /// Number of cache misses since creation.
    public private(set) var cacheMisses: UInt64 = 0

    /// Cache hit rate (0.0 to 1.0).
    public var hitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0.0
    }

    // MARK: - Initialization

    /// Creates a new geometry cache.
    ///
    /// - Parameters:
    ///   - maxEntries: Maximum number of entries to cache (default: 256).
    ///   - maxMemoryBytes: Maximum memory in bytes (default: 64MB).
    public init(maxEntries: Int = 256, maxMemoryBytes: Int = 64 * 1024 * 1024) {
        self.maxEntries = maxEntries
        self.maxMemoryBytes = maxMemoryBytes
    }

    // MARK: - Public Methods

    /// Gets cached tessellated geometry if it exists.
    ///
    /// - Parameter key: The cache key.
    /// - Returns: The cached geometry, or nil if not found.
    public func getGeometry(for key: GeometryCacheKey) -> TessellatedGeometry? {
        guard var entry = cache[key] else {
            cacheMisses += 1
            return nil
        }

        // Update access tracking
        entry.lastAccessFrame = currentFrame
        entry.accessCount += 1
        cache[key] = entry
        cacheHits += 1

        return entry.geometry
    }

    /// Caches tessellated geometry.
    ///
    /// - Parameters:
    ///   - geometry: The tessellated geometry to cache.
    ///   - key: The cache key.
    public func cacheGeometry(_ geometry: TessellatedGeometry, for key: GeometryCacheKey) {
        // Remove existing entry if present
        if let existing = cache[key] {
            currentMemoryBytes -= existing.geometry.memorySize
        }

        let memorySize = geometry.memorySize
        evictIfNeeded(forNewMemory: memorySize)

        let entry = GeometryCacheEntry(
            geometry: geometry,
            lastAccessFrame: currentFrame,
            accessCount: 1
        )
        cache[key] = entry
        currentMemoryBytes += memorySize
    }

    /// Gets cached geometry or creates it using the provided factory.
    ///
    /// - Parameters:
    ///   - key: The cache key.
    ///   - factory: A closure that creates the geometry if not cached.
    /// - Returns: The cached or newly created geometry.
    public func getOrCreateGeometry(
        for key: GeometryCacheKey,
        factory: () -> TessellatedGeometry
    ) -> TessellatedGeometry {
        if let cached = getGeometry(for: key) {
            return cached
        }

        let geometry = factory()
        cacheGeometry(geometry, for: key)
        return geometry
    }

    /// Removes a specific geometry from the cache.
    ///
    /// - Parameter key: The cache key.
    public func removeGeometry(for key: GeometryCacheKey) {
        if let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.geometry.memorySize
        }
    }

    /// Advances the frame counter for LRU tracking.
    ///
    /// Call this at the end of each frame.
    public func advanceFrame() {
        currentFrame += 1
    }

    /// Clears all cached geometry.
    public func clearAll() {
        cache.removeAll()
        currentMemoryBytes = 0
    }

    /// Invalidates the cache.
    public func invalidate() {
        clearAll()
    }

    /// Evicts geometry that hasn't been used for the specified number of frames.
    ///
    /// - Parameter frameThreshold: Number of frames after which unused geometry is evicted.
    public func evictStale(olderThan frameThreshold: UInt64) {
        let cutoffFrame = currentFrame > frameThreshold ? currentFrame - frameThreshold : 0

        var keysToRemove: [GeometryCacheKey] = []
        for (key, entry) in cache {
            if entry.lastAccessFrame < cutoffFrame {
                keysToRemove.append(key)
            }
        }

        for key in keysToRemove {
            if let entry = cache.removeValue(forKey: key) {
                currentMemoryBytes -= entry.geometry.memorySize
            }
        }
    }

    // MARK: - Private Methods

    /// Evicts least recently used entries if the cache is over capacity.
    private func evictIfNeeded(forNewMemory newMemory: Int) {
        // Check if we need to evict based on count
        while cache.count >= maxEntries {
            evictLeastRecentlyUsed()
        }

        // Check if we need to evict based on memory
        while currentMemoryBytes + newMemory > maxMemoryBytes && !cache.isEmpty {
            evictLeastRecentlyUsed()
        }
    }

    /// Evicts the single least recently used entry.
    private func evictLeastRecentlyUsed() {
        guard !cache.isEmpty else { return }

        // Find the entry with the oldest last access frame
        var oldestKey: GeometryCacheKey?
        var oldestFrame: UInt64 = .max

        for (key, entry) in cache {
            if entry.lastAccessFrame < oldestFrame {
                oldestFrame = entry.lastAccessFrame
                oldestKey = key
            }
        }

        if let key = oldestKey, let entry = cache.removeValue(forKey: key) {
            currentMemoryBytes -= entry.geometry.memorySize
        }
    }
}

// MARK: - WebGPU Renderer

/// A renderer that uses WebGPU to render layer trees in WASM/Web environments.
///
/// This is the primary renderer for OpenCoreAnimation in production.
public final class CAWebGPURenderer: CARenderer {

    // MARK: - Constants

    /// Maximum number of layers that can be rendered per frame.
    private static let maxLayers = 256

    /// Uniform buffer alignment requirement (WebGPU minimum is 256 bytes).
    private static let uniformAlignment: UInt64 = 256

    /// Size of aligned uniform data per layer.
    private static var alignedUniformSize: UInt64 {
        let baseSize = UInt64(MemoryLayout<CARendererUniforms>.stride)
        return ((baseSize + uniformAlignment - 1) / uniformAlignment) * uniformAlignment
    }

    // MARK: - Properties

    /// The WebGPU device.
    private var device: GPUDevice?

    /// The canvas context.
    private var context: GPUCanvasContext?

    /// The render pipeline.
    private var pipeline: GPURenderPipeline?

    /// The vertex buffer pool (triple buffered).
    private var vertexBufferPool: VertexBufferPool?

    /// The uniform buffer pool (triple buffered).
    private var uniformBufferPool: UniformBufferPool?

    /// The vertex buffer (large enough for all layers).
    /// - Note: This is a computed property that returns the current buffer from the pool.
    private var vertexBuffer: GPUBuffer? {
        return vertexBufferPool?.currentBuffer
    }

    /// The uniform buffer (large enough for all layers with alignment).
    /// - Note: This is a computed property that returns the current buffer from the pool.
    private var uniformBuffer: GPUBuffer? {
        return uniformBufferPool?.currentBuffer
    }

    /// The bind group layout for dynamic uniform access.
    private var bindGroupLayout: GPUBindGroupLayout?

    /// The bind group.
    /// - Note: This is a computed property that returns the current bind group from the pool.
    private var bindGroup: GPUBindGroup? {
        return uniformBufferPool?.currentBindGroup
    }

    /// The depth texture.
    private var depthTexture: GPUTexture?

    /// The current size.
    public private(set) var size: CGSize = CGSize(width: 0, height: 0)

    /// The preferred texture format.
    private var preferredFormat: GPUTextureFormat = .bgra8unorm

    /// The canvas element (JavaScript object).
    private let canvas: JSObject

    /// Current layer index during rendering.
    private var currentLayerIndex: Int = 0

    // MARK: - Clipping (masksToBounds)

    /// Represents a clip rectangle in screen coordinates.
    private struct ClipRect {
        var x: Int32
        var y: Int32
        var width: UInt32
        var height: UInt32

        /// Returns the intersection of two clip rects.
        func intersection(with other: ClipRect) -> ClipRect {
            let x1 = max(self.x, other.x)
            let y1 = max(self.y, other.y)
            let x2 = min(self.x + Int32(self.width), other.x + Int32(other.width))
            let y2 = min(self.y + Int32(self.height), other.y + Int32(other.height))

            let w = max(0, x2 - x1)
            let h = max(0, y2 - y1)

            return ClipRect(x: x1, y: y1, width: UInt32(w), height: UInt32(h))
        }
    }

    /// Stack of clip rectangles for nested masksToBounds.
    private var clipRectStack: [ClipRect] = []

    /// The full viewport clip rect.
    private var viewportClipRect: ClipRect {
        ClipRect(x: 0, y: 0, width: UInt32(size.width), height: UInt32(size.height))
    }

    /// Calculates the screen-space clip rect for a layer.
    ///
    /// Transforms the layer's bounds corners through the model matrix and
    /// returns an axis-aligned bounding box in screen coordinates.
    private func calculateClipRect(layer: CALayer, modelMatrix: Matrix4x4) -> ClipRect {
        let bounds = layer.bounds

        // Transform all four corners of the bounds
        let corners: [SIMD4<Float>] = [
            SIMD4(Float(bounds.minX), Float(bounds.minY), 0, 1),
            SIMD4(Float(bounds.maxX), Float(bounds.minY), 0, 1),
            SIMD4(Float(bounds.minX), Float(bounds.maxY), 0, 1),
            SIMD4(Float(bounds.maxX), Float(bounds.maxY), 0, 1)
        ]

        var minX: Float = .greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude
        var maxX: Float = -.greatestFiniteMagnitude
        var maxY: Float = -.greatestFiniteMagnitude

        for corner in corners {
            // Apply model matrix (which includes projection)
            let transformed = modelMatrix * corner
            // Perspective divide (w should be 1 for orthographic, but handle it anyway)
            let w = transformed.w != 0 ? transformed.w : 1
            let x = transformed.x / w
            let y = transformed.y / w

            minX = min(minX, x)
            minY = min(minY, y)
            maxX = max(maxX, x)
            maxY = max(maxY, y)
        }

        // Clamp to viewport bounds
        let viewWidth = Float(size.width)
        let viewHeight = Float(size.height)

        let clampedMinX = max(0, minX)
        let clampedMinY = max(0, minY)
        let clampedMaxX = min(viewWidth, maxX)
        let clampedMaxY = min(viewHeight, maxY)

        let width = max(0, clampedMaxX - clampedMinX)
        let height = max(0, clampedMaxY - clampedMinY)

        return ClipRect(
            x: Int32(clampedMinX),
            y: Int32(clampedMinY),
            width: UInt32(width),
            height: UInt32(height)
        )
    }

    /// Returns the current effective clip rect (intersection of all stacked clip rects).
    private var currentClipRect: ClipRect {
        clipRectStack.last ?? viewportClipRect
    }

    /// Applies the current clip rect to the render pass.
    private func applyScissorRect(_ renderPass: GPURenderPassEncoder) {
        let clip = currentClipRect
        renderPass.setScissorRect(x: clip.x, y: clip.y, width: clip.width, height: clip.height)
    }

    // MARK: - Texture Rendering

    /// The textured render pipeline.
    private var texturedPipeline: GPURenderPipeline?

    /// The bind group layout for textured rendering.
    private var texturedBindGroupLayout: GPUBindGroupLayout?

    /// The texture sampler.
    private var textureSampler: GPUSampler?

    /// Texture manager with LRU cache for efficient texture memory management.
    private var textureManager: GPUTextureManager?

    /// Legacy texture cache access for compatibility.
    /// - Note: This is a bridge to the texture manager for existing code.
    private var textureCache: [ObjectIdentifier: GPUTexture] {
        get { [:] }  // Not used for reading - use textureManager instead
        set { }  // Not used for writing - use textureManager instead
    }

    // MARK: - Shadow Rendering

    /// Shadow mask texture for blur operations.
    private var shadowMaskTexture: GPUTexture?

    /// Intermediate texture for blur ping-pong.
    private var shadowBlurTexture: GPUTexture?

    /// Shadow blur pipeline (horizontal pass).
    private var shadowBlurHorizontalPipeline: GPURenderPipeline?

    /// Shadow blur pipeline (vertical pass).
    private var shadowBlurVerticalPipeline: GPURenderPipeline?

    /// Shadow composite pipeline.
    private var shadowCompositePipeline: GPURenderPipeline?

    /// Shadow mask pipeline.
    private var shadowMaskPipeline: GPURenderPipeline?

    /// Shadow bind group layout.
    private var shadowBindGroupLayout: GPUBindGroupLayout?

    /// Full-screen quad sampler for blur.
    private var blurSampler: GPUSampler?

    /// Blur uniform buffer.
    private var blurUniformBuffer: GPUBuffer?

    /// Bind group for horizontal blur pass (samples from shadowMaskTexture).
    private var blurHorizontalBindGroup: GPUBindGroup?

    /// Bind group for vertical blur pass (samples from shadowBlurTexture).
    private var blurVerticalBindGroup: GPUBindGroup?

    /// Cache of pre-blurred shadow textures keyed by layer identity.
    private var blurredShadowCache: [ObjectIdentifier: GPUTexture] = [:]

    // MARK: - Mask Rendering (Stencil-based)

    /// Mask texture for layer masking (texture-based approach).
    private var maskTexture: GPUTexture?

    /// Masked rendering pipeline (texture-based).
    private var maskedPipeline: GPURenderPipeline?

    /// Mask bind group layout.
    private var maskBindGroupLayout: GPUBindGroupLayout?

    /// Pipeline for writing mask to stencil buffer (stencil-based approach).
    /// Writes to stencil where mask alpha > 0, no color output.
    private var stencilWritePipeline: GPURenderPipeline?

    /// Pipeline for rendering with stencil test.
    /// Only renders where stencil value matches.
    private var stencilTestPipeline: GPURenderPipeline?

    /// Current stencil reference value for nested masks.
    private var currentStencilValue: UInt32 = 0

    // MARK: - Particle System (CAEmitterLayer)

    /// Particle instance buffer.
    private var particleBuffer: GPUBuffer?

    /// Particle pipeline.
    private var particlePipeline: GPURenderPipeline?

    /// Maximum number of particles.
    private static let maxParticles = 10000

    /// Active particle data.
    private var activeParticles: [EmitterParticle] = []

    // MARK: - Shader Code

    private let shaderCode = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
        borderWidth: f32,
        renderMode: f32,  // 0 = fill, 1 = border, 2 = gradient
        gradientStartPoint: vec2<f32>,
        gradientEndPoint: vec2<f32>,
        gradientColorCount: f32,
        padding3: vec3<f32>,
        gradientColor0: vec4<f32>,
        gradientColor1: vec4<f32>,
        gradientColor2: vec4<f32>,
        gradientColor3: vec4<f32>,
        gradientColor4: vec4<f32>,
        gradientColor5: vec4<f32>,
        gradientColor6: vec4<f32>,
        gradientColor7: vec4<f32>,
        gradientLocations: vec4<f32>,
        gradientLocations2: vec4<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) texCoord: vec2<f32>,
        @location(2) color: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
        @location(1) color: vec4<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.texCoord = input.texCoord;
        output.color = input.color * uniforms.opacity;
        return output;
    }

    // Signed distance function for a rounded rectangle
    fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
        let q = abs(p) - halfSize + radius;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    }

    // Get gradient color at index
    fn getGradientColor(index: i32) -> vec4<f32> {
        switch (index) {
            case 0: { return uniforms.gradientColor0; }
            case 1: { return uniforms.gradientColor1; }
            case 2: { return uniforms.gradientColor2; }
            case 3: { return uniforms.gradientColor3; }
            case 4: { return uniforms.gradientColor4; }
            case 5: { return uniforms.gradientColor5; }
            case 6: { return uniforms.gradientColor6; }
            case 7: { return uniforms.gradientColor7; }
            default: { return vec4<f32>(0.0); }
        }
    }

    // Get gradient location at index
    fn getGradientLocation(index: i32) -> f32 {
        switch (index) {
            case 0: { return uniforms.gradientLocations.x; }
            case 1: { return uniforms.gradientLocations.y; }
            case 2: { return uniforms.gradientLocations.z; }
            case 3: { return uniforms.gradientLocations.w; }
            case 4: { return uniforms.gradientLocations2.x; }
            case 5: { return uniforms.gradientLocations2.y; }
            case 6: { return uniforms.gradientLocations2.z; }
            case 7: { return uniforms.gradientLocations2.w; }
            default: { return 0.0; }
        }
    }

    // Calculate gradient color at position t (0-1)
    fn sampleGradient(t: f32) -> vec4<f32> {
        let colorCount = i32(uniforms.gradientColorCount);
        if (colorCount <= 0) {
            return vec4<f32>(0.0);
        }
        if (colorCount == 1) {
            return getGradientColor(0);
        }

        let clampedT = clamp(t, 0.0, 1.0);

        // Find the two colors to interpolate between
        for (var i = 1; i < colorCount; i++) {
            let loc0 = getGradientLocation(i - 1);
            let loc1 = getGradientLocation(i);
            if (clampedT <= loc1) {
                let localT = (clampedT - loc0) / max(loc1 - loc0, 0.0001);
                let color0 = getGradientColor(i - 1);
                let color1 = getGradientColor(i);
                return mix(color0, color1, clamp(localT, 0.0, 1.0));
            }
        }

        // Return last color if past all stops
        return getGradientColor(colorCount - 1);
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        // Handle case where layer size is zero
        if (uniforms.layerSize.x <= 0.0 || uniforms.layerSize.y <= 0.0) {
            return input.color;
        }

        // Convert texCoord (0-1) to pixel coordinates centered at origin
        let pixelCoord = (input.texCoord - 0.5) * uniforms.layerSize;

        // Calculate half-size of the rectangle in pixels
        let halfSize = uniforms.layerSize * 0.5;

        // Clamp corner radius to half the smaller dimension
        let maxRadius = min(halfSize.x, halfSize.y);
        let radius = min(uniforms.cornerRadius, maxRadius);

        // Gradient rendering mode
        if (uniforms.renderMode > 1.5) {
            // Calculate gradient position
            let gradientDir = uniforms.gradientEndPoint - uniforms.gradientStartPoint;
            let gradientLen = length(gradientDir);
            var t: f32 = 0.0;
            if (gradientLen > 0.0001) {
                let normalizedDir = gradientDir / gradientLen;
                let relativePos = input.texCoord - uniforms.gradientStartPoint;
                t = dot(relativePos, normalizedDir) / gradientLen;
            }

            // Sample gradient
            var gradientColor = sampleGradient(t);
            gradientColor.a *= uniforms.opacity;

            // Apply corner radius if set
            if (radius > 0.0) {
                let dist = sdRoundedBox(pixelCoord, halfSize, radius);
                let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
                gradientColor.a *= alpha;
            }

            return gradientColor;
        }

        // Border rendering mode
        if (uniforms.renderMode > 0.5) {
            // Calculate outer and inner signed distances
            let outerDist = sdRoundedBox(pixelCoord, halfSize, radius);
            let innerHalfSize = halfSize - uniforms.borderWidth;
            let innerRadius = max(0.0, radius - uniforms.borderWidth);
            let innerDist = sdRoundedBox(pixelCoord, innerHalfSize, innerRadius);

            // Border is where we're inside outer but outside inner
            let outerAlpha = 1.0 - smoothstep(-1.0, 1.0, outerDist);
            let innerAlpha = 1.0 - smoothstep(-1.0, 1.0, innerDist);
            let borderAlpha = outerAlpha - innerAlpha;

            return vec4<f32>(input.color.rgb, input.color.a * borderAlpha);
        }

        // Fill rendering mode (default)
        if (uniforms.cornerRadius <= 0.0) {
            return input.color;
        }

        // Calculate signed distance for fill
        let dist = sdRoundedBox(pixelCoord, halfSize, radius);

        // Anti-aliased edge (smooth over 1 pixel)
        let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);

        return vec4<f32>(input.color.rgb, input.color.a * alpha);
    }
    """

    /// Shader code for textured rendering.
    private let texturedShaderCode = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var textureSampler: sampler;
    @group(0) @binding(2) var textureData: texture_2d<f32>;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) texCoord: vec2<f32>,
        @location(2) color: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
        @location(1) color: vec4<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.texCoord = input.texCoord;
        output.color = input.color * uniforms.opacity;
        return output;
    }

    // Signed distance function for a rounded rectangle
    fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
        let q = abs(p) - halfSize + radius;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        // Sample texture
        var texColor = textureSample(textureData, textureSampler, input.texCoord);

        // Apply opacity
        texColor.a *= uniforms.opacity;

        // Apply corner radius if set
        if (uniforms.cornerRadius > 0.0 && uniforms.layerSize.x > 0.0 && uniforms.layerSize.y > 0.0) {
            let pixelCoord = (input.texCoord - 0.5) * uniforms.layerSize;
            let halfSize = uniforms.layerSize * 0.5;
            let maxRadius = min(halfSize.x, halfSize.y);
            let radius = min(uniforms.cornerRadius, maxRadius);
            let dist = sdRoundedBox(pixelCoord, halfSize, radius);
            let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
            texColor.a *= alpha;
        }

        return texColor;
    }
    """

    /// Shader code for shadow mask generation.
    private let shadowMaskShaderCode = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        // Full-screen quad from vertex index
        let positions = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
            vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
        );
        let pos = positions[vertexIndex];
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(pos, 0.0, 1.0);
        output.texCoord = pos;
        return output;
    }

    fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
        let q = abs(p) - halfSize + radius;
        return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        if (uniforms.layerSize.x <= 0.0 || uniforms.layerSize.y <= 0.0) {
            return vec4<f32>(1.0);
        }
        let pixelCoord = (input.texCoord - 0.5) * uniforms.layerSize;
        let halfSize = uniforms.layerSize * 0.5;
        let maxRadius = min(halfSize.x, halfSize.y);
        let radius = min(uniforms.cornerRadius, maxRadius);
        let dist = sdRoundedBox(pixelCoord, halfSize, radius);
        let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);
        return vec4<f32>(alpha, alpha, alpha, alpha);
    }
    """

    /// Shader code for Gaussian blur (horizontal pass).
    private let blurHorizontalShaderCode = """
    struct BlurUniforms {
        texelSize: vec2<f32>,
        blurRadius: f32,
        padding: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: BlurUniforms;
    @group(0) @binding(1) var inputTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        let positions = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
            vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
        );
        let texCoords = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
            vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0)
        );
        var output: VertexOutput;
        output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
        output.texCoord = texCoords[vertexIndex];
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
        var result = textureSample(inputTexture, texSampler, input.texCoord) * weights[0];

        for (var i = 1; i < 5; i++) {
            let offset = vec2<f32>(f32(i) * uniforms.blurRadius * uniforms.texelSize.x, 0.0);
            result += textureSample(inputTexture, texSampler, input.texCoord + offset) * weights[i];
            result += textureSample(inputTexture, texSampler, input.texCoord - offset) * weights[i];
        }
        return result;
    }
    """

    /// Shader code for Gaussian blur (vertical pass).
    private let blurVerticalShaderCode = """
    struct BlurUniforms {
        texelSize: vec2<f32>,
        blurRadius: f32,
        padding: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: BlurUniforms;
    @group(0) @binding(1) var inputTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        let positions = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
            vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
        );
        let texCoords = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
            vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0)
        );
        var output: VertexOutput;
        output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
        output.texCoord = texCoords[vertexIndex];
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
        var result = textureSample(inputTexture, texSampler, input.texCoord) * weights[0];

        for (var i = 1; i < 5; i++) {
            let offset = vec2<f32>(0.0, f32(i) * uniforms.blurRadius * uniforms.texelSize.y);
            result += textureSample(inputTexture, texSampler, input.texCoord + offset) * weights[i];
            result += textureSample(inputTexture, texSampler, input.texCoord - offset) * weights[i];
        }
        return result;
    }
    """

    /// Shader code for shadow compositing.
    private let shadowCompositeShaderCode = """
    struct ShadowUniforms {
        mvpMatrix: mat4x4<f32>,
        shadowColor: vec4<f32>,
        shadowOffset: vec2<f32>,
        layerSize: vec2<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: ShadowUniforms;
    @group(0) @binding(1) var shadowTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) texCoord: vec2<f32>,
        @location(2) color: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.texCoord = input.texCoord;
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let shadowAlpha = textureSample(shadowTexture, texSampler, input.texCoord).r;
        return vec4<f32>(uniforms.shadowColor.rgb, uniforms.shadowColor.a * shadowAlpha);
    }
    """

    /// Shader code for masked rendering.
    private let maskedShaderCode = """
    struct Uniforms {
        mvpMatrix: mat4x4<f32>,
        opacity: f32,
        cornerRadius: f32,
        layerSize: vec2<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var maskTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct VertexInput {
        @location(0) position: vec2<f32>,
        @location(1) texCoord: vec2<f32>,
        @location(2) color: vec4<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
        @location(1) color: vec4<f32>,
    }

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 0.0, 1.0);
        output.texCoord = input.texCoord;
        output.color = input.color * uniforms.opacity;
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let maskAlpha = textureSample(maskTexture, texSampler, input.texCoord).a;
        return vec4<f32>(input.color.rgb, input.color.a * maskAlpha);
    }
    """

    /// Shader code for particle rendering.
    private let particleShaderCode = """
    struct ParticleUniforms {
        mvpMatrix: mat4x4<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: ParticleUniforms;
    @group(0) @binding(1) var particleTexture: texture_2d<f32>;
    @group(0) @binding(2) var texSampler: sampler;

    struct ParticleInstance {
        @location(0) position: vec3<f32>,
        @location(1) color: vec4<f32>,
        @location(2) scaleRotation: vec2<f32>,
    }

    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
        @location(1) color: vec4<f32>,
    }

    @vertex
    fn vertexMain(
        @builtin(vertex_index) vertexIndex: u32,
        instance: ParticleInstance
    ) -> VertexOutput {
        let corners = array<vec2<f32>, 6>(
            vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5), vec2<f32>(-0.5, 0.5),
            vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.5)
        );
        var corner = corners[vertexIndex];

        // Apply rotation
        let cos_r = cos(instance.scaleRotation.y);
        let sin_r = sin(instance.scaleRotation.y);
        corner = vec2<f32>(
            corner.x * cos_r - corner.y * sin_r,
            corner.x * sin_r + corner.y * cos_r
        );

        // Apply scale
        corner *= instance.scaleRotation.x;

        var output: VertexOutput;
        output.position = uniforms.mvpMatrix * vec4<f32>(instance.position + vec3<f32>(corner, 0.0), 1.0);
        output.texCoord = corners[vertexIndex] + vec2<f32>(0.5);
        output.color = instance.color;
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
        let texColor = textureSample(particleTexture, texSampler, input.texCoord);
        return texColor * input.color;
    }
    """

    // MARK: - Initialization

    /// Creates a new WebGPU renderer with the specified canvas element.
    ///
    /// - Parameter canvas: The JavaScript canvas element to render to.
    public init(canvas: JSObject) {
        self.canvas = canvas
    }

    // MARK: - CARenderer

    public func initialize() async throws {
        // Get GPU
        guard let gpu = GPU.shared else {
            throw CARendererError.deviceNotAvailable
        }

        // Request adapter
        guard let adapter = try await gpu.requestAdapter() else {
            throw CARendererError.deviceNotAvailable
        }

        // Request device
        let device = try await adapter.requestDevice()
        self.device = device

        // Get preferred format
        preferredFormat = gpu.preferredCanvasFormat

        // Configure canvas context
        let ctx = canvas.getContext!("webgpu")
        guard let ctxObject = ctx.object else {
            throw CARendererError.canvasNotConfigured
        }
        context = GPUCanvasContext(jsObject: ctxObject)

        context?.configure(GPUCanvasConfiguration(
            device: device,
            format: preferredFormat
        ))

        // Create shader module
        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: shaderCode
        ))

        // Create bind group layout with dynamic uniform buffer
        let layout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: .vertex,
                    buffer: GPUBufferBindingLayout(
                        type: .uniform,
                        hasDynamicOffset: true,
                        minBindingSize: UInt64(MemoryLayout<CARendererUniforms>.stride)
                    )
                )
            ]
        ))
        bindGroupLayout = layout

        // Create pipeline layout
        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [layout]
        ))

        // Create render pipeline with depth testing
        pipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: true,
                depthCompare: .lessEqual,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        // Create triple-buffered vertex buffer pool
        let vertexBufferSize = UInt64(MemoryLayout<CARendererVertex>.stride * 6 * Self.maxLayers)
        vertexBufferPool = VertexBufferPool(
            device: device,
            bufferSize: vertexBufferSize,
            bufferCount: 3
        )

        // Create triple-buffered uniform buffer pool with bind groups
        let uniformBufferSize = Self.alignedUniformSize * UInt64(Self.maxLayers)
        uniformBufferPool = UniformBufferPool(
            device: device,
            bufferSize: uniformBufferSize,
            bindGroupLayout: layout,
            bindingSize: UInt64(MemoryLayout<CARendererUniforms>.stride),
            bufferCount: 3
        )

        // Create texture manager with LRU cache
        textureManager = GPUTextureManager(
            device: device,
            maxTextures: 256,
            maxMemoryBytes: 256 * 1024 * 1024  // 256 MB
        )

        // Get initial canvas size
        let width = canvas.width.number ?? 800
        let height = canvas.height.number ?? 600
        size = CGSize(width: width, height: height)

        // Create depth texture
        createDepthTexture(width: Int(width), height: Int(height))

        // Create textured pipeline
        try createTexturedPipeline(device: device)

        // Create shadow/blur pipelines
        try createShadowPipelines(device: device)

        // Create shadow textures
        createShadowTextures(width: Int(width), height: Int(height))

        // Create stencil pipelines for CALayer.mask
        try createStencilPipelines(device: device)
    }

    /// Creates stencil pipelines for CALayer.mask functionality.
    ///
    /// - stencilWritePipeline: Writes mask alpha to stencil buffer (increment stencil where alpha > 0)
    /// - stencilTestPipeline: Only renders where stencil equals reference value
    private func createStencilPipelines(device: GPUDevice) throws {
        guard let bindGroupLayout = bindGroupLayout else {
            throw CARendererError.pipelineCreationFailed
        }

        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [bindGroupLayout]
        ))

        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: shaderCode
        ))

        // Stencil write pipeline: writes to stencil where fragment alpha > 0
        // Uses increment operation to support nested masks
        stencilWritePipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: true,
                depthCompare: .lessEqual,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .incrementClamp  // Increment stencil on pass
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .incrementClamp
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        writeMask: []  // No color output, only stencil
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))

        // Stencil test pipeline: only renders where stencil equals reference
        stencilTestPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: true,
                depthCompare: .lessEqual,
                stencilFront: GPUStencilFaceState(
                    compare: .equal,  // Only pass where stencil == reference
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .equal,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))
    }

    /// Creates the textured render pipeline for layer.contents rendering.
    private func createTexturedPipeline(device: GPUDevice) throws {
        // Create shader module
        let shaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: texturedShaderCode
        ))

        // Create sampler
        textureSampler = device.createSampler(descriptor: GPUSamplerDescriptor(
            magFilter: .linear,
            minFilter: .linear,
            mipmapFilter: .linear,
            addressModeU: .clampToEdge,
            addressModeV: .clampToEdge,
            addressModeW: .clampToEdge
        ))

        // Create bind group layout with uniform, sampler, and texture
        texturedBindGroupLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(
                        type: .uniform,
                        hasDynamicOffset: true,
                        minBindingSize: UInt64(MemoryLayout<TexturedUniforms>.stride)
                    )
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(
                        sampleType: .float,
                        viewDimension: .dimension2D
                    )
                )
            ]
        ))

        guard let texturedBindGroupLayout = texturedBindGroupLayout else {
            throw CARendererError.resourceCreationFailed
        }

        // Create pipeline layout
        let pipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [texturedBindGroupLayout]
        ))

        // Create textured render pipeline
        texturedPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: true,
                depthCompare: .lessEqual,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(pipelineLayout)
        ))
    }

    /// Creates shadow and blur pipelines.
    private func createShadowPipelines(device: GPUDevice) throws {
        // Create blur sampler
        blurSampler = device.createSampler(descriptor: GPUSamplerDescriptor(
            magFilter: .linear,
            minFilter: .linear,
            addressModeU: .clampToEdge,
            addressModeV: .clampToEdge,
            addressModeW: .clampToEdge
        ))

        // Create blur bind group layout
        let blurBindGroupLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(type: .uniform)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .dimension2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                )
            ]
        ))

        shadowBindGroupLayout = blurBindGroupLayout

        let blurPipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [blurBindGroupLayout]
        ))

        // Create horizontal blur pipeline
        let blurHShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: blurHorizontalShaderCode
        ))

        shadowBlurHorizontalPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: blurHShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: blurHShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(format: .rgba8unorm)
                ]
            ),
            layout: .layout(blurPipelineLayout)
        ))

        // Create vertical blur pipeline
        let blurVShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: blurVerticalShaderCode
        ))

        shadowBlurVerticalPipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: blurVShaderModule,
                entryPoint: "vertexMain"
            ),
            fragment: GPUFragmentState(
                module: blurVShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(format: .rgba8unorm)
                ]
            ),
            layout: .layout(blurPipelineLayout)
        ))

        // Create shadow composite pipeline
        let shadowCompositeLayout = device.createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor(
            entries: [
                GPUBindGroupLayoutEntry(
                    binding: 0,
                    visibility: [.vertex, .fragment],
                    buffer: GPUBufferBindingLayout(type: .uniform)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 1,
                    visibility: .fragment,
                    texture: GPUTextureBindingLayout(sampleType: .float, viewDimension: .dimension2D)
                ),
                GPUBindGroupLayoutEntry(
                    binding: 2,
                    visibility: .fragment,
                    sampler: GPUSamplerBindingLayout(type: .filtering)
                )
            ]
        ))

        let shadowCompositePipelineLayout = device.createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor(
            bindGroupLayouts: [shadowCompositeLayout]
        ))

        let shadowCompositeShaderModule = device.createShaderModule(descriptor: GPUShaderModuleDescriptor(
            code: shadowCompositeShaderCode
        ))

        shadowCompositePipeline = device.createRenderPipeline(descriptor: GPURenderPipelineDescriptor(
            vertex: GPUVertexState(
                module: shadowCompositeShaderModule,
                entryPoint: "vertexMain",
                buffers: [
                    GPUVertexBufferLayout(
                        arrayStride: UInt64(MemoryLayout<CARendererVertex>.stride),
                        attributes: [
                            GPUVertexAttribute(format: .float32x2, offset: 0, shaderLocation: 0),
                            GPUVertexAttribute(format: .float32x2, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride), shaderLocation: 1),
                            GPUVertexAttribute(format: .float32x4, offset: UInt64(MemoryLayout<SIMD2<Float>>.stride * 2), shaderLocation: 2)
                        ]
                    )
                ]
            ),
            depthStencil: GPUDepthStencilState(
                format: .depth24plusStencil8,
                depthWriteEnabled: false,
                depthCompare: .lessEqual,
                stencilFront: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                ),
                stencilBack: GPUStencilFaceState(
                    compare: .always,
                    failOp: .keep,
                    depthFailOp: .keep,
                    passOp: .keep
                )
            ),
            fragment: GPUFragmentState(
                module: shadowCompositeShaderModule,
                entryPoint: "fragmentMain",
                targets: [
                    GPUColorTargetState(
                        format: preferredFormat,
                        blend: GPUBlendState(
                            color: GPUBlendComponent(
                                srcFactor: .srcAlpha,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            ),
                            alpha: GPUBlendComponent(
                                srcFactor: .one,
                                dstFactor: .oneMinusSrcAlpha,
                                operation: .add
                            )
                        )
                    )
                ]
            ),
            layout: .layout(shadowCompositePipelineLayout)
        ))
    }

    /// Creates or recreates shadow textures for the given size.
    private func createShadowTextures(width: Int, height: Int) {
        guard let device = device,
              let shadowBindGroupLayout = shadowBindGroupLayout,
              let blurSampler = blurSampler,
              width > 0, height > 0 else { return }

        // Shadow mask texture (stores layer shape)
        shadowMaskTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.renderAttachment, .textureBinding]
        ))

        // Shadow blur texture (for ping-pong blur)
        shadowBlurTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.renderAttachment, .textureBinding]
        ))

        // Create blur uniform buffer
        blurUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: UInt64(MemoryLayout<BlurUniforms>.stride),
            usage: [.uniform, .copyDst]
        ))

        // Create bind groups for blur passes
        guard let shadowMaskTexture = shadowMaskTexture,
              let shadowBlurTexture = shadowBlurTexture,
              let blurUniformBuffer = blurUniformBuffer else { return }

        // Horizontal blur: samples from shadowMaskTexture, outputs to shadowBlurTexture
        blurHorizontalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(shadowMaskTexture.createView())),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))

        // Vertical blur: samples from shadowBlurTexture, outputs to shadowMaskTexture
        blurVerticalBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: shadowBindGroupLayout,
            entries: [
                GPUBindGroupEntry(binding: 0, resource: .buffer(blurUniformBuffer, offset: 0, size: UInt64(MemoryLayout<BlurUniforms>.stride))),
                GPUBindGroupEntry(binding: 1, resource: .textureView(shadowBlurTexture.createView())),
                GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
            ]
        ))
    }

    /// Creates or recreates the depth texture for the given size.
    /// Uses depth24plus-stencil8 format to support both depth testing and stencil-based masking.
    private func createDepthTexture(width: Int, height: Int) {
        guard let device = device, width > 0, height > 0 else { return }

        depthTexture = device.createTexture(descriptor: GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .depth24plusStencil8,
            usage: .renderAttachment
        ))
    }

    public func resize(width: Int, height: Int) {
        size = CGSize(width: width, height: height)

        // Update canvas size
        canvas.width = .number(Double(width))
        canvas.height = .number(Double(height))

        // Recreate depth texture
        createDepthTexture(width: width, height: height)

        // Recreate shadow textures
        createShadowTextures(width: width, height: height)
    }

    /// Tracks whether shadow pre-rendering has been done for this frame.
    private var shadowsPrerendered: Bool = false

    /// Stores the pre-blurred shadow alpha value for the current frame.
    /// After blur, this is stored in shadowMaskTexture.
    private var hasPrerenderredShadow: Bool = false

    public func render(layer rootLayer: CALayer) {
        guard let device = device,
              let context = context,
              let pipeline = pipeline,
              bindGroup != nil,
              let depthTexture = depthTexture else { return }

        // Reset layer index for this frame
        currentLayerIndex = 0

        // Reset clip rect stack for this frame
        clipRectStack.removeAll()

        // Reset shadow pre-rendering state
        shadowsPrerendered = false
        hasPrerenderredShadow = false

        // Get current texture
        let currentTexture = context.getCurrentTexture()
        let textureView = currentTexture.createView()
        let depthTextureView = depthTexture.createView()

        // Create command encoder
        let encoder = device.createCommandEncoder()

        // Create projection matrix (needed for shadow pre-rendering)
        let projectionMatrix = Matrix4x4.orthographic(
            left: 0,
            right: Float(size.width),
            bottom: Float(size.height),
            top: 0,
            near: -1000,
            far: 1000
        )

        // Pre-render shadows with 2-pass Gaussian blur
        prerenderShadows(rootLayer, encoder: encoder, projectionMatrix: projectionMatrix)

        // Begin render pass with depth attachment
        let renderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: textureView,
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 1),
                    loadOp: .clear,
                    storeOp: .store
                )
            ],
            depthStencilAttachment: GPURenderPassDepthStencilAttachment(
                view: depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: .clear,
                depthStoreOp: .store,
                stencilClearValue: 0,
                stencilLoadOp: .clear,
                stencilStoreOp: .store
            )
        ))

        renderPass.setPipeline(pipeline)

        // Render layer tree (projectionMatrix already created above for shadow pre-rendering)
        renderLayer(rootLayer, renderPass: renderPass, parentMatrix: projectionMatrix)

        renderPass.end()

        // Submit command buffer
        device.queue.submit([encoder.finish()])

        // Advance buffer pools and texture manager to the next frame (triple buffering + LRU)
        vertexBufferPool?.advanceFrame()
        uniformBufferPool?.advanceFrame()
        textureManager?.advanceFrame()

        // Periodically evict stale textures (not used in last 300 frames = ~5 seconds at 60fps)
        textureManager?.evictStale(olderThan: 300)
    }

    public func invalidate() {
        // Invalidate buffer pools
        vertexBufferPool?.invalidate()
        vertexBufferPool = nil
        uniformBufferPool?.invalidate()
        uniformBufferPool = nil

        // Invalidate texture manager
        textureManager?.invalidate()
        textureManager = nil

        bindGroupLayout = nil
        depthTexture = nil
        pipeline = nil
        texturedPipeline = nil
        texturedBindGroupLayout = nil
        textureSampler = nil

        // Shadow resources
        shadowMaskTexture = nil
        shadowBlurTexture = nil
        shadowBlurHorizontalPipeline = nil
        shadowBlurVerticalPipeline = nil
        shadowCompositePipeline = nil
        shadowMaskPipeline = nil
        shadowBindGroupLayout = nil
        blurSampler = nil
        blurUniformBuffer = nil
        blurHorizontalBindGroup = nil
        blurVerticalBindGroup = nil
        blurredShadowCache.removeAll()

        // Mask resources
        maskTexture = nil
        maskedPipeline = nil
        maskBindGroupLayout = nil

        // Stencil resources
        stencilWritePipeline = nil
        stencilTestPipeline = nil
        currentStencilValue = 0

        // Particle resources
        particleBuffer = nil
        particlePipeline = nil
        activeParticles.removeAll()

        context = nil
        device = nil
    }

    // MARK: - Private Methods

    /// Converts Swift data to a JavaScript Float32Array for WebGPU buffer writes.
    private func createFloat32Array<T>(from data: inout T) -> JSObject {
        let byteCount = MemoryLayout<T>.stride
        let floatCount = byteCount / 4
        let float32Array = JSObject.global.Float32Array.function!.new(floatCount)
        withUnsafeBytes(of: &data) { bytes in
            for i in 0..<floatCount {
                let value = bytes.load(fromByteOffset: i * 4, as: Float.self)
                float32Array[i] = .number(Double(value))
            }
        }
        return float32Array
    }

    /// Converts an array of vertices to a JavaScript Float32Array.
    private func createFloat32Array(from vertices: inout [CARendererVertex]) -> JSObject {
        let floatCount = vertices.count * (MemoryLayout<CARendererVertex>.stride / 4)
        let float32Array = JSObject.global.Float32Array.function!.new(floatCount)
        vertices.withUnsafeBytes { bytes in
            for i in 0..<floatCount {
                let value = bytes.load(fromByteOffset: i * 4, as: Float.self)
                float32Array[i] = .number(Double(value))
            }
        }
        return float32Array
    }

    private func renderLayer(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        guard let device = device,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        // Get the presentation layer for animated values, fall back to model layer
        // This is critical for animations to be visible - the presentation layer
        // reflects the current animated state of all properties
        let presentationLayer = layer.presentation() ?? layer

        // Skip hidden layers (using presentation layer values)
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }

        // Check layer limit
        guard currentLayerIndex < Self.maxLayers else { return }

        // Calculate model matrix using presentation layer values
        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Handle CALayer.mask if set
        let hasMask = presentationLayer.mask != nil
        if hasMask, let maskLayer = presentationLayer.mask {
            // Render mask to stencil buffer
            renderMaskToStencil(maskLayer, renderPass: renderPass, parentMatrix: modelMatrix)
        }

        // Handle special layer types first
        // CATransformLayer: Only render sublayers, skip own properties
        if presentationLayer is CATransformLayer {
            renderTransformLayerSublayers(layer, renderPass: renderPass, parentMatrix: modelMatrix)
            return
        }

        // CAEmitterLayer: Render particle system
        if let emitterLayer = presentationLayer as? CAEmitterLayer {
            renderEmitterLayer(emitterLayer, device: device, renderPass: renderPass,
                             modelMatrix: modelMatrix, bindGroup: bindGroup)
            // Render sublayers after particles
            if let sublayers = layer.sublayers {
                for sublayer in sublayers {
                    self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: modelMatrix)
                }
            }
            return
        }

        // CATiledLayer: Render tiled content
        if let tiledLayer = presentationLayer as? CATiledLayer {
            renderTiledLayer(tiledLayer, device: device, renderPass: renderPass,
                           modelMatrix: modelMatrix, bindGroup: bindGroup)
            return
        }

        // Render shadow before layer content (if shadow is visible)
        if presentationLayer.shadowOpacity > 0 && presentationLayer.shadowColor != nil {
            renderLayerShadow(presentationLayer, device: device, renderPass: renderPass,
                            modelMatrix: modelMatrix)
        }

        // Check if this is a text layer
        if let textLayer = presentationLayer as? CATextLayer, textLayer.string != nil {
            renderTextLayer(textLayer, device: device, renderPass: renderPass,
                           modelMatrix: modelMatrix, bindGroup: bindGroup)
        }
        // Check if this is a shape layer
        else if let shapeLayer = presentationLayer as? CAShapeLayer, shapeLayer.path != nil {
            renderShapeLayer(shapeLayer, device: device, renderPass: renderPass,
                            modelMatrix: modelMatrix, bindGroup: bindGroup)
        }
        // Check if this is a gradient layer
        else if let gradientLayer = presentationLayer as? CAGradientLayer,
           let colors = gradientLayer.colors, !colors.isEmpty {
            renderGradientLayer(gradientLayer, device: device, renderPass: renderPass,
                              modelMatrix: modelMatrix, bindGroup: bindGroup)
        }
        // Check if layer has contents (CGImage)
        else if let contents = presentationLayer.contents {
            renderContentsLayer(presentationLayer, contents: contents, device: device,
                               renderPass: renderPass, modelMatrix: modelMatrix)
        }
        // Render background color if set (and not a gradient layer, or gradient has no colors)
        else if presentationLayer.backgroundColor != nil {
            let layerIndex = currentLayerIndex
            currentLayerIndex += 1

            // Create scale matrix for layer bounds (column-major order)
            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
                SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let finalMatrix = modelMatrix * scaleMatrix

            // Update uniforms at the correct offset
            var uniforms = CARendererUniforms(
                mvpMatrix: finalMatrix,
                opacity: presentationLayer.opacity,
                cornerRadius: Float(presentationLayer.cornerRadius),
                layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height))
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(
                uniformBuffer,
                bufferOffset: uniformOffset,
                data: uniformData
            )

            // Create vertices using presentation layer color
            let color = presentationLayer.backgroundColorComponents
            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            ]

            // Write vertices at the correct offset
            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            // Set bind group with dynamic offset for this layer's uniforms
            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Render border if set
        if presentationLayer.borderWidth > 0 && presentationLayer.borderColor != nil {
            guard currentLayerIndex < Self.maxLayers else { return }

            let layerIndex = currentLayerIndex
            currentLayerIndex += 1

            // Create scale matrix for layer bounds (column-major order)
            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
                SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let finalMatrix = modelMatrix * scaleMatrix

            // Update uniforms with border mode (renderMode = 1)
            var uniforms = CARendererUniforms(
                mvpMatrix: finalMatrix,
                opacity: presentationLayer.opacity,
                cornerRadius: Float(presentationLayer.cornerRadius),
                layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height)),
                borderWidth: Float(presentationLayer.borderWidth),
                renderMode: 1.0  // Border mode
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(
                uniformBuffer,
                bufferOffset: uniformOffset,
                data: uniformData
            )

            // Create vertices using border color
            let color = presentationLayer.borderColorComponents
            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: color),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: color),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: color),
            ]

            // Write vertices at the correct offset
            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            // Set bind group with dynamic offset for this layer's uniforms
            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Render sublayers (use model layer hierarchy, but presentation layer's sublayerTransform)
        if let sublayers = layer.sublayers {
            var sublayerMatrix = modelMatrix
            if !CATransform3DIsIdentity(presentationLayer.sublayerTransform) {
                sublayerMatrix = sublayerMatrix * presentationLayer.sublayerTransform.matrix4x4
            }

            // Apply masksToBounds clipping if enabled
            let shouldClip = presentationLayer.masksToBounds
            if shouldClip {
                // Calculate the clip rect for this layer
                let layerClipRect = calculateClipRect(layer: presentationLayer, modelMatrix: modelMatrix)

                // Intersect with current clip rect (for nested clipping)
                let newClipRect = currentClipRect.intersection(with: layerClipRect)
                clipRectStack.append(newClipRect)

                // Apply the scissor rect
                applyScissorRect(renderPass)
            }

            // Check if this is a replicator layer
            if let replicatorLayer = presentationLayer as? CAReplicatorLayer {
                renderReplicatorSublayers(
                    replicatorLayer: replicatorLayer,
                    sublayers: sublayers,
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix
                )
            } else {
                for sublayer in sublayers {
                    self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
                }
            }

            // Restore previous clip rect if we pushed one
            if shouldClip {
                _ = clipRectStack.popLast()
                applyScissorRect(renderPass)
            }
        }

        // Clear stencil mask if we used one
        if hasMask {
            clearStencilMask(renderPass: renderPass)
        }
    }

    // MARK: - Mask Rendering (Stencil)

    /// Renders a mask layer to the stencil buffer.
    ///
    /// This writes to the stencil buffer where the mask layer has visible content.
    /// Subsequent layer rendering will only appear where the stencil value matches.
    private func renderMaskToStencil(
        _ maskLayer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        guard let device = device,
              let stencilWritePipeline = stencilWritePipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        // Increment stencil reference for nested masks
        currentStencilValue += 1

        // Switch to stencil write pipeline
        renderPass.setPipeline(stencilWritePipeline)
        renderPass.setStencilReference(currentStencilValue)

        // Render the mask layer (this writes to stencil buffer)
        let maskPresentationLayer = maskLayer.presentation() ?? maskLayer
        let maskModelMatrix = maskPresentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Render mask layer content to stencil
        guard currentLayerIndex < Self.maxLayers else { return }
        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(maskPresentationLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(maskPresentationLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = maskModelMatrix * scaleMatrix

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: 1.0,  // Full opacity for stencil write
            cornerRadius: Float(maskPresentationLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(maskPresentationLayer.bounds.width), Float(maskPresentationLayer.bounds.height))
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: uniformOffset, data: uniformData)

        // Use white color (alpha = 1 everywhere mask is visible)
        let maskColor = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: maskColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: maskColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: maskColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: maskColor),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: maskColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: maskColor),
        ]

        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch to stencil test pipeline for subsequent rendering
        if let stencilTestPipeline = stencilTestPipeline {
            renderPass.setPipeline(stencilTestPipeline)
        }
    }

    /// Clears the stencil mask after layer rendering is complete.
    private func clearStencilMask(renderPass: GPURenderPassEncoder) {
        // Decrement stencil reference for nested masks
        if currentStencilValue > 0 {
            currentStencilValue -= 1
        }

        // Switch back to main pipeline
        if let pipeline = pipeline {
            renderPass.setPipeline(pipeline)
            renderPass.setStencilReference(0)
        }
    }

    // MARK: - Replicator Layer Rendering

    /// Renders the sublayers of a CAReplicatorLayer with instance transformations.
    ///
    /// Supports the following CAReplicatorLayer features:
    /// - instanceCount: Number of copies to render
    /// - instanceTransform: Cumulative transform applied to each instance
    /// - instanceDelay: Animation time offset between instances (staggered animations)
    /// - instanceColor/instanceRedOffset/etc.: Color variations per instance
    private func renderReplicatorSublayers(
        replicatorLayer: CAReplicatorLayer,
        sublayers: [CALayer],
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let instanceCount = max(1, replicatorLayer.instanceCount)
        let instanceTransform = replicatorLayer.instanceTransform
        let instanceDelay = replicatorLayer.instanceDelay

        // Get base instance color (defaults to white if not set)
        let baseColor: SIMD4<Float>
        if let color = replicatorLayer.instanceColor,
           let components = color.components, components.count >= 4 {
            baseColor = SIMD4<Float>(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
        } else {
            baseColor = SIMD4<Float>(1, 1, 1, 1)
        }

        // Color offsets per instance
        let redOffset = replicatorLayer.instanceRedOffset
        let greenOffset = replicatorLayer.instanceGreenOffset
        let blueOffset = replicatorLayer.instanceBlueOffset
        let alphaOffset = replicatorLayer.instanceAlphaOffset

        // Render each instance
        var cumulativeTransform = CATransform3DIdentity
        for instanceIndex in 0..<instanceCount {
            // Calculate color multiplier for this instance
            let colorMultiplier = SIMD4<Float>(
                clamp(baseColor.x + Float(instanceIndex) * redOffset, 0, 1),
                clamp(baseColor.y + Float(instanceIndex) * greenOffset, 0, 1),
                clamp(baseColor.z + Float(instanceIndex) * blueOffset, 0, 1),
                clamp(baseColor.w + Float(instanceIndex) * alphaOffset, 0, 1)
            )

            // Calculate instance matrix
            let instanceMatrix = parentMatrix * cumulativeTransform.matrix4x4

            // Calculate time offset for this instance
            // Positive instanceDelay means later instances start their animations later
            // So instance N's animations are evaluated at (currentTime - N * instanceDelay)
            let timeOffset = CFTimeInterval(instanceIndex) * instanceDelay

            // Render all sublayers with this instance's transform, color, and time offset
            for sublayer in sublayers {
                renderLayerWithColorMultiplierAndTimeOffset(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: instanceMatrix,
                    colorMultiplier: colorMultiplier,
                    timeOffset: timeOffset
                )
            }

            // Apply instance transform for next iteration
            cumulativeTransform = CATransform3DConcat(cumulativeTransform, instanceTransform)
        }
    }

    /// Renders a layer with a color multiplier applied (for replicator instances).
    private func renderLayerWithColorMultiplier(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4,
        colorMultiplier: SIMD4<Float>
    ) {
        guard let device = device,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        let presentationLayer = layer.presentation() ?? layer
        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }
        guard currentLayerIndex < Self.maxLayers else { return }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // For now, render background color with the color multiplier applied
        if presentationLayer.backgroundColor != nil {
            let layerIndex = currentLayerIndex
            currentLayerIndex += 1

            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
                SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let finalMatrix = modelMatrix * scaleMatrix

            var uniforms = CARendererUniforms(
                mvpMatrix: finalMatrix,
                opacity: presentationLayer.opacity,
                cornerRadius: Float(presentationLayer.cornerRadius),
                layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height))
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(
                uniformBuffer,
                bufferOffset: uniformOffset,
                data: uniformData
            )

            // Apply color multiplier to the layer's background color
            var baseColor = presentationLayer.backgroundColorComponents
            baseColor.x *= colorMultiplier.x
            baseColor.y *= colorMultiplier.y
            baseColor.z *= colorMultiplier.z
            baseColor.w *= colorMultiplier.w

            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
            ]

            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Recursively render sublayers
        if let sublayers = layer.sublayers {
            var sublayerMatrix = modelMatrix
            if !CATransform3DIsIdentity(presentationLayer.sublayerTransform) {
                sublayerMatrix = sublayerMatrix * presentationLayer.sublayerTransform.matrix4x4
            }

            for sublayer in sublayers {
                renderLayerWithColorMultiplier(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix,
                    colorMultiplier: colorMultiplier
                )
            }
        }
    }

    /// Renders a layer with a color multiplier and time offset applied (for replicator instances with instanceDelay).
    ///
    /// The time offset is used to evaluate animations at a different point in time,
    /// creating staggered animation effects across replicator instances.
    private func renderLayerWithColorMultiplierAndTimeOffset(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4,
        colorMultiplier: SIMD4<Float>,
        timeOffset: CFTimeInterval
    ) {
        guard let device = device,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup else { return }

        // Get presentation layer at the specified time offset
        let presentationLayer: CALayer
        if timeOffset != 0 {
            presentationLayer = layer.presentationAtTimeOffset(timeOffset)
        } else {
            presentationLayer = layer.presentation() ?? layer
        }

        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else { return }
        guard currentLayerIndex < Self.maxLayers else { return }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Render background color with the color multiplier applied
        if presentationLayer.backgroundColor != nil {
            let layerIndex = currentLayerIndex
            currentLayerIndex += 1

            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(Float(presentationLayer.bounds.width), 0, 0, 0),
                SIMD4<Float>(0, Float(presentationLayer.bounds.height), 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let finalMatrix = modelMatrix * scaleMatrix

            var uniforms = CARendererUniforms(
                mvpMatrix: finalMatrix,
                opacity: presentationLayer.opacity,
                cornerRadius: Float(presentationLayer.cornerRadius),
                layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height))
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(
                uniformBuffer,
                bufferOffset: uniformOffset,
                data: uniformData
            )

            // Apply color multiplier to the layer's background color
            var baseColor = presentationLayer.backgroundColorComponents
            baseColor.x *= colorMultiplier.x
            baseColor.y *= colorMultiplier.y
            baseColor.z *= colorMultiplier.z
            baseColor.w *= colorMultiplier.w

            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
                CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: baseColor),
                CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: baseColor),
                CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: baseColor),
            ]

            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(
                vertexBuffer,
                bufferOffset: vertexOffset,
                data: vertexData
            )

            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }

        // Recursively render sublayers with the same time offset
        if let sublayers = layer.sublayers {
            var sublayerMatrix = modelMatrix
            if !CATransform3DIsIdentity(presentationLayer.sublayerTransform) {
                sublayerMatrix = sublayerMatrix * presentationLayer.sublayerTransform.matrix4x4
            }

            for sublayer in sublayers {
                renderLayerWithColorMultiplierAndTimeOffset(
                    sublayer,
                    renderPass: renderPass,
                    parentMatrix: sublayerMatrix,
                    colorMultiplier: colorMultiplier,
                    timeOffset: timeOffset
                )
            }
        }
    }

    /// Clamps a float value between min and max.
    private func clamp(_ value: Float, _ minVal: Float, _ maxVal: Float) -> Float {
        return min(max(value, minVal), maxVal)
    }

    // MARK: - Contents Layer Rendering (CGImage)

    /// Renders a layer with CGImage contents.
    private func renderContentsLayer(
        _ layer: CALayer,
        contents: CGImage,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let texturedPipeline = texturedPipeline,
              let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        guard currentLayerIndex < Self.maxLayers else { return }

        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        // Get or create GPU texture from CGImage using the texture manager
        let textureKey = ObjectIdentifier(contents)
        let imageWidth = contents.width
        let imageHeight = contents.height
        guard let gpuTexture = textureManager?.getOrCreateTexture(
            for: textureKey,
            width: imageWidth,
            height: imageHeight,
            factory: { [weak self] in
                self?.createGPUTexture(from: contents, device: device)
            }
        ) else { return }

        // Create scale matrix for layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(layer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(layer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Create uniforms for textured rendering
        var uniforms = TexturedUniforms(
            mvpMatrix: finalMatrix,
            opacity: layer.opacity,
            cornerRadius: Float(layer.cornerRadius),
            layerSize: SIMD2<Float>(Float(layer.bounds.width), Float(layer.bounds.height))
        )

        // Write uniforms
        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        // Create vertices with white color (texture will provide color)
        let white = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
        ]

        // Write vertices
        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Create bind group with texture
        let texturedBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: texturedBindGroupLayout,
            entries: [
                GPUBindGroupEntry(
                    binding: 0,
                    resource: .bufferBinding(GPUBufferBinding(
                        buffer: uniformBuffer,
                        size: UInt64(MemoryLayout<TexturedUniforms>.stride)
                    ))
                ),
                GPUBindGroupEntry(
                    binding: 1,
                    resource: .sampler(textureSampler)
                ),
                GPUBindGroupEntry(
                    binding: 2,
                    resource: .textureView(gpuTexture.createView())
                )
            ]
        ))

        // Switch to textured pipeline and render
        renderPass.setPipeline(texturedPipeline)
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch back to non-textured pipeline for subsequent rendering
        if let pipeline = pipeline {
            renderPass.setPipeline(pipeline)
        }
    }

    /// Creates a GPU texture from a CGImage.
    private func createGPUTexture(from cgImage: CGImage, device: GPUDevice) -> GPUTexture? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0 && height > 0 else { return nil }

        // Get RGBA data
        guard let rgbaData = getRGBAData(from: cgImage) else { return nil }

        // Create texture
        let textureDescriptor = GPUTextureDescriptor(
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
            format: .rgba8unorm,
            usage: [.textureBinding, .copyDst, .renderAttachment]
        )

        let texture = device.createTexture(descriptor: textureDescriptor)

        // Create JS Uint8Array from data
        let jsArray = createUint8Array(from: rgbaData)

        // Upload data
        device.queue.writeTexture(
            destination: GPUImageCopyTexture(texture: texture),
            data: jsArray,
            dataLayout: GPUImageDataLayout(
                offset: 0,
                bytesPerRow: UInt32(width * 4),
                rowsPerImage: UInt32(height)
            ),
            size: GPUExtent3D(width: UInt32(width), height: UInt32(height))
        )

        return texture
    }

    /// Converts CGImage data to RGBA format.
    private func getRGBAData(from cgImage: CGImage) -> Data? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0 && height > 0 else { return nil }

        // If CGImage has data, try to use it directly
        guard let sourceData = cgImage.data else { return nil }

        let bytesPerPixel = cgImage.bitsPerPixel / 8
        let sourceBytesPerRow = cgImage.bytesPerRow
        let destBytesPerRow = width * 4
        let totalBytes = destBytesPerRow * height

        // If already RGBA8, use directly
        if bytesPerPixel == 4 && sourceBytesPerRow == destBytesPerRow {
            let isRGBA = cgImage.alphaInfo == .premultipliedLast ||
                        cgImage.alphaInfo == .last ||
                        cgImage.alphaInfo == .noneSkipLast
            if isRGBA {
                return sourceData
            }
        }

        // Need to convert
        var destData = Data(count: totalBytes)

        sourceData.withUnsafeBytes { sourceBuffer in
            destData.withUnsafeMutableBytes { destBuffer in
                guard let sourceBase = sourceBuffer.baseAddress,
                      let destBase = destBuffer.baseAddress else { return }

                for y in 0..<height {
                    for x in 0..<width {
                        let destOffset = y * destBytesPerRow + x * 4
                        let d = destBase.advanced(by: destOffset).assumingMemoryBound(to: UInt8.self)

                        if bytesPerPixel == 4 {
                            let sourceOffset = y * sourceBytesPerRow + x * 4
                            let s = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)

                            // Handle different alpha positions
                            switch cgImage.alphaInfo {
                            case .premultipliedFirst, .first:
                                // ARGB -> RGBA
                                d[0] = s[1]; d[1] = s[2]; d[2] = s[3]; d[3] = s[0]
                            case .noneSkipFirst:
                                // xRGB -> RGBA
                                d[0] = s[1]; d[1] = s[2]; d[2] = s[3]; d[3] = 255
                            case .noneSkipLast:
                                // RGBx -> RGBA
                                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255
                            default:
                                // Assume RGBA
                                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3]
                            }
                        } else if bytesPerPixel == 3 {
                            let sourceOffset = y * sourceBytesPerRow + x * 3
                            let s = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)
                            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255
                        } else if bytesPerPixel == 1 {
                            let sourceOffset = y * sourceBytesPerRow + x
                            let gray = sourceBase.advanced(by: sourceOffset).assumingMemoryBound(to: UInt8.self)[0]
                            d[0] = gray; d[1] = gray; d[2] = gray; d[3] = 255
                        }
                    }
                }
            }
        }

        return destData
    }

    /// Creates a JavaScript Uint8Array from Data.
    private func createUint8Array(from data: Data) -> JSObject {
        let uint8Array = JSObject.global.Uint8Array.function!.new(data.count)
        data.withUnsafeBytes { bytes in
            for i in 0..<data.count {
                uint8Array[i] = .number(Double(bytes.load(fromByteOffset: i, as: UInt8.self)))
            }
        }
        return uint8Array
    }

    /// Clears the texture cache for a specific CGImage.
    public func removeTexture(for cgImage: CGImage) {
        let key = ObjectIdentifier(cgImage)
        textureManager?.removeTexture(for: key)
    }

    /// Clears all cached textures.
    public func clearTextureCache() {
        textureManager?.clearAll()
    }

    // MARK: - Text Layer Rendering

    /// Cache for text textures to avoid recreating them every frame.
    private var textTextureCache: [String: GPUTexture] = [:]

    /// Renders a CATextLayer using Canvas2D for text rasterization and texture-based rendering.
    ///
    /// This method uses an offscreen Canvas2D to render the text, creates a WebGPU texture
    /// from the canvas data, and displays it using the textured pipeline.
    private func renderTextLayer(
        _ textLayer: CATextLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let string = textLayer.string else { return }
        guard let texturedPipeline = texturedPipeline,
              let texturedBindGroupLayout = texturedBindGroupLayout,
              let textureSampler = textureSampler,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let pipeline = pipeline else { return }

        let text: String
        if let str = string as? String {
            text = str
        } else {
            text = String(describing: string)
        }

        guard !text.isEmpty else { return }
        guard currentLayerIndex < Self.maxLayers else { return }

        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        let width = Int(textLayer.bounds.width)
        let height = Int(textLayer.bounds.height)

        guard width > 0 && height > 0 else { return }

        // Create cache key based on text content and properties
        let cacheKey = "\(text)_\(width)x\(height)_\(textLayer.fontSize)_\(textLayer.alignmentMode.rawValue)"

        // Check cache first
        let gpuTexture: GPUTexture
        if let cached = textTextureCache[cacheKey] {
            gpuTexture = cached
        } else {
            // Create offscreen canvas for text rendering
            let document = JSObject.global.document
            let offscreenCanvas = document.createElement("canvas")
            offscreenCanvas.width = .number(Double(width))
            offscreenCanvas.height = .number(Double(height))

            guard let ctx = offscreenCanvas.getContext("2d").object else { return }

            // Clear canvas with background color if set
            if let bgColor = textLayer.backgroundColor {
                let components = bgColor.components ?? [0, 0, 0, 1]
                let r = Int((components.count > 0 ? components[0] : 0) * 255)
                let g = Int((components.count > 1 ? components[1] : 0) * 255)
                let b = Int((components.count > 2 ? components[2] : 0) * 255)
                let a = components.count > 3 ? components[3] : 1.0
                ctx.fillStyle = .string("rgba(\(r),\(g),\(b),\(a))")
                _ = ctx.fillRect!(0, 0, width, height)
            } else {
                _ = ctx.clearRect!(0, 0, width, height)
            }

            // Set font
            let fontName: String
            if let font = textLayer.font as? String {
                fontName = font
            } else {
                fontName = "sans-serif"
            }
            ctx.font = .string("\(Int(textLayer.fontSize))px \(fontName)")

            // Set text color
            if let fgColor = textLayer.foregroundColor {
                let components = fgColor.components ?? [0, 0, 0, 1]
                let r = Int((components.count > 0 ? components[0] : 0) * 255)
                let g = Int((components.count > 1 ? components[1] : 0) * 255)
                let b = Int((components.count > 2 ? components[2] : 0) * 255)
                let a = components.count > 3 ? components[3] : 1.0
                ctx.fillStyle = .string("rgba(\(r),\(g),\(b),\(a))")
            } else {
                ctx.fillStyle = .string("rgba(255,255,255,1)")
            }

            // Set text alignment
            switch textLayer.alignmentMode {
            case .left:
                ctx.textAlign = .string("left")
            case .right:
                ctx.textAlign = .string("right")
            case .center:
                ctx.textAlign = .string("center")
            case .justified, .natural:
                ctx.textAlign = .string("start")
            default:
                ctx.textAlign = .string("start")
            }

            ctx.textBaseline = .string("top")

            // Calculate text position based on alignment
            let x: Double
            switch textLayer.alignmentMode {
            case .center:
                x = Double(width) / 2
            case .right:
                x = Double(width)
            default:
                x = 0
            }

            // Draw text (simple single-line for now)
            // For wrapped text, we would need to implement line breaking
            if textLayer.isWrapped {
                // Simple word wrapping
                drawWrappedText(ctx: ctx, text: text, x: x, y: 0,
                               maxWidth: Double(width), lineHeight: textLayer.fontSize * 1.2)
            } else {
                _ = ctx.fillText!(text, x, Double(textLayer.fontSize * 0.1))
            }

            // Get image data from canvas
            guard let imageData = ctx.getImageData!(0, 0, width, height).object else { return }
            guard let dataArray = imageData.data.object else { return }

            // Create WebGPU texture
            let textureDescriptor = GPUTextureDescriptor(
                size: GPUExtent3D(width: UInt32(width), height: UInt32(height)),
                format: .rgba8unorm,
                usage: [.textureBinding, .copyDst, .renderAttachment]
            )

            let texture = device.createTexture(descriptor: textureDescriptor)

            // Copy image data to texture
            device.queue.writeTexture(
                destination: GPUImageCopyTexture(texture: texture),
                data: dataArray,
                dataLayout: GPUImageDataLayout(
                    offset: 0,
                    bytesPerRow: UInt32(width * 4),
                    rowsPerImage: UInt32(height)
                ),
                size: GPUExtent3D(width: UInt32(width), height: UInt32(height))
            )

            // Cache the texture
            textTextureCache[cacheKey] = texture
            gpuTexture = texture
        }

        // Create texture view
        let textureView = gpuTexture.createView()

        // Create bind group with the texture
        let texturedBindGroup = device.createBindGroup(
            descriptor: GPUBindGroupDescriptor(
                layout: texturedBindGroupLayout,
                entries: [
                    GPUBindGroupEntry(binding: 0, resource: .buffer(GPUBufferBinding(buffer: uniformBuffer))),
                    GPUBindGroupEntry(binding: 1, resource: .sampler(textureSampler)),
                    GPUBindGroupEntry(binding: 2, resource: .textureView(textureView))
                ]
            )
        )

        // Setup matrices
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(textLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(textLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Update uniforms
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: textLayer.opacity,
            cornerRadius: Float(textLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(textLayer.bounds.width), Float(textLayer.bounds.height))
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        // White vertex color for texture rendering (texture provides actual color)
        let white = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
        ]

        // Write vertices
        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Switch to textured pipeline and draw
        renderPass.setPipeline(texturedPipeline)
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)

        // Switch back to standard pipeline
        renderPass.setPipeline(pipeline)
    }

    /// Draws wrapped text on a Canvas2D context.
    private func drawWrappedText(ctx: JSObject, text: String, x: Double, y: Double,
                                 maxWidth: Double, lineHeight: CGFloat) {
        let words = text.split(separator: " ")
        var line = ""
        var currentY = y

        for word in words {
            let testLine = line.isEmpty ? String(word) : line + " " + String(word)
            let metrics = ctx.measureText!(testLine)
            let testWidth = metrics.width.number ?? 0

            if testWidth > maxWidth && !line.isEmpty {
                _ = ctx.fillText!(line, x, currentY)
                line = String(word)
                currentY += Double(lineHeight)
            } else {
                line = testLine
            }
        }

        if !line.isEmpty {
            _ = ctx.fillText!(line, x, currentY)
        }
    }

    // MARK: - Shape Layer Rendering

    /// Flattens a CGPath into an array of polylines (arrays of points).
    /// Each subpath becomes a separate polyline.
    private func flattenPath(_ path: CGPath, flatness: CGFloat = 0.5) -> [[CGPoint]] {
        var polylines: [[CGPoint]] = []
        var currentPolyline: [CGPoint] = []
        var currentPoint = CGPoint.zero
        var subpathStart = CGPoint.zero

        path.applyWithBlock { elementPtr in
            let element = elementPtr.pointee
            switch element.type {
            case .moveToPoint:
                if !currentPolyline.isEmpty {
                    polylines.append(currentPolyline)
                }
                currentPolyline = []
                let point = element.points![0]
                currentPolyline.append(point)
                currentPoint = point
                subpathStart = point

            case .addLineToPoint:
                let point = element.points![0]
                currentPolyline.append(point)
                currentPoint = point

            case .addQuadCurveToPoint:
                let control = element.points![0]
                let end = element.points![1]
                // Flatten quadratic bezier
                self.flattenQuadBezier(from: currentPoint, control: control, to: end,
                                       flatness: flatness, into: &currentPolyline)
                currentPoint = end

            case .addCurveToPoint:
                let control1 = element.points![0]
                let control2 = element.points![1]
                let end = element.points![2]
                // Flatten cubic bezier
                self.flattenCubicBezier(from: currentPoint, control1: control1,
                                        control2: control2, to: end,
                                        flatness: flatness, into: &currentPolyline)
                currentPoint = end

            case .closeSubpath:
                if !currentPolyline.isEmpty && currentPolyline.first != currentPolyline.last {
                    currentPolyline.append(subpathStart)
                }
                currentPoint = subpathStart

            @unknown default:
                break
            }
        }

        if !currentPolyline.isEmpty {
            polylines.append(currentPolyline)
        }

        return polylines
    }

    /// Flattens a quadratic bezier curve into line segments.
    private func flattenQuadBezier(from p0: CGPoint, control p1: CGPoint, to p2: CGPoint,
                                   flatness: CGFloat, into polyline: inout [CGPoint]) {
        // Simple recursive subdivision
        let midX = (p0.x + 2 * p1.x + p2.x) / 4
        let midY = (p0.y + 2 * p1.y + p2.y) / 4
        let dx = (p0.x + p2.x) / 2 - midX
        let dy = (p0.y + p2.y) / 2 - midY

        if dx * dx + dy * dy <= flatness * flatness {
            polyline.append(p2)
        } else {
            let p01 = CGPoint(x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2)
            let p12 = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
            let mid = CGPoint(x: (p01.x + p12.x) / 2, y: (p01.y + p12.y) / 2)
            flattenQuadBezier(from: p0, control: p01, to: mid, flatness: flatness, into: &polyline)
            flattenQuadBezier(from: mid, control: p12, to: p2, flatness: flatness, into: &polyline)
        }
    }

    /// Flattens a cubic bezier curve into line segments.
    private func flattenCubicBezier(from p0: CGPoint, control1 p1: CGPoint,
                                    control2 p2: CGPoint, to p3: CGPoint,
                                    flatness: CGFloat, into polyline: inout [CGPoint]) {
        // Check if curve is flat enough
        let dx1 = p1.x - p0.x
        let dy1 = p1.y - p0.y
        let dx2 = p2.x - p3.x
        let dy2 = p2.y - p3.y

        let d = sqrt(max(dx1 * dx1 + dy1 * dy1, dx2 * dx2 + dy2 * dy2))

        if d <= flatness {
            polyline.append(p3)
        } else {
            // Subdivide
            let p01 = CGPoint(x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2)
            let p12 = CGPoint(x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2)
            let p23 = CGPoint(x: (p2.x + p3.x) / 2, y: (p2.y + p3.y) / 2)
            let p012 = CGPoint(x: (p01.x + p12.x) / 2, y: (p01.y + p12.y) / 2)
            let p123 = CGPoint(x: (p12.x + p23.x) / 2, y: (p12.y + p23.y) / 2)
            let mid = CGPoint(x: (p012.x + p123.x) / 2, y: (p012.y + p123.y) / 2)

            flattenCubicBezier(from: p0, control1: p01, control2: p012, to: mid,
                               flatness: flatness, into: &polyline)
            flattenCubicBezier(from: mid, control1: p123, control2: p23, to: p3,
                               flatness: flatness, into: &polyline)
        }
    }

    /// Triangulates a simple polygon using ear-clipping algorithm.
    /// Returns array of vertex indices forming triangles.
    private func triangulatePolygon(_ polygon: [CGPoint]) -> [Int] {
        guard polygon.count >= 3 else { return [] }

        // Remove duplicate last point if closed
        var points = polygon
        if let first = points.first, let last = points.last,
           first.x == last.x && first.y == last.y && points.count > 1 {
            points.removeLast()
        }

        guard points.count >= 3 else { return [] }

        var indices: [Int] = []
        var remaining = Array(0..<points.count)

        // Ensure polygon is counter-clockwise
        let area = signedArea(points)
        if area < 0 {
            remaining.reverse()
        }

        var safetyCounter = remaining.count * remaining.count

        while remaining.count > 3 && safetyCounter > 0 {
            safetyCounter -= 1
            var earFound = false

            for i in 0..<remaining.count {
                let prev = (i + remaining.count - 1) % remaining.count
                let next = (i + 1) % remaining.count

                let a = points[remaining[prev]]
                let b = points[remaining[i]]
                let c = points[remaining[next]]

                if isEar(a: a, b: b, c: c, polygon: points, remaining: remaining) {
                    indices.append(remaining[prev])
                    indices.append(remaining[i])
                    indices.append(remaining[next])
                    remaining.remove(at: i)
                    earFound = true
                    break
                }
            }

            if !earFound {
                // Fallback: just use first three vertices as triangle
                if remaining.count >= 3 {
                    indices.append(remaining[0])
                    indices.append(remaining[1])
                    indices.append(remaining[2])
                    remaining.remove(at: 1)
                }
            }
        }

        // Add final triangle
        if remaining.count == 3 {
            indices.append(remaining[0])
            indices.append(remaining[1])
            indices.append(remaining[2])
        }

        return indices
    }

    /// Calculates the signed area of a polygon.
    private func signedArea(_ polygon: [CGPoint]) -> CGFloat {
        var area: CGFloat = 0
        let n = polygon.count
        for i in 0..<n {
            let j = (i + 1) % n
            area += polygon[i].x * polygon[j].y
            area -= polygon[j].x * polygon[i].y
        }
        return area / 2
    }

    /// Checks if vertex b is an ear (can be clipped).
    private func isEar(a: CGPoint, b: CGPoint, c: CGPoint,
                       polygon: [CGPoint], remaining: [Int]) -> Bool {
        // Check if triangle is convex (counter-clockwise)
        let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if cross <= 0 {
            return false
        }

        // Check if any other point is inside the triangle
        for idx in remaining {
            let p = polygon[idx]
            if (p.x == a.x && p.y == a.y) ||
               (p.x == b.x && p.y == b.y) ||
               (p.x == c.x && p.y == c.y) {
                continue
            }
            if pointInTriangle(p: p, a: a, b: b, c: c) {
                return false
            }
        }

        return true
    }

    /// Checks if point p is inside triangle abc.
    private func pointInTriangle(p: CGPoint, a: CGPoint, b: CGPoint, c: CGPoint) -> Bool {
        let v0x = c.x - a.x
        let v0y = c.y - a.y
        let v1x = b.x - a.x
        let v1y = b.y - a.y
        let v2x = p.x - a.x
        let v2y = p.y - a.y

        let dot00 = v0x * v0x + v0y * v0y
        let dot01 = v0x * v1x + v0y * v1y
        let dot02 = v0x * v2x + v0y * v2y
        let dot11 = v1x * v1x + v1y * v1y
        let dot12 = v1x * v2x + v1y * v2y

        let invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        let u = (dot11 * dot02 - dot01 * dot12) * invDenom
        let v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= 0) && (v >= 0) && (u + v <= 1)
    }

    /// Generates stroke geometry for a polyline.
    /// Returns vertices forming triangle strips along the path.
    private func generateStrokeGeometry(polyline: [CGPoint], lineWidth: CGFloat,
                                        lineCap: CAShapeLayerLineCap,
                                        lineJoin: CAShapeLayerLineJoin) -> [CGPoint] {
        guard polyline.count >= 2 else { return [] }

        let halfWidth = lineWidth / 2
        var leftPoints: [CGPoint] = []
        var rightPoints: [CGPoint] = []

        for i in 0..<polyline.count {
            let curr = polyline[i]

            // Calculate tangent
            var tangent: CGPoint
            if i == 0 {
                let next = polyline[i + 1]
                tangent = CGPoint(x: next.x - curr.x, y: next.y - curr.y)
            } else if i == polyline.count - 1 {
                let prev = polyline[i - 1]
                tangent = CGPoint(x: curr.x - prev.x, y: curr.y - prev.y)
            } else {
                let prev = polyline[i - 1]
                let next = polyline[i + 1]
                tangent = CGPoint(x: next.x - prev.x, y: next.y - prev.y)
            }

            // Normalize tangent
            let len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y)
            if len > 0.0001 {
                tangent.x /= len
                tangent.y /= len
            }

            // Calculate normal (perpendicular)
            let normal = CGPoint(x: -tangent.y, y: tangent.x)

            // Offset points
            leftPoints.append(CGPoint(x: curr.x + normal.x * halfWidth,
                                      y: curr.y + normal.y * halfWidth))
            rightPoints.append(CGPoint(x: curr.x - normal.x * halfWidth,
                                       y: curr.y - normal.y * halfWidth))
        }

        // Build triangle strip as triangles
        var vertices: [CGPoint] = []
        for i in 0..<(leftPoints.count - 1) {
            // Triangle 1
            vertices.append(leftPoints[i])
            vertices.append(rightPoints[i])
            vertices.append(leftPoints[i + 1])

            // Triangle 2
            vertices.append(rightPoints[i])
            vertices.append(rightPoints[i + 1])
            vertices.append(leftPoints[i + 1])
        }

        // Add end caps
        if lineCap == .round {
            // Round cap at start
            let startCenter = polyline[0]
            let startTangent = CGPoint(x: polyline[1].x - polyline[0].x,
                                       y: polyline[1].y - polyline[0].y)
            addRoundCap(center: startCenter, tangent: startTangent, halfWidth: halfWidth,
                        isStart: true, into: &vertices)

            // Round cap at end
            let endCenter = polyline[polyline.count - 1]
            let endTangent = CGPoint(x: polyline[polyline.count - 1].x - polyline[polyline.count - 2].x,
                                     y: polyline[polyline.count - 1].y - polyline[polyline.count - 2].y)
            addRoundCap(center: endCenter, tangent: endTangent, halfWidth: halfWidth,
                        isStart: false, into: &vertices)
        } else if lineCap == .square {
            // Square caps extend by halfWidth
            let startCenter = polyline[0]
            let startDir = CGPoint(x: polyline[0].x - polyline[1].x,
                                   y: polyline[0].y - polyline[1].y)
            let startLen = sqrt(startDir.x * startDir.x + startDir.y * startDir.y)
            if startLen > 0 {
                let startNorm = CGPoint(x: startDir.x / startLen, y: startDir.y / startLen)
                let extendedStart = CGPoint(x: startCenter.x + startNorm.x * halfWidth,
                                            y: startCenter.y + startNorm.y * halfWidth)
                let perpStart = CGPoint(x: -startNorm.y, y: startNorm.x)

                vertices.append(leftPoints[0])
                vertices.append(CGPoint(x: extendedStart.x + perpStart.x * halfWidth,
                                        y: extendedStart.y + perpStart.y * halfWidth))
                vertices.append(rightPoints[0])

                vertices.append(rightPoints[0])
                vertices.append(CGPoint(x: extendedStart.x + perpStart.x * halfWidth,
                                        y: extendedStart.y + perpStart.y * halfWidth))
                vertices.append(CGPoint(x: extendedStart.x - perpStart.x * halfWidth,
                                        y: extendedStart.y - perpStart.y * halfWidth))
            }

            let endCenter = polyline[polyline.count - 1]
            let endDir = CGPoint(x: polyline[polyline.count - 1].x - polyline[polyline.count - 2].x,
                                 y: polyline[polyline.count - 1].y - polyline[polyline.count - 2].y)
            let endLen = sqrt(endDir.x * endDir.x + endDir.y * endDir.y)
            if endLen > 0 {
                let endNorm = CGPoint(x: endDir.x / endLen, y: endDir.y / endLen)
                let extendedEnd = CGPoint(x: endCenter.x + endNorm.x * halfWidth,
                                          y: endCenter.y + endNorm.y * halfWidth)
                let perpEnd = CGPoint(x: -endNorm.y, y: endNorm.x)

                vertices.append(leftPoints[leftPoints.count - 1])
                vertices.append(CGPoint(x: extendedEnd.x + perpEnd.x * halfWidth,
                                        y: extendedEnd.y + perpEnd.y * halfWidth))
                vertices.append(rightPoints[rightPoints.count - 1])

                vertices.append(rightPoints[rightPoints.count - 1])
                vertices.append(CGPoint(x: extendedEnd.x + perpEnd.x * halfWidth,
                                        y: extendedEnd.y + perpEnd.y * halfWidth))
                vertices.append(CGPoint(x: extendedEnd.x - perpEnd.x * halfWidth,
                                        y: extendedEnd.y - perpEnd.y * halfWidth))
            }
        }

        return vertices
    }

    /// Adds a round cap to the stroke geometry.
    private func addRoundCap(center: CGPoint, tangent: CGPoint, halfWidth: CGFloat,
                             isStart: Bool, into vertices: inout [CGPoint]) {
        let len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y)
        guard len > 0 else { return }

        let dir = CGPoint(x: tangent.x / len, y: tangent.y / len)
        let normal = CGPoint(x: -dir.y, y: dir.x)

        // Create semicircle with 8 segments
        let segments = 8
        let startAngle: CGFloat = isStart ? CGFloat.pi / 2 : -CGFloat.pi / 2
        let endAngle: CGFloat = isStart ? 3 * CGFloat.pi / 2 : CGFloat.pi / 2

        for i in 0..<segments {
            let angle1 = startAngle + CGFloat(i) * (endAngle - startAngle) / CGFloat(segments)
            let angle2 = startAngle + CGFloat(i + 1) * (endAngle - startAngle) / CGFloat(segments)

            let p1 = CGPoint(
                x: center.x + halfWidth * (cos(angle1) * normal.x - sin(angle1) * dir.x),
                y: center.y + halfWidth * (cos(angle1) * normal.y - sin(angle1) * dir.y)
            )
            let p2 = CGPoint(
                x: center.x + halfWidth * (cos(angle2) * normal.x - sin(angle2) * dir.x),
                y: center.y + halfWidth * (cos(angle2) * normal.y - sin(angle2) * dir.y)
            )

            vertices.append(center)
            vertices.append(p1)
            vertices.append(p2)
        }
    }

    /// Renders a CAShapeLayer with its path.
    private func renderShapeLayer(
        _ shapeLayer: CAShapeLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let path = shapeLayer.path else { return }
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        // Flatten the path
        let polylines = flattenPath(path)
        guard !polylines.isEmpty else { return }

        // Render fill if fillColor is set
        if let fillColor = shapeLayer.fillColor {
            for polyline in polylines {
                guard polyline.count >= 3 else { continue }

                // Triangulate the polygon
                let indices = triangulatePolygon(polyline)
                guard !indices.isEmpty else { continue }

                guard currentLayerIndex < Self.maxLayers else { return }
                let layerIndex = currentLayerIndex
                currentLayerIndex += 1

                // Create vertices from triangulation
                var vertices: [CARendererVertex] = []
                let colorComponents = cgColorToSIMD4(fillColor)

                for idx in indices {
                    let point = polyline[idx]
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: SIMD2(0, 0),  // Not used for solid color
                        color: colorComponents
                    ))
                }

                guard !vertices.isEmpty else { continue }

                // Update uniforms
                var uniforms = CARendererUniforms(
                    mvpMatrix: modelMatrix,
                    opacity: shapeLayer.opacity,
                    cornerRadius: 0,
                    layerSize: .zero  // No SDF-based corner radius for shapes
                )

                let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
                let uniformData = createFloat32Array(from: &uniforms)
                device.queue.writeBuffer(
                    uniformBuffer,
                    bufferOffset: uniformOffset,
                    data: uniformData
                )

                // Write vertices
                let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
                let vertexData = createFloat32Array(from: &vertices)
                device.queue.writeBuffer(
                    vertexBuffer,
                    bufferOffset: vertexOffset,
                    data: vertexData
                )

                // Draw
                renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
                renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
                renderPass.draw(vertexCount: UInt32(vertices.count))
            }
        }

        // Render stroke if strokeColor is set
        if let strokeColor = shapeLayer.strokeColor, shapeLayer.lineWidth > 0 {
            for polyline in polylines {
                guard polyline.count >= 2 else { continue }

                // Generate stroke geometry
                let strokeVertices = generateStrokeGeometry(
                    polyline: polyline,
                    lineWidth: shapeLayer.lineWidth,
                    lineCap: shapeLayer.lineCap,
                    lineJoin: shapeLayer.lineJoin
                )
                guard !strokeVertices.isEmpty else { continue }

                guard currentLayerIndex < Self.maxLayers else { return }
                let layerIndex = currentLayerIndex
                currentLayerIndex += 1

                // Create vertices
                var vertices: [CARendererVertex] = []
                let colorComponents = cgColorToSIMD4(strokeColor)

                for point in strokeVertices {
                    vertices.append(CARendererVertex(
                        position: SIMD2(Float(point.x), Float(point.y)),
                        texCoord: SIMD2(0, 0),
                        color: colorComponents
                    ))
                }

                // Update uniforms
                var uniforms = CARendererUniforms(
                    mvpMatrix: modelMatrix,
                    opacity: shapeLayer.opacity,
                    cornerRadius: 0,
                    layerSize: .zero
                )

                let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
                let uniformData = createFloat32Array(from: &uniforms)
                device.queue.writeBuffer(
                    uniformBuffer,
                    bufferOffset: uniformOffset,
                    data: uniformData
                )

                // Write vertices
                let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
                let vertexData = createFloat32Array(from: &vertices)
                device.queue.writeBuffer(
                    vertexBuffer,
                    bufferOffset: vertexOffset,
                    data: vertexData
                )

                // Draw
                renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
                renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
                renderPass.draw(vertexCount: UInt32(vertices.count))
            }
        }
    }

    /// Converts CGColor to SIMD4<Float>.
    private func cgColorToSIMD4(_ color: CGColor) -> SIMD4<Float> {
        guard let components = color.components, components.count >= 4 else {
            return SIMD4<Float>(0, 0, 0, 1)
        }
        return SIMD4<Float>(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
    }

    // MARK: - Gradient Layer Rendering

    /// Renders a CAGradientLayer with its gradient colors.
    private func renderGradientLayer(
        _ gradientLayer: CAGradientLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let colors = gradientLayer.colors, !colors.isEmpty else { return }

        guard currentLayerIndex < Self.maxLayers else { return }

        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        // Create scale matrix for layer bounds
        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(gradientLayer.bounds.width), 0, 0, 0),
            SIMD4<Float>(0, Float(gradientLayer.bounds.height), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * scaleMatrix

        // Prepare gradient uniforms
        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: gradientLayer.opacity,
            cornerRadius: Float(gradientLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(gradientLayer.bounds.width), Float(gradientLayer.bounds.height)),
            borderWidth: 0,
            renderMode: 2.0,  // Gradient mode
            gradientStartPoint: SIMD2<Float>(Float(gradientLayer.startPoint.x), Float(gradientLayer.startPoint.y)),
            gradientEndPoint: SIMD2<Float>(Float(gradientLayer.endPoint.x), Float(gradientLayer.endPoint.y)),
            gradientColorCount: Float(min(colors.count, kMaxGradientStops))
        )

        // Extract gradient colors
        var gradientColors: [SIMD4<Float>] = []
        for colorAny in colors.prefix(kMaxGradientStops) {
            if let cgColor = colorAny as? CGColor,
               let components = cgColor.components,
               components.count >= 4 {
                gradientColors.append(SIMD4<Float>(
                    Float(components[0]),
                    Float(components[1]),
                    Float(components[2]),
                    Float(components[3])
                ))
            } else {
                gradientColors.append(SIMD4<Float>(0, 0, 0, 1))
            }
        }

        // Pad to 8 colors
        while gradientColors.count < kMaxGradientStops {
            gradientColors.append(.zero)
        }

        uniforms.gradientColors = (
            gradientColors[0], gradientColors[1], gradientColors[2], gradientColors[3],
            gradientColors[4], gradientColors[5], gradientColors[6], gradientColors[7]
        )

        // Extract or generate locations
        let colorCount = min(colors.count, kMaxGradientStops)
        var locations: [Float] = []
        if let providedLocations = gradientLayer.locations, !providedLocations.isEmpty {
            locations = providedLocations.prefix(kMaxGradientStops).map { Float($0) }
        } else {
            // Generate evenly spaced locations
            for i in 0..<colorCount {
                locations.append(Float(i) / Float(max(colorCount - 1, 1)))
            }
        }

        // Pad locations to 8
        while locations.count < kMaxGradientStops {
            locations.append(1.0)
        }

        uniforms.gradientLocations = SIMD4<Float>(locations[0], locations[1], locations[2], locations[3])
        uniforms.gradientLocations2 = SIMD4<Float>(locations[4], locations[5], locations[6], locations[7])

        // Write uniforms
        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        // Create vertices (color doesn't matter for gradient, shader uses gradient uniforms)
        let dummyColor = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: dummyColor),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: dummyColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: dummyColor),
        ]

        // Write vertices
        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        // Draw
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    // MARK: - Shadow Rendering (2-Pass Gaussian Blur)

    /// Pre-renders shadows with 2-pass Gaussian blur before the main render pass.
    ///
    /// This method finds the first layer needing a shadow and renders it with blur.
    /// Currently supports one shadow per frame due to texture sharing constraints.
    /// For multiple shadows, a texture pool would be needed.
    private func prerenderShadows(
        _ rootLayer: CALayer,
        encoder: GPUCommandEncoder,
        projectionMatrix: Matrix4x4
    ) {
        guard let device = device,
              let pipeline = pipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let shadowMaskTexture = shadowMaskTexture,
              let shadowBlurTexture = shadowBlurTexture,
              let shadowBlurHorizontalPipeline = shadowBlurHorizontalPipeline,
              let shadowBlurVerticalPipeline = shadowBlurVerticalPipeline,
              let blurHorizontalBindGroup = blurHorizontalBindGroup,
              let blurVerticalBindGroup = blurVerticalBindGroup,
              let blurUniformBuffer = blurUniformBuffer else { return }

        // Find first layer with shadow to pre-render
        guard let (shadowLayer, layerMatrix) = findFirstShadowLayer(rootLayer, parentMatrix: projectionMatrix) else {
            return
        }

        let presentationLayer = shadowLayer.presentation() ?? shadowLayer
        let shadowRadius = presentationLayer.shadowRadius
        let shadowOffset = presentationLayer.shadowOffset

        // Calculate expanded bounds for shadow (includes blur radius)
        let expandedWidth = presentationLayer.bounds.width + shadowRadius * 4
        let expandedHeight = presentationLayer.bounds.height + shadowRadius * 4

        // Step 1: Render layer shape to shadow mask texture
        let maskRenderPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: shadowMaskTexture.createView(),
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        // Create mask uniforms - render as white shape
        let offsetMatrix = Matrix4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(Float(shadowOffset.width), Float(shadowOffset.height), 0, 1)
        ))
        let finalMatrix = layerMatrix * offsetMatrix

        var maskUniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: 1.0,
            cornerRadius: Float(presentationLayer.cornerRadius),
            layerSize: SIMD2<Float>(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height))
        )

        // Write mask uniforms
        let maskUniformData = createFloat32Array(from: &maskUniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: 0, data: maskUniformData)

        // Create white vertices for the mask
        let whiteColor = SIMD4<Float>(1, 1, 1, 1)
        var maskVertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: whiteColor),
            CARendererVertex(position: SIMD2(Float(presentationLayer.bounds.width), 0), texCoord: SIMD2(1, 0), color: whiteColor),
            CARendererVertex(position: SIMD2(0, Float(presentationLayer.bounds.height)), texCoord: SIMD2(0, 1), color: whiteColor),
            CARendererVertex(position: SIMD2(Float(presentationLayer.bounds.width), 0), texCoord: SIMD2(1, 0), color: whiteColor),
            CARendererVertex(position: SIMD2(Float(presentationLayer.bounds.width), Float(presentationLayer.bounds.height)), texCoord: SIMD2(1, 1), color: whiteColor),
            CARendererVertex(position: SIMD2(0, Float(presentationLayer.bounds.height)), texCoord: SIMD2(0, 1), color: whiteColor),
        ]

        let maskVertexData = createFloat32Array(from: &maskVertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: 0, data: maskVertexData)

        maskRenderPass.setPipeline(pipeline)
        if let bindGroup = bindGroup {
            maskRenderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [0])
        }
        maskRenderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: 0)
        maskRenderPass.draw(vertexCount: 6)
        maskRenderPass.end()

        // Step 2: Horizontal blur (shadowMaskTexture → shadowBlurTexture)
        var blurUniforms = BlurUniforms(
            texelSize: SIMD2<Float>(1.0 / Float(size.width), 1.0 / Float(size.height)),
            blurRadius: Float(shadowRadius) * 0.5
        )
        let blurUniformData = createFloat32Array(from: &blurUniforms)
        device.queue.writeBuffer(blurUniformBuffer, bufferOffset: 0, data: blurUniformData)

        let hBlurPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: shadowBlurTexture.createView(),
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        hBlurPass.setPipeline(shadowBlurHorizontalPipeline)
        hBlurPass.setBindGroup(0, bindGroup: blurHorizontalBindGroup)
        hBlurPass.draw(vertexCount: 6)
        hBlurPass.end()

        // Step 3: Vertical blur (shadowBlurTexture → shadowMaskTexture)
        let vBlurPass = encoder.beginRenderPass(descriptor: GPURenderPassDescriptor(
            colorAttachments: [
                GPURenderPassColorAttachment(
                    view: shadowMaskTexture.createView(),
                    clearValue: GPUColor(r: 0, g: 0, b: 0, a: 0),
                    loadOp: .clear,
                    storeOp: .store
                )
            ]
        ))

        vBlurPass.setPipeline(shadowBlurVerticalPipeline)
        vBlurPass.setBindGroup(0, bindGroup: blurVerticalBindGroup)
        vBlurPass.draw(vertexCount: 6)
        vBlurPass.end()

        // Mark that shadow was pre-rendered
        hasPrerenderredShadow = true
        shadowsPrerendered = true
    }

    /// Finds the first layer in the hierarchy that has a visible shadow.
    private func findFirstShadowLayer(
        _ layer: CALayer,
        parentMatrix: Matrix4x4
    ) -> (layer: CALayer, matrix: Matrix4x4)? {
        let presentationLayer = layer.presentation() ?? layer

        guard !presentationLayer.isHidden && presentationLayer.opacity > 0 else {
            return nil
        }

        let modelMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Check if this layer has a shadow
        if presentationLayer.shadowOpacity > 0 && presentationLayer.shadowColor != nil {
            return (layer, modelMatrix)
        }

        // Recursively check sublayers
        if let sublayers = layer.sublayers {
            var sublayerMatrix = modelMatrix
            if !CATransform3DIsIdentity(presentationLayer.sublayerTransform) {
                sublayerMatrix = sublayerMatrix * presentationLayer.sublayerTransform.matrix4x4
            }

            for sublayer in sublayers {
                if let result = findFirstShadowLayer(sublayer, parentMatrix: sublayerMatrix) {
                    return result
                }
            }
        }

        return nil
    }

    /// Renders the shadow for a layer using the pre-blurred shadow texture.
    ///
    /// If the shadow was pre-rendered with 2-pass Gaussian blur, this composites
    /// the blurred texture. Otherwise, falls back to a simplified shadow approximation.
    private func renderLayerShadow(
        _ layer: CALayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4
    ) {
        guard let shadowColor = layer.shadowColor,
              let pipeline = pipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let bindGroup = bindGroup,
              currentLayerIndex < Self.maxLayers else { return }

        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        // Get shadow properties
        let shadowOpacity = layer.shadowOpacity
        let shadowOffset = layer.shadowOffset
        let shadowRadius = layer.shadowRadius

        // Get shadow color components
        let colorComponents: SIMD4<Float>
        if let components = shadowColor.components, components.count >= 3 {
            colorComponents = SIMD4<Float>(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                components.count > 3 ? Float(components[3]) * shadowOpacity : shadowOpacity
            )
        } else {
            colorComponents = SIMD4<Float>(0, 0, 0, shadowOpacity)
        }

        // If shadow was pre-rendered with blur, use the textured composite pipeline
        if hasPrerenderredShadow,
           let shadowMaskTexture = shadowMaskTexture,
           let shadowCompositePipeline = shadowCompositePipeline,
           let blurSampler = blurSampler {

            // Create shadow uniforms for the composite shader
            var shadowUniforms = ShadowUniforms(
                mvpMatrix: Matrix4x4.orthographic(
                    left: 0, right: Float(size.width),
                    bottom: Float(size.height), top: 0,
                    near: -1000, far: 1000
                ),
                shadowColor: colorComponents,
                shadowOffset: SIMD2<Float>(Float(shadowOffset.width), Float(shadowOffset.height)),
                layerSize: SIMD2<Float>(Float(size.width), Float(size.height))
            )

            // Create a temporary buffer for shadow uniforms
            let shadowUniformBuffer = device.createBuffer(descriptor: GPUBufferDescriptor(
                size: UInt64(MemoryLayout<ShadowUniforms>.stride),
                usage: [.uniform, .copyDst]
            ))

            let shadowUniformData = createFloat32Array(from: &shadowUniforms)
            device.queue.writeBuffer(shadowUniformBuffer, bufferOffset: 0, data: shadowUniformData)

            // Create shadow composite bind group with the blurred texture
            let compositeBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
                layout: shadowCompositePipeline.getBindGroupLayout(0),
                entries: [
                    GPUBindGroupEntry(binding: 0, resource: .buffer(shadowUniformBuffer, offset: 0, size: UInt64(MemoryLayout<ShadowUniforms>.stride))),
                    GPUBindGroupEntry(binding: 1, resource: .textureView(shadowMaskTexture.createView())),
                    GPUBindGroupEntry(binding: 2, resource: .sampler(blurSampler))
                ]
            ))

            // Full-screen quad vertices for compositing the shadow texture
            var vertices: [CARendererVertex] = [
                CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: colorComponents),
                CARendererVertex(position: SIMD2(Float(size.width), 0), texCoord: SIMD2(1, 0), color: colorComponents),
                CARendererVertex(position: SIMD2(0, Float(size.height)), texCoord: SIMD2(0, 1), color: colorComponents),
                CARendererVertex(position: SIMD2(Float(size.width), 0), texCoord: SIMD2(1, 0), color: colorComponents),
                CARendererVertex(position: SIMD2(Float(size.width), Float(size.height)), texCoord: SIMD2(1, 1), color: colorComponents),
                CARendererVertex(position: SIMD2(0, Float(size.height)), texCoord: SIMD2(0, 1), color: colorComponents),
            ]

            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

            renderPass.setPipeline(shadowCompositePipeline)
            renderPass.setBindGroup(0, bindGroup: compositeBindGroup)
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)

            // Mark that we've consumed the pre-rendered shadow
            hasPrerenderredShadow = false
            return
        }

        // Fallback: Simplified shadow without blur (for subsequent shadows or when blur is unavailable)
        // Calculate shadow size (larger than layer for blur effect)
        let expandedWidth = layer.bounds.width + shadowRadius * 2
        let expandedHeight = layer.bounds.height + shadowRadius * 2

        // Create offset matrix for shadow (render behind the layer)
        let offsetMatrix = Matrix4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(Float(shadowOffset.width - shadowRadius), Float(shadowOffset.height - shadowRadius), 0.001, 1)
        ))

        let scaleMatrix = Matrix4x4(columns: (
            SIMD4<Float>(Float(expandedWidth), 0, 0, 0),
            SIMD4<Float>(0, Float(expandedHeight), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let finalMatrix = modelMatrix * offsetMatrix * scaleMatrix

        // Use larger corner radius to simulate blur
        let effectiveCornerRadius = Float(layer.cornerRadius + shadowRadius * 0.5)

        var uniforms = CARendererUniforms(
            mvpMatrix: finalMatrix,
            opacity: 1.0,
            cornerRadius: effectiveCornerRadius,
            layerSize: SIMD2<Float>(Float(expandedWidth), Float(expandedHeight))
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(
            uniformBuffer,
            bufferOffset: uniformOffset,
            data: uniformData
        )

        // Create shadow vertices
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: colorComponents),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: colorComponents),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: colorComponents),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: colorComponents),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: colorComponents),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: colorComponents),
        ]

        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(
            vertexBuffer,
            bufferOffset: vertexOffset,
            data: vertexData
        )

        renderPass.setPipeline(pipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    // MARK: - CATransformLayer Rendering

    /// Renders a CATransformLayer's sublayers without rendering the layer's own properties.
    ///
    /// CATransformLayer is different from regular layers in that:
    /// 1. It does not render its own content (backgroundColor, contents, etc.)
    /// 2. It does NOT flatten sublayers - they maintain their full 3D positions
    /// 3. Depth buffer z-testing is used for correct occlusion of overlapping layers
    ///
    /// Unlike regular layers where sublayers are composited as flat 2D images,
    /// CATransformLayer sublayers participate in true 3D depth testing.
    /// The depth buffer (z-buffer) handles occlusion correctly based on the
    /// transformed z-coordinates of each sublayer.
    private func renderTransformLayerSublayers(
        _ layer: CALayer,
        renderPass: GPURenderPassEncoder,
        parentMatrix: Matrix4x4
    ) {
        let presentationLayer = layer.presentation() ?? layer

        guard let sublayers = layer.sublayers else { return }

        // Apply the CATransformLayer's own transform (but not its content)
        var sublayerMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

        // Apply sublayerTransform if specified
        if !CATransform3DIsIdentity(presentationLayer.sublayerTransform) {
            sublayerMatrix = sublayerMatrix * presentationLayer.sublayerTransform.matrix4x4
        }

        // Render sublayers in array order.
        // The depth buffer handles z-ordering correctly - no pre-sorting needed.
        // Each sublayer's full 3D transform (including zPosition and CATransform3D)
        // is applied, and the GPU's depth test determines visibility.
        for sublayer in sublayers {
            self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
        }
    }

    // MARK: - CAEmitterLayer Rendering

    /// Last update time for particle simulation.
    private var lastParticleUpdateTime: CFTimeInterval = 0

    /// Random number generator state for particles.
    private var particleRandomSeed: UInt32 = 12345

    /// Tracks the last seed used to detect seed changes.
    private var lastEmitterSeed: UInt32 = 0

    /// Whether the random seed has been initialized.
    private var randomSeedInitialized: Bool = false

    /// Generates a random float in [0, 1).
    private func randomFloat() -> Float {
        particleRandomSeed = particleRandomSeed &* 1103515245 &+ 12345
        return Float(particleRandomSeed % 65536) / 65536.0
    }

    /// Generates a random float in [-1, 1).
    private func randomSignedFloat() -> Float {
        return randomFloat() * 2.0 - 1.0
    }

    /// Rotates a 2D point around the center (0.5, 0.5) by the given angle in radians.
    private func rotatePoint(_ point: SIMD2<Float>, angle: Float) -> SIMD2<Float> {
        let center = SIMD2<Float>(0.5, 0.5)
        let p = point - center
        let cosA = cos(angle)
        let sinA = sin(angle)
        let rotated = SIMD2<Float>(
            p.x * cosA - p.y * sinA,
            p.x * sinA + p.y * cosA
        )
        return rotated + center
    }

    /// Renders a CAEmitterLayer with its particle system.
    private func renderEmitterLayer(
        _ emitterLayer: CAEmitterLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let emitterCells = emitterLayer.emitterCells, !emitterCells.isEmpty,
              let pipeline = pipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        // Initialize or update random seed only when seed changes
        if !randomSeedInitialized || emitterLayer.seed != lastEmitterSeed {
            particleRandomSeed = emitterLayer.seed
            lastEmitterSeed = emitterLayer.seed
            randomSeedInitialized = true
        }

        // Calculate delta time
        let currentTime = CACurrentMediaTime()
        let deltaTime: Float
        if lastParticleUpdateTime > 0 {
            deltaTime = min(Float(currentTime - lastParticleUpdateTime), 0.1)
        } else {
            deltaTime = 1.0 / 60.0
        }
        lastParticleUpdateTime = currentTime

        // Update existing particles
        for i in 0..<activeParticles.count {
            activeParticles[i].update(deltaTime: deltaTime)
        }

        // Remove dead particles
        activeParticles.removeAll { !$0.isAlive }

        // Spawn new particles
        for cell in emitterCells where cell.isEnabled {
            let birthRate = cell.birthRate * emitterLayer.birthRate
            let particlesToSpawn = Int(birthRate * deltaTime + 0.5)

            for _ in 0..<particlesToSpawn {
                guard activeParticles.count < Self.maxParticles else { break }

                var particle = EmitterParticle()
                particle.position = calculateEmissionPosition(
                    shape: emitterLayer.emitterShape,
                    position: emitterLayer.emitterPosition,
                    zPosition: emitterLayer.emitterZPosition,
                    size: emitterLayer.emitterSize,
                    depth: emitterLayer.emitterDepth
                )

                // Calculate velocity
                let angle = Float(cell.emissionLongitude) + randomSignedFloat() * Float(cell.emissionRange)
                let velocity = Float(cell.velocity + randomSignedFloat() * CGFloat(cell.velocityRange)) * emitterLayer.velocity
                particle.velocity = SIMD3(velocity * cos(angle), velocity * sin(angle), 0)
                particle.acceleration = SIMD3(Float(cell.xAcceleration), Float(cell.yAcceleration), Float(cell.zAcceleration))

                // Set lifetime
                let lifetime = cell.lifetime + randomSignedFloat() * cell.lifetimeRange
                particle.lifetime = lifetime * emitterLayer.lifetime
                particle.maxLifetime = particle.lifetime

                // Set scale
                particle.scale = Float(cell.scale + randomSignedFloat() * CGFloat(cell.scaleRange)) * emitterLayer.scale
                particle.scaleSpeed = Float(cell.scaleSpeed)

                // Set rotation
                particle.rotationSpeed = Float(cell.spin + randomSignedFloat() * CGFloat(cell.spinRange)) * emitterLayer.spin

                // Set color
                if let cellColor = cell.color, let components = cellColor.components, components.count >= 3 {
                    particle.color = SIMD4(
                        Float(components[0]) + randomSignedFloat() * cell.redRange,
                        Float(components[1]) + randomSignedFloat() * cell.greenRange,
                        Float(components[2]) + randomSignedFloat() * cell.blueRange,
                        (components.count > 3 ? Float(components[3]) : 1.0) + randomSignedFloat() * cell.alphaRange
                    )
                } else {
                    particle.color = SIMD4(1, 1, 1, 1)
                }

                particle.colorSpeed = SIMD4(cell.redSpeed, cell.greenSpeed, cell.blueSpeed, cell.alphaSpeed)
                particle.isAlive = true
                activeParticles.append(particle)
            }
        }

        // Render particles
        for particle in activeParticles where particle.isAlive {
            guard currentLayerIndex < Self.maxLayers else { break }

            let layerIndex = currentLayerIndex
            currentLayerIndex += 1

            let scale = particle.scale * 20
            let scaleMatrix = Matrix4x4(columns: (
                SIMD4<Float>(scale, 0, 0, 0),
                SIMD4<Float>(0, scale, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))

            let translateMatrix = Matrix4x4(columns: (
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(particle.position.x, particle.position.y, particle.position.z, 1)
            ))

            let particleMatrix = modelMatrix * translateMatrix * scaleMatrix

            var uniforms = CARendererUniforms(
                mvpMatrix: particleMatrix,
                opacity: 1.0,
                cornerRadius: scale * 0.5,
                layerSize: SIMD2<Float>(scale, scale)
            )

            let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
            let uniformData = createFloat32Array(from: &uniforms)
            device.queue.writeBuffer(uniformBuffer, bufferOffset: uniformOffset, data: uniformData)

            // Apply rotation to vertices
            let rotation = particle.rotation
            let p0 = rotatePoint(SIMD2(0, 0), angle: rotation)
            let p1 = rotatePoint(SIMD2(1, 0), angle: rotation)
            let p2 = rotatePoint(SIMD2(0, 1), angle: rotation)
            let p3 = rotatePoint(SIMD2(1, 1), angle: rotation)

            var vertices: [CARendererVertex] = [
                CARendererVertex(position: p0, texCoord: SIMD2(0, 0), color: particle.color),
                CARendererVertex(position: p1, texCoord: SIMD2(1, 0), color: particle.color),
                CARendererVertex(position: p2, texCoord: SIMD2(0, 1), color: particle.color),
                CARendererVertex(position: p1, texCoord: SIMD2(1, 0), color: particle.color),
                CARendererVertex(position: p3, texCoord: SIMD2(1, 1), color: particle.color),
                CARendererVertex(position: p2, texCoord: SIMD2(0, 1), color: particle.color),
            ]

            let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
            let vertexData = createFloat32Array(from: &vertices)
            device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

            renderPass.setPipeline(pipeline)
            renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
            renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
            renderPass.draw(vertexCount: 6)
        }
    }

    /// Calculates an emission position based on the emitter shape.
    private func calculateEmissionPosition(
        shape: CAEmitterLayerEmitterShape,
        position: CGPoint,
        zPosition: CGFloat,
        size: CGSize,
        depth: CGFloat
    ) -> SIMD3<Float> {
        let x = Float(position.x)
        let y = Float(position.y)
        let z = Float(zPosition)
        let w = Float(size.width)
        let h = Float(size.height)

        switch shape {
        case .point:
            return SIMD3(x, y, z)
        case .line:
            return SIMD3(x - w/2 + w * randomFloat(), y, z)
        case .rectangle:
            return SIMD3(x - w/2 + w * randomFloat(), y - h/2 + h * randomFloat(), z)
        case .circle:
            let angle = randomFloat() * 2 * .pi
            let radius = sqrt(randomFloat()) * min(w, h) / 2
            return SIMD3(x + radius * cos(angle), y + radius * sin(angle), z)
        case .sphere:
            let theta = randomFloat() * 2 * .pi
            let phi = acos(2 * randomFloat() - 1)
            let r = cbrt(randomFloat()) * min(w, h, Float(depth)) / 2
            return SIMD3(x + r * sin(phi) * cos(theta), y + r * sin(phi) * sin(theta), z + r * cos(phi))
        default:
            return SIMD3(x, y, z)
        }
    }

    // MARK: - CATiledLayer Rendering

    /// Calculates the current LOD level based on the layer's transform.
    /// Returns the LOD level (0 = highest detail, higher = lower detail).
    private func calculateLODLevel(
        tiledLayer: CATiledLayer,
        modelMatrix: Matrix4x4
    ) -> Int {
        // Extract scale from the model matrix (approximate using column magnitudes)
        let scaleX = sqrt(modelMatrix.columns.0.x * modelMatrix.columns.0.x +
                         modelMatrix.columns.0.y * modelMatrix.columns.0.y)
        let scaleY = sqrt(modelMatrix.columns.1.x * modelMatrix.columns.1.x +
                         modelMatrix.columns.1.y * modelMatrix.columns.1.y)
        let scale = max(scaleX, scaleY)

        // Calculate LOD based on scale
        // scale > 1: zoomed in (use higher detail, lower LOD number)
        // scale < 1: zoomed out (use lower detail, higher LOD number)
        let maxLOD = tiledLayer.levelsOfDetail - 1
        let lodBias = tiledLayer.levelsOfDetailBias

        if scale >= 1.0 {
            // Zoomed in: use LOD 0 (highest detail)
            return max(0, -lodBias)
        } else {
            // Zoomed out: calculate LOD based on scale
            // Each LOD level represents a 2x reduction in detail
            let lodFromScale = Int(-log2(Double(scale)))
            return max(0, min(maxLOD, lodFromScale - lodBias))
        }
    }

    /// Renders a CATiledLayer with tile-based rendering and LOD support.
    private func renderTiledLayer(
        _ tiledLayer: CATiledLayer,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        modelMatrix: Matrix4x4,
        bindGroup: GPUBindGroup
    ) {
        guard let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let pipeline = pipeline else { return }

        // Calculate current LOD level
        let lodLevel = calculateLODLevel(tiledLayer: tiledLayer, modelMatrix: modelMatrix)

        // Adjust tile size based on LOD level
        // Higher LOD = larger tiles (covering more area with less detail)
        let lodScale = pow(2.0, CGFloat(lodLevel))
        let adjustedTileSize = CGSize(
            width: tiledLayer.tileSize.width * lodScale,
            height: tiledLayer.tileSize.height * lodScale
        )

        let bounds = tiledLayer.bounds
        let tilesX = Int(ceil(bounds.width / adjustedTileSize.width))
        let tilesY = Int(ceil(bounds.height / adjustedTileSize.height))

        // Render tiles
        for ty in 0..<tilesY {
            for tx in 0..<tilesX {
                guard currentLayerIndex < Self.maxLayers else { return }

                let tileKey = CATiledLayer.TileKey(column: tx, row: ty, lodLevel: lodLevel)
                let tileX = CGFloat(tx) * adjustedTileSize.width
                let tileY = CGFloat(ty) * adjustedTileSize.height
                let tileW = min(adjustedTileSize.width, bounds.width - tileX)
                let tileH = min(adjustedTileSize.height, bounds.height - tileY)

                let tileTranslate = Matrix4x4(columns: (
                    SIMD4<Float>(1, 0, 0, 0),
                    SIMD4<Float>(0, 1, 0, 0),
                    SIMD4<Float>(0, 0, 1, 0),
                    SIMD4<Float>(Float(tileX), Float(tileY), 0, 1)
                ))

                let tileScale = Matrix4x4(columns: (
                    SIMD4<Float>(Float(tileW), 0, 0, 0),
                    SIMD4<Float>(0, Float(tileH), 0, 0),
                    SIMD4<Float>(0, 0, 1, 0),
                    SIMD4<Float>(0, 0, 0, 1)
                ))

                let tileMatrix = modelMatrix * tileTranslate * tileScale

                // Check if tile has cached content
                if let cachedImage = tiledLayer.cachedImage(for: tileKey) {
                    // Render cached tile as texture
                    renderTileWithImage(
                        cachedImage,
                        device: device,
                        renderPass: renderPass,
                        tileMatrix: tileMatrix,
                        tileSize: CGSize(width: tileW, height: tileH),
                        opacity: tiledLayer.opacity
                    )
                } else {
                    // Render placeholder and request tile
                    renderTilePlaceholder(
                        device: device,
                        renderPass: renderPass,
                        bindGroup: bindGroup,
                        tileMatrix: tileMatrix,
                        tileSize: CGSize(width: tileW, height: tileH),
                        tileKey: tileKey,
                        opacity: tiledLayer.opacity,
                        lodLevel: lodLevel
                    )

                    // Request tile from delegate if not already loading
                    if !tiledLayer.loadingTiles.contains(tileKey) {
                        requestTileFromDelegate(
                            tiledLayer: tiledLayer,
                            tileKey: tileKey,
                            tileRect: CGRect(x: tileX, y: tileY, width: tileW, height: tileH),
                            lodScale: lodScale
                        )
                    }
                }
            }
        }

        // Render sublayers
        if let sublayers = tiledLayer.sublayers {
            for sublayer in sublayers {
                self.renderLayer(sublayer, renderPass: renderPass, parentMatrix: modelMatrix)
            }
        }
    }

    /// Renders a tile with a cached image texture.
    private func renderTileWithImage(
        _ image: CGImage,
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        tileMatrix: Matrix4x4,
        tileSize: CGSize,
        opacity: Float
    ) {
        guard let texturedPipeline = texturedPipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer,
              let textureSampler = textureSampler else { return }

        guard currentLayerIndex < Self.maxLayers else { return }
        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        // Get or create texture for image using the texture manager
        let imageId = ObjectIdentifier(image as AnyObject)
        let imageWidth = image.width
        let imageHeight = image.height
        guard let texture = textureManager?.getOrCreateTexture(
            for: imageId,
            width: imageWidth,
            height: imageHeight,
            factory: { [weak self] in
                self?.createTexture(from: image, device: device)
            }
        ) else { return }

        // Create uniforms
        var uniforms = CARendererUniforms(
            mvpMatrix: tileMatrix,
            opacity: opacity,
            cornerRadius: 0,
            layerSize: SIMD2<Float>(Float(tileSize.width), Float(tileSize.height))
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: uniformOffset, data: uniformData)

        // Create vertices with white color (texture provides color)
        let white = SIMD4<Float>(1, 1, 1, 1)
        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: white),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: white),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: white),
        ]

        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

        // Create bind group for textured rendering
        guard let texturedBindGroupLayout = texturedBindGroupLayout else { return }
        let texturedBindGroup = device.createBindGroup(descriptor: GPUBindGroupDescriptor(
            layout: texturedBindGroupLayout,
            entries: [
                GPUBindGroupEntry(
                    binding: 0,
                    resource: .bufferBinding(GPUBufferBinding(
                        buffer: uniformBuffer,
                        size: UInt64(MemoryLayout<CARendererUniforms>.stride)
                    ))
                ),
                GPUBindGroupEntry(binding: 1, resource: .sampler(textureSampler)),
                GPUBindGroupEntry(binding: 2, resource: .textureView(texture.createView()))
            ]
        ))

        renderPass.setPipeline(texturedPipeline)
        renderPass.setBindGroup(0, bindGroup: texturedBindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    /// Renders a placeholder for a tile that hasn't been loaded yet.
    private func renderTilePlaceholder(
        device: GPUDevice,
        renderPass: GPURenderPassEncoder,
        bindGroup: GPUBindGroup,
        tileMatrix: Matrix4x4,
        tileSize: CGSize,
        tileKey: CATiledLayer.TileKey,
        opacity: Float,
        lodLevel: Int
    ) {
        guard let pipeline = pipeline,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else { return }

        guard currentLayerIndex < Self.maxLayers else { return }
        let layerIndex = currentLayerIndex
        currentLayerIndex += 1

        var uniforms = CARendererUniforms(
            mvpMatrix: tileMatrix,
            opacity: opacity,
            cornerRadius: 0,
            layerSize: SIMD2<Float>(Float(tileSize.width), Float(tileSize.height))
        )

        let uniformOffset = UInt64(layerIndex) * Self.alignedUniformSize
        let uniformData = createFloat32Array(from: &uniforms)
        device.queue.writeBuffer(uniformBuffer, bufferOffset: uniformOffset, data: uniformData)

        // Generate placeholder color based on LOD level and position
        let isEven = (tileKey.column + tileKey.row) % 2 == 0
        let lodTint = 1.0 - Float(lodLevel) * 0.1
        let baseGray: Float = isEven ? 0.85 : 0.75
        let tileColor = SIMD4<Float>(baseGray * lodTint, baseGray * lodTint, baseGray, 1.0)

        var vertices: [CARendererVertex] = [
            CARendererVertex(position: SIMD2(0, 0), texCoord: SIMD2(0, 0), color: tileColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: tileColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: tileColor),
            CARendererVertex(position: SIMD2(1, 0), texCoord: SIMD2(1, 0), color: tileColor),
            CARendererVertex(position: SIMD2(1, 1), texCoord: SIMD2(1, 1), color: tileColor),
            CARendererVertex(position: SIMD2(0, 1), texCoord: SIMD2(0, 1), color: tileColor),
        ]

        let vertexOffset = UInt64(layerIndex * 6 * MemoryLayout<CARendererVertex>.stride)
        let vertexData = createFloat32Array(from: &vertices)
        device.queue.writeBuffer(vertexBuffer, bufferOffset: vertexOffset, data: vertexData)

        renderPass.setPipeline(pipeline)
        renderPass.setBindGroup(0, bindGroup: bindGroup, dynamicOffsets: [UInt32(uniformOffset)])
        renderPass.setVertexBuffer(0, buffer: vertexBuffer, offset: vertexOffset)
        renderPass.draw(vertexCount: 6)
    }

    /// Requests a tile from the layer's delegate.
    ///
    /// This method calls the delegate's draw(_:in:) method with a CGContext
    /// configured for the specific tile's bounds and scale.
    private func requestTileFromDelegate(
        tiledLayer: CATiledLayer,
        tileKey: CATiledLayer.TileKey,
        tileRect: CGRect,
        lodScale: CGFloat
    ) {
        guard tiledLayer.delegate != nil else { return }

        // Mark tile as loading
        tiledLayer.loadingTiles.insert(tileKey)

        // In a full implementation, this would:
        // 1. Create a CGContext for the tile
        // 2. Set up the context transformation for the tile's position
        // 3. Call delegate.draw(tiledLayer, in: context)
        // 4. Convert the context to a CGImage
        // 5. Cache the result with tiledLayer.cacheImage(image, for: tileKey)
        //
        // For now, we just mark it as loading. The actual implementation
        // would require CGContext rendering to textures which depends on
        // the OpenCoreGraphics context implementation.
    }
}

#endif
