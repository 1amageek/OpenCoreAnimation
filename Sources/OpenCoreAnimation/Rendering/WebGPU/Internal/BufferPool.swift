#if arch(wasm32)
import Foundation
import SwiftWebGPU

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

#endif
