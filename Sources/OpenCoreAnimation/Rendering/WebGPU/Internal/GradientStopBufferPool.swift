#if arch(wasm32)
import Foundation
import SwiftWebGPU

final class GradientStopBufferPool {
    private struct Slot {
        var buffer: GPUBuffer
        var capacity: UInt64
        var supersededBuffers: [GPUBuffer] = []
    }

    private let device: GPUDevice
    private let maximumCapacity: UInt64
    private var slots: [Slot]
    private var currentIndex = 0

    init(
        device: GPUDevice,
        initialCapacity: UInt64,
        maximumCapacity: UInt64,
        bufferCount: Int = 3
    ) {
        precondition(bufferCount > 0)
        precondition(initialCapacity > 0)
        precondition(initialCapacity <= maximumCapacity)

        self.device = device
        self.maximumCapacity = maximumCapacity
        self.slots = (0..<bufferCount).map { _ in
            Slot(
                buffer: device.createBuffer(descriptor: GPUBufferDescriptor(
                    size: initialCapacity,
                    usage: [.storage, .copyDst]
                )),
                capacity: initialCapacity
            )
        }
    }

    var currentBuffer: GPUBuffer {
        slots[currentIndex].buffer
    }

    var currentCapacity: UInt64 {
        slots[currentIndex].capacity
    }

    @discardableResult
    func ensureCapacity(_ requiredCapacity: UInt64) throws -> Bool {
        guard requiredCapacity <= maximumCapacity else {
            throw GradientStopBufferPoolError.capacityExceeded(
                required: requiredCapacity,
                maximum: maximumCapacity
            )
        }
        guard requiredCapacity > currentCapacity else { return false }

        var newCapacity = currentCapacity > maximumCapacity / 2
            ? maximumCapacity
            : max(currentCapacity * 2, 32)
        while newCapacity < requiredCapacity {
            if newCapacity > maximumCapacity / 2 {
                newCapacity = maximumCapacity
            } else {
                newCapacity *= 2
            }
        }

        let replacement = device.createBuffer(descriptor: GPUBufferDescriptor(
            size: newCapacity,
            usage: [.storage, .copyDst]
        ))
        slots[currentIndex].supersededBuffers.append(slots[currentIndex].buffer)
        slots[currentIndex].buffer = replacement
        slots[currentIndex].capacity = newCapacity
        return true
    }

    func advanceFrame() {
        currentIndex = (currentIndex + 1) % slots.count
        for buffer in slots[currentIndex].supersededBuffers {
            buffer.destroy()
        }
        slots[currentIndex].supersededBuffers.removeAll(keepingCapacity: true)
    }

    func invalidate() {
        for slot in slots {
            slot.buffer.destroy()
            for buffer in slot.supersededBuffers {
                buffer.destroy()
            }
        }
        slots.removeAll(keepingCapacity: false)
        currentIndex = 0
    }
}
#endif
