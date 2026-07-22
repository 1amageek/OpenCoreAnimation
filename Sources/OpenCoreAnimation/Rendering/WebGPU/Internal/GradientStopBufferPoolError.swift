#if arch(wasm32)
import Foundation

enum GradientStopBufferPoolError: Error, Equatable {
    case capacityExceeded(required: UInt64, maximum: UInt64)
}
#endif
