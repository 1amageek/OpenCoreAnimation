#if canImport(CoreVideo)
import CoreVideo

extension CARenderer {
    /// Begins evaluation of a frame at the supplied layer media time.
    public func beginFrame(
        atTime time: CFTimeInterval,
        timeStamp: UnsafeMutablePointer<CVTimeStamp>?
    ) {
        beginFrame(atTime: time)
    }
}
#endif
