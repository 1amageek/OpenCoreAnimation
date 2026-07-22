
/// A protocol for receiving display link callbacks.
///
/// On WASM, implement this protocol on your target object to receive display link updates.
/// The selector parameter in `init(target:selector:)` is ignored; this protocol method is called instead.
@MainActor public protocol CADisplayLinkDelegate: AnyObject {
    /// Called when the display link fires.
    ///
    /// - Parameter displayLink: The display link that fired.
    func displayLinkDidFire(_ displayLink: CADisplayLink)
}

/// A structure that represents a range of frame rates.
public struct CAFrameRateRange: Equatable, Sendable {
    /// The minimum frame rate.
    public var minimum: Float

    /// The maximum frame rate.
    public var maximum: Float

    /// The preferred frame rate.
    public var preferred: Float?

    /// Creates a default frame rate range.
    public init() {
        self.minimum = 0
        self.maximum = 0
        self.preferred = nil
    }

    /// Creates a frame rate range with the specified values.
    public init(minimum: Float, maximum: Float, preferred: Float?) {
        self.minimum = minimum
        self.maximum = maximum
        self.preferred = preferred
    }

    /// The default frame rate range.
    public static let `default` = CAFrameRateRange()

    internal var effectiveFrameRate: Float? {
        func validRate(_ value: Float?) -> Float? {
            guard let value, value.isFinite, value > 0 else { return nil }
            return value
        }

        if var resolved = validRate(preferred) {
            if let minimum = validRate(minimum) {
                resolved = max(resolved, minimum)
            }
            if let maximum = validRate(maximum) {
                resolved = min(resolved, maximum)
            }
            return resolved
        }
        return validRate(maximum) ?? validRate(minimum)
    }
}
