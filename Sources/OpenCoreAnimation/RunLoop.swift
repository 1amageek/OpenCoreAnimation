//
//  RunLoop.swift
//  OpenCoreAnimation
//
//  Stub RunLoop implementation for API compatibility with Apple platforms.
//

#if canImport(Foundation) && !arch(wasm32)
import Foundation
// On native platforms with Foundation, re-export Foundation's RunLoop
public typealias RunLoop = Foundation.RunLoop
#else

/// A stub RunLoop for API compatibility with Apple platforms.
///
/// On WASM, JavaScript's event loop handles all scheduling. The `requestAnimationFrame` API
/// is used for display synchronization, which has no concept of "run loop modes."
/// This type exists solely for API compatibility so that code written for Apple platforms
/// compiles without modification.
///
/// ## Usage
///
/// ```swift
/// let displayLink = CADisplayLink(target: self, selector: Selector(""))
/// displayLink.add(to: .main, forMode: .common)
/// ```
///
/// - Note: On WASM, the `RunLoop` and `RunLoop.Mode` parameters are ignored.
///   All display links use `requestAnimationFrame` regardless of the mode specified.
public final class RunLoop: @unchecked Sendable {

    /// The run loop for the main thread.
    public static let main = RunLoop()

    /// The current run loop.
    public static var current: RunLoop { main }

    private init() {}

    /// Modes that a run loop operates in.
    ///
    /// On WASM, modes are ignored. The `requestAnimationFrame` API always fires
    /// when the browser is ready to paint, which is equivalent to "common" mode behavior.
    public struct Mode: RawRepresentable, Hashable, Sendable {
        public var rawValue: String

        public init(rawValue: String) {
            self.rawValue = rawValue
        }

        /// A mode that includes input sources and timers from all common modes.
        ///
        /// On WASM, `requestAnimationFrame` inherently behaves like common mode,
        /// firing during all browser states including scrolling.
        public static let common = Mode(rawValue: "kCFRunLoopCommonModes")

        /// The default run loop mode.
        ///
        /// On WASM, this is treated the same as `.common`.
        public static let `default` = Mode(rawValue: "kCFRunLoopDefaultMode")

        /// The mode used for tracking events.
        ///
        /// On WASM, this is treated the same as `.common`.
        public static let tracking = Mode(rawValue: "UITrackingRunLoopMode")
    }
}

#endif
