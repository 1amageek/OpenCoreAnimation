
/// A type alias for time intervals, matching CoreFoundation's CFTimeInterval.
public typealias CFTimeInterval = Double

/// Methods that model a hierarchical timing system, allowing objects to map time between their parent and local time.
///
/// Absolute time is defined as mach time converted to seconds. The `CACurrentMediaTime()` function
/// is provided as a convenience for getting the current absolute time.
///
/// The conversion from parent time to local time has two stages:
/// 1. Conversion to "active local time." This includes the point at which the object appears
///    in the parent object's timeline and how fast it plays relative to the parent.
/// 2. Conversion from "active local time" to "basic local time." The timing model allows for
///    objects to repeat their basic duration multiple times and, optionally, to play backwards before repeating.
public protocol CAMediaTiming {
    /// Specifies the begin time of the receiver in relation to its parent object, if applicable.
    var beginTime: CFTimeInterval { get set }

    /// Specifies an additional time offset in active local time.
    var timeOffset: CFTimeInterval { get set }

    /// Determines the number of times the animation will repeat.
    var repeatCount: Float { get set }

    /// Determines how many seconds the animation will repeat for.
    var repeatDuration: CFTimeInterval { get set }

    /// Specifies the basic duration of the animation, in seconds.
    var duration: CFTimeInterval { get set }

    /// Specifies how time is mapped to receiver's time space from the parent time space.
    var speed: Float { get set }

    /// Determines if the receiver plays in the reverse upon completion.
    var autoreverses: Bool { get set }

    /// Determines if the receiver's presentation is frozen or removed once its active duration has completed.
    var fillMode: CAMediaTimingFillMode { get set }
}

// MARK: - CACurrentMediaTime

#if arch(wasm32)
import JavaScriptKit

/// Returns the current absolute time, in seconds.
///
/// This uses JavaScript's `performance.now()` API which returns a high-resolution
/// timestamp in milliseconds, which is then converted to seconds.
public func CACurrentMediaTime() -> CFTimeInterval {
    let performance = JSObject.global.performance
    let milliseconds = performance.now().number ?? 0
    return milliseconds / 1000.0
}

#else
import Foundation

/// Returns the current absolute time, in seconds.
///
/// This uses `ProcessInfo.processInfo.systemUptime` for native platforms (testing).
public func CACurrentMediaTime() -> CFTimeInterval {
    return ProcessInfo.processInfo.systemUptime
}

#endif
