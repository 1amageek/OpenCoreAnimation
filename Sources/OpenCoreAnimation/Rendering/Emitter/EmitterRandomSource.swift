import Foundation

/// Deterministic random source used by one emitter-layer simulation.
struct EmitterRandomSource {
    private(set) var state: UInt32

    init(seed: UInt32) {
        state = seed
    }

    mutating func reset(seed: UInt32) {
        state = seed
    }

    mutating func unitFloat() -> Float {
        state = state &* 1_103_515_245 &+ 12_345
        return Float(state % 65_536) / 65_536
    }

    mutating func signedFloat() -> Float {
        unitFloat() * 2 - 1
    }
}
