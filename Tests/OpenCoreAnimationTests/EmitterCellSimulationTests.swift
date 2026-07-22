import Testing
@testable import OpenCoreAnimation

@Suite("Emitter Cell Simulation Tests")
struct EmitterCellSimulationTests {
    @Test("Default infinite duration emits continuously after begin time")
    func unboundedDefaultTiming() throws {
        let cell = CAEmitterCell()
        cell.beginTime = 1

        #expect(cell.duration == .infinity)
        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 0, to: 0.5) == 0)
        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 0.5, to: 1.5) == 0.5)
    }

    @Test("Positive infinite repeat values keep emission active")
    func infiniteRepeatTiming() throws {
        let repeatCountCell = CAEmitterCell()
        repeatCountCell.duration = 1
        repeatCountCell.repeatCount = .infinity

        let repeatDurationCell = CAEmitterCell()
        repeatDurationCell.duration = 1
        repeatDurationCell.repeatDuration = .infinity

        #expect(
            try EmitterCellSimulation.activeEmissionDelta(
                for: repeatCountCell,
                from: 100,
                to: 101
            ) == 1
        )
        #expect(
            try EmitterCellSimulation.activeEmissionDelta(
                for: repeatDurationCell,
                from: 100,
                to: 101
            ) == 1
        )
    }

    @Test("Invalid non-finite timing is reported as an error")
    func invalidNonFiniteTiming() {
        let notANumberCell = CAEmitterCell()
        notANumberCell.duration = .nan

        let negativeInfinityCell = CAEmitterCell()
        negativeInfinityCell.repeatDuration = -.infinity

        #expect(throws: EmitterCellSimulationError.nonFiniteTiming) {
            try EmitterCellSimulation.activeEmissionDelta(
                for: notANumberCell,
                from: 0,
                to: 1
            )
        }
        #expect(throws: EmitterCellSimulationError.nonFiniteTiming) {
            try EmitterCellSimulation.activeEmissionDelta(
                for: negativeInfinityCell,
                from: 0,
                to: 1
            )
        }
    }

    @Test("Speed and finite repeat duration clip active emission time")
    func finiteTimingWindow() throws {
        let cell = CAEmitterCell()
        cell.duration = 1
        cell.repeatCount = 2
        cell.speed = 2

        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 0, to: 0.25) == 0.5)
        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 0.75, to: 1.25) == 0.5)
        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 1.25, to: 2) == 0)
    }

    @Test("Reverse timing consumes a positive active interval")
    func reverseTimingWindow() throws {
        let cell = CAEmitterCell()
        cell.duration = 2
        cell.speed = -1
        cell.timeOffset = 2

        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 0, to: 0.5) == 0.5)
        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 1.5, to: 2.5) == 0.5)
    }

    @Test("Child direction is relative to parent direction")
    func relativeChildDirection() throws {
        let localForward = SIMD3<Float>(0, 0, 1)
        let localRight = SIMD3<Float>(1, 0, 0)
        let parentDirection = SIMD3<Float>(0, 1, 0)

        #expect(
            try EmitterCellSimulation.childDirection(
                localDirection: localForward,
                parentDirection: parentDirection
            ) == parentDirection
        )
        #expect(
            try EmitterCellSimulation.childDirection(
                localDirection: localRight,
                parentDirection: SIMD3<Float>(0, 0, 1)
            ) == SIMD3<Float>(1, 0, 0)
        )
    }
}
