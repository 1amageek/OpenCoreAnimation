import Testing
@testable import OpenCoreAnimation

@Suite("Emitter Cell Simulation Tests")
struct EmitterCellSimulationTests {
    @Test("Default zero duration emits continuously after begin time")
    func unboundedDefaultTiming() throws {
        let cell = CAEmitterCell()
        cell.beginTime = 1

        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 0, to: 0.5) == 0)
        #expect(try EmitterCellSimulation.activeEmissionDelta(for: cell, from: 0.5, to: 1.5) == 0.5)
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
