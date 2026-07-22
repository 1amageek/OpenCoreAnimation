import Testing
@testable import OpenCoreAnimation

@Suite("Projected depth")
struct CAProjectedDepthTests {
    @Test("Finite homogeneous coordinates produce normalized depth")
    func validDepth() throws {
        #expect(try CAProjectedDepth.resolve(z: 3, w: 2) == 1.5)
        #expect(try CAProjectedDepth.resolve(z: -1, w: 4) == -0.25)
    }

    @Test("Non-finite and zero homogeneous coordinates retain exact reasons")
    func invalidCoordinates() {
        #expect(throws: CAProjectedDepthError.nonFiniteHomogeneousCoordinate(
            z: .infinity,
            w: 1
        )) {
            try CAProjectedDepth.resolve(z: .infinity, w: 1)
        }
        #expect(throws: CAProjectedDepthError.zeroHomogeneousCoordinate) {
            try CAProjectedDepth.resolve(z: 1, w: 0)
        }
    }

    @Test("Finite division overflow is rejected")
    func normalizedOverflow() {
        #expect(throws: CAProjectedDepthError.nonFiniteNormalizedDepth) {
            try CAProjectedDepth.resolve(
                z: Float.greatestFiniteMagnitude,
                w: Float.leastNonzeroMagnitude
            )
        }
    }
}
