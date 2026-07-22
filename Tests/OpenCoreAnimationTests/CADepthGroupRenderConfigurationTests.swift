import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Depth group render configuration")
struct CADepthGroupRenderConfigurationTests {
    @Test("Root and nested groups produce exact state transitions")
    func validStateTransitions() throws {
        let root = try CADepthGroupRenderConfiguration(currentNestingDepth: 0)
        #expect(root.requiresDepthClear)
        #expect(root.enteredNestingDepth == 1)

        let nested = try CADepthGroupRenderConfiguration(currentNestingDepth: 3)
        #expect(!nested.requiresDepthClear)
        #expect(nested.enteredNestingDepth == 4)
    }

    @Test("Invalid nesting states retain exact internal reasons")
    func invalidStateTransitions() {
        #expect(throws: CADepthGroupStateFailure.invalidNestingDepth(-1)) {
            try CADepthGroupRenderConfiguration(currentNestingDepth: -1)
        }
        #expect(throws: CADepthGroupStateFailure.nestingDepthOverflow) {
            try CADepthGroupRenderConfiguration(currentNestingDepth: Int.max)
        }
    }

    @Test("Every renderer maps invalid nesting to its public diagnostic")
    func invalidNestingMappings() {
        let reason = CADepthGroupStateFailure.invalidNestingDepth(-1)
        #expect(CATransformDepthRenderFailure.depthGroupStateFailure(reason)
            == .invalidNestingDepth(-1))
        #expect(CAReplicatorRenderFailure.depthGroupStateFailure(reason)
            == .invalidDepthNesting(-1))
        #expect(CAEmitterFailure.depthGroupStateFailure(reason)
            == .invalidDepthNesting(-1))
    }

    @Test("Every renderer maps overflow to its public diagnostic")
    func overflowMappings() {
        let reason = CADepthGroupStateFailure.nestingDepthOverflow
        #expect(CATransformDepthRenderFailure.depthGroupStateFailure(reason)
            == .nestingDepthOverflow)
        #expect(CAReplicatorRenderFailure.depthGroupStateFailure(reason)
            == .depthNestingOverflow)
        #expect(CAEmitterFailure.depthGroupStateFailure(reason)
            == .depthNestingOverflow)
    }
}
