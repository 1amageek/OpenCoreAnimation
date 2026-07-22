import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Transform depth render configuration")
struct CATransformDepthRenderConfigurationTests {
    @Test("Root and nested groups produce exact state transitions")
    func validStateTransitions() throws {
        let root = try CATransformDepthRenderConfiguration(currentNestingDepth: 0)
        #expect(root.requiresDepthClear)
        #expect(root.enteredNestingDepth == 1)

        let nested = try CATransformDepthRenderConfiguration(currentNestingDepth: 3)
        #expect(!nested.requiresDepthClear)
        #expect(nested.enteredNestingDepth == 4)
    }

    @Test("Invalid nesting states retain exact typed reasons")
    func invalidStateTransitions() {
        #expect(throws: CATransformDepthRenderFailure.invalidNestingDepth(-1)) {
            try CATransformDepthRenderConfiguration(currentNestingDepth: -1)
        }
        #expect(throws: CATransformDepthRenderFailure.nestingDepthOverflow) {
            try CATransformDepthRenderConfiguration(currentNestingDepth: Int.max)
        }
    }
}
