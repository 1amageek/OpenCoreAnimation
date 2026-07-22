import Foundation
import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

@Suite("CAReplicatorLayer render configuration")
struct CAReplicatorRenderConfigurationTests {
    @Test("Valid input preserves normalized renderer state")
    func validConfiguration() throws {
        let layer = CAReplicatorLayer()
        layer.instanceCount = 3
        layer.preservesDepth = true
        layer.instanceDelay = 0.25
        layer.instanceTransform = CATransform3DMakeTranslation(10, 20, 30)
        layer.instanceColor = CGColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1)
        layer.instanceRedOffset = 0.1
        layer.instanceAlphaOffset = -0.2

        let configuration = try CAReplicatorRenderConfiguration(
            layer: layer,
            maximumInstanceCount: 16
        )

        #expect(configuration.instanceCount == 3)
        #expect(configuration.preservesDepth)
        #expect(configuration.instanceDelay == 0.25)
        #expect(configuration.instanceTransform == layer.instanceTransform)
        #expect(configuration.baseColor == SIMD4(0.25, 0.5, 0.75, 1))
        #expect(try configuration.color(at: 2) == SIMD4(0.45, 0.5, 0.75, 0.6))
        #expect(try configuration.timeOffset(at: 2) == 0.5)
    }

    @Test("Negative counts normalize to an empty replication")
    func negativeInstanceCount() throws {
        let layer = CAReplicatorLayer()
        layer.instanceCount = -4

        let configuration = try CAReplicatorRenderConfiguration(
            layer: layer,
            maximumInstanceCount: 16
        )

        #expect(configuration.instanceCount == 0)
    }

    @Test("Counts beyond renderer capacity fail before traversal")
    func excessiveInstanceCount() {
        let layer = CAReplicatorLayer()
        layer.instanceCount = 17

        #expect(throws: CAReplicatorRenderFailure.instanceCountExceedsRendererCapacity(
            actual: 17,
            maximum: 16
        )) {
            try CAReplicatorRenderConfiguration(layer: layer, maximumInstanceCount: 16)
        }
    }

    @Test("Non-finite public values fail before GPU work")
    func nonFiniteInput() {
        let layer = CAReplicatorLayer()
        layer.instanceDelay = .infinity
        #expect(throws: CAReplicatorRenderFailure.nonFiniteInstanceDelay) {
            try CAReplicatorRenderConfiguration(layer: layer, maximumInstanceCount: 16)
        }

        layer.instanceDelay = 0
        layer.instanceTransform.m22 = .nan
        #expect(throws: CAReplicatorRenderFailure.nonFiniteInstanceTransform) {
            try CAReplicatorRenderConfiguration(layer: layer, maximumInstanceCount: 16)
        }

        layer.instanceTransform = CATransform3DIdentity
        layer.instanceBlueOffset = .infinity
        #expect(throws: CAReplicatorRenderFailure.nonFiniteInstanceColorOffset) {
            try CAReplicatorRenderConfiguration(layer: layer, maximumInstanceCount: 16)
        }

        layer.instanceBlueOffset = 0
        layer.instanceColor = CGColor(red: .nan, green: 0, blue: 0, alpha: 1)
        #expect(throws: CAReplicatorRenderFailure.invalidInstanceColor) {
            try CAReplicatorRenderConfiguration(layer: layer, maximumInstanceCount: 16)
        }
    }

    @Test("Cumulative values report the exact overflowing instance")
    func cumulativeOverflow() throws {
        let colorLayer = CAReplicatorLayer()
        colorLayer.instanceCount = 3
        colorLayer.instanceRedOffset = .greatestFiniteMagnitude
        let colorConfiguration = try CAReplicatorRenderConfiguration(
            layer: colorLayer,
            maximumInstanceCount: 16
        )
        #expect(throws: CAReplicatorRenderFailure.instanceColorOverflow(instanceIndex: 2)) {
            try colorConfiguration.color(at: 2)
        }

        let timeLayer = CAReplicatorLayer()
        timeLayer.instanceCount = 3
        timeLayer.instanceDelay = .greatestFiniteMagnitude
        let timeConfiguration = try CAReplicatorRenderConfiguration(
            layer: timeLayer,
            maximumInstanceCount: 16
        )
        #expect(throws: CAReplicatorRenderFailure.instanceTimeOffsetOverflow(instanceIndex: 2)) {
            try timeConfiguration.timeOffset(at: 2)
        }

        let transformLayer = CAReplicatorLayer()
        transformLayer.instanceCount = 3
        transformLayer.instanceTransform = CATransform3DMakeScale(
            .greatestFiniteMagnitude,
            1,
            1
        )
        let transformConfiguration = try CAReplicatorRenderConfiguration(
            layer: transformLayer,
            maximumInstanceCount: 16
        )
        let secondTransform = try transformConfiguration.nextTransform(
            after: CATransform3DIdentity,
            nextInstanceIndex: 1
        )
        #expect(throws: CAReplicatorRenderFailure.cumulativeTransformOverflow(instanceIndex: 2)) {
            try transformConfiguration.nextTransform(
                after: secondTransform,
                nextInstanceIndex: 2
            )
        }
    }
}
