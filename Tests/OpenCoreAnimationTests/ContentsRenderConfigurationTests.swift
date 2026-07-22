import Testing
@_spi(RendererDiagnostics)
@testable import OpenCoreAnimation

@Suite("Contents render configuration")
struct ContentsRenderConfigurationTests {
    @Test("nine-slice source coordinates apply contentsRect first")
    func contentsRectPrecedesContentsCenter() throws {
        let configuration = try ContentsRenderConfiguration(
            imageSize: CGSize(width: 8, height: 4),
            boundsSize: CGSize(width: 8, height: 8),
            contentsRect: CGRect(x: 0.5, y: 0, width: 0.5, height: 1),
            contentsCenter: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5),
            contentsScale: 1,
            gravity: .resize
        )

        #expect(configuration.patches.count == 9)
        #expect(configuration.patches[4].sourceUnitRect == CGRect(
            x: 0.625,
            y: 0.25,
            width: 0.25,
            height: 0.5
        ))
        #expect(configuration.patches[4].destinationRect == CGRect(
            x: 1,
            y: 1,
            width: 6,
            height: 6
        ))
    }

    @Test("contentsScale converts fixed source pixels to destination points")
    func contentsScaleControlsFixedEdges() throws {
        let configuration = try ContentsRenderConfiguration(
            imageSize: CGSize(width: 8, height: 8),
            boundsSize: CGSize(width: 8, height: 8),
            contentsRect: CGRect(x: 0, y: 0, width: 1, height: 1),
            contentsCenter: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5),
            contentsScale: 2,
            gravity: .resize
        )

        #expect(configuration.patches[0].destinationRect == CGRect(
            x: 0,
            y: 7,
            width: 1,
            height: 1
        ))
        #expect(configuration.patches[4].destinationRect == CGRect(
            x: 1,
            y: 1,
            width: 6,
            height: 6
        ))
    }

    @Test("single-quad gravity uses the selected source size and contentsScale")
    func singleQuadUsesSelectedLogicalSize() throws {
        let destination = try ContentsRenderConfiguration.destinationRect(
            imageSize: CGSize(width: 12, height: 8),
            boundsSize: CGSize(width: 10, height: 10),
            contentsRect: CGRect(x: 0.5, y: 0, width: 0.5, height: 1),
            contentsScale: 2,
            gravity: .center
        )

        #expect(destination == CGRect(x: 3.5, y: 3, width: 3, height: 4))
    }

    @Test("nine-slice is restricted to resizing gravity modes")
    func gravityControlsNineSliceSelection() {
        let center = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)

        #expect(ContentsRenderConfiguration.usesNineSlice(
            gravity: .resize,
            contentsCenter: center
        ))
        #expect(ContentsRenderConfiguration.usesNineSlice(
            gravity: .resizeAspect,
            contentsCenter: center
        ))
        #expect(ContentsRenderConfiguration.usesNineSlice(
            gravity: .resizeAspectFill,
            contentsCenter: center
        ))
        #expect(!ContentsRenderConfiguration.usesNineSlice(
            gravity: .center,
            contentsCenter: center
        ))
    }

    @Test("invalid geometry is rejected with typed errors")
    func invalidGeometryFails() {
        #expect(throws: ContentsRenderConfigurationError.invalidContentsCenter) {
            _ = try ContentsRenderConfiguration(
                imageSize: CGSize(width: 8, height: 8),
                boundsSize: CGSize(width: 8, height: 8),
                contentsRect: CGRect(x: 0, y: 0, width: 1, height: 1),
                contentsCenter: CGRect(x: -0.25, y: 0, width: 0.5, height: 1),
                contentsScale: 1,
                gravity: .resize
            )
        }
        #expect(throws: ContentsRenderConfigurationError.invalidContentsScale) {
            _ = try ContentsRenderConfiguration(
                imageSize: CGSize(width: 8, height: 8),
                boundsSize: CGSize(width: 8, height: 8),
                contentsRect: CGRect(x: 0, y: 0, width: 1, height: 1),
                contentsCenter: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5),
                contentsScale: 0,
                gravity: .resize
            )
        }
        #expect(throws: ContentsRenderConfigurationError.unsupportedGravity("unknown")) {
            _ = try ContentsRenderConfiguration.destinationRect(
                imageSize: CGSize(width: 8, height: 8),
                boundsSize: CGSize(width: 8, height: 8),
                contentsRect: CGRect(x: 0, y: 0, width: 1, height: 1),
                contentsScale: 1,
                gravity: CALayerContentsGravity(rawValue: "unknown")
            )
        }
    }
}
