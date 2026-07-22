import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

@Suite("Delegate backing-store format")
struct CADelegateBackingStoreFormatTests {
    private final class ProbeDelegate: CALayerDelegate {}

    @Test("Explicit layer formats retain their storage widths")
    func explicitFormats() throws {
        let rgba8 = try CADelegateBackingStoreFormat.resolve(
            contentsFormat: .RGBA8Uint,
            contentsHeadroom: 1
        )
        let rgba16 = try CADelegateBackingStoreFormat.resolve(
            contentsFormat: .RGBA16Float,
            contentsHeadroom: 1
        )
        let gray8 = try CADelegateBackingStoreFormat.resolve(
            contentsFormat: .gray8Uint,
            contentsHeadroom: 1
        )

        #expect(rgba8 == .rgba8Uint)
        #expect(rgba8.bitsPerComponent == 8)
        #expect(rgba8.bitsPerPixel == 32)
        #expect(rgba16 == .rgba16Float)
        #expect(rgba16.bitsPerComponent == 16)
        #expect(rgba16.bitsPerPixel == 64)
        #expect(gray8 == .gray8Uint)
        #expect(gray8.bitsPerComponent == 8)
        #expect(gray8.bitsPerPixel == 8)
    }

    @Test("Automatic storage preserves extended headroom")
    func automaticStorage() throws {
        #expect(try CADelegateBackingStoreFormat.resolve(
            contentsFormat: .automatic,
            contentsHeadroom: 1
        ) == .rgba8Uint)
        #expect(try CADelegateBackingStoreFormat.resolve(
            contentsFormat: .automatic,
            contentsHeadroom: 4
        ) == .rgba16Float)
    }

    @Test("Unknown storage formats fail explicitly")
    func unknownStorageFormat() {
        #expect(throws: CADelegateBackingStoreError.unsupportedContentsFormat("FutureFormat")) {
            try CADelegateBackingStoreFormat.resolve(
                contentsFormat: CALayerContentsFormat(rawValue: "FutureFormat"),
                contentsHeadroom: 1
            )
        }
    }

    @Test("Storage-affecting changes invalidate delegate drawing")
    func storageChangesInvalidateDelegateDrawing() {
        let delegate = ProbeDelegate()
        let layer = CALayer()
        layer.delegate = delegate

        layer.contentsFormat = .RGBA16Float
        #expect(layer.needsDisplay())
        layer.displayIfNeeded()

        layer.contentsHeadroom = 4
        #expect(layer.needsDisplay())
        layer.displayIfNeeded()

        layer.contentsScale = 2
        #expect(layer.needsDisplay())
    }
}
