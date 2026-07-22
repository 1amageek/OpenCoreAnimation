import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CGImage texture storage conversion")
struct CGImageTextureStorageConverterTests {
    @Test("Premultiplied RGBA8 is normalized to straight alpha")
    func premultipliedRGBA8IsUnpremultiplied() throws {
        let image = try makeImage(
            width: 2,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 8,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            data: Data([64, 32, 16, 128, 99, 88, 77, 0])
        )

        let storage = try CGImageTextureStorageConverter.convert(image)
        #expect(storage.format == .rgba8Unorm)
        #expect(storage.data == Data([
            128, 64, 32, 128,
            0, 0, 0, 0,
        ]))
    }

    @Test("Straight RGBA8 with row padding is tightly repacked")
    func straightRGBA8WithPaddingIsRepacked() throws {
        let image = try makeImage(
            width: 1,
            height: 2,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 8,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            data: Data([
                10, 20, 30, 255, 1, 2, 3, 4,
                40, 50, 60, 255,
            ])
        )

        let storage = try CGImageTextureStorageConverter.convert(image)
        #expect(storage.data == Data([
            10, 20, 30, 255,
            40, 50, 60, 255,
        ]))
    }

    @Test("ARGB, skipped-alpha, and grayscale layouts use color-buffer conversion")
    func convertsARGBSkippedAlphaAndGrayscale() throws {
        let argb = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 4,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
            data: Data([128, 64, 32, 16])
        )
        let gray = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: 1,
            colorSpace: .deviceGray,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            data: Data([77])
        )
        let skippedAlpha = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 4,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            data: Data([10, 20, 30, 5])
        )

        #expect(try CGImageTextureStorageConverter.convert(argb).data == Data([128, 64, 32, 128]))
        #expect(try CGImageTextureStorageConverter.convert(skippedAlpha).data == Data([10, 20, 30, 255]))
        #expect(try CGImageTextureStorageConverter.convert(gray).data == Data([77, 77, 77, 255]))
    }

    @Test("Premultiplied extended-linear half-float remains extended and becomes straight alpha")
    func premultipliedRGBA16FloatPreservesHeadroom() throws {
        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.extendedLinearSRGB))
        let image = try makeImage(
            headroom: 4,
            width: 1,
            height: 1,
            bitsPerComponent: 16,
            bitsPerPixel: 64,
            bytesPerRow: 8,
            colorSpace: colorSpace,
            bitmapInfo: CGBitmapInfo(
                alpha: .premultipliedLast,
                component: .float,
                byteOrder: .order16Little
            ),
            data: halfFloatData([1, 0.25, 0.125, 0.5])
        )

        let storage = try CGImageTextureStorageConverter.convert(image)

        #expect(storage.format == .rgba16Float)
        #expect(halfFloatComponents(in: storage.data) == [2, 0.5, 0.25, 0.5])
    }

    @Test("Device RGB Float32 extended values convert through the public color conversion path")
    func deviceRGBFloat32PreservesExtendedValues() throws {
        let image = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 32,
            bitsPerPixel: 128,
            bytesPerRow: 16,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(
                alpha: .last,
                component: .float,
                byteOrder: .order32Little
            ),
            data: float32Data([2, 0.5, 0.25, 1])
        )

        let storage = try CGImageTextureStorageConverter.convert(image)

        #expect(storage.format == .rgba16Float)
        #expect(halfFloatComponents(in: storage.data) == [2, 0.5, 0.25, 1])
    }

    @Test("Non-finite extended components fail explicitly")
    func nonFiniteComponentFails() throws {
        let image = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 32,
            bitsPerPixel: 128,
            bytesPerRow: 16,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(
                alpha: .last,
                component: .float,
                byteOrder: .order32Little
            ),
            data: float32Data([.nan, 0, 0, 1])
        )

        #expect(throws: CAImageContentsConversionError.nonFinitePixelComponent(
            pixelIndex: 0,
            componentIndex: 0
        )) {
            try CGImageTextureStorageConverter.convert(image)
        }
    }

    @Test("Mip generation weights straight RGB by alpha")
    func rgba8MipGenerationIsAlphaCorrect() throws {
        let source = CGImageTextureStorage(
            format: .rgba8Unorm,
            width: 2,
            height: 1,
            data: Data([
                255, 0, 0, 0,
                0, 0, 255, 255,
            ])
        )

        let mip = try CGImageTextureMipGenerator.nextLevel(from: source)

        #expect(mip.width == 1)
        #expect(mip.height == 1)
        #expect(mip.data == Data([0, 0, 255, 128]))
    }

    @Test("Half-float mip generation preserves extended values")
    func rgba16FloatMipGenerationPreservesExtendedValues() throws {
        let source = CGImageTextureStorage(
            format: .rgba16Float,
            width: 2,
            height: 1,
            data: halfFloatData([
                4, 0, 0, 0.5,
                0, 2, 0, 0.5,
            ])
        )

        let mip = try CGImageTextureMipGenerator.nextLevel(from: source)

        #expect(halfFloatComponents(in: mip.data) == [2, 1, 0, 0.5])
    }

    @Test("Mip memory accounting uses the selected bytes per pixel")
    func mipMemoryAccountingUsesPixelFormat() throws {
        #expect(try CGImageTextureStorage.mipmappedByteCount(
            width: 4,
            height: 4,
            format: .rgba8Unorm
        ) == 84)
        #expect(try CGImageTextureStorage.mipmappedByteCount(
            width: 4,
            height: 4,
            format: .rgba16Float
        ) == 168)
    }

    @Test("Insufficient storage is rejected before pointer access")
    func insufficientStorageFails() throws {
        let image = try makeImage(
            width: 2,
            height: 2,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 8,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            data: Data(repeating: 0, count: 15)
        )

        #expect(throws: CAImageContentsConversionError.insufficientPixelData(
            required: 16,
            actual: 15
        )) {
            try CGImageTextureStorageConverter.convert(image)
        }
    }

    @Test("Invalid stride and unsupported component packing fail explicitly")
    func invalidLayoutsFail() throws {
        let invalidStride = try makeImage(
            width: 2,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 4,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            data: Data(repeating: 0, count: 8)
        )
        let unsupportedPacking = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 7,
            bitsPerPixel: 28,
            bytesPerRow: 4,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            data: Data(repeating: 0, count: 4)
        )

        #expect(throws: CAImageContentsConversionError.invalidBytesPerRow(minimum: 8, actual: 4)) {
            try CGImageTextureStorageConverter.convert(invalidStride)
        }
        #expect(throws: CAImageContentsConversionError.unsupportedPixelLayout(
            bitsPerComponent: 7,
            bitsPerPixel: 28
        )) {
            try CGImageTextureStorageConverter.convert(unsupportedPacking)
        }
    }

    @Test("Overflowing pixel storage is rejected before allocation")
    func pixelStorageOverflowFails() throws {
        let image = try makeImage(
            width: Int.max,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: Int.max,
            colorSpace: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            data: Data()
        )

        #expect(throws: CAImageContentsConversionError.pixelStorageOverflow) {
            try CGImageTextureStorageConverter.convert(image)
        }
    }

    private func makeImage(
        headroom: Float? = nil,
        width: Int,
        height: Int,
        bitsPerComponent: Int,
        bitsPerPixel: Int,
        bytesPerRow: Int,
        colorSpace: CGColorSpace,
        bitmapInfo: CGBitmapInfo,
        data: Data
    ) throws -> CGImage {
        let provider = CGDataProvider(data: data)
        if let headroom {
            return try #require(CGImage(
                headroom: headroom,
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bitsPerPixel: bitsPerPixel,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            ))
        }
        return try #require(CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ))
    }

    private func halfFloatData(_ components: [Float16]) -> Data {
        var data = Data()
        data.reserveCapacity(components.count * 2)
        for component in components {
            var bits = component.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }

    private func float32Data(_ components: [Float]) -> Data {
        var data = Data()
        data.reserveCapacity(components.count * 4)
        for component in components {
            var bits = component.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }

    private func halfFloatComponents(in data: Data) -> [Float16] {
        data.withUnsafeBytes { bytes in
            (0..<(data.count / 2)).map { componentIndex in
                let offset = componentIndex * 2
                let bits = UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8
                return Float16(bitPattern: bits)
            }
        }
    }
}
