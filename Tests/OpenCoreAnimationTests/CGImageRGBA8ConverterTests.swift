import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CGImage RGBA8 conversion")
struct CGImageRGBA8ConverterTests {
    @Test("Premultiplied RGBA is normalized to straight alpha")
    func premultipliedRGBAIsUnpremultiplied() throws {
        let image = try makeImage(
            width: 2,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 8,
            colorSpace: .deviceRGB,
            alphaInfo: .premultipliedLast,
            data: Data([64, 32, 16, 128, 99, 88, 77, 0])
        )

        #expect(try CGImageRGBA8Converter.convert(image) == Data([
            128, 64, 32, 128,
            0, 0, 0, 0,
        ]))
    }

    @Test("Straight RGBA with row padding is tightly repacked")
    func straightRGBAWithPaddingIsRepacked() throws {
        let image = try makeImage(
            width: 1,
            height: 2,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 8,
            colorSpace: .deviceRGB,
            alphaInfo: .last,
            data: Data([
                10, 20, 30, 255, 1, 2, 3, 4,
                40, 50, 60, 255,
            ])
        )

        #expect(try CGImageRGBA8Converter.convert(image) == Data([
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
            alphaInfo: .premultipliedFirst,
            data: Data([128, 64, 32, 16])
        )
        let gray = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: 1,
            colorSpace: .deviceGray,
            alphaInfo: .none,
            data: Data([77])
        )
        let skippedAlpha = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 4,
            colorSpace: .deviceRGB,
            alphaInfo: .noneSkipLast,
            data: Data([10, 20, 30, 5])
        )

        #expect(try CGImageRGBA8Converter.convert(argb) == Data([128, 64, 32, 128]))
        #expect(try CGImageRGBA8Converter.convert(skippedAlpha) == Data([10, 20, 30, 255]))
        #expect(try CGImageRGBA8Converter.convert(gray) == Data([77, 77, 77, 255]))
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
            alphaInfo: .premultipliedLast,
            data: Data(repeating: 0, count: 15)
        )

        #expect(throws: CAImageContentsConversionError.insufficientPixelData(
            required: 16,
            actual: 15
        )) {
            try CGImageRGBA8Converter.convert(image)
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
            alphaInfo: .premultipliedLast,
            data: Data(repeating: 0, count: 8)
        )
        let unsupportedPacking = try makeImage(
            width: 1,
            height: 1,
            bitsPerComponent: 7,
            bitsPerPixel: 28,
            bytesPerRow: 4,
            colorSpace: .deviceRGB,
            alphaInfo: .premultipliedLast,
            data: Data(repeating: 0, count: 4)
        )

        #expect(throws: CAImageContentsConversionError.invalidBytesPerRow(minimum: 8, actual: 4)) {
            try CGImageRGBA8Converter.convert(invalidStride)
        }
        #expect(throws: CAImageContentsConversionError.unsupportedPixelLayout(
            bitsPerComponent: 7,
            bitsPerPixel: 28
        )) {
            try CGImageRGBA8Converter.convert(unsupportedPacking)
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
            alphaInfo: .premultipliedLast,
            data: Data()
        )

        #expect(throws: CAImageContentsConversionError.pixelStorageOverflow) {
            try CGImageRGBA8Converter.convert(image)
        }
    }

    private func makeImage(
        width: Int,
        height: Int,
        bitsPerComponent: Int,
        bitsPerPixel: Int,
        bytesPerRow: Int,
        colorSpace: CGColorSpace,
        alphaInfo: CGImageAlphaInfo,
        data: Data
    ) throws -> CGImage {
        try #require(CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: alphaInfo.rawValue),
            provider: CGDataProvider(data: data),
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ))
    }
}
