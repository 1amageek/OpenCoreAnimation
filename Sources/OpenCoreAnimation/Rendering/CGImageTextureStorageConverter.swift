import Foundation

/// Converts `CGImage` storage into the straight-alpha formats sampled by the renderer.
internal struct CGImageTextureStorageConverter {
    private struct SourceStorage {
        let data: Data
        let requiredByteCount: Int
    }

    internal static func convert(
        _ image: CGImage,
        to format: CGImageTexturePixelFormat? = nil
    ) throws(CAImageContentsConversionError) -> CGImageTextureStorage {
        let selectedFormat = format ?? .recommended(for: image)
        let source = try validatedSourceStorage(for: image)
        let data: Data
        switch selectedFormat {
        case .rgba8Unorm:
            data = try convertToRGBA8(image, source: source)
        case .rgba16Float:
            data = try convertToRGBA16Float(image, source: source)
        }
        return CGImageTextureStorage(
            format: selectedFormat,
            width: image.width,
            height: image.height,
            data: data
        )
    }

    private static func validatedSourceStorage(
        for image: CGImage
    ) throws(CAImageContentsConversionError) -> SourceStorage {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else {
            throw .invalidDimensions(width: width, height: height)
        }
        guard image.bitsPerComponent > 0,
              image.bitsPerComponent % 8 == 0,
              image.bitsPerPixel > 0,
              image.bitsPerPixel % 8 == 0,
              image.bitsPerPixel % image.bitsPerComponent == 0 else {
            throw .unsupportedPixelLayout(
                bitsPerComponent: image.bitsPerComponent,
                bitsPerPixel: image.bitsPerPixel
            )
        }

        let bytesPerPixel = image.bitsPerPixel / 8
        let (minimumBytesPerRow, rowOverflow) = width.multipliedReportingOverflow(by: bytesPerPixel)
        guard !rowOverflow else { throw .pixelStorageOverflow }
        guard image.bytesPerRow >= minimumBytesPerRow else {
            throw .invalidBytesPerRow(minimum: minimumBytesPerRow, actual: image.bytesPerRow)
        }

        let (precedingRowsByteCount, rowsOverflow) = image.bytesPerRow.multipliedReportingOverflow(
            by: height - 1
        )
        let (requiredByteCount, totalOverflow) = precedingRowsByteCount.addingReportingOverflow(
            minimumBytesPerRow
        )
        guard !rowsOverflow, !totalOverflow else { throw .pixelStorageOverflow }

        guard let sourceData = image.data ?? image.dataProvider?.data else {
            throw .missingPixelData
        }
        guard sourceData.count >= requiredByteCount else {
            throw .insufficientPixelData(required: requiredByteCount, actual: sourceData.count)
        }
        return SourceStorage(data: sourceData, requiredByteCount: requiredByteCount)
    }

    private static func convertToRGBA8(
        _ image: CGImage,
        source: SourceStorage
    ) throws(CAImageContentsConversionError) -> Data {
        let destinationBytesPerRow: Int
        let destinationByteCount: Int
        do {
            destinationBytesPerRow = try multiplied(image.width, by: 4)
            destinationByteCount = try multiplied(destinationBytesPerRow, by: image.height)
        } catch {
            throw .pixelStorageOverflow
        }

        if canUseStraightRGBAStorageDirectly(image, packedBytesPerRow: destinationBytesPerRow) {
            return source.data.count == source.requiredByteCount
                ? source.data
                : Data(source.data.prefix(source.requiredByteCount))
        }

        guard let convertedImage = image.copy(colorSpace: .deviceRGB),
              convertedImage.bitsPerComponent == 8,
              convertedImage.bitsPerPixel == 32,
              convertedImage.bytesPerRow == destinationBytesPerRow,
              convertedImage.alphaInfo == .premultipliedLast,
              let convertedData = convertedImage.data ?? convertedImage.dataProvider?.data,
              convertedData.count >= destinationByteCount else {
            throw .conversionFailed
        }

        return straightAlphaRGBA8(
            fromPremultipliedRGBA: convertedData,
            byteCount: destinationByteCount
        )
    }

    private static func convertToRGBA16Float(
        _ image: CGImage,
        source: SourceStorage
    ) throws(CAImageContentsConversionError) -> Data {
        guard let sourceColorSpace = image.colorSpace else {
            throw .missingColorSpace
        }
        guard let destinationColorSpace = CGColorSpace(
            name: CGColorSpace.extendedLinearSRGB
        ), let conversion = CGColorConversionInfo(
            src: sourceColorSpace,
            dst: destinationColorSpace
        ) else {
            throw .unsupportedColorSpace
        }

        let bytesPerRow: Int
        let byteCount: Int
        do {
            bytesPerRow = try multiplied(image.width, by: 8)
            byteCount = try multiplied(bytesPerRow, by: image.height)
        } catch {
            throw .pixelStorageOverflow
        }
        let destinationFormat = CGColorBufferFormat(
            version: 0,
            bitmapInfo: CGBitmapInfo(
                alpha: .last,
                component: .float,
                byteOrder: .order16Little
            ),
            bitsPerComponent: 16,
            bitsPerPixel: 64,
            bytesPerRow: bytesPerRow
        )
        let sourceFormat = CGColorBufferFormat(
            version: 0,
            bitmapInfo: image.bitmapInfo,
            bitsPerComponent: image.bitsPerComponent,
            bitsPerPixel: image.bitsPerPixel,
            bytesPerRow: image.bytesPerRow
        )
        var destination = Data(count: byteCount)
        let converted = source.data.withUnsafeBytes { sourceBytes -> Bool in
            guard let sourceAddress = sourceBytes.baseAddress else { return false }
            return destination.withUnsafeMutableBytes { destinationBytes -> Bool in
                guard let destinationAddress = destinationBytes.baseAddress else { return false }
                return conversion.convert(
                    width: image.width,
                    height: image.height,
                    to: destinationAddress,
                    format: destinationFormat,
                    from: sourceAddress,
                    format: sourceFormat,
                    options: nil
                )
            }
        }
        guard converted else { throw .conversionFailed }
        try validateRGBA16Float(destination)
        return destination
    }

    private static func canUseStraightRGBAStorageDirectly(
        _ image: CGImage,
        packedBytesPerRow: Int
    ) -> Bool {
        guard image.bitsPerComponent == 8,
              image.bitsPerPixel == 32,
              image.bytesPerRow == packedBytesPerRow,
              image.pixelFormatInfo == .packed,
              image.byteOrderInfo == .orderDefault,
              image.colorSpace == .deviceRGB else {
            return false
        }
        return image.alphaInfo == .last
    }

    private static func straightAlphaRGBA8(
        fromPremultipliedRGBA source: Data,
        byteCount: Int
    ) -> Data {
        var destination = Data(count: byteCount)
        source.withUnsafeBytes { sourceBytes in
            destination.withUnsafeMutableBytes { destinationBytes in
                guard let sourceBase = sourceBytes.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let destinationBase = destinationBytes.baseAddress?.assumingMemoryBound(
                        to: UInt8.self
                      ) else {
                    return
                }
                for offset in stride(from: 0, to: byteCount, by: 4) {
                    let alpha = UInt32(sourceBase[offset + 3])
                    destinationBase[offset + 3] = UInt8(alpha)
                    guard alpha > 0 else {
                        destinationBase[offset] = 0
                        destinationBase[offset + 1] = 0
                        destinationBase[offset + 2] = 0
                        continue
                    }
                    for component in 0..<3 {
                        let premultiplied = UInt32(sourceBase[offset + component])
                        let straight = min(255, (premultiplied * 255 + alpha / 2) / alpha)
                        destinationBase[offset + component] = UInt8(straight)
                    }
                }
            }
        }
        return destination
    }

    private static func validateRGBA16Float(
        _ data: Data
    ) throws(CAImageContentsConversionError) {
        var validationError: CAImageContentsConversionError?
        data.withUnsafeBytes { bytes in
            let pixelCount = data.count / 8
            for pixelIndex in 0..<pixelCount {
                for componentIndex in 0..<4 {
                    let offset = pixelIndex * 8 + componentIndex * 2
                    let bits = UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8
                    let value = Float16(bitPattern: bits)
                    guard value.isFinite else {
                        validationError = .nonFinitePixelComponent(
                            pixelIndex: pixelIndex,
                            componentIndex: componentIndex
                        )
                        return
                    }
                    if componentIndex == 3, value < 0 || value > 1 {
                        validationError = .invalidAlphaComponent(pixelIndex: pixelIndex)
                        return
                    }
                }
            }
        }
        if let validationError { throw validationError }
    }

    private static func multiplied(_ lhs: Int, by rhs: Int) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw CAImageContentsConversionError.pixelStorageOverflow }
        return result
    }
}
