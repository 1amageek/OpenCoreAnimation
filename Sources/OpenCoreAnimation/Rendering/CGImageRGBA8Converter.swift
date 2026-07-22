import Foundation

/// Normalizes `CGImage` storage to tightly packed, straight-alpha RGBA8 pixels.
internal struct CGImageRGBA8Converter {
    internal static func convert(_ image: CGImage) throws(CAImageContentsConversionError) -> Data {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else {
            throw .invalidDimensions(width: width, height: height)
        }
        guard image.bitsPerComponent > 0,
              image.bitsPerComponent % 8 == 0,
              image.bitsPerPixel > 0,
              image.bitsPerPixel % 8 == 0 else {
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

        let (destinationBytesPerRow, destinationRowOverflow) = width.multipliedReportingOverflow(by: 4)
        let (destinationByteCount, destinationSizeOverflow) = destinationBytesPerRow
            .multipliedReportingOverflow(by: height)
        guard !destinationRowOverflow, !destinationSizeOverflow else {
            throw .pixelStorageOverflow
        }

        if canUseStraightRGBAStorageDirectly(image, packedBytesPerRow: minimumBytesPerRow) {
            return sourceData.count == requiredByteCount
                ? sourceData
                : Data(sourceData.prefix(requiredByteCount))
        }

        guard let convertedImage = image.copy(colorSpace: .deviceRGB),
              convertedImage.bitsPerComponent == 8,
              convertedImage.bitsPerPixel == 32,
              convertedImage.bytesPerRow == destinationBytesPerRow,
              convertedImage.alphaInfo == .premultipliedLast,
              let convertedData = convertedImage.data ?? convertedImage.dataProvider?.data else {
            throw .conversionFailed
        }

        guard convertedData.count >= destinationByteCount else {
            throw .conversionFailed
        }
        return straightAlphaRGBA(
            fromPremultipliedRGBA: convertedData,
            byteCount: destinationByteCount
        )
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

    private static func straightAlphaRGBA(
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
}
