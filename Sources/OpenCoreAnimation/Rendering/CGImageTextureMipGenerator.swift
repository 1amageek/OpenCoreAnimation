import Foundation

/// Builds alpha-correct box-filtered mip levels for straight-alpha image textures.
internal enum CGImageTextureMipGenerator {
    internal static func nextLevel(
        from source: CGImageTextureStorage
    ) throws(CAImageContentsConversionError) -> CGImageTextureStorage {
        guard source.width > 0, source.height > 0 else {
            throw .invalidDimensions(width: source.width, height: source.height)
        }
        let expectedByteCount = try byteCount(
            width: source.width,
            height: source.height,
            bytesPerPixel: source.format.bytesPerPixel
        )
        guard source.data.count >= expectedByteCount else {
            throw .insufficientPixelData(required: expectedByteCount, actual: source.data.count)
        }
        guard source.width > 1 || source.height > 1 else { return source }

        let destinationWidth = max(1, source.width / 2)
        let destinationHeight = max(1, source.height / 2)
        let destinationByteCount = try byteCount(
            width: destinationWidth,
            height: destinationHeight,
            bytesPerPixel: source.format.bytesPerPixel
        )
        var destination = Data(count: destinationByteCount)

        var generationError: CAImageContentsConversionError?
        source.data.withUnsafeBytes { sourceBytes in
            destination.withUnsafeMutableBytes { destinationBytes in
                guard let sourceBase = sourceBytes.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let destinationBase = destinationBytes.baseAddress?.assumingMemoryBound(
                        to: UInt8.self
                      ) else {
                    generationError = .missingPixelData
                    return
                }
                for destinationY in 0..<destinationHeight {
                    for destinationX in 0..<destinationWidth {
                        guard generationError == nil else { return }
                        let sourceMinX = destinationX * source.width / destinationWidth
                        let sourceMinY = destinationY * source.height / destinationHeight
                        let sourceMaxX = max(
                            sourceMinX + 1,
                            (destinationX + 1) * source.width / destinationWidth
                        )
                        let sourceMaxY = max(
                            sourceMinY + 1,
                            (destinationY + 1) * source.height / destinationHeight
                        )
                        var premultipliedRGB = SIMD3<Float>.zero
                        var alphaSum: Float = 0
                        var sampleCount: Float = 0
                        for sourceY in sourceMinY..<sourceMaxY {
                            for sourceX in sourceMinX..<sourceMaxX {
                                let pixelIndex = sourceY * source.width + sourceX
                                let sample: SIMD4<Float>
                                switch readPixel(
                                    sourceBase,
                                    pixelIndex: pixelIndex,
                                    format: source.format
                                ) {
                                case .success(let value):
                                    sample = value
                                case .failure(let error):
                                    generationError = error
                                    return
                                }
                                premultipliedRGB += SIMD3(sample.x, sample.y, sample.z) * sample.w
                                alphaSum += sample.w
                                sampleCount += 1
                            }
                        }
                        let alpha = alphaSum / sampleCount
                        let rgb = alphaSum > 0
                            ? premultipliedRGB / alphaSum
                            : SIMD3<Float>.zero
                        let destinationPixelIndex = destinationY * destinationWidth + destinationX
                        writePixel(
                            SIMD4(rgb.x, rgb.y, rgb.z, alpha),
                            to: destinationBase,
                            pixelIndex: destinationPixelIndex,
                            format: source.format
                        )
                    }
                }
            }
        }
        if let generationError { throw generationError }
        return CGImageTextureStorage(
            format: source.format,
            width: destinationWidth,
            height: destinationHeight,
            data: destination
        )
    }

    private static func readPixel(
        _ bytes: UnsafePointer<UInt8>,
        pixelIndex: Int,
        format: CGImageTexturePixelFormat
    ) -> Result<SIMD4<Float>, CAImageContentsConversionError> {
        switch format {
        case .rgba8Unorm:
            let offset = pixelIndex * 4
            return .success(SIMD4(
                Float(bytes[offset]) / 255,
                Float(bytes[offset + 1]) / 255,
                Float(bytes[offset + 2]) / 255,
                Float(bytes[offset + 3]) / 255
            ))
        case .rgba16Float:
            let offset = pixelIndex * 8
            var result = SIMD4<Float>.zero
            for componentIndex in 0..<4 {
                let componentOffset = offset + componentIndex * 2
                let bits = UInt16(bytes[componentOffset])
                    | UInt16(bytes[componentOffset + 1]) << 8
                let value = Float(Float16(bitPattern: bits))
                guard value.isFinite else {
                    return .failure(.nonFinitePixelComponent(
                        pixelIndex: pixelIndex,
                        componentIndex: componentIndex
                    ))
                }
                if componentIndex == 3, value < 0 || value > 1 {
                    return .failure(.invalidAlphaComponent(pixelIndex: pixelIndex))
                }
                result[componentIndex] = value
            }
            return .success(result)
        }
    }

    private static func writePixel(
        _ pixel: SIMD4<Float>,
        to bytes: UnsafeMutablePointer<UInt8>,
        pixelIndex: Int,
        format: CGImageTexturePixelFormat
    ) {
        switch format {
        case .rgba8Unorm:
            let offset = pixelIndex * 4
            for componentIndex in 0..<4 {
                let normalized = min(1, max(0, pixel[componentIndex]))
                bytes[offset + componentIndex] = UInt8((normalized * 255).rounded())
            }
        case .rgba16Float:
            let offset = pixelIndex * 8
            for componentIndex in 0..<4 {
                let bits = Float16(pixel[componentIndex]).bitPattern.littleEndian
                bytes[offset + componentIndex * 2] = UInt8(truncatingIfNeeded: bits)
                bytes[offset + componentIndex * 2 + 1] = UInt8(truncatingIfNeeded: bits >> 8)
            }
        }
    }

    private static func byteCount(
        width: Int,
        height: Int,
        bytesPerPixel: Int
    ) throws(CAImageContentsConversionError) -> Int {
        let (pixelCount, pixelOverflow) = width.multipliedReportingOverflow(by: height)
        let (result, byteOverflow) = pixelCount.multipliedReportingOverflow(by: bytesPerPixel)
        guard !pixelOverflow, !byteOverflow else { throw .pixelStorageOverflow }
        return result
    }
}
