import Foundation

/// Tightly packed, straight-alpha pixels ready for GPU texture upload.
internal struct CGImageTextureStorage {
    internal let format: CGImageTexturePixelFormat
    internal let width: Int
    internal let height: Int
    internal let data: Data

    internal var bytesPerRow: Int {
        width * format.bytesPerPixel
    }

    internal static func mipmappedByteCount(
        width: Int,
        height: Int,
        format: CGImageTexturePixelFormat
    ) throws(CAImageContentsConversionError) -> UInt64 {
        guard width > 0, height > 0 else {
            throw .invalidDimensions(width: width, height: height)
        }

        var levelWidth = UInt64(width)
        var levelHeight = UInt64(height)
        let bytesPerPixel = UInt64(format.bytesPerPixel)
        var byteCount: UInt64 = 0
        while true {
            let (pixelCount, pixelCountOverflow) = levelWidth.multipliedReportingOverflow(
                by: levelHeight
            )
            let (levelByteCount, levelByteCountOverflow) = pixelCount.multipliedReportingOverflow(
                by: bytesPerPixel
            )
            let (newByteCount, totalOverflow) = byteCount.addingReportingOverflow(levelByteCount)
            guard !pixelCountOverflow, !levelByteCountOverflow, !totalOverflow else {
                throw .pixelStorageOverflow
            }
            byteCount = newByteCount
            guard levelWidth > 1 || levelHeight > 1 else { return byteCount }
            levelWidth = max(1, levelWidth / 2)
            levelHeight = max(1, levelHeight / 2)
        }
    }
}
