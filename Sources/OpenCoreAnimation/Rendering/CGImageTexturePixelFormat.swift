import Foundation

/// GPU storage formats supported for `CGImage` layer contents.
internal enum CGImageTexturePixelFormat: Hashable, Sendable {
    case rgba8Unorm
    case rgba16Float

    internal var bytesPerPixel: Int {
        switch self {
        case .rgba8Unorm: return 4
        case .rgba16Float: return 8
        }
    }

    /// Chooses storage from the image's intrinsic sample representation and
    /// color metadata. The result is independent of the layer using the image,
    /// so one `CGImage` always maps to one cached GPU texture format.
    internal static func recommended(for image: CGImage) -> Self {
        let usesExtendedColorSpace = image.colorSpace?.name?.contains("Extended") == true
        if image.bitmapInfo.isFloatComponents
            || image.contentHeadroom > 1
            || image.colorSpace?.isHDR() == true
            || usesExtendedColorSpace {
            return .rgba16Float
        }
        return .rgba8Unorm
    }
}
