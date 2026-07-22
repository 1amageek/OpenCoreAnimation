import Foundation

/// Concrete CPU storage selected from a layer's contents-format hint.
internal enum CADelegateBackingStoreFormat: Equatable, Sendable {
    case rgba8Uint
    case rgba16Float
    case gray8Uint

    static func resolve(
        contentsFormat: CALayerContentsFormat,
        contentsHeadroom: CGFloat
    ) throws(CADelegateBackingStoreError) -> Self {
        switch contentsFormat {
        case .automatic:
            return contentsHeadroom > 1 ? .rgba16Float : .rgba8Uint
        case .RGBA8Uint:
            return .rgba8Uint
        case .RGBA16Float:
            return .rgba16Float
        case .gray8Uint:
            return .gray8Uint
        default:
            throw .unsupportedContentsFormat(contentsFormat.rawValue)
        }
    }

    var contentsFormat: CALayerContentsFormat {
        switch self {
        case .rgba8Uint: return .RGBA8Uint
        case .rgba16Float: return .RGBA16Float
        case .gray8Uint: return .gray8Uint
        }
    }

    var bitsPerComponent: Int {
        switch self {
        case .rgba8Uint, .gray8Uint: return 8
        case .rgba16Float: return 16
        }
    }

    var bitsPerPixel: Int {
        switch self {
        case .rgba8Uint: return 32
        case .rgba16Float: return 64
        case .gray8Uint: return 8
        }
    }
}
