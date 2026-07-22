import Foundation

enum ContentsRenderConfigurationError: Error, Equatable {
    case invalidImageSize
    case invalidBounds
    case invalidContentsRect
    case invalidContentsCenter
    case invalidContentsScale
    case unsupportedGravity(String)
}
