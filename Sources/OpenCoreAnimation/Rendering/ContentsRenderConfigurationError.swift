import Foundation

@_spi(RendererDiagnostics)
public enum ContentsRenderConfigurationError: Error, Equatable, Sendable {
    case invalidImageSize
    case invalidBounds
    case invalidContentsRect
    case invalidContentsCenter
    case invalidContentsScale
}
