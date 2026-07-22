import Foundation

/// Describes why a text layer could not enter the renderer's text pipeline.
@_spi(RendererDiagnostics)
public enum CATextRenderFailure: Error, Equatable, Sendable {
    case unsupportedStringValue
    case unsupportedFontValue
    case invalidFontSize
    case invalidContentsScale
    case invalidBounds
    case invalidForegroundColor
    case unsupportedAlignmentMode(String)
    case unsupportedTruncationMode(String)
    case rendererResourcesUnavailable
    case canvas2DUnavailable
    case textMeasurementUnavailable
    case textureDimensionsUnsupported
    case imageDataUnavailable
    case imageDataStorageUnavailable
}

/// Validated, renderer-independent text input.
internal struct CATextRenderConfiguration {
    let text: String
    let fontFamily: String
    let cssFontFamily: String
    let fontSize: CGFloat
    let contentsScale: CGFloat
    let foregroundRGBA: SIMD4<Float>
    let alignmentMode: CATextLayerAlignmentMode
    let truncationMode: CATextLayerTruncationMode
    let bounds: CGRect
    let isWrapped: Bool

    init(layer: CATextLayer) throws(CATextRenderFailure) {
        guard let text = layer.string as? String else {
            throw .unsupportedStringValue
        }
        guard let fontFamily = layer.font as? String,
              !fontFamily.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw .unsupportedFontValue
        }
        guard layer.fontSize.isFinite, layer.fontSize > 0 else {
            throw .invalidFontSize
        }
        guard layer.contentsScale.isFinite, layer.contentsScale > 0 else {
            throw .invalidContentsScale
        }
        guard layer.bounds.origin.x.isFinite,
              layer.bounds.origin.y.isFinite,
              layer.bounds.width.isFinite,
              layer.bounds.height.isFinite,
              layer.bounds.width >= 0,
              layer.bounds.height >= 0 else {
            throw .invalidBounds
        }

        let validAlignmentModes: [CATextLayerAlignmentMode] = [
            .left, .right, .center, .justified, .natural,
        ]
        guard validAlignmentModes.contains(layer.alignmentMode) else {
            throw .unsupportedAlignmentMode(layer.alignmentMode.rawValue)
        }
        let validTruncationModes: [CATextLayerTruncationMode] = [
            .none, .start, .middle, .end,
        ]
        guard validTruncationModes.contains(layer.truncationMode) else {
            throw .unsupportedTruncationMode(layer.truncationMode.rawValue)
        }

        let foreground = layer.foregroundColor ?? .white
        guard let converted = foreground.converted(
            to: .deviceRGB,
            intent: .defaultIntent,
            options: nil
        ), let components = converted.components,
           components.count == 4,
           components.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw .invalidForegroundColor
        }

        self.text = text
        self.fontFamily = fontFamily
        self.cssFontFamily = Self.cssFontFamily(from: fontFamily)
        self.fontSize = layer.fontSize
        self.contentsScale = layer.contentsScale
        self.foregroundRGBA = SIMD4(
            Float(components[0]),
            Float(components[1]),
            Float(components[2]),
            Float(components[3])
        )
        self.alignmentMode = layer.alignmentMode
        self.truncationMode = layer.truncationMode
        self.bounds = layer.bounds
        self.isWrapped = layer.isWrapped
    }

    private static func cssFontFamily(from fontFamily: String) -> String {
        let genericFamilies: Set<String> = [
            "serif", "sans-serif", "monospace", "cursive", "fantasy",
            "system-ui", "ui-serif", "ui-sans-serif", "ui-monospace",
            "ui-rounded", "math", "emoji", "fangsong",
        ]
        if genericFamilies.contains(fontFamily.lowercased()) {
            return fontFamily.lowercased()
        }
        let escaped = fontFamily
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }
}
