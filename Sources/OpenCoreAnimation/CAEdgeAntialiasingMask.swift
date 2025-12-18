
/// A mask used by the `edgeAntialiasingMask` property.
public struct CAEdgeAntialiasingMask: OptionSet, Sendable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    /// Antialias the left edge of the layer.
    public static let layerLeftEdge = CAEdgeAntialiasingMask(rawValue: 1 << 0)

    /// Antialias the right edge of the layer.
    public static let layerRightEdge = CAEdgeAntialiasingMask(rawValue: 1 << 1)

    /// Antialias the bottom edge of the layer.
    public static let layerBottomEdge = CAEdgeAntialiasingMask(rawValue: 1 << 2)

    /// Antialias the top edge of the layer.
    public static let layerTopEdge = CAEdgeAntialiasingMask(rawValue: 1 << 3)
}
