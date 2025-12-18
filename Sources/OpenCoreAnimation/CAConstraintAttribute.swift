
/// The constraint attribute type.
public enum CAConstraintAttribute: Int32, Sendable {
    /// The left edge of a layer's frame.
    case minX = 1
    /// The horizontal location of the center of a layer's frame.
    case midX = 2
    /// The right edge of a layer's frame.
    case maxX = 3
    /// The width of a layer.
    case width = 4
    /// The bottom edge of a layer's frame.
    case minY = 5
    /// The vertical location of the center of a layer's frame.
    case midY = 6
    /// The top edge of a layer's frame.
    case maxY = 7
    /// The height of a layer.
    case height = 8
}
