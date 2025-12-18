
/// Constants that specify the format of the layer's contents.
public struct CALayerContentsFormat: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// A 32-bit RGBA pixel format.
    public static let RGBA8Uint = CALayerContentsFormat(rawValue: "RGBA8")

    /// A 16-bit float RGBA pixel format.
    public static let RGBA16Float = CALayerContentsFormat(rawValue: "RGBAh")

    /// A gray pixel format.
    public static let gray8Uint = CALayerContentsFormat(rawValue: "Gray8")
}
