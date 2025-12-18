
/// Constants that specify the filter used when reducing the size of the content.
public struct CALayerContentsFilter: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Linear interpolation filter.
    public static let linear = CALayerContentsFilter(rawValue: "linear")

    /// Nearest neighbor interpolation filter.
    public static let nearest = CALayerContentsFilter(rawValue: "nearest")

    /// Trilinear minification filter, which enables mipmap generation.
    public static let trilinear = CALayerContentsFilter(rawValue: "trilinear")
}
