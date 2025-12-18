
/// Constants that specify the rendering mode.
public struct CAEmitterLayerRenderMode: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Particles are rendered in the order they were created.
    public static let unordered = CAEmitterLayerRenderMode(rawValue: "unordered")

    /// Particles are rendered from oldest to youngest.
    public static let oldestFirst = CAEmitterLayerRenderMode(rawValue: "oldestFirst")

    /// Particles are rendered from youngest to oldest.
    public static let oldestLast = CAEmitterLayerRenderMode(rawValue: "oldestLast")

    /// Particles are rendered from back to front.
    public static let backToFront = CAEmitterLayerRenderMode(rawValue: "backToFront")

    /// Particles are rendered additively.
    public static let additive = CAEmitterLayerRenderMode(rawValue: "additive")
}
