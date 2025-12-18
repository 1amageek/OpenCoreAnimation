
/// Constants that specify the emitter mode.
public struct CAEmitterLayerEmitterMode: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Particles are emitted from points on the emitter shape.
    public static let points = CAEmitterLayerEmitterMode(rawValue: "points")

    /// Particles are emitted from the outline of the emitter shape.
    public static let outline = CAEmitterLayerEmitterMode(rawValue: "outline")

    /// Particles are emitted from the surface of the emitter shape.
    public static let surface = CAEmitterLayerEmitterMode(rawValue: "surface")

    /// Particles are emitted from the volume of the emitter shape.
    public static let volume = CAEmitterLayerEmitterMode(rawValue: "volume")
}
