
/// Constants that specify the shape of the emitter.
public struct CAEmitterLayerEmitterShape: Hashable, Equatable, RawRepresentable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Particles are emitted from a single point.
    public static let point = CAEmitterLayerEmitterShape(rawValue: "point")

    /// Particles are emitted from points on a line segment.
    public static let line = CAEmitterLayerEmitterShape(rawValue: "line")

    /// Particles are emitted from points on a rectangle.
    public static let rectangle = CAEmitterLayerEmitterShape(rawValue: "rectangle")

    /// Particles are emitted from points on a cubic volume.
    public static let cuboid = CAEmitterLayerEmitterShape(rawValue: "cuboid")

    /// Particles are emitted from points on a circle.
    public static let circle = CAEmitterLayerEmitterShape(rawValue: "circle")

    /// Particles are emitted from points on a sphere.
    public static let sphere = CAEmitterLayerEmitterShape(rawValue: "sphere")
}
