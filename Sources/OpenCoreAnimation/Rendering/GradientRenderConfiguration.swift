import Foundation

struct GradientRenderConfiguration {
    let renderMode: Float
    let colors: [CGColor]
    let locations: [Float]

    init(
        type: CAGradientLayerType,
        colors colorValues: [Any],
        locations locationValues: [CGFloat]?,
        startPoint: CGPoint,
        endPoint: CGPoint
    ) throws {
        renderMode = try Self.renderMode(for: type)
        try Self.validateGeometry(startPoint: startPoint, endPoint: endPoint)

        var validatedColors: [CGColor] = []
        validatedColors.reserveCapacity(colorValues.count)
        for (index, value) in colorValues.enumerated() {
            guard let color = value as? CGColor else {
                throw GradientRenderConfigurationError.invalidColor(index: index)
            }
            guard let components = color.components,
                  !components.isEmpty,
                  components.allSatisfy(\.isFinite) else {
                throw GradientRenderConfigurationError.invalidColorComponents(index: index)
            }
            validatedColors.append(color)
        }
        colors = validatedColors

        if let locationValues {
            guard locationValues.count == colorValues.count else {
                throw GradientRenderConfigurationError.invalidLocationCount(
                    expected: colorValues.count,
                    actual: locationValues.count
                )
            }

            var previousLocation: CGFloat?
            var validatedLocations: [Float] = []
            validatedLocations.reserveCapacity(locationValues.count)
            for (index, location) in locationValues.enumerated() {
                guard location.isFinite else {
                    throw GradientRenderConfigurationError.nonFiniteLocation(index: index)
                }
                guard (0...1).contains(location) else {
                    throw GradientRenderConfigurationError.locationOutOfRange(index: index)
                }
                if let previousLocation, location < previousLocation {
                    throw GradientRenderConfigurationError.locationsNotMonotonic(index: index)
                }
                validatedLocations.append(Float(location))
                previousLocation = location
            }
            locations = validatedLocations
        } else {
            let denominator = max(colorValues.count - 1, 1)
            locations = colorValues.indices.map { Float($0) / Float(denominator) }
        }
    }

    static func parameter(
        at point: CGPoint,
        type: CAGradientLayerType,
        startPoint: CGPoint,
        endPoint: CGPoint
    ) throws -> CGFloat? {
        try validateGeometry(startPoint: startPoint, endPoint: endPoint)
        guard point.x.isFinite, point.y.isFinite else {
            throw GradientRenderConfigurationError.nonFiniteGeometry
        }

        let deltaX = endPoint.x - startPoint.x
        let deltaY = endPoint.y - startPoint.y
        let relativeX = point.x - startPoint.x
        let relativeY = point.y - startPoint.y

        switch type {
        case .axial:
            let squaredLength = deltaX * deltaX + deltaY * deltaY
            guard squaredLength > 0 else { return 0 }
            return (relativeX * deltaX + relativeY * deltaY) / squaredLength
        case .radial:
            let radiusX = abs(deltaX)
            let radiusY = abs(deltaY)
            guard radiusX > 0, radiusY > 0 else { return nil }
            let normalizedX = relativeX / radiusX
            let normalizedY = relativeY / radiusY
            return sqrt(normalizedX * normalizedX + normalizedY * normalizedY)
        case .conic:
            let directionAngle = atan2(deltaY, deltaX)
            let pointAngle = atan2(relativeY, relativeX)
            let turns = (pointAngle - directionAngle) / (2 * .pi) + 1
            return turns - floor(turns)
        default:
            throw GradientRenderConfigurationError.unsupportedType(type.rawValue)
        }
    }

    private static func renderMode(for type: CAGradientLayerType) throws -> Float {
        switch type {
        case .axial:
            return 2
        case .radial:
            return 3
        case .conic:
            return 4
        default:
            throw GradientRenderConfigurationError.unsupportedType(type.rawValue)
        }
    }

    private static func validateGeometry(
        startPoint: CGPoint,
        endPoint: CGPoint
    ) throws {
        guard startPoint.x.isFinite,
              startPoint.y.isFinite,
              endPoint.x.isFinite,
              endPoint.y.isFinite else {
            throw GradientRenderConfigurationError.nonFiniteGeometry
        }
    }
}
