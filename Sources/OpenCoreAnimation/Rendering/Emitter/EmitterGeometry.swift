import Foundation

/// Samples documented Core Animation emitter geometry and three-dimensional emission cones.
enum EmitterGeometry {
    static func position(
        shape: CAEmitterLayerEmitterShape,
        mode: CAEmitterLayerEmitterMode,
        position: CGPoint,
        zPosition: CGFloat,
        size: CGSize,
        depth: CGFloat,
        random: inout EmitterRandomSource
    ) -> SIMD3<Float>? {
        guard isSupported(mode: mode) else { return nil }

        let center = SIMD3(Float(position.x), Float(position.y), Float(zPosition))
        let halfWidth = abs(Float(size.width)) * 0.5
        let halfHeight = abs(Float(size.height)) * 0.5
        let halfDepth = abs(Float(depth)) * 0.5
        guard center.x.isFinite, center.y.isFinite, center.z.isFinite else { return nil }

        switch shape {
        case .point:
            return center
        case .line:
            guard halfWidth.isFinite else { return nil }
            return center + lineOffset(
                halfWidth: halfWidth,
                mode: mode,
                random: &random
            )
        case .rectangle:
            guard halfWidth.isFinite, halfHeight.isFinite else { return nil }
            return center + rectangleOffset(
                halfWidth: halfWidth,
                halfHeight: halfHeight,
                mode: mode,
                random: &random
            )
        case .cuboid:
            guard halfWidth.isFinite, halfHeight.isFinite, halfDepth.isFinite else { return nil }
            return center + cuboidOffset(
                halfWidth: halfWidth,
                halfHeight: halfHeight,
                halfDepth: halfDepth,
                mode: mode,
                random: &random
            )
        case .circle:
            guard halfWidth.isFinite else { return nil }
            return center + circleOffset(
                radius: abs(Float(size.width)),
                mode: mode,
                random: &random
            )
        case .sphere:
            guard halfWidth.isFinite else { return nil }
            return center + sphereOffset(
                radius: abs(Float(size.width)),
                mode: mode,
                random: &random
            )
        default:
            return nil
        }
    }

    static func direction(
        longitude: CGFloat,
        latitude: CGFloat,
        range: CGFloat,
        random: inout EmitterRandomSource
    ) -> SIMD3<Float>? {
        let longitude = Float(longitude)
        let latitude = Float(latitude)
        let range = Float(range)
        guard longitude.isFinite, latitude.isFinite, range.isFinite else { return nil }
        let sinLatitude = sin(latitude)
        let axis = SIMD3<Float>(
            sinLatitude * cos(longitude),
            sinLatitude * sin(longitude),
            cos(latitude)
        )

        let halfAngle = min(abs(range) * 0.5, .pi)
        guard halfAngle > 0 else { return axis }

        let cosine = 1 - random.unitFloat() * (1 - cos(halfAngle))
        let sine = sqrt(max(0, 1 - cosine * cosine))
        let azimuth = random.unitFloat() * 2 * Float.pi
        let reference = abs(axis.z) < 0.999
            ? SIMD3<Float>(0, 0, 1)
            : SIMD3<Float>(1, 0, 0)
        let tangent = normalized(cross(reference, axis))
        let bitangent = cross(axis, tangent)
        return normalized(
            axis * cosine
                + tangent * (sine * cos(azimuth))
                + bitangent * (sine * sin(azimuth))
        )
    }

    private static func isSupported(mode: CAEmitterLayerEmitterMode) -> Bool {
        mode == .points || mode == .outline || mode == .surface || mode == .volume
    }

    private static func lineOffset(
        halfWidth: Float,
        mode: CAEmitterLayerEmitterMode,
        random: inout EmitterRandomSource
    ) -> SIMD3<Float> {
        if mode == .points || mode == .outline {
            return SIMD3(random.unitFloat() < 0.5 ? -halfWidth : halfWidth, 0, 0)
        }
        return SIMD3(random.signedFloat() * halfWidth, 0, 0)
    }

    private static func rectangleOffset(
        halfWidth: Float,
        halfHeight: Float,
        mode: CAEmitterLayerEmitterMode,
        random: inout EmitterRandomSource
    ) -> SIMD3<Float> {
        if mode == .points {
            return SIMD3(randomSign(&random) * halfWidth, randomSign(&random) * halfHeight, 0)
        }
        if mode == .outline {
            let width = halfWidth * 2
            let height = halfHeight * 2
            let perimeter = 2 * (width + height)
            guard perimeter > 0 else { return .zero }
            var distance = random.unitFloat() * perimeter
            if distance < width {
                return SIMD3(-halfWidth + distance, -halfHeight, 0)
            }
            distance -= width
            if distance < height {
                return SIMD3(halfWidth, -halfHeight + distance, 0)
            }
            distance -= height
            if distance < width {
                return SIMD3(halfWidth - distance, halfHeight, 0)
            }
            distance -= width
            return SIMD3(-halfWidth, halfHeight - distance, 0)
        }
        return SIMD3(
            random.signedFloat() * halfWidth,
            random.signedFloat() * halfHeight,
            0
        )
    }

    private static func cuboidOffset(
        halfWidth: Float,
        halfHeight: Float,
        halfDepth: Float,
        mode: CAEmitterLayerEmitterMode,
        random: inout EmitterRandomSource
    ) -> SIMD3<Float> {
        if mode == .points {
            return SIMD3(
                randomSign(&random) * halfWidth,
                randomSign(&random) * halfHeight,
                randomSign(&random) * halfDepth
            )
        }
        if mode == .outline {
            let width = halfWidth * 2
            let height = halfHeight * 2
            let depth = halfDepth * 2
            let totalAxisLength = width + height + depth
            guard totalAxisLength > 0 else { return .zero }
            let selection = random.unitFloat() * totalAxisLength
            if selection < width {
                return SIMD3(
                    random.signedFloat() * halfWidth,
                    randomSign(&random) * halfHeight,
                    randomSign(&random) * halfDepth
                )
            }
            if selection < width + height {
                return SIMD3(
                    randomSign(&random) * halfWidth,
                    random.signedFloat() * halfHeight,
                    randomSign(&random) * halfDepth
                )
            }
            return SIMD3(
                randomSign(&random) * halfWidth,
                randomSign(&random) * halfHeight,
                random.signedFloat() * halfDepth
            )
        }
        if mode == .surface {
            let xFaceArea = halfHeight * halfDepth
            let yFaceArea = halfWidth * halfDepth
            let zFaceArea = halfWidth * halfHeight
            let totalFaceArea = xFaceArea + yFaceArea + zFaceArea
            guard totalFaceArea > 0 else {
                return rectangleOffset(
                    halfWidth: halfWidth,
                    halfHeight: halfHeight,
                    mode: .surface,
                    random: &random
                )
            }
            let selection = random.unitFloat() * totalFaceArea
            if selection < xFaceArea {
                return SIMD3(
                    randomSign(&random) * halfWidth,
                    random.signedFloat() * halfHeight,
                    random.signedFloat() * halfDepth
                )
            }
            if selection < xFaceArea + yFaceArea {
                return SIMD3(
                    random.signedFloat() * halfWidth,
                    randomSign(&random) * halfHeight,
                    random.signedFloat() * halfDepth
                )
            }
            return SIMD3(
                random.signedFloat() * halfWidth,
                random.signedFloat() * halfHeight,
                randomSign(&random) * halfDepth
            )
        }
        return SIMD3(
            random.signedFloat() * halfWidth,
            random.signedFloat() * halfHeight,
            random.signedFloat() * halfDepth
        )
    }

    private static func circleOffset(
        radius: Float,
        mode: CAEmitterLayerEmitterMode,
        random: inout EmitterRandomSource
    ) -> SIMD3<Float> {
        let angle = random.unitFloat() * 2 * Float.pi
        let sampledRadius = mode == .surface || mode == .volume
            ? sqrt(random.unitFloat()) * radius
            : radius
        return SIMD3(sampledRadius * cos(angle), sampledRadius * sin(angle), 0)
    }

    private static func sphereOffset(
        radius: Float,
        mode: CAEmitterLayerEmitterMode,
        random: inout EmitterRandomSource
    ) -> SIMD3<Float> {
        let z = 1 - 2 * random.unitFloat()
        let azimuth = random.unitFloat() * 2 * Float.pi
        let radial = sqrt(max(0, 1 - z * z))
        let sampledRadius = mode == .volume
            ? cbrt(random.unitFloat()) * radius
            : radius
        return SIMD3(
            sampledRadius * radial * cos(azimuth),
            sampledRadius * radial * sin(azimuth),
            sampledRadius * z
        )
    }

    private static func randomSign(_ random: inout EmitterRandomSource) -> Float {
        random.unitFloat() < 0.5 ? -1 : 1
    }

    private static func cross(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x
        )
    }

    private static func normalized(_ value: SIMD3<Float>) -> SIMD3<Float> {
        let length = sqrt(value.x * value.x + value.y * value.y + value.z * value.z)
        guard length > 0 else { return SIMD3(0, 0, 1) }
        return value / length
    }
}
