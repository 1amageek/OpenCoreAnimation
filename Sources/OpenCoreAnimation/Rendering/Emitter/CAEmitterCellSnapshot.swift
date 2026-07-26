import Foundation

/// Immutable, renderer-owned emitter-cell input captured at transaction commit.
internal struct CAEmitterCellSnapshot: Equatable, Sendable {
    internal struct Image: Equatable, Sendable {
        internal let storage: CGImageTextureStorage
        internal let contentsRect: CGRect
        internal let contentsScale: Float
        internal let sampling: CAContentsSampling
        internal let minificationFilterBias: Float
    }

    internal let identity: UInt64
    internal let image: Image?
    internal let childCells: [Self]
    internal let color: SIMD4<Float>
    internal let birthRate: Float
    internal let lifetime: Float
    internal let lifetimeRange: Float
    internal let redRange: Float
    internal let greenRange: Float
    internal let blueRange: Float
    internal let alphaRange: Float
    internal let redSpeed: Float
    internal let greenSpeed: Float
    internal let blueSpeed: Float
    internal let alphaSpeed: Float
    internal let velocity: Float
    internal let velocityRange: Float
    internal let acceleration: SIMD3<Float>
    internal let scale: Float
    internal let scaleRange: Float
    internal let scaleSpeed: Float
    internal let spin: Float
    internal let spinRange: Float
    internal let emissionLatitude: Float
    internal let emissionLongitude: Float
    internal let emissionRange: Float
    internal let isEnabled: Bool
    internal let beginTime: CFTimeInterval
    internal let timeOffset: CFTimeInterval
    internal let repeatCount: Float
    internal let repeatDuration: CFTimeInterval
    internal let duration: CFTimeInterval
    internal let speed: Float
    internal let autoreverses: Bool

    internal var canEmit: Bool {
        isEnabled && birthRate > 0
    }

    internal static func capture(
        _ cells: [CAEmitterCell]
    ) throws(CARenderSnapshotEmitterError) -> [Self] {
        var activePath: Set<ObjectIdentifier> = []
        var snapshots: [Self] = []
        snapshots.reserveCapacity(cells.count)
        for (index, cell) in cells.enumerated() {
            snapshots.append(
                try capture(
                    cell,
                    path: [index],
                    activePath: &activePath
                )
            )
        }
        return snapshots
    }

    private static func capture(
        _ cell: CAEmitterCell,
        path: [Int],
        activePath: inout Set<ObjectIdentifier>
    ) throws(CARenderSnapshotEmitterError) -> Self {
        let referenceIdentity = ObjectIdentifier(cell)
        guard activePath.insert(referenceIdentity).inserted else {
            throw .cyclicCellHierarchy(path: path)
        }
        defer { activePath.remove(referenceIdentity) }
        let identity = cell.simulationIdentity

        guard cell.beginTime.isFinite,
              cell.timeOffset.isFinite,
              isFiniteOrPositiveInfinity(cell.repeatCount),
              isFiniteOrPositiveInfinity(cell.repeatDuration),
              isFiniteOrPositiveInfinity(cell.duration),
              cell.speed.isFinite else {
            throw .invalidCellTiming(path: path)
        }
        guard cell.birthRate.isFinite,
              cell.lifetime.isFinite,
              cell.lifetimeRange.isFinite else {
            throw .invalidCellBirthRate(path: path)
        }
        let convertedValues = [
            cell.velocity,
            cell.velocityRange,
            cell.xAcceleration,
            cell.yAcceleration,
            cell.zAcceleration,
            cell.scale,
            cell.scaleRange,
            cell.scaleSpeed,
            cell.spin,
            cell.spinRange,
            cell.emissionLatitude,
            cell.emissionLongitude,
            cell.emissionRange,
        ].map(Float.init)
        guard convertedValues.allSatisfy(\.isFinite),
              cell.redRange.isFinite,
              cell.greenRange.isFinite,
              cell.blueRange.isFinite,
              cell.alphaRange.isFinite,
              cell.redSpeed.isFinite,
              cell.greenSpeed.isFinite,
              cell.blueSpeed.isFinite,
              cell.alphaSpeed.isFinite else {
            throw .nonFiniteCellSimulationValue(path: path)
        }

        let color: SIMD4<Float>
        if let sourceColor = cell.color {
            guard let converted = sourceColor.converted(
                to: .deviceRGB,
                intent: .defaultIntent,
                options: nil
            ), let components = converted.components,
               components.count == 4,
               components.allSatisfy(\.isFinite) else {
                throw .invalidCellColor(path: path)
            }
            color = SIMD4(
                Float(components[0]),
                Float(components[1]),
                Float(components[2]),
                Float(components[3])
            )
            guard color.x.isFinite,
                  color.y.isFinite,
                  color.z.isFinite,
                  color.w.isFinite else {
                throw .invalidCellColor(path: path)
            }
        } else {
            color = SIMD4(repeating: 1)
        }

        let image: Image?
        if let contents = cell.contents {
            guard let sourceImage = contents as? CGImage,
                  sourceImage.width > 0,
                  sourceImage.height > 0,
                  cell.contentsScale.isFinite,
                  cell.contentsScale > 0,
                  cell.contentsRect.origin.x.isFinite,
                  cell.contentsRect.origin.y.isFinite,
                  cell.contentsRect.width.isFinite,
                  cell.contentsRect.height.isFinite,
                  cell.contentsRect.width > 0,
                  cell.contentsRect.height > 0,
                  cell.minificationFilterBias.isFinite,
                  let sampling = CAContentsSampling(
                      magnificationFilter: cell.magnificationFilter,
                      minificationFilter: cell.minificationFilter
                  ) else {
                throw .invalidCellContents(path: path)
            }
            let storage: CGImageTextureStorage
            do {
                storage = try CGImageTextureStorageConverter.convert(
                    sourceImage
                )
            } catch {
                throw .cellImageConversion(
                    path: path,
                    reason: error
                )
            }
            image = Image(
                storage: storage,
                contentsRect: cell.contentsRect,
                contentsScale: Float(cell.contentsScale),
                sampling: sampling,
                minificationFilterBias: min(
                    max(cell.minificationFilterBias, -16),
                    15.99
                )
            )
        } else {
            image = nil
        }

        let sourceChildren = cell.emitterCells ?? []
        var children: [Self] = []
        children.reserveCapacity(sourceChildren.count)
        for (childIndex, child) in sourceChildren.enumerated() {
            children.append(
                try capture(
                    child,
                    path: path + [childIndex],
                    activePath: &activePath
                )
            )
        }
        return Self(
            identity: identity,
            image: image,
            childCells: children,
            color: color,
            birthRate: cell.birthRate,
            lifetime: cell.lifetime,
            lifetimeRange: cell.lifetimeRange,
            redRange: cell.redRange,
            greenRange: cell.greenRange,
            blueRange: cell.blueRange,
            alphaRange: cell.alphaRange,
            redSpeed: cell.redSpeed,
            greenSpeed: cell.greenSpeed,
            blueSpeed: cell.blueSpeed,
            alphaSpeed: cell.alphaSpeed,
            velocity: convertedValues[0],
            velocityRange: convertedValues[1],
            acceleration: SIMD3(
                convertedValues[2],
                convertedValues[3],
                convertedValues[4]
            ),
            scale: convertedValues[5],
            scaleRange: convertedValues[6],
            scaleSpeed: convertedValues[7],
            spin: convertedValues[8],
            spinRange: convertedValues[9],
            emissionLatitude: convertedValues[10],
            emissionLongitude: convertedValues[11],
            emissionRange: convertedValues[12],
            isEnabled: cell.isEnabled,
            beginTime: cell.beginTime,
            timeOffset: cell.timeOffset,
            repeatCount: cell.repeatCount,
            repeatDuration: cell.repeatDuration,
            duration: cell.duration,
            speed: cell.speed,
            autoreverses: cell.autoreverses
        )
    }

    private static func isFiniteOrPositiveInfinity<
        Value: BinaryFloatingPoint
    >(_ value: Value) -> Bool {
        value.isFinite || value == .infinity
    }
}
