import Foundation
#if arch(wasm32)
import OpenCoreImage
#endif

internal enum CARenderSnapshotFilterParameter: Equatable, Sendable {
    case boolean(Bool)
    case integer(Int)
    case scalar(Double)
    case vector([Double])
    case color(SIMD4<Double>)
    case point(SIMD2<Double>)
    case size(SIMD2<Double>)
    case rectangle(SIMD4<Double>)
    case affineTransform(
        a: Double,
        b: Double,
        c: Double,
        d: Double,
        tx: Double,
        ty: Double
    )

    internal static func capture(
        _ value: Any,
        filterName: String,
        key: String
    ) throws(CARenderSnapshotFilterError) -> Self {
        let parameter: Self
        if let values = value as? [Float] {
            parameter = .vector(values.map(Double.init))
        } else if let values = value as? [Double] {
            parameter = .vector(values)
        } else if let values = value as? [CGFloat] {
            parameter = .vector(values.map(Double.init))
        } else {
            switch value {
            case let value as Bool:
                parameter = .boolean(value)
            case let value as Int:
                parameter = .integer(value)
            case let value as Float:
                parameter = .scalar(Double(value))
            case let value as CGFloat:
                parameter = .scalar(Double(value))
            case let value as Double:
                parameter = .scalar(value)
            case let value as CGPoint:
                parameter = .point(
                    SIMD2(Double(value.x), Double(value.y))
                )
            case let value as CGSize:
                parameter = .size(
                    SIMD2(Double(value.width), Double(value.height))
                )
            case let value as CGRect:
                parameter = .rectangle(
                    SIMD4(
                        Double(value.origin.x),
                        Double(value.origin.y),
                        Double(value.size.width),
                        Double(value.size.height)
                    )
                )
            case let value as CGAffineTransform:
                parameter = .affineTransform(
                    a: Double(value.a),
                    b: Double(value.b),
                    c: Double(value.c),
                    d: Double(value.d),
                    tx: Double(value.tx),
                    ty: Double(value.ty)
                )
            #if arch(wasm32)
            case let value as CIVector:
                parameter = .vector(
                    (0..<value.count).map { Double(value.value(at: $0)) }
                )
            case let value as CIColor:
                parameter = .color(
                    SIMD4(
                        Double(value.red),
                        Double(value.green),
                        Double(value.blue),
                        Double(value.alpha)
                    )
                )
            #endif
            default:
                throw .unsupportedCoreImageParameter(
                    filter: filterName,
                    key: key,
                    valueType: String(reflecting: type(of: value))
                )
            }
        }
        guard parameter.hasFiniteFloatingPointValues else {
            throw .nonFiniteCoreImageParameter(
                filter: filterName,
                key: key
            )
        }
        return parameter
    }

    #if arch(wasm32)
    internal var materializedValue: Any {
        switch self {
        case let .boolean(value):
            return value
        case let .integer(value):
            return value
        case let .scalar(value):
            return CGFloat(value)
        case let .vector(values):
            var materialized = values.map(CGFloat.init)
            return CIVector(
                values: &materialized,
                count: materialized.count
            )
        case let .color(value):
            return CIColor(
                red: CGFloat(value.x),
                green: CGFloat(value.y),
                blue: CGFloat(value.z),
                alpha: CGFloat(value.w)
            )
        case let .point(value):
            return CGPoint(
                x: CGFloat(value.x),
                y: CGFloat(value.y)
            )
        case let .size(value):
            return CGSize(
                width: CGFloat(value.x),
                height: CGFloat(value.y)
            )
        case let .rectangle(value):
            return CGRect(
                x: CGFloat(value.x),
                y: CGFloat(value.y),
                width: CGFloat(value.z),
                height: CGFloat(value.w)
            )
        case let .affineTransform(a, b, c, d, tx, ty):
            return CGAffineTransform(
                a: CGFloat(a),
                b: CGFloat(b),
                c: CGFloat(c),
                d: CGFloat(d),
                tx: CGFloat(tx),
                ty: CGFloat(ty)
            )
        }
    }
    #endif

    private var hasFiniteFloatingPointValues: Bool {
        switch self {
        case .boolean, .integer:
            return true
        case let .scalar(value):
            return value.isFinite
        case let .vector(values):
            return values.allSatisfy(\.isFinite)
        case let .affineTransform(a, b, c, d, tx, ty):
            return a.isFinite
                && b.isFinite
                && c.isFinite
                && d.isFinite
                && tx.isFinite
                && ty.isFinite
        case let .color(value), let .rectangle(value):
            return value.x.isFinite
                && value.y.isFinite
                && value.z.isFinite
                && value.w.isFinite
        case let .point(value), let .size(value):
            return value.x.isFinite && value.y.isFinite
        }
    }
}

internal enum CARenderSnapshotFilterStage: Equatable, Sendable {
    case renderer(CAFilterOperation)
    case coreImage(
        name: String,
        parameters: [String: CARenderSnapshotFilterParameter]
    )

    internal static func capture(
        _ values: [Any]
    ) throws(CARenderSnapshotFilterError) -> [Self] {
        var stages: [Self] = []
        stages.reserveCapacity(values.count)

        for value in values {
            if let filter = value as? CAFilter {
                let executionPlan: CAFilterExecutionPlan
                do {
                    executionPlan = try filter.executionPlan()
                } catch {
                    throw .invalidConfiguration(error)
                }
                switch executionPlan {
                case let .renderer(operation):
                    stages.append(.renderer(operation))
                case let .coreImage(name, parameters):
                    stages.append(.coreImage(
                        name: name,
                        parameters: parameters.mapValues {
                            .scalar(Double($0))
                        }
                    ))
                }
                continue
            }

            #if arch(wasm32)
            guard let filter = value as? CIFilter else {
                throw .unsupportedFilterValue(
                    String(reflecting: type(of: value))
                )
            }
            guard filter.isEnabled else { continue }
            guard !filter.name.isEmpty else {
                throw .invalidCoreImageFilterName
            }

            var parameters: [
                String: CARenderSnapshotFilterParameter
            ] = [:]
            for key in filter.inputKeys.sorted()
            where key != kCIInputImageKey
                && key != kCIInputBackgroundImageKey {
                guard let value = filter.value(forKey: key) else {
                    continue
                }
                parameters[key] = try .capture(
                    value,
                    filterName: filter.name,
                    key: key
                )
            }
            stages.append(.coreImage(
                name: filter.name,
                parameters: parameters
            ))
            #else
            throw .unsupportedFilterValue(
                String(reflecting: type(of: value))
            )
            #endif
        }
        return stages
    }
}

internal struct CARenderSnapshotCompositingFilter: Equatable, Sendable {
    internal let name: String
    internal let parameters: [
        String: CARenderSnapshotFilterParameter
    ]
    internal let isEnabled: Bool

    internal static func capture(
        _ value: Any?
    ) throws(CARenderSnapshotFilterError) -> Self? {
        guard let value else { return nil }
        #if arch(wasm32)
        guard let filter = value as? CIFilter else {
            throw .unsupportedFilterValue(
                String(reflecting: type(of: value))
            )
        }
        guard !filter.name.isEmpty else {
            throw .invalidCoreImageFilterName
        }

        var parameters: [
            String: CARenderSnapshotFilterParameter
        ] = [:]
        for key in filter.inputKeys.sorted()
        where key != kCIInputImageKey
            && key != kCIInputBackgroundImageKey {
            guard let parameter = filter.value(forKey: key) else {
                continue
            }
            parameters[key] = try .capture(
                parameter,
                filterName: filter.name,
                key: key
            )
        }
        return Self(
            name: filter.name,
            parameters: parameters,
            isEnabled: filter.isEnabled
        )
        #else
        throw .unsupportedFilterValue(
            String(reflecting: type(of: value))
        )
        #endif
    }
}
