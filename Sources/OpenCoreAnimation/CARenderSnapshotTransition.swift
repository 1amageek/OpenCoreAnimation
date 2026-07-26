import Foundation
#if arch(wasm32)
import OpenCoreImage
#endif

/// Immutable transition input captured at the transaction boundary.
internal struct CARenderSnapshotTransition: Equatable, Sendable {
    internal struct Filter: Equatable, Sendable {
        internal let name: String
        internal let parameters: [
            String: CARenderSnapshotFilterParameter
        ]

        internal static func capture(
            _ value: Any?
        ) throws(CATransitionRenderFailure) -> Self? {
            guard let value else {
                return nil
            }
            #if arch(wasm32)
            guard let filter = value as? CIFilter else {
                throw .unsupportedFilterValue(
                    String(reflecting: Swift.type(of: value))
                )
            }
            guard !filter.name.isEmpty else {
                throw .filterSnapshotCaptureFailed(
                    .invalidCoreImageFilterName
                )
            }

            var parameters: [
                String: CARenderSnapshotFilterParameter
            ] = [:]
            for key in filter.inputKeys.sorted()
            where key != kCIInputImageKey
                && key != kCIInputTargetImageKey {
                guard let parameter = filter.value(forKey: key) else {
                    continue
                }
                do {
                    parameters[key] = try .capture(
                        parameter,
                        filterName: filter.name,
                        key: key
                    )
                } catch {
                    throw .filterSnapshotCaptureFailed(error)
                }
            }
            return Self(
                name: filter.name,
                parameters: parameters
            )
            #else
            throw .unsupportedFilterValue(
                String(reflecting: Swift.type(of: value))
            )
            #endif
        }
    }

    internal let resourceIdentity: UInt64
    internal let sourceRootIndex: Int
    internal let type: CATransitionType
    internal let subtype: CATransitionSubtype?
    internal let filter: Filter?
    internal let progress: CFTimeInterval

    internal static func capture(
        _ state: CATransitionRenderState,
        sourceRootIndex: Int
    ) throws(CATransitionRenderFailure) -> Self {
        guard state.progress.isFinite else {
            throw .invalidProgress(state.progress)
        }
        let filter = try Filter.capture(state.filter)
        if filter == nil {
            try validateBuiltIn(
                type: state.type,
                subtype: state.subtype
            )
        }
        return Self(
            resourceIdentity: state.resourceIdentity,
            sourceRootIndex: sourceRootIndex,
            type: state.type,
            subtype: state.subtype,
            filter: filter,
            progress: state.progress
        )
    }

    internal static func validateBuiltIn(
        type: CATransitionType,
        subtype: CATransitionSubtype?
    ) throws(CATransitionRenderFailure) {
        switch type {
        case .fade:
            return
        case .moveIn, .push, .reveal:
            switch subtype {
            case .fromRight, .fromLeft, .fromTop, .fromBottom, nil:
                return
            default:
                throw .unsupportedTransitionSubtype(
                    subtype?.rawValue ?? "nil"
                )
            }
        default:
            throw .unsupportedTransitionType(type.rawValue)
        }
    }
}
