/// The immutable texture-sampling contract captured for one layer contents image.
internal enum CAContentsSampling: CaseIterable, Equatable, Hashable, Sendable {
    case nearestNearest
    case nearestLinear
    case nearestTrilinear
    case linearNearest
    case linearLinear
    case linearTrilinear

    internal init?(
        magnificationFilter: CALayerContentsFilter,
        minificationFilter: CALayerContentsFilter
    ) {
        let magnificationIsNearest: Bool
        switch magnificationFilter {
        case .nearest:
            magnificationIsNearest = true
        case .linear, .trilinear:
            magnificationIsNearest = false
        default:
            return nil
        }

        switch (magnificationIsNearest, minificationFilter) {
        case (true, .nearest):
            self = .nearestNearest
        case (true, .linear):
            self = .nearestLinear
        case (true, .trilinear):
            self = .nearestTrilinear
        case (false, .nearest):
            self = .linearNearest
        case (false, .linear):
            self = .linearLinear
        case (false, .trilinear):
            self = .linearTrilinear
        default:
            return nil
        }
    }

    internal init?(
        magnificationFilter: String,
        minificationFilter: String
    ) {
        self.init(
            magnificationFilter: CALayerContentsFilter(
                rawValue: magnificationFilter
            ),
            minificationFilter: CALayerContentsFilter(
                rawValue: minificationFilter
            )
        )
    }

    internal var usesMipmaps: Bool {
        switch self {
        case .nearestTrilinear, .linearTrilinear:
            return true
        default:
            return false
        }
    }
}
