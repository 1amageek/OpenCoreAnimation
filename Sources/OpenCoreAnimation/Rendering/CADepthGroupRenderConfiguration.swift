import Foundation

/// Describes an invalid renderer depth-group state transition.
internal enum CADepthGroupStateFailure: Error, Equatable, Sendable {
    case invalidNestingDepth(Int)
    case nestingDepthOverflow
}

/// Validated state transition for entering any renderer depth group.
internal struct CADepthGroupRenderConfiguration: Equatable, Sendable {
    let requiresDepthClear: Bool
    let enteredNestingDepth: Int

    init(currentNestingDepth: Int) throws(CADepthGroupStateFailure) {
        guard currentNestingDepth >= 0 else {
            throw .invalidNestingDepth(currentNestingDepth)
        }
        guard currentNestingDepth < Int.max else {
            throw .nestingDepthOverflow
        }

        requiresDepthClear = currentNestingDepth == 0
        enteredNestingDepth = currentNestingDepth + 1
    }
}
