import Foundation

/// Validated state transition for entering a transform-layer depth group.
internal struct CATransformDepthRenderConfiguration: Equatable {
    let requiresDepthClear: Bool
    let enteredNestingDepth: Int

    init(currentNestingDepth: Int) throws(CATransformDepthRenderFailure) {
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
