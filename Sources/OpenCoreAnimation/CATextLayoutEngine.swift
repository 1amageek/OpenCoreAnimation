//
//  CATextLayoutEngine.swift
//  OpenCoreAnimation
//

import Foundation

/// Width-driven text truncation shared by the Web renderer and native tests.
internal enum CATextLayoutEngine {
    private static let ellipsis = "…"

    static func truncatedText(
        _ text: String,
        mode: CATextLayerTruncationMode,
        maximumWidth: CGFloat,
        measure: (String) -> CGFloat
    ) -> String {
        guard mode == .start || mode == .middle || mode == .end else { return text }
        guard measure(text) > maximumWidth else { return text }
        guard maximumWidth > 0, measure(ellipsis) <= maximumWidth else { return "" }

        let characters = Array(text)
        guard !characters.isEmpty else { return text }

        if mode == .start {
            let count = maximumRetainedCharacterCount(
                upperBound: characters.count,
                fits: { count in
                    measure(ellipsis + String(characters.suffix(count))) <= maximumWidth
                }
            )
            return ellipsis + String(characters.suffix(count))
        }

        if mode == .middle {
            let count = maximumRetainedCharacterCount(
                upperBound: characters.count,
                fits: { count in
                    let prefixCount = (count + 1) / 2
                    let suffixCount = count / 2
                    return measure(
                        String(characters.prefix(prefixCount))
                            + ellipsis
                            + String(characters.suffix(suffixCount))
                    ) <= maximumWidth
                }
            )
            let prefixCount = (count + 1) / 2
            let suffixCount = count / 2
            return String(characters.prefix(prefixCount))
                + ellipsis
                + String(characters.suffix(suffixCount))
        }

        let count = maximumRetainedCharacterCount(
            upperBound: characters.count,
            fits: { count in
                measure(String(characters.prefix(count)) + ellipsis) <= maximumWidth
            }
        )
        return String(characters.prefix(count)) + ellipsis
    }

    private static func maximumRetainedCharacterCount(
        upperBound: Int,
        fits: (Int) -> Bool
    ) -> Int {
        var lower = 0
        var upper = upperBound
        while lower < upper {
            let candidate = lower + (upper - lower + 1) / 2
            if fits(candidate) {
                lower = candidate
            } else {
                upper = candidate - 1
            }
        }
        return lower
    }
}
