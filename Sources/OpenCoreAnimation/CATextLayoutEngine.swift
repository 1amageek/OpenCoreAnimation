//
//  CATextLayoutEngine.swift
//  OpenCoreAnimation
//

import Foundation

internal struct CATextLayoutLine: Equatable {
    let text: String
    let separatorAfter: String
    let isParagraphFinal: Bool
}

/// Width-driven text truncation shared by the Web renderer and native tests.
internal enum CATextLayoutEngine {
    private static let ellipsis = "…"

    private enum LineBreakToken {
        case text(String, leadingSpace: Bool)
        case paragraphBreak
    }

    static func wrappedLines(
        _ text: String,
        maximumWidth: CGFloat,
        wrapsToWidth: Bool = true,
        measure: (String) -> CGFloat
    ) -> [CATextLayoutLine] {
        let tokens = lineBreakTokens(in: text)
        var lines: [CATextLayoutLine] = []
        var currentLine = ""

        func appendLine(separatorAfter: String, isParagraphFinal: Bool) {
            lines.append(CATextLayoutLine(
                text: currentLine,
                separatorAfter: separatorAfter,
                isParagraphFinal: isParagraphFinal
            ))
            currentLine = ""
        }

        for token in tokens {
            switch token {
            case .paragraphBreak:
                appendLine(separatorAfter: "\n", isParagraphFinal: true)

            case let .text(value, leadingSpace):
                let separator = !currentLine.isEmpty && leadingSpace ? " " : ""
                let candidate = currentLine + separator + value
                if wrapsToWidth, !currentLine.isEmpty, measure(candidate) > maximumWidth {
                    appendLine(separatorAfter: leadingSpace ? " " : "", isParagraphFinal: false)
                }

                if wrapsToWidth, measure(value) > maximumWidth {
                    let chunks = oversizedTokenChunks(
                        value,
                        maximumWidth: maximumWidth,
                        measure: measure
                    )
                    for chunk in chunks.dropLast() {
                        currentLine = chunk
                        appendLine(separatorAfter: "", isParagraphFinal: false)
                    }
                    currentLine = chunks.last ?? ""
                } else {
                    currentLine += currentLine.isEmpty ? value : separator + value
                }
            }
        }

        if !currentLine.isEmpty || lines.isEmpty || endsInParagraphBreak(text) {
            appendLine(separatorAfter: "", isParagraphFinal: true)
        } else if let last = lines.last, !last.isParagraphFinal {
            lines[lines.count - 1] = CATextLayoutLine(
                text: last.text,
                separatorAfter: last.separatorAfter,
                isParagraphFinal: true
            )
        }
        return lines
    }

    static func joinedText(_ lines: ArraySlice<CATextLayoutLine>) -> String {
        var result = ""
        for line in lines {
            result += line.text
            if line.separatorAfter == "\n" {
                result += " "
            } else {
                result += line.separatorAfter
            }
        }
        return result
    }

    static func justificationSegments(for text: String) -> [String] {
        var segments: [String] = []
        for token in lineBreakTokens(in: text) {
            if case let .text(value, _) = token {
                segments.append(value)
            }
        }
        return segments
    }

    static func containsParagraphBreak(_ text: String) -> Bool {
        text.unicodeScalars.contains(where: isParagraphBreak)
    }

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

    private static func lineBreakTokens(in text: String) -> [LineBreakToken] {
        var tokens: [LineBreakToken] = []
        var current = ""
        var currentLeadingSpace = false
        var pendingSpace = false

        func flush() {
            guard !current.isEmpty else { return }
            tokens.append(.text(current, leadingSpace: currentLeadingSpace))
            current = ""
            currentLeadingSpace = false
        }

        for character in text {
            if character.unicodeScalars.contains(where: isParagraphBreak) {
                flush()
                tokens.append(.paragraphBreak)
                pendingSpace = false
                continue
            }
            if character.isWhitespace {
                flush()
                pendingSpace = true
                continue
            }
            if character.unicodeScalars.contains(where: isCJKLineBreakable) {
                flush()
                tokens.append(.text(String(character), leadingSpace: pendingSpace))
                pendingSpace = false
                continue
            }
            if current.isEmpty {
                currentLeadingSpace = pendingSpace
                pendingSpace = false
            }
            current.append(character)
            if character.unicodeScalars.contains(where: { $0.value == 0x00AD }) {
                flush()
            }
        }
        flush()
        return tokens
    }

    private static func oversizedTokenChunks(
        _ token: String,
        maximumWidth: CGFloat,
        measure: (String) -> CGFloat
    ) -> [String] {
        var chunks: [String] = []
        var buffer = ""
        for character in token {
            let candidate = buffer + String(character)
            if !buffer.isEmpty, measure(candidate) > maximumWidth {
                chunks.append(buffer)
                buffer = String(character)
            } else {
                buffer = candidate
            }
        }
        if !buffer.isEmpty {
            chunks.append(buffer)
        }
        return chunks
    }

    private static func isCJKLineBreakable(_ scalar: Unicode.Scalar) -> Bool {
        let value = scalar.value
        return (0x4E00...0x9FFF).contains(value)
            || (0x3400...0x4DBF).contains(value)
            || (0x3040...0x309F).contains(value)
            || (0x30A0...0x30FF).contains(value)
            || (0xAC00...0xD7AF).contains(value)
    }

    private static func isParagraphBreak(_ scalar: Unicode.Scalar) -> Bool {
        scalar.value == 0x000A || scalar.value == 0x000D
    }

    private static func endsInParagraphBreak(_ text: String) -> Bool {
        text.last?.unicodeScalars.contains(where: isParagraphBreak) == true
    }
}
