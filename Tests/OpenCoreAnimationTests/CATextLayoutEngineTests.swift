import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CATextLayer truncation layout")
struct CATextLayoutEngineTests {
    private func monospacedWidth(_ text: String) -> CGFloat {
        CGFloat(text.count)
    }

    @Test("Truncation modes retain the expected side of a line")
    func truncationModesRetainExpectedText() {
        #expect(CATextLayoutEngine.truncatedText(
            "abcdefghij",
            mode: .start,
            maximumWidth: 6,
            measure: monospacedWidth
        ) == "…fghij")
        #expect(CATextLayoutEngine.truncatedText(
            "abcdefghij",
            mode: .middle,
            maximumWidth: 6,
            measure: monospacedWidth
        ) == "abc…ij")
        #expect(CATextLayoutEngine.truncatedText(
            "abcdefghij",
            mode: .end,
            maximumWidth: 6,
            measure: monospacedWidth
        ) == "abcde…")
    }

    @Test("None and fitting text remain unchanged")
    func unchangedText() {
        #expect(CATextLayoutEngine.truncatedText(
            "abcdefghij",
            mode: .none,
            maximumWidth: 3,
            measure: monospacedWidth
        ) == "abcdefghij")
        #expect(CATextLayoutEngine.truncatedText(
            "abc",
            mode: .end,
            maximumWidth: 3,
            measure: monospacedWidth
        ) == "abc")
        #expect(CATextLayoutEngine.truncatedText(
            "abcdefghij",
            mode: CATextLayerTruncationMode(rawValue: "unknown"),
            maximumWidth: 3,
            measure: monospacedWidth
        ) == "abcdefghij")
    }

    @Test("Truncation preserves extended grapheme clusters")
    func preservesGraphemeClusters() {
        let text = "A👨‍👩‍👧‍👦BCDE"
        let result = CATextLayoutEngine.truncatedText(
            text,
            mode: .end,
            maximumWidth: 4,
            measure: monospacedWidth
        )

        #expect(result == "A👨‍👩‍👧‍👦B…")
    }

    @Test("A bound narrower than the ellipsis produces no fake fit")
    func boundNarrowerThanEllipsis() {
        #expect(CATextLayoutEngine.truncatedText(
            "abc",
            mode: .end,
            maximumWidth: 0,
            measure: monospacedWidth
        ).isEmpty)
    }

    @Test("Wrapping preserves whitespace, CJK adjacency, and paragraph breaks")
    func wrappingPreservesBreakSemantics() {
        let latin = CATextLayoutEngine.wrappedLines(
            "hello world",
            maximumWidth: 7,
            measure: monospacedWidth
        )
        #expect(latin == [
            CATextLayoutLine(text: "hello", separatorAfter: " ", isParagraphFinal: false),
            CATextLayoutLine(text: "world", separatorAfter: "", isParagraphFinal: true),
        ])

        let cjk = CATextLayoutEngine.wrappedLines(
            "これはテスト",
            maximumWidth: 3,
            measure: monospacedWidth
        )
        #expect(cjk.map(\.text) == ["これは", "テスト"])
        #expect(cjk[0].separatorAfter.isEmpty)

        let paragraphs = CATextLayoutEngine.wrappedLines(
            "first\n\nsecond",
            maximumWidth: 20,
            measure: monospacedWidth
        )
        #expect(paragraphs.map(\.text) == ["first", "", "second"])
        #expect(paragraphs.allSatisfy { $0.isParagraphFinal })

        let windowsLines = CATextLayoutEngine.wrappedLines(
            "first\r\nsecond\r",
            maximumWidth: 20,
            measure: monospacedWidth
        )
        #expect(windowsLines.map(\.text) == ["first", "second", ""])
        #expect(windowsLines.allSatisfy { $0.isParagraphFinal })
    }

    @Test("Joined overflow text does not invent CJK spaces")
    func joinedOverflowPreservesSeparators() {
        let lines = CATextLayoutEngine.wrappedLines(
            "これはテスト",
            maximumWidth: 3,
            measure: monospacedWidth
        )
        #expect(CATextLayoutEngine.joinedText(lines[...]) == "これはテスト")
    }

    @Test("Justification uses words for Latin and characters for CJK")
    func justificationSegments() {
        #expect(CATextLayoutEngine.justificationSegments(for: "one two three") == [
            "one", "two", "three",
        ])
        #expect(CATextLayoutEngine.justificationSegments(for: "日本語") == ["日", "本", "語"])
        #expect(CATextLayoutEngine.justificationSegments(for: "Swift日本語") == [
            "Swift", "日", "本", "語",
        ])
        #expect(CATextLayoutEngine.justificationSegments(for: "unbroken") == ["unbroken"])
    }

    @Test("Wrapping preserves graphemes when an oversized token is split")
    func oversizedTokensPreserveGraphemes() {
        let lines = CATextLayoutEngine.wrappedLines(
            "A👨‍👩‍👧‍👦BC",
            maximumWidth: 2,
            measure: monospacedWidth
        )
        #expect(lines.map(\.text) == ["A👨‍👩‍👧‍👦", "BC"])
    }

    @Test("Width wrapping can be disabled while paragraph breaks remain active")
    func paragraphLayoutWithoutWidthWrapping() {
        let lines = CATextLayoutEngine.wrappedLines(
            "long line\nnext",
            maximumWidth: 2,
            wrapsToWidth: false,
            measure: monospacedWidth
        )
        #expect(lines.map(\.text) == ["long line", "next"])
        #expect(lines.allSatisfy { $0.isParagraphFinal })
    }
}
