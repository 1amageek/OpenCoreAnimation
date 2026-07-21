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
}
