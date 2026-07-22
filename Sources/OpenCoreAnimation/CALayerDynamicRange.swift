extension CALayer {
    /// Options that control the dynamic range used to render layer contents.
    public struct DynamicRange: Hashable, Equatable, RawRepresentable, Sendable {
        public let rawValue: String

        public init(rawValue: String) {
            self.rawValue = rawValue
        }

        /// Lets the output system choose the dynamic range.
        public static let automatic = DynamicRange(rawValue: "automatic")

        /// Restricts content to standard dynamic range.
        public static let standard = DynamicRange(rawValue: "standard")

        /// Uses extended range while balancing it with surrounding content.
        public static let constrainedHigh = DynamicRange(rawValue: "constrainedHigh")

        /// Uses the highest dynamic range available to the output.
        public static let high = DynamicRange(rawValue: "high")
    }
}
