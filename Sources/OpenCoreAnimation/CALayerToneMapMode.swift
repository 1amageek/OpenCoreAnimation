extension CALayer {
    /// Options that control when layer contents are tone mapped.
    public struct ToneMapMode: Hashable, Equatable, RawRepresentable, Sendable {
        public let rawValue: String

        public init(rawValue: String) {
            self.rawValue = rawValue
        }

        /// Lets the output system select the appropriate tone-mapping behavior.
        public static let automatic = ToneMapMode(rawValue: "automatic")

        /// Preserves extended-range values without tone mapping.
        public static let never = ToneMapMode(rawValue: "never")

        /// Tone maps extended-range values whenever the output system supports it.
        public static let ifSupported = ToneMapMode(rawValue: "ifSupported")
    }
}
