// CAFilter.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework

import Foundation

internal enum CAFilterOperation: Equatable {
    case gaussianBlur(radius: CGFloat)
    case brightness(amount: CGFloat)
    case contrast(amount: CGFloat)
    case saturation(amount: CGFloat)
    case colorInvert
}

internal enum CAFilterConfigurationError: Error, Equatable {
    case unexpectedParameter(String)
    case invalidParameterType(String)
    case nonFiniteParameter(String)
    case parameterOutOfRange(String, minimum: CGFloat, maximum: CGFloat?)
}

internal enum CAFilterExecutionPlan: Equatable {
    case renderer(CAFilterOperation)
    case coreImage(name: String, parameters: [String: CGFloat])
}


/// A filter effect that can be applied to layer content.
///
/// `CAFilter` provides a cross-platform abstraction for common visual effects
/// that can be applied to layer content. On Apple platforms, you would typically
/// use `CIFilter` objects, but for WASM/WebGPU environments, `CAFilter` provides
/// similar functionality using GPU shaders.
///
/// ## Supported Filter Types
///
/// - `gaussianBlur`: Applies a Gaussian blur effect
/// - `brightness`: Adjusts the brightness of the content
/// - `contrast`: Adjusts the contrast of the content
/// - `saturation`: Adjusts the color saturation
/// - `colorInvert`: Inverts the colors
///
/// ## Usage Example
///
/// ```swift
/// let layer = CALayer()
/// layer.filters = [
///     CAFilter(type: .gaussianBlur, parameters: ["inputRadius": 5.0]),
///     CAFilter(type: .brightness, parameters: ["inputBrightness": 0.2])
/// ]
/// ```
public struct CAFilter: Hashable {

    private enum ParameterValue: Hashable {
        case bool(Bool)
        case int(Int)
        case double(Double)
        case string(String)
        case unsupported(String)
    }

    /// The type of filter effect.
    public enum FilterType: String, Hashable, Sendable {
        /// Gaussian blur effect. Parameter: inputRadius (CGFloat)
        case gaussianBlur = "CIGaussianBlur"

        /// Brightness adjustment. Parameter: inputBrightness (CGFloat, -1 to 1)
        case brightness = "CIColorControls.brightness"

        /// Contrast adjustment. Parameter: inputContrast (CGFloat, 0 to 4)
        case contrast = "CIColorControls.contrast"

        /// Saturation adjustment. Parameter: inputSaturation (CGFloat, 0 to 2)
        case saturation = "CIColorControls.saturation"

        /// Color inversion.
        case colorInvert = "CIColorInvert"

        /// Sepia tone effect. Parameter: inputIntensity (CGFloat, 0 to 1)
        case sepiaTone = "CISepiaTone"

        /// Vignette effect. Parameters: inputRadius, inputIntensity (CGFloat)
        case vignette = "CIVignette"
    }

    /// The type of this filter.
    public let type: FilterType

    /// The name of the filter (for compatibility with CIFilter naming).
    public var name: String {
        return type.rawValue
    }

    /// Filter parameters as key-value pairs.
    ///
    /// Common parameters include:
    /// - "inputRadius": Blur radius (CGFloat)
    /// - "inputBrightness": Brightness adjustment (-1 to 1)
    /// - "inputContrast": Contrast adjustment (0 to 4)
    /// - "inputSaturation": Saturation adjustment (0 to 2)
    /// - "inputIntensity": Effect intensity (0 to 1)
    public var parameters: [String: Any]

    /// Creates a new filter with the specified type and parameters.
    ///
    /// - Parameters:
    ///   - type: The type of filter effect.
    ///   - parameters: Key-value pairs of filter parameters.
    public init(type: FilterType, parameters: [String: Any] = [:]) {
        self.type = type
        self.parameters = parameters
    }

    /// Creates a Gaussian blur filter.
    ///
    /// - Parameter radius: The blur radius in points.
    /// - Returns: A configured blur filter.
    public static func blur(radius: CGFloat) -> CAFilter {
        return CAFilter(type: .gaussianBlur, parameters: ["inputRadius": radius])
    }

    /// Creates a brightness adjustment filter.
    ///
    /// - Parameter amount: The brightness adjustment (-1 to 1). 0 is no change.
    /// - Returns: A configured brightness filter.
    public static func brightness(_ amount: CGFloat) -> CAFilter {
        return CAFilter(type: .brightness, parameters: ["inputBrightness": amount])
    }

    /// Creates a contrast adjustment filter.
    ///
    /// - Parameter amount: The contrast multiplier (0 to 4). 1 is no change.
    /// - Returns: A configured contrast filter.
    public static func contrast(_ amount: CGFloat) -> CAFilter {
        return CAFilter(type: .contrast, parameters: ["inputContrast": amount])
    }

    /// Creates a saturation adjustment filter.
    ///
    /// - Parameter amount: The saturation multiplier (0 to 2). 1 is no change.
    /// - Returns: A configured saturation filter.
    public static func saturation(_ amount: CGFloat) -> CAFilter {
        return CAFilter(type: .saturation, parameters: ["inputSaturation": amount])
    }

    /// Creates a color inversion filter.
    ///
    /// - Returns: A color inversion filter.
    public static func colorInvert() -> CAFilter {
        return CAFilter(type: .colorInvert)
    }

    // MARK: - Hashable

    private static func normalizedParameterValue(_ value: Any) -> ParameterValue {
        switch value {
        case let value as Bool:
            return .bool(value)
        case let value as Int:
            return .int(value)
        case let value as CGFloat:
            return .double(Double(value))
        case let value as Double:
            return .double(value)
        case let value as Float:
            return .double(Double(value))
        case let value as String:
            return .string(value)
        default:
            return .unsupported(String(describing: value))
        }
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(type)
        for key in parameters.keys.sorted() {
            hasher.combine(key)
            if let value = parameters[key] {
                hasher.combine(Self.normalizedParameterValue(value))
            }
        }
    }

    public static func == (lhs: CAFilter, rhs: CAFilter) -> Bool {
        guard lhs.type == rhs.type, lhs.parameters.count == rhs.parameters.count else {
            return false
        }

        for key in lhs.parameters.keys {
            guard let lhsValue = lhs.parameters[key], let rhsValue = rhs.parameters[key] else {
                return false
            }
            guard normalizedParameterValue(lhsValue) == normalizedParameterValue(rhsValue) else {
                return false
            }
        }

        return true
    }

    // MARK: - Parameter Access

    /// Gets a CGFloat parameter value.
    public func floatValue(forKey key: String) -> CGFloat? {
        if let value = parameters[key] as? CGFloat {
            return value
        } else if let value = parameters[key] as? Double {
            return CGFloat(value)
        } else if let value = parameters[key] as? Float {
            return CGFloat(value)
        } else if let value = parameters[key] as? Int {
            return CGFloat(value)
        }
        return nil
    }

    /// Gets the blur radius if this is a blur filter.
    public var blurRadius: CGFloat? {
        guard type == .gaussianBlur else { return nil }
        return configuredNumber(forKey: "inputRadius", default: 0)
    }

    /// Gets the brightness amount if this is a brightness filter.
    public var brightnessAmount: CGFloat? {
        guard type == .brightness else { return nil }
        return configuredNumber(forKey: "inputBrightness", default: 0)
    }

    /// Gets the contrast amount if this is a contrast filter.
    public var contrastAmount: CGFloat? {
        guard type == .contrast else { return nil }
        return configuredNumber(forKey: "inputContrast", default: 1)
    }

    /// Gets the saturation amount if this is a saturation filter.
    public var saturationAmount: CGFloat? {
        guard type == .saturation else { return nil }
        return configuredNumber(forKey: "inputSaturation", default: 1)
    }

    internal func executionPlan() throws -> CAFilterExecutionPlan {
        switch type {
        case .gaussianBlur:
            return .renderer(.gaussianBlur(radius: try validatedNumber(
                forKey: "inputRadius",
                default: 0,
                minimum: 0
            )))
        case .brightness:
            return .renderer(.brightness(amount: try validatedNumber(
                forKey: "inputBrightness",
                default: 0,
                minimum: -1,
                maximum: 1
            )))
        case .contrast:
            return .renderer(.contrast(amount: try validatedNumber(
                forKey: "inputContrast",
                default: 1,
                minimum: 0,
                maximum: 4
            )))
        case .saturation:
            return .renderer(.saturation(amount: try validatedNumber(
                forKey: "inputSaturation",
                default: 1,
                minimum: 0,
                maximum: 2
            )))
        case .colorInvert:
            try rejectUnexpectedParameters(allowedKeys: [])
            return .renderer(.colorInvert)
        case .sepiaTone:
            return .coreImage(name: type.rawValue, parameters: [
                "inputIntensity": try validatedNumber(
                    forKey: "inputIntensity",
                    default: 1,
                    minimum: 0,
                    maximum: 1
                )
            ])
        case .vignette:
            try rejectUnexpectedParameters(allowedKeys: ["inputRadius", "inputIntensity"])
            return .coreImage(name: type.rawValue, parameters: [
                "inputRadius": try validatedNumber(
                    forKey: "inputRadius",
                    default: 1,
                    minimum: 0,
                    validateKeys: false
                ),
                "inputIntensity": try validatedNumber(
                    forKey: "inputIntensity",
                    default: 0,
                    minimum: 0,
                    maximum: 1,
                    validateKeys: false
                )
            ])
        }
    }

    internal var operation: CAFilterOperation? {
        do {
            guard case let .renderer(operation) = try executionPlan() else {
                return nil
            }
            return operation
        } catch {
            return nil
        }
    }

    private func validatedNumber(
        forKey key: String,
        default defaultValue: CGFloat,
        minimum: CGFloat,
        maximum: CGFloat? = nil,
        validateKeys: Bool = true
    ) throws -> CGFloat {
        if validateKeys {
            try rejectUnexpectedParameters(allowedKeys: [key])
        }
        guard parameters[key] != nil else {
            return defaultValue
        }
        guard let value = floatValue(forKey: key) else {
            throw CAFilterConfigurationError.invalidParameterType(key)
        }
        guard value.isFinite else {
            throw CAFilterConfigurationError.nonFiniteParameter(key)
        }
        guard value >= minimum, maximum.map({ value <= $0 }) ?? true else {
            throw CAFilterConfigurationError.parameterOutOfRange(
                key,
                minimum: minimum,
                maximum: maximum
            )
        }
        return value
    }

    private func configuredNumber(forKey key: String, default defaultValue: CGFloat) -> CGFloat? {
        guard parameters[key] != nil else {
            return defaultValue
        }
        guard let value = floatValue(forKey: key), value.isFinite else {
            return nil
        }
        return value
    }

    private func rejectUnexpectedParameters(allowedKeys: Set<String>) throws {
        if let unexpected = parameters.keys
            .filter({ !allowedKeys.contains($0) })
            .sorted()
            .first {
            throw CAFilterConfigurationError.unexpectedParameter(unexpected)
        }
    }
}

// MARK: - CALayer Filter Extensions

extension CALayer {

    /// Extracts CAFilter objects from the filters array.
    ///
    /// This helper method extracts any CAFilter instances from the filters array,
    /// which may contain mixed types for compatibility with CIFilter on Apple platforms.
    internal var activeFilters: [CAFilter] {
        guard let filters = filters else { return [] }
        return filters.compactMap { $0 as? CAFilter }
    }

    /// Returns supported GPU filter operations in the order they should be applied.
    internal var supportedFilterOperations: [CAFilterOperation] {
        activeFilters.compactMap(\.operation)
    }

    /// Checks if the layer has any blur filters.
    internal var hasBlurFilter: Bool {
        return supportedFilterOperations.contains {
            if case .gaussianBlur = $0 {
                return true
            }
            return false
        }
    }

    /// Gets the total blur radius from all blur filters.
    internal var totalBlurRadius: CGFloat {
        supportedFilterOperations.reduce(0) { partialResult, operation in
            guard case let .gaussianBlur(radius) = operation else {
                return partialResult
            }
            return partialResult + radius
        }
    }
}
