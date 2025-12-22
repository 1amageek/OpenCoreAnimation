// CGImageMetadataTag.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework

import Foundation
import OpenCoreGraphics


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

    public func hash(into hasher: inout Hasher) {
        hasher.combine(type)
        // Note: parameters are not included in hash for simplicity
    }

    public static func == (lhs: CAFilter, rhs: CAFilter) -> Bool {
        return lhs.type == rhs.type
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
        return floatValue(forKey: "inputRadius") ?? 0
    }

    /// Gets the brightness amount if this is a brightness filter.
    public var brightnessAmount: CGFloat? {
        guard type == .brightness else { return nil }
        return floatValue(forKey: "inputBrightness") ?? 0
    }

    /// Gets the contrast amount if this is a contrast filter.
    public var contrastAmount: CGFloat? {
        guard type == .contrast else { return nil }
        return floatValue(forKey: "inputContrast") ?? 1
    }

    /// Gets the saturation amount if this is a saturation filter.
    public var saturationAmount: CGFloat? {
        guard type == .saturation else { return nil }
        return floatValue(forKey: "inputSaturation") ?? 1
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

    /// Checks if the layer has any blur filters.
    internal var hasBlurFilter: Bool {
        return activeFilters.contains { $0.type == .gaussianBlur }
    }

    /// Gets the total blur radius from all blur filters.
    internal var totalBlurRadius: CGFloat {
        return activeFilters
            .filter { $0.type == .gaussianBlur }
            .compactMap { $0.blurRadius }
            .reduce(0, +)
    }
}
