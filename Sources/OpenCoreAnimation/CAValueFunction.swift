//
//  CAValueFunction.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics


/// An object that provides a flexible method of defining animated transformations.
///
/// Value functions define how a single input value is transformed into a `CATransform3D`.
/// They are typically used with `CAPropertyAnimation` to animate transform properties
/// using a single scalar value instead of specifying the full transform matrix.
open class CAValueFunction {

    /// The name of the value function.
    open private(set) var name: CAValueFunctionName

    /// Creates a new value function with the specified name.
    public init?(name: CAValueFunctionName) {
        self.name = name
    }

    /// The number of arguments this value function expects.
    ///
    /// All value functions take exactly 1 argument:
    /// - `rotateX`, `rotateY`, `rotateZ`: angle in radians
    /// - `scale`, `scaleX`, `scaleY`, `scaleZ`: scale factor
    /// - `translateX`, `translateY`, `translateZ`: translation amount in points
    open var inputCount: Int {
        return 1
    }

    /// Applies this value function to the given input value(s) and returns the resulting transform.
    ///
    /// - Parameter values: The input values for the transformation.
    /// - Returns: A `CATransform3D` representing the transformation.
    open func apply(values: [CGFloat]) -> CATransform3D {
        guard !values.isEmpty else { return CATransform3DIdentity }
        let value = values[0]

        switch name {
        case .rotateX:
            return CATransform3DMakeRotation(value, 1, 0, 0)
        case .rotateY:
            return CATransform3DMakeRotation(value, 0, 1, 0)
        case .rotateZ:
            return CATransform3DMakeRotation(value, 0, 0, 1)
        case .scale:
            return CATransform3DMakeScale(value, value, value)
        case .scaleX:
            return CATransform3DMakeScale(value, 1, 1)
        case .scaleY:
            return CATransform3DMakeScale(1, value, 1)
        case .scaleZ:
            return CATransform3DMakeScale(1, 1, value)
        case .translateX:
            return CATransform3DMakeTranslation(value, 0, 0)
        case .translateY:
            return CATransform3DMakeTranslation(0, value, 0)
        case .translateZ:
            return CATransform3DMakeTranslation(0, 0, value)
        default:
            return CATransform3DIdentity
        }
    }

    /// Applies this value function to interpolate between two transforms.
    ///
    /// - Parameters:
    ///   - fromValue: The starting value.
    ///   - toValue: The ending value.
    ///   - progress: The interpolation progress (0 to 1).
    /// - Returns: A `CATransform3D` representing the interpolated transformation.
    open func interpolate(from fromValue: CGFloat, to toValue: CGFloat, progress: CGFloat) -> CATransform3D {
        let interpolatedValue = fromValue + (toValue - fromValue) * progress
        return apply(values: [interpolatedValue])
    }
}
