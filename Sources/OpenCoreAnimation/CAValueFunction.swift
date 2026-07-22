//
//  CAValueFunction.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation


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
        guard Self.supportedNames.contains(name) else { return nil }
        self.name = name
    }

    private static let supportedNames: Set<CAValueFunctionName> = [
        .rotateX, .rotateY, .rotateZ,
        .scale, .scaleX, .scaleY, .scaleZ,
        .translate, .translateX, .translateY, .translateZ,
    ]

    internal var componentCount: Int {
        name == .scale || name == .translate ? 3 : 1
    }

    internal var neutralComponents: [CGFloat] {
        switch name {
        case .scale:
            return [1, 1, 1]
        case .scaleX, .scaleY, .scaleZ:
            return [1]
        default:
            return Array(repeating: 0, count: componentCount)
        }
    }

    internal func components(from value: Any) -> [CGFloat]? {
        if componentCount == 1 {
            guard let scalar = Self.scalar(from: value) else { return nil }
            return [scalar]
        }

        let components: [CGFloat]?
        if let values = value as? [CGFloat] {
            components = values
        } else if let values = value as? [Double] {
            components = values.map { CGFloat($0) }
        } else if let values = value as? [Float] {
            components = values.map { CGFloat($0) }
        } else if let values = value as? [Int] {
            components = values.map { CGFloat($0) }
        } else if let values = value as? [Any] {
            var converted: [CGFloat] = []
            converted.reserveCapacity(values.count)
            for value in values {
                guard let scalar = Self.scalar(from: value) else { return nil }
                converted.append(scalar)
            }
            components = converted
        } else {
            components = nil
        }

        guard let components, components.count == componentCount else { return nil }
        return components
    }

    internal func resolveInputs(
        from fromValue: Any?,
        to toValue: Any?,
        by byValue: Any?
    ) -> (from: [CGFloat], to: [CGFloat])? {
        guard fromValue != nil || toValue != nil || byValue != nil else { return nil }

        let from = fromValue.flatMap { components(from: $0) }
        let to = toValue.flatMap { components(from: $0) }
        let by = byValue.flatMap { components(from: $0) }
        guard fromValue == nil || from != nil,
              toValue == nil || to != nil,
              byValue == nil || by != nil else {
            return nil
        }

        if let from, let to {
            return (from, to)
        }
        if let from, let by {
            return (from, Self.add(from, by))
        }
        if let to, let by {
            return (Self.subtract(to, by), to)
        }
        if let by {
            return (neutralComponents, Self.add(neutralComponents, by))
        }
        if let from {
            return (from, neutralComponents)
        }
        if let to {
            return (neutralComponents, to)
        }
        return nil
    }

    internal func transform(for value: Any) -> CATransform3D? {
        guard let components = components(from: value) else { return nil }
        return transform(for: components)
    }

    internal func transform(for components: [CGFloat]) -> CATransform3D? {
        guard components.count == componentCount else { return nil }
        switch name {
        case .rotateX:
            return CATransform3DMakeRotation(components[0], 1, 0, 0)
        case .rotateY:
            return CATransform3DMakeRotation(components[0], 0, 1, 0)
        case .rotateZ:
            return CATransform3DMakeRotation(components[0], 0, 0, 1)
        case .scale:
            return CATransform3DMakeScale(components[0], components[1], components[2])
        case .scaleX:
            return CATransform3DMakeScale(components[0], 1, 1)
        case .scaleY:
            return CATransform3DMakeScale(1, components[0], 1)
        case .scaleZ:
            return CATransform3DMakeScale(1, 1, components[0])
        case .translate:
            return CATransform3DMakeTranslation(components[0], components[1], components[2])
        case .translateX:
            return CATransform3DMakeTranslation(components[0], 0, 0)
        case .translateY:
            return CATransform3DMakeTranslation(0, components[0], 0)
        case .translateZ:
            return CATransform3DMakeTranslation(0, 0, components[0])
        default:
            return nil
        }
    }

    internal func interpolatedTransform(from: Any, to: Any, progress: CGFloat) -> CATransform3D? {
        guard let from = components(from: from),
              let to = components(from: to) else {
            return nil
        }
        return interpolatedTransform(fromComponents: from, toComponents: to, progress: progress)
    }

    internal func interpolatedTransform(
        fromComponents from: [CGFloat],
        toComponents to: [CGFloat],
        progress: CGFloat
    ) -> CATransform3D? {
        guard from.count == componentCount, to.count == componentCount else { return nil }
        let components = zip(from, to).map { start, end in
            start + (end - start) * progress
        }
        return transform(for: components)
    }

    internal func distance(from: Any, to: Any) -> CGFloat? {
        guard let from = components(from: from),
              let to = components(from: to) else {
            return nil
        }
        let squaredDistance = zip(from, to).reduce(CGFloat(0)) { partialResult, pair in
            let difference = pair.1 - pair.0
            return partialResult + difference * difference
        }
        return sqrt(squaredDistance)
    }

    private static func scalar(from value: Any) -> CGFloat? {
        if let value = value as? CGFloat { return value }
        if let value = value as? Double { return CGFloat(value) }
        if let value = value as? Float { return CGFloat(value) }
        if let value = value as? Int { return CGFloat(value) }
        return nil
    }

    private static func add(_ lhs: [CGFloat], _ rhs: [CGFloat]) -> [CGFloat] {
        zip(lhs, rhs).map(+)
    }

    private static func subtract(_ lhs: [CGFloat], _ rhs: [CGFloat]) -> [CGFloat] {
        zip(lhs, rhs).map(-)
    }
}
