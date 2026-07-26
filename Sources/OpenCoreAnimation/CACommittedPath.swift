import Foundation

/// An immutable, Sendable command snapshot of a Core Graphics path.
///
/// The snapshot owns point values rather than retaining the mutable reference
/// type supplied by a layer or animation. `CGPath` is reconstructed only after
/// the committed value reaches its owning evaluation context.
internal struct CACommittedPath: Sendable {
    private enum Element: Sendable {
        case move(CGPoint)
        case line(CGPoint)
        case quadratic(control: CGPoint, end: CGPoint)
        case cubic(
            control1: CGPoint,
            control2: CGPoint,
            end: CGPoint
        )
        case close
    }

    private let elements: [Element]

    internal init(
        capturing path: CGPath
    ) throws(CACommittedAnimationCaptureError) {
        do {
            try ShapeFillTessellator.validate(path)
        } catch {
            throw .nonFiniteValue("CGPath")
        }

        var captured: [Element] = []
        var captureError: CACommittedAnimationCaptureError?
        path.applyWithBlock { elementPointer in
            guard captureError == nil else { return }
            let element = elementPointer.pointee
            switch element.type {
            case .moveToPoint:
                guard let points = element.points else {
                    captureError = .unsupportedValueType("CGPath")
                    return
                }
                captured.append(.move(points[0]))
            case .addLineToPoint:
                guard let points = element.points else {
                    captureError = .unsupportedValueType("CGPath")
                    return
                }
                captured.append(.line(points[0]))
            case .addQuadCurveToPoint:
                guard let points = element.points else {
                    captureError = .unsupportedValueType("CGPath")
                    return
                }
                captured.append(
                    .quadratic(
                        control: points[0],
                        end: points[1]
                    )
                )
            case .addCurveToPoint:
                guard let points = element.points else {
                    captureError = .unsupportedValueType("CGPath")
                    return
                }
                captured.append(
                    .cubic(
                        control1: points[0],
                        control2: points[1],
                        end: points[2]
                    )
                )
            case .closeSubpath:
                captured.append(.close)
            @unknown default:
                captureError = .unsupportedValueType("CGPath")
            }
        }
        if let captureError {
            throw captureError
        }
        elements = captured
    }

    internal func materialize() -> CGPath {
        let path = CGMutablePath()
        for element in elements {
            switch element {
            case .move(let point):
                path.move(to: point)
            case .line(let point):
                path.addLine(to: point)
            case .quadratic(let control, let end):
                path.addQuadCurve(
                    to: end,
                    control: control
                )
            case .cubic(let control1, let control2, let end):
                path.addCurve(
                    to: end,
                    control1: control1,
                    control2: control2
                )
            case .close:
                path.closeSubpath()
            }
        }
        return path
    }
}
