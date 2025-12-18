
/// A function that defines the pacing of an animation as a timing curve.
///
/// CAMediaTimingFunction represents one segment of a function that defines the pacing of an animation
/// as a timing curve. The function maps an input time normalized to the range [0,1] to an output time
/// also in the range [0,1].
public class CAMediaTimingFunction: Hashable {

    /// The control points of the cubic Bézier curve.
    private let controlPoints: [Float]

    // MARK: - Initialization

    /// Creates and returns a new instance of CAMediaTimingFunction configured with the predefined
    /// timing function specified by name.
    ///
    /// - Parameter name: The name of the predefined timing function.
    public convenience init(name: CAMediaTimingFunctionName) {
        switch name {
        case .linear:
            self.init(controlPoints: 0.0, 0.0, 1.0, 1.0)
        case .easeIn:
            self.init(controlPoints: 0.42, 0.0, 1.0, 1.0)
        case .easeOut:
            self.init(controlPoints: 0.0, 0.0, 0.58, 1.0)
        case .easeInEaseOut:
            self.init(controlPoints: 0.42, 0.0, 0.58, 1.0)
        case .default:
            self.init(controlPoints: 0.25, 0.1, 0.25, 1.0)
        default:
            self.init(controlPoints: 0.0, 0.0, 1.0, 1.0)
        }
    }

    /// Returns an initialized timing function modeled as a cubic Bézier curve using the specified control points.
    ///
    /// - Parameters:
    ///   - c1x: The x value of the first control point.
    ///   - c1y: The y value of the first control point.
    ///   - c2x: The x value of the second control point.
    ///   - c2y: The y value of the second control point.
    public init(controlPoints c1x: Float, _ c1y: Float, _ c2x: Float, _ c2y: Float) {
        // The curve starts at (0,0) and ends at (1,1)
        // Control points are (c1x, c1y) and (c2x, c2y)
        self.controlPoints = [c1x, c1y, c2x, c2y]
    }

    // MARK: - Accessing Control Points

    /// Returns the control point for the specified index.
    ///
    /// - Parameters:
    ///   - idx: The index of the control point to return (0-3).
    ///   - ptr: On output, the x and y values of the control point.
    public func getControlPoint(at idx: Int, values ptr: UnsafeMutablePointer<Float>) {
        switch idx {
        case 0:
            ptr[0] = 0.0
            ptr[1] = 0.0
        case 1:
            ptr[0] = controlPoints[0]
            ptr[1] = controlPoints[1]
        case 2:
            ptr[0] = controlPoints[2]
            ptr[1] = controlPoints[3]
        case 3:
            ptr[0] = 1.0
            ptr[1] = 1.0
        default:
            break
        }
    }

    // MARK: - Evaluation

    /// Evaluates the timing function at the given input time.
    ///
    /// - Parameter t: The input time, normalized to the range [0, 1].
    /// - Returns: The output time, also in the range [0, 1].
    public func evaluate(at t: Float) -> Float {
        // Handle edge cases
        if t <= 0 { return 0 }
        if t >= 1 { return 1 }

        // Cubic Bézier curve evaluation
        // P(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        // where P0 = (0,0), P1 = (c1x, c1y), P2 = (c2x, c2y), P3 = (1,1)

        let c1x = controlPoints[0]
        let c1y = controlPoints[1]
        let c2x = controlPoints[2]
        let c2y = controlPoints[3]

        // Find the parameter 'u' for which x(u) = t using Newton-Raphson
        // with convergence check and fallback to bisection
        var u = t
        let epsilon: Float = 1e-6
        let maxIterations = 16

        for _ in 0..<maxIterations {
            let x = bezierX(u, c1x: c1x, c2x: c2x)
            let error = x - t

            // Check for convergence
            if abs(error) < epsilon {
                break
            }

            let dx = bezierDX(u, c1x: c1x, c2x: c2x)

            // If derivative is too small, use bisection instead
            if abs(dx) < epsilon {
                // Fallback to bisection method
                u = bisectionSolve(t: t, c1x: c1x, c2x: c2x)
                break
            }

            u -= error / dx
            u = max(0, min(1, u))
        }

        // Calculate y at the found parameter u
        return bezierY(u, c1y: c1y, c2y: c2y)
    }

    /// Fallback bisection method for finding the parameter u where x(u) = t.
    private func bisectionSolve(t: Float, c1x: Float, c2x: Float) -> Float {
        var low: Float = 0
        var high: Float = 1
        var mid: Float = t

        for _ in 0..<20 {
            mid = (low + high) / 2
            let x = bezierX(mid, c1x: c1x, c2x: c2x)

            if abs(x - t) < 1e-6 {
                break
            }

            if x < t {
                low = mid
            } else {
                high = mid
            }
        }

        return mid
    }

    /// Calculates the x-coordinate of the Bézier curve at parameter u.
    private func bezierX(_ u: Float, c1x: Float, c2x: Float) -> Float {
        let oneMinusU = 1 - u
        let oneMinusU2 = oneMinusU * oneMinusU
        let oneMinusU3 = oneMinusU2 * oneMinusU
        let u2 = u * u
        let u3 = u2 * u
        return oneMinusU3 * 0 + 3 * oneMinusU2 * u * c1x + 3 * oneMinusU * u2 * c2x + u3 * 1
    }

    /// Calculates the y-coordinate of the Bézier curve at parameter u.
    private func bezierY(_ u: Float, c1y: Float, c2y: Float) -> Float {
        let oneMinusU = 1 - u
        let oneMinusU2 = oneMinusU * oneMinusU
        let oneMinusU3 = oneMinusU2 * oneMinusU
        let u2 = u * u
        let u3 = u2 * u
        return oneMinusU3 * 0 + 3 * oneMinusU2 * u * c1y + 3 * oneMinusU * u2 * c2y + u3 * 1
    }

    /// Calculates the derivative of x with respect to u.
    private func bezierDX(_ u: Float, c1x: Float, c2x: Float) -> Float {
        let oneMinusU = 1 - u
        return 3 * oneMinusU * oneMinusU * c1x +
               6 * oneMinusU * u * (c2x - c1x) +
               3 * u * u * (1 - c2x)
    }

    // MARK: - Hashable

    public func hash(into hasher: inout Hasher) {
        hasher.combine(controlPoints)
    }

    public static func == (lhs: CAMediaTimingFunction, rhs: CAMediaTimingFunction) -> Bool {
        return lhs.controlPoints == rhs.controlPoints
    }
}
