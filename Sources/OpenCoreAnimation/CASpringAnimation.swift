//
//  CASpringAnimation.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation
import OpenCoreGraphics


/// An animation that applies a spring-like force to a layer's properties.
open class CASpringAnimation: CABasicAnimation {

    /// The mass of the object attached to the end of the spring.
    /// Must be greater than 0. Default is 1.
    open var mass: CGFloat = 1 {
        didSet { mass = max(0.001, mass) }
    }

    /// The spring stiffness coefficient.
    /// Must be greater than 0. Default is 100.
    open var stiffness: CGFloat = 100 {
        didSet { stiffness = max(0.001, stiffness) }
    }

    /// The damping coefficient.
    /// Must be greater than 0. Default is 10.
    open var damping: CGFloat = 10 {
        didSet { damping = max(0.001, damping) }
    }

    /// The initial velocity of the object attached to the spring.
    open var initialVelocity: CGFloat = 0

    /// The estimated duration required for the spring system to be considered at rest.
    ///
    /// This is calculated based on the spring parameters. The spring is considered
    /// at rest when oscillations have decayed to less than 0.1% of the initial displacement.
    open var settlingDuration: CFTimeInterval {
        // Ensure valid parameters to prevent division by zero
        let safeMass = max(0.001, mass)
        let safeStiffness = max(0.001, stiffness)
        let safeDamping = max(0.001, damping)

        // Calculate damping ratio: ζ = c / (2 * sqrt(k * m))
        let dampingRatio = safeDamping / (2 * sqrt(safeStiffness * safeMass))

        // Use tolerance for damping regime classification to avoid
        // floating-point boundary issues (e.g., dampingRatio = 0.9999999).
        let criticalDampingTolerance = 1e-6
        let naturalFrequency = sqrt(safeStiffness / safeMass)

        if abs(dampingRatio - 1) < criticalDampingTolerance {
            // Critically damped: settling time ≈ 4 / ω_n
            return 4 / naturalFrequency
        } else if dampingRatio > 1 {
            // Overdamped: settling time ≈ 4 / (ω_n * (ζ - sqrt(ζ² - 1)))
            // Uses the slower eigenvalue for accurate settling
            let slowEigenvalue = naturalFrequency * (dampingRatio - sqrt(dampingRatio * dampingRatio - 1))
            return 4 / slowEigenvalue
        } else {
            // Underdamped: settling time ≈ 4 / (ζ * ω_n)
            return 4 / (dampingRatio * naturalFrequency)
        }
    }

    /// Override to use settlingDuration when duration is not explicitly set.
    internal override var effectiveBaseDuration: CFTimeInterval {
        // If duration is explicitly set, use it; otherwise use settlingDuration
        return duration > 0 ? duration : settlingDuration
    }

    // MARK: - Spring Physics

    /// Calculates the spring interpolation value at a given time.
    ///
    /// This implements the damped harmonic oscillator equation:
    /// - Underdamped (ζ < 1): Oscillates with decreasing amplitude
    /// - Critically damped (ζ = 1): Returns to rest without oscillation, fastest
    /// - Overdamped (ζ > 1): Returns to rest without oscillation, slower
    ///
    /// - Parameter time: The elapsed time in seconds since animation start.
    /// - Returns: The interpolated value from 0 to 1, may overshoot for underdamped springs.
    internal func springValue(at time: CFTimeInterval) -> CGFloat {
        let safeMass = max(0.001, mass)
        let safeStiffness = max(0.001, stiffness)
        let safeDamping = max(0.001, damping)

        // Natural frequency: ω_n = sqrt(k / m)
        let omega_n = sqrt(safeStiffness / safeMass)

        // Damping ratio: ζ = c / (2 * sqrt(k * m))
        let zeta = safeDamping / (2 * sqrt(safeStiffness * safeMass))

        // Initial conditions: starting at 0, moving towards 1
        // x(0) = 0 (start position)
        // x(∞) = 1 (target position)
        // v(0) = initialVelocity

        let t = CGFloat(time)

        // Use tolerance for damping regime classification (consistent with settlingDuration)
        if zeta < 1 - 1e-6 {
            // Underdamped: oscillates
            // x(t) = 1 - e^(-ζω_n*t) * (cos(ω_d*t) + ((ζω_n - v0) / ω_d) * sin(ω_d*t))
            let omega_d = omega_n * sqrt(1 - zeta * zeta)  // Damped frequency
            let decay = exp(-zeta * omega_n * t)
            let cosComponent = cos(omega_d * t)
            let sinCoefficient = (zeta * omega_n - initialVelocity) / omega_d
            let sinComponent = sinCoefficient * sin(omega_d * t)

            return 1 - decay * (cosComponent + sinComponent)
        } else if abs(zeta - 1) < 1e-6 {
            // Critically damped: fastest return without oscillation
            // x(t) = 1 - e^(-ω_n*t) * (1 + (ω_n - v0) * t)
            let decay = exp(-omega_n * t)
            return 1 - decay * (1 + (omega_n - initialVelocity) * t)
        } else {
            // Overdamped (zeta > 1): slow return without oscillation
            // x(t) = 1 - A*e^(r1*t) - B*e^(r2*t)
            // where r1, r2 = -ω_n * (ζ ± sqrt(ζ² - 1))
            let sqrtTerm = sqrt(zeta * zeta - 1)
            let r1 = -omega_n * (zeta - sqrtTerm)
            let r2 = -omega_n * (zeta + sqrtTerm)

            // Solve for A and B using initial conditions:
            // x(0) = 0 → 1 - A - B = 0 → A + B = 1
            // v(0) = v0 → -A*r1 - B*r2 = v0
            // A = (r2 + v0) / (r2 - r1)
            // B = 1 - A
            let A = (r2 + initialVelocity) / (r2 - r1)
            let B = 1 - A

            return 1 - A * exp(r1 * t) - B * exp(r2 * t)
        }
    }
}
