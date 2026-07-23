//
//  CASpringAnimation.swift
//  OpenCoreAnimation
//
//  Internal delegate protocol for rendering layer trees.
//

import Foundation


/// An animation that applies a spring-like force to a layer's properties.
open class CASpringAnimation: CABasicAnimation {

    private static let restDisplacement: CGFloat = 0.001
    private static let settlingSampleInterval: CFTimeInterval = 0.1

    /// The mass of the object attached to the end of the spring.
    /// Must be greater than 0. Default is 1.
    private var _mass: CGFloat = 1
    open var mass: CGFloat {
        get { _mass }
        set {
            guard newValue > 0 else { return }
            _mass = newValue
        }
    }

    /// The spring stiffness coefficient.
    /// Must be greater than 0. Default is 100.
    private var _stiffness: CGFloat = 100
    open var stiffness: CGFloat {
        get { _stiffness }
        set {
            guard newValue > 0 else { return }
            _stiffness = newValue
        }
    }

    /// The damping coefficient.
    /// Must be greater than or equal to 0. Default is 10.
    private var _damping: CGFloat = 10
    open var damping: CGFloat {
        get { _damping }
        set {
            guard newValue >= 0 else { return }
            _damping = newValue
        }
    }

    /// The initial velocity of the object attached to the spring.
    open var initialVelocity: CGFloat = 0

    /// Controls whether damping ratios greater than one use an overdamped response.
    ///
    /// When this is `false`, damping above the critical value is evaluated as a
    /// critically damped spring. The physical coefficient remains available through
    /// `damping`, matching QuartzCore's separation between stored parameters and the
    /// response policy.
    open var allowsOverdamping: Bool = false

    /// The duration of one cycle of the corresponding undamped oscillator.
    open var perceptualDuration: CFTimeInterval {
        2 * .pi * sqrt(CFTimeInterval(mass / stiffness))
    }

    /// A normalized description of the spring's overshoot.
    ///
    /// Positive values describe underdamping, zero is critical damping, and
    /// negative values describe overdamping.
    open var bounce: CGFloat {
        let ratio = physicalDampingRatio
        if ratio <= 1 {
            return 1 - ratio
        }
        return (1 / ratio) - 1
    }

    public required init() {
        super.init()
    }

    /// Creates a spring from perceptual duration and bounce parameters.
    public convenience init(perceptualDuration: CFTimeInterval, bounce: CGFloat) {
        self.init()

        guard perceptualDuration.isFinite, perceptualDuration > 0, bounce.isFinite else {
            return
        }

        let angularFrequency = 2 * Double.pi / perceptualDuration
        stiffness = CGFloat(angularFrequency * angularFrequency)
        allowsOverdamping = true

        let dampingRatio: CGFloat
        if bounce >= 0 {
            dampingRatio = 1 - bounce
        } else if bounce > -1 {
            dampingRatio = 1 / (1 + bounce)
        } else {
            dampingRatio = .infinity
        }
        damping = 2 * dampingRatio * sqrt(mass * stiffness)
        duration = settlingDuration
    }

    public required init(animation: CAAnimation) {
        super.init(animation: animation)
        if let source = animation as? CASpringAnimation {
            self.mass = source.mass
            self.stiffness = source.stiffness
            self.damping = source.damping
            self.initialVelocity = source.initialVelocity
            self.allowsOverdamping = source.allowsOverdamping
        }
    }

    open override func shouldArchiveValue(forKey key: String) -> Bool {
        switch key {
        case "mass":
            return mass != 1
        case "stiffness":
            return stiffness != 100
        case "damping":
            return damping != 10
        case "initialVelocity":
            // QuartzCore derives this from the runtime spring configuration
            // and does not persist it through shouldArchiveValue(forKey:).
            return false
        case "allowsOverdamping":
            return allowsOverdamping
        default:
            return super.shouldArchiveValue(forKey: key)
        }
    }

    /// Returns the default value for a spring animation property.
    open override class func defaultValue(forKey key: String) -> Any? {
        switch key {
        case "mass":
            return CGFloat(1)
        case "stiffness":
            return CGFloat(100)
        case "damping":
            return CGFloat(10)
        case "allowsOverdamping":
            return false
        default:
            return super.defaultValue(forKey: key)
        }
    }

    /// The estimated duration required for the spring system to be considered at rest.
    ///
    /// This is calculated based on the spring parameters. The spring is considered
    /// at rest when oscillations have decayed to less than 0.1% of the initial displacement.
    open var settlingDuration: CFTimeInterval {
        if damping == 0 {
            return CFTimeInterval(Float.greatestFiniteMagnitude)
        }
        if damping == .infinity, allowsOverdamping {
            return 0
        }

        let dampingRatio = effectiveDampingRatio
        let naturalFrequency = sqrt(stiffness / mass)
        if dampingRatio < 1 {
            let dampedFrequency = naturalFrequency * sqrt(1 - dampingRatio * dampingRatio)
            let decayRate = dampingRatio * naturalFrequency
            let sineCoefficient = (decayRate - initialVelocity) / dampedFrequency
            let envelope = sqrt(1 + sineCoefficient * sineCoefficient)
            let envelopeDuration = log(envelope / Self.restDisplacement) / decayRate

            // QuartzCore leaves a phase-dependent fraction of the natural response
            // after the exponential envelope reaches the rest threshold. This term
            // keeps the estimate aligned with the actual oscillator instead of the
            // former four-time-constant approximation.
            let phaseAllowance = sqrt(1 - dampingRatio) / naturalFrequency
            return max(0, CFTimeInterval(envelopeDuration + phaseAllowance))
        }

        if dampingRatio == 1 {
            return sampledCriticalSettlingDuration(naturalFrequency: naturalFrequency)
        }

        // The perceptual overdamping policy intentionally avoids the arbitrarily
        // long tail of the slow physical eigenvalue. QuartzCore quantizes this
        // estimate to tenths of a second.
        let dimensionlessDuration = 9.3 * (1 + log(dampingRatio))
        let estimatedDuration = CFTimeInterval(dimensionlessDuration / naturalFrequency)
        return ceil(estimatedDuration / Self.settlingSampleInterval)
            * Self.settlingSampleInterval
    }

    /// Override to use settlingDuration when duration is not explicitly set.
    internal override var effectiveBaseDuration: CFTimeInterval {
        return durationOrFallback(settlingDuration)
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
        if damping == .infinity, allowsOverdamping {
            return time <= 0 ? 0 : 1
        }

        let safeMass = mass
        let safeStiffness = stiffness

        // Natural frequency: ω_n = sqrt(k / m)
        let omega_n = sqrt(safeStiffness / safeMass)

        // Damping ratio: ζ = c / (2 * sqrt(k * m))
        let zeta = effectiveDampingRatio

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

    private var physicalDampingRatio: CGFloat {
        damping / (2 * sqrt(stiffness * mass))
    }

    private var effectiveDampingRatio: CGFloat {
        allowsOverdamping ? physicalDampingRatio : min(physicalDampingRatio, 1)
    }

    private func sampledCriticalSettlingDuration(
        naturalFrequency: CGFloat
    ) -> CFTimeInterval {
        let linearCoefficient = naturalFrequency - initialVelocity
        var time = Self.settlingSampleInterval

        while time < CFTimeInterval(Float.greatestFiniteMagnitude) {
            let scalarTime = CGFloat(time)
            let displacement = exp(-naturalFrequency * scalarTime)
                * (1 + linearCoefficient * scalarTime)
            if abs(displacement) < Self.restDisplacement {
                return time
            }
            time += Self.settlingSampleInterval
        }
        return CFTimeInterval(Float.greatestFiniteMagnitude)
    }
}
