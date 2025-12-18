
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

        if dampingRatio >= 1 {
            // Overdamped or critically damped
            // Settling time ≈ 4 * τ where τ = m / c
            return 4 * safeMass / safeDamping
        } else {
            // Underdamped
            // Natural frequency: ω_n = sqrt(k / m)
            // Settling time ≈ 4 / (ζ * ω_n)
            let naturalFrequency = sqrt(safeStiffness / safeMass)
            return 4 / (dampingRatio * naturalFrequency)
        }
    }

    /// Override to use settlingDuration when duration is not explicitly set.
    internal override var effectiveBaseDuration: CFTimeInterval {
        // If duration is explicitly set, use it; otherwise use settlingDuration
        return duration > 0 ? duration : settlingDuration
    }
}
