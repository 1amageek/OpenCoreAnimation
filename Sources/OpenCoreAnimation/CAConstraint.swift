
/// A representation of a single layout constraint between two layers.
///
/// Each CAConstraint instance encapsulates one geometry relationship between two layers on the same axis.
/// Sibling layers are referenced by name, using the name property of each layer. The special name "superlayer"
/// is used to refer to the layer's superlayer.
public final class CAConstraint: Hashable, Sendable {
    /// The attribute the constraint affects.
    public let attribute: CAConstraintAttribute

    /// Offset value of the constraint attribute.
    public let offset: CGFloat

    /// Scale factor of the constraint attribute.
    public let scale: CGFloat

    /// The constraint attribute of the layer the receiver is calculated relative to.
    public let sourceAttribute: CAConstraintAttribute

    /// Name of the layer that the constraint is calculated relative to.
    public let sourceName: String

    /// Returns a CAConstraint object with the specified parameters.
    ///
    /// - Parameters:
    ///   - attribute: The attribute of the layer to constrain.
    ///   - sourceName: The name of the layer to constrain the attribute to.
    ///   - sourceAttribute: The attribute of the source layer to constrain.
    ///   - scale: The scale factor to apply.
    ///   - offset: The offset value.
    public init(
        attribute: CAConstraintAttribute,
        relativeTo sourceName: String,
        attribute sourceAttribute: CAConstraintAttribute,
        scale: CGFloat,
        offset: CGFloat
    ) {
        self.attribute = attribute
        self.sourceName = sourceName
        self.sourceAttribute = sourceAttribute
        self.scale = scale
        self.offset = offset
    }

    /// Creates and returns a CAConstraint object with the specified parameters.
    ///
    /// - Parameters:
    ///   - attribute: The attribute of the layer to constrain.
    ///   - sourceName: The name of the layer to constrain the attribute to.
    ///   - sourceAttribute: The attribute of the source layer to constrain.
    ///   - offset: The offset value.
    public convenience init(
        attribute: CAConstraintAttribute,
        relativeTo sourceName: String,
        attribute sourceAttribute: CAConstraintAttribute,
        offset: CGFloat
    ) {
        self.init(attribute: attribute, relativeTo: sourceName, attribute: sourceAttribute, scale: 1.0, offset: offset)
    }

    /// Creates and returns a CAConstraint object with the specified parameters.
    ///
    /// - Parameters:
    ///   - attribute: The attribute of the layer to constrain.
    ///   - sourceName: The name of the layer to constrain the attribute to.
    ///   - sourceAttribute: The attribute of the source layer to constrain.
    public convenience init(
        attribute: CAConstraintAttribute,
        relativeTo sourceName: String,
        attribute sourceAttribute: CAConstraintAttribute
    ) {
        self.init(attribute: attribute, relativeTo: sourceName, attribute: sourceAttribute, scale: 1.0, offset: 0.0)
    }

    // MARK: - Hashable

    public func hash(into hasher: inout Hasher) {
        hasher.combine(attribute)
        hasher.combine(sourceName)
        hasher.combine(sourceAttribute)
        hasher.combine(scale)
        hasher.combine(offset)
    }

    public static func == (lhs: CAConstraint, rhs: CAConstraint) -> Bool {
        return lhs.attribute == rhs.attribute &&
               lhs.sourceName == rhs.sourceName &&
               lhs.sourceAttribute == rhs.sourceAttribute &&
               lhs.scale == rhs.scale &&
               lhs.offset == rhs.offset
    }
}
