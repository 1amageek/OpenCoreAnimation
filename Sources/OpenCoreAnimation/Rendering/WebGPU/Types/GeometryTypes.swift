#if arch(wasm32)
import Foundation
import OpenCoreGraphics

// MARK: - Geometry Cache Types

/// A key for caching tessellated geometry.
///
/// This key uniquely identifies a tessellated path based on:
/// - The path identity (via ObjectIdentifier for reference types, or hash for value types)
/// - Rendering parameters (line width, cap, join, fill rule)
/// - Stroke/fill mode
///
/// ## Note on Path Identity
///
/// For best cache hit rates, use `ObjectIdentifier` of the CGPath object when the path
/// object is reused across frames. If the path is recreated each frame with the same content,
/// consider using a content-based hash instead.
public struct GeometryCacheKey: Hashable {
    /// Unique identifier for the path object (ObjectIdentifier if available, or hash).
    public let pathIdentifier: Int

    /// Whether this is for stroke (true) or fill (false).
    public let isStroke: Bool

    /// Line width for strokes.
    public let lineWidth: CGFloat

    /// Line cap style for strokes.
    public let lineCap: CAShapeLayerLineCap

    /// Line join style for strokes.
    public let lineJoin: CAShapeLayerLineJoin

    /// Miter limit for miter joins.
    public let miterLimit: CGFloat

    /// Fill rule for fills.
    public let fillRule: CAShapeLayerFillRule

    /// Stroke start (0.0 to 1.0).
    public let strokeStart: CGFloat

    /// Stroke end (0.0 to 1.0).
    public let strokeEnd: CGFloat

    /// Creates a geometry cache key from a path object identifier.
    ///
    /// - Parameters:
    ///   - pathIdentifier: Use `ObjectIdentifier(path).hashValue` for CGPath reference types,
    ///     or `path.hashValue` for content-based hashing.
    ///   - isStroke: Whether this is for stroke rendering.
    ///   - lineWidth: Line width for strokes.
    ///   - lineCap: Line cap style.
    ///   - lineJoin: Line join style.
    ///   - miterLimit: Miter limit for miter joins.
    ///   - fillRule: Fill rule for fills.
    ///   - strokeStart: Stroke start position (0.0 to 1.0).
    ///   - strokeEnd: Stroke end position (0.0 to 1.0).
    public init(
        pathIdentifier: Int,
        isStroke: Bool,
        lineWidth: CGFloat = 1.0,
        lineCap: CAShapeLayerLineCap = .butt,
        lineJoin: CAShapeLayerLineJoin = .miter,
        miterLimit: CGFloat = 10.0,
        fillRule: CAShapeLayerFillRule = .nonZero,
        strokeStart: CGFloat = 0.0,
        strokeEnd: CGFloat = 1.0
    ) {
        self.pathIdentifier = pathIdentifier
        self.isStroke = isStroke
        self.lineWidth = lineWidth
        self.lineCap = lineCap
        self.lineJoin = lineJoin
        self.miterLimit = miterLimit
        self.fillRule = fillRule
        self.strokeStart = strokeStart
        self.strokeEnd = strokeEnd
    }

    /// Creates a geometry cache key from a CGPath.
    ///
    /// Uses ObjectIdentifier for stable identity across frames when the same path object is reused.
    ///
    /// - Parameters:
    ///   - path: The CGPath to create a key for.
    ///   - isStroke: Whether this is for stroke rendering.
    ///   - Other parameters same as main initializer.
    public init(
        path: CGPath,
        isStroke: Bool,
        lineWidth: CGFloat = 1.0,
        lineCap: CAShapeLayerLineCap = .butt,
        lineJoin: CAShapeLayerLineJoin = .miter,
        miterLimit: CGFloat = 10.0,
        fillRule: CAShapeLayerFillRule = .nonZero,
        strokeStart: CGFloat = 0.0,
        strokeEnd: CGFloat = 1.0
    ) {
        // Use ObjectIdentifier for stable identity when path object is reused
        self.pathIdentifier = ObjectIdentifier(path).hashValue
        self.isStroke = isStroke
        self.lineWidth = lineWidth
        self.lineCap = lineCap
        self.lineJoin = lineJoin
        self.miterLimit = miterLimit
        self.fillRule = fillRule
        self.strokeStart = strokeStart
        self.strokeEnd = strokeEnd
    }
}

/// Cached tessellated geometry data.
public struct TessellatedGeometry {
    /// Triangle vertices for rendering.
    public let vertices: [CGPoint]

    /// Number of triangles (vertices.count / 3).
    public var triangleCount: Int {
        return vertices.count / 3
    }

    /// Approximate memory usage in bytes.
    public var memorySize: Int {
        return vertices.count * MemoryLayout<CGPoint>.stride
    }

    public init(vertices: [CGPoint]) {
        self.vertices = vertices
    }
}

#endif
