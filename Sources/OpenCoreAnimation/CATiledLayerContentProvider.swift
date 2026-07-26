import Foundation

/// A tile delegate that can create immutable content for committed rendering.
///
/// Assign the conforming object through `CALayer.delegate`. The provider is
/// called synchronously during transaction capture. The renderer retains only
/// the returned snapshot, never the mutable provider or model layer.
public protocol CATiledLayerContentProvider:
    CALayerDelegate,
    Sendable {
    /// Captures the tile content visible at the current transaction boundary.
    func makeTileContentSnapshot()
        throws -> any CATiledLayerContentSnapshot
}
