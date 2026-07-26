import Foundation

/// A value-owned tile drawing implementation captured at transaction commit.
///
/// Conforming values must preserve the content visible at the instant the
/// provider created the snapshot. Later provider or model-layer mutations must
/// not change this snapshot's output.
public protocol CATiledLayerContentSnapshot: Sendable {
    /// Draws one requested tile into a preconfigured bitmap context.
    func drawTile(
        _ tile: CATiledLayerTileDrawingInfo,
        in context: CGContext
    ) throws
}
