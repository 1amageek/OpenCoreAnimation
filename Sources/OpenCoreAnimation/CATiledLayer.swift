
/// A layer that provides a way to asynchronously provide tiles of the layer's content,
/// potentially cached at multiple levels of detail.
open class CATiledLayer: CALayer {

    /// The number of levels of detail maintained by this layer.
    open var levelsOfDetail: Int = 1

    /// The number of magnified levels of detail for this layer.
    open var levelsOfDetailBias: Int = 0

    /// The maximum size of each tile.
    open var tileSize: CGSize = CGSize(width: 256, height: 256)

    /// Returns the fading duration for a given view.
    open class func fadeDuration() -> CFTimeInterval {
        return 0.25
    }
}
