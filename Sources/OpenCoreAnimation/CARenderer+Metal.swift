#if canImport(Metal)
import Metal

extension CARenderer {
    /// Creates a layer renderer whose destination is the supplied Metal texture.
    public convenience init(
        mtlTexture texture: any MTLTexture,
        options: [AnyHashable: Any]? = nil
    ) {
        let metalBackend: CAMetalRenderer
        do {
            metalBackend = try CAMetalRenderer(destination: texture)
        } catch {
            preconditionFailure("Unable to create CARenderer: \(error)")
        }
        self.init(backend: metalBackend)
    }

    /// Replaces the Metal texture receiving subsequent frames.
    public func setDestination(_ texture: any MTLTexture) {
        guard let metalBackend = backend as? CAMetalRenderer else {
            preconditionFailure("The renderer was not created with a Metal destination")
        }
        do {
            try metalBackend.setDestination(texture)
        } catch {
            preconditionFailure("Unable to replace CARenderer destination: \(error)")
        }
    }
}
#endif
