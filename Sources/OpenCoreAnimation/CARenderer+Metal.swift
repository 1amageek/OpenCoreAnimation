#if canImport(Metal)
import Metal

extension CARenderer {
    /// Creates a layer renderer whose destination is the supplied Metal texture.
    public convenience init(
        mtlTexture texture: any MTLTexture,
        options: [AnyHashable: Any]? = nil
    ) {
        switch CAMetalRendererOptions.parse(options) {
        case .success(let parsedOptions):
            self.init(
                backend: CAMetalRenderer(
                    destination: texture,
                    commandQueue: parsedOptions.commandQueue,
                    outputColorSpace: parsedOptions.outputColorSpace
                )
            )
        case .failure(let error):
            self.init(backend: CAMetalRenderer())
            recordInterfaceRenderError(error)
        }
    }

    /// Replaces the Metal texture receiving subsequent frames.
    public func setDestination(_ texture: any MTLTexture) {
        guard let metalBackend = backend as? CAMetalRenderer else {
            recordInterfaceRenderError(.incompatibleRendererBackend)
            return
        }
        recordInterfaceRenderError(nil)
        metalBackend.replaceDestination(texture)
    }
}

private struct CAMetalRendererOptions {
    internal let commandQueue: (any MTLCommandQueue)?
    internal let outputColorSpace:
        OpenCoreGraphics.CGColorSpace?

    internal static func parse(
        _ options: [AnyHashable: Any]?
    ) -> Result<Self, CARendererError> {
        var commandQueue: (any MTLCommandQueue)?
        var outputColorSpace:
            OpenCoreGraphics.CGColorSpace?

        for (rawKey, value) in options ?? [:] {
            guard let key = rawKey.base as? String else {
                return .failure(
                    .unsupportedRendererOption(
                        String(describing: rawKey)
                    )
                )
            }
            switch key {
            case kCARendererMetalCommandQueue:
                guard let queue =
                        value as? any MTLCommandQueue else {
                    return .failure(
                        .invalidRendererOption(
                            key: key,
                            expected: "MTLCommandQueue",
                            actual: String(
                                reflecting: type(of: value)
                            )
                        )
                    )
                }
                commandQueue = queue
            case kCARendererColorSpace:
                guard let colorSpace =
                        value as?
                        OpenCoreGraphics.CGColorSpace else {
                    return .failure(
                        .invalidRendererOption(
                            key: key,
                            expected:
                                "OpenCoreGraphics.CGColorSpace",
                            actual: String(
                                reflecting: type(of: value)
                            )
                        )
                    )
                }
                guard colorSpace.model == .rgb,
                      colorSpace.numberOfComponents == 3 else {
                    return .failure(
                        .unsupportedRendererColorSpace(
                            colorSpace.name
                                ?? String(
                                    describing: colorSpace.model
                                )
                        )
                    )
                }
                outputColorSpace = colorSpace
            default:
                return .failure(
                    .unsupportedRendererOption(key)
                )
            }
        }

        return .success(
            Self(
                commandQueue: commandQueue,
                outputColorSpace: outputColorSpace
            )
        )
    }
}
#endif
