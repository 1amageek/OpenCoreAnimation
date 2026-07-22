import Foundation

struct ContentsRenderConfiguration: Equatable {
    struct Patch: Equatable {
        let destinationRect: CGRect
        let sourceUnitRect: CGRect
    }

    let patches: [Patch]

    static func usesNineSlice(
        gravity: CALayerContentsGravity,
        contentsCenter: CGRect
    ) -> Bool {
        guard contentsCenter != CGRect(x: 0, y: 0, width: 1, height: 1) else {
            return false
        }
        switch gravity {
        case .resize, .resizeAspect, .resizeAspectFill:
            return true
        default:
            return false
        }
    }

    init(
        imageSize: CGSize,
        boundsSize: CGSize,
        contentsRect: CGRect,
        contentsCenter: CGRect,
        contentsScale: CGFloat,
        gravity: CALayerContentsGravity
    ) throws(ContentsRenderConfigurationError) {
        guard imageSize.width.isFinite,
              imageSize.height.isFinite,
              imageSize.width > 0,
              imageSize.height > 0 else {
            throw ContentsRenderConfigurationError.invalidImageSize
        }
        guard boundsSize.width.isFinite,
              boundsSize.height.isFinite,
              boundsSize.width > 0,
              boundsSize.height > 0 else {
            throw ContentsRenderConfigurationError.invalidBounds
        }
        guard Self.isFinite(contentsRect),
              contentsRect.width > 0,
              contentsRect.height > 0 else {
            throw ContentsRenderConfigurationError.invalidContentsRect
        }
        guard Self.isFinite(contentsCenter),
              contentsCenter.minX >= 0,
              contentsCenter.minY >= 0,
              contentsCenter.maxX <= 1,
              contentsCenter.maxY <= 1,
              contentsCenter.width >= 0,
              contentsCenter.height >= 0 else {
            throw ContentsRenderConfigurationError.invalidContentsCenter
        }
        guard contentsScale.isFinite, contentsScale > 0 else {
            throw ContentsRenderConfigurationError.invalidContentsScale
        }

        let selectedPixelSize = CGSize(
            width: imageSize.width * contentsRect.width,
            height: imageSize.height * contentsRect.height
        )
        guard selectedPixelSize.width.isFinite,
              selectedPixelSize.height.isFinite,
              selectedPixelSize.width > 0,
              selectedPixelSize.height > 0 else {
            throw ContentsRenderConfigurationError.invalidContentsRect
        }

        let destinationRect = try Self.resolvedDestinationRect(
            sourcePointSize: CGSize(
                width: selectedPixelSize.width / contentsScale,
                height: selectedPixelSize.height / contentsScale
            ),
            boundsSize: boundsSize,
            gravity: gravity
        )

        var leftWidth = selectedPixelSize.width * contentsCenter.minX / contentsScale
        var rightWidth = selectedPixelSize.width * (1 - contentsCenter.maxX) / contentsScale
        let fixedWidth = leftWidth + rightWidth
        if fixedWidth > destinationRect.width, fixedWidth > 0 {
            let scale = destinationRect.width / fixedWidth
            leftWidth *= scale
            rightWidth *= scale
        }
        let centerWidth = max(0, destinationRect.width - leftWidth - rightWidth)

        var topHeight = selectedPixelSize.height * contentsCenter.minY / contentsScale
        var bottomHeight = selectedPixelSize.height * (1 - contentsCenter.maxY) / contentsScale
        let fixedHeight = topHeight + bottomHeight
        if fixedHeight > destinationRect.height, fixedHeight > 0 {
            let scale = destinationRect.height / fixedHeight
            topHeight *= scale
            bottomHeight *= scale
        }
        let centerHeight = max(0, destinationRect.height - topHeight - bottomHeight)

        let destinationColumns = [
            (destinationRect.minX, leftWidth),
            (destinationRect.minX + leftWidth, centerWidth),
            (destinationRect.maxX - rightWidth, rightWidth),
        ]
        let destinationRows = [
            (destinationRect.maxY - topHeight, topHeight),
            (destinationRect.minY + bottomHeight, centerHeight),
            (destinationRect.minY, bottomHeight),
        ]

        let sourceX = [
            contentsRect.minX,
            contentsRect.minX + contentsRect.width * contentsCenter.minX,
            contentsRect.minX + contentsRect.width * contentsCenter.maxX,
            contentsRect.maxX,
        ]
        let sourceY = [
            contentsRect.minY,
            contentsRect.minY + contentsRect.height * contentsCenter.minY,
            contentsRect.minY + contentsRect.height * contentsCenter.maxY,
            contentsRect.maxY,
        ]

        var resolvedPatches: [Patch] = []
        resolvedPatches.reserveCapacity(9)
        for row in 0..<3 {
            for column in 0..<3 {
                let destination = CGRect(
                    x: destinationColumns[column].0,
                    y: destinationRows[row].0,
                    width: destinationColumns[column].1,
                    height: destinationRows[row].1
                )
                let source = CGRect(
                    x: sourceX[column],
                    y: sourceY[row],
                    width: sourceX[column + 1] - sourceX[column],
                    height: sourceY[row + 1] - sourceY[row]
                )
                guard destination.width > 0,
                      destination.height > 0,
                      source.width > 0,
                      source.height > 0 else {
                    continue
                }
                resolvedPatches.append(Patch(
                    destinationRect: destination,
                    sourceUnitRect: source
                ))
            }
        }
        patches = resolvedPatches
    }

    static func destinationRect(
        imageSize: CGSize,
        boundsSize: CGSize,
        contentsRect: CGRect,
        contentsScale: CGFloat,
        gravity: CALayerContentsGravity
    ) throws(ContentsRenderConfigurationError) -> CGRect {
        guard imageSize.width.isFinite,
              imageSize.height.isFinite,
              imageSize.width > 0,
              imageSize.height > 0 else {
            throw ContentsRenderConfigurationError.invalidImageSize
        }
        guard boundsSize.width.isFinite,
              boundsSize.height.isFinite,
              boundsSize.width > 0,
              boundsSize.height > 0 else {
            throw ContentsRenderConfigurationError.invalidBounds
        }
        guard isFinite(contentsRect),
              contentsRect.width > 0,
              contentsRect.height > 0 else {
            throw ContentsRenderConfigurationError.invalidContentsRect
        }
        guard contentsScale.isFinite, contentsScale > 0 else {
            throw ContentsRenderConfigurationError.invalidContentsScale
        }
        let sourcePointSize = CGSize(
            width: imageSize.width * contentsRect.width / contentsScale,
            height: imageSize.height * contentsRect.height / contentsScale
        )
        guard sourcePointSize.width.isFinite,
              sourcePointSize.height.isFinite,
              sourcePointSize.width > 0,
              sourcePointSize.height > 0 else {
            throw ContentsRenderConfigurationError.invalidContentsRect
        }
        return try resolvedDestinationRect(
            sourcePointSize: sourcePointSize,
            boundsSize: boundsSize,
            gravity: gravity
        )
    }

    private static func isFinite(_ rect: CGRect) -> Bool {
        rect.origin.x.isFinite
            && rect.origin.y.isFinite
            && rect.width.isFinite
            && rect.height.isFinite
    }

    private static func resolvedDestinationRect(
        sourcePointSize: CGSize,
        boundsSize: CGSize,
        gravity: CALayerContentsGravity
    ) throws(ContentsRenderConfigurationError) -> CGRect {
        switch gravity {
        case .center:
            return CGRect(
                x: (boundsSize.width - sourcePointSize.width) / 2,
                y: (boundsSize.height - sourcePointSize.height) / 2,
                width: sourcePointSize.width,
                height: sourcePointSize.height
            )
        case .resize:
            return CGRect(origin: .zero, size: boundsSize)
        case .resizeAspect:
            let scale = min(
                boundsSize.width / sourcePointSize.width,
                boundsSize.height / sourcePointSize.height
            )
            let size = CGSize(
                width: sourcePointSize.width * scale,
                height: sourcePointSize.height * scale
            )
            return CGRect(
                x: (boundsSize.width - size.width) / 2,
                y: (boundsSize.height - size.height) / 2,
                width: size.width,
                height: size.height
            )
        case .resizeAspectFill:
            let scale = max(
                boundsSize.width / sourcePointSize.width,
                boundsSize.height / sourcePointSize.height
            )
            let size = CGSize(
                width: sourcePointSize.width * scale,
                height: sourcePointSize.height * scale
            )
            return CGRect(
                x: (boundsSize.width - size.width) / 2,
                y: (boundsSize.height - size.height) / 2,
                width: size.width,
                height: size.height
            )
        case .top:
            return CGRect(
                x: (boundsSize.width - sourcePointSize.width) / 2,
                y: boundsSize.height - sourcePointSize.height,
                width: sourcePointSize.width,
                height: sourcePointSize.height
            )
        case .bottom:
            return CGRect(
                x: (boundsSize.width - sourcePointSize.width) / 2,
                y: 0,
                width: sourcePointSize.width,
                height: sourcePointSize.height
            )
        case .left:
            return CGRect(
                x: 0,
                y: (boundsSize.height - sourcePointSize.height) / 2,
                width: sourcePointSize.width,
                height: sourcePointSize.height
            )
        case .right:
            return CGRect(
                x: boundsSize.width - sourcePointSize.width,
                y: (boundsSize.height - sourcePointSize.height) / 2,
                width: sourcePointSize.width,
                height: sourcePointSize.height
            )
        case .topLeft:
            return CGRect(
                x: 0,
                y: boundsSize.height - sourcePointSize.height,
                width: sourcePointSize.width,
                height: sourcePointSize.height
            )
        case .topRight:
            return CGRect(
                x: boundsSize.width - sourcePointSize.width,
                y: boundsSize.height - sourcePointSize.height,
                width: sourcePointSize.width,
                height: sourcePointSize.height
            )
        case .bottomLeft:
            return CGRect(origin: .zero, size: sourcePointSize)
        case .bottomRight:
            return CGRect(
                x: boundsSize.width - sourcePointSize.width,
                y: 0,
                width: sourcePointSize.width,
                height: sourcePointSize.height
            )
        default:
            throw ContentsRenderConfigurationError.unsupportedGravity(gravity.rawValue)
        }
    }
}
