import Foundation
#if arch(wasm32)
@_spi(SoftwareBitmapContext) import OpenCoreGraphics
#endif

/// A layer-owned immutable result of one delegate drawing pass.
internal struct CADelegateBackingStore {
    internal let image: CGImage
    internal let format: CADelegateBackingStoreFormat

    internal static func render(
        layer: CALayer,
        delegate: any CALayerDelegate,
        invalidation: CALayer.DisplayInvalidation,
        previous: CADelegateBackingStore?,
        maximumPixelDimension: Int
    ) throws(CADelegateBackingStoreError) -> Self {
        let bounds = layer.bounds
        let scale = layer.contentsScale
        guard bounds.width.isFinite,
              bounds.height.isFinite,
              bounds.width > 0,
              bounds.height > 0,
              scale.isFinite,
              scale > 0 else {
            throw .invalidGeometry
        }

        let pixelWidthValue = ceil(bounds.width * scale)
        let pixelHeightValue = ceil(bounds.height * scale)
        let maximumDimension = CGFloat(maximumPixelDimension)
        guard pixelWidthValue.isFinite,
              pixelHeightValue.isFinite,
              pixelWidthValue <= maximumDimension,
              pixelHeightValue <= maximumDimension,
              let pixelWidth = Int(exactly: pixelWidthValue),
              let pixelHeight = Int(exactly: pixelHeightValue) else {
            let reportedWidth = finiteDimension(pixelWidthValue)
            let reportedHeight = finiteDimension(pixelHeightValue)
            throw .dimensionsExceedTextureLimit(
                width: reportedWidth,
                height: reportedHeight,
                maximum: maximumPixelDimension
            )
        }

        let format = try CADelegateBackingStoreFormat.resolve(
            contentsFormat: layer.contentsFormat,
            contentsHeadroom: layer.contentsHeadroom
        )
        let storageMetrics = try validatedStorageMetrics(
            width: pixelWidth,
            height: pixelHeight,
            bitsPerPixel: format.bitsPerPixel
        )
        let colorSpace: CGColorSpace
        let bitmapInfo: CGBitmapInfo
        switch format {
        case .rgba8Uint:
            colorSpace = .deviceRGB
            bitmapInfo = CGBitmapInfo(
                rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        case .rgba16Float:
            guard let extendedColorSpace = CGColorSpace(
                name: CGColorSpace.extendedLinearSRGB
            ) else {
                throw .extendedColorSpaceUnavailable
            }
            colorSpace = extendedColorSpace
            bitmapInfo = CGBitmapInfo(
                rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
            )
            .union(.floatComponents)
            .union(.byteOrder16Little)
        case .gray8Uint:
            colorSpace = .deviceGray
            bitmapInfo = CGBitmapInfo(
                rawValue: CGImageAlphaInfo.none.rawValue
            )
        }
        let context: CGContext?
        #if arch(wasm32)
        context = CGContext(
            softwareData: nil,
            width: pixelWidth,
            height: pixelHeight,
            bitsPerComponent: format.bitsPerComponent,
            bytesPerRow: storageMetrics.bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        )
        #else
        context = CGContext(
            data: nil,
            width: pixelWidth,
            height: pixelHeight,
            bitsPerComponent: format.bitsPerComponent,
            bytesPerRow: storageMetrics.bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        )
        #endif
        guard let context else {
            throw .contextCreationFailed
        }
        if format == .rgba16Float,
           layer.contentsHeadroom > 1,
           !context.setEDRTargetHeadroom(Float(layer.contentsHeadroom)) {
            throw .extendedHeadroomRejected(layer.contentsHeadroom)
        }

        let invalidationRect: CGRect
        switch invalidation {
        case .full:
            invalidationRect = bounds
        case .partial(let requestedRect):
            invalidationRect = requestedRect.intersection(bounds)
            copyPreservedPixels(
                from: previous,
                into: context,
                width: pixelWidth,
                height: pixelHeight,
                byteCount: storageMetrics.byteCount,
                format: format,
                colorSpace: colorSpace,
                bitmapInfo: bitmapInfo
            )
        }

        if drawable(invalidationRect) {
            context.scaleBy(x: scale, y: scale)
            if layer.contentsAreFlipped() {
                context.translateBy(x: -bounds.minX, y: -bounds.minY)
            } else {
                context.translateBy(x: -bounds.minX, y: bounds.maxY)
                context.scaleBy(x: 1, y: -1)
            }
            context.clip(to: invalidationRect)
            context.clear(invalidationRect)
            delegate.layerWillDraw(layer)
            layer.draw(in: context)
        }
        guard let image = context.makeImage() else {
            throw .snapshotFailed
        }
        return Self(image: image, format: format)
    }

    private static func copyPreservedPixels(
        from previous: CADelegateBackingStore?,
        into context: CGContext,
        width: Int,
        height: Int,
        byteCount: Int,
        format: CADelegateBackingStoreFormat,
        colorSpace: CGColorSpace,
        bitmapInfo: CGBitmapInfo
    ) {
        guard let previous,
              previous.format == format,
              previous.image.width == width,
              previous.image.height == height,
              previous.image.bitsPerComponent == format.bitsPerComponent,
              previous.image.bitsPerPixel == format.bitsPerPixel,
              previous.image.bytesPerRow == context.bytesPerRow,
              previous.image.colorSpace == colorSpace,
              previous.image.bitmapInfo == bitmapInfo,
              let previousData = previous.image.data,
              previousData.count >= byteCount,
              let destination = CGBitmapContextGetData(context) else {
            return
        }
        // Partial redraw requires a copy because the previous image is
        // immutable while the new context owns independently mutable storage.
        previousData.withUnsafeBytes { source in
            guard let sourceAddress = source.baseAddress else { return }
            destination.copyMemory(
                from: sourceAddress,
                byteCount: byteCount
            )
        }
    }

    private static func validatedStorageMetrics(
        width: Int,
        height: Int,
        bitsPerPixel: Int
    ) throws(CADelegateBackingStoreError) -> (
        bytesPerRow: Int,
        byteCount: Int
    ) {
        let (rowBits, rowBitsOverflow) = width.multipliedReportingOverflow(
            by: bitsPerPixel
        )
        let (roundedRowBits, roundingOverflow) =
            rowBits.addingReportingOverflow(7)
        let minimumBytesPerRow = roundedRowBits / 8
        let (paddedBytesPerRow, paddingOverflow) =
            minimumBytesPerRow.addingReportingOverflow(15)
        let bytesPerRow = paddedBytesPerRow & ~15
        let (byteCount, byteCountOverflow) =
            bytesPerRow.multipliedReportingOverflow(by: height)
        guard !rowBitsOverflow,
              !roundingOverflow,
              !paddingOverflow,
              !byteCountOverflow else {
            throw .pixelStorageSizeOverflow(
                width: width,
                height: height,
                bitsPerPixel: bitsPerPixel
            )
        }
        return (bytesPerRow, byteCount)
    }

    private static func drawable(_ rect: CGRect) -> Bool {
        rect.origin.x.isFinite
            && rect.origin.y.isFinite
            && rect.width.isFinite
            && rect.height.isFinite
            && rect.width > 0
            && rect.height > 0
    }

    private static func finiteDimension(_ value: CGFloat) -> Int {
        guard value.isFinite,
              value >= CGFloat(Int.min),
              value <= CGFloat(Int.max) else {
            return Int.max
        }
        return Int(value)
    }
}
