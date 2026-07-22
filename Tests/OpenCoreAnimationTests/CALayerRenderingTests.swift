import Foundation
import Testing
@testable import OpenCoreAnimation
@testable import OpenCoreGraphics

private final class RecordingRenderer: CGContextStatefulRendererDelegate, @unchecked Sendable {
    struct ImageDrawCall {
        let image: CGImage
        let rect: CGRect
        let blendMode: CGBlendMode
        let clipPaths: [CGClipPath]
    }

    struct FillCall {
        let path: CGPath
        let color: CGColor
        let clipPaths: [CGClipPath]
    }

    var imageDrawCalls: [ImageDrawCall] = []
    var fillCalls: [FillCall] = []
    var transparencyLayerBeginCount = 0
    var transparencyLayerEndCount = 0

    func fill(
        path: CGPath,
        color: CGColor,
        alpha: CGFloat,
        blendMode: CGBlendMode,
        rule: CGPathFillRule
    ) {}

    func stroke(
        path: CGPath,
        color: CGColor,
        lineWidth: CGFloat,
        lineCap: CGLineCap,
        lineJoin: CGLineJoin,
        miterLimit: CGFloat,
        dashPhase: CGFloat,
        dashLengths: [CGFloat],
        alpha: CGFloat,
        blendMode: CGBlendMode
    ) {}

    func draw(
        image: CGImage,
        in rect: CGRect,
        alpha: CGFloat,
        blendMode: CGBlendMode,
        interpolationQuality: CGInterpolationQuality
    ) {}

    func fill(
        path: CGPath,
        color: CGColor,
        alpha: CGFloat,
        blendMode: CGBlendMode,
        rule: CGPathFillRule,
        state: CGDrawingState
    ) {
        fillCalls.append(FillCall(path: path, color: color, clipPaths: state.clipPaths))
    }

    func draw(
        image: CGImage,
        in rect: CGRect,
        alpha: CGFloat,
        blendMode: CGBlendMode,
        interpolationQuality: CGInterpolationQuality,
        state: CGDrawingState
    ) {
        imageDrawCalls.append(ImageDrawCall(image: image, rect: rect, blendMode: blendMode, clipPaths: state.clipPaths))
    }

    func beginTransparencyLayer(in rect: CGRect?, auxiliaryInfo: [String: Any]?, state: CGDrawingState) {
        transparencyLayerBeginCount += 1
    }

    func endTransparencyLayer(alpha: CGFloat, blendMode: CGBlendMode, state: CGDrawingState) {
        transparencyLayerEndCount += 1
    }
}

@Suite("CALayer Rendering Tests")
struct CALayerRenderingTests {
    private func makeContext(renderer: RecordingRenderer, width: Int = 64, height: Int = 64) -> CGContext {
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: .deviceRGB,
            bitmapInfo: bitmapInfo
        ) else {
            fatalError("Expected bitmap context")
        }
        context.rendererDelegate = renderer
        return context
    }

    private func makeImage(width: Int, height: Int) -> CGImage {
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let bytesPerRow = width * 4
        let pixelData = Data(repeating: 255, count: bytesPerRow * height)
        let provider = CGDataProvider(data: pixelData)

        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: .deviceRGB,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            fatalError("Expected test image")
        }

        return image
    }

    private func curveElementCount(in path: CGPath) -> Int {
        var count = 0
        path.applyWithBlock { element in
            if element.pointee.type == .addCurveToPoint {
                count += 1
            }
        }
        return count
    }

    @Test("contentsRect crops the source image before drawing")
    func contentsRectCropsSourceImage() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 20, height: 10)
        layer.contents = makeImage(width: 4, height: 2)
        layer.contentsRect = CGRect(x: 0.5, y: 0, width: 0.5, height: 1)

        layer.render(in: context)

        #expect(renderer.imageDrawCalls.count == 1)
        guard let drawCall = renderer.imageDrawCalls.first else { return }
        #expect(drawCall.image.width == 2)
        #expect(drawCall.image.height == 2)
        #expect(drawCall.rect == layer.bounds)
    }

    @Test("contentsCenter uses nine-slice scaling after contentsRect is applied")
    func contentsCenterUsesNineSliceScaling() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        layer.contents = makeImage(width: 8, height: 4)
        layer.contentsGravity = .resize
        layer.contentsRect = CGRect(x: 0.5, y: 0, width: 0.5, height: 1)
        layer.contentsCenter = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)

        layer.render(in: context)

        #expect(renderer.imageDrawCalls.count == 9)

        let centerRect = CGRect(x: 1, y: 1, width: 6, height: 6)
        let centerCall = renderer.imageDrawCalls.first { $0.rect == centerRect }
        #expect(centerCall?.image.width == 2)
        #expect(centerCall?.image.height == 2)

        let topEdgeRect = CGRect(x: 1, y: 0, width: 6, height: 1)
        let topEdgeCall = renderer.imageDrawCalls.first { $0.rect == topEdgeRect }
        #expect(topEdgeCall?.image.width == 2)
        #expect(topEdgeCall?.image.height == 1)

        let rightEdgeRect = CGRect(x: 7, y: 1, width: 1, height: 6)
        let rightEdgeCall = renderer.imageDrawCalls.first { $0.rect == rightEdgeRect }
        #expect(rightEdgeCall?.image.width == 1)
        #expect(rightEdgeCall?.image.height == 2)
    }

    @Test("contentsCenter is ignored when contentsGravity does not resize")
    func contentsCenterRequiresResizingGravity() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        layer.contents = makeImage(width: 4, height: 4)
        layer.contentsGravity = .center
        layer.contentsCenter = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)

        layer.render(in: context)

        #expect(renderer.imageDrawCalls.count == 1)
        guard let drawCall = renderer.imageDrawCalls.first else { return }
        #expect(drawCall.rect == CGRect(x: 2, y: 2, width: 4, height: 4))
    }

    @Test("masksToBounds pushes a clip path into the renderer state")
    func masksToBoundsAddsClipPathToRendererState() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        layer.masksToBounds = true

        layer.render(in: context)

        #expect(renderer.fillCalls.count == 1)
        guard let fillCall = renderer.fillCalls.first else { return }
        #expect(fillCall.clipPaths.count == 1)
    }

    @Test("maskedCorners can disable rounding for the native render path")
    func maskedCornersCanDisableCornerRounding() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        layer.cornerRadius = 4
        layer.maskedCorners = []

        layer.render(in: context)

        #expect(renderer.fillCalls.count == 1)
        guard let fillCall = renderer.fillCalls.first else { return }
        #expect(curveElementCount(in: fillCall.path) == 0)
    }

    @Test("maskedCorners applies selective corner rounding for clipping")
    func maskedCornersAppliesSelectiveCornerRoundingForClipping() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        layer.cornerRadius = 4
        layer.maskedCorners = [.layerMinXMinYCorner]
        layer.masksToBounds = true

        layer.render(in: context)

        #expect(renderer.fillCalls.count == 1)
        guard let clipPath = renderer.fillCalls.first?.clipPaths.first else { return }
        #expect(curveElementCount(in: clipPath.path) == 1)
        #expect(clipPath.rule == .winding)
    }

    @Test("continuous corner curves preserve more corner area than circular curves")
    func continuousCornerCurveChangesRenderedGeometry() {
        func renderedPath(for curve: CALayerCornerCurve) -> CGPath? {
            let renderer = RecordingRenderer()
            let context = makeContext(renderer: renderer)
            let layer = CALayer()
            layer.bounds = CGRect(x: 0, y: 0, width: 60, height: 60)
            layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
            layer.cornerRadius = 30
            layer.cornerCurve = curve
            layer.render(in: context)
            return renderer.fillCalls.first?.path
        }

        let discriminatingPoint = CGPoint(x: 4, y: 14)
        let circularPath = renderedPath(for: .circular)
        let continuousPath = renderedPath(for: .continuous)

        #expect(circularPath?.contains(discriminatingPoint) == false)
        #expect(continuousPath?.contains(discriminatingPoint) == true)
        guard let continuousPath else { return }
        #expect(curveElementCount(in: continuousPath) == 64)
    }

    @Test("unsupported corner curves do not render a fallback shape")
    func unsupportedCornerCurveDoesNotRenderFallbackShape() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 60, height: 60)
        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        layer.cornerRadius = 30
        layer.cornerCurve = CALayerCornerCurve(rawValue: "future-curve")

        layer.render(in: context)

        #expect(renderer.fillCalls.isEmpty)
    }

    @Test("mask renders through a transparency layer using destinationIn blending")
    func maskUsesTransparencyLayerAndDestinationInBlendMode() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        layer.contents = makeImage(width: 4, height: 4)

        let maskLayer = CALayer()
        maskLayer.frame = layer.bounds
        maskLayer.contents = makeImage(width: 4, height: 4)
        layer.mask = maskLayer

        layer.render(in: context)

        #expect(renderer.transparencyLayerBeginCount == 1)
        #expect(renderer.transparencyLayerEndCount == 1)
        #expect(renderer.imageDrawCalls.count == 2)
        #expect(renderer.imageDrawCalls.last?.blendMode == .destinationIn)
    }
}
