import Foundation
import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation
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
        let transform: CGAffineTransform
        let alpha: CGFloat
    }

    var imageDrawCalls: [ImageDrawCall] = []
    var fillCalls: [FillCall] = []
    var transparencyLayerBeginCount = 0
    var transparencyLayerEndCount = 0
    var transparencyLayerEndAlphas: [CGFloat] = []
    var transparencyLayerEndBlendModes: [CGBlendMode] = []

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
        fillCalls.append(FillCall(
            path: path,
            color: color,
            clipPaths: state.clipPaths,
            transform: state.ctm,
            alpha: alpha
        ))
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
        transparencyLayerEndAlphas.append(alpha)
        transparencyLayerEndBlendModes.append(blendMode)
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

    private func makeSoftwareContext(width: Int, height: Int) throws -> CGContext {
        try #require(CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: .deviceRGB,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        ))
    }

    private func pixel(in context: CGContext, x: Int, y: Int) throws -> [UInt8] {
        let image = try #require(context.makeImage())
        let data = try #require(image.data ?? image.dataProvider?.data)
        let offset = y * image.bytesPerRow + x * 4
        return Array(data[offset..<(offset + 4)])
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

    @Test("contentsScale converts cropped image pixels to points")
    func contentsScaleControlsCroppedDestinationSize() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)

        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        layer.contents = makeImage(width: 4, height: 4)
        layer.contentsGravity = .center
        layer.contentsRect = CGRect(x: 0.5, y: 0, width: 0.5, height: 1)
        layer.contentsScale = 2

        layer.render(in: context)

        #expect(renderer.imageDrawCalls.count == 1)
        #expect(renderer.imageDrawCalls.first?.rect == CGRect(
            x: 3.5,
            y: 3,
            width: 1,
            height: 2
        ))
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
        #expect(layer.lastContextRenderError == .unsupportedCornerCurve("future-curve"))
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

        #expect(renderer.transparencyLayerBeginCount == 2)
        #expect(renderer.transparencyLayerEndCount == 2)
        #expect(renderer.imageDrawCalls.count == 2)
        #expect(renderer.imageDrawCalls.last?.blendMode == .normal)
        #expect(renderer.transparencyLayerEndBlendModes.first == .destinationIn)
    }

    @Test("render ignores the root layer transform in its own coordinate space")
    func rootTransformIsIgnored() throws {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 4, height: 4)
        layer.position = CGPoint(x: 20, y: 20)
        layer.transform = CATransform3DMakeTranslation(10, 12, 0)
        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)

        layer.render(in: context)

        let call = try #require(renderer.fillCalls.first)
        #expect(call.path.boundingBox.applying(call.transform) == layer.bounds)
    }

    @Test("sublayer geometry applies position transform anchor and bounds origin in order")
    func sublayerGeometryComposition() throws {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)

        let child = CALayer()
        child.bounds = CGRect(x: 3, y: 4, width: 4, height: 4)
        child.position = CGPoint(x: 10, y: 10)
        child.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        child.transform = CATransform3DMakeRotation(.pi / 2, 0, 0, 1)
        child.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        parent.addSublayer(child)

        parent.render(in: context)

        let call = try #require(renderer.fillCalls.first)
        let renderedBounds = call.path.boundingBox
        #expect(abs(renderedBounds.minX - 8) < 0.001)
        #expect(abs(renderedBounds.minY - 8) < 0.001)
        #expect(abs(renderedBounds.width - 4) < 0.001)
        #expect(abs(renderedBounds.height - 4) < 0.001)
    }

    @Test("sublayer geometry reaches the expected software pixels")
    func sublayerGeometrySoftwarePixels() throws {
        let context = try makeSoftwareContext(width: 30, height: 30)
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)

        let child = CALayer()
        child.bounds = CGRect(x: 3, y: 4, width: 4, height: 4)
        child.position = CGPoint(x: 10, y: 10)
        child.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        child.transform = CATransform3DMakeRotation(.pi / 2, 0, 0, 1)
        child.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        parent.addSublayer(child)

        parent.render(in: context)

        #expect(try pixel(in: context, x: 9, y: 9) == [255, 0, 0, 255])
        #expect(try pixel(in: context, x: 13, y: 13) == [0, 0, 0, 0])
    }

    @Test("sublayerTransform rotates descendants around the parent anchor")
    func sublayerTransformUsesParentAnchor() throws {
        let context = try makeSoftwareContext(width: 40, height: 40)
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)
        parent.sublayerTransform = CATransform3DMakeRotation(.pi / 2, 0, 0, 1)

        let child = CALayer()
        child.bounds = CGRect(x: 0, y: 0, width: 4, height: 4)
        child.position = CGPoint(x: 10, y: 10)
        child.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        child.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        parent.addSublayer(child)

        parent.render(in: context)

        #expect(try pixel(in: context, x: 19, y: 9) == [255, 0, 0, 255])
        #expect(try pixel(in: context, x: 9, y: 9) == [0, 0, 0, 0])
    }

    @Test("layer opacity composites the complete subtree as one group")
    func opacityUsesTransparencyGroup() throws {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)
        let parent = CALayer()
        parent.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        parent.opacity = 0.5

        let child = CALayer()
        child.frame = parent.bounds
        child.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        parent.addSublayer(child)

        parent.render(in: context)

        #expect(renderer.transparencyLayerBeginCount == 1)
        #expect(renderer.transparencyLayerEndCount == 1)
        #expect(try #require(renderer.transparencyLayerEndAlphas.first) == 0.5)
        #expect(try #require(renderer.fillCalls.first).alpha == 1)
    }

    @Test("CAShapeLayer render draws fill and trimmed stroke content")
    func shapeLayerRendersSpecializedContent() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)
        let shape = CAShapeLayer()
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 2, y: 8))
        path.addLine(to: CGPoint(x: 18, y: 8))
        shape.path = path
        shape.fillColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        shape.strokeColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        shape.lineWidth = 4
        shape.strokeStart = 0.5
        shape.strokeEnd = 1

        shape.render(in: context)

        #expect(renderer.fillCalls.count == 2)
        #expect(renderer.fillCalls.last?.path.boundingBox.minX ?? 0 >= 9.9)
        #expect(shape.lastContextRenderError == nil)
    }

    @Test("CAShapeLayer render reports unknown stroke styles")
    func shapeLayerReportsUnknownStrokeStyle() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)
        let shape = CAShapeLayer()
        shape.path = CGPath(rect: CGRect(x: 0, y: 0, width: 10, height: 10), transform: nil)
        shape.fillColor = nil
        shape.strokeColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        shape.lineCap = CAShapeLayerLineCap(rawValue: "future-cap")

        shape.render(in: context)

        #expect(shape.lastContextRenderError == .unsupportedShapeLineCap("future-cap"))
        #expect(renderer.fillCalls.isEmpty)
    }

    @Test("CAShapeLayer render writes specialized content to software pixels")
    func shapeLayerSoftwarePixels() throws {
        let context = try makeSoftwareContext(width: 16, height: 16)
        let shape = CAShapeLayer()
        shape.path = CGPath(
            rect: CGRect(x: 2, y: 3, width: 8, height: 6),
            transform: nil
        )
        shape.fillColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
        shape.strokeColor = nil

        shape.render(in: context)

        #expect(try pixel(in: context, x: 5, y: 5) == [0, 255, 0, 255])
        #expect(try pixel(in: context, x: 12, y: 5) == [0, 0, 0, 0])
    }

    @Test("layer mask clears pixels outside the rendered mask buffer")
    func layerMaskSoftwarePixels() throws {
        let context = try makeSoftwareContext(width: 10, height: 10)
        let layer = CALayer()
        layer.bounds = CGRect(x: 0, y: 0, width: 10, height: 10)
        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)

        let maskLayer = CALayer()
        maskLayer.frame = CGRect(x: 0, y: 0, width: 5, height: 10)
        maskLayer.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
        layer.mask = maskLayer

        layer.render(in: context)

        #expect(try pixel(in: context, x: 2, y: 5) == [255, 0, 0, 255])
        #expect(try pixel(in: context, x: 7, y: 5) == [0, 0, 0, 0])
    }

    @Test("CAGradientLayer render draws an axial gradient in unit coordinates")
    func axialGradientSoftwarePixels() throws {
        let context = try makeSoftwareContext(width: 4, height: 1)
        let gradient = CAGradientLayer()
        gradient.bounds = CGRect(x: 0, y: 0, width: 4, height: 1)
        gradient.colors = [
            CGColor(red: 0, green: 0, blue: 0, alpha: 1),
            CGColor(red: 1, green: 1, blue: 1, alpha: 1),
        ]
        gradient.startPoint = CGPoint(x: 0, y: 0.5)
        gradient.endPoint = CGPoint(x: 1, y: 0.5)

        gradient.render(in: context)

        #expect(try pixel(in: context, x: 0, y: 0) == [32, 32, 32, 255])
        #expect(try pixel(in: context, x: 3, y: 0) == [223, 223, 223, 255])
        #expect(gradient.lastContextRenderError == nil)
    }

    @Test("CAGradientLayer render draws elliptical radial bands")
    func radialGradientSoftwarePixels() throws {
        let context = try makeSoftwareContext(width: 9, height: 9)
        let gradient = CAGradientLayer()
        gradient.bounds = CGRect(x: 0, y: 0, width: 9, height: 9)
        gradient.colors = [
            CGColor(red: 0, green: 0, blue: 0, alpha: 1),
            CGColor(red: 1, green: 1, blue: 1, alpha: 1),
        ]
        gradient.startPoint = CGPoint(x: 0.5, y: 0.5)
        gradient.endPoint = CGPoint(x: 1, y: 1)
        gradient.type = .radial

        gradient.render(in: context)

        let center = try pixel(in: context, x: 4, y: 4)
        let edge = try pixel(in: context, x: 4, y: 0)
        #expect(center[0] < 8)
        #expect(edge[0] > 180)
        #expect(gradient.lastContextRenderError == nil)
    }

    @Test("CAGradientLayer render draws conic angle progression")
    func conicGradientSoftwarePixels() throws {
        let context = try makeSoftwareContext(width: 9, height: 9)
        let gradient = CAGradientLayer()
        gradient.bounds = CGRect(x: 0, y: 0, width: 9, height: 9)
        gradient.colors = [
            CGColor(red: 1, green: 0, blue: 0, alpha: 1),
            CGColor(red: 0, green: 0, blue: 1, alpha: 1),
        ]
        gradient.startPoint = CGPoint(x: 0.5, y: 0.5)
        gradient.endPoint = CGPoint(x: 1, y: 0.5)
        gradient.type = .conic

        gradient.render(in: context)

        let right = try pixel(in: context, x: 8, y: 4)
        let top = try pixel(in: context, x: 4, y: 0)
        #expect(right[0] > right[2])
        #expect(top[2] > top[0])
        #expect(gradient.lastContextRenderError == nil)
    }

    @Test("CAGradientLayer render reports invalid public configuration")
    func gradientReportsInvalidConfiguration() {
        let renderer = RecordingRenderer()
        let context = makeContext(renderer: renderer)
        let gradient = CAGradientLayer()
        gradient.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
        gradient.colors = [
            CGColor(red: 1, green: 0, blue: 0, alpha: 1),
            "not-a-color",
        ]

        gradient.render(in: context)

        #expect(gradient.lastContextRenderError == .invalidGradientColor(index: 1))
    }
}
