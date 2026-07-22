import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CALayer default values")
struct CALayerDefaultValueTests {
    private func expectColor(
        _ value: Any?,
        red: CGFloat,
        green: CGFloat,
        blue: CGFloat,
        alpha: CGFloat,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        guard let color = value as? CGColor,
              let components = color.components,
              components.count >= 4 else {
            Issue.record("Expected an RGBA color", sourceLocation: sourceLocation)
            return
        }
        #expect(abs(components[0] - red) < 0.000_001, sourceLocation: sourceLocation)
        #expect(abs(components[1] - green) < 0.000_001, sourceLocation: sourceLocation)
        #expect(abs(components[2] - blue) < 0.000_001, sourceLocation: sourceLocation)
        #expect(abs(components[3] - alpha) < 0.000_001, sourceLocation: sourceLocation)
    }

    @Test("Base geometry and contents defaults match QuartzCore")
    func baseGeometryAndContentsDefaults() {
        #expect(CALayer.defaultValue(forKey: "anchorPoint") as? CGPoint == CGPoint(x: 0.5, y: 0.5))
        #expect(CALayer.defaultValue(forKey: "contentsRect") as? CGRect == CGRect(x: 0, y: 0, width: 1, height: 1))
        #expect(CALayer.defaultValue(forKey: "contentsCenter") as? CGRect == CGRect(x: 0, y: 0, width: 1, height: 1))
        #expect(CALayer.defaultValue(forKey: "contentsGravity") as? CALayerContentsGravity == .resize)
        #expect(CALayer.defaultValue(forKey: "contentsScale") as? CGFloat == 1)
        #expect(CALayer.defaultValue(forKey: "contentsFormat") as? CALayerContentsFormat == .RGBA8Uint)
        #expect(CALayer.defaultValue(forKey: "minificationFilter") as? CALayerContentsFilter == .linear)
        #expect(CALayer.defaultValue(forKey: "magnificationFilter") as? CALayerContentsFilter == .linear)

        let corners = CALayer.defaultValue(forKey: "maskedCorners") as? CACornerMask
        #expect(corners?.rawValue == 15)
        #expect(CALayer.defaultValue(forKey: "cornerCurve") as? CALayerCornerCurve == .circular)
    }

    @Test("Base appearance, rendering, and timing defaults match QuartzCore")
    func baseAppearanceRenderingAndTimingDefaults() {
        #expect(CALayer.defaultValue(forKey: "opacity") as? Float == 1)
        #expect(CALayer.defaultValue(forKey: "hidden") as? Bool == false)
        #expect(CALayer.defaultValue(forKey: "doubleSided") as? Bool == true)
        #expect(CALayer.defaultValue(forKey: "masksToBounds") as? Bool == false)
        #expect(CALayer.defaultValue(forKey: "geometryFlipped") as? Bool == false)
        #expect(CALayer.defaultValue(forKey: "opaque") as? Bool == false)
        #expect(CALayer.defaultValue(forKey: "allowsEdgeAntialiasing") as? Bool == true)
        #expect(CALayer.defaultValue(forKey: "allowsGroupOpacity") as? Bool == true)
        #expect(CALayer.defaultValue(forKey: "shouldRasterize") as? Bool == false)
        #expect(CALayer.defaultValue(forKey: "drawsAsynchronously") as? Bool == false)
        #expect(CALayer.defaultValue(forKey: "needsDisplayOnBoundsChange") as? Bool == false)
        #expect(CALayer.defaultValue(forKey: "edgeAntialiasingMask") as? CAEdgeAntialiasingMask == [
            .layerLeftEdge, .layerRightEdge, .layerBottomEdge, .layerTopEdge
        ])
        #expect(CALayer.defaultValue(forKey: "rasterizationScale") as? CGFloat == 1)
        #expect(CALayer.defaultValue(forKey: "shadowOffset") as? CGSize == CGSize(width: 0, height: -3))
        #expect(CALayer.defaultValue(forKey: "shadowRadius") as? CGFloat == 3)
        #expect(CALayer.defaultValue(forKey: "duration") as? CFTimeInterval == .infinity)
        #expect(CALayer.defaultValue(forKey: "speed") as? Float == 1)
        #expect(CALayer.defaultValue(forKey: "fillMode") as? CAMediaTimingFillMode == .removed)
        expectColor(CALayer.defaultValue(forKey: "borderColor"), red: 0, green: 0, blue: 0, alpha: 1)
        expectColor(CALayer.defaultValue(forKey: "shadowColor"), red: 0, green: 0, blue: 0, alpha: 1)
    }

    @Test("Layer instances use the same nonzero defaults")
    func layerInstanceDefaults() {
        let layer = CALayer()

        #expect(layer.allowsEdgeAntialiasing)
        #expect(layer.duration == .infinity)
        expectColor(layer.borderColor, red: 0, green: 0, blue: 0, alpha: 1)
    }

    @Test("Known zero and unknown keys remain nil when QuartzCore has no stored default")
    func nilDefaultsRemainNil() {
        #expect(CALayer.defaultValue(forKey: "bounds") == nil)
        #expect(CALayer.defaultValue(forKey: "position") == nil)
        #expect(CALayer.defaultValue(forKey: "shadowOpacity") == nil)
        #expect(CALayer.defaultValue(forKey: "timeOffset") == nil)
        #expect(CALayer.defaultValue(forKey: "isHidden") == nil)
        #expect(CALayer.defaultValue(forKey: "unknownProperty") == nil)
    }

    @Test("Shape-layer defaults match QuartzCore")
    func shapeLayerDefaults() {
        expectColor(CAShapeLayer.defaultValue(forKey: "fillColor"), red: 0, green: 0, blue: 0, alpha: 1)
        #expect(CAShapeLayer.defaultValue(forKey: "fillRule") as? CAShapeLayerFillRule == .nonZero)
        #expect(CAShapeLayer.defaultValue(forKey: "strokeEnd") as? CGFloat == 1)
        #expect(CAShapeLayer.defaultValue(forKey: "lineWidth") as? CGFloat == 1)
        #expect(CAShapeLayer.defaultValue(forKey: "miterLimit") as? CGFloat == 10)
        #expect(CAShapeLayer.defaultValue(forKey: "lineCap") as? CAShapeLayerLineCap == .butt)
        #expect(CAShapeLayer.defaultValue(forKey: "lineJoin") as? CAShapeLayerLineJoin == .miter)
        #expect(CAShapeLayer.defaultValue(forKey: "strokeStart") == nil)
    }

    @Test("Gradient-layer defaults match QuartzCore")
    func gradientLayerDefaults() {
        #expect(CAGradientLayer.defaultValue(forKey: "startPoint") as? CGPoint == CGPoint(x: 0.5, y: 0))
        #expect(CAGradientLayer.defaultValue(forKey: "endPoint") as? CGPoint == CGPoint(x: 0.5, y: 1))
        #expect(CAGradientLayer.defaultValue(forKey: "type") as? CAGradientLayerType == .axial)
        #expect(CAGradientLayer.defaultValue(forKey: "opacity") as? Float == 1)
    }

    @Test("Replicator and emitter defaults match QuartzCore")
    func replicatorAndEmitterDefaults() {
        #expect(CAReplicatorLayer.defaultValue(forKey: "instanceCount") as? Int == 1)
        expectColor(CAReplicatorLayer.defaultValue(forKey: "instanceColor"), red: 1, green: 1, blue: 1, alpha: 1)
        #expect(CAReplicatorLayer.defaultValue(forKey: "instanceDelay") == nil)

        #expect(CAEmitterLayer.defaultValue(forKey: "birthRate") as? Float == 1)
        #expect(CAEmitterLayer.defaultValue(forKey: "lifetime") as? Float == 1)
        #expect(CAEmitterLayer.defaultValue(forKey: "velocity") as? Float == 1)
        #expect(CAEmitterLayer.defaultValue(forKey: "scale") as? Float == 1)
        #expect(CAEmitterLayer.defaultValue(forKey: "spin") as? Float == 1)
        #expect(CAEmitterLayer.defaultValue(forKey: "emitterShape") as? CAEmitterLayerEmitterShape == .point)
        #expect(CAEmitterLayer.defaultValue(forKey: "emitterMode") as? CAEmitterLayerEmitterMode == .volume)
        #expect(CAEmitterLayer.defaultValue(forKey: "renderMode") as? CAEmitterLayerRenderMode == .unordered)
    }

    @Test("Text-layer defaults match QuartzCore")
    func textLayerDefaults() {
        #expect(CATextLayer.defaultValue(forKey: "font") as? String == "Helvetica")
        #expect(CATextLayer.defaultValue(forKey: "fontSize") as? CGFloat == 36)
        expectColor(CATextLayer.defaultValue(forKey: "foregroundColor"), red: 1, green: 1, blue: 1, alpha: 1)
        #expect(
            CATextLayer.defaultValue(forKey: "truncationMode") as? CATextLayerTruncationMode
                == CATextLayerTruncationMode.none
        )
        #expect(CATextLayer.defaultValue(forKey: "alignmentMode") as? CATextLayerAlignmentMode == .natural)
        #expect(CATextLayer().font as? String == "Helvetica")
    }

    @Test("Tiled and scroll layer defaults match QuartzCore")
    func tiledAndScrollLayerDefaults() {
        #expect(CATiledLayer.defaultValue(forKey: "levelsOfDetail") as? Int == 1)
        #expect(CATiledLayer.defaultValue(forKey: "levelsOfDetailBias") as? Int == 0)
        #expect(CATiledLayer.defaultValue(forKey: "tileSize") as? CGSize == CGSize(width: 256, height: 256))
        #expect(CAScrollLayer.defaultValue(forKey: "scrollMode") as? CAScrollLayerScrollMode == .both)
    }
}
