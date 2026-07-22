import Foundation
import Testing
@testable import OpenCoreAnimation

@Suite("CAAnimation default values")
struct CAAnimationDefaultValueTests {
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

    @Test("Base animation stores only QuartzCore defaults")
    func baseDefaults() {
        #expect(CAAnimation.defaultValue(forKey: "speed") as? Float == 1)
        #expect(CAAnimation.defaultValue(forKey: "fillMode") as? CAMediaTimingFillMode == .removed)
        #expect(CAAnimation.defaultValue(forKey: "removedOnCompletion") as? Bool == true)

        #expect(CAAnimation.defaultValue(forKey: "duration") == nil)
        #expect(CAAnimation.defaultValue(forKey: "beginTime") == nil)
        #expect(CAAnimation.defaultValue(forKey: "autoreverses") == nil)
        #expect(CAAnimation.defaultValue(forKey: "isRemovedOnCompletion") == nil)
        #expect(CAAnimation.defaultValue(forKey: "unknownProperty") == nil)
    }

    @Test("Specialized animation defaults inherit base values")
    func specializedDefaults() {
        #expect(
            CAKeyframeAnimation.defaultValue(forKey: "calculationMode") as? CAAnimationCalculationMode
                == .linear
        )
        #expect(CASpringAnimation.defaultValue(forKey: "mass") as? CGFloat == 1)
        #expect(CASpringAnimation.defaultValue(forKey: "stiffness") as? CGFloat == 100)
        #expect(CASpringAnimation.defaultValue(forKey: "damping") as? CGFloat == 10)
        #expect(CASpringAnimation.defaultValue(forKey: "initialVelocity") == nil)
        #expect(CATransition.defaultValue(forKey: "type") as? CATransitionType == .fade)
        #expect(CATransition.defaultValue(forKey: "endProgress") as? Float == 1)
        #expect(CATransition.defaultValue(forKey: "startProgress") == nil)
        #expect(CATransition.defaultValue(forKey: "speed") as? Float == 1)
    }

    @Test("Emitter cell defaults match stored QuartzCore values")
    func emitterCellDefaults() {
        #expect(CAEmitterCell.defaultValue(forKey: "enabled") as? Bool == true)
        #expect(CAEmitterCell.defaultValue(forKey: "isEnabled") == nil)
        #expect(CAEmitterCell.defaultValue(forKey: "contentsRect") as? CGRect == CGRect(x: 0, y: 0, width: 1, height: 1))
        #expect(CAEmitterCell.defaultValue(forKey: "contentsScale") as? CGFloat == 1)
        #expect(CAEmitterCell.defaultValue(forKey: "scale") as? CGFloat == 1)
        #expect(CAEmitterCell.defaultValue(forKey: "duration") as? CFTimeInterval == .infinity)
        #expect(CAEmitterCell.defaultValue(forKey: "speed") as? Float == 1)
        #expect(CAEmitterCell.defaultValue(forKey: "fillMode") as? CAMediaTimingFillMode == .removed)
        expectColor(CAEmitterCell.defaultValue(forKey: "color"), red: 1, green: 1, blue: 1, alpha: 1)

        #expect(CAEmitterCell.defaultValue(forKey: "birthRate") == nil)
        #expect(CAEmitterCell.defaultValue(forKey: "velocity") == nil)
        #expect(CAEmitterCell.defaultValue(forKey: "beginTime") == nil)
        #expect(CAEmitterCell.defaultValue(forKey: "unknownProperty") == nil)
    }

    @Test("Emitter cell instances expose nonzero defaults and style storage")
    func emitterCellInstanceDefaults() {
        let cell = CAEmitterCell()
        cell.style = ["purpose": "spark"]

        #expect(cell.isEnabled)
        #expect(cell.scale == 1)
        #expect(cell.duration == .infinity)
        #expect(cell.contentsRect == CGRect(x: 0, y: 0, width: 1, height: 1))
        #expect(cell.contentsScale == 1)
        #expect(cell.magnificationFilter == CALayerContentsFilter.linear.rawValue)
        #expect(cell.minificationFilter == CALayerContentsFilter.linear.rawValue)
        #expect(cell.style?["purpose"] as? String == "spark")
        expectColor(cell.color, red: 1, green: 1, blue: 1, alpha: 1)
    }
}
