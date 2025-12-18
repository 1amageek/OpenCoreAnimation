import JavaScriptKit
import OpenCoreAnimation
import OpenCoreGraphics

// MARK: - Application Entry Point

/// Basic animation demo showcasing OpenCoreAnimation capabilities.
///
/// This example demonstrates:
/// - Layer hierarchy creation
/// - Basic animations (opacity, position, transform)
/// - Gradient layers
/// - Shape layers with paths
/// - Animation engine integration

@main
struct BasicAnimationApp {
    static func main() async throws {
        print("OpenCoreAnimation BasicAnimation Demo Starting...")

        // Get the document and create a canvas
        let document = JSObject.global.document
        let canvas = document.createElement("canvas")
        canvas.id = "animation-canvas"
        canvas.width = 800
        canvas.height = 600
        canvas.style.border = "1px solid #333"
        _ = document.body.appendChild(canvas)

        // Create info display
        let info = document.createElement("div")
        info.id = "info"
        info.innerHTML = """
            <h2>OpenCoreAnimation Demo</h2>
            <p>Demonstrating: Layer hierarchy, animations, gradients, and shapes</p>
        """
        _ = document.body.appendChild(info)

        // Initialize the renderer
        do {
            let renderer = try await CAWebGPURenderer(canvas: canvas.object!)
            await runDemo(renderer: renderer, canvas: canvas.object!)
        } catch {
            print("Failed to initialize renderer: \(error)")
            let errorDiv = document.createElement("div")
            errorDiv.style.color = "red"
            errorDiv.innerHTML = "Error: WebGPU not supported or initialization failed"
            _ = document.body.appendChild(errorDiv)
        }
    }

    static func runDemo(renderer: CAWebGPURenderer, canvas: JSObject) async {
        // Create root layer
        let rootLayer = CALayer()
        rootLayer.bounds = CGRect(x: 0, y: 0, width: 800, height: 600)
        rootLayer.position = CGPoint(x: 400, y: 300)
        rootLayer.backgroundColor = CGColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1.0)

        // Create animated box layer
        let boxLayer = CALayer()
        boxLayer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
        boxLayer.position = CGPoint(x: 150, y: 150)
        boxLayer.backgroundColor = CGColor(red: 0.2, green: 0.6, blue: 1.0, alpha: 1.0)
        boxLayer.cornerRadius = 10
        boxLayer.borderWidth = 2
        boxLayer.borderColor = CGColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.5)
        rootLayer.addSublayer(boxLayer)

        // Create gradient layer
        let gradientLayer = CAGradientLayer()
        gradientLayer.bounds = CGRect(x: 0, y: 0, width: 150, height: 150)
        gradientLayer.position = CGPoint(x: 400, y: 150)
        gradientLayer.colors = [
            CGColor(red: 1.0, green: 0.4, blue: 0.4, alpha: 1.0),
            CGColor(red: 1.0, green: 0.8, blue: 0.2, alpha: 1.0),
            CGColor(red: 0.4, green: 1.0, blue: 0.4, alpha: 1.0)
        ]
        gradientLayer.startPoint = CGPoint(x: 0, y: 0)
        gradientLayer.endPoint = CGPoint(x: 1, y: 1)
        gradientLayer.cornerRadius = 20
        rootLayer.addSublayer(gradientLayer)

        // Create shape layer with a star path
        let shapeLayer = CAShapeLayer()
        shapeLayer.bounds = CGRect(x: 0, y: 0, width: 120, height: 120)
        shapeLayer.position = CGPoint(x: 650, y: 150)
        shapeLayer.path = createStarPath(center: CGPoint(x: 60, y: 60), radius: 50, points: 5)
        shapeLayer.fillColor = CGColor(red: 1.0, green: 0.8, blue: 0.0, alpha: 1.0)
        shapeLayer.strokeColor = CGColor(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0)
        shapeLayer.lineWidth = 3
        rootLayer.addSublayer(shapeLayer)

        // Create text layer
        let textLayer = CATextLayer()
        textLayer.bounds = CGRect(x: 0, y: 0, width: 200, height: 50)
        textLayer.position = CGPoint(x: 150, y: 320)
        textLayer.string = "Hello WASM!"
        textLayer.fontSize = 24
        textLayer.foregroundColor = CGColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        textLayer.backgroundColor = CGColor(red: 0.2, green: 0.2, blue: 0.3, alpha: 0.8)
        textLayer.alignmentMode = .center
        textLayer.cornerRadius = 8
        rootLayer.addSublayer(textLayer)

        // Create replicator layer with rotating copies
        let replicatorLayer = CAReplicatorLayer()
        replicatorLayer.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)
        replicatorLayer.position = CGPoint(x: 400, y: 420)
        replicatorLayer.instanceCount = 8
        replicatorLayer.instanceTransform = CATransform3DMakeRotation(CGFloat.pi * 2 / 8, 0, 0, 1)
        replicatorLayer.instanceAlphaOffset = -0.1
        rootLayer.addSublayer(replicatorLayer)

        // Add a small rectangle to be replicated
        let replicatedLayer = CALayer()
        replicatedLayer.bounds = CGRect(x: 0, y: 0, width: 20, height: 60)
        replicatedLayer.position = CGPoint(x: 100, y: 30)
        replicatedLayer.backgroundColor = CGColor(red: 0.4, green: 0.8, blue: 1.0, alpha: 1.0)
        replicatedLayer.cornerRadius = 5
        replicatorLayer.addSublayer(replicatedLayer)

        // Create rounded rectangle
        let roundedRect = CALayer()
        roundedRect.bounds = CGRect(x: 0, y: 0, width: 150, height: 60)
        roundedRect.position = CGPoint(x: 650, y: 320)
        roundedRect.backgroundColor = CGColor(red: 0.2, green: 0.8, blue: 0.6, alpha: 1.0)
        roundedRect.cornerRadius = 15
        roundedRect.borderWidth = 2
        roundedRect.borderColor = CGColor(red: 0.1, green: 0.5, blue: 0.4, alpha: 1.0)
        rootLayer.addSublayer(roundedRect)

        // Create shape layer with bezier curve
        let curveLayer = CAShapeLayer()
        curveLayer.bounds = CGRect(x: 0, y: 0, width: 150, height: 100)
        curveLayer.position = CGPoint(x: 650, y: 450)
        curveLayer.path = createCurvePath()
        curveLayer.fillColor = nil
        curveLayer.strokeColor = CGColor(red: 0.4, green: 0.8, blue: 1.0, alpha: 1.0)
        curveLayer.lineWidth = 4
        curveLayer.lineCap = .round
        rootLayer.addSublayer(curveLayer)

        // Create circle layer
        let circleLayer = CALayer()
        circleLayer.bounds = CGRect(x: 0, y: 0, width: 60, height: 60)
        circleLayer.position = CGPoint(x: 150, y: 450)
        circleLayer.backgroundColor = CGColor(red: 0.8, green: 0.2, blue: 0.8, alpha: 1.0)
        circleLayer.cornerRadius = 30
        rootLayer.addSublayer(circleLayer)

        // Add animations
        addAnimations(to: boxLayer, gradientLayer: gradientLayer, shapeLayer: shapeLayer,
                      circleLayer: circleLayer, roundedRect: roundedRect)

        // Setup animation engine
        let engine = CAAnimationEngine.shared
        engine.rootLayer = rootLayer
        engine.renderer = renderer

        print("Starting animation loop...")
        engine.start()

        // Keep the program running
        // The animation loop will continue via requestAnimationFrame
    }

    // MARK: - Animation Setup

    static func addAnimations(to boxLayer: CALayer, gradientLayer: CAGradientLayer,
                              shapeLayer: CAShapeLayer, circleLayer: CALayer,
                              roundedRect: CALayer) {
        // Box layer: position animation (bounce)
        let positionAnim = CABasicAnimation(keyPath: "position")
        positionAnim.fromValue = CGPoint(x: 150, y: 150)
        positionAnim.toValue = CGPoint(x: 150, y: 250)
        positionAnim.duration = 1.0
        positionAnim.autoreverses = true
        positionAnim.repeatCount = .infinity
        positionAnim.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        boxLayer.add(positionAnim, forKey: "bounce")

        // Box layer: opacity animation
        let opacityAnim = CABasicAnimation(keyPath: "opacity")
        opacityAnim.fromValue = Float(1.0)
        opacityAnim.toValue = Float(0.5)
        opacityAnim.duration = 0.5
        opacityAnim.autoreverses = true
        opacityAnim.repeatCount = .infinity
        boxLayer.add(opacityAnim, forKey: "fade")

        // Gradient layer: rotation animation
        let rotationAnim = CABasicAnimation(keyPath: "transform.rotation.z")
        rotationAnim.fromValue = CGFloat(0)
        rotationAnim.toValue = CGFloat.pi * 2
        rotationAnim.duration = 4.0
        rotationAnim.repeatCount = .infinity
        gradientLayer.add(rotationAnim, forKey: "spin")

        // Shape layer: scale animation
        let scaleAnim = CABasicAnimation(keyPath: "transform.scale")
        scaleAnim.fromValue = CGFloat(1.0)
        scaleAnim.toValue = CGFloat(1.2)
        scaleAnim.duration = 0.8
        scaleAnim.autoreverses = true
        scaleAnim.repeatCount = .infinity
        scaleAnim.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        shapeLayer.add(scaleAnim, forKey: "pulse")

        // Circle layer: horizontal movement
        let moveAnim = CABasicAnimation(keyPath: "position.x")
        moveAnim.fromValue = CGFloat(150)
        moveAnim.toValue = CGFloat(250)
        moveAnim.duration = 1.5
        moveAnim.autoreverses = true
        moveAnim.repeatCount = .infinity
        moveAnim.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        circleLayer.add(moveAnim, forKey: "slide")

        // Rounded rect: transform animation
        let transformAnim = CABasicAnimation(keyPath: "transform")
        transformAnim.fromValue = CATransform3DIdentity
        transformAnim.toValue = CATransform3DMakeScale(1.1, 1.1, 1.0)
        transformAnim.duration = 1.2
        transformAnim.autoreverses = true
        transformAnim.repeatCount = .infinity
        transformAnim.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        roundedRect.add(transformAnim, forKey: "grow")
    }

    // MARK: - Path Creation

    static func createStarPath(center: CGPoint, radius: CGFloat, points: Int) -> CGMutablePath {
        let path = CGMutablePath()
        let angleIncrement = CGFloat.pi * 2 / CGFloat(points * 2)
        let innerRadius = radius * 0.4

        for i in 0..<(points * 2) {
            let r = (i % 2 == 0) ? radius : innerRadius
            let angle = CGFloat(i) * angleIncrement - CGFloat.pi / 2

            let x = center.x + r * cos(angle)
            let y = center.y + r * sin(angle)

            if i == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        path.closeSubpath()
        return path
    }

    static func createCurvePath() -> CGMutablePath {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 10, y: 80))
        path.addCurve(to: CGPoint(x: 140, y: 80),
                      control1: CGPoint(x: 40, y: 10),
                      control2: CGPoint(x: 110, y: 90))
        return path
    }
}
