// OpenCoreAnimation WASM smoke-test executable.
//
// Boots a `CAAnimationEngine` against a freshly created <canvas>, attaches a
// minimal `CALayer` tree (3 solid-color sublayers on a dark root), and runs
// the display link for a few frames. Success is asserted from JS through
// `window.__oca_test`, including direct readback of the submitted WebGPU
// texture so an empty render pass cannot satisfy the smoke test.
//
// The harness / reactor-ABI boot / global-init race defenses come from
// `swift-wasm-testing` — this file only contains what is specific to the
// OCA pipeline.

import Foundation
import WasmTesting
@_spi(RendererDiagnostics) import OpenCoreAnimation
import OpenCoreImage
#if canImport(Testing)
import Testing
#endif

// MARK: - Captured state (populated by performSetup)

nonisolated(unsafe) var statusText: String = "initializing"
nonisolated(unsafe) var canvasWidth: Int = 0
nonisolated(unsafe) var canvasHeight: Int = 0
nonisolated(unsafe) var sublayerCount: Int = 0
nonisolated(unsafe) var rootLayerRef: CALayer?
nonisolated(unsafe) var rasterizedGroupRef: CALayer?
nonisolated(unsafe) var tiledLayerRef: CATiledLayer?
nonisolated(unsafe) var transitioningLayerRef: CALayer?
nonisolated(unsafe) var filteredTransitioningLayerRef: CALayer?
nonisolated(unsafe) var unsupportedTransitioningLayerRef: CALayer?
nonisolated(unsafe) var tileDelegateRef: SmokeTileDelegate?
nonisolated(unsafe) var tileDrawCount: Int = 0
nonisolated(unsafe) var pixelReadbackResult: String = "pending"
nonisolated(unsafe) var transitionFilterProbeResult: String = "pending"
nonisolated(unsafe) var layerFilterProbeResult: String = "pending"
nonisolated(unsafe) var shadowProbeResult: String = "pending"
nonisolated(unsafe) var displayLinkProbeResult: String = "pending"

final class SmokeTileDelegate: CALayerDelegate {
    func draw(_ layer: CALayer, in context: CGContext) {
        tileDrawCount += 1
        context.setFillColor(CGColor(red: 1, green: 0, blue: 1, alpha: 1))
        context.fill(layer.bounds)
    }
}

@MainActor
final class DisplayLinkProbeTarget: CADisplayLinkDelegate {
    private(set) var callbackCount = 0

    func displayLinkDidFire(_ displayLink: CADisplayLink) {
        callbackCount += 1
    }
}

// MARK: - WASM entry point

@_cdecl("setup")
public func setup() {
    WasmTestingReactor.boot(
        touchGlobals: {
            // Force the lazy-once initializer for every module-scope global
            // the Task reads. Without this, `statusText` has been observed
            // reading uninitialised memory inside the first JSClosure.
            statusText = "initializing"
            canvasWidth = 0
            canvasHeight = 0
            sublayerCount = 0
            rootLayerRef = nil
            tiledLayerRef = nil
            transitioningLayerRef = nil
            filteredTransitioningLayerRef = nil
            unsupportedTransitioningLayerRef = nil
            tileDelegateRef = nil
            tileDrawCount = 0
            pixelReadbackResult = "pending"
            transitionFilterProbeResult = "pending"
            layerFilterProbeResult = "pending"
            shadowProbeResult = "pending"
            displayLinkProbeResult = "pending"
        },
        then: { await performSetup() }
    )
}

// MARK: - Scene construction

@MainActor
func performSetup() async {
    installHarness()

    let document = JSObject.global.document
    let canvas = document.createElement("canvas")
    canvas.id = "oca-canvas"
    canvas.width = .number(400)
    canvas.height = .number(300)
    canvas.style.border = "1px solid #333"
    _ = document.body.appendChild(canvas)

    canvasWidth = 400
    canvasHeight = 300

    guard let canvasObject = canvas.object else {
        statusText = "error: canvas element has no JSObject"
        return
    }

    let engine = CAAnimationEngine.shared

    do {
        try await engine.setCanvas(canvasObject)
    } catch {
        statusText = "error: setCanvas failed: \(error)"
        return
    }

    let root = CALayer()
    root.bounds = CGRect(x: 0, y: 0, width: 400, height: 300)
    root.position = CGPoint(x: 200, y: 150)
    root.backgroundColor = CGColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1.0)

    let red = CALayer()
    red.bounds = CGRect(x: 0, y: 0, width: 80, height: 80)
    red.position = CGPoint(x: 80, y: 80)
    red.backgroundColor = CGColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0)
    root.addSublayer(red)

    let green = CALayer()
    green.bounds = CGRect(x: 0, y: 0, width: 80, height: 80)
    green.position = CGPoint(x: 200, y: 80)
    green.backgroundColor = CGColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 1.0)
    root.addSublayer(green)

    let blue = CALayer()
    blue.bounds = CGRect(x: 0, y: 0, width: 80, height: 80)
    blue.position = CGPoint(x: 320, y: 80)
    blue.backgroundColor = CGColor(red: 0.0, green: 0.0, blue: 1.0, alpha: 1.0)
    blue.shouldRasterize = true
    blue.rasterizationScale = 2.0

    let inner1 = CALayer()
    inner1.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)
    inner1.position = CGPoint(x: 25, y: 25)
    inner1.backgroundColor = CGColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
    blue.addSublayer(inner1)

    let inner2 = CALayer()
    inner2.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)
    inner2.position = CGPoint(x: 55, y: 55)
    inner2.backgroundColor = CGColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 1.0)
    blue.addSublayer(inner2)

    root.addSublayer(blue)
    rasterizedGroupRef = blue

    let tileDelegate = SmokeTileDelegate()
    let tiled = CATiledLayer()
    tiled.bounds = CGRect(x: 0, y: 0, width: 80, height: 80)
    tiled.position = CGPoint(x: 200, y: 220)
    tiled.tileSize = CGSize(width: 80, height: 80)
    tiled.delegate = tileDelegate
    root.addSublayer(tiled)
    tiledLayerRef = tiled
    tileDelegateRef = tileDelegate

    let transitioning = CALayer()
    transitioning.bounds = CGRect(x: 0, y: 0, width: 80, height: 80)
    transitioning.position = CGPoint(x: 320, y: 220)
    transitioning.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 0.5)
    root.addSublayer(transitioning)
    transitioningLayerRef = transitioning

    let transition = CATransition()
    transition.type = .fade
    transition.duration = 1
    transition.speed = 0
    transition.timeOffset = 0.5
    transitioning.add(transition, forKey: "browserCrossfade")
    transitioning.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 0.5)

    let filteredTransitioning = CALayer()
    filteredTransitioning.bounds = CGRect(x: 0, y: 0, width: 80, height: 80)
    filteredTransitioning.position = CGPoint(x: 80, y: 220)
    filteredTransitioning.backgroundColor = CGColor(red: 1, green: 1, blue: 0, alpha: 1)
    root.addSublayer(filteredTransitioning)
    filteredTransitioningLayerRef = filteredTransitioning

    guard let dissolve = CIFilter(name: "CIDissolveTransition") else {
        statusText = "error: CIDissolveTransition unavailable"
        return
    }
    let filteredTransition = CATransition()
    filteredTransition.filter = dissolve
    filteredTransition.duration = 1
    filteredTransition.speed = 0
    filteredTransition.timeOffset = 0.25
    filteredTransitioning.add(filteredTransition, forKey: "browserFilteredTransition")
    filteredTransitioning.backgroundColor = CGColor(red: 0, green: 1, blue: 1, alpha: 1)

    rootLayerRef = root
    sublayerCount = root.sublayers?.count ?? 0

    engine.rootLayer = root
    engine.renderFrame()
    engine.renderFrame()
    engine.start()

    statusText = "ready"
    print("OCASmoke ready: canvas \(canvasWidth)x\(canvasHeight), sublayers \(sublayerCount)")
}

// MARK: - JS harness

@MainActor
func installHarness() {
    Harness.install(as: "__oca_test") { h in
        h.expose("getStatus", returning: { .string(statusText) })
        h.expose("getCanvasWidth", returning: { .number(Double(canvasWidth)) })
        h.expose("getCanvasHeight", returning: { .number(Double(canvasHeight)) })
        h.expose("getSublayerCount", returning: { .number(Double(sublayerCount)) })
        h.expose("isEngineRunning", returning: {
            let isRunning = MainActor.assumeIsolated {
                CAAnimationEngine.shared.isRunning
            }
            return .boolean(isRunning)
        })
        h.expose("getRasterizedGroupChildCount", returning: {
            .number(Double(rasterizedGroupRef?.sublayers?.count ?? 0))
        })
        h.expose("isRasterizedGroupEnabled", returning: {
            .boolean(rasterizedGroupRef?.shouldRasterize ?? false)
        })
        h.expose("getTileDrawCount", returning: {
            .number(Double(tileDrawCount))
        })
        h.expose("getTileState", returning: {
            let layer = tiledLayerRef
            return .string("delegate=\(layer?.delegate != nil),bounds=\(layer?.bounds.width ?? -1)x\(layer?.bounds.height ?? -1)")
        })
        h.expose("getPixelReadback", returning: {
            .string(pixelReadbackResult)
        })
        h.expose("getTransitionFilterProbeResult", returning: {
            .string(transitionFilterProbeResult)
        })
        h.expose("getLayerFilterProbeResult", returning: {
            .string(layerFilterProbeResult)
        })
        h.expose("getShadowProbeResult", returning: {
            .string(shadowProbeResult)
        })
        h.expose("getDisplayLinkProbeResult", returning: {
            .string(displayLinkProbeResult)
        })
        h.expose("getTransitionSourceCaptureCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .transitionSourceCaptureCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getTransitionTargetCaptureCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .transitionTargetCaptureCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getActiveTransitionTextureCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .activeTransitionTextureCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getTransitionFilterDispatchCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .transitionFilterDispatchCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getTransitionFilterFailureCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .transitionFilterFailureCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getActiveFilterResourceCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .activeFilterResourceCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getActiveShadowResourceCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .activeShadowResourceCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getShadowRenderFailureCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .shadowRenderFailureCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("mutateTransitionTarget", action: {
            MainActor.assumeIsolated {
                CATransaction.begin()
                CATransaction.setDisableActions(true)
                transitioningLayerRef?.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 0.5)
                CATransaction.commit()
                CAAnimationEngine.shared.renderFrame()
            }
        })
        h.expose("exerciseUnsupportedTransitionFilter", action: {
            MainActor.assumeIsolated {
                let layer = CALayer()
                layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
                layer.position = CGPoint(x: 4, y: 4)
                layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                rootLayerRef?.addSublayer(layer)

                let transition = CATransition()
                transition.filter = "unsupported-filter"
                transition.duration = 1
                transition.speed = 0
                transition.timeOffset = 0.5
                layer.add(transition, forKey: "unsupportedFilteredTransition")
                layer.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                unsupportedTransitioningLayerRef = layer
                CAAnimationEngine.shared.renderFrame()
            }
        })
        h.expose("beginTransitionFilterProbes", action: {
            Task { @MainActor in
                transitionFilterProbeResult = "running"
                let engine = CAAnimationEngine.shared
                engine.pause()
                guard let layer = filteredTransitioningLayerRef,
                      let renderer = engine.renderer as? CAWebGPURenderer else {
                    transitionFilterProbeResult = "error: probe layer or renderer unavailable"
                    return
                }

                typealias Probe = (
                    label: String,
                    filterName: String,
                    progress: CFTimeInterval,
                    configure: (CIFilter) -> Void
                )
                let probes: [Probe] = [
                    ("dissolve", "CIDissolveTransition", 0.25, { _ in }),
                    ("swipe", "CISwipeTransition", 0.36324, { filter in
                        filter.setValue(Float(0), forKey: kCIInputAngleKey)
                        filter.setValue(Float(8), forKey: "inputWidth")
                        filter.setValue(Float(1), forKey: "inputOpacity")
                        filter.setValue(CIColor(red: 0, green: 1, blue: 0), forKey: kCIInputColorKey)
                    }),
                    ("bars", "CIBarsSwipeTransition", 0.5, { filter in
                        filter.setValue(Float(0), forKey: kCIInputAngleKey)
                        filter.setValue(Float(20), forKey: "inputWidth")
                        filter.setValue(Float(0), forKey: "inputBarOffset")
                    }),
                    ("mod", "CIModTransition", 0.5, { filter in
                        filter.setValue([Float(40), Float(40)], forKey: kCIInputCenterKey)
                        filter.setValue(Float(0), forKey: kCIInputAngleKey)
                        filter.setValue(Float(10), forKey: kCIInputRadiusKey)
                        filter.setValue(Float(1), forKey: "inputCompression")
                    }),
                    ("flash", "CIFlashTransition", 0.75, { filter in
                        filter.setValue([Float(40), Float(40)], forKey: kCIInputCenterKey)
                        filter.setValue(Float(1), forKey: "inputFadeThreshold")
                        filter.setValue(CIColor(red: 0, green: 1, blue: 0), forKey: kCIInputColorKey)
                    }),
                    ("copy", "CICopyMachineTransition", 0.5, { filter in
                        filter.setValue(Float(0), forKey: kCIInputAngleKey)
                        filter.setValue(Float(20), forKey: "inputWidth")
                        filter.setValue(Float(1), forKey: "inputOpacity")
                        filter.setValue(CIColor(red: 0, green: 1, blue: 0), forKey: kCIInputColorKey)
                    }),
                    ("ripple", "CIRippleTransition", 0.5, { filter in
                        filter.setValue([Float(40), Float(40)], forKey: kCIInputCenterKey)
                        filter.setValue(Float(10), forKey: "inputWidth")
                        filter.setValue(Float(0), forKey: kCIInputScaleKey)
                    }),
                ]

                layer.removeAnimation(forKey: "browserFilteredTransition")
                var results: [String] = []
                do {
                    for probe in probes {
                        CATransaction.begin()
                        CATransaction.setDisableActions(true)
                        layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                        CATransaction.commit()
                        engine.renderFrame()

                        guard let filter = CIFilter(name: probe.filterName) else {
                            transitionFilterProbeResult = "error: \(probe.filterName) unavailable"
                            return
                        }
                        probe.configure(filter)
                        let transition = CATransition()
                        transition.filter = filter
                        transition.duration = 1
                        transition.speed = 0
                        transition.timeOffset = probe.progress
                        layer.add(transition, forKey: "activeFilterProbe")

                        CATransaction.begin()
                        CATransaction.setDisableActions(true)
                        layer.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                        CATransaction.commit()
                        engine.renderFrame()

                        let pixel = try await renderer.readbackPixel(x: 80, y: 80)
                        results.append("\(probe.label)=\(pixel.map(String.init).joined(separator: ","))")
                        layer.removeAnimation(forKey: "activeFilterProbe")
                        engine.renderFrame()
                    }
                    transitionFilterProbeResult = results.joined(separator: ";")
                } catch {
                    transitionFilterProbeResult = "error: \(error)"
                }
            }
        })
        h.expose("beginLayerFilterProbe", action: {
            Task { @MainActor in
                layerFilterProbeResult = "running"
                let engine = CAAnimationEngine.shared
                engine.pause()
                guard let root = rootLayerRef,
                      let renderer = engine.renderer as? CAWebGPURenderer else {
                    layerFilterProbeResult = "error: root layer or renderer unavailable"
                    return
                }

                let chained = CALayer()
                chained.bounds = CGRect(x: 0, y: 0, width: 70, height: 70)
                chained.position = CGPoint(x: 40, y: 40)
                chained.zPosition = 100
                chained.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                chained.filters = [CAFilter.brightness(-0.5), CAFilter.colorInvert()]
                root.addSublayer(chained)

                let sibling = CALayer()
                sibling.bounds = CGRect(x: 0, y: 0, width: 70, height: 70)
                sibling.position = CGPoint(x: 140, y: 40)
                sibling.zPosition = 100
                sibling.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                sibling.filters = [CAFilter.brightness(0.25)]
                root.addSublayer(sibling)

                let parent = CALayer()
                parent.bounds = CGRect(x: 0, y: 0, width: 90, height: 70)
                parent.position = CGPoint(x: 260, y: 40)
                parent.zPosition = 100
                parent.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                parent.filters = [CAFilter.colorInvert()]

                let child = CALayer()
                child.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)
                child.position = CGPoint(x: 45, y: 35)
                child.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                child.filters = [CAFilter.colorInvert()]
                parent.addSublayer(child)
                root.addSublayer(parent)

                engine.renderFrame()
                do {
                    let pixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 40, y: 260),
                        CGPoint(x: 140, y: 260),
                        CGPoint(x: 225, y: 260),
                        CGPoint(x: 260, y: 260),
                    ])
                    layerFilterProbeResult = pixels
                        .map { $0.map(String.init).joined(separator: ",") }
                        .joined(separator: ";")
                } catch {
                    layerFilterProbeResult = "error: \(error)"
                }

                chained.removeFromSuperlayer()
                sibling.removeFromSuperlayer()
                parent.removeFromSuperlayer()
                engine.renderFrame()
            }
        })
        h.expose("beginShadowProbe", action: {
            Task { @MainActor in
                shadowProbeResult = "running"
                let engine = CAAnimationEngine.shared
                engine.pause()
                guard let root = rootLayerRef,
                      let renderer = engine.renderer as? CAWebGPURenderer else {
                    shadowProbeResult = "error: root layer or renderer unavailable"
                    return
                }

                func makeShadowLayer(
                    x: CGFloat,
                    color: CGColor,
                    opacity: Float,
                    parent: CALayer
                ) -> CALayer {
                    let layer = CALayer()
                    layer.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                    layer.position = CGPoint(x: x, y: 150)
                    layer.zPosition = 100
                    layer.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                    layer.opacity = opacity
                    layer.shadowColor = color
                    layer.shadowOpacity = 1
                    layer.shadowOffset = CGSize(width: 40, height: 0)
                    layer.shadowRadius = 4
                    parent.addSublayer(layer)
                    return layer
                }

                let movingContainer = CALayer()
                movingContainer.bounds = root.bounds
                movingContainer.position = root.position
                movingContainer.zPosition = 100
                root.addSublayer(movingContainer)

                let red = makeShadowLayer(
                    x: 40,
                    color: CGColor(red: 1, green: 0, blue: 0, alpha: 1),
                    opacity: 1,
                    parent: movingContainer
                )
                let green = makeShadowLayer(
                    x: 150,
                    color: CGColor(red: 0, green: 1, blue: 0, alpha: 1),
                    opacity: 0.5,
                    parent: root
                )
                let animated = makeShadowLayer(
                    x: 260,
                    color: CGColor(red: 0, green: 0, blue: 1, alpha: 1),
                    opacity: 1,
                    parent: root
                )
                animated.shadowOpacity = 0
                let shadowAnimation = CABasicAnimation(keyPath: "shadowOpacity")
                shadowAnimation.fromValue = Float(0)
                shadowAnimation.toValue = Float(1)
                shadowAnimation.duration = 1
                shadowAnimation.speed = 0
                shadowAnimation.timeOffset = 0.5
                shadowAnimation.fillMode = .both
                shadowAnimation.isRemovedOnCompletion = false
                animated.add(shadowAnimation, forKey: "animatedShadowProbe")

                let emptyPath = makeShadowLayer(
                    x: 340,
                    color: CGColor(red: 1, green: 0, blue: 0, alpha: 1),
                    opacity: 1,
                    parent: root
                )
                emptyPath.shadowPath = CGMutablePath()

                engine.renderFrame()
                var result: String
                do {
                    let pixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 80, y: 150),
                        CGPoint(x: 190, y: 150),
                        CGPoint(x: 300, y: 150),
                        CGPoint(x: 380, y: 150),
                    ])
                    result = pixels
                        .map { $0.map(String.init).joined(separator: ",") }
                        .joined(separator: ";")

                    movingContainer.position.x += 20
                    engine.renderFrame()
                    let movedPixel = try await renderer.readbackPixel(x: 100, y: 150)
                    result += ";" + movedPixel.map(String.init).joined(separator: ",")
                } catch {
                    result = "error: \(error)"
                }

                red.removeFromSuperlayer()
                green.removeFromSuperlayer()
                animated.removeFromSuperlayer()
                emptyPath.removeFromSuperlayer()
                movingContainer.removeFromSuperlayer()
                engine.renderFrame()
                shadowProbeResult = result
            }
        })
        h.expose("beginDisplayLinkProbe", action: {
            Task { @MainActor in
                displayLinkProbeResult = "running"
                let target = DisplayLinkProbeTarget()
                let displayLink = CADisplayLink(
                    target: target,
                    selector: Selector("displayLinkDidFire")
                )
                displayLink.add(to: .main, forMode: .default)
                displayLink.add(to: .main, forMode: .common)

                do {
                    try await Task.sleep(for: .milliseconds(100))
                    let initialCount = target.callbackCount

                    displayLink.remove(from: .main, forMode: .default)
                    try await Task.sleep(for: .milliseconds(100))
                    let retainedModeCount = target.callbackCount

                    displayLink.remove(from: .main, forMode: .common)
                    let stoppedCount = target.callbackCount
                    try await Task.sleep(for: .milliseconds(100))
                    let finalCount = target.callbackCount

                    displayLinkProbeResult = [
                        "started=\(initialCount > 0)",
                        "retained=\(retainedModeCount > initialCount)",
                        "stopped=\(finalCount == stoppedCount)",
                        "duration=\(displayLink.duration > 0)",
                    ].joined(separator: ",")
                } catch {
                    displayLinkProbeResult = "error: \(error)"
                }
                displayLink.invalidate()
            }
        })
        h.expose("removeTransition", action: {
            MainActor.assumeIsolated {
                transitioningLayerRef?.removeAnimation(forKey: "browserCrossfade")
                filteredTransitioningLayerRef?.removeAnimation(forKey: "browserFilteredTransition")
                unsupportedTransitioningLayerRef?.removeFromSuperlayer()
                CAAnimationEngine.shared.renderFrame()
            }
        })
        h.expose("beginPixelReadback", action: {
            Task { @MainActor in
                do {
                    try await Task.sleep(for: .milliseconds(300))
                } catch {
                    pixelReadbackResult = "error: fade wait failed: \(error)"
                    return
                }

                let engine = CAAnimationEngine.shared
                engine.pause()
                engine.renderFrame()

                guard let renderer = engine.renderer as? CAWebGPURenderer else {
                    pixelReadbackResult = "error: renderer unavailable"
                    return
                }

                do {
                    let pixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 80, y: 220),
                        CGPoint(x: 200, y: 220),
                        CGPoint(x: 340, y: 240),
                        CGPoint(x: 10, y: 10),
                        CGPoint(x: 200, y: 80),
                        CGPoint(x: 320, y: 80),
                        CGPoint(x: 80, y: 80),
                    ])
                    pixelReadbackResult = pixels
                        .map { $0.map(String.init).joined(separator: ",") }
                        .joined(separator: ";")
                } catch {
                    pixelReadbackResult = "error: \(error)"
                }
            }
        })
    }
}
