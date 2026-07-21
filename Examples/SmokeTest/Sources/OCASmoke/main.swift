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
nonisolated(unsafe) var unsupportedBuiltInTransitioningLayerRef: CALayer?
nonisolated(unsafe) var unsupportedTransitionSubtypeLayerRef: CALayer?
nonisolated(unsafe) var tileDelegateRef: SmokeTileDelegate?
nonisolated(unsafe) var tileDrawCount: Int = 0
nonisolated(unsafe) var pixelReadbackResult: String = "pending"
nonisolated(unsafe) var transitionFilterProbeResult: String = "pending"
nonisolated(unsafe) var layerFilterProbeResult: String = "pending"
nonisolated(unsafe) var shadowProbeResult: String = "pending"
nonisolated(unsafe) var displayLinkProbeResult: String = "pending"
nonisolated(unsafe) var emitterProbeResult: String = "pending"
nonisolated(unsafe) var replicatorProbeResult: String = "pending"
nonisolated(unsafe) var compositionProbeResult: String = "pending"
nonisolated(unsafe) var transformDepthProbeResult: String = "pending"

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
            unsupportedBuiltInTransitioningLayerRef = nil
            unsupportedTransitionSubtypeLayerRef = nil
            tileDelegateRef = nil
            tileDrawCount = 0
            pixelReadbackResult = "pending"
            transitionFilterProbeResult = "pending"
            layerFilterProbeResult = "pending"
            shadowProbeResult = "pending"
            displayLinkProbeResult = "pending"
            emitterProbeResult = "pending"
            replicatorProbeResult = "pending"
            compositionProbeResult = "pending"
            transformDepthProbeResult = "pending"
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
        h.expose("getEmitterProbeResult", returning: {
            .string(emitterProbeResult)
        })
        h.expose("getReplicatorProbeResult", returning: {
            .string(replicatorProbeResult)
        })
        h.expose("getCompositionProbeResult", returning: {
            .string(compositionProbeResult)
        })
        h.expose("getTransformDepthProbeResult", returning: {
            .string(transformDepthProbeResult)
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
        h.expose("getTransitionRenderFailureCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .transitionRenderFailureCount ?? -1
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
        h.expose("getLayerFilterFailureCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .layerFilterFailureCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getCompositionFilterFailureCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .compositionFilterFailureCount ?? -1
            }
            return .number(Double(count))
        })
        h.expose("getActiveCompositionResourceCount", returning: {
            let count = MainActor.assumeIsolated {
                (CAAnimationEngine.shared.renderer as? CAWebGPURenderer)?
                    .activeCompositionResourceCount ?? -1
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
        h.expose("exerciseUnsupportedBuiltInTransition", action: {
            MainActor.assumeIsolated {
                let layer = CALayer()
                layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
                layer.position = CGPoint(x: 12, y: 4)
                layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                rootLayerRef?.addSublayer(layer)

                let transition = CATransition()
                transition.type = CATransitionType(rawValue: "unsupported")
                transition.duration = 1
                transition.speed = 0
                transition.timeOffset = 0.5
                layer.add(transition, forKey: "unsupportedBuiltInTransition")
                layer.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                unsupportedBuiltInTransitioningLayerRef = layer
                CAAnimationEngine.shared.renderFrame()
            }
        })
        h.expose("exerciseUnsupportedTransitionSubtype", action: {
            MainActor.assumeIsolated {
                let layer = CALayer()
                layer.bounds = CGRect(x: 0, y: 0, width: 8, height: 8)
                layer.position = CGPoint(x: 20, y: 4)
                layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                rootLayerRef?.addSublayer(layer)

                let transition = CATransition()
                transition.type = .push
                transition.subtype = CATransitionSubtype(rawValue: "unsupported")
                transition.duration = 1
                transition.speed = 0
                transition.timeOffset = 0.5
                layer.add(transition, forKey: "unsupportedTransitionSubtype")
                layer.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                unsupportedTransitionSubtypeLayerRef = layer
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
                guard let chainColorInvert = CIFilter(name: "CIColorInvert") else {
                    layerFilterProbeResult = "error: chain CIColorInvert unavailable"
                    return
                }
                chained.filters = [CAFilter.brightness(-0.5), chainColorInvert]
                root.addSublayer(chained)

                let sibling = CALayer()
                sibling.bounds = CGRect(x: 0, y: 0, width: 70, height: 70)
                sibling.position = CGPoint(x: 140, y: 40)
                sibling.zPosition = 100
                sibling.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                guard let coreImageBrightness = CIFilter(name: "CIColorControls") else {
                    layerFilterProbeResult = "error: CIColorControls unavailable"
                    return
                }
                coreImageBrightness.setValue(Float(0.25), forKey: kCIInputBrightnessKey)
                sibling.filters = [coreImageBrightness, CAFilter.colorInvert()]
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

                guard let incompatibleFilter = CIFilter(name: "CISourceOverCompositing") else {
                    layerFilterProbeResult = "error: CISourceOverCompositing unavailable"
                    return
                }
                guard let alphaColorInvert = CIFilter(name: "CIColorInvert") else {
                    layerFilterProbeResult = "error: CIColorInvert unavailable"
                    return
                }

                CATransaction.begin()
                CATransaction.setDisableActions(true)
                let opacityGroup = CALayer()
                opacityGroup.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                opacityGroup.position = CGPoint(x: 360, y: 40)
                opacityGroup.zPosition = 100
                opacityGroup.opacity = 0.5
                opacityGroup.allowsGroupOpacity = true
                for _ in 0..<2 {
                    let component = CALayer()
                    component.frame = opacityGroup.bounds
                    component.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                    opacityGroup.addSublayer(component)
                }
                root.addSublayer(opacityGroup)

                let translucentGroup = CALayer()
                translucentGroup.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                translucentGroup.position = CGPoint(x: 360, y: 100)
                translucentGroup.zPosition = 100
                translucentGroup.opacity = 0.5
                translucentGroup.allowsGroupOpacity = true
                for _ in 0..<2 {
                    let component = CALayer()
                    component.frame = translucentGroup.bounds
                    component.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 0.5)
                    translucentGroup.addSublayer(component)
                }
                root.addSublayer(translucentGroup)

                let rejected = CALayer()
                rejected.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                rejected.position = CGPoint(x: 140, y: 140)
                rejected.zPosition = 100
                rejected.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                rejected.filters = [incompatibleFilter]
                root.addSublayer(rejected)

                let alphaFiltered = CALayer()
                alphaFiltered.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                alphaFiltered.position = CGPoint(x: 300, y: 140)
                alphaFiltered.zPosition = 100
                alphaFiltered.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 0.5)
                alphaFiltered.filters = [alphaColorInvert]
                root.addSublayer(alphaFiltered)
                CATransaction.commit()

                engine.renderFrame()
                do {
                    let pixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 40, y: 260),
                        CGPoint(x: 140, y: 260),
                        CGPoint(x: 225, y: 260),
                        CGPoint(x: 260, y: 260),
                        CGPoint(x: 360, y: 260),
                        CGPoint(x: 360, y: 200),
                        CGPoint(x: 140, y: 160),
                        CGPoint(x: 300, y: 160),
                    ])
                    let groupedPixel = pixels[4]
                    let translucentGroupedPixel = pixels[5]
                    let rejectedPixel = pixels[6]
                    let alphaFilteredPixel = pixels[7]
                    opacityGroup.allowsGroupOpacity = false
                    translucentGroup.allowsGroupOpacity = false
                    engine.renderFrame()
                    let ungroupedPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 360, y: 260),
                        CGPoint(x: 360, y: 200),
                    ])
                    let ungroupedPixel = ungroupedPixels[0]
                    let translucentUngroupedPixel = ungroupedPixels[1]
                    let grouped = groupedPixel == [140, 13, 19, 255]
                    let ungrouped = ungroupedPixel[0] >= 195
                        && ungroupedPixel[1] <= 7
                        && ungroupedPixel[2] <= 11
                        && ungroupedPixel[3] == 255
                    let translucentGrouped = (108...115).contains(translucentGroupedPixel[0])
                        && (13...19).contains(translucentGroupedPixel[1])
                        && (20...27).contains(translucentGroupedPixel[2])
                        && translucentGroupedPixel[3] == 255
                    let translucentUngrouped = (122...130).contains(translucentUngroupedPixel[0])
                        && (12...18).contains(translucentUngroupedPixel[1])
                        && (18...25).contains(translucentUngroupedPixel[2])
                        && translucentUngroupedPixel[3] == 255
                    let rejectedExplicitly = rejectedPixel == [26, 26, 38, 255]
                    let alphaFilteredCorrectly = alphaFilteredPixel == [13, 141, 147, 255]
                    layerFilterProbeResult = pixels.prefix(4)
                        .map { $0.map(String.init).joined(separator: ",") }
                        .joined(separator: ";")
                        + ";group=\(grouped),ungrouped=\(ungrouped)"
                        + ",translucentGroup=\(translucentGrouped)"
                        + ",translucentUngrouped=\(translucentUngrouped)"
                        + ",rejected=\(rejectedExplicitly)"
                        + ",alphaFilter=\(alphaFilteredCorrectly)"
                        + ",alphaPixel=\(alphaFilteredPixel.map(String.init).joined(separator: ","))"
                } catch {
                    layerFilterProbeResult = "error: \(error)"
                }

                chained.removeFromSuperlayer()
                sibling.removeFromSuperlayer()
                parent.removeFromSuperlayer()
                opacityGroup.removeFromSuperlayer()
                translucentGroup.removeFromSuperlayer()
                rejected.removeFromSuperlayer()
                alphaFiltered.removeFromSuperlayer()
                engine.renderFrame()
            }
        })
        h.expose("beginTransformDepthProbe", action: {
            Task { @MainActor in
                transformDepthProbeResult = "running"
                let engine = CAAnimationEngine.shared
                engine.pause()
                guard let root = rootLayerRef,
                      let renderer = engine.renderer as? CAWebGPURenderer else {
                    transformDepthProbeResult = "error: transform depth dependencies unavailable"
                    return
                }

                CATransaction.begin()
                CATransaction.setDisableActions(true)
                let originalRootBackground = root.backgroundColor
                let existingLayerStates = (root.sublayers ?? []).map { ($0, $0.isHidden) }
                for (layer, _) in existingLayerStates {
                    layer.isHidden = true
                }
                root.backgroundColor = CGColor(red: 0, green: 0, blue: 0, alpha: 1)

                let crossingGroup = CATransformLayer()
                crossingGroup.bounds = root.bounds
                crossingGroup.anchorPoint = .zero
                crossingGroup.position = .zero
                crossingGroup.backgroundColor = CGColor(red: 1, green: 1, blue: 0, alpha: 1)
                crossingGroup.filters = [CAFilter.colorInvert()]
                crossingGroup.mask = CALayer()
                crossingGroup.shadowColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                crossingGroup.shadowOpacity = 1
                crossingGroup.shouldRasterize = true

                let flatPlane = CALayer()
                flatPlane.bounds = CGRect(x: 0, y: 0, width: 120, height: 80)
                flatPlane.position = CGPoint(x: 200, y: 220)
                flatPlane.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                crossingGroup.addSublayer(flatPlane)

                let crossingPlane = CALayer()
                crossingPlane.bounds = flatPlane.bounds
                crossingPlane.position = flatPlane.position
                crossingPlane.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                crossingPlane.transform = CATransform3DMakeRotation(.pi / 3, 0, 1, 0)
                crossingGroup.addSublayer(crossingPlane)

                let flattenedContainer = CALayer()
                flattenedContainer.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                flattenedContainer.position = CGPoint(x: 280, y: 250)
                let flattenedRedChild = CALayer()
                flattenedRedChild.frame = flattenedContainer.bounds
                flattenedRedChild.zPosition = 100
                flattenedRedChild.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                flattenedContainer.addSublayer(flattenedRedChild)
                crossingGroup.addSublayer(flattenedContainer)

                let flattenedOccluder = CALayer()
                flattenedOccluder.bounds = flattenedContainer.bounds
                flattenedOccluder.position = flattenedContainer.position
                flattenedOccluder.zPosition = 50
                flattenedOccluder.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                crossingGroup.addSublayer(flattenedOccluder)

                let nestedTransform = CATransformLayer()
                nestedTransform.bounds = root.bounds
                nestedTransform.anchorPoint = .zero
                nestedTransform.position = .zero
                let nestedRedChild = CALayer()
                nestedRedChild.bounds = flattenedContainer.bounds
                nestedRedChild.position = CGPoint(x: 350, y: 250)
                nestedRedChild.zPosition = 100
                nestedRedChild.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                nestedTransform.addSublayer(nestedRedChild)
                crossingGroup.addSublayer(nestedTransform)

                let nestedOccluder = CALayer()
                nestedOccluder.bounds = nestedRedChild.bounds
                nestedOccluder.position = nestedRedChild.position
                nestedOccluder.zPosition = 50
                nestedOccluder.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                crossingGroup.addSublayer(nestedOccluder)

                let opacityGroup = CALayer()
                opacityGroup.bounds = flattenedContainer.bounds
                opacityGroup.position = CGPoint(x: 30, y: 250)
                opacityGroup.opacity = 0.5
                opacityGroup.allowsGroupOpacity = true
                let opacityRedChild = CALayer()
                opacityRedChild.frame = opacityGroup.bounds
                opacityRedChild.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                opacityGroup.addSublayer(opacityRedChild)
                let opacityGreenChild = CALayer()
                opacityGreenChild.frame = opacityGroup.bounds
                opacityGreenChild.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                opacityGroup.addSublayer(opacityGreenChild)
                crossingGroup.addSublayer(opacityGroup)

                let filteredGroup = CALayer()
                filteredGroup.bounds = flattenedContainer.bounds
                filteredGroup.position = CGPoint(x: 30, y: 180)
                filteredGroup.filters = [CAFilter.colorInvert()]
                let filteredRedChild = CALayer()
                filteredRedChild.frame = filteredGroup.bounds
                filteredRedChild.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                filteredGroup.addSublayer(filteredRedChild)
                crossingGroup.addSublayer(filteredGroup)

                let nestedEffectGroup = CALayer()
                nestedEffectGroup.bounds = flattenedContainer.bounds
                nestedEffectGroup.position = CGPoint(x: 90, y: 180)
                let nestedFilteredChild = CALayer()
                nestedFilteredChild.frame = nestedEffectGroup.bounds
                nestedFilteredChild.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                nestedFilteredChild.filters = [CAFilter.colorInvert()]
                nestedEffectGroup.addSublayer(nestedFilteredChild)
                crossingGroup.addSublayer(nestedEffectGroup)

                let maskedGroup = CALayer()
                maskedGroup.bounds = flattenedContainer.bounds
                maskedGroup.position = CGPoint(x: 30, y: 110)
                let maskedRedChild = CALayer()
                maskedRedChild.frame = maskedGroup.bounds
                maskedRedChild.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                maskedGroup.addSublayer(maskedRedChild)
                let halfMask = CALayer()
                halfMask.bounds = CGRect(x: 0, y: 0, width: 20, height: 40)
                halfMask.position = CGPoint(x: 10, y: 20)
                halfMask.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                maskedGroup.mask = halfMask
                crossingGroup.addSublayer(maskedGroup)

                let shadowGroup = CALayer()
                shadowGroup.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
                shadowGroup.position = CGPoint(x: 150, y: 140)
                shadowGroup.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                shadowGroup.shadowColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                shadowGroup.shadowOpacity = 1
                shadowGroup.shadowOffset = CGSize(width: 20, height: 0)
                shadowGroup.shadowRadius = 4
                crossingGroup.addSublayer(shadowGroup)

                let pathShadowGroup = CALayer()
                pathShadowGroup.bounds = shadowGroup.bounds
                pathShadowGroup.position = CGPoint(x: 220, y: 140)
                pathShadowGroup.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                pathShadowGroup.shadowColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                pathShadowGroup.shadowOpacity = 1
                pathShadowGroup.shadowOffset = CGSize(width: 20, height: 0)
                pathShadowGroup.shadowRadius = 0
                let narrowShadowPath = CGMutablePath()
                narrowShadowPath.addRect(CGRect(x: 0, y: 0, width: 10, height: 20))
                pathShadowGroup.shadowPath = narrowShadowPath
                crossingGroup.addSublayer(pathShadowGroup)

                let compositionBackdropPlane = CALayer()
                compositionBackdropPlane.bounds = CGRect(x: 0, y: 0, width: 60, height: 40)
                compositionBackdropPlane.position = CGPoint(x: 300, y: 40)
                compositionBackdropPlane.zPosition = -100
                compositionBackdropPlane.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                crossingGroup.addSublayer(compositionBackdropPlane)

                let compositionPlane = CALayer()
                compositionPlane.bounds = compositionBackdropPlane.bounds
                compositionPlane.position = compositionBackdropPlane.position
                compositionPlane.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                guard let compositionScreenFilter = CIFilter(name: "CIScreenCompositing") else {
                    crossingGroup.removeFromSuperlayer()
                    root.backgroundColor = originalRootBackground
                    for (layer, wasHidden) in existingLayerStates {
                        layer.isHidden = wasHidden
                    }
                    CATransaction.commit()
                    engine.renderFrame()
                    transformDepthProbeResult = "error: composition depth filter unavailable"
                    return
                }
                compositionPlane.compositingFilter = compositionScreenFilter
                crossingGroup.addSublayer(compositionPlane)

                let compositionCrossingPlane = CALayer()
                compositionCrossingPlane.bounds = compositionBackdropPlane.bounds
                compositionCrossingPlane.position = compositionBackdropPlane.position
                compositionCrossingPlane.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                compositionCrossingPlane.transform = CATransform3DMakeRotation(.pi / 3, 0, 1, 0)
                crossingGroup.addSublayer(compositionCrossingPlane)
                root.addSublayer(crossingGroup)

                let transparencyGroup = CATransformLayer()
                transparencyGroup.bounds = root.bounds
                transparencyGroup.anchorPoint = .zero
                transparencyGroup.position = .zero

                let imageData = Data([
                    255, 255, 255, 255,
                    0, 0, 0, 0,
                    255, 255, 255, 255,
                ])
                guard let transparentImage = CGImage(
                    width: 3,
                    height: 1,
                    bitsPerComponent: 8,
                    bitsPerPixel: 32,
                    bytesPerRow: 12,
                    space: .deviceRGB,
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                    provider: CGDataProvider(data: imageData),
                    decode: nil,
                    shouldInterpolate: false,
                    intent: .defaultIntent
                ) else {
                    crossingGroup.removeFromSuperlayer()
                    root.backgroundColor = originalRootBackground
                    for (layer, wasHidden) in existingLayerStates {
                        layer.isHidden = wasHidden
                    }
                    CATransaction.commit()
                    engine.renderFrame()
                    transformDepthProbeResult = "error: transparent depth image unavailable"
                    return
                }

                let transparentFront = CALayer()
                transparentFront.bounds = CGRect(x: 0, y: 0, width: 90, height: 30)
                transparentFront.position = CGPoint(x: 100, y: 80)
                transparentFront.contents = transparentImage
                transparentFront.magnificationFilter = .nearest
                transparentFront.transform = CATransform3DMakeTranslation(0, 0, 100)
                transparencyGroup.addSublayer(transparentFront)

                let transparentBack = CALayer()
                transparentBack.bounds = transparentFront.bounds
                transparentBack.position = transparentFront.position
                transparentBack.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                transparencyGroup.addSublayer(transparentBack)
                root.addSublayer(transparencyGroup)

                let firstIndependentGroup = CATransformLayer()
                firstIndependentGroup.bounds = root.bounds
                firstIndependentGroup.anchorPoint = .zero
                firstIndependentGroup.position = .zero
                let firstIndependentPlane = CALayer()
                firstIndependentPlane.bounds = CGRect(x: 0, y: 0, width: 50, height: 50)
                firstIndependentPlane.position = CGPoint(x: 330, y: 100)
                firstIndependentPlane.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                firstIndependentPlane.transform = CATransform3DMakeTranslation(0, 0, 500)
                firstIndependentGroup.addSublayer(firstIndependentPlane)
                root.addSublayer(firstIndependentGroup)

                let secondIndependentGroup = CATransformLayer()
                secondIndependentGroup.bounds = root.bounds
                secondIndependentGroup.anchorPoint = .zero
                secondIndependentGroup.position = .zero
                let secondIndependentPlane = CALayer()
                secondIndependentPlane.bounds = firstIndependentPlane.bounds
                secondIndependentPlane.position = firstIndependentPlane.position
                secondIndependentPlane.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                secondIndependentPlane.transform = CATransform3DMakeTranslation(0, 0, -500)
                secondIndependentGroup.addSublayer(secondIndependentPlane)
                root.addSublayer(secondIndependentGroup)
                CATransaction.commit()

                do {
                    engine.renderFrame()
                    let pixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 185, y: 80),
                        CGPoint(x: 215, y: 80),
                        CGPoint(x: 70, y: 220),
                        CGPoint(x: 100, y: 220),
                        CGPoint(x: 130, y: 220),
                        CGPoint(x: 330, y: 200),
                        CGPoint(x: 280, y: 50),
                        CGPoint(x: 350, y: 50),
                        CGPoint(x: 30, y: 50),
                        CGPoint(x: 30, y: 120),
                        CGPoint(x: 20, y: 190),
                        CGPoint(x: 45, y: 190),
                        CGPoint(x: 90, y: 120),
                        CGPoint(x: 170, y: 160),
                        CGPoint(x: 235, y: 160),
                        CGPoint(x: 245, y: 160),
                        CGPoint(x: 185, y: 160),
                        CGPoint(x: 285, y: 260),
                        CGPoint(x: 315, y: 260),
                    ])
                    let flatteningCaptureCount = renderer.transformFlatteningCaptureCount
                    let flatteningCompositeCount = renderer.transformFlatteningCompositeCount
                    flattenedRedChild.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                    flattenedOccluder.isHidden = true
                    engine.renderFrame()
                    let updatedFlattenedPixel = try await renderer.readbackPixel(x: 280, y: 50)
                    let flatteningRecaptureCount = renderer.transformFlatteningCaptureCount
                    let flatteningUpdatedCompositeCount = renderer.transformFlatteningCompositeCount
                    engine.renderFrame()
                    let reusedFlattenedPixel = try await renderer.readbackPixel(x: 280, y: 50)
                    let flatteningReuseCaptureCount = renderer.transformFlatteningCaptureCount
                    let flatteningReuseCompositeCount = renderer.transformFlatteningCompositeCount
                    crossingGroup.removeFromSuperlayer()
                    transparencyGroup.removeFromSuperlayer()
                    firstIndependentGroup.removeFromSuperlayer()
                    secondIndependentGroup.removeFromSuperlayer()
                    root.backgroundColor = originalRootBackground
                    for (layer, wasHidden) in existingLayerStates {
                        layer.isHidden = wasHidden
                    }
                    engine.renderFrame()
                    let crossingIsDepthCorrect = pixels[0] == [255, 0, 0, 255]
                        && pixels[1] == [0, 0, 255, 255]
                    let transparentPixelsAreCorrect = pixels[2][0] >= 240
                        && pixels[2][1] >= 240
                        && pixels[2][2] >= 240
                        && pixels[2][3] == 255
                        && pixels[3][0] == 0
                        && pixels[3][1] >= 245
                        && pixels[3][2] == 0
                        && pixels[3][3] == 255
                        && pixels[4] == [255, 255, 255, 255]
                    let independentGroupsAreIsolated = pixels[5] == [0, 255, 0, 255]
                    let normalSubtreeIsFlattened = pixels[6] == [0, 255, 0, 255]
                    let nestedTransformPreservesDepth = pixels[7] == [255, 0, 0, 255]
                    let groupOpacityIsAppliedOnce = pixels[8][0] <= 1
                        && (127...128).contains(pixels[8][1])
                        && pixels[8][2] <= 1
                        && pixels[8][3] == 255
                    let filterUsesLocalPixels = pixels[9] == [0, 255, 255, 255]
                    let contentMaskUsesLocalBounds = pixels[10] == [255, 0, 0, 255]
                        && pixels[11] == [0, 0, 0, 255]
                    let nestedFilterUsesLocalPixels = pixels[12] == [0, 255, 255, 255]
                    let expandedShadowIsVisible = pixels[13][0] >= 240
                        && pixels[13][1] >= 240
                        && pixels[13][2] >= 240
                        && pixels[13][3] == 255
                        && pixels[16][0] > 0
                        && pixels[16][1] > 0
                        && pixels[16][2] > 0
                        && pixels[16][3] == 255
                    let customShadowPathIsRespected = pixels[14] == [255, 255, 255, 255]
                        && pixels[15] == [0, 0, 0, 255]
                    let compositionPlaneWritesDepth = pixels[17] == [0, 0, 255, 255]
                        && pixels[18] == [255, 255, 0, 255]
                    let changedSubtreeWasRecaptured = updatedFlattenedPixel == [0, 0, 255, 255]
                        && flatteningRecaptureCount == 1
                        && flatteningUpdatedCompositeCount == 7
                    let unchangedSubtreeWasReused = reusedFlattenedPixel == [0, 0, 255, 255]
                        && flatteningReuseCaptureCount == 0
                        && flatteningReuseCompositeCount == 7
                    transformDepthProbeResult = "crossing=\(crossingIsDepthCorrect)"
                        + ",transparent=\(transparentPixelsAreCorrect)"
                        + ",isolated=\(independentGroupsAreIsolated)"
                        + ",flattened=\(normalSubtreeIsFlattened)"
                        + ",nested=\(nestedTransformPreservesDepth)"
                        + ",captures=\(flatteningCaptureCount)"
                        + ",composites=\(flatteningCompositeCount)"
                        + ",groupOpacity=\(groupOpacityIsAppliedOnce)"
                        + ",filter=\(filterUsesLocalPixels)"
                        + ",mask=\(contentMaskUsesLocalBounds)"
                        + ",nestedFilter=\(nestedFilterUsesLocalPixels)"
                        + ",shadow=\(expandedShadowIsVisible)"
                        + ",shadowPath=\(customShadowPathIsRespected)"
                        + ",compositionDepth=\(compositionPlaneWritesDepth)"
                        + ",updated=\(changedSubtreeWasRecaptured)"
                        + ",reused=\(unchangedSubtreeWasReused)"
                } catch {
                    crossingGroup.removeFromSuperlayer()
                    transparencyGroup.removeFromSuperlayer()
                    firstIndependentGroup.removeFromSuperlayer()
                    secondIndependentGroup.removeFromSuperlayer()
                    root.backgroundColor = originalRootBackground
                    for (layer, wasHidden) in existingLayerStates {
                        layer.isHidden = wasHidden
                    }
                    engine.renderFrame()
                    transformDepthProbeResult = "error: \(error)"
                }
            }
        })
        h.expose("beginCompositionProbe", action: {
            Task { @MainActor in
                compositionProbeResult = "running"
                let engine = CAAnimationEngine.shared
                engine.pause()
                guard let root = rootLayerRef,
                      let renderer = engine.renderer as? CAWebGPURenderer,
                      let multiply = CIFilter(name: "CIMultiplyCompositing"),
                      let screen = CIFilter(name: "CIScreenCompositing"),
                      let sourceOver = CIFilter(name: "CISourceOverCompositing"),
                      let invert = CIFilter(name: "CIColorInvert"),
                      let halfAlphaMask = CIFilter(name: "CIColorMatrix") else {
                    compositionProbeResult = "error: composition dependencies unavailable"
                    return
                }

                CATransaction.begin()
                CATransaction.setDisableActions(true)
                let originalRootBackground = root.backgroundColor
                let existingLayerStates = (root.sublayers ?? []).map { ($0, $0.isHidden) }
                for (layer, _) in existingLayerStates {
                    layer.isHidden = true
                }
                root.backgroundColor = CGColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 0.5)
                let opacityBackdrop = CALayer()
                opacityBackdrop.bounds = CGRect(x: 0, y: 0, width: 60, height: 60)
                opacityBackdrop.position = CGPoint(x: 120, y: 80)
                opacityBackdrop.zPosition = 199
                opacityBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(opacityBackdrop)

                let backdrop = CALayer()
                backdrop.bounds = CGRect(x: 0, y: 0, width: 360, height: 60)
                backdrop.position = CGPoint(x: 200, y: 140)
                backdrop.zPosition = 200
                backdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(backdrop)

                let source = CALayer()
                source.bounds = CGRect(x: 0, y: 0, width: 60, height: 60)
                source.position = CGPoint(x: 160, y: 140)
                source.zPosition = 201
                source.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                source.compositingFilter = multiply
                root.addSublayer(source)

                let screenedSource = CALayer()
                screenedSource.bounds = source.bounds
                screenedSource.position = CGPoint(x: 240, y: 140)
                screenedSource.zPosition = 202
                screenedSource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                screenedSource.compositingFilter = screen
                root.addSublayer(screenedSource)

                let translucentSource = CALayer()
                translucentSource.bounds = source.bounds
                translucentSource.position = CGPoint(x: 320, y: 140)
                translucentSource.zPosition = 203
                translucentSource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 0.5)
                translucentSource.compositingFilter = multiply
                root.addSublayer(translucentSource)

                let laterSource = CALayer()
                laterSource.bounds = source.bounds
                laterSource.position = CGPoint(x: 370, y: 140)
                laterSource.zPosition = 204
                laterSource.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                root.addSublayer(laterSource)

                let backdropFiltered = CALayer()
                backdropFiltered.bounds = source.bounds
                backdropFiltered.position = CGPoint(x: 80, y: 140)
                backdropFiltered.zPosition = 205
                backdropFiltered.cornerRadius = 20
                backdropFiltered.backgroundFilters = [CAFilter.brightness(0), invert]
                root.addSublayer(backdropFiltered)

                let transparentSource = CALayer()
                transparentSource.bounds = source.bounds
                transparentSource.position = CGPoint(x: 50, y: 240)
                transparentSource.zPosition = 206
                transparentSource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 0.5)
                transparentSource.compositingFilter = sourceOver
                root.addSublayer(transparentSource)

                let opacitySource = CALayer()
                opacitySource.bounds = source.bounds
                opacitySource.position = CGPoint(x: 120, y: 80)
                opacitySource.zPosition = 207
                opacitySource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                opacitySource.opacity = 0.5
                opacitySource.compositingFilter = multiply
                root.addSublayer(opacitySource)

                let replicatorBackdrop = CALayer()
                replicatorBackdrop.bounds = CGRect(x: 0, y: 0, width: 100, height: 30)
                replicatorBackdrop.position = CGPoint(x: 240, y: 40)
                replicatorBackdrop.zPosition = 198
                replicatorBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(replicatorBackdrop)

                let compositionReplicator = CAReplicatorLayer()
                compositionReplicator.bounds = root.bounds
                compositionReplicator.anchorPoint = .zero
                compositionReplicator.position = .zero
                compositionReplicator.zPosition = 208
                compositionReplicator.instanceCount = 2
                compositionReplicator.instanceTransform = CATransform3DMakeTranslation(40, 0, 0)
                compositionReplicator.instanceColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                compositionReplicator.instanceGreenOffset = -1
                let replicatedSource = CALayer()
                replicatedSource.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)
                replicatedSource.position = CGPoint(x: 220, y: 40)
                replicatedSource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                replicatedSource.compositingFilter = screen
                compositionReplicator.addSublayer(replicatedSource)
                root.addSublayer(compositionReplicator)

                let nestedBackdrop = CALayer()
                nestedBackdrop.bounds = CGRect(x: 0, y: 0, width: 60, height: 30)
                nestedBackdrop.position = CGPoint(x: 330, y: 40)
                nestedBackdrop.zPosition = 197
                nestedBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(nestedBackdrop)

                let outerComposition = CALayer()
                outerComposition.bounds = nestedBackdrop.bounds
                outerComposition.position = nestedBackdrop.position
                outerComposition.zPosition = 209
                outerComposition.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                outerComposition.opacity = 0.5
                outerComposition.compositingFilter = screen
                let innerComposition = CALayer()
                innerComposition.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)
                innerComposition.position = CGPoint(x: 15, y: 15)
                innerComposition.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                innerComposition.compositingFilter = multiply
                outerComposition.addSublayer(innerComposition)
                root.addSublayer(outerComposition)

                let inheritedOpacityBackdrop = CALayer()
                inheritedOpacityBackdrop.bounds = CGRect(x: 0, y: 0, width: 30, height: 30)
                inheritedOpacityBackdrop.position = CGPoint(x: 50, y: 40)
                inheritedOpacityBackdrop.zPosition = 196
                inheritedOpacityBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(inheritedOpacityBackdrop)

                let opacityParent = CALayer()
                opacityParent.bounds = inheritedOpacityBackdrop.bounds
                opacityParent.position = inheritedOpacityBackdrop.position
                opacityParent.zPosition = 210
                opacityParent.opacity = 0.5
                opacityParent.allowsGroupOpacity = false
                let inheritedOpacitySource = CALayer()
                inheritedOpacitySource.bounds = inheritedOpacityBackdrop.bounds
                inheritedOpacitySource.position = CGPoint(x: 15, y: 15)
                inheritedOpacitySource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                inheritedOpacitySource.compositingFilter = multiply
                opacityParent.addSublayer(inheritedOpacitySource)
                root.addSublayer(opacityParent)

                let groupOpacityBackdrop = CALayer()
                groupOpacityBackdrop.bounds = inheritedOpacityBackdrop.bounds
                groupOpacityBackdrop.position = CGPoint(x: 85, y: 40)
                groupOpacityBackdrop.zPosition = 195
                groupOpacityBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(groupOpacityBackdrop)

                let groupOpacityParent = CALayer()
                groupOpacityParent.bounds = groupOpacityBackdrop.bounds
                groupOpacityParent.position = groupOpacityBackdrop.position
                groupOpacityParent.zPosition = 211
                groupOpacityParent.opacity = 0.5
                let groupOpacitySource = CALayer()
                groupOpacitySource.bounds = groupOpacityBackdrop.bounds
                groupOpacitySource.position = CGPoint(x: 15, y: 15)
                groupOpacitySource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                groupOpacitySource.compositingFilter = multiply
                groupOpacityParent.addSublayer(groupOpacitySource)
                root.addSublayer(groupOpacityParent)

                let filteredScopeBackdrop = CALayer()
                filteredScopeBackdrop.bounds = inheritedOpacityBackdrop.bounds
                filteredScopeBackdrop.position = CGPoint(x: 160, y: 40)
                filteredScopeBackdrop.zPosition = 194
                filteredScopeBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(filteredScopeBackdrop)

                let filteredScopeParent = CALayer()
                filteredScopeParent.bounds = filteredScopeBackdrop.bounds
                filteredScopeParent.position = filteredScopeBackdrop.position
                filteredScopeParent.zPosition = 212
                filteredScopeParent.filters = [CAFilter.colorInvert()]
                let filteredScopeSource = CALayer()
                filteredScopeSource.bounds = filteredScopeBackdrop.bounds
                filteredScopeSource.position = CGPoint(x: 15, y: 15)
                filteredScopeSource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                filteredScopeSource.compositingFilter = multiply
                filteredScopeParent.addSublayer(filteredScopeSource)
                root.addSublayer(filteredScopeParent)

                let clippedBackdrop = CALayer()
                clippedBackdrop.bounds = CGRect(x: 0, y: 0, width: 100, height: 60)
                clippedBackdrop.position = CGPoint(x: 250, y: 250)
                clippedBackdrop.zPosition = 193
                clippedBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(clippedBackdrop)

                let clippingParent = CALayer()
                clippingParent.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                clippingParent.position = CGPoint(x: 250, y: 250)
                clippingParent.zPosition = 213
                clippingParent.cornerRadius = 10
                clippingParent.masksToBounds = true
                let innerClippingParent = CALayer()
                innerClippingParent.bounds = CGRect(x: 0, y: 0, width: 30, height: 40)
                innerClippingParent.position = CGPoint(x: 25, y: 20)
                innerClippingParent.cornerRadius = 5
                innerClippingParent.masksToBounds = true
                let clippedSource = CALayer()
                clippedSource.bounds = CGRect(x: 0, y: 0, width: 80, height: 40)
                clippedSource.position = CGPoint(x: 15, y: 20)
                clippedSource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                clippedSource.compositingFilter = screen
                innerClippingParent.addSublayer(clippedSource)
                clippingParent.addSublayer(innerClippingParent)
                root.addSublayer(clippingParent)

                let clippedFilterBackdrop = CALayer()
                clippedFilterBackdrop.bounds = clippedBackdrop.bounds
                clippedFilterBackdrop.position = CGPoint(x: 350, y: 250)
                clippedFilterBackdrop.zPosition = 192
                clippedFilterBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(clippedFilterBackdrop)

                let filterClippingParent = CALayer()
                filterClippingParent.bounds = clippingParent.bounds
                filterClippingParent.position = CGPoint(x: 350, y: 250)
                filterClippingParent.zPosition = 214
                filterClippingParent.cornerRadius = 10
                filterClippingParent.masksToBounds = true
                let clippedBackdropFilter = CALayer()
                clippedBackdropFilter.bounds = clippedSource.bounds
                clippedBackdropFilter.position = CGPoint(x: 20, y: 20)
                clippedBackdropFilter.backgroundFilters = [CAFilter.colorInvert()]
                filterClippingParent.addSublayer(clippedBackdropFilter)
                root.addSublayer(filterClippingParent)

                func makeHalfMask() -> CALayer {
                    let mask = CALayer()
                    mask.bounds = CGRect(x: 0, y: 0, width: 30, height: 60)
                    mask.position = CGPoint(x: 15, y: 30)
                    mask.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                    return mask
                }

                let contentMaskBackdrop = CALayer()
                contentMaskBackdrop.bounds = CGRect(x: 0, y: 0, width: 60, height: 60)
                contentMaskBackdrop.position = CGPoint(x: 50, y: 200)
                contentMaskBackdrop.zPosition = 191
                contentMaskBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(contentMaskBackdrop)

                let contentMaskParent = CALayer()
                contentMaskParent.bounds = contentMaskBackdrop.bounds
                contentMaskParent.position = contentMaskBackdrop.position
                contentMaskParent.zPosition = 215
                halfAlphaMask.setValue(
                    CIVector(x: 0, y: 0, z: 0, w: 0.5),
                    forKey: "inputAVector"
                )
                let filteredHalfMask = makeHalfMask()
                filteredHalfMask.filters = [CAFilter.brightness(0), halfAlphaMask]
                contentMaskParent.mask = filteredHalfMask
                let contentMaskedSource = CALayer()
                contentMaskedSource.bounds = contentMaskBackdrop.bounds
                contentMaskedSource.position = CGPoint(x: 30, y: 30)
                contentMaskedSource.backgroundColor = CGColor(red: 0, green: 1, blue: 0, alpha: 1)
                contentMaskedSource.compositingFilter = screen
                contentMaskParent.addSublayer(contentMaskedSource)
                root.addSublayer(contentMaskParent)

                let targetMaskBackdrop = CALayer()
                targetMaskBackdrop.bounds = contentMaskBackdrop.bounds
                targetMaskBackdrop.position = CGPoint(x: 120, y: 200)
                targetMaskBackdrop.zPosition = 190
                targetMaskBackdrop.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                root.addSublayer(targetMaskBackdrop)

                let targetMaskedFilter = CALayer()
                targetMaskedFilter.bounds = targetMaskBackdrop.bounds
                targetMaskedFilter.position = targetMaskBackdrop.position
                targetMaskedFilter.zPosition = 216
                targetMaskedFilter.mask = makeHalfMask()
                targetMaskedFilter.backgroundFilters = [CAFilter.colorInvert()]
                root.addSublayer(targetMaskedFilter)
                CATransaction.commit()

                engine.renderFrame()
                do {
                    let pixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 160, y: 160),
                        CGPoint(x: 240, y: 160),
                        CGPoint(x: 320, y: 160),
                        CGPoint(x: 370, y: 160),
                        CGPoint(x: 80, y: 140),
                        CGPoint(x: 52, y: 188),
                        CGPoint(x: 50, y: 60),
                        CGPoint(x: 120, y: 220),
                        CGPoint(x: 220, y: 260),
                        CGPoint(x: 260, y: 260),
                        CGPoint(x: 315, y: 260),
                        CGPoint(x: 345, y: 260),
                        CGPoint(x: 50, y: 260),
                        CGPoint(x: 85, y: 260),
                        CGPoint(x: 160, y: 260),
                        CGPoint(x: 250, y: 50),
                        CGPoint(x: 215, y: 50),
                        CGPoint(x: 232, y: 68),
                        CGPoint(x: 235, y: 50),
                        CGPoint(x: 350, y: 50),
                        CGPoint(x: 315, y: 50),
                        CGPoint(x: 332, y: 68),
                        CGPoint(x: 35, y: 100),
                        CGPoint(x: 65, y: 100),
                        CGPoint(x: 105, y: 100),
                        CGPoint(x: 135, y: 100),
                    ])
                    let composited = pixels[0] == [0, 0, 0, 255]
                        && pixels[1] == [255, 255, 0, 255]
                        && pixels[2] == [127, 0, 0, 255]
                        && pixels[3] == [0, 0, 255, 255]
                        && pixels[4] == [0, 255, 255, 255]
                        && pixels[5] == [255, 0, 0, 255]
                        && pixels[6] == [7, 135, 10, 192]
                        && pixels[7] == [127, 0, 0, 255]
                        && pixels[8] == [255, 255, 0, 255]
                        && pixels[9] == [255, 0, 0, 255]
                        && pixels[10] == [255, 0, 0, 255]
                        && pixels[11] == [255, 128, 0, 255]
                        && pixels[12] == [127, 0, 0, 255]
                        && pixels[13] == [128, 128, 0, 255]
                        && pixels[14] == [255, 0, 255, 255]
                        && pixels[15] == [255, 255, 0, 255]
                        && pixels[16] == [255, 0, 0, 255]
                        && pixels[17] == [255, 0, 0, 255]
                        && pixels[18] == [255, 0, 0, 255]
                        && pixels[19] == [0, 255, 255, 255]
                        && pixels[20] == [255, 0, 0, 255]
                        && pixels[21] == [255, 0, 0, 255]
                        && pixels[22] == [255, 128, 0, 255]
                        && pixels[23] == [255, 0, 0, 255]
                        && pixels[24] == [0, 255, 255, 255]
                        && pixels[25] == [255, 0, 0, 255]
                    opacityBackdrop.removeFromSuperlayer()
                    backdrop.removeFromSuperlayer()
                    source.removeFromSuperlayer()
                    screenedSource.removeFromSuperlayer()
                    translucentSource.removeFromSuperlayer()
                    laterSource.removeFromSuperlayer()
                    backdropFiltered.removeFromSuperlayer()
                    transparentSource.removeFromSuperlayer()
                    opacitySource.removeFromSuperlayer()
                    replicatorBackdrop.removeFromSuperlayer()
                    compositionReplicator.removeFromSuperlayer()
                    nestedBackdrop.removeFromSuperlayer()
                    outerComposition.removeFromSuperlayer()
                    inheritedOpacityBackdrop.removeFromSuperlayer()
                    opacityParent.removeFromSuperlayer()
                    groupOpacityBackdrop.removeFromSuperlayer()
                    groupOpacityParent.removeFromSuperlayer()
                    filteredScopeBackdrop.removeFromSuperlayer()
                    filteredScopeParent.removeFromSuperlayer()
                    clippedBackdrop.removeFromSuperlayer()
                    clippingParent.removeFromSuperlayer()
                    clippedFilterBackdrop.removeFromSuperlayer()
                    filterClippingParent.removeFromSuperlayer()
                    contentMaskBackdrop.removeFromSuperlayer()
                    contentMaskParent.removeFromSuperlayer()
                    targetMaskBackdrop.removeFromSuperlayer()
                    targetMaskedFilter.removeFromSuperlayer()
                    root.backgroundColor = originalRootBackground
                    for (layer, wasHidden) in existingLayerStates {
                        layer.isHidden = wasHidden
                    }
                    engine.renderFrame()
                    compositionProbeResult = "ordered=\(composited),pixels=\(pixels.map { $0.map(String.init).joined(separator: ",") }.joined(separator: ";")),failures=\(renderer.compositionFilterFailureCount),after=\(renderer.activeCompositionResourceCount)"
                } catch {
                    compositionProbeResult = "error: \(error)"
                }
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

                let emptyBaselines: [[UInt8]]
                do {
                    emptyBaselines = try await renderer.readbackPixels(at: [
                        CGPoint(x: 205, y: 160),
                        CGPoint(x: 300, y: 220),
                        CGPoint(x: 75, y: 275),
                        CGPoint(x: 165, y: 275),
                        CGPoint(x: 160, y: 160),
                        CGPoint(x: 200, y: 160),
                    ])
                } catch {
                    shadowProbeResult = "error: \(error)"
                    return
                }

                let silhouetteParent = CALayer()
                silhouetteParent.bounds = CGRect(x: 0, y: 0, width: 80, height: 50)
                silhouetteParent.position = CGPoint(x: 100, y: 140)
                silhouetteParent.zPosition = 100
                silhouetteParent.shadowColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                silhouetteParent.shadowOpacity = 1
                silhouetteParent.shadowOffset = CGSize(width: 80, height: 0)
                silhouetteParent.shadowRadius = 0

                let silhouetteChild = CALayer()
                silhouetteChild.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
                silhouetteChild.position = CGPoint(x: 20, y: 25)
                silhouetteChild.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                let silhouetteAnimation = CABasicAnimation(keyPath: "position")
                silhouetteAnimation.fromValue = CGPoint(x: 20, y: 25)
                silhouetteAnimation.toValue = CGPoint(x: 60, y: 25)
                silhouetteAnimation.duration = 1
                silhouetteAnimation.speed = 0
                silhouetteAnimation.timeOffset = 0
                silhouetteAnimation.fillMode = .both
                silhouetteAnimation.isRemovedOnCompletion = false
                silhouetteChild.add(silhouetteAnimation, forKey: "shadowSilhouettePosition")
                silhouetteParent.addSublayer(silhouetteChild)

                let emptyContent = CALayer()
                emptyContent.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                emptyContent.position = CGPoint(x: 260, y: 80)
                emptyContent.zPosition = 100
                emptyContent.shadowColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                emptyContent.shadowOpacity = 1
                emptyContent.shadowOffset = CGSize(width: 40, height: 0)
                emptyContent.shadowRadius = 0

                let imageData = Data([
                    0, 0, 0, 0,
                    255, 255, 255, 255,
                    0, 0, 0, 0,
                ])
                let imageProvider = CGDataProvider(data: imageData)
                guard let transparentImage = CGImage(
                    width: 3,
                    height: 1,
                    bitsPerComponent: 8,
                    bitsPerPixel: 32,
                    bytesPerRow: 12,
                    space: .deviceRGB,
                    bitmapInfo: CGBitmapInfo(
                        rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                    ),
                    provider: imageProvider,
                    decode: nil,
                    shouldInterpolate: false,
                    intent: .defaultIntent
                ) else {
                    shadowProbeResult = "error: transparent image unavailable"
                    return
                }
                let imageContent = CALayer()
                imageContent.bounds = CGRect(x: 0, y: 0, width: 60, height: 20)
                imageContent.position = CGPoint(x: 100, y: 25)
                imageContent.zPosition = 100
                imageContent.contents = transparentImage
                imageContent.shadowColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                imageContent.shadowOpacity = 1
                imageContent.shadowOffset = CGSize(width: 40, height: 0)
                imageContent.shadowRadius = 0
                root.addSublayer(silhouetteParent)
                root.addSublayer(emptyContent)
                root.addSublayer(imageContent)

                engine.renderFrame()
                do {
                    let pixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 160, y: 160),
                        CGPoint(x: 205, y: 160),
                        CGPoint(x: 300, y: 220),
                        CGPoint(x: 75, y: 275),
                        CGPoint(x: 140, y: 275),
                        CGPoint(x: 165, y: 275),
                    ])
                    result += ";" + pixels[0].map(String.init).joined(separator: ",")
                    result += ";emptyRegion=\(pixels[1] == emptyBaselines[0])"
                    result += ";emptyLayer=\(pixels[2] == emptyBaselines[1])"
                    result += ";imageEdges=\(pixels[3] == emptyBaselines[2] && pixels[5] == emptyBaselines[3])"
                    let imageCenterIsShadow = pixels[4][0] >= 240
                        && pixels[4][1] <= 5
                        && pixels[4][2] <= 5
                        && pixels[4][3] == 255
                    result += ";imageCenter=\(imageCenterIsShadow)"

                    if let storedAnimation = silhouetteChild.animation(
                        forKey: "shadowSilhouettePosition"
                    ) {
                        storedAnimation.timeOffset = 1
                        engine.renderFrame()
                        let animatedPixels = try await renderer.readbackPixels(at: [
                            CGPoint(x: 160, y: 160),
                            CGPoint(x: 200, y: 160),
                        ])
                        let newPositionIsShadow = animatedPixels[1][0] >= 240
                            && animatedPixels[1][1] <= 5
                            && animatedPixels[1][2] <= 5
                            && animatedPixels[1][3] == 255
                        let oldPositionCleared = animatedPixels[0] == emptyBaselines[4]
                        result += ";animatedSilhouette=\(oldPositionCleared && newPositionIsShadow)"
                    } else {
                        result = "error: silhouette animation unavailable"
                    }
                } catch {
                    result = "error: \(error)"
                }

                silhouetteParent.removeFromSuperlayer()
                emptyContent.removeFromSuperlayer()
                imageContent.removeFromSuperlayer()
                engine.renderFrame()
                shadowProbeResult = result
            }
        })
        h.expose("beginReplicatorProbe", action: {
            Task { @MainActor in
                replicatorProbeResult = "running"
                let engine = CAAnimationEngine.shared
                engine.pause()
                guard let root = rootLayerRef,
                      let renderer = engine.renderer as? CAWebGPURenderer else {
                    replicatorProbeResult = "error: root layer or renderer unavailable"
                    return
                }

                let replicator = CAReplicatorLayer()
                CATransaction.begin()
                CATransaction.setDisableActions(true)
                replicator.bounds = root.bounds
                replicator.position = root.position
                replicator.zPosition = 100
                replicator.instanceCount = 2
                replicator.instanceTransform = CATransform3DMakeTranslation(40, 0, 0)
                replicator.instanceColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                replicator.instanceRedOffset = -1
                replicator.instanceGreenOffset = 1

                let background = CALayer()
                background.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
                background.position = CGPoint(x: 30, y: 140)
                background.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                replicator.addSublayer(background)

                let bordered = CALayer()
                bordered.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
                bordered.position = CGPoint(x: 100, y: 140)
                bordered.borderColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                bordered.borderWidth = 3
                replicator.addSublayer(bordered)

                let shape = CAShapeLayer()
                shape.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
                shape.position = CGPoint(x: 170, y: 140)
                let path = CGMutablePath()
                path.addRect(shape.bounds)
                shape.path = path
                shape.fillColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                replicator.addSublayer(shape)

                let imageData = Data(repeating: 255, count: 8 * 8 * 4)
                guard let image = CGImage(
                    width: 8,
                    height: 8,
                    bitsPerComponent: 8,
                    bitsPerPixel: 32,
                    bytesPerRow: 8 * 4,
                    space: .deviceRGB,
                    bitmapInfo: CGBitmapInfo(
                        rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                    ),
                    provider: CGDataProvider(data: imageData),
                    decode: nil,
                    shouldInterpolate: false,
                    intent: .defaultIntent
                ) else {
                    CATransaction.commit()
                    replicatorProbeResult = "error: image unavailable"
                    return
                }
                let imageLayer = CALayer()
                imageLayer.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
                imageLayer.position = CGPoint(x: 240, y: 140)
                imageLayer.contents = image
                replicator.addSublayer(imageLayer)

                let gradient = CAGradientLayer()
                gradient.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
                gradient.position = CGPoint(x: 310, y: 140)
                gradient.colors = [
                    CGColor(red: 1, green: 1, blue: 1, alpha: 1),
                    CGColor(red: 1, green: 1, blue: 1, alpha: 1),
                ]
                replicator.addSublayer(gradient)

                root.addSublayer(replicator)
                CATransaction.commit()
                engine.renderFrame()
                do {
                    let pixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 30, y: 160),
                        CGPoint(x: 70, y: 160),
                        CGPoint(x: 95, y: 160),
                        CGPoint(x: 135, y: 160),
                        CGPoint(x: 170, y: 160),
                        CGPoint(x: 210, y: 160),
                        CGPoint(x: 240, y: 160),
                        CGPoint(x: 280, y: 160),
                        CGPoint(x: 310, y: 160),
                        CGPoint(x: 350, y: 160),
                    ])
                    let colorsMatch = pixels == [
                        [255, 0, 0, 255],
                        [0, 255, 0, 255],
                        [255, 0, 0, 255],
                        [0, 255, 0, 255],
                        [255, 0, 0, 255],
                        [0, 255, 0, 255],
                        [255, 0, 0, 255],
                        [0, 255, 0, 255],
                        [255, 0, 0, 255],
                        [0, 255, 0, 255],
                    ]

                    replicator.instanceCount = 0
                    engine.renderFrame()
                    let emptyPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 30, y: 160),
                        CGPoint(x: 70, y: 160),
                        CGPoint(x: 95, y: 160),
                        CGPoint(x: 135, y: 160),
                        CGPoint(x: 170, y: 160),
                        CGPoint(x: 210, y: 160),
                        CGPoint(x: 240, y: 160),
                        CGPoint(x: 280, y: 160),
                        CGPoint(x: 310, y: 160),
                        CGPoint(x: 350, y: 160),
                    ])
                    let zeroCountMatches = emptyPixels.allSatisfy { pixel in
                        pixel == [26, 26, 38, 255]
                    }

                    replicator.removeFromSuperlayer()
                    CATransaction.begin()
                    CATransaction.setDisableActions(true)
                    let delayedReplicator = CAReplicatorLayer()
                    delayedReplicator.bounds = root.bounds
                    delayedReplicator.position = root.position
                    delayedReplicator.zPosition = 100
                    delayedReplicator.instanceCount = 2
                    delayedReplicator.instanceDelay = 0.5
                    delayedReplicator.instanceTransform = CATransform3DMakeTranslation(40, 0, 0)

                    let animatedLayer = CALayer()
                    animatedLayer.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
                    animatedLayer.position = CGPoint(x: 40, y: 200)
                    animatedLayer.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                    delayedReplicator.addSublayer(animatedLayer)
                    root.addSublayer(delayedReplicator)
                    CATransaction.commit()

                    let opacity = CABasicAnimation(keyPath: "opacity")
                    opacity.fromValue = Float(0)
                    opacity.toValue = Float(1)
                    opacity.duration = 2
                    opacity.beginTime = animatedLayer.convertTime(
                        CACurrentMediaTime(),
                        from: nil
                    ) - 1
                    opacity.fillMode = .both
                    opacity.isRemovedOnCompletion = false
                    animatedLayer.add(opacity, forKey: "replicatorDelayOpacity")

                    engine.renderFrame()
                    let delayedPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 40, y: 100),
                        CGPoint(x: 80, y: 100),
                    ])
                    let delayMatches = Int(delayedPixels[0][0]) > Int(delayedPixels[1][0]) + 40
                        && delayedPixels.allSatisfy { $0[3] == 255 }
                    delayedReplicator.removeFromSuperlayer()

                    func makeDelayedColorReplicator(
                        x: CGFloat,
                        configure: (CALayer) -> Void
                    ) -> CAReplicatorLayer {
                        let result = CAReplicatorLayer()
                        result.bounds = root.bounds
                        result.position = root.position
                        result.zPosition = 100
                        result.instanceCount = 2
                        result.instanceDelay = 0.5
                        result.instanceTransform = CATransform3DMakeTranslation(40, 0, 0)

                        let source = CALayer()
                        source.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
                        source.position = CGPoint(x: x, y: 250)
                        source.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                        configure(source)
                        result.addSublayer(source)

                        let color = CABasicAnimation(keyPath: "backgroundColor")
                        color.fromValue = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
                        color.toValue = CGColor(red: 0, green: 0, blue: 1, alpha: 1)
                        color.duration = 2
                        color.beginTime = source.convertTime(CACurrentMediaTime(), from: nil) - 1
                        color.fillMode = .both
                        color.isRemovedOnCompletion = false
                        source.add(color, forKey: "replicatorDelayedColor")
                        return result
                    }

                    CATransaction.begin()
                    CATransaction.setDisableActions(true)
                    let filteredReplicator = makeDelayedColorReplicator(x: 40) { source in
                        source.filters = [CAFilter.brightness(0)]
                    }
                    root.addSublayer(filteredReplicator)
                    CATransaction.commit()
                    engine.renderFrame()
                    let filteredPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 40, y: 50),
                        CGPoint(x: 80, y: 50),
                    ])
                    let filterMatches = Int(filteredPixels[0][2]) > Int(filteredPixels[1][2]) + 40
                        && Int(filteredPixels[0][0]) + 40 < Int(filteredPixels[1][0])
                    filteredReplicator.removeFromSuperlayer()

                    CATransaction.begin()
                    CATransaction.setDisableActions(true)
                    let shadowReplicator = CAReplicatorLayer()
                    shadowReplicator.bounds = root.bounds
                    shadowReplicator.position = root.position
                    shadowReplicator.zPosition = 100
                    shadowReplicator.instanceCount = 2
                    shadowReplicator.instanceTransform = CATransform3DMakeTranslation(40, 0, 0)
                    let shadowSource = CALayer()
                    shadowSource.bounds = CGRect(x: 0, y: 0, width: 12, height: 12)
                    shadowSource.position = CGPoint(x: 160, y: 250)
                    let shadowPath = CGMutablePath()
                    shadowPath.addRect(shadowSource.bounds)
                    shadowSource.shadowPath = shadowPath
                    shadowSource.shadowColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
                    shadowSource.shadowOpacity = 1
                    shadowSource.shadowRadius = 0
                    shadowSource.shadowOffset = .zero
                    shadowReplicator.addSublayer(shadowSource)
                    root.addSublayer(shadowReplicator)
                    CATransaction.commit()
                    engine.renderFrame()
                    let shadowPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 160, y: 50),
                        CGPoint(x: 200, y: 50),
                    ])
                    let shadowMatches = shadowPixels.allSatisfy { pixel in
                        pixel[0] >= 240 && pixel[1] >= 240 && pixel[2] >= 240 && pixel[3] == 255
                    }
                    shadowReplicator.removeFromSuperlayer()

                    CATransaction.begin()
                    CATransaction.setDisableActions(true)
                    let rasterizedReplicator = makeDelayedColorReplicator(x: 280) { source in
                        source.shouldRasterize = true
                    }
                    root.addSublayer(rasterizedReplicator)
                    CATransaction.commit()
                    engine.renderFrame()
                    let rasterizedPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 280, y: 50),
                        CGPoint(x: 320, y: 50),
                    ])
                    let rasterMatches = Int(rasterizedPixels[0][2]) > Int(rasterizedPixels[1][2]) + 40
                        && Int(rasterizedPixels[0][0]) + 40 < Int(rasterizedPixels[1][0])
                    rasterizedReplicator.removeFromSuperlayer()

                    replicatorProbeResult = "content=\(colorsMatch),zero=\(zeroCountMatches),delay=\(delayMatches),filter=\(filterMatches),shadow=\(shadowMatches),raster=\(rasterMatches)"
                } catch {
                    replicatorProbeResult = "error: \(error)"
                }
                replicator.removeFromSuperlayer()
                engine.renderFrame()
            }
        })
        h.expose("beginEmitterProbe", action: {
            Task { @MainActor in
                emitterProbeResult = "running"
                let engine = CAAnimationEngine.shared
                engine.pause()
                guard let root = rootLayerRef,
                      let renderer = engine.renderer as? CAWebGPURenderer else {
                    emitterProbeResult = "error: root layer or renderer unavailable"
                    return
                }
                func makeParticleImage(width: Int, height: Int, data: Data) -> CGImage? {
                    CGImage(
                        width: width,
                        height: height,
                        bitsPerComponent: 8,
                        bitsPerPixel: 32,
                        bytesPerRow: width * 4,
                        space: .deviceRGB,
                        bitmapInfo: CGBitmapInfo(
                            rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                        ),
                        provider: CGDataProvider(data: data),
                        decode: nil,
                        shouldInterpolate: false,
                        intent: .defaultIntent
                    )
                }
                let particleData = Data(repeating: 255, count: 8 * 8 * 4)
                guard let particleImage = makeParticleImage(
                    width: 8,
                    height: 8,
                    data: particleData
                ) else {
                    emitterProbeResult = "error: particle image unavailable"
                    return
                }

                func makeEmitter(
                    x: CGFloat,
                    color: CGColor,
                    shape: CAEmitterLayerEmitterShape,
                    mode: CAEmitterLayerEmitterMode,
                    size: CGSize,
                    latitude: CGFloat,
                    longitude: CGFloat
                ) -> (layer: CAEmitterLayer, cell: CAEmitterCell) {
                    let cell = CAEmitterCell()
                    cell.birthRate = 10
                    cell.lifetime = 5
                    cell.velocity = 10
                    cell.scale = 1
                    cell.contents = particleImage
                    cell.color = color
                    cell.emissionLatitude = latitude
                    cell.emissionLongitude = longitude

                    let layer = CAEmitterLayer()
                    layer.bounds = CGRect(x: 0, y: 0, width: 40, height: 40)
                    layer.position = CGPoint(x: x, y: 140)
                    layer.zPosition = 100
                    layer.emitterPosition = CGPoint(x: 10, y: 10)
                    layer.emitterShape = shape
                    layer.emitterMode = mode
                    layer.emitterSize = size
                    layer.emitterCells = [cell]
                    return (layer, cell)
                }

                let firstEmitter = makeEmitter(
                    x: 70,
                    color: CGColor(red: 1, green: 0, blue: 0, alpha: 1),
                    shape: .rectangle,
                    mode: .outline,
                    size: CGSize(width: 20, height: 12),
                    latitude: 0,
                    longitude: 0
                )
                let secondEmitter = makeEmitter(
                    x: 170,
                    color: CGColor(red: 0, green: 1, blue: 0, alpha: 1),
                    shape: .sphere,
                    mode: .surface,
                    size: CGSize(width: 10, height: 1),
                    latitude: .pi / 2,
                    longitude: .pi / 2
                )
                let first = firstEmitter.layer
                let second = secondEmitter.layer
                var transientEmitterLayers: [CAEmitterLayer] = []
                root.addSublayer(first)
                root.addSublayer(second)

                engine.renderFrame()
                do {
                    try await Task.sleep(for: .milliseconds(100))
                    engine.renderFrame()
                    let firstCount = renderer.activeParticleCount(for: first)
                    let secondCount = renderer.activeParticleCount(for: second)
                    guard let firstPosition = renderer.activeParticlePositions(for: first).first,
                          let secondPosition = renderer.activeParticlePositions(for: second).first,
                          let firstVelocity = renderer.activeParticleVelocities(for: first).first,
                          let secondVelocity = renderer.activeParticleVelocities(for: second).first else {
                        first.removeFromSuperlayer()
                        second.removeFromSuperlayer()
                        engine.renderFrame()
                        emitterProbeResult = "error: particle diagnostics unavailable"
                        return
                    }
                    let firstOnOutline = abs(abs(firstPosition.x - 10) - 10) < 0.001
                        || abs(abs(firstPosition.y - 10) - 6) < 0.001
                    let sphereOffset = secondPosition - SIMD3<Float>(10, 10, 0)
                    let sphereRadius = sqrt(
                        sphereOffset.x * sphereOffset.x
                            + sphereOffset.y * sphereOffset.y
                            + sphereOffset.z * sphereOffset.z
                    )
                    let geometryMatches = firstOnOutline && abs(sphereRadius - 10) < 0.001
                    let directionsMatch = abs(firstVelocity.x) < 0.001
                        && abs(firstVelocity.y) < 0.001
                        && firstVelocity.z > 9.999
                        && abs(secondVelocity.x) < 0.001
                        && secondVelocity.y > 9.999
                        && abs(secondVelocity.z) < 0.001
                    var result = "before=\(firstCount),\(secondCount),states=\(renderer.activeEmitterStateCount),geometry=\(geometryMatches),directions=\(directionsMatch),failures=\(renderer.emitterSpawnFailureCount)"

                    firstEmitter.cell.emissionLatitude = .pi / 2
                    firstEmitter.cell.emissionLongitude = 0
                    try await Task.sleep(for: .milliseconds(100))
                    engine.renderFrame()
                    firstEmitter.cell.birthRate = 0
                    secondEmitter.cell.birthRate = 0

                    first.renderMode = .oldestFirst
                    engine.renderFrame()
                    let oldestFirst = renderer.lastRenderedParticleSequences(for: first) == [0, 1]
                    first.renderMode = .oldestLast
                    engine.renderFrame()
                    let oldestLast = renderer.lastRenderedParticleSequences(for: first) == [1, 0]
                    first.renderMode = .backToFront
                    engine.renderFrame()
                    let backToFront = renderer.lastRenderedParticleSequences(for: first) == [1, 0]
                    first.renderMode = .additive
                    engine.renderFrame()
                    let additive = renderer.lastEmitterRenderUsedAdditiveBlending(for: first)
                        && renderer.lastRenderedParticleSequences(for: first) == [0, 1]
                    first.renderMode = CAEmitterLayerRenderMode(rawValue: "unsupported")
                    engine.renderFrame()
                    let unknownRejected = renderer.lastRenderedParticleSequences(for: first).isEmpty
                        && renderer.emitterRenderFailureCount == 1
                    result += ";orders=\(oldestFirst && oldestLast && backToFront),additive=\(additive),unknown=\(unknownRejected)"

                    first.removeFromSuperlayer()
                    engine.renderFrame()
                    result += ";after=\(renderer.activeParticleCount(for: first)),\(renderer.activeParticleCount(for: second)),states=\(renderer.activeEmitterStateCount)"

                    second.removeFromSuperlayer()
                    engine.renderFrame()
                    let sourceBlendEmitter = makeEmitter(
                        x: 250,
                        color: CGColor(red: 1, green: 0, blue: 0, alpha: 0.5),
                        shape: .point,
                        mode: .volume,
                        size: .zero,
                        latitude: 0,
                        longitude: 0
                    )
                    let additiveBlendEmitter = makeEmitter(
                        x: 310,
                        color: CGColor(red: 1, green: 0, blue: 0, alpha: 0.5),
                        shape: .point,
                        mode: .volume,
                        size: .zero,
                        latitude: 0,
                        longitude: 0
                    )
                    sourceBlendEmitter.cell.birthRate = 20
                    sourceBlendEmitter.cell.velocity = 0
                    additiveBlendEmitter.cell.birthRate = 20
                    additiveBlendEmitter.cell.velocity = 0
                    additiveBlendEmitter.layer.renderMode = .additive
                    transientEmitterLayers = [
                        sourceBlendEmitter.layer,
                        additiveBlendEmitter.layer,
                    ]
                    root.addSublayer(sourceBlendEmitter.layer)
                    root.addSublayer(additiveBlendEmitter.layer)
                    engine.renderFrame()
                    try await Task.sleep(for: .milliseconds(100))
                    engine.renderFrame()
                    let blendPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 240, y: 170),
                        CGPoint(x: 300, y: 170),
                    ])
                    let blendPixelsMatch = blendPixels.count == 2
                        && blendPixels.allSatisfy { $0.count >= 4 }
                        && blendPixels[1][0] > blendPixels[0][0] + 30
                    for layer in transientEmitterLayers {
                        layer.removeFromSuperlayer()
                    }
                    engine.renderFrame()
                    transientEmitterLayers.removeAll(keepingCapacity: true)

                    var croppedData = Data()
                    croppedData.reserveCapacity(8 * 8 * 4)
                    for _ in 0..<8 {
                        for x in 0..<8 {
                            croppedData.append(contentsOf: x < 4
                                ? [255, 0, 0, 255]
                                : [0, 255, 0, 255])
                        }
                    }
                    guard let croppedImage = makeParticleImage(
                        width: 8,
                        height: 8,
                        data: croppedData
                    ) else {
                        emitterProbeResult = "error: cropped particle image unavailable"
                        return
                    }
                    let croppedEmitter = makeEmitter(
                        x: 370,
                        color: CGColor(red: 1, green: 1, blue: 1, alpha: 1),
                        shape: .point,
                        mode: .volume,
                        size: .zero,
                        latitude: 0,
                        longitude: 0
                    )
                    croppedEmitter.cell.contents = croppedImage
                    croppedEmitter.cell.contentsRect = CGRect(x: 0.5, y: 0, width: 0.5, height: 1)
                    croppedEmitter.cell.contentsScale = 2
                    croppedEmitter.cell.velocity = 0
                    transientEmitterLayers.append(croppedEmitter.layer)
                    root.addSublayer(croppedEmitter.layer)
                    engine.renderFrame()
                    try await Task.sleep(for: .milliseconds(100))
                    engine.renderFrame()
                    let croppedPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 360, y: 170),
                        CGPoint(x: 363, y: 170),
                    ])
                    let croppedImageMatches = croppedPixels.count == 2
                        && croppedPixels.allSatisfy { $0.count >= 4 }
                        && croppedPixels[0][1] > 200
                        && croppedPixels[0][0] < 40
                        && Int(croppedPixels[0][1]) > Int(croppedPixels[1][1]) + 100
                    croppedEmitter.layer.removeFromSuperlayer()
                    engine.renderFrame()
                    transientEmitterLayers.removeAll(keepingCapacity: true)

                    var minificationData = Data()
                    minificationData.reserveCapacity(8 * 8 * 4)
                    for _ in 0..<8 {
                        for x in 0..<8 {
                            minificationData.append(contentsOf: x < 7
                                ? [255, 0, 0, 255]
                                : [0, 0, 0, 255])
                        }
                    }
                    guard let minificationImage = makeParticleImage(
                        width: 8,
                        height: 8,
                        data: minificationData
                    ) else {
                        emitterProbeResult = "error: minification image unavailable"
                        return
                    }
                    let linearEmitter = makeEmitter(
                        x: 250,
                        color: CGColor(red: 1, green: 1, blue: 1, alpha: 1),
                        shape: .point,
                        mode: .volume,
                        size: .zero,
                        latitude: 0,
                        longitude: 0
                    )
                    let trilinearEmitter = makeEmitter(
                        x: 310,
                        color: CGColor(red: 1, green: 1, blue: 1, alpha: 1),
                        shape: .point,
                        mode: .volume,
                        size: .zero,
                        latitude: 0,
                        longitude: 0
                    )
                    for emitter in [linearEmitter, trilinearEmitter] {
                        emitter.cell.contents = minificationImage
                        emitter.cell.birthRate = 20
                        emitter.cell.velocity = 0
                        emitter.cell.scale = 0.25
                    }
                    linearEmitter.cell.minificationFilter = CALayerContentsFilter.linear.rawValue
                    trilinearEmitter.cell.minificationFilter = CALayerContentsFilter.trilinear.rawValue
                    trilinearEmitter.cell.minificationFilterBias = 4
                    transientEmitterLayers = [linearEmitter.layer, trilinearEmitter.layer]
                    root.addSublayer(linearEmitter.layer)
                    root.addSublayer(trilinearEmitter.layer)
                    engine.renderFrame()
                    try await Task.sleep(for: .milliseconds(100))
                    engine.renderFrame()
                    let minificationPixels = try await renderer.readbackPixels(at: [
                        CGPoint(x: 240, y: 170),
                        CGPoint(x: 300, y: 170),
                    ])
                    let minificationMatches = minificationPixels.count == 2
                        && minificationPixels.allSatisfy { $0.count >= 4 }
                        && Int(minificationPixels[0][0]) > Int(minificationPixels[1][0]) + 10
                        && minificationPixels[1][0] > 150
                    for layer in transientEmitterLayers {
                        layer.removeFromSuperlayer()
                    }
                    engine.renderFrame()
                    transientEmitterLayers.removeAll(keepingCapacity: true)

                    let invisibleEmitter = makeEmitter(
                        x: 100,
                        color: CGColor(red: 1, green: 0, blue: 0, alpha: 1),
                        shape: .point,
                        mode: .volume,
                        size: .zero,
                        latitude: 0,
                        longitude: 0
                    )
                    invisibleEmitter.cell.contents = nil
                    invisibleEmitter.cell.velocity = 0
                    let unsupportedEmitter = makeEmitter(
                        x: 160,
                        color: CGColor(red: 1, green: 0, blue: 0, alpha: 1),
                        shape: .point,
                        mode: .volume,
                        size: .zero,
                        latitude: 0,
                        longitude: 0
                    )
                    unsupportedEmitter.cell.contents = "unsupported"
                    unsupportedEmitter.cell.velocity = 0
                    transientEmitterLayers = [invisibleEmitter.layer, unsupportedEmitter.layer]
                    root.addSublayer(invisibleEmitter.layer)
                    root.addSublayer(unsupportedEmitter.layer)
                    engine.renderFrame()
                    try await Task.sleep(for: .milliseconds(100))
                    engine.renderFrame()
                    let invisibleMatches = renderer.activeParticleCount(for: invisibleEmitter.layer) == 1
                        && renderer.lastRenderedParticleSequences(for: invisibleEmitter.layer).isEmpty
                    let unsupportedRejected = renderer.activeParticleCount(for: unsupportedEmitter.layer) == 0
                        && renderer.emitterSpawnFailureCount == 1
                    for layer in transientEmitterLayers {
                        layer.removeFromSuperlayer()
                    }
                    engine.renderFrame()

                    let childCell = CAEmitterCell()
                    childCell.birthRate = 20
                    childCell.beginTime = 0.025
                    childCell.lifetime = 1
                    childCell.velocity = 5
                    childCell.scale = 2
                    childCell.color = CGColor(red: 0.5, green: 1, blue: 1, alpha: 0.5)
                    childCell.contents = particleImage
                    let parentEmitter = makeEmitter(
                        x: 220,
                        color: CGColor(red: 1, green: 0.5, blue: 0.25, alpha: 1),
                        shape: .point,
                        mode: .volume,
                        size: .zero,
                        latitude: .pi / 2,
                        longitude: 0
                    )
                    parentEmitter.cell.lifetime = 2
                    parentEmitter.cell.scale = 0.5
                    parentEmitter.cell.emitterCells = [childCell]
                    transientEmitterLayers = [parentEmitter.layer]
                    root.addSublayer(parentEmitter.layer)
                    engine.renderFrame()
                    try await Task.sleep(for: .milliseconds(100))
                    engine.renderFrame()
                    let childWasDelayed = !renderer.activeParticleGenerations(
                        for: parentEmitter.layer
                    ).contains(1)
                    parentEmitter.cell.birthRate = 0
                    try await Task.sleep(for: .milliseconds(100))
                    engine.renderFrame()
                    let generations = renderer.activeParticleGenerations(for: parentEmitter.layer)
                    let positions = renderer.activeParticlePositions(for: parentEmitter.layer)
                    let velocities = renderer.activeParticleVelocities(for: parentEmitter.layer)
                    let colors = renderer.activeParticleColors(for: parentEmitter.layer)
                    let scales = renderer.activeParticleScales(for: parentEmitter.layer)
                    let childMatches: Bool
                    if let childIndex = generations.firstIndex(of: 1),
                       positions.indices.contains(childIndex),
                       velocities.indices.contains(childIndex),
                       colors.indices.contains(childIndex),
                       scales.indices.contains(childIndex) {
                        let childPosition = positions[childIndex]
                        let childVelocity = velocities[childIndex]
                        let childColor = colors[childIndex]
                        childMatches = childWasDelayed
                            && abs(childPosition.x - 10.5) < 0.15
                            && abs(childPosition.y - 10) < 0.01
                            && childVelocity.x > 4.99
                            && abs(childVelocity.y) < 0.01
                            && abs(childVelocity.z) < 0.01
                            && abs(childColor.x - 0.5) < 0.01
                            && abs(childColor.y - 0.5) < 0.01
                            && abs(childColor.z - 0.25) < 0.01
                            && abs(childColor.w - 0.5) < 0.01
                            && abs(scales[childIndex] - 1) < 0.01
                    } else {
                        childMatches = false
                    }
                    parentEmitter.layer.removeFromSuperlayer()
                    engine.renderFrame()
                    result += ";image=\(croppedImageMatches),sampling=\(minificationMatches),nil=\(invisibleMatches),rejected=\(unsupportedRejected),child=\(childMatches);blend=\(blendPixelsMatch),final=\(renderer.activeEmitterStateCount)"
                    emitterProbeResult = result
                } catch {
                    first.removeFromSuperlayer()
                    second.removeFromSuperlayer()
                    for layer in transientEmitterLayers {
                        layer.removeFromSuperlayer()
                    }
                    engine.renderFrame()
                    emitterProbeResult = "error: \(error)"
                }
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
                unsupportedBuiltInTransitioningLayerRef?.removeFromSuperlayer()
                unsupportedTransitionSubtypeLayerRef?.removeFromSuperlayer()
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
