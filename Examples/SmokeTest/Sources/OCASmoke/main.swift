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
import OpenCoreAnimation
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
nonisolated(unsafe) var tileDelegateRef: SmokeTileDelegate?
nonisolated(unsafe) var tileDrawCount: Int = 0
nonisolated(unsafe) var pixelReadbackResult: String = "pending"

final class SmokeTileDelegate: CALayerDelegate {
    func draw(_ layer: CALayer, in context: CGContext) {
        tileDrawCount += 1
        context.setFillColor(CGColor(red: 1, green: 0, blue: 1, alpha: 1))
        context.fill(layer.bounds)
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
            tileDelegateRef = nil
            tileDrawCount = 0
            pixelReadbackResult = "pending"
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
    transitioning.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
    root.addSublayer(transitioning)

    let transition = CATransition()
    transition.type = .fade
    transition.duration = 1
    transition.speed = 0
    transition.timeOffset = 0.5
    transitioning.add(transition, forKey: "browserCrossfade")
    transitioning.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)

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
                        CGPoint(x: 10, y: 10),
                        CGPoint(x: 200, y: 80),
                        CGPoint(x: 320, y: 80),
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
