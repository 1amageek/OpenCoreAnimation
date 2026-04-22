// OpenCoreAnimation WASM smoke-test executable.
//
// Boots a `CAAnimationEngine` against a freshly created <canvas>, attaches a
// minimal `CALayer` tree (3 solid-color sublayers on a dark root), and runs
// the display link for a few frames. Success is asserted from JS through
// `window.__oca_test`, which reads Swift-side state rather than trying to
// pixel-read a WebGPU canvas (swap textures are destroyed on present, so
// `drawImage` into a 2D context is unreliable in most browsers).

import Foundation
import JavaScriptKit
import JavaScriptEventLoop
import OpenCoreAnimation

// MARK: - Captured state (populated by performSetup)

nonisolated(unsafe) var statusText: String = "initializing"
nonisolated(unsafe) var canvasWidth: Int = 0
nonisolated(unsafe) var canvasHeight: Int = 0
nonisolated(unsafe) var sublayerCount: Int = 0
nonisolated(unsafe) var rootLayerRef: CALayer?

// Retain JSClosures so JavaScriptKit does not deallocate them while JS still
// holds references through window.__oca_test.
nonisolated(unsafe) var installedClosures: [JSClosure] = []

// Canvas dimensions are inlined at each use site as `400` / `300`. Do NOT
// hoist them into file-scope `let` constants — in the WASI reactor ABI used
// here, Swift module-scope globals are driven by the lazy-once runtime path
// and the first read inside the `JavaScriptEventLoop` Task has been observed
// returning 0. Inlining the literal sidesteps the initializer race. See
// memory: feedback_wasm_reactor_global_init_race.

// MARK: - WASM entry point

@_cdecl("setup")
public func setup() {
    // Touching every `nonisolated(unsafe) var` before scheduling the Task
    // forces Swift's lazy-once global initializers to run on the main WASM
    // call so the async Task can't observe them uninitialised. Without this,
    // `statusText` is read before its initializer in the `getStatus`
    // JSClosure path and the JSString bridge traps with `RuntimeError:
    // unreachable` on the first `window.__oca_test.getStatus()` call from JS.
    statusText = "initializing"
    canvasWidth = 0
    canvasHeight = 0
    sublayerCount = 0
    rootLayerRef = nil
    installedClosures = []
    JavaScriptEventLoop.installGlobalExecutor()
    Task { await performSetup() }
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
    root.addSublayer(blue)

    rootLayerRef = root
    sublayerCount = root.sublayers?.count ?? 0

    engine.rootLayer = root
    engine.renderFrame()
    engine.start()

    statusText = "ready"
    print("OCASmoke ready: canvas \(canvasWidth)x\(canvasHeight), sublayers \(sublayerCount)")
}

// MARK: - JS harness

@MainActor
func installHarness() {
    let getStatus = JSClosure { _ -> JSValue in
        .string(statusText)
    }
    let getCanvasWidth = JSClosure { _ -> JSValue in
        .number(Double(canvasWidth))
    }
    let getCanvasHeight = JSClosure { _ -> JSValue in
        .number(Double(canvasHeight))
    }
    let getSublayerCount = JSClosure { _ -> JSValue in
        .number(Double(sublayerCount))
    }
    let isEngineRunning = JSClosure { _ -> JSValue in
        .boolean(CAAnimationEngine.shared.isRunning)
    }
    installedClosures = [getStatus, getCanvasWidth, getCanvasHeight, getSublayerCount, isEngineRunning]

    let harness = JSObject.global.Object.function!.new()
    harness.getStatus = .object(getStatus)
    harness.getCanvasWidth = .object(getCanvasWidth)
    harness.getCanvasHeight = .object(getCanvasHeight)
    harness.getSublayerCount = .object(getSublayerCount)
    harness.isEngineRunning = .object(isEngineRunning)
    JSObject.global.__oca_test = .object(harness)
}
