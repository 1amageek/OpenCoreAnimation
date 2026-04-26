// Smoke-test the full CALayer → CAWebGPURenderer pipeline inside a real
// Chromium:
//   1. setCanvas(canvas) resolves (adapter + device available)
//   2. A CALayer tree with backgroundColor sublayers is built
//   3. engine.renderFrame() runs once without trapping
//   4. engine.start() schedules rAF and isRunning stays true
//
// Assertions go through the Swift-side harness (`window.__oca_test`) via the
// typed proxy from `swift-wasm-testing`. Pixel-reading the WebGPU canvas is
// unreliable because swap textures are destroyed on present; the harness
// reads state Swift already owns, which is deterministic by construction.

import { test as base, expect, type Harness } from "swift-wasm-testing";

interface OCA extends Harness {
    getStatus: () => string;
    getCanvasWidth: () => number;
    getCanvasHeight: () => number;
    getSublayerCount: () => number;
    isEngineRunning: () => boolean;
}

const test = base;
test.use({ harnessGlobalName: "__oca_test" });

test.describe("OpenCoreAnimation smoke", () => {
    test("boot: harness installs and reports ready", async ({ harness }) => {
        const h = harness as unknown as {
            [K in keyof OCA]: (...a: Parameters<OCA[K]>) => Promise<ReturnType<OCA[K]>>;
        };

        expect(await h.getCanvasWidth(), "canvas width").toBe(400);
        expect(await h.getCanvasHeight(), "canvas height").toBe(300);
        expect(await h.getSublayerCount(), "root sublayer count").toBe(3);
    });

    test("engine: display link is running", async ({ harness }) => {
        const h = harness as unknown as {
            [K in keyof OCA]: (...a: Parameters<OCA[K]>) => Promise<ReturnType<OCA[K]>>;
        };

        // engine.start() installs a CADisplayLink that schedules rAF.
        // isRunning stays true until stop()/deinit — asserting it proves the
        // start() call landed without trapping.
        expect(await h.isEngineRunning(), "CAAnimationEngine.isRunning after start()").toBe(true);
    });
});
