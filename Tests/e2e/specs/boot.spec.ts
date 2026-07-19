// Smoke-test the full CALayer → CAWebGPURenderer pipeline inside a real
// Chromium:
//   1. setCanvas(canvas) resolves (adapter + device available)
//   2. A CALayer tree with backgroundColor sublayers is built
//   3. engine.renderFrame() runs once without trapping
//   4. engine.start() schedules rAF and isRunning stays true
//
// State assertions go through the Swift-side harness (`window.__oca_test`).
// The final assertion captures the compositor output so a command submission
// that renders no pixels cannot satisfy the smoke test.

import { test as base, expect, type Harness } from "swift-wasm-testing";

interface OCA extends Harness {
    getStatus: () => string;
    getCanvasWidth: () => number;
    getCanvasHeight: () => number;
    getSublayerCount: () => number;
    isEngineRunning: () => boolean;
    getPixelReadback: () => string;
    beginPixelReadback: () => void;
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

    test("rendering: WebGPU readback contains layer colors", async ({ harness }) => {
        const h = harness as unknown as {
            [K in keyof OCA]: (...a: Parameters<OCA[K]>) => Promise<ReturnType<OCA[K]>>;
        };
        expect(await h.getStatus()).toBe("ready");
        await h.beginPixelReadback();

        await expect.poll(() => h.getPixelReadback()).not.toBe("pending");
        expect(await h.getPixelReadback()).toBe(
            "255,0,0,255;0,255,0,255;26,26,38,255"
        );
    });
});
