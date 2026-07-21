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
    getTileDrawCount: () => number;
    getTileState: () => string;
    isEngineRunning: () => boolean;
    getPixelReadback: () => string;
    getTransitionSourceCaptureCount: () => number;
    getActiveTransitionSourceTextureCount: () => number;
    removeTransition: () => void;
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
        expect(await h.getSublayerCount(), "root sublayer count").toBe(5);
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
        expect(await h.getTileState()).toBe("delegate=true,bounds=80.0x80.0");
        await expect.poll(() => h.getTileDrawCount(), { timeout: 2_000 }).toBeGreaterThan(0);
        await expect.poll(() => h.getTransitionSourceCaptureCount()).toBe(1);
        expect(await h.getActiveTransitionSourceTextureCount()).toBe(1);
        await h.beginPixelReadback();

        await expect.poll(() => h.getPixelReadback()).not.toBe("pending");
        expect(await h.getPixelReadback()).toBe(
            "255,0,0,255;0,255,0,255;0,0,255,255;26,26,38,255;255,0,255,255;106,10,78,255"
        );

        await h.removeTransition();
        await expect.poll(() => h.getActiveTransitionSourceTextureCount()).toBe(0);
    });
});
