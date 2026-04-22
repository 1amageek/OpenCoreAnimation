import { test, expect, type Page } from "@playwright/test";

// Smoke-test the full CALayer → CAWebGPURenderer pipeline inside a real
// Chromium:
//   1. setCanvas(canvas) resolves (adapter + device available)
//   2. A CALayer tree with backgroundColor sublayers is built
//   3. engine.renderFrame() runs once without trapping
//   4. engine.start() schedules rAF and isRunning stays true
//
// Assertions go through the Swift-side harness (`window.__oca_test`) rather
// than pixel-reading the WebGPU canvas — swap textures are destroyed on
// present so `drawImage` into a 2D context is unreliable. The harness reads
// state Swift already owns, which is deterministic by construction.

async function waitForHarness(page: Page): Promise<void> {
    await page.waitForFunction(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        () => !!(window as unknown as { __oca_test?: unknown }).__oca_test,
        null,
        { timeout: 10_000 }
    );
}

async function getStatus(page: Page): Promise<string> {
    return await page.evaluate(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const h = (window as unknown as { __oca_test: { getStatus: () => string } }).__oca_test;
        return h.getStatus();
    });
}

async function waitForReady(page: Page): Promise<void> {
    // setCanvas + first renderFrame can trail the harness install by a couple
    // of rAF ticks — poll until the harness signals "ready".
    await expect.poll(async () => getStatus(page), { timeout: 30_000 })
        .toBe("ready");
}

test.describe("OpenCoreAnimation smoke", () => {
    test("boot: harness installs and reports ready", async ({ page }) => {
        await page.goto("/");
        await waitForHarness(page);
        await waitForReady(page);

        const { width, height, sublayers } = await page.evaluate(() => {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const h = (window as unknown as { __oca_test: {
                getCanvasWidth: () => number;
                getCanvasHeight: () => number;
                getSublayerCount: () => number;
            } }).__oca_test;
            return {
                width: h.getCanvasWidth(),
                height: h.getCanvasHeight(),
                sublayers: h.getSublayerCount(),
            };
        });

        expect(width, "canvas width").toBe(400);
        expect(height, "canvas height").toBe(300);
        expect(sublayers, "root sublayer count").toBe(3);
    });

    test("engine: display link is running", async ({ page }) => {
        await page.goto("/");
        await waitForHarness(page);
        await waitForReady(page);

        // engine.start() installs a CADisplayLink that schedules
        // requestAnimationFrame. isRunning stays true until stop()/deinit —
        // asserting it proves the start() call landed without trapping.
        const running = await page.evaluate(() => {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const h = (window as unknown as { __oca_test: {
                isEngineRunning: () => boolean;
            } }).__oca_test;
            return h.isEngineRunning();
        });
        expect(running, "CAAnimationEngine.isRunning after start()").toBe(true);
    });
});
