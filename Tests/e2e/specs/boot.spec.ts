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
    getTransitionFilterProbeResult: () => string;
    getLayerFilterProbeResult: () => string;
    getTextProbeResult: () => string;
    getDelegateDrawProbeResult: () => string;
    getBooleanAnimationProbeResult: () => string;
    getContentsAnimationProbeResult: () => string;
    getRasterizationScaleProbeResult: () => string;
    getShadowPathKeyframeProbeResult: () => string;
    getShadowProbeResult: () => string;
    getDisplayLinkProbeResult: () => string;
    getEmitterProbeResult: () => string;
    getReplicatorProbeResult: () => string;
    getCompositionProbeResult: () => string;
    getTransformDepthProbeResult: () => string;
    getTransitionSourceCaptureCount: () => number;
    getTransitionTargetCaptureCount: () => number;
    getActiveTransitionTextureCount: () => number;
    getTransitionFilterDispatchCount: () => number;
    getTransitionFilterFailureCount: () => number;
    getTransitionRenderFailureCount: () => number;
    getFirstUncapturedGPUError: () => string;
    getActiveFilterResourceCount: () => number;
    getLayerFilterFailureCount: () => number;
    getCompositionFilterFailureCount: () => number;
    getActiveCompositionResourceCount: () => number;
    getActiveShadowResourceCount: () => number;
    getShadowRenderFailureCount: () => number;
    mutateTransitionTarget: () => void;
    exerciseUnsupportedTransitionFilter: () => void;
    exerciseUnsupportedBuiltInTransition: () => void;
    exerciseUnsupportedTransitionSubtype: () => void;
    beginTransitionFilterProbes: () => void;
    beginLayerFilterProbe: () => void;
    beginTextProbe: () => void;
    beginDelegateDrawProbe: () => void;
    beginBooleanAnimationProbe: () => void;
    beginContentsAnimationProbe: () => void;
    beginRasterizationScaleProbe: () => void;
    beginShadowPathKeyframeProbe: () => void;
    beginCompositionProbe: () => void;
    beginTransformDepthProbe: () => void;
    beginShadowProbe: () => void;
    beginDisplayLinkProbe: () => void;
    beginEmitterProbe: () => void;
    beginReplicatorProbe: () => void;
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
        expect(await h.getSublayerCount(), "root sublayer count").toBe(6);
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
        await expect.poll(() => h.getTransitionSourceCaptureCount()).toBe(2);
        expect(await h.getTransitionTargetCaptureCount()).toBe(2);
        expect(await h.getActiveTransitionTextureCount()).toBe(5);
        await expect.poll(() => h.getTransitionFilterDispatchCount()).toBeGreaterThan(0);
        expect(await h.getTransitionFilterFailureCount()).toBe(0);
        expect(await h.getTransitionRenderFailureCount()).toBe(0);
        await h.mutateTransitionTarget();
        expect(await h.getTransitionSourceCaptureCount()).toBe(2);
        expect(await h.getTransitionTargetCaptureCount()).toBe(2);
        await h.beginPixelReadback();

        await expect.poll(() => h.getPixelReadback()).not.toBe("pending");
        expect(await h.getFirstUncapturedGPUError()).toBe("none");
        expect(await h.getPixelReadback()).toBe(
            "255,0,0,255;0,255,0,255;0,0,255,255;26,26,38,255;255,0,255,255;77,13,83,255;191,255,64,255"
        );

        await h.beginTransitionFilterProbes();
        await expect.poll(() => h.getTransitionFilterProbeResult(), { timeout: 10_000 }).toContain("ripple=");
        expect(await h.getTransitionFilterProbeResult()).toBe(
            "dissolve=191,0,64,255;swipe=0,255,0,255;bars=0,0,255,255;mod=255,0,0,255;flash=0,128,128,255;copy=0,255,0,255;ripple=0,0,255,255"
        );
        expect(await h.getTransitionSourceCaptureCount()).toBe(9);
        expect(await h.getTransitionTargetCaptureCount()).toBe(9);
        expect(await h.getActiveTransitionTextureCount()).toBe(2);
        expect(await h.getTransitionFilterFailureCount()).toBe(0);

        await h.beginLayerFilterProbe();
        await expect.poll(() => h.getLayerFilterProbeResult(), { timeout: 10_000 }).toBe(
            "127,255,255,255;191,191,0,255;255,0,255,255;255,0,0,255;group=true,ungrouped=true,translucentGroup=true,translucentUngrouped=true,rejected=true,alphaFilter=true,alphaPixel=13,141,147,255"
        );
        expect(await h.getActiveFilterResourceCount()).toBe(0);
        expect(await h.getLayerFilterFailureCount()).toBe(1);

        await h.beginTextProbe();
        await expect.poll(() => h.getTextProbeResult(), { timeout: 10_000 }).toBe(
            "initial=0,0,0,255;255,255,255,255;255,255,255,255;0,0,0,255;255,255,255,255;255,255,255,255;0,0,0,255;0,0,0,255;255,255,255,255;0,0,0,255;255,255,255,255;0,0,0,255;255,255,255,255;255,255,255,255;0,0,0,255;0,0,0,255,mutated=0,0,0,255;255,255,255,255;255,255,255,255;0,0,0,255,multiline=255,255,255,255;255,255,255,255;255,255,255,255;0,0,0,255;255,255,255,255;255,255,255,255;0,0,0,255"
        );

        await h.beginDelegateDrawProbe();
        await expect.poll(() => h.getDelegateDrawProbeResult(), { timeout: 10_000 }).toBe(
            "initial=255,0,0,255;0,255,0,255,updated=0,0,255,255;255,255,0,255,normalVertical=255,0,0,255;255,255,255,255,flippedVertical=255,255,255,255;0,0,255,255,callbacks=true,display=true,retained=true,replaced=true,released=true,rejected=true,failures=1"
        );

        await h.beginEdgeAntialiasingProbe();
        await expect.poll(() => h.getEdgeAntialiasingProbeResult(), { timeout: 10_000 }).toBe(
            "initial=255,255,255,255;40,40,40,255;255,255,255,255;255,255,255,255;215,215,215,255;40,40,40,255,mutated=255,255,255,255;215,215,215,255,bottom=40,40,40,255;255,255,255,255,top=255,255,255,255;215,215,215,255"
        );

        await h.beginBooleanAnimationProbe();
        await expect.poll(() => h.getBooleanAnimationProbeResult(), { timeout: 10_000 }).toBe(
            "0,0,0,255;0,0,0,255;0,255,0,255;0,0,255,255;255,255,0,255,captures=1"
        );

        await h.beginContentsAnimationProbe();
        await expect.poll(() => h.getContentsAnimationProbeResult(), { timeout: 10_000 }).toBe(
            "255,0,0,255;0,0,255,255;0,255,0,255;0,255,0,255"
        );

        await h.beginRasterizationScaleProbe();
        await expect.poll(() => h.getRasterizationScaleProbeResult(), { timeout: 10_000 }).toBe(
            "pixel=255,0,0,255,scale=1.5,size=60x60,captures=1"
        );

        await h.beginShadowPathKeyframeProbe();
        await expect.poll(() => h.getShadowPathKeyframeProbeResult(), { timeout: 10_000 }).toBe(
            "255,0,0,255;0,0,0,255;0,0,0,255;255,0,0,255;0,0,0,255"
        );

        await h.beginTransformDepthProbe();
        await expect.poll(() => h.getTransformDepthProbeResult(), { timeout: 10_000 }).toBe(
            "crossing=true,transparent=true,isolated=true,flattened=true,nested=true,captures=10,composites=10,groupOpacity=true,filter=true,mask=true,directMask=true,rasterMask=true,maskUpdated=true,nestedFilter=true,shadow=true,shadowPath=true,compositionDepth=true,nestedComposition=true,overflow=true,updated=true,reused=true"
        );

        await h.beginCompositionProbe();
        await expect.poll(() => h.getCompositionProbeResult(), { timeout: 10_000 }).toBe(
            "ordered=true,unbounded=true,maskBackdrop=true,pixels=0,0,0,255;255,255,0,255;127,0,0,255;0,0,255,255;0,255,255,255;255,0,0,255;7,135,10,192;127,0,0,255;255,255,0,255;255,0,0,255;255,0,0,255;255,128,0,255;127,0,0,255;128,128,0,255;255,0,255,255;255,255,0,255;255,0,0,255;255,0,0,255;255,0,0,255;0,255,255,255;255,0,0,255;255,0,0,255;255,128,0,255;255,0,0,255;0,255,255,255;255,0,0,255;0,255,255,255;0,255,255,255;10,10,78,160;54,11,16,149,failures=0,after=0"
        );
        expect(await h.getCompositionFilterFailureCount()).toBe(0);
        expect(await h.getActiveCompositionResourceCount()).toBe(0);

        await h.beginShadowProbe();
        await expect.poll(() => h.getShadowProbeResult(), { timeout: 10_000 }).toBe(
            "255,0,0,255;13,140,19,255;13,13,146,255;26,26,38,255;255,0,0,255;255,0,0,255;emptyRegion=true;emptyLayer=true;imageEdges=true;imageCenter=true;maskTransition=true;animatedSilhouette=true"
        );
        expect(await h.getActiveShadowResourceCount()).toBe(0);
        expect(await h.getShadowRenderFailureCount()).toBe(0);

        await h.beginEmitterProbe();
        await expect.poll(() => h.getEmitterProbeResult(), { timeout: 10_000 }).toBe(
            "before=1,1,states=2,geometry=true,directions=true,failures=0;orders=true,additive=true,unknown=true;after=0,2,states=1;image=true,sampling=true,nil=true,rejected=true,child=true;blend=true,final=0"
        );

        await h.beginReplicatorProbe();
        await expect.poll(() => h.getReplicatorProbeResult(), { timeout: 10_000 }).toBe(
            "content=true,zero=true,delay=true,filter=true,shadow=true,raster=true"
        );

        await h.beginDisplayLinkProbe();
        await expect.poll(() => h.getDisplayLinkProbeResult(), { timeout: 10_000 }).toBe(
            "started=true,retained=true,stopped=true,duration=true"
        );

        await h.exerciseUnsupportedTransitionFilter();
        await expect.poll(() => h.getTransitionFilterFailureCount()).toBe(1);
        expect(await h.getActiveTransitionTextureCount()).toBe(2);

        await h.exerciseUnsupportedBuiltInTransition();
        await expect.poll(() => h.getTransitionRenderFailureCount()).toBe(1);
        expect(await h.getActiveTransitionTextureCount()).toBe(2);

        await h.exerciseUnsupportedTransitionSubtype();
        await expect.poll(() => h.getTransitionRenderFailureCount()).toBe(2);
        expect(await h.getActiveTransitionTextureCount()).toBe(2);

        await h.removeTransition();
        await expect.poll(() => h.getActiveTransitionTextureCount()).toBe(0);
        expect(await h.getFirstUncapturedGPUError()).toBe("none");
    });
});
