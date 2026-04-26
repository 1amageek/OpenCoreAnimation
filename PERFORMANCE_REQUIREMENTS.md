# OpenCoreAnimation Performance Requirements

This document lists every optimization mechanism that Apple's Core Animation
provides, the role each plays in keeping frame time bounded, and the gap to the
current OpenCoreAnimation implementation. It is the input to the optimization
work that closes the FPS gap observed in `megaman` and other E2E targets.

The goal is **architectural parity with Core Animation's render pipeline**, not
"add a few caches." Symptoms (`presentation()` thrashing, per-draw
`Float32Array` allocation, full-tree walks) are downstream of the missing
architecture below.

---

## 1. Reference architecture (Apple)

### 1.1 Three-tree model

Core Animation maintains three parallel trees per process:

| Tree | Owned by | Purpose | User-visible? |
|---|---|---|---|
| **Model tree** | Main thread | Target values written by app | Yes — every `layer.foo = x` |
| **Presentation tree** | Main thread | In-flight animated values | Read-only, via `layer.presentation()` |
| **Render tree** | Render server | What the GPU actually draws | Private |

Source: *Core Animation Programming Guide — Core Animation Basics, "Layer Trees
Reflect Different Aspects of the Animation State."*

### 1.2 Four-phase commit cycle

Every runloop iteration executes the following phases in order
(WWDC 2014 #419 *Advanced Graphics and Animations for iOS Apps*):

| Phase | Work | Cost class |
|---|---|---|
| **Layout** | `layoutSublayers`, view-tree creation, content population | CPU |
| **Display** | `drawRect:` / `display()` → render into backing store | CPU + memory |
| **Prepare** | Image decode, GPU-format conversion | CPU/GPU |
| **Commit** | Recursive walk of layer tree, serialize delta, ship to render server | CPU + IPC |

Two practical consequences:

1. The **render server** holds an independent frozen render tree and runs
   animation evaluation itself. The main thread is free between commits.
2. Static layers are **not re-evaluated** unless something marks them dirty.

### 1.3 Backing store

Per `setNeedsDisplay()` docs:
> Calling this method causes the layer to recache its content. […] The
> existing content in the layer's `contents` property is removed to make way
> for the new content.

The backing store is a CPU-side or GPU-side bitmap that survives across frames.
A layer with `contents` set and no `setNeedsDisplay()` call **never re-runs
`display()`** — its bitmap is reused indefinitely.

### 1.4 Atomic batching via `CATransaction`

Per `CATransaction` docs:
> CATransaction is the Core Animation mechanism for batching multiple
> layer-tree operations into atomic updates to the render tree. Every
> modification to a layer tree must be part of a transaction.

Implicit transactions auto-commit on the next runloop iteration. Explicit
transactions (`begin()` / `commit()`) bound the user's edits. The render
server only ever sees committed snapshots — never half-mutated trees.

---

## 2. Current state in OpenCoreAnimation

### 2.1 Present and working

| Mechanism | Where | Note |
|---|---|---|
| `CALayer.presentation()` | `CALayer.swift:136` | Exists but recomputed on every call |
| `CATransaction.begin/commit/flush` | `CATransaction.swift:102,128,182` | Implicit-animation context only |
| `setNeedsDisplay()` flag | `CALayer.swift:3127` | Bool flipped, never read by renderer |
| `setNeedsLayout()` flag | `CALayer.swift:3285` | Bool flipped, never read by renderer |
| `_subtreeShadowCount` / `_subtreeFilterCount` | `CALayer.swift` (Task #5) | Subtree counter for prerender skip |
| Texture LRU cache | `Rendering/WebGPU/Internal/TextureManager.swift` | CGImage → GPUTexture |
| Triple-buffered uniform/vertex pool | `Rendering/WebGPU/Internal/BufferPool.swift` | `advanceFrame()` based |
| Geometry tessellation cache | `Rendering/WebGPU/Internal/GeometryCache.swift` | path → vertex |

### 2.2 Absent or stub

| Mechanism | Symptom |
|---|---|
| **Render-tree / model-tree separation** | `presentation()` synthesized on the fly from model every frame |
| **Per-layer dirty propagation** | All static layers re-walked & re-evaluated every frame |
| **Backing store** | No bitmap retention; `display()` paths re-run unconditionally |
| **`shouldRasterize` honored** | Property exists but `CAWebGPURenderer` never reads it (grep: 0 hits) |
| **`isOpaque` honored** | Property exists but renderer always assumes alpha-blended |
| **`drawsAsynchronously`** | Not implemented |
| **`shadowPath` fast path** | Path property exists but renderer recomputes shadow geometry every frame |
| **`CATransaction.commit()` driving render** | Renderer runs from `CADisplayLink` independently of commits |
| **`presentation()` per-frame cache** | Recomputed N×log N times per frame inside `sortedByZPosition` |
| **`sortedByZPosition` cache** | Re-sorted from scratch every parent every frame |
| **JS `Float32Array` pool** | New TypedArray per drawcall × ~200 drawcalls = ~400 JS allocations/frame |
| **`needsDisplay(forKey:)` invalidation** | Returns false unconditionally; no implicit-redraw on KVO-style keys |

---

## 3. Requirements

Grouped into four phases. **Phase 1 and Phase 2 are the prerequisites** —
later phases are only effective once dirty propagation exists.

### Phase 1 — Dirty propagation infrastructure (foundation)

| ID | Requirement | Apple reference |
|---|---|---|
| **R1.1** | `CALayer` has a `_dirtyMask: CALayerDirtyFlags` (OptionSet). Setters for animatable properties (`bounds`, `position`, `transform`, `opacity`, `contents`, `backgroundColor`, `cornerRadius`, `shadowOpacity`, `shadowRadius`, `shadowOffset`, `shadowPath`, …) set the corresponding bit. | Implicit in render-tree commit model. |
| **R1.2** | Adding/removing/reordering a sublayer marks the parent's `.sublayerHierarchy` bit. Changing a sublayer's `zPosition` marks parent's `.sublayerOrdering` bit. | Required for R3.2. |
| **R1.3** | Dirty bits propagate to the root via a `_subtreeDirty` counter (analogous to existing `_subtreeShadowCount`). The renderer can early-return on a clean subtree. | Mirror of Task #5 design. |
| **R1.4** | `setNeedsDisplay()` sets `.contentsRedraw`. The renderer consults this flag to decide whether to re-run `display()` for delegate-backed contents. | `CALayer.setNeedsDisplay()` Discussion. |
| **R1.5** | `needsDisplay(forKey:)` class method is consulted on property change for custom `CALayer` subclasses. Return value triggers `setNeedsDisplay()`. | `class func needsDisplay(forKey:)`. |
| **R1.6** | All dirty bits cleared at the end of a successful `render()` pass, *after* the GPU command buffer has been submitted. | Standard double-buffered commit. |

### Phase 2 — Presentation cache & sublayer ordering cache

| ID | Requirement | Apple reference |
|---|---|---|
| **R2.1** | `CALayer._presentationFrameToken: UInt64` — when the renderer's frame counter matches the cached token, `presentation()` returns the existing `_presentationLayer` without recomputation. | Optimization, not API surface. |
| **R2.2** | When `_animations.isEmpty` AND no Phase-1 dirty bit on the model layer, `presentation()` returns `self` (no presentation layer allocation, no property copy). Acceptable per Apple: *"the presentation layer's values match the model layer's values."* | Programming Guide §"Layer-Based Animations". |
| **R2.3** | `sublayers` ordering cache: `CALayer._sortedSublayers: [CALayer]?` invalidated by R1.2's `.sublayerHierarchy` and `.sublayerOrdering` bits. `CAWebGPURenderer.sortedByZPosition` returns the cached array on hit. | Optimization. |
| **R2.4** | The presentation cache is per-renderer-frame, not global. Two concurrent renderers (rare on WASM but valid) must not stomp on each other. Use a `CARenderContext` token to scope. | Defensive. |

### Phase 3 — Backing store & rasterization

| ID | Requirement | Apple reference |
|---|---|---|
| **R3.1** | A layer with `contents != nil` AND `.contentsRedraw` clean must reuse its existing GPU texture. No re-upload per frame. | `setNeedsDisplay()` Discussion: backing store is the contract. |
| **R3.2** | `CALayer.shouldRasterize == true` → renderer captures the layer **and its entire subtree** into a single offscreen GPUTexture sized by `bounds × rasterizationScale`. Subsequent frames composite the cached texture as a single textured quad until any descendant goes dirty. | `shouldRasterize` Discussion. |
| **R3.3** | The rasterized cache includes filters and shadows but **excludes the layer's own opacity** (opacity is applied at composite time). | `shouldRasterize` Discussion: *"the current opacity of the layer is not rasterized."* |
| **R3.4** | Cache eviction: rasterized textures unused for ≥100 ms (≈6 frames @60 Hz) are released. Cache memory bound is configurable, default ≤ 2.5× viewport area to mirror Apple's documented guidance. | WWDC 2014 #419: *"cache size … 2.5 × screen", "evicted after 100 ms unused."* |
| **R3.5** | `isOpaque == true` → renderer's `display()` allocates a backing store without alpha channel; compositor skips per-pixel blend for the layer's quad. | `isOpaque` Discussion. |
| **R3.6** | `shadowPath != nil` → use the path directly to generate the blurred-mask geometry; skip the silhouette extraction from `contents`. | Performance Guide §"Specify a Shadow Path". |
| **R3.7** | Filter prerender (Task #5) and shadow prerender skip when the source subtree is clean (no Phase-1 dirty bit). | Extends Task #5. |

### Phase 4 — Commit-driven rendering

| ID | Requirement | Apple reference |
|---|---|---|
| **R4.1** | The renderer's frame loop pulls a **render snapshot**, not the live model tree. The snapshot is produced inside `CATransaction.commit()` and is the only object the render path reads. | Programming Guide §"Layer Trees Reflect Different Aspects of the Animation State". |
| **R4.2** | `CATransaction.commit()` runs four ordered sub-phases mirroring Apple: **Layout → Display → Prepare → Commit**. | WWDC 2014 #419. |
| **R4.3** | `CADisplayLink`-driven `render()` is decoupled from commits: if no commit has happened since the last frame AND no live animation is in flight, the renderer reuses the previous command buffer's outputs (skip submit). Only when a live animation is in flight does the renderer re-evaluate `presentation()` for the affected subtree. | Implicit in the render-server design. |
| **R4.4** | Animation list is held by the render snapshot, not the model tree. Adding an animation marks the layer dirty so the next commit captures it. | `CATransaction` Overview. |
| **R4.5** | Implicit transaction completion blocks fire after the render snapshot's frame has been submitted, not at `commit()` return. | `setCompletionBlock(_:)` semantics. |

### Phase 5 — Drawcall-level optimizations

These are downstream of Phase 1–4 but cheap to land independently.

| ID | Requirement |
|---|---|
| **R5.1** | Pool persistent JS `Float32Array` objects in `CAWebGPURenderer`. Reuse via `.set(swiftArray, offset)`. Eliminates ~400 JS allocations per frame. |
| **R5.2** | `supportedFilterOperations` cached on `CALayer` as a `Bool` (`_hasSupportedFilters`), invalidated by `filters` setter. |
| **R5.3** | `calculateClipRect` skipped when `masksToBounds == false`. Currently the boolean check exists but the matrix work runs unconditionally if any ancestor has it. |
| **R5.4** | `CAWebGPURenderer.renderLayer` early-return on `presentationLayer.opacity == 0 || isHidden` consults the **cached** presentation, not a fresh one. (Trivially follows from R2.1.) |

---

## 4. Out of scope (intentionally)

These are documented for completeness but **not** included in the planned work:

- Multi-process render server (`CAContext` host-client). WASM is single-threaded;
  IPC parity provides no benefit.
- `CAMetalLayer` / `CAEAGLLayer` / `CAOpenGLLayer`. Excluded per project policy
  (see existing `CLAUDE.md` §"Excluded APIs").
- `drawsAsynchronously`. Background-thread Core Graphics drawing requires a
  thread pool not available on WASM single-threaded runtime. Track as future
  work for native testing only.
- HDR / `CAEDRMetadata`. Out of project scope.
- `CADisplayLink.preferredFrameRateRange` (variable refresh). Browser
  `requestAnimationFrame` doesn't expose target rates pre-paint.

---

## 5. Verification strategy

| Phase | Verification |
|---|---|
| Phase 1 | Unit tests in `CALayerBehaviorTests`: setting a property flips the bit; ancestor counter increments; setter idempotent for same value. |
| Phase 2 | Counter-based test: a renderer wrapper that counts `presentation()` deep evaluations; assert that for N renders of a static tree the count drops from O(N×N) to O(N) after R2.1, and to 0 after R2.2. |
| Phase 3 | Integration test: a layer tree with `shouldRasterize = true` on a deep subtree renders identical pixels to one without (golden image); CPU-side check that the rasterized texture is written once across N frames. |
| Phase 4 | Behavior test: mutating the model tree mid-frame does not affect the in-flight render; the next commit captures the change exactly once. |
| Phase 5 | Microbenchmark: `Float32Array` allocation count per frame measured via JS heap snapshot. |
| End-to-end | `megaman` Playwright spec measures frame-time histogram via `__megaman_test` harness; pre-/post comparison reported per phase. |

---

## 6. Sequencing

```
Phase 1 (dirty bits)
   │
   ├──▶ Phase 2 (presentation + sublayer cache)
   │       │
   │       └──▶ Phase 3 (backing store + shouldRasterize)
   │
   ├──▶ Phase 4 (commit-driven snapshot)        ← largest refactor, lands last
   │
   └──▶ Phase 5 (drawcall pool, supportedFilters cache, clipRect skip)
            (independent — can land any time)
```

Phase 5 is a quick early win and can land in parallel with Phase 1.
Phase 4 is intentionally last because it touches the public `render()` /
`CATransaction` contract and depends on Phase 1's dirty-bit foundation.

---

## 7. References

- *Core Animation Programming Guide — Core Animation Basics*  
  https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CoreAnimation_guide/CoreAnimationBasics/CoreAnimationBasics.html
- *Core Animation Programming Guide — Improving Animation Performance*  
  https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CoreAnimation_guide/ImprovingAnimationPerformance/ImprovingAnimationPerformance.html
- `CATransaction` — https://developer.apple.com/documentation/quartzcore/catransaction
- `CALayer.shouldRasterize` — https://developer.apple.com/documentation/quartzcore/calayer/shouldrasterize
- `CALayer.isOpaque` — https://developer.apple.com/documentation/quartzcore/calayer/isopaque
- `CALayer.drawsAsynchronously` — https://developer.apple.com/documentation/quartzcore/calayer/drawsasynchronously
- `CALayer.setNeedsDisplay()` — https://developer.apple.com/documentation/quartzcore/calayer/setneedsdisplay()
- WWDC 2014 #419 — *Advanced Graphics and Animations for iOS Apps*
- WWDC 2010 #424 — *Core Animation in Practice, Part 1*
- WWDC 2021 — *Demystify and eliminate hitches in the render phase*  
  https://developer.apple.com/videos/play/tech-talks/10857/
