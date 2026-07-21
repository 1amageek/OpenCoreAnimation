# OpenCoreAnimation Performance — Detailed Design (v2)

This document is the implementation-level design that pairs with
`PERFORMANCE_REQUIREMENTS.md`. It is the artifact that gets verified before any
TDD test is written: it must close every requirement, identify every edge
case, and sequence every test cycle.

The reader is expected to have `PERFORMANCE_REQUIREMENTS.md` open. Each section
references the requirement IDs (R1.1 … R5.4) from there.

**v2 changelog (post multi-perspective review).** Ten blockers (B1–B10) found
in the 5-perspective review are folded in. Cross-references appear inline
where the resolution lives:

| ID | Blocker | Resolution location |
|---|---|---|
| B1 | Public `presentation()` must remain Apple-contract-compliant | §4.1 / §4.2 — R2.2 moved to internal `_renderTimePresentation()` |
| B2 | `static var _currentFrameToken` violates Swift 6 strict concurrency | §4.1 — `nonisolated(unsafe)` + WASM-only gate |
| B3 | `CARenderSnapshot: Sendable` cannot hold a `CALayer` reference | §6.1 — `Node` carries `ObjectIdentifier`, not `CALayer` |
| B4 | Re-parent counter delta not enumerated per mutator | §3.4 — table of 7 mutators with explicit ± delta lines |
| B5 | R2.2 disabled by lingering `isRemovedOnCompletion = false` animations | §3.5 / §4.1 / §6.2 — finished-animation condition + Phase A sweep |
| B6 | Completion-block / dirty-clear ordering race | §6.5 — `submit → clear → completionBlocks` |
| B7 | `_dirtyMask = .all` includes `.contentsRedraw` and triggers spurious `display()` | §3.1 / §3.7 — initial mask is `.all.subtracting(.contentsRedraw)`; `_needsDisplay` is an orthogonal axis |
| B8 | Pixel test 3.4 untestable through a mock | §5.7 — split into 3.4a (pipeline state) and 3.4b (composite-time alpha multiply) |
| B9 | Performance suites mutate global state and must be serialized | §10 — every `Performance/*` suite is `@Suite(.serialized)` with `init` reset |
| B10 | Float32Array pool benefit ambiguous | §7.1 — `JSTypedArray<Float32>` bulk copy; R5.1 acceptance is JS↔WASM boundary count |

---

## 0. TDD methodology

| Cycle | Action |
|---|---|
| **Red** | Write one focused `@Test` (swift-testing) that asserts the target behavior. The test must fail for the documented reason — not for an unrelated build error. |
| **Green** | Write the smallest production change that makes the test pass. No incidental refactors. |
| **Refactor** | Tighten naming, dedupe, doc-comment. No behavior change. |

**Rules specific to this work:**

1. **One requirement → at least one test.** A requirement without a test is
   not implemented.
2. **One test asserts one observable behavior.** "All bits clear after render"
   is one test; "bit X clears after render" is not — the granularity is on the
   observable (renderer skip vs. dirty re-evaluation), not on each property.
3. **Test the contract, not the implementation.** Tests reference the public
   `presentation()` / `render()` surface plus a new `internal` introspection
   surface (frame counter, dirty mask). They must not reach into `private`
   storage.
4. **Native + WASM parity.** Every Phase 1 / 2 / 5 test runs natively
   (`swift test`) because the dirty-bit and cache logic is platform-agnostic.
   Phase 3 / 4 tests that touch the WebGPU renderer have a native-only stub
   `MockCARenderer` so they run under `swift test` too.
5. **No brittle frame-time assertions.** Performance gain is measured E2E in
   `megaman` Playwright; correctness tests assert call counts, cache hits, and
   bitfield state — not wall-clock.

---

## 1. Self-review checklist (gap audit before coding)

Before any code is written, the design must answer "yes" to every item below.
If any is "no", the design is updated, not the code.

| # | Question | Answer |
|---|---|---|
| 1 | Does every R1–R5 requirement have a corresponding section here? | Yes — sections 3–7. |
| 2 | Is dirty-bit propagation safe under re-parenting? | Yes — section 3.6. |
| 3 | Does `presentation()` cache invalidate when an animation completes mid-frame? | Yes — section 4.4. |
| 4 | Can a presentation layer mutate the model's dirty mask via setters? | No — section 3.5 forbids it. |
| 5 | Does `init(layer:)` cleanly seed `_dirtyMask` and `_subtreeDirtyCount`? | Yes — section 3.7. |
| 6 | Do all sublayer-mutating APIs propagate the new `.sublayerHierarchy` bit? | Yes — section 3.4 enumerates them. |
| 7 | Does the rasterization cache evict before unbounded growth? | Yes — section 5.3. |
| 8 | Does `shouldRasterize` correctly composite at the model layer's opacity (not include it)? | Yes — section 5.4. |
| 9 | Does a mid-frame model mutation never leak into the in-flight render? | Yes — section 6.2 (snapshot). |
| 10 | Does `CADisplayLink` continue to drive frames when no commit happens (live animations)? | Yes — section 6.3. |
| 11 | Does the JS Float32Array pool survive `invalidate()` correctly? | Yes — section 7.1. |
| 12 | Are subtree counters (`_subtreeDirtyCount`) consistent with the existing `_subtreeShadowCount` pattern, so we don't introduce two divergent traversal models? | Yes — section 3.3 mirrors it. |
| 13 | Does the design allow Phase 5 to land in parallel with Phase 1 without merge conflicts? | Yes — section 8 sequencing. |

Each item is re-asserted in the relevant section below.

---

## 2. Vocabulary

- **Model layer** — the `CALayer` instance the user holds and mutates.
- **Presentation layer** — the per-frame copy with animations evaluated. Owned
  by the model layer; returned by `presentation()`.
- **Render snapshot** — a per-commit immutable structure introduced in Phase 4
  that the renderer reads from. Holds presentation values, animation refs, and
  the resolved sublayer order.
- **Frame token** — a `UInt64` bumped by the renderer at the start of each
  `render(layer:)` call. Used as a cache key to invalidate per-frame caches
  without explicit invalidation calls.
- **Subtree counter** — an `Int` on each layer that mirrors the count of
  descendants (incl. self) carrying a particular flag. Updated by propagating
  a delta to the root every time the contributing condition changes.

---

## 3. Phase 1 — Dirty propagation infrastructure

### 3.1 Public/internal surface

Nothing public is added. All new state is `internal` (or `internal package`)
to allow the renderer and tests to read but not the user.

```swift
// CALayer+Dirty.swift  (new file)
extension CALayer {
    /// Bit-flag set tracking which categories of state are stale on the model layer.
    @frozen
    internal struct DirtyFlags: OptionSet, Sendable {
        let rawValue: UInt32

        static let geometry          = DirtyFlags(rawValue: 1 << 0)  // bounds, position, anchorPoint, transform, sublayerTransform, zPosition, anchorPointZ
        static let appearance        = DirtyFlags(rawValue: 1 << 1)  // backgroundColor, borderColor, borderWidth, opacity, cornerRadius, isHidden, masksToBounds
        static let contents          = DirtyFlags(rawValue: 1 << 2)  // contents, contentsRect, contentsCenter, contentsGravity, contentsScale
        static let contentsRedraw    = DirtyFlags(rawValue: 1 << 3)  // setNeedsDisplay()
        static let shadow            = DirtyFlags(rawValue: 1 << 4)  // shadowOpacity, shadowRadius, shadowOffset, shadowColor, shadowPath
        static let filters           = DirtyFlags(rawValue: 1 << 5)  // filters, compositingFilter, backgroundFilters
        static let mask              = DirtyFlags(rawValue: 1 << 6)  // mask
        static let sublayerHierarchy = DirtyFlags(rawValue: 1 << 7)  // add/remove/insert/replace
        static let sublayerOrdering  = DirtyFlags(rawValue: 1 << 8)  // any sublayer's zPosition changed
        static let rasterization     = DirtyFlags(rawValue: 1 << 9)  // shouldRasterize, rasterizationScale, isOpaque
        static let animations        = DirtyFlags(rawValue: 1 << 10) // add/remove animation
        static let timing            = DirtyFlags(rawValue: 1 << 11) // beginTime, duration, speed, repeatCount, …

        static let all: DirtyFlags = [.geometry, .appearance, .contents, .contentsRedraw,
                                       .shadow, .filters, .mask, .sublayerHierarchy,
                                       .sublayerOrdering, .rasterization, .animations, .timing]
    }

    /// Categories dirty on this specific model layer (not its descendants).
    internal var _dirtyMask: DirtyFlags { get set }

    /// Number of descendants (incl. self) with `_dirtyMask != []`.
    /// Mirrors `_subtreeShadowCount`; renderer can early-return on 0.
    internal var _subtreeDirtyCount: Int { get }
}
```

The storage is added to `CALayer` proper as two stored properties:

```swift
// .contentsRedraw is intentionally NOT included in the initial mask.
// It is the explicit `setNeedsDisplay()` axis — see B7 / §3.7.
internal static let _initialDirtyMask: DirtyFlags = DirtyFlags.all.subtracting(.contentsRedraw)
internal var _dirtyMask: DirtyFlags = CALayer._initialDirtyMask
internal fileprivate(set) var _subtreeDirtyCount: Int = 1  // self contributes 1 because _dirtyMask != []
```

Initial state: a brand-new `CALayer` is dirty for every render-affecting
category **except `.contentsRedraw`**, with subtree count 1. This matches
today's behavior (every property is computed on first render) and prevents
skipping the very first frame, while keeping `.contentsRedraw` purely
opt-in via `setNeedsDisplay()`.

**Why `.contentsRedraw` is excluded (B7).** The existing `_needsDisplay`
boolean (CALayer.swift:3124) and `displayIfNeeded()` (line 3140) form an
orthogonal axis from `_dirtyMask`. If `.contentsRedraw` were initially set on
every fresh layer, `displayIfNeededRecursive` (Phase 4 §6.2) would invoke
`display()` once per layer per first frame — exactly the regression Phase 1
must avoid. The two axes are kept independent:

| Axis | Trigger | Consumer |
|---|---|---|
| `_needsDisplay` (existing Bool) | `setNeedsDisplay()`, `setNeedsDisplay(_:)` | `displayIfNeeded()` → `display()` |
| `_dirtyMask.contentsRedraw` (new bit) | the same setter mirrors into the mask for renderer-side cache invalidation (R3.1) | renderer's "skip texture re-upload?" check |

The clear at end of frame (§3.8) zeroes `_dirtyMask` but **leaves
`_needsDisplay` untouched** — the latter is cleared by `displayIfNeeded()`
itself, exactly as today.

### 3.2 Property → bit mapping (canonical)

Every animatable property becomes a setter that:

1. Compares old vs. new (early-return if equal — preserves idempotence).
2. Writes the new value to `_<name>`.
3. Calls `markDirty(.<bit>)` (defined below).
4. Where applicable, recomputes its subtree contribution and propagates a
   delta (e.g., `propagateShadowDelta` continues to run for the existing
   shadow counter).

| Property | Bit | Notes |
|---|---|---|
| `bounds`, `position`, `anchorPoint`, `transform`, `sublayerTransform`, `zPosition`, `anchorPointZ` | `.geometry` | `zPosition` *also* dirties **parent**.`.sublayerOrdering` (separate call). |
| `opacity`, `isHidden`, `backgroundColor`, `borderColor`, `borderWidth`, `cornerRadius`, `masksToBounds`, `maskedCorners`, `cornerCurve` | `.appearance` | |
| `contents`, `contentsRect`, `contentsCenter`, `contentsGravity`, `contentsScale`, `contentsFormat` | `.contents` | |
| `setNeedsDisplay()` | `.contentsRedraw` | Flag separate from `.contents` so a property change without explicit redraw does not trigger `display()`. |
| `shadowOpacity`, `shadowRadius`, `shadowOffset`, `shadowColor`, `shadowPath` | `.shadow` | Continues to propagate the existing `_subtreeShadowCount`. |
| `filters`, `compositingFilter`, `backgroundFilters` | `.filters` | Continues to propagate the existing `_subtreeFilterCount`. |
| `mask` | `.mask` | Setter must dirty mask *and* propagate the mask layer's full subtree dirty count up through the parent (mask is rendered into a stencil pass). |
| `add/insert/replace/removeSublayer*`, `sublayers=` | `.sublayerHierarchy` | All hierarchy mutators get the bit; the parent's bit, not the moving child's. |
| Any sublayer's `zPosition` change | `.sublayerOrdering` | Set on *parent*, not on the changed child. Implemented by `zPosition`'s setter calling `_superlayer?.markDirty(.sublayerOrdering)`. |
| `shouldRasterize`, `rasterizationScale`, `isOpaque` | `.rasterization` | |
| `add/remove animation` | `.animations` | `add(_:forKey:)` always dirties the bit. `removeAnimation`/`removeAllAnimations` dirty when the set was non-empty. |
| `beginTime`, `duration`, `speed`, `repeatCount`, `repeatDuration`, `autoreverses`, `fillMode`, `timeOffset` | `.timing` | |

**Properties that intentionally do NOT mark dirty:**

- `delegate` (weak ref, no rendering effect by itself; if delegate-driven
  display is needed the user calls `setNeedsDisplay()`).
- `name`, `style` (identification only).
- `actions` (consulted lazily during animation triggers).
- `layoutManager`, `constraints` (Phase 4 layout pass; not in scope yet).

### 3.3 The propagation primitive

Mirror the existing `propagateShadowDelta` pattern verbatim:

```swift
fileprivate static func propagateDirtyDelta(_ delta: Int, startingAt layer: CALayer?) {
    guard delta != 0 else { return }
    var node = layer
    while let n = node {
        n._subtreeDirtyCount += delta
        node = n._superlayer
    }
}
```

Centralized helper:

```swift
internal func markDirty(_ flags: DirtyFlags) {
    if _isPresentation { return }                           // R3.5 (section 3.5)
    let wasClean = _dirtyMask.isEmpty
    _dirtyMask.formUnion(flags)
    if wasClean { Self.propagateDirtyDelta(+1, startingAt: self) }
    invalidatePresentationCache()                            // section 4.1
}
```

`wasClean → +1` ensures the subtree counter increments only on the
clean→dirty transition. Idempotent: marking the same layer dirty twice in a
row contributes only one to the counter.

### 3.4 Sublayer mutating APIs (R1.2 enumeration)

These methods (already present, see `CALayer.swift:2963–3120`) must perform
**three** updates per mutation, mirroring the existing
`_subtreeShadowCount` pattern exactly (`propagateShadowDelta` at
CALayer.swift:3025) so we don't introduce a divergent traversal model:

1. **Subtract the moving child's `_subtreeDirtyCount`** from the *old*
   ancestor chain (if any) — done **before** mutating `_superlayer`.
2. **Set the parent's `.sublayerHierarchy`** dirty bit.
3. **Add the moving child's `_subtreeDirtyCount`** to the *new* ancestor
   chain — done **after** assigning `_superlayer = self`.

Per-mutator delta plan (matches the existing shadow-counter handling in
the same methods, lines 3043–3120):

| Mutator | Sub-step ordering |
|---|---|
| `set sublayers` (replace whole array) | For each removed child: `propagateDirtyDelta(-child._subtreeDirtyCount, startingAt: self)` then clear `_superlayer`. For each added child: detach from its old parent (delta -=), set `_superlayer = self`, `propagateDirtyDelta(+child._subtreeDirtyCount, startingAt: self)`. Finally `markDirty(.sublayerHierarchy)`. |
| `addSublayer(_:)` | If child has an old superlayer, detach (delta -=). Set `_superlayer = self`. `propagateDirtyDelta(+child._subtreeDirtyCount, startingAt: self)`. `markDirty(.sublayerHierarchy)`. |
| `insertSublayer(_:at:)` | Same as `addSublayer`. The index doesn't change the count math. |
| `insertSublayer(_:below:)` | Same as `addSublayer`. |
| `insertSublayer(_:above:)` | Same as `addSublayer`. |
| `replaceSublayer(_:with:)` | `propagateDirtyDelta(-old._subtreeDirtyCount, startingAt: self)`; clear `old._superlayer`; if new has an old parent, detach it (delta -=); set `new._superlayer = self`; `propagateDirtyDelta(+new._subtreeDirtyCount, startingAt: self)`; `markDirty(.sublayerHierarchy)`. |
| `removeFromSuperlayer()` | `propagateDirtyDelta(-self._subtreeDirtyCount, startingAt: oldSuper)`; `oldSuper.markDirty(.sublayerHierarchy)`; clear `_superlayer`. |

**`zPosition` setter** additionally calls
`_superlayer?.markDirty(.sublayerOrdering)`. The child marks its own
`.geometry` bit (because painter's-algorithm position changes affect the
child's transform too). No subtree-count mutation needed for `zPosition` —
the count tracks dirtiness, not order.

**Why subtract on detach is a delta, not a clear-and-rebuild.** A moving
subtree's internal dirty bits stay valid — they describe the subtree's own
state, not its relationship to the parent. Reattaching restores the
contribution to the new ancestors. This matches `_subtreeShadowCount` /
`_subtreeFilterCount` semantics 1:1.

### 3.5 Presentation layer is a *consumer*, never a *producer*

`init(layer:)` runs on a fresh presentation layer to seed it; it must NOT
propagate dirty deltas. Two safeguards:

1. The `_isPresentation` flag is set **before** any `markDirty` could fire
   (see section 3.7 for init order).
2. `markDirty` early-returns if `_isPresentation == true`.

Same applies to `updatePresentationLayer()` — it writes to the presentation's
backing-store storage `_<field>` directly, never through the public setter.
This is already the pattern today (see `CALayer.swift:196–215`); the design
preserves it.

**Animation completion produces dirtiness, but the presentation layer doesn't
mark it.** When an animation reaches its end during
`updatePresentationLayer()`, the *model* layer's `_animations` dictionary
must be drained on the next commit (Phase A in §6.2). Until that happens, a
finished animation with `isRemovedOnCompletion = false` would otherwise
disable the R2.2 self-return path forever (B5). The fix is to gate the
fast-path on `_animations.allSatisfy { $0.value.isFinished }`, not on
`_animations.isEmpty` — see §4.1.

### 3.6 Re-parenting safety

When a dirty subtree is moved from parent A to parent B:

- `removeFromSuperlayer()` walks up from A subtracting
  `self._subtreeDirtyCount`.
- The new attach (`addSublayer` / `insert*`) walks up from B adding
  `self._subtreeDirtyCount`.
- B is also marked `.sublayerHierarchy` dirty, which in turn bumps B's own
  contribution if it was clean.

The order matters: subtract on detach **before** clearing `_superlayer`, add
on attach **after** setting `_superlayer = self`. This is exactly how the
existing shadow/filter counters work — duplicate the pattern, do not invent a
new one.

### 3.7 `init` and `init(layer:)` order

Initial state per section 3.1 is `_initialDirtyMask` dirty (everything except
`.contentsRedraw`) with `_subtreeDirtyCount = 1`. Three consequences:

- `init()` does not need to do anything beyond the default initializers; the
  defaults seed the right state.
- `_needsDisplay` (existing Bool, CALayer.swift:3124) starts at `false` and
  is **not** touched by Phase 1. The `.contentsRedraw` bit and
  `_needsDisplay` are on/off in lock-step only because `setNeedsDisplay()`
  sets both — see §3.2. This separation is what prevents B7's
  spurious-`display()` regression.
- `init(layer:)` (presentation layer) must reset:
  ```swift
  self._isPresentation = true              // set first
  self._dirtyMask = []                     // presentation layer is read-only
  self._subtreeDirtyCount = 0              // does not contribute
  self._presentationCacheToken = 0         // §4.1 token storage
  ```
  These four lines go at the top of the `init(layer:)` body — after the
  `Hashable` storage but before any property assignment. Property assignments
  in `init(layer:)` already use `_<name>` (direct backing store), so they
  bypass `markDirty` even without the early-return guard. The early-return
  guard is the belt to that suspenders.

### 3.8 Bit clearing at end of frame (R1.6)

`CAWebGPURenderer.render(layer:)` calls a new `internal func
recursivelyClearDirtyAfterCommit()` on the root **after**
`device.queue.submit([encoder.finish()])` and **before** any user-visible
side-effect (completion blocks, animation-finished delegate calls). The
ordering is enforced in §6.5:

```
device.queue.submit([encoder.finish()])
rootLayer.recursivelyClearDirtyAfterCommit()       // ← bits cleared here
snapshot.completionBlocks.forEach { $0() }         // user callbacks observe a clean tree
```

This prevents the B6 race: a completion block that mutates the layer
graph (e.g., `addSublayer` in a CAAnimation `animationDidStop` callback)
would otherwise see a tree with stale dirty bits, and the next frame
would either over-render or — worse — `propagateDirtyDelta` would
operate on a partially-cleared subtree counter.

`recursivelyClearDirtyAfterCommit()` is implemented as:

```swift
internal func recursivelyClearDirtyAfterCommit() {
    guard _subtreeDirtyCount > 0 else { return }
    if !_dirtyMask.isEmpty {
        _dirtyMask = []
        Self.propagateDirtyDelta(-1, startingAt: self)
    }
    // _needsDisplay is intentionally NOT cleared here — it has its own
    // lifecycle managed by displayIfNeeded(). See §3.7 (B7).
    _sublayers?.forEach { $0.recursivelyClearDirtyAfterCommit() }
}
```

Walks only dirty subtrees. Verified by test: after this call, the root's
`_subtreeDirtyCount == 0` AND `_needsDisplay` is unchanged from its
pre-call value.

### 3.9 Edge cases checklist

| Case | Handling |
|---|---|
| Setting a property to its current value | Setter compares first; no bit set, no propagation. |
| Hierarchy operation that is a no-op (e.g. inserting at the same index) | Detect equality (same parent + same index); skip propagation. *Defer to section 3.10 — currently treat as dirty since detection is cheap to skip.* |
| Mid-frame mutation between commit phases | Phase 1 alone does not solve this; Phase 4 introduces the snapshot. Until then, model mutations during render visibly leak — which is the status quo. |
| Animation with from-value matching current model | The `.animations` bit is set unconditionally on `add()`; over-marking is acceptable (it just causes one re-evaluation) and avoids equality comparison on the animation graph. |
| `removeAnimation` when key missing | No bit set (no animation removed). |
| Subclass overrides a setter | Subclass must call `super.<setter>` for the bit to fire. Documented, not enforced. |
| Bit mutated on a *presentation* layer (e.g. test code calls `add(_:forKey:)` on a presentation copy by mistake) | `markDirty` early-returns; subtree counter unaffected. |

### 3.10 TDD test sequence — Phase 1

Tests live in `Tests/OpenCoreAnimationTests/Performance/DirtyPropagationTests.swift`.

| # | Test name | What it asserts | Implementation step |
|---|---|---|---|
| 1.1 | `freshLayerIsAllDirtyExceptContentsRedraw` | `CALayer()._dirtyMask == DirtyFlags.all.subtracting(.contentsRedraw)`; `_subtreeDirtyCount == 1`; `_needsDisplay == false` (B7) | Add storage + defaults. |
| 1.2 | `geometrySetterMarksGeometryBit` | `layer.bounds = …` → `.geometry` ⊆ `_dirtyMask` | Wrap one setter; verify pattern. |
| 1.3 | `idempotentSetterDoesNotPropagate` | After clearing, setting the same value does not increment subtree counter | Equality guard. |
| 1.4 | `appearanceShadowFiltersBitsCorrect` | Each property in section 3.2 sets its bit (parametrized helper) | Wrap remaining setters. |
| 1.5 | `addSublayerMarksParentHierarchyBit` | Parent's `.sublayerHierarchy` set, child unaffected | Wrap hierarchy mutators. |
| 1.6 | `zPositionMarksParentOrderingBit` | Setting child's `zPosition` marks parent `.sublayerOrdering` | `zPosition` setter. |
| 1.7 | `subtreeCounterIncrementsOnDirty` | Dirtying a leaf bumps every ancestor's counter by 1 | Verify `propagateDirtyDelta`. |
| 1.8 | `subtreeCounterIdempotent` | Dirtying an already-dirty leaf does not double-count | `wasClean` check. |
| 1.9 | `reparentingPreservesCounter` | Moving a dirty subtree A→B: A's count drops, B's rises | Hierarchy mutators handle it. |
| 1.10 | `presentationLayerCannotMarkDirty` | `markDirty` no-op on presentation layer | `_isPresentation` early-return. |
| 1.11 | `clearDirtyAfterCommitZeroesCounter` | After `recursivelyClearDirtyAfterCommit`, all bits + counters are 0 | Add the recursive clear. |
| 1.12 | `setNeedsDisplaySetsBothAxes` | `setNeedsDisplay()` sets BOTH `.contentsRedraw` bit AND `_needsDisplay = true`; setting `bounds` does NOT touch either (B7 — verifies the two axes are independent). | Implementation. |
| 1.13 | `needsDisplayForKeyTriggersRedrawForOverride` | A subclass returning true from `needsDisplay(forKey:)` causes property change to mark `.contentsRedraw` | R1.5 wiring. |
| 1.14 | `clearDirtyAfterCommitLeavesNeedsDisplayUntouched` | After `recursivelyClearDirtyAfterCommit()`, `_needsDisplay` retains its value (only `displayIfNeeded()` clears it). (B7) | Verify clear scope. |

Each test is independent. `clearDirty` is exposed as `internal func` so tests
can drive a clean→dirty→clean cycle without invoking the renderer.

---

## 4. Phase 2 — Presentation cache & sublayer ordering cache

### 4.1 Frame token

Add to `CALayer`:

```swift
// CALayer+FrameToken.swift  (new file)
extension CALayer {
    /// Monotonic per-render-frame counter. Bumped by the renderer at the top of
    /// each render(layer:). Single-threaded by construction:
    ///   - WASM is single-threaded by virtue of the host.
    ///   - On native (test-only paths), we serialize Performance/* suites
    ///     via @Suite(.serialized) (see §10) and reset to 0 in suite init.
    /// `nonisolated(unsafe)` is the explicit acknowledgment that we own the
    /// invariant — Swift 6 strict concurrency demands the annotation. (B2)
    nonisolated(unsafe) internal static var _currentFrameToken: UInt64 = 0

    internal var _presentationCacheToken: UInt64 = 0
}
```

`CAWebGPURenderer.render(layer:)` increments the global at the top:

```swift
CALayer._currentFrameToken &+= 1
```

#### Public surface — `presentation()` stays Apple-contract-compliant (B1)

Apple's `presentation()` contract states that the returned object is a
distinct copy whose values reflect the in-flight animations. Returning
`self` from the *public* method violates that contract: existing user code
written against Apple's QuartzCore would observe identity equality between
the model and presentation, which is a behavioral difference. The public
method therefore retains the original semantics; only **internal** callers
inside `CAWebGPURenderer` opt into the self-return fast path.

```swift
// Public, Apple-compatible: always returns a presentation copy when called
// outside the render loop. Cached per frame token so repeated calls in the
// same frame don't re-evaluate animations (R2.1).
open func presentation() -> Self? {
    if _isPresentation { return self as? Self }

    // R2.1 cache hit: same frame, already computed.
    if _presentationCacheToken == Self._currentFrameToken,
       let cached = _presentationLayer {
        return cached as? Self
    }

    if _presentationLayer == nil {
        _presentationLayer = createPresentationLayer()
    }
    updatePresentationLayer()
    _presentationCacheToken = Self._currentFrameToken
    return _presentationLayer as? Self
}

// Internal, renderer-only: applies the R2.2 self-return fast path.
// CAWebGPURenderer reads through this. User code reads through public
// presentation() and observes Apple-compatible behavior.
internal func _renderTimePresentation() -> CALayer {
    if _isPresentation { return self }

    // R2.2: when there is no live animation AND no presentation-affecting
    // mutation since last commit, the model values *are* the presentation
    // values. Skip the per-frame allocation.
    let allAnimationsFinished = _animations.allSatisfy { $0.value.isFinished }
    if allAnimationsFinished
        && _dirtyMask.isDisjoint(with: .presentationAffecting) {
        return self
    }

    return presentation() ?? self
}
```

`.presentationAffecting` is a static union of every bit that affects rendered
output: `[.geometry, .appearance, .contents, .shadow, .filters, .mask,
.rasterization, .animations]`. It excludes `.sublayerHierarchy` and
`.sublayerOrdering` (those affect *children* selection, not *self*) and
`.timing` (already handled by `_animations` evaluation timing).

`invalidatePresentationCache()` (called from `markDirty`) sets
`_presentationCacheToken = 0` so the next `presentation()` recomputes.

#### Why finished-animation gating matters (B5)

`CAAnimation.isRemovedOnCompletion = false` is the canonical pattern for
holding a final value at the end of an animation (e.g. UIKit's `setHidden`
toggles via fade-out animation). The animation stays in `_animations` after
its end time. Naive `_animations.isEmpty` gating would make every layer
that has ever held such an animation fall off the R2.2 fast path forever.
The `isFinished`-based predicate fires when:

- `anim.beginTime + anim.duration <= currentMediaTime` (forward)
- ...with `repeatCount` / `repeatDuration` taken into account (handled
  inside `CAAnimation.isFinished` already).

`processAnimationCompletions()` in §6.2 Phase A is what eventually drains
the dictionary; until then, `isFinished` keeps R2.2 enabled.

### 4.2 R2.2 self-return correctness

The R2.2 fast path lives **only** in the internal
`_renderTimePresentation()` (§4.1). The public `presentation()` retains
Apple's documented behavior: it always returns a distinct presentation copy
when called outside a render frame.

Why it is safe inside the renderer:

- Per Apple's docs, when no animations are running, presentation values match
  model values byte-for-byte. The renderer reads geometry/appearance/etc.
  from this object and writes nothing back.
- The renderer is the only consumer of `_renderTimePresentation()`. User
  code never sees the result, so the `===` identity invariant Apple's docs
  imply is preserved at the public surface.
- If a future render path needs to **mutate** the result (e.g. inject
  computed values), it must call `presentation()` instead. The naming
  contract enforces this: `_renderTimePresentation` is read-only by name.

Renderer migration (CAWebGPURenderer.swift:1718):
```swift
// Before:
let rootPresentation = rootLayer.presentation() ?? rootLayer
// After:
let rootPresentation = rootLayer._renderTimePresentation()
```
Same observable behavior at the public surface, fewer allocations on the
hot path.

#### Audit: callers that legitimately need public `presentation()`

Phase 1/2 leaves the public method untouched. The only renderer-internal
read site that switches to `_renderTimePresentation()` is the geometry
read inside `renderLayer` and the comparator in `sortedSublayers()` (§4.3).
All test code, public API surface, and animation delegate code continues
to call `presentation()` and observes the Apple-compatible cached copy.

### 4.3 Sublayer ordering cache (R2.3)

```swift
extension CALayer {
    internal var _sortedSublayersToken: UInt64 = 0
    internal var _sortedSublayers: [CALayer] = []
}
```

`CAWebGPURenderer.sortedByZPosition` (currently at line 190) becomes a method
on `CALayer`:

```swift
internal func sortedSublayers() -> [CALayer] {
    guard let subs = _sublayers, !subs.isEmpty else { return [] }
    if _dirtyMask.isDisjoint(with: [.sublayerHierarchy, .sublayerOrdering]),
       _sortedSublayersToken == Self._currentFrameToken,
       _sortedSublayers.count == subs.count {
        return _sortedSublayers
    }
    let sorted = subs.enumerated().sorted { a, b in
        let zA = a.element._renderTimePresentation().zPosition
        let zB = b.element._renderTimePresentation().zPosition
        if zA != zB { return zA < zB }
        return a.offset < b.offset
    }.map(\.element)
    _sortedSublayers = sorted
    _sortedSublayersToken = Self._currentFrameToken
    return sorted
}
```

The fast path skips the O(N log N) sort *and* the O(N) `presentation()` calls
inside the comparator. Renderer change is a one-line swap.

Cache is invalidated by:
- `.sublayerHierarchy` bit (add/remove/insert/replace).
- `.sublayerOrdering` bit (any child's `zPosition` change).
- New frame token (because cache stores presentation-evaluated z, which can
  drift mid-animation even with bits clean — but this case is dominated by
  R2.2 returning model anyway).

### 4.4 Animation lifecycle and cache invalidation

When an animation fires `markFinished(completed:)` (e.g., animation reaches
its end time during `updatePresentationLayer`), the layer is implicitly
dirty for the next frame because the presentation value just changed. Two
mechanisms handle this:

1. **Mid-frame**: the cache token is set after `updatePresentationLayer`, so
   subsequent `presentation()` calls *in the same frame* return the same
   copy — even if the animation finished. Correct: a single frame should see
   one consistent presentation.
2. **Next frame**: the renderer increments `_currentFrameToken`, so the cache
   stale-detects automatically. No explicit `markDirty` needed for animation
   finish — the token bump is the invalidation.

For R2.2's "self return" path: when the last animation completes, the
`_animations` dict empties; the next `presentation()` enters the R2.2 path
and returns `self`. The dropped `_presentationLayer` is freed by ARC.

### 4.5 Multi-renderer safety (R2.4)

Two renderers running against the same tree (rare; pre-render texture passes
*do* re-enter `presentation()` within a single `render(layer:)` call) must
share the frame token within one frame. Solution: increment only at top of
`render`, not at top of pre-render passes — they all see the same token. This
already matches the design above.

If a future use case requires concurrent renderers (separate canvases), the
fix is to make `_currentFrameToken` a parameter of a `CARenderContext` and
have each renderer maintain its own token. Out of scope for this design but
documented to keep the door open.

### 4.6 Edge cases checklist

| Case | Handling |
|---|---|
| Renderer wraps frame token (`UInt64.max`) | Saturating wraparound `&+=` cycles cleanly. At 60 fps this happens after ~9.7×10¹⁰ years. Not a concern. |
| User calls `presentation()` outside a render frame | Returns the model fallback (R2.2 path) for clean trees, or builds a fresh presentation for animated ones. Same as today. |
| Layer never been rendered → `_presentationCacheToken == 0`, `_currentFrameToken == 0` | `_presentationLayer == nil` so the cache hit is skipped by the `cached` unwrap. Safe. After first frame `_currentFrameToken == 1` so the `0 == 0` race never fires. |
| Test increments token but does not call render | Test helper sets `_currentFrameToken = N` directly via `internal` setter for deterministic cache tests. |

### 4.7 TDD test sequence — Phase 2

`Tests/OpenCoreAnimationTests/Performance/PresentationCacheTests.swift`.

| # | Test name | Asserts | Step |
|---|---|---|---|
| 2.1 | `cleanLayerWithoutAnimationsReturnsSelf` | `presentation() === self` after a clean frame | R2.2. |
| 2.2 | `dirtyLayerBuildsPresentationCopy` | After `bounds` change, `presentation() !== self`, returns copy | Cache miss path. |
| 2.3 | `secondPresentationInSameFrameReturnsCachedCopy` | Two `presentation()` calls in the same token return the same instance | R2.1 token check. |
| 2.4 | `nextFrameInvalidatesCache` | Incrementing token + dirty mask returns a new copy | Token-based invalidation. |
| 2.5 | `sortedSublayersCachedAcrossFrames` | When parent has clean ordering bits + same token, returns same array | R2.3. |
| 2.6 | `addSublayerInvalidatesSortedCache` | After `addSublayer`, cache returns fresh sort | `.sublayerHierarchy` triggers. |
| 2.7 | `presentationCallCountInRender` | A counter-instrumented `MockCALayer` reports O(N) `presentation()` calls per frame instead of O(N log N) | Integration via mock. |

The MockCALayer subclass overrides `presentation()` to bump a counter then
calls `super`. Used to verify renderer integration without parsing render
output.

---

## 5. Phase 3 — Backing store, rasterization, isOpaque, shadowPath

### 5.1 Backing store (R3.1)

A layer with `contents != nil` and `.contentsRedraw` clean must reuse its
existing GPU texture. Today, `CAWebGPURenderer` already keys
`textureManager` by `ObjectIdentifier(image as AnyObject)`, so the texture is
de-duplicated *across layers* with the same image. The missing piece is
**not regenerating the upload command** when the same layer renders the same
image across frames.

Implementation:

- `GPUTextureManager.texture(for:)` already does the cache lookup.
- The renderer currently issues a new `device.queue.writeTexture` only on
  cache miss — verify by reading `texture(for:)` (`TextureManager.swift`).
  If today it re-uploads on hit, fix it.
- Add a guard in `renderTextureContent`:
  ```swift
  if !layer._dirtyMask.contains(.contentsRedraw),
     let cached = textureManager.texture(for: image) {
      // reuse, no upload
  }
  ```

R3.1 verification: a unit test wraps `MockTextureManager` and asserts that 2
frames of the same image cause 1 upload, not 2.

### 5.2 `shouldRasterize` cache (R3.2 / R3.3 / R3.4)

#### Data model

```swift
internal struct RasterizedLayer: Sendable {
    let texture: GPUTexture
    let pixelSize: CGSize          // bounds.size × rasterizationScale
    let contentBoundsHash: Int     // hash of bounds + transform (for fast invalidation)
    var lastUsedFrame: UInt64
}

extension CAWebGPURenderer {
    private var rasterizationCache: [ObjectIdentifier: RasterizedLayer] = [:]
    private var rasterizationCacheBytes: Int = 0
    private let rasterizationCacheMaxBytes: Int  // computed from viewport on init/resize
}
```

`rasterizationCacheMaxBytes` defaults to `Int(viewport.width * viewport.height
* 4 * 2.5)` per WWDC 2014 #419 guidance.

#### Capture algorithm

For a layer with `shouldRasterize == true`:

1. If cache hit AND no descendant of the layer is dirty
   (`layer._subtreeDirtyCount == 0`) AND
   `layer.contentBoundsHash == cached.contentBoundsHash` AND the captured
   subtree does not contain backdrop-dependent composition:
    - Composite the cached texture as a single textured quad at `layer`'s
      transform.
    - Apply `layer.opacity` at composite time (R3.3 — opacity excluded from
      the cache).
    - Update `lastUsedFrame = _currentFrameToken`.
2. Else:
    - Allocate offscreen `GPUTexture` of `pixelSize`.
    - Recursively render `layer` and its subtree into the texture (calling
      the existing `renderLayer` with a redirected render-pass target).
    - Replace cache entry.
    - Composite as in (1).

A captured subtree containing `compositingFilter` or `backgroundFilters`
depends on pixels outside that subtree. Its ancestor rasterization is therefore
deferred until backdrop composition completes and is recaptured every frame;
ordinary layer dirty state cannot prove that the external backdrop is unchanged.

#### Eviction (R3.4)

After `device.queue.submit`, walk `rasterizationCache`:

- Drop entries with `lastUsedFrame + 6 < _currentFrameToken` (≈100 ms @ 60 Hz).
- If `rasterizationCacheBytes > rasterizationCacheMaxBytes`, drop oldest by
  `lastUsedFrame` until under bound.

`invalidate()` (renderer teardown) drops the whole cache.

### 5.3 `isOpaque` honoring (R3.5)

`isOpaque` becomes a hint to `prerenderShadows` / `prerenderFilteredLayers` /
the rasterized capture path: the offscreen texture format chooses `bgra8unorm`
without alpha pre-multiplication when `isOpaque == true`. The composite path
also omits the alpha-blend pipeline state for the layer's quad.

Caveat: `isOpaque` does not affect the *user's* `display()` callback contract
(they may still draw with alpha) — it only affects the texture format and
blend mode. This is honest with Apple's docs, which describe `isOpaque` as a
*hint*.

### 5.4 `shadowPath` fast path (R3.6)

Today, shadow silhouette is extracted from `contents` via the
`shadowMaskPipeline`. When `shadowPath != nil`:

- Skip silhouette extraction.
- Tessellate the path through the existing `PathTessellator`.
- Render the tessellated geometry directly into `shadowMaskTexture`.

This is a localized change in `prerenderShadows` — no architecture impact.

### 5.5 Filter / shadow prerender skip (R3.7)

Existing Task #5 fast-path already uses `_subtreeShadowCount` /
`_subtreeFilterCount` to skip the recursive walk when no contributor exists.
R3.7 layers on top: even when a contributor exists, if neither the
contributing layer nor any ancestor is dirty (`_subtreeDirtyCount == 0` at the
contributor) AND the prerendered texture is cached, reuse the cache.

Storage: extend `blurredShadowCache` with a `lastUsedFrame` and check the
contributor's `_dirtyMask` before re-running the blur passes.

### 5.6 Edge cases checklist

| Case | Handling |
|---|---|
| `shouldRasterize` toggled mid-animation | `.rasterization` bit set; cache evicted on next render. |
| Rasterized layer's child mutates | `_subtreeDirtyCount > 0` at the rasterized root → recapture. |
| Rasterized subtree contains backdrop composition | Resolve the descendant composition first, then recapture every frame because external backdrop pixels are outside the subtree dirty contract. |
| Layer scrolls (transform-only change) | Transform is a uniform, so cached texture is reused; `contentBoundsHash` excludes the parent's transform — only the *self* transform that affects internal layout matters, which is captured in `bounds`. |
| `rasterizationScale` change | `.rasterization` bit; cache evicted. |
| Off-screen layer | Optimization opportunity (skip capture) but out of scope; cache still works. |
| Mask layer with `shouldRasterize` | Defer to mask first, then rasterize composed result. Documented in mask section. |

### 5.7 TDD test sequence — Phase 3

`Tests/OpenCoreAnimationTests/Performance/RasterizationTests.swift`.

| # | Test name | Asserts | Step |
|---|---|---|---|
| 3.1 | `contentsReuseAcrossFrames` | Same image, no `setNeedsDisplay`, 2 frames → 1 upload | R3.1. |
| 3.2 | `shouldRasterizeCachesSubtree` | Frame 1 captures, frame 2 reuses cached texture | R3.2. |
| 3.3 | `descendantDirtyEvictsRasterizationCache` | Mutating a child after frame 1 → frame 2 recaptures | R3.2 + dirty hook. |
| 3.4a | `rasterizationPipelineExcludesOpacityFromCapture` | Pipeline state used to render *into* the offscreen texture has `clearColor.a == 1.0` regardless of `layer.opacity`; opacity is not multiplied into the captured pixels (B8 — assertable through pipeline-descriptor introspection on `MockCARenderer`). | R3.3 capture path. |
| 3.4b | `compositeMultipliesLayerOpacityAtRenderTime` | When compositing the cached texture as a quad, the per-quad uniform `opacity` equals the *current* `layer.opacity`. Verified by reading the uniform buffer the mock recorded; opacity may differ across frames without re-capturing. | R3.3 composite path. |
| 3.5 | `cacheEvictsAfterIdleFrames` | After 7 frames of disuse, entry removed | R3.4. |
| 3.6 | `cacheRespectsByteBudget` | When `bytes > maxBytes`, oldest entry dropped | R3.4. |
| 3.7 | `isOpaqueOmitsAlphaBlend` | Pipeline descriptor for the quad has `blend: nil` | R3.5. |
| 3.8 | `shadowPathSkipsSilhouetteExtraction` | When `shadowPath != nil`, `shadowMaskPipeline` not invoked | R3.6. |
| 3.9 | `shadowPrerenderSkipsCleanSubtree` | After clean frame, second frame's `prerenderShadows` reuses cached blurred texture | R3.7. |

Tests 3.1, 3.5, 3.6, 3.9 use `MockCARenderer` (native test fallback) that
stubs out actual GPU calls and records calls to a counter.

---

## 6. Phase 4 — Commit-driven rendering

### 6.1 `CARenderSnapshot`

```swift
internal struct CARenderSnapshot: Sendable {
    struct Node: Sendable {
        /// Identity, NOT a strong reference. Used as the key into renderer
        /// caches (texture cache, rasterization cache). The snapshot must
        /// not retain CALayer because CALayer is a non-Sendable class. (B3)
        let identity: ObjectIdentifier

        let presentationValues: PresentationValues
        let sortedChildIndices: [Int]
        let activeAnimations: [SnapshotAnimationRef]   // also Sendable (see below)
        let backingStoreToken: BackingStoreToken?
    }

    /// A snapshot-time copy of just the fields the renderer reads from a
    /// CAAnimation (start time, evaluator function, key path). Holding a
    /// CAAnimation directly would re-introduce non-Sendable state (delegate
    /// closures, mutable timing offsets).
    struct SnapshotAnimationRef: Sendable {
        let key: String
        let beginTime: CFTimeInterval
        let duration: CFTimeInterval
        // ... other immutable scalars needed at evaluation time ...
    }

    let nodes: [Node]                       // depth-first index
    let frameToken: UInt64
    let rootIndex: Int
    let completionBlocks: [@Sendable () -> Void]   // queued from CATransaction
}
```

`PresentationValues` is a `struct` carrying every render-affecting field
(geometry, appearance, contents handle, etc.). Built once during `commit()`,
read by the renderer. **The model `CALayer` is never referenced from the
snapshot** (B3) — the renderer keeps a parallel `[ObjectIdentifier:
CALayer]` weak map for the rare cache lookup that needs the live object,
which is also why the rasterization cache (§5.2) keys by `ObjectIdentifier`.

#### Why this pattern (B3 detail)

`CARenderSnapshot: Sendable` is the contract that lets the snapshot cross
isolation boundaries (e.g. an off-main-actor encode pass on native
platforms; a Worker on WASM if we ever add OffscreenCanvas + workers).
`CALayer` is a `class` with non-`Sendable` state (delegate, mutable
animation dictionary, weak superlayer link). Holding it inside a
`Sendable` struct would require `@unchecked Sendable`, which the project
rules explicitly forbid. The `ObjectIdentifier` indirection costs nothing
(it's a `UInt`) and makes the Sendable claim honest.

### 6.2 Commit's four sub-phases (R4.2)

`CATransaction.commit()` becomes:

```swift
public class func commit() {
    guard !_transactionStack.isEmpty else { return }
    let state = _transactionStack.removeLast()

    // Phase A — Layout & animation reaping
    rootLayer.processAnimationCompletions()  // (B5) drain finished animations from _animations
                                             // before snapshot capture, so the snapshot
                                             // doesn't carry zombie animations into the next frame.
    rootLayer.layoutIfNeeded()               // walks .needsLayout subtrees

    // Phase B — Display
    rootLayer.displayIfNeededRecursive()     // .contentsRedraw

    // Phase C — Prepare
    let snapshot = CARenderSnapshot.capture(
        rootLayer,
        frameToken: _nextFrameToken,
        completionBlocks: state.completionBlocks
    )
    _nextFrameToken &+= 1

    // Phase D — Commit
    activeRenderer?.submit(snapshot)         // queues for next CADisplayLink fire

    // Note: completion blocks are NOT fired here. They fire in the renderer
    // after submit + dirty-clear, see §6.5 (B6).
}
```

`processAnimationCompletions()`:

```swift
internal func processAnimationCompletions() {
    let now = CACurrentMediaTime()
    for (key, anim) in _animations where anim.isFinished(at: now) {
        if anim.isRemovedOnCompletion {
            _animations.removeValue(forKey: key)
        } else {
            // Bake final value into model layer if delegate requests it,
            // then mark the animation as "settled" — it stays in the dict
            // but reports isFinished=true so R2.2 fast path can fire.
            anim.markSettled()
        }
        anim.delegate?.animationDidStop(anim, finished: true)
    }
    _sublayers?.forEach { $0.processAnimationCompletions() }
}
```

The `markSettled()` step is what makes B5 fully resolve: even with
`isRemovedOnCompletion = false`, the finished animation no longer blocks
R2.2 because `_renderTimePresentation()` checks `isFinished`, not
membership.

### 6.3 Commit-snapshot decoupling from `CADisplayLink` (R4.3)

The renderer holds the latest snapshot:

```swift
final class CAWebGPURenderer {
    private var pendingSnapshot: CARenderSnapshot?

    func submit(_ snapshot: CARenderSnapshot) {
        pendingSnapshot = snapshot
    }

    public func render(layer: CALayer) {
        // Use pendingSnapshot if available; fall back to live tree for
        // backwards compatibility during migration.
        let snapshot = pendingSnapshot ?? CARenderSnapshot.capture(layer, frameToken: ...)

        // No new commit + no live animation → render the previous snapshot's output
        // (rebind the same vertex/uniform buffers, no re-evaluation).
        let hasActiveAnimations = snapshot.nodes.contains { !$0.activeAnimations.isEmpty }
        if snapshot.frameToken == lastRenderedFrameToken && !hasActiveAnimations {
            return                                     // skip submit
        }

        renderSnapshot(snapshot)
        lastRenderedFrameToken = snapshot.frameToken
    }
}
```

`render(layer:)` keeps its current public signature for backwards
compatibility. Internally it calls `renderSnapshot(_:)` which is the new
snapshot-driven implementation.

### 6.4 Animation list ownership (R4.4)

Adding an animation:

```swift
open func add(_ anim: CAAnimation, forKey key: String?) {
    let copied = anim.copy()
    // ... existing setup ...
    _animations[animKey] = copied
    markDirty(.animations)            // ← new line
    copied.delegate?.animationDidStart(copied)
}
```

`CARenderSnapshot.capture` reads `_animations` per layer at capture time.
Mutations after capture do not affect the in-flight snapshot.

### 6.5 Completion blocks fire post-submit (R4.5)

`CATransaction.setCompletionBlock(_:)` queues the block in the snapshot. The
renderer fires it after `device.queue.submit` returns AND after dirty bits
have been cleared (B6):

```swift
device.queue.submit([encoder.finish()])
rootLayer.recursivelyClearDirtyAfterCommit()       // ① clear bits FIRST
snapshot.completionBlocks.forEach { $0() }         // ② then user callbacks
```

**Why this order (B6 detail).** A completion block can legally mutate the
layer graph — the canonical pattern is "fade-out animation finishes →
removeFromSuperlayer". If completion blocks fired before the clear, the
mutating block would set new dirty bits, then `recursivelyClearDirtyAfterCommit`
would zero them, and the *next* frame would skip the just-mutated subtree.
Reversing the order means the next frame's snapshot capture sees the
mutations correctly because:

1. Snapshot for frame N submitted.
2. Bits from frame N cleared (state is "clean").
3. Completion blocks run; their mutations set fresh bits via `markDirty`.
4. Next commit captures a snapshot that includes those mutations.

The tradeoff is that a misbehaving completion block that calls back into
`render()` synchronously would observe a "clean tree with pending
mutations". This is the same edge case Apple's QuartzCore exhibits and
matches its documented behavior.

### 6.6 Backwards-compatibility fallback

Until every test path has been migrated, `pendingSnapshot == nil` (no commit
happened) falls back to capturing live. This keeps existing callers
(`CADisplayLink.displayLinkDidFire` direct → `renderer.render(layer:)`)
green while the snapshot path is rolled out.

When `pendingSnapshot != nil`, the live tree is no longer read by the
renderer — proving snapshot fidelity is the Phase 4 acceptance test.

### 6.7 Edge cases checklist

| Case | Handling |
|---|---|
| User calls `commit()` from inside `setCompletionBlock` | Implicit transaction reused; nested commits drain in order. |
| Renderer is not yet attached when commit happens | `pendingSnapshot` accumulated; latest wins. |
| Live animation finishes between snapshots | `activeAnimations` empty in next snapshot → R4.3 skip kicks in. |
| Mutation during render | Mutates model layer; next commit picks up. In-flight snapshot unaffected (it holds copies). |

### 6.8 TDD test sequence — Phase 4

`Tests/OpenCoreAnimationTests/Performance/CommitDrivenRenderingTests.swift`.

| # | Test name | Asserts | Step |
|---|---|---|---|
| 4.1 | `commitProducesSnapshot` | After `CATransaction.commit()`, `pendingSnapshot != nil`, frame token incremented | Snapshot capture. |
| 4.2 | `snapshotIsImmutableAcrossModelMutation` | Mutate model after commit → snapshot.presentationValues unchanged | Defensive copy. |
| 4.3 | `cleanRenderWithoutCommitSkipsSubmit` | If no commit + no active animation, `MockRenderer.submitCount` does not increment frame-over-frame | R4.3. |
| 4.4 | `liveAnimationForcesEvaluation` | When animation active, render evaluates presentation each frame regardless of token equality | R4.3 escape hatch. |
| 4.5 | `completionBlockFiresAfterSubmit` | Block fires only after `submit` returns; ordering verified by counter | R4.5. |
| 4.6 | `addAnimationDirtiesAndIsCapturedNextCommit` | Adding animation mid-commit appears in *next* snapshot, not current | R4.4. |

---

## 7. Phase 5 — Drawcall-level optimizations

### 7.1 Float32Array pool (R5.1)

The current implementation (`CAWebGPURenderer.swift:1896–1916`,
`createFloat32Array`) marshals an `[Float]` to JS by **per-element**
property writes through `JSValue.number(...)`. Each write crosses the
JS↔WASM ABI boundary. For a 4 KB vertex buffer that's ~1000 boundary
crossings per allocation, multiplied by 40+ call sites per frame.

The fix is **two**, not one (B10):

1. **Bulk copy**: replace per-element marshaling with
   `JSTypedArray<Float32>` — JavaScriptKit's typed-array binding lets
   Swift hand a `UnsafeBufferPointer<Float>` to the JS side as a single
   memcpy through the linear-memory view.
2. **Pool reuse**: avoid allocating a fresh `Float32Array` per frame by
   recycling JS-side buffers across frames.

```swift
final class Float32ArrayPool {
    private struct Bucket { var freeIndex: Int = 0; var arrays: [JSObject] = [] }
    private var buckets: [Int: Bucket] = [:]   // key = log2(capacity)

    /// Returns a JSObject of capacity ≥ count, filled with `floats`.
    /// One JS↔WASM boundary crossing per call (the typed-array set()).
    func acquireFilled(_ floats: [Float]) -> JSObject {
        let pow = nextPowerOfTwo(floats.count)
        let key = bucketKey(pow)
        var bucket = buckets[key, default: Bucket()]
        defer { buckets[key] = bucket }

        let array: JSObject
        if bucket.freeIndex < bucket.arrays.count {
            array = bucket.arrays[bucket.freeIndex]
        } else {
            array = JSObject.global.Float32Array.function!.new(pow).object!
            bucket.arrays.append(array)
        }
        bucket.freeIndex += 1

        // ONE boundary crossing — bulk copy via typed-array slot view.
        floats.withUnsafeBufferPointer { buf in
            let typed = JSTypedArray<Float32>(unsafelyWrapping: array)
            typed.copyMemory(from: buf)            // memcpy through wasm linear memory
        }
        return array
    }

    func resetForNewFrame() {
        for key in buckets.keys { buckets[key]!.freeIndex = 0 }
    }

    func invalidate() { buckets.removeAll() }
}
```

Lifetime:
- Created in `CAWebGPURenderer.init`.
- `resetForNewFrame()` called at start of `render()`.
- `invalidate()` clears the pool (releases JS references).

Call sites (40+ in `CAWebGPURenderer`): every place that currently does
`JSObject.global.Float32Array.function!.new(jsArray)` becomes
`pool.acquireFilled(swiftFloats)`.

#### R5.1 acceptance metric

The benefit isn't "fewer allocations" (the GC handles those) — it's
**fewer JS↔WASM boundary crossings** per frame, which cost ~50–200 ns
each and dominate WASM render budgets at high vertex counts.

Acceptance test: instrument `JSObject.global.Float32Array.function!.new`
and `JSValue.number(...)` to bump a counter. After R5.1 lands:

- Boundary count per frame must be **constant** with respect to
  per-vertex iteration (one crossing per pool acquire instead of one per
  vertex).
- Per-frame total scales O(call sites), not O(call sites × vertices).

Numerical target: for the megaman idle scene (~2 K vertices, 40 buffer
allocations per frame), boundary count drops from ~80 K to ~40. Reported
in PR description; see test 5.1.

### 7.2 `_hasSupportedFilters` cache (R5.2)

```swift
extension CALayer {
    internal var _hasSupportedFilters: Bool? {
        // Computed lazily; reset by .filters bit.
    }
}
```

Filter setter resets it to `nil`; getter computes and memoizes. Renderer
reads `layer._hasSupportedFilters ?? recompute()`. One-time compute per
`.filters` change.

### 7.3 `calculateClipRect` early-return (R5.3)

```swift
private func calculateClipRect(layer: CALayer, modelMatrix: Matrix4x4) -> ClipRect {
    guard layer.masksToBounds else { return currentClipRect }   // ← new
    // ... existing matrix work ...
}
```

Saves the 4-corner matrix multiply for non-clipping layers. Wholly local.

### 7.4 Cached opacity check (R5.4)

`renderLayer` opens with:
```swift
let presentation = layer.presentation() ?? layer
if presentation.opacity == 0 || presentation.isHidden { return }
```
With Phase 2's R2.1 cache, `presentation()` is now O(1) for clean trees, so
this early-return becomes essentially free. No code change beyond Phase 2.

### 7.5 TDD test sequence — Phase 5

`Tests/OpenCoreAnimationTests/Performance/DrawcallOptTests.swift`.

| # | Test | Asserts |
|---|---|---|
| 5.1 | `float32PoolReusesAcrossFrames` | Pool's `allocCount` does not grow after frame 2 |
| 5.2 | `hasSupportedFiltersCachedUntilFiltersChange` | Setting `.filters` once → 1 evaluation; querying 100 times → still 1 |
| 5.3 | `clipRectSkippedWhenMasksToBoundsFalse` | Counter on `calculateClipRect` does not increment for non-clipping layers |

Phase 5 tests are the simplest and most parallelizable — they can be written
alongside Phase 1 to deliver early wins.

---

## 8. Sequencing

```
Phase 1 (dirty bits + subtree counter)
        ├── Phase 5.2/5.3 (independent, can land same PR)
        ├── Phase 2 (presentation cache + sublayer ordering cache)
        │      └── Phase 3 (backing store + shouldRasterize + isOpaque + shadowPath + cleanup)
        └── Phase 4 (CARenderSnapshot + commit-driven render)
              └── Phase 5.1 (Float32Array pool — touches renderer body, easier after snapshot landing)
```

PR plan:

1. **PR 1**: Phase 1 + Phase 5.2/5.3. Adds dirty-bit foundation; no observable
   change in behavior. Smallest possible high-confidence change.
2. **PR 2**: Phase 2. First measurable perf win (presentation + sort caches).
3. **PR 3**: Phase 3 (rasterization). Largest pixel-correctness surface.
4. **PR 4**: Phase 4 (snapshot + commit-driven). Refactor; semantics
   unchanged.
5. **PR 5**: Phase 5.1 (Float32Array pool). Drop-in inside the renderer.

Each PR ships with its phase's full TDD test set green.

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Subtree counter drifts under aggressive re-parenting | Low | High (silent stale renders) | Mirror existing `_subtreeShadowCount` exactly; extend `subtreeCounterIdempotent` test to cover re-parenting. |
| R2.2 `self`-return breaks identity-sensitive code | Medium | Medium (subtle bugs) | Audit all `presentation() ===` checks (currently 0 in repo). Keep the renderer's `?? layer` fallback. |
| Rasterization cache pixel-diff vs. uncached | Medium | High (visual regressions) | Golden-image test in Phase 3.4 / 3.7. |
| Snapshot capture cost dominates the savings for small trees | Low | Medium | Capture only dirty subtrees from the previous snapshot; leaf reuse keeps cost O(dirty). |
| Float32Array pool grows unboundedly | Low | Medium | Bucket-by-power-of-two with `maxBuckets` cap; periodic shrink. |
| Frame-token global state breaks unit-test isolation | Medium | Low | Test helper resets `_currentFrameToken = 0` in setup. |

---

## 10. Test infrastructure

New file: `Tests/OpenCoreAnimationTests/Performance/_TestHelpers.swift`

```swift
import Testing
@testable import OpenCoreAnimation

/// Mock layer that counts presentation() invocations.
final class CountingLayer: CALayer {
    nonisolated(unsafe) static var presentationCallCount = 0
    override func presentation() -> Self? {
        Self.presentationCallCount += 1
        return super.presentation()
    }
    static func reset() { presentationCallCount = 0 }
}

/// Resets all global render counters between tests.
/// Called from every Performance/* suite's init().
func resetPerformanceTestState() {
    CALayer._currentFrameToken = 0
    CountingLayer.reset()
    // Add new global resets here as Phases 3/4/5 introduce them
    // (rasterization cache, snapshot pool, Float32Array pool counters).
}

/// Force-clears dirty state without invoking the renderer.
extension CALayer {
    func _testClearDirty() { recursivelyClearDirtyAfterCommit() }
}
```

#### Suite-serialization contract (B9)

Every `Tests/OpenCoreAnimationTests/Performance/*Tests.swift` file MUST:

1. Annotate the suite with `@Suite(.serialized)` so swift-testing does
   not parallelise tests within the suite — they share the
   `CALayer._currentFrameToken` global and would race.
2. Reset global state in `init()`:

```swift
@Suite(.serialized, "Dirty propagation")
struct DirtyPropagationTests {
    init() { resetPerformanceTestState() }

    @Test func freshLayerIsAllDirty() { ... }
    // ... rest of suite
}
```

Cross-suite parallelism is the next concern. swift-testing parallelises
*across* suites by default; even with `.serialized` *within* a suite,
two Performance suites running in parallel would still race on
`_currentFrameToken`. The fix:

- All Performance suites declare `@Suite(.serialized, .tags(.performance))`.
- A test plan or CI invocation passes `--no-parallel` for the
  `.performance` tag.
- For local `swift test`, a comment at the top of `_TestHelpers.swift`
  documents: "If you add a new Performance suite, run it via
  `swift test --filter Performance --no-parallel`."

A native-only `MockCARenderer` lives at
`Tests/OpenCoreAnimationTests/Performance/MockCARenderer.swift` and records
`submit` / `writeTexture` / `dispatch` calls and pipeline-state mutations.
Phase 3/4 tests assert against its call log instead of pixels. The
mock's `clearLog()` is part of `resetPerformanceTestState()`.

---

## 11. Summary — what gets built

- **Phase 1**: 1 new file (`CALayer+Dirty.swift`), ~80 lines of additions to
  `CALayer.swift` (setters), 1 internal recursive method.
- **Phase 2**: ~30 lines in `CALayer.swift` (token storage + `presentation()`
  rewrite), ~20 lines in `CALayer.swift` (`sortedSublayers()`), 1 line in
  `CAWebGPURenderer.swift` (token bump).
- **Phase 3**: 1 new file (`Rendering/WebGPU/Internal/RasterizationCache.swift`),
  ~100 lines in `CAWebGPURenderer.swift` (capture + composite + eviction),
  ~30 lines for `isOpaque`/`shadowPath` paths.
- **Phase 4**: 1 new file (`CARenderSnapshot.swift`), ~80 lines in
  `CATransaction.swift` (4-phase commit), ~50 lines refactor in
  `CAWebGPURenderer.render`.
- **Phase 5**: 1 new file (`Rendering/WebGPU/Internal/Float32ArrayPool.swift`),
  ~50 lines of call-site replacements, ~10 lines for `_hasSupportedFilters`,
  1 line for `calculateClipRect`.

Test files: 5 new files under
`Tests/OpenCoreAnimationTests/Performance/`, ~40 tests total.

End-to-end: `megaman` Playwright spec captures a 3 s frame-time histogram
before/after each PR; results posted in PR description.
