# OpenCoreAnimation

A Swift library providing CoreAnimation (QuartzCore) API compatibility for WebAssembly, powered by WebGPU.

## Overview

OpenCoreAnimation enables CoreAnimation-style code to run in the browser via WebAssembly. Full compatibility is the target; current completion is established by tests of specific API and renderer paths.

## Verified Status

| Evidence | Result |
|---|---|
| Native package | 761 tests passed |
| Browser | 3 checks passed, including immutable committed layer, compositing, background filters, scale-correct rasterization, true-3D transform containers, committed replicator instances, and value-owned emitter cells/images with clean-tree continuation and typed retry; float16 extended-dynamic-range color and `CGImage` output with SDR restoration; straight-alpha image normalization and alpha-correct mipmaps with typed malformed/non-finite-storage rejection; contents geometry; rasterized/tiled pixels; frozen transition pairs; multiple shadows; shape fill-rule holes; trimmed/dashed strokes; axial/radial/conic and 12-stop gradients; depth-preserving emitters/replicators; and replicated background/border/shape/image/gradient pixels read back from WebGPU |
| Layer defaults | `CALayer.defaultValue(forKey:)` returns QuartzCore-compatible typed defaults for geometry, contents, appearance, rasterization, and timing keys instead of treating every key as unknown. Shape, gradient, replicator, emitter, text, tiled, and scroll layers override their specialized defaults while inheriting base values. Fresh layers now use opaque-black borders, enabled edge antialiasing, infinite layer duration, and Helvetica text to match QuartzCore; native tests compare stored, zero/unknown, inherited, and instance defaults, while browser readback verifies default edge coverage and black-border rendering |
| Layer archiving | `CALayer.shouldArchiveValue(forKey:)` compares each supported base property with its QuartzCore archive default instead of returning a fixed success value. Shape, gradient, replicator, emitter, text, tiled, and scroll layers own their specialized decisions and defer unknown keys through the class hierarchy. Fresh and changed values are cross-checked against QuartzCore, including collection, delegate, timing, derived, and unknown-key behavior |
| Context rendering | `CALayer.render(in:)` renders in the receiver's coordinate space, composes sublayer position/anchor/bounds/transform geometry, groups subtree opacity once, and evaluates masks through a complete nested `destinationIn` buffer. `CAShapeLayer` fill/trimmed stroke and axial/radial/conic `CAGradientLayer` content use the same path with typed configuration failures; native tests inspect both renderer calls and final bitmap pixels |
| Text rendering | `CATextLayer` validates string, font, Float-safe quad geometry, scale, layout modes, converted foreground color, effective opacity, replicator tint, active corner geometry, final transform, and vertex capacity before entering Canvas2D or GPU upload. Commit-time configuration errors and frame-time Canvas/GPU failures use distinct typed contracts, so unsupported values, missing browser measurements, and capacity loss cannot become description strings, estimated sizes, or silent dropped draws. Static commits value-own the complete validated text/layout/style input; later model mutations cannot alter the submitted frame. New textures enter the LRU cache only after vertex allocation succeeds, so rejected draws cannot leave a cached partial result. CSS generic font families remain unquoted while concrete font names are escaped and quoted, avoiding environment-dependent fallback metrics. Text textures remain transparent over the separate layer-background pass, use typed cache keys, and rasterize at `contentsScale` while preserving point-space geometry. Browser readback verifies truncation, wrapping, multiline and justified pixels, immutable committed text pixels, and exact typed capacity rejection followed by retry recovery |
| Tiled rendering | `CATiledLayer` invalidates cached and in-flight tiles when bounds, scale, tile geometry, LOD, or display regions change. Every asynchronous request carries a cache generation, so work completed after invalidation cannot overwrite newer content or clear a replacement request. Tile geometry, scale, detail levels, renderer capacity, resource allocation, drawing-context creation, and asynchronous image creation now report typed WebGPU diagnostics instead of disappearing as empty tiles. Copies preserve configuration without copying cache state or manufacturing display work |
| Scrolling | `CAScrollLayer` applies point and rectangle scrolling only along the axes selected by the four documented modes. Unknown raw modes preserve the existing bounds origin for both overloads, matching QuartzCore instead of inconsistently acting like `.both` or jumping to the rectangle origin. Native tests cover all five mode paths and a browser probe executes the unknown-mode contract in WASM |
| Display invalidation | `CALayer.needsDisplay(forKey:)` preserves the QuartzCore base and non-text `false` contract while `CATextLayer` identifies its ten text/style/scale redraw keys. Text mutations set the public display-invalid state only when a stored value changes, and copy/presentation initialization transfers backing values without manufacturing redraw work |
| Explicit renderer | `CARenderer` is a QuartzCore-compatible class rather than the former backend protocol. `beginFrame`, automatic and explicit update regions, supplied media-time presentation evaluation, `nextFrameTime`, `render`, and `endFrame` execute over the same internal WebGPU/Metal backend contract used by the animation engine. Automatic regions include overflowing descendants plus both old and new extents so moved or removed content cannot leave stale pixels; active animations, unprocessed terminal frames, and effects conservatively invalidate the destination bounds. Layer and detached-mask trees share cycle-safe layout, scheduling, effect, and animation-completion traversal, so mask animations publish their next frame time and cannot strand their final frame or completion. WebGPU initialization, resize, and frame entry share one finite, integral, positive, device-limited render-target configuration. Missing or nonnumeric browser canvas dimensions are reported as exact typed configuration failures instead of being replaced with a fabricated `800×600` target. Frame entry reports the exact missing device, context, base pipeline/bind group, depth resource, committed capture failure, or revision-capture failure through `CAWebGPUFrameRenderFailure`; rejected resize requests preserve the previous public size and canvas state, and replaced depth textures are explicitly destroyed. After delegate backing stores are resolved, WebGPU captures layer and detached-mask content revisions. Successful submission clears only matching revisions and acknowledges only the committed generation it began with, so a later mutation or commit cannot be erased by an older frame. Native tests cover render-target boundaries, revision-safe clearing, lifecycle decisions, future/active/paused/terminal scheduling, mask traversal, and descendant removal, while browser diagnostics prove invalid initialization and resize requests are rejected atomically before continuing normal WebGPU rendering. The native renderer also reads an actual submitted Metal texture pixel; the release WASM build includes the canvas-backed initializer and Core Video timestamp stand-ins |
| Unified invalidation | Geometry, contents, shadows, filters, timing, masks, and hierarchy changes share `_dirtyMask` and `_subtreeDirtyCount`. The obsolete shadow/filter subtree counters and their hierarchy-wide propagation were removed because the renderer no longer consumed them and model-only counts could not represent animated presentation effects. Existing dirty-propagation tests cover clean-to-dirty idempotence, reparenting, ordering, masks, presentation isolation, and commit clearing |
| Transaction scheduling | Implicit transactions schedule only after a mutation creates work. Native delivery uses the owning thread's RunLoop; browser delivery validates `setTimeout`, its numeric safe-integer handle, and `clearTimeout`. Native and WASM share one pthread TLS stack contract rather than process-global or Objective-C dictionary storage. Unavailable or malformed host behavior remains visible through typed diagnostics instead of trapping or fabricating success. Manual outermost commits and `flush()` cancel active timers, while a shared generation prevents stale RunLoop or browser callbacks from committing a later transaction. If an explicit transaction temporarily blocks delivery, committing it reschedules the remaining implicit level instead of stranding its work. Native tests prove two-thread isolation, owning-thread callback delivery, exactly-once thread-exit release, handle boundaries, diagnostics, and explicit-over-implicit ordering; the browser probe covers unavailable APIs, malformed handles, exact boundary-handle cancellation, recovery, stale-callback isolation, and blocked-delivery rescheduling |
| Transaction completion | Completion blocks for property, hierarchy, mask, and display mutations are coordinated across nested transactions and distinct render roots. A renderer releases each render obligation only after command submission and committed dirty-state clearing; animation obligations remain independently outstanding until their animations stop. Completion callbacks can mutate the tree without those new dirty bits being erased by the preceding frame. Native tests cover disabled actions, explicit and implicit animations, groups, removal, detached masks, multiple mutation categories, and callback reentrancy. A Chromium probe verifies that commit leaves the block pending and the following WebGPU frame releases it. Static common trees publish immutable snapshots and WebGPU encodes their value-owned geometry, transforms, colors, borders, corners, nested rectangular/rounded clipping, ordinary and nested `CALayer.mask` trees, true versus distributed group opacity, scale-qualified rasterization policy, layer-filter and backdrop-composition plans, subtree and explicit-path shadows, backface policy, dynamic-range policy, ordinary and delegate-generated image bytes and sampling state, gradient stops and geometry, commit-time tessellated shape fill/stroke vertices and colors, complete text layout/style configuration, true-3D `CATransformLayer` child depth, expanded `CAReplicatorLayer` instance subtrees, emitter simulation input and image bytes, and stable child order without reading `CALayer`; native tests prove source-image release, full/partial delegate drawing, explicit-content replacement, typed capture failure, rasterization/filter/backdrop/mask/group/shadow/gradient/shape/text/transform/replicator/emitter value isolation, and commit-time shadow/shape-path tessellation, while browser readback verifies captured color, clipping, backface, ordinary image, delegate pixels, multiplicative nested-mask alpha, overlapping group-opacity semantics, immutable rasterized pixels at the captured GPU scale, immutable gradient, shape, text, transform, replicated, and emitter output, immutable `CAFilter`, `CIFilter`, `compositingFilter`, and `backgroundFilters` results, and both shadow silhouettes after model/delegate/filter/mask/path mutation. Invalid committed transform, replicator projected depth, or emitter graph remains an exact typed failure, retains completion, and recovers through a corrected commit. Non-`CGImage` model or emitter-cell contents now fail at commit with their concrete typed contract instead of entering a static path that silently draws nothing; a valid layer delegate bitmap still takes precedence. Snapshot capture or solid, clipping-mask, contents, shape, text, transform-depth, replicator, emitter, rasterization, filter, backdrop-composition, shadow, and subtree-composite prepass encoding failures preserve the pending render completion instead of submitting or clearing dirty state; browser retries prove the same committed frame can recover and only then releases completion. Base-only `CALayer` subclasses, `CAScrollLayer`, `CATextLayer`, `CATransformLayer`, `CAReplicatorLayer`, and static `CAEmitterLayer` trees use the immutable value path, including committed scroll offsets, validated text configuration, insertion-order child capture, cumulative instance transform/color/time, emitter simulation identity/state, and depth-group state. Tiled foreground resources and transitions publish an explicit typed live-resource requirement until their value/resource ownership is migrated. Immutable animation evaluation and the remaining resource-backed WebGPU snapshot encoding remain open Phase 4 work |
| Native engine verification | The native `CAAnimationEngine` uses the real offscreen `CAMetalRenderer`, lazily creates a destination from the root bounds, submits a command buffer, records synchronous failures explicitly, and clears dirty state only after submission. Outermost transactions resolve parent-to-child layout to a fixed point before publishing static render roots through a mutex-protected `Sendable`, CALayer-free value snapshot; layout callbacks that add or resize descendants are therefore captured in the same committed generation. Metal consumes the snapshot features it supports and returns `unsupportedCommittedSnapshotFeature(.imageContents)`, `.contentMask`, `.groupOpacity`, `.rasterization`, `.filters`, `.backdropComposition`, `.shadow`, `.gradient`, `.shape`, `.text`, `.transformDepth`, `.replicatorInstances`, or `.emitter` for committed resources it cannot encode instead of silently omitting them. Its recursive encoder reads only snapshot nodes. Node revision matching prevents an older submission from clearing a mutation made after capture, while commit generation matching prevents an older frame from acknowledging newer state. Native readback captures green, mutates the model to red, submits green, and proves that the red mutation remains dirty for the next commit. Device-gray backgrounds are converted to device RGB, while non-finite geometry, colors, rasterization scale, filters, gradients, shapes, text, transforms, replicators, emitters, and shadows remain typed committed failures instead of becoming successful fallback pixels. Animated trees remain explicitly marked for live evaluation until immutable animation evaluators exist. Display-link ticks skip submission for clean static trees, future animations, and paused animations after their committed state is drawn. Dirty descendants, progressing animations, one terminal frame, and renderer-owned tile or emitter work submit; manual `renderFrame()` remains unconditional. Detached masks participate in the same cycle-safe layout, submission, frame-rate, and completion traversal as sublayers. Immutable animation evaluators, remaining WebGPU snapshot migration, and active-subtree-only presentation evaluation remain open Phase 4 work |
| Animation and emitter-cell defaults | `CAAnimation`, every concrete animation subclass, and `CAEmitterCell` expose QuartzCore-compatible defaults and `shouldArchiveValue(forKey:)` decisions for persistent state. Runtime-only frame-rate hints and spring initial velocity remain intentionally unarchived, while unknown keys fail closed. `CAEmitterCell` includes `style`, defaults to white, enabled, and infinite duration, and distinguishes positive infinity from invalid non-finite timing. Native tests cover typed values, subclass inheritance, every archive key, canonical/unknown keys, style storage, indefinite emission, infinite repeats, and typed timing failure; browser emitter diagnostics exercise the default infinite duration with zero spawn failures |
| Shape fills | `CAShapeLayer` tessellates all path contours as one fill, preserving `.nonZero` winding and `.evenOdd` parity for nested, overlapping, coincident, curved, open, and self-intersecting subpaths. Path validation and tessellation use typed throws; unknown rules, non-finite paths, invalid device-RGB colors, missing renderer resources, and vertex-capacity exhaustion retain a `CAShapeRenderFailure` instead of producing fallback geometry or silently dropping work. A failed fill does not suppress an independently valid stroke. Shape draws select their own solid/stencil/depth pipeline so output does not depend on the preceding sibling type. Native geometry tests and browser pixel readback cover both fill rules, holes, submitted draw/vertex counts, and exact typed rejection of an unknown rule |
| Shape strokes | `strokeStart` and `strokeEnd` trim against total length across all subpaths, then the shared OpenCoreGraphics geometry path applies `lineDashPattern`, phase continuity, line caps, line joins, and miter limits before the outline enters the same WebGPU tessellator as fills. Raw animation values remain unclamped in the model and clamp only at rendering. Invalid geometry, dash patterns, unknown styles, non-finite widths, invalid device-RGB colors, and vertex-capacity exhaustion report typed failures. Native tests cover trim, multi-subpath ranges, dash/phase, cap styles, and failure contracts; browser pixels prove trimmed alternating dash segments reach WebGPU |
| Gradients | `CAGradientLayer.type` selects axial projection, unit-coordinate elliptical radial distance, or conic angular progression around `startPoint` with the `startPoint`→`endPoint` ray as zero. Colors convert to finite device-RGB upload components and locations are validated in one typed configuration; unknown types, non-finite geometry, non-color values, and invalid location sequences are rejected instead of being replaced. Gradient-stop resource loss, byte-count/capacity overflow, buffer-pool capacity, stop-offset range, and vertex capacity retain a `CAGradientRenderFailure` rather than incrementing an undifferentiated count or silently dropping a draw. Gradient stops use dynamically growing, triple-buffered read-only GPU storage rather than a fixed uniform array. The explicit `render(in:)` path draws all three modes and maps the same typed configuration failures. Native tests cover color conversion, geometry, final bitmap pixels, unbounded valid stop lists, and every input-failure class; browser pixels verify all three modes, a 12-stop gradient that crosses the former eight-stop boundary, and exact typed rejection of an unsupported type through WebGPU readback |
| Corner curves, content masks, and stencil clips | `.circular` and `.continuous` use distinct calibrated corner geometry in the native path and matching Lp signed-distance fields in WebGPU. The curve exponent is carried through solid fills, borders, gradients, textured contents, rounded stencil clips, content masks, composition clips, and shadow cache identity. Ordinary `CALayer.mask` rendering captures the complete detached mask tree into transparent storage and multiplies its rendered alpha into the captured layer subtree, preserving partial coverage and filtered descendants without treating mask bounds as an opaque rectangle. Unknown raw values retain a `CACornerCurveRenderFailure` with the exact configuration error and renderer context (`layer`, `mask`, or `roundedClip`) instead of falling back to circular corners or dropping a requested mask without a cause. Mask resources, Float-safe geometry and transforms, vertex capacity, stencil-reference overflow, pipeline availability, and depth/reference state invariants retain `CAMaskRenderFailure`. Every stencil increment used for rounded and transformed `masksToBounds` clips owns the exact vertex/uniform ranges needed to draw a matching decrement during unwind; only after that GPU restoration succeeds do depth and reference state move to the parent. This prevents one rounded sibling's stencil values from corrupting a later overlapping sibling. Scissor remains limited to rejection optimization. Masked image, nine-slice, text, and tile draws never substitute an unmasked pipeline when their stencil variant is unavailable. Native boundary tests distinguish both curve shapes and invalid mask geometry; browser readback verifies solid, texture, gradient, `masksToBounds`, unfiltered partial-alpha mask trees, overlapping sibling restoration, rejected paths, and exact typed frame rejection |
| Backgrounds and borders | Background and border color, opacity, Float-safe bounds, corner geometry, border width, final transform, resources, vertex capacity, and stencil/depth-aware pipeline selection enter one validated solid-quad path. Failures retain `CASolidRenderFailure`; pipeline selection now completes before vertex reservation or buffer writes, so unavailable pipelines cannot consume frame capacity. Extended-range finite colors remain supported. Native tests cover valid HDR inputs and typed invalid opacity/border values; browser pixels cover ordinary, rounded, masked, depth-preserving, replicated, and animated backgrounds and borders |
| Boolean and discrete animations | `hidden`, `masksToBounds`, `doubleSided`, and `shouldRasterize` participate in basic and keyframe presentation evaluation. Discrete keyframes select the latest reached value, single-value sequences apply directly, scalar position components animate independently, and values hold at the final key time. Browser pixels verify visibility, clipping, backface rendering, and explicit rasterization capture |
| Path keyframe animations | `CAKeyframeAnimation.path` preserves line, quadratic, cubic, closing, and discontinuous subpath geometry. Linear/cubic modes use segment timing, paced modes use adaptive arc-length traversal, discrete mode holds segment starts, and valid `keyTimes` plus per-segment timing functions control non-paced motion. Additive and cumulative position contributions compose with model state, while automatic rotation concatenates with the existing transform using QuartzCore-compatible `CATransform3D` ordering. Invalid or non-finite paths leave the complete presentation state unchanged. Native tests compare measured QuartzCore positions and transforms; browser presentation checks plus GPU pixel readback cover linear, paced, discrete, keyed, curved, and rotated paths |
| Contents animations | `contents` uses QuartzCore-compatible midpoint selection for basic and interpolated keyframe segments, while discrete keyframes hold the latest reached image. Presentation copies use backing storage so evaluation does not enqueue model transactions. Browser readback verifies red, green, and blue `CGImage` selection at the boundary times |
| Contents geometry | Single-quad and nine-slice image rendering validate image, bounds, crop, and scale through typed configurations. The twelve documented gravity values resolve explicitly, while unknown raw values use QuartzCore's measured center layout consistently in `render(in:)` and WebGPU instead of becoming resize output or a renderer-specific failure. `contentsRect` controls both UVs and the logical size used by gravity before `contentsCenter` subdivides the selected image; `contentsScale` converts fixed source pixels to points. Magnification/minification filters and mip bias select sampler-aware cache entries rather than reusing stale bind groups. At transaction commit, ordinary `CGImage` contents are converted into value-owned storage with all geometry and sampling state; later image or layer mutations cannot change the in-flight frame. Configuration, image conversion/mipmap generation, texture-manager/factory resources, renderer resources, and vertex capacity retain a `CAContentsRenderFailure`. Nine-slice reserves every patch vertex before uploading any draw, preventing capacity pressure from producing a partially rendered image. Texture conversion failures propagate into emitter and tiled-layer diagnostics instead of becoming generic resource loss. Native tests cover crop/center/scale/gravity and snapshot ownership decisions while browser readback verifies cropped blue/magenta/yellow slices, committed red/green pixels after blue/magenta mutation, unknown-gravity centering, centered non-sliced contents, invalid-center rejection, and typed malformed-image and snapshot-capacity rejection |
| Rasterization | `rasterizationScale` mutations resolve layer actions and basic/keyframe presentation values drive the explicit offscreen capture dimensions. Capture bounds, scale, scaled extent, Float-safe projection, composite resources, destination bounds, vertex capacity, and pipeline selection now report typed failures. Invalid captures cannot reuse an earlier pipeline or submit a partial composite, and each frame starts without a stale rasterized texture. Browser diagnostics verify a 40-point layer animated from scale 1 to 2 captures at 60×60 pixels at half progress and prove that an invalid scale is rejected with its exact failure reason |
| Shadow path keyframes | `shadowPath` accepts single-value and discrete keyframes and applies compatible linear/cubic path morphs to the presentation layer. Browser shadow pixels verify that discrete and cubic path selection reaches the explicit-path tessellation renderer |
| Transform component animations | `transform.rotation[.x/.y/.z]`, `transform.scale[.x/.y/.z]`, and `transform.translation[.x/.y/.z]` read and replace decomposed transform components instead of multiplying the model matrix. Basic and keyframe animations share linear, discrete, cubic, additive, and cumulative evaluation. Browser pixels verify that translation, scale, and rotation presentation transforms reach WebGPU vertex rendering |
| Basic aggregate animations | `CABasicAnimation` applies the complete from/to/by endpoint contract to rectangles, colors, gradient color/location arrays, and full layer, sublayer, and replicator transforms. Color endpoints use managed sRGB conversion when a source profile is available and an explicit device-RGB conversion otherwise, then require finite interpolation and additive/cumulative results; monochrome, CMYK, and profiled RGB inputs are never reinterpreted as raw RGBA. Unsupported pattern colors, non-finite components, arithmetic overflow, singular transforms, and incompatible arrays leave the complete presentation value unchanged instead of producing black, partial, or fabricated values. `toValue + byValue` uses verified inverse transform resolution. Additive colors start from transparent black, explicit color and gradient-array endpoints work without model values, and presentation evaluation writes backing storage without registering model transactions. Native tests cover every endpoint family and failure path; browser pixels verify bounds expansion, RGB addition, CMYK interpolation, and full-transform translation reach WebGPU rendering |
| Keyframe timing | Value and path animations share one finite unit-range `keyTimes` validator. Linear and cubic modes require one time per value with `0` and `1` endpoints; discrete value animations use the documented interval form with one more time than values; paced modes ignore explicit times. Invalid or inappropriate arrays receive the documented evenly spaced timing rather than being partially consumed, while unknown calculation modes leave presentation state unchanged instead of fabricating linear interpolation. Native tests cover count, endpoint, range, ordering, non-finite, discrete-interval, path, and unknown-mode behavior; browser presentation and pixel readback verify invalid-time recovery and unknown-mode rejection reach WebGPU rendering |
| Timing-function failures | Basic, keyframe, path, group, transition, and scheduling paths use one checked Bézier evaluator. Non-finite input time or control points remain an explicit non-finite public evaluation result and cause animation application to leave the complete presentation state unchanged; malformed timing can no longer inject `NaN` geometry or partially advance an animation graph. Native tests exercise direct evaluation and property/graph failure paths, while browser presentation and WebGPU pixel readback verify model-state preservation |
| Timing-function names | The five predefined names resolve to their documented cubic Bézier control points. Unknown raw names terminate initialization, matching QuartzCore's invalid-name contract, instead of silently fabricating a linear timing function. A native subprocess exit test exercises the real failure path without terminating the test runner |
| Media-timing failures | Animation, transition, group, completion, renderer scheduling, and frame-rate arbitration share one validity result for `beginTime`, `timeOffset`, `speed`, duration, repeat count, and repeat duration. Non-finite timing is preserved through duration resolution, never substituted with transaction defaults, never changes presentation values, and terminates with `finished: false` instead of being reported as a successful completion. Native tests cover every timing input plus presentation, completion, and scheduling behavior; browser presentation and WebGPU pixel readback verify invalid-speed rejection |
| Additive keyframes | Linear, paced, discrete, cubic, and single-value keyframes share one typed interpolation-to-application path. Scalar, geometry, RGBA, gradient-array, full-transform, and specialized-layer values add to presentation state instead of replacing it; `CFTimeInterval` and `CGFloat` scalar inputs normalize at the API boundary. Cumulative gradient colors and locations carry complete terminal arrays across repeats, while incompatible array lengths or elements leave the complete presentation array unchanged. Native tests cover base, shape, text, emitter, gradient, and replicator layers; browser pixels verify additive position, bounds, and color reach WebGPU rendering |
| Animation graphs | Root and nested `CAAnimationGroup` trees are evaluated in graph-wide non-additive and additive passes, so child-array or top-level dictionary order cannot overwrite additive contributions. Group timing functions remap child basic time and compose with each child's own timing function. Grouped transitions receive a recursively captured source tree and evaluate in the group's repeating basic time. Lifecycle callbacks belong to the group attached to the layer; nested groups and child animations are evaluated without emitting partial start-only delegate events. Native tests cover direct, nested, and root-crossing ordering, hierarchical pacing, grouped transition state, natural completion, explicit removal, retention, and transaction completion; browser pixels exercise a deliberately additive-first mixed group through presentation and WebGPU rendering |
| Constraint layout | `CAConstraintLayoutManager` solves each independent system of sibling and superlayer equations simultaneously, preserving unconstrained geometry and isolating inconsistent components instead of depending on sublayer or constraint order. Bounds, constraint, layout-manager, and hierarchy mutations invalidate layout, notify the manager once per clean-to-dirty transition, and converge repeated invalidations before `CAAnimationEngine` renders parent-to-child. Native tests cover nonzero bounds origins, coupled edge sizing, reversed sibling chains, missing sources, conflicts, notification/reentry, and automatic render-time layout; browser pixels verify the solved sibling frames reach WebGPU rendering without a manual layout call |
| Value-function animations | `CAValueFunction` rejects unknown names and validates input arity instead of producing an identity fallback. Scalar functions and the three-component `.scale` / `.translate` functions participate in basic and keyframe linear, discrete, cubic, paced, additive, and cumulative evaluation. Non-additive animations replace the model transform, while additive animations concatenate with it. Native tests cover integer and floating-point inputs, and browser pixels verify aggregate keyframe translation reaches WebGPU rendering |
| Edge antialiasing | `allowsEdgeAntialiasing` and each `CAEdgeAntialiasingMask` bit drive derivative-based WebGPU coverage in layer-local coordinates. Solid, border, gradient, shape, image, text, nine-slice, and exterior tiled edges share the contract; internal tile seams remain untouched and captured layers are not antialiased again during final composition. Browser readback verifies disabled, left-only, right-only, runtime mask mutation, and textured-content pixels |
| Filters and transitions | Sibling and nested `CAFilter` chains use per-layer WebGPU resources with browser pixel and cleanup evidence. Built-in fades interpolate frozen premultiplied source/target RGBA in one fragment pass, preserving translucent and transparent pixels under stencil and true-3D depth. Transition source/target bounds, contents scale, device-limited pixel extent, Float-safe projection, filter setup, progress, and dispatch preserve participant-aware typed failures instead of returning `nil` or a count alone. Composite resources, Float-safe bounds/offsets, opacity, vertex capacity, and pipeline selection are also typed; directional source/target allocations are reserved as one batch before either draw, and every successful path restores the base pipeline. All 7 executable OpenCoreImage transition pipelines have filter-specific browser pixel evidence, while unsupported transition filter objects, unknown built-in transition types, and unknown directional subtypes are rejected with their exact reason and without target-image or `.fromLeft` fallback |
| Shadows | Every visible shadow owns an independent mask, blur target, and uniform set; a nil `shadowPath` derives its silhouette from the rendered subtree alpha, while an explicit path uses the same complete-path non-zero tessellation as shape fills. Shadow geometry, device-RGB color, effective opacity, replicator color, offset, and viewport enter typed configurations before GPU work, preserving finite extended-range colors while rejecting non-finite or Float-overflowing output. Path tessellation, rasterized-shadow resources, pre-render resources, composite pipelines, and vertex capacity report `CAShadowRenderFailure`; pre-render and display failures have independent deduplicated state, a failed path is never blurred or cached as a successful empty mask, and a missing stencil pipeline never falls back to unmasked rendering. The display path resolves pipelines and reserves vertex capacity before any uniform or vertex-buffer write. Silhouettes are captured at the content position and `shadowOffset` is applied once during display for raw, filtered, masked, and explicit-path inputs. Rendered mask-tree alpha, including filtered descendants, active transitions, and partial coverage, shapes the shadow; detached mask mutations invalidate the cached silhouette. Native tests cover color conversion, composite color math, Float boundaries, and exact invalid geometry/color/opacity/viewport reasons; browser evidence covers exact pre-render and display-stage typed rejection, transparent image pixels, sublayer-only content, empty content, multiple shadows, inherited opacity, animated `shadowOpacity` from a zero model value, ancestor transform invalidation, mask transition and mutation, empty and holed `shadowPath` values, and resource cleanup |
| Emitters | Particle simulation state, fractional birth accumulation, random state, and cleanup are isolated per stable model `CAEmitterLayer` identity. Shape, emission mode, render mode, geometry, and simulation multipliers enter one validated immutable configuration before state mutation. Cell timing, birth rate, contents, converted color, direction, finite particle state, particle capacity, image conversion, texture/pipeline resources, and vertex capacity now report typed spawn/render failures rather than returning an undifferentiated count or silently dropping work. A mutex-protected weak owner registry propagates root or nested `CAEmitterCell` mutations to every attached emitter layer, so an already committed frame remains immutable while the next transaction captures the new recursive cell graph; detached cells stop invalidating former owners. All documented shape/mode combinations and uniform 3D emission cones honor emitter geometry, latitude, longitude, and range. Every render mode is active, including z-sorted back-to-front rendering, source-additive compositing, and stencil-aware masked particles. `preservesDepth` enters the shared overflow-checked depth-group transition, restores the exact prior nesting value, and selects direct particle depth writes inside 3D hierarchies; the default path captures particles as one plane, and emitter-containing rasterization captures are refreshed rather than freezing simulation behind a clean dirty mask. `CGImage` cells commit value-owned `contentsRect`, `contentsScale`, tint, magnification/minification filters, mip bias, and converted bytes; nil contents remain simulated but invisible, while unsupported content is rejected. Nested child cells honor their media-timing window, emit from the parent's moving position, orient relative to its current direction, inherit its animated color and scale, and support later generations. Native tests cover layer validation, shared-owner revision invalidation and recapture, recursive attachment, and detachment; browser evidence covers immutable commit/clean-frame continuation, cell-driven typed retry, child-cell semantics, image cropping/scaling, linear versus trilinear pixels, ordering, additive pixel readback, depth-path selection, concurrent low-rate emitters, rectangle-outline and sphere-surface geometry, orthogonal 3D velocities, and independent removal |
| Replicators | Instances traverse the normal layer renderer, with cumulative transforms, color multiplication/offsets, nested inherited state, `instanceDelay` animation evaluation, and a true zero-instance result. Static commits expand every instance into an independent immutable node subtree with copy-on-write value storage, absolute inherited color/time, a fixed pre-parent transform, and unique node indices for filter, shadow, mask, rasterization, and backdrop prepasses. Later changes to instance count, transform, color offsets, or descendants cannot alter the submitted frame. Instance count, delay, transform, converted color, and offsets enter one validated configuration; capacity violations and per-instance color/time/transform/projected-depth overflow report typed renderer failures instead of traversing unbounded state or submitting non-finite GPU data. Depth-preserving instances resolve every homogeneous center depth and enter the shared overflow-checked depth-group transition before clearing GPU depth state, then restore the exact prior nesting value. With `preservesDepth`, replicated descendants share a depth group, draw far-to-near for translucent blending, and use per-pixel depth tests instead of flattening in instance order. Native tests cover immutable expansion, value isolation, the depth-container contract, validation, and Metal rejection; browser pixels cover committed instance immutability, typed capture/depth failure recovery, background, border, shape, image, gradient, delayed opacity/color, all three offscreen paths, and the visible flat-versus-depth occlusion difference |
| Group opacity | Translucent layers with `allowsGroupOpacity` capture their complete subtree and apply opacity once during premultiplied-alpha composite; disabling the property preserves per-component opacity. Browser pixels verify overlapping opaque and translucent children in both modes |
| Layer filters | `CAFilter` stages and executable `CIFilter` objects run in declared mixed order against the captured subtree. Static commits freeze `CAFilter` operations plus supported scalar, integer, Boolean, vector, color, point, size, rectangle, and affine `CIFilter` parameters into a `Sendable` value plan; WebGPU reconstructs private execution objects from that plan and never retains or rereads the caller's mutable filter. A typed execution plan validates parameter names, numeric types, finite values, and documented ranges before dispatch, and materializes sepia/vignette defaults explicitly at the Core Image boundary. Unsupported or non-finite Core Image parameter values fail snapshot capture through `CARenderSnapshotFilterError` instead of being ignored. Explicit straight/premultiplied-alpha conversions preserve translucent edges there. Final display validates effective opacity and replicator color, resolves its sampler, stencil-aware pipeline, and restoration pipeline before writing uniforms, and rejects a missing stencil variant instead of drawing an unmasked fallback. Pre-render, rasterized-filter execution, and display failures retain exact per-layer state so a persistent failure is counted once and retried without being mistaken for successful filtering. Invalid configurations, unavailable filter types, renderer operations, Core Image dispatch, alpha conversion, content-mask, and display failures retain an exact `CALayerFilterRenderFailure`; incompatible multi-input filters are rejected and never silently render an unfiltered layer. A content-mask preparation failure is promoted to `CAWebGPUFrameRenderFailure.contentMaskPreparationFailed`, aborts command submission and dirty-state clearing, and is retried from fresh per-frame state until the caller fixes the cause. Native snapshot tests verify value isolation and typed rejection; browser readback proves committed `CAFilter` and `CIFilter` output survives later deletion/disablement, while unsupported parameters retain the committed completion obligation |
| Compositing filters | Executable two-input `CIFilter` operations receive a draw-order-accurate backdrop captured immediately before the layer. Static commits freeze the compositing filter name, enabled state, supported parameters, background-filter stages, geometry, masks, and hierarchy into `Sendable` values; the snapshot compositor reconstructs private filter objects and evaluates deepest targets first without retaining or rereading mutable model layers. Directly inherited opacity and replicator color scale only the premultiplied source before that operation; the backdrop is not faded. Group-opacity and filtered ancestors create transparent local scopes, apply their opacity/filter once after child composition, and then rejoin their parent backdrop. Replicator instances own distinct backdrop resources and preserve instance transform, color offsets, and ordering. Nested composition is evaluated deepest-first; each parent source is recaptured after its children complete, preventing global backdrop leakage and double composition. Ancestor transform/effect flattening and explicit `shouldRasterize` captures are deferred until their descendant compositions exist, then recaptured without stale backdrop-cache reuse. Local captures reproject viewport composition coordinates through validated homogeneous interpolation, while true-3D display planes validate Float-safe layer/viewport geometry and final transforms before retaining depth writes. Display resources and vertex capacity resolve before uniform or vertex-buffer writes, so a prepared texture cannot fail silently or partially update its final composite. Consecutive operations feed the preceding composite into the next backdrop, later siblings remain on top, and the cumulative snapshot replaces rather than re-blends the framebuffer. Transformed, rounded, nested `masksToBounds` shapes and rendered `CALayer.mask` trees are rasterized independently and intersected as full-viewport coverage masks for both source composition and backdrop filtering. Detached mask trees own transparent backdrop roots, recursively resolve nested `compositingFilter` and `backgroundFilters` targets before the main tree, and retain all sibling-context resources until command submission. Mask-root `filters` execute mixed `CAFilter`/`CIFilter` stages with explicit alpha conversion before becoming coverage. `backgroundFilters` execute the same mixed stages against the layer bounds when clipped and the parent/full backdrop extent when unclipped. Capture distinguishes ordinary, compositing, and background filter failures with typed errors. Planning and execution failures for both background and content-mask filters retain the nested `CALayerFilterRenderFailure`; source/backdrop capture, filter support, alpha conversion, mask geometry, mixing, composition dispatch, final-display resource, transform, sampling, and capacity failures retain their exact `CACompositionFilterRenderFailure`. Failed or unprepared paths never fall through to unprocessed source-over. Browser pixels cover multiply, screen, source-in mask alpha, mask-local backdrop blur, direct/group source opacity, translucent backdrop replacement, mixed background/mask-filter stages, nested rounded and content-mask clipping, replicated color/transform, nested filtered and rasterized scopes, projective reprojection, ordered chaining, immutable committed composition/background filters, typed capture/unsupported-object/content-mask dispatch rejection, and resource eviction |
| Display link | Fresh timing values match QuartzCore's zero state. After delivery, `duration` estimates the maximum physical refresh cadence while `targetTimestamp - timestamp` reports the selected callback interval, including factor-based 30 fps delivery on a faster display. Browser timestamps and unsigned request identifiers are validated before entering state; ID zero remains valid, malformed values produce `CADisplayLinkSchedulingFailure`, delegates do not receive fabricated time, and a new scheduling generation prevents stale callbacks from crossing pause/restart boundaries. `CACurrentMediaTime()` returns an explicit invalid value when `performance.now()` is unavailable or malformed instead of substituting time zero. Non-finite frame-rate hints cannot create invalid timers, pause/resume is terminally separated from mode registration, and independent modes retain or stop delivery correctly. Native run-loop callbacks and browser rAF verify monotonic timing, throttling, typed rejection, recovery, pause/resume, removal, invalidation, and target release |
| Animation frame-rate hints | `CAAnimation.preferredFrameRateRange` is preserved by defensive animation copies and arbitrated across active animations, nested groups, and the complete layer tree. Future and completed animations do not affect the current request; the highest-demand active range is submitted to `CADisplayLink`, while the engine baseline is restored when no explicit hint is active |
| Dynamic range | `CALayer.ToneMapMode`, `CALayer.DynamicRange`, `toneMapMode`, `preferredDynamicRange`, and `contentsHeadroom` match current QuartzCore names, raw values, defaults, and copy/presentation behavior. The WebGPU renderer uses an `rgba16float` canvas, validates the complete visible layer tree, switches canvas tone mapping between standard and extended modes, and reports typed failures for invalid policy, invalid headroom, or unavailable explicit HDR output. Float, HDR-tagged, and extended-space `CGImage` contents are converted to straight-alpha extended-linear RGBA16Float without the former RGBA8 quantization; SDR images stay RGBA8. Both formats receive alpha-correct mip chains and format-aware cache accounting. Browser GPU readback proves that `(2.0, 0.5, 0.25, 1.0)` survives both color and image texture paths, an invalid `0.5` headroom rejects the frame, and removing HDR content restores standard output. Rasterization budgets account for the eight-byte pixel format |
| Delegate drawing | Ordinary `CALayer` instances with a delegate consume `setNeedsDisplay()` during transaction snapshot preparation, run `display(_:)` before `layerWillDraw(_:)` / `draw(_:in:)`, rasterize the draw callback into a layer-owned `contentsScale`-aware software backing store, and copy its immutable pixels into the same committed image contract as explicit contents. Layers without a delegate retain the independent display-invalidation axis. `contentsFormat` selects RGBA8, extended-linear RGBA16Float, or Gray8 storage; `.automatic` upgrades to Float16 for extended headroom, while unknown formats publish a typed capture failure. Rectangular invalidations union until display, preserve pixels outside the clipped update region in the same storage format, clear stale pixels inside it, and retain invalidations raised reentrantly by `display(_:)`. A `display(_:)` contents assignment supersedes software drawing, explicit contents releases an older store, full redraw replaces it, and detached layers release it with their owner. Invalid/non-finite extents, texture-limit violations, and checked row/storage-size overflow reject the complete frame before bitmap allocation rather than submitting missing delegate pixels. Failed invalidations remain pending, so committed and live-tree retries continue to fail explicitly until the cause is corrected; successful recovery consumes the request. Native tests verify bitmap bytes, partial preservation, repeated-snapshot reuse, display override, re-invalidation, explicit replacement, allocation-boundary rejection, and recovery behavior; browser readback verifies commit-time capture against a later delegate mutation, partial/full redraw, both Y-up and `isGeometryFlipped` orientation, HDR values above 1.0, Gray8 expansion, repeated live-tree failure, unknown-format rejection, and pending completion on capture failure |
| Text layout | `CATextLayer` shares one Canvas-measured layout path between sizing and rendering. Width wrapping preserves Latin separators without inventing spaces between CJK characters, explicit LF/CRLF/CR paragraph breaks render even when `isWrapped` is false, oversized tokens remain grapheme-safe, and `.justified` distributes Latin words or CJK characters while leaving each paragraph's final line unchanged. `.start`, `.middle`, and `.end` truncation retain extended grapheme clusters; unknown modes do not silently fall back. Browser pixels verify truncation placement, wrapped overflow, mode-change cache invalidation, explicit multiline rendering, and first-line justification |
| Cubic keyframes | `.cubic` and `.cubicPaced` use Kochanek-Bartels tangents with per-control-point tension, continuity, and bias defaults. First and last segment control points are extrapolated from adjacent differences, matching QuartzCore endpoint slopes instead of halving them through duplicated endpoints. Cubic-paced values derive key times from finite Euclidean component distance and scale adjacent tangents for the resulting nonuniform intervals; unsupported paced values leave presentation state unchanged instead of receiving fabricated equal spacing. Scalar, point, size, rectangle, color, transform, gradient-array, and compatible-path values retain their QuartzCore-specific interpolation or even-spacing contracts without linear fallback. Native tests assert measured scalar, point, size, interior, endpoint, and failure behavior; browser presentation and pixel readback exercise the paced tangent path through WebGPU rendering |
| Spring animations | `CASpringAnimation` exposes the current QuartzCore perceptual API (`init(perceptualDuration:bounce:)`, `perceptualDuration`, `bounce`, and `allowsOverdamping`) and derives its physical coefficients from the same damping-ratio mapping. The response evaluator distinguishes critical and explicitly enabled overdamped motion without rewriting stored coefficients. Settling estimates replace the former four-time-constant approximation and are checked against measured QuartzCore underdamped, critical, initial-velocity, and infinite-damping boundaries |
| Geometry flipping | `isGeometryFlipped` reflects descendant geometry around the layer bounds while leaving the layer's own contents plane unchanged. Coordinate conversion, inverse hit testing, ordinary traversal, transform/depth groups, filters, masks, shadows, and rasterization use the same flipped parent matrix. Native tests match QuartzCore values for nonzero bounds origins, arbitrary anchors, and rotation; browser readback verifies that asymmetric child layers exchange vertical positions after a runtime flip |
| Hit testing | Normal layer trees traverse stable `zPosition` order. Coordinate conversion and picking compose each hierarchy's full 4×4 transform, including parent `sublayerTransform`, then project and invert the resulting plane homography so perspective and out-of-plane rotations remain aligned with rendered pixels. A `nil` conversion endpoint uses the receiver's superlayer coordinate space for points and all four rectangle corners, including anchor, transform, and geometry-flip effects. Singular projections return non-finite coordinates instead of silently using an unrelated affine result. `CATransformLayer` rejects 2D hit testing with `nil`, because its true-3D hierarchy has no single 2D coordinate space; callers must provide scene-specific 3D picking |
| Transform layers | `CATransformLayer` preserves `zPosition` and `anchorPointZ`, orders blended children by projected center depth, and enables per-pixel WebGPU depth writes/tests only within the true-3D hierarchy. Homogeneous `z/w` normalization rejects non-finite coordinates, zero `w`, and division overflow with the exact sublayer index before depth clear or child draw; invalid geometry is never reinterpreted as depth zero. Independent transform groups reset depth without changing color; missing depth-clear resources and invalid or overflowing nesting transitions retain `CATransformDepthRenderFailure` instead of silently dropping the group, while empty groups avoid unnecessary GPU work. Transparent texels discard before depth write. A normal `CALayer` child subtree is captured with transparent clear and composited as one plane, while a nested `CATransformLayer` keeps the shared depth space. Capture extents recursively union transformed out-of-bounds descendants unless an ancestor clips with `masksToBounds`; oversized logical extents reduce resolution to WebGPU limits instead of cropping geometry. Layer-sized effect captures preserve rendered mask trees, partial mask alpha, mask-descendant filters, group opacity, root filters, and nested filters without falling back to viewport-sized composites. Premultiplied capture composition applies mask alpha once, and detached mask-tree revisions invalidate explicit raster caches after mutation. Shadow captures expand beyond layer bounds for offsets and blur, derive silhouettes from subtree alpha or `shadowPath`, and composite behind content on the same 3D plane. Native tests cover exact depth-state transitions, homogeneous normalization boundaries, and invalid reasons; browser readback verifies exact zero-`w` rejection plus intersecting planes, transparent cutouts, group isolation, flattening, nested 3D, overflow clipping, effects, masks, shadows, mutation, and cache reuse |
| Remaining boundary | Core Image transition types without executable WGSL remain unavailable. Complete QuartzCore parity is not claimed |

```swift
#if canImport(QuartzCore)
import QuartzCore      // Native platforms (iOS, macOS)
#else
import OpenCoreAnimation  // WASM/Web
#endif

// Same API works in both environments
let layer = CALayer()
layer.frame = CGRect(x: 0, y: 0, width: 100, height: 100)
layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
layer.cornerRadius = 10
```

## Features

### Layers

| Layer Type | Description |
|------------|-------------|
| `CALayer` | Base layer rendering for opacity, transform, shadow, border, mask, and filters |
| `CAShapeLayer` | Vector path rendering with strokes and complete-path non-zero/even-odd fills |
| `CAGradientLayer` | Axial, elliptical radial, and conic gradients with validated stops |
| `CATextLayer` | Text rendering with font and alignment |
| `CAReplicatorLayer` | Instance replication with cumulative transforms, color offsets, and animation delay |
| `CAScrollLayer` | Scrollable content with bounds.origin-based scrolling |
| `CATransformLayer` | 3D transform container (does not flatten sublayers) |
| `CATiledLayer` | Tiled content rendering with level-of-detail support |
| `CAEmitterLayer` | Particle systems with physics simulation |

### Animations

| Animation Type | Description |
|----------------|-------------|
| `CABasicAnimation` | Simple from/to/by animations |
| `CAKeyframeAnimation` | Multi-keyframe animations with path support |
| `CASpringAnimation` | Physics-based spring animations |
| `CAAnimationGroup` | Animation composition |
| `CATransition` | Layer transitions (fade, push, moveIn, reveal) |
| `CAValueFunction` | Transform animations from scalar values |

### Animation Features

- **Timing Functions**: Linear, ease-in, ease-out, ease-in-ease-out, custom cubic bezier
- **Path Animations**: Animate position along CGPath with auto-rotation
- **Value Functions**: Animate transforms using scalar values (rotateX/Y/Z, scaleX/Y/Z, translateX/Y/Z)
- **Spring Physics**: Configurable mass, stiffness, damping, and initial velocity

### Layer Properties

- **Visual**: backgroundColor, borderColor, borderWidth, cornerRadius, opacity, isHidden
- **Shadow**: shadowColor, shadowOpacity, shadowOffset, shadowRadius
- **Transform**: transform, sublayerTransform, anchorPoint, anchorPointZ, zPosition
- **Content**: contents (CGImage), contentsGravity, contentsRect, contentsCenter, contentsScale
- **Masking**: mask, masksToBounds, cornerRadius with maskedCorners
- **Filters**: CAFilter for blur and color effects

### Filters (CAFilter)

```swift
// Apply blur filter to layer
layer.filters = [CAFilter.blur(radius: 10)]

// Color adjustments
layer.filters = [
    CAFilter.brightness(0.2),
    CAFilter.contrast(1.5),
    CAFilter.saturation(0.8)
```

Supported filter types:
- `gaussianBlur` - Gaussian blur effect
- `brightness` - Brightness adjustment (-1 to 1)
- `contrast` - Contrast adjustment (0 to 4)
- `saturation` - Saturation adjustment (0 to 2)
- `colorInvert` - Color inversion
- `sepiaTone` - Sepia tone effect
- `vignette` - Vignette effect

### Core Components

- `CATransaction` - Implicit animation batching with nested transaction support
- `CADisplayLink` - Frame-synchronized callbacks using requestAnimationFrame
- `CATransform3D` - Full 3D transformation matrix support
- `CAMediaTiming` - Animation timing protocol

## Installation

Add OpenCoreAnimation to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/OpenCoreAnimation.git", branch: "main")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["OpenCoreAnimation"]
)
```

### Dependencies

- [OpenCoreGraphics](https://github.com/1amageek/OpenCoreGraphics) - CoreGraphics types for WASM
- [swift-webgpu](https://github.com/1amageek/swift-webgpu) - WebGPU bindings (WASM only)

## Usage

### Basic Layer Setup

```swift
import OpenCoreAnimation

// Create a layer hierarchy
let rootLayer = CALayer()
rootLayer.bounds = CGRect(x: 0, y: 0, width: 800, height: 600)
rootLayer.position = CGPoint(x: 400, y: 300)
rootLayer.backgroundColor = CGColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1.0)

// Add a sublayer
let boxLayer = CALayer()
boxLayer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)
boxLayer.position = CGPoint(x: 150, y: 150)
boxLayer.backgroundColor = CGColor(red: 0.2, green: 0.6, blue: 1.0, alpha: 1.0)
boxLayer.cornerRadius = 10
boxLayer.shadowOpacity = 0.5
boxLayer.shadowRadius = 10
boxLayer.shadowOffset = CGSize(width: 5, height: 5)
rootLayer.addSublayer(boxLayer)
```

### Image Content with Gravity

```swift
let imageLayer = CALayer()
imageLayer.bounds = CGRect(x: 0, y: 0, width: 200, height: 200)
imageLayer.contents = myCGImage

// Content positioning
imageLayer.contentsGravity = .resizeAspectFill
imageLayer.contentsRect = CGRect(x: 0, y: 0, width: 1, height: 1)  // Full image
imageLayer.contentsScale = 2.0  // Retina

// 9-patch scaling for stretchable images
imageLayer.contentsCenter = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
```

### Animations

```swift
// Basic animation
let positionAnim = CABasicAnimation(keyPath: "position")
positionAnim.fromValue = CGPoint(x: 100, y: 100)
positionAnim.toValue = CGPoint(x: 300, y: 300)
positionAnim.duration = 1.0
positionAnim.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
layer.add(positionAnim, forKey: "move")

// Spring animation
let springAnim = CASpringAnimation(keyPath: "transform.scale")
springAnim.fromValue = 1.0
springAnim.toValue = 1.5
springAnim.damping = 10
springAnim.stiffness = 100
springAnim.mass = 1
layer.add(springAnim, forKey: "spring")

// Keyframe animation with values
let keyframeAnim = CAKeyframeAnimation(keyPath: "position")
keyframeAnim.values = [
    CGPoint(x: 0, y: 0),
    CGPoint(x: 100, y: 50),
    CGPoint(x: 200, y: 0)
]
keyframeAnim.keyTimes = [0, 0.5, 1.0]
keyframeAnim.duration = 2.0
layer.add(keyframeAnim, forKey: "path")

// Path-based animation with auto-rotation
let pathAnim = CAKeyframeAnimation(keyPath: "position")
let path = CGMutablePath()
path.move(to: CGPoint(x: 50, y: 50))
path.addCurve(to: CGPoint(x: 350, y: 350),
              control1: CGPoint(x: 200, y: 50),
              control2: CGPoint(x: 200, y: 350))
pathAnim.path = path
pathAnim.rotationMode = .rotateAuto  // Rotate along path tangent
pathAnim.duration = 3.0
layer.add(pathAnim, forKey: "followPath")

// Value function animation (rotate around Y axis)
let rotateAnim = CABasicAnimation(keyPath: "transform")
rotateAnim.valueFunction = CAValueFunction(name: .rotateY)
rotateAnim.fromValue = 0
rotateAnim.toValue = CGFloat.pi * 2
rotateAnim.duration = 2.0
layer.add(rotateAnim, forKey: "rotate")

// Transition
let transition = CATransition()
transition.type = .push
transition.subtype = .fromRight
transition.duration = 0.5
layer.add(transition, forKey: "transition")
```

### Shape Layers

```swift
let shapeLayer = CAShapeLayer()
shapeLayer.bounds = CGRect(x: 0, y: 0, width: 100, height: 100)

// Create a path
let path = CGMutablePath()
path.addEllipse(in: CGRect(x: 0, y: 0, width: 100, height: 100))

shapeLayer.path = path
shapeLayer.fillColor = CGColor(red: 1, green: 0.5, blue: 0, alpha: 1)
shapeLayer.strokeColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
shapeLayer.lineWidth = 2
shapeLayer.strokeStart = 0
shapeLayer.strokeEnd = 1  // Animate this for drawing effect
```

### Gradient Layers

```swift
let gradientLayer = CAGradientLayer()
gradientLayer.bounds = CGRect(x: 0, y: 0, width: 150, height: 150)
gradientLayer.colors = [
    CGColor(red: 1.0, green: 0.4, blue: 0.4, alpha: 1.0),
    CGColor(red: 0.4, green: 1.0, blue: 0.4, alpha: 1.0)
]
gradientLayer.startPoint = CGPoint(x: 0, y: 0)
gradientLayer.endPoint = CGPoint(x: 1, y: 1)
gradientLayer.locations = [0.0, 1.0]
```

### Replicator Layers

```swift
let replicator = CAReplicatorLayer()
replicator.instanceCount = 10
replicator.instanceDelay = 0.1
replicator.instanceTransform = CATransform3DMakeRotation(.pi / 5, 0, 0, 1)
replicator.instanceRedOffset = -0.1
replicator.instanceGreenOffset = -0.1
replicator.instanceBlueOffset = 0.0
replicator.instanceAlphaOffset = -0.1

let dot = CALayer()
dot.bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
dot.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
dot.cornerRadius = 10
replicator.addSublayer(dot)
```

### WebGPU Rendering (WASM)

```swift
import JavaScriptKit
import OpenCoreAnimation

@main
struct MyApp {
    static func main() async throws {
        let document = JSObject.global.document
        let canvas = document.createElement("canvas")
        canvas.width = 800
        canvas.height = 600
        _ = document.body.appendChild(canvas)

        // Create layer hierarchy
        let rootLayer = CALayer()
        rootLayer.bounds = CGRect(x: 0, y: 0, width: 800, height: 600)
        rootLayer.position = CGPoint(x: 400, y: 300)
        // ... configure layers ...

        // Initialize and start animation engine
        let engine = CAAnimationEngine.shared
        try await engine.setCanvas(canvas.object!)
        engine.rootLayer = rootLayer
        engine.start()
    }
}
```

## Building

### Native (for testing)

```bash
swift build
perl -e 'alarm 30; exec @ARGV' -- \
  xcodebuild test -scheme OpenCoreAnimation -destination 'platform=macOS' \
  -only-testing:OpenCoreAnimationTests
```

### WASM

```bash
TOOLCHAINS=org.swift.64202607171a xcrun swift build \
  --swift-sdk swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-17-a_wasm
cd Tests/e2e && npm test
```

## Platform Strategy

| Platform | Rendering | Timing | Usage |
|----------|-----------|--------|-------|
| WASM/Web | WebGPU | `requestAnimationFrame` | **Production** |
| macOS/iOS | Metal offscreen verification | `Timer` | Testing only |

On native Apple platforms, use Apple's QuartzCore directly for production. OpenCoreAnimation's native implementations are for testing purposes only.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OpenCoreAnimation API                       │
│ (CALayer, CAAnimation, CADisplayLink, CARenderer - API)     │
├─────────────────────────────────────────────────────────────┤
│                  WebGPU Rendering Layer                      │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ CAWebGPURenderer│  │ CAAnimationEngine│                  │
│  │ (Layer drawing) │  │ (Timing/frames)  │                  │
│  └─────────────────┘  └──────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                     swift-webgpu                             │
│              (Type-safe WebGPU bindings)                     │
├─────────────────────────────────────────────────────────────┤
│                     JavaScriptKit                            │
│              (Swift-to-JavaScript bridge)                    │
├─────────────────────────────────────────────────────────────┤
│                   Browser WebGPU API                         │
└─────────────────────────────────────────────────────────────┘
```

## Rendering Features

The WebGPU renderer provides full support for:

- **Layer Hierarchy**: Proper sublayer rendering with transform inheritance
- **3D Transforms**: Full CATransform3D support with perspective
- **Shadows**: Gaussian blur shadows with configurable radius, offset, and color
- **Masks**: Layer masking via the `mask` property
- **Clipping**: `masksToBounds` and corner radius clipping
- **Edge Antialiasing**: Independently selectable left, right, bottom, and top edge coverage
- **Image Content**: CGImage rendering with contentsGravity, contentsRect, and contentsCenter
- **9-Patch Scaling**: Stretchable images via contentsCenter
- **Filters**: Blur and color adjustment filters
- **Blend Modes**: Standard alpha blending
- **Depth Testing**: Proper z-ordering for 3D scenes

## Examples

See the [Examples](Examples/) directory for complete working demos:

- **BasicAnimation** - Layer hierarchy, animations, gradients, and shapes

## Requirements

- Swift 6.4 development snapshot `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-17-a`
- For WASM: Browser with WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly)

## License

MIT License

## References

- [Core Animation Documentation](https://developer.apple.com/documentation/quartzcore)
- [Core Animation Programming Guide](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CoreAnimation_guide/Introduction/Introduction.html)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
