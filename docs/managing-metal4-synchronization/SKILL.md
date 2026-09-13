---
name: managing-metal4-synchronization
description: Metal 4 explicit synchronization — barriers, fences, events, and cross-API state translation (from D3D12, Vulkan, OpenGL). Use when porting GPU synchronization, translating resource state transitions, or debugging barrier-related visual corruption.
---

# Metal 4 Synchronization

## Overview

Metal 4 requires fully explicit synchronization — there is no driver hazard tracking. All resource dependencies between encoders and passes must be expressed through barriers, fences, or events. This is typically the longest phase of a port — expect barrier and synchronization work to account for a significant portion of total porting effort.

The key insight: **all source API synchronization models reduce to Metal 4 primitives** — producer barriers (on outgoing encoder), consumer barriers (on incoming encoder), intra-pass barriers (within an encoder), fences (between specific encoders on the same queue), and events (cross-queue or CPU-GPU). The source API determines *when* synchronization is needed. Metal 4's primitives determine *how* to express it.

## References

**Read the relevant Metal 4 SDK header before writing synchronization code** — the headers are the source of truth for property names, types, and method signatures.

- Metal 4 SDK headers - `$(xcrun --show-sdk-path)/System/Library/Frameworks/Metal.framework/Headers/` — focus on `MTL4CommandEncoder.h` (barriers), `MTLEvent.h` (events), `MTLFence.h` (fences)
- Apple documentation - [Resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)

## Design Patterns

### Barrier Primitives

Metal 4 provides three barrier types on `MTL4CommandEncoder`. The APIs use two scope levels for stage masks:

- **Queue stages** (`afterQueueStages` / `beforeQueueStages`) — affect all encoders on the queue with matching stages, both past and future. Used in inter-pass (producer/consumer) barriers.
- **Encoder stages** (`afterEncoderStages` / `beforeEncoderStages`) — affect only the current encoder. Used in intra-pass barriers.

#### 1. Consumer Barrier (inter-pass, on incoming encoder)

```objc
[encoder barrierAfterQueueStages:afterStages
                    beforeStages:beforeStages
               visibilityOptions:visOptions];
```

Prior encoders' stages must complete before the current and future encoders' stages proceed. Emitted on the **incoming** encoder at the start of a pass.

#### 2. Producer Barrier (inter-pass, on outgoing encoder)

```objc
[encoder barrierAfterStages:afterStages
          beforeQueueStages:beforeQueueStages
          visibilityOptions:visOptions];
```

Current and prior encoders' stages must complete before subsequent encoders' stages proceed. Emitted on the **outgoing** encoder before ending it.

#### 3. Intra-Pass Barrier (within same encoder)

```objc
[encoder barrierAfterEncoderStages:afterStages
              beforeEncoderStages:beforeStages
                visibilityOptions:visOptions];
```

Within the same encoder. Used for read-write resource barriers and same-pass dependencies.

**TBDR constraint:** Cannot wait for `Fragment` or `Tile` in `afterEncoderStages` on a render encoder — tiles execute concurrently and fragment output is only in tile memory, not yet stored to device memory. Split into separate render passes + inter-pass barriers instead.

### Visibility Options

| Option | When to Use |
|---|---|
| `MTL4VisibilityOptionNone` | Execution barrier only, no cache flush. Read-to-read transitions. |
| `MTL4VisibilityOptionDevice` | Flush to GPU device memory coherence point. Most resource transitions (write→read, write→write). |
| `MTL4VisibilityOptionResourceAlias` | When multiple **virtual addresses** can simultaneously be backed by the same physical address. Canonical case: placement-sparse resources whose tiles map the same heap page through more than one sparse resource. May flush to system memory coherence point on some hardware — heavier than `Device`. **Do not use** for ordinary placement-heap aliasing where two resources share an offset range across non-overlapping lifetimes — `Device` is sufficient there. |

**Default to `MTL4VisibilityOptionDevice`** for any write→read or write→write transition. Use `None` only for pure execution ordering without data dependencies.

### Fences and Events

| Mechanism | Scope | API |
|---|---|---|
| `MTLFence` | Same queue, between encoders | `updateFence:afterEncoderStages:` / `waitForFence:beforeEncoderStages:` |
| `MTLEvent` | Cross-queue on same device | `signalEvent:value:` / `waitForEvent:value:` on queue |
| `MTLSharedEvent` | Cross-queue, cross-device, CPU-GPU | `signalEvent:value:` / `waitUntilSignaledValue:timeoutMS:` |

**Queue barriers vs fences:** Queue barriers affect all encoders with matching stages — they reason about encoder ordering within the queue broadly. Fences are point-to-point connections between specific encoders that don't affect anything else in the queue. Metal is allowed to reorder encoders as long as declared dependencies (barriers, fences) are maintained. Use fences when barriers over-synchronize work that could otherwise run concurrently, or when your pass dependency graph already has point-to-point dependency information that maps naturally to fence update/wait pairs.

### Deferred Barrier Pattern

The recommended implementation pattern for engines with D3D12-style state transitions:

1. **On state-transition call** — When the engine receives a D3D12-style state-transition request (`ResourceBarrier(Transition)` or the engine's equivalent), translate source/destination states to `MTLStages` using the state-to-stage mapping table. Accumulate (OR) stage masks on a per-resource pending-barrier record. Do not emit Metal barriers yet.

2. **Ending outgoing encoder** — Emit a **producer barrier** with the accumulated `afterStages`. Clear the producer flags.

3. **Starting incoming encoder** — Emit a **consumer barrier** with the accumulated `beforeStages`. Clear the consumer flags.

4. **Encoder type switch** (render → compute or vice versa) — Flush all pending barriers on the outgoing encoder before ending it.

5. **Consecutive read-write access** — When the same resource is written and then read or written again within the same encoder, emit an **immediate intra-pass barrier** (not deferred):

```objc
[encoder barrierAfterEncoderStages:MTLStageDispatch
              beforeEncoderStages:MTLStageDispatch
                visibilityOptions:MTL4VisibilityOptionDevice];
```

`MTL4VisibilityOptionDevice` is critical here — without it, writes may remain in cache.

### Render Pass Transition Barriers (TBDR)

Metal 4's explicit barrier model requires producer/consumer barriers between sequential render passes sharing the same attachment. Metal 3 handled this implicitly.

When a render pass uses `MTLLoadActionLoad` on an attachment, the previous render pass's fragment output must be visible. Emit:
- **Producer barrier** on the outgoing encoder: `afterStages:Fragment beforeQueueStages:Fragment visibilityOptions:Device`
- **Consumer barrier** on the incoming encoder: `afterQueueStages:Fragment beforeStages:Fragment visibilityOptions:Device`

Without these, TBDR tile memory may show stale data in random tiles.

### Resource State to Metal Stage Mapping

The D3D12 and Vulkan cross-API translation sections below contain complete state-to-stage mapping tables. For engines with custom state abstractions, discover the engine's state model during codebase discovery and map it to Metal stages using those tables as a guide.

**Where to integrate:** The source of resource state transitions varies by engine — it may be a render graph (automatic dependency tracking), explicit barrier calls from the application, or the engine's command recording layer. The agent must identify where the engine expresses resource transitions; this is where Metal 4 barrier translation must be integrated.

**Key principle:** The resource's previous usage determines the producer stage (`afterStages`). The resource's new usage determines the consumer stage (`beforeStages`). If a state is read-only, only a consumer barrier is needed. If write-only, only a producer barrier is needed. For "shader resource read," the consumer stage must match the actual consuming stage(s) — there is no single "shader resource" stage in Metal.

### Corner Cases

**Split barriers (D3D12 begin-only / end-only):**
D3D12 optimization for overlapping work. No direct Metal equivalent.
- Begin-only → record pending state, don't emit yet
- End-only → emit the actual barrier using the recorded state
- Both 0 → immediate (the deferred pattern handles this naturally)

**Queue ownership transfers (Vulkan concept):**
Vulkan concept. No-op on Metal — Metal queues share memory visibility by default.

**Initial / common / undefined state (`D3D12_RESOURCE_STATE_COMMON`, Vulkan `VK_IMAGE_LAYOUT_UNDEFINED`):** No barrier needed.

**Present state (`D3D12_RESOURCE_STATE_PRESENT`, Vulkan `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`):** No barrier — handled by `signalDrawable`.

**Multiple barriers per call:** A single state-transition call (D3D12 `ResourceBarrier`, or the engine's equivalent) typically receives an array. Process ALL barriers and OR the stage masks together before emitting.

**Compute ↔ render transitions:** Pending barriers MUST be flushed on the outgoing encoder before ending it when switching encoder types.

## Cross-API Translation

### D3D12 → Metal 4

| D3D12 Concept | Metal 4 Equivalent |
|---|---|
| `ResourceBarrier` (state transition) | Queue barriers (producer/consumer). Use D3D12 state mapping below. |
| `ResourceBarrier` (UAV) | Intra-pass barrier with `MTL4VisibilityOptionDevice` |
| Split barriers (`BEGIN_ONLY`/`END_ONLY`) | Record on begin, emit on end. No Metal equivalent. |
| `ID3D12Fence` | `MTLSharedEvent` for CPU-GPU, queue barriers for GPU-GPU |
| Command list (monolithic) | Metal 4 requires explicit encoder transitions. Flush barriers at encoder boundaries. |

**D3D12 resource state → Metal stage mapping:**

| D3D12 Resource State | Producer Stage | Consumer Stage |
|---|---|---|
| `D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER` | — | `Vertex` |
| `D3D12_RESOURCE_STATE_INDEX_BUFFER` | — | `Vertex` |
| `D3D12_RESOURCE_STATE_RENDER_TARGET` | `Fragment` | — |
| `D3D12_RESOURCE_STATE_UNORDERED_ACCESS` | `Dispatch` | `Dispatch` |
| `D3D12_RESOURCE_STATE_DEPTH_WRITE` | `Fragment` | — |
| `D3D12_RESOURCE_STATE_DEPTH_READ` | — | `Fragment` |
| `D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE` | — | `Vertex \| Dispatch` |
| `D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE` | — | `Fragment` |
| `D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT` | — | `Vertex \| Dispatch` |
| `D3D12_RESOURCE_STATE_COPY_DEST` | `Blit` | — |
| `D3D12_RESOURCE_STATE_COPY_SOURCE` | — | `Blit` |
| `D3D12_RESOURCE_STATE_PRESENT` | no barrier | no barrier |
| `D3D12_RESOURCE_STATE_COMMON` | no barrier | no barrier |
| `D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE` | `AccelerationStructure` | `AccelerationStructure` |

### Vulkan → Metal 4

| Vulkan Concept | Metal 4 Equivalent |
|---|---|
| `vkCmdPipelineBarrier` (src/dst stages) | Queue barriers. Use Vulkan stage mapping below. |
| Access masks (`SHADER_WRITE_BIT` etc.) | `MTL4VisibilityOptions`. Write+Read → `Device`. Read-to-read → `None`. |
| Subpass dependencies | Load/store actions handle tile memory on TBDR. Self-dependencies → intra-pass barrier or split passes. |
| `vkCmdPipelineBarrier` within render pass | Intra-pass barrier if the stages are valid on the current encoder type. If the barrier involves fragment-to-fragment dependencies across tiles (the TBDR Fragment constraint), split the render pass at that point into separate passes with inter-pass barriers. |
| Events (set/wait) | Producer/consumer barrier pair. |
| Semaphores | `MTLSharedEvent` for cross-queue. |
| Queue ownership transfers | No-op on Metal. |

**Vulkan pipeline stage → Metal stage mapping:**

| Vulkan Pipeline Stage | Metal Stage |
|---|---|
| `VK_PIPELINE_STAGE_VERTEX_INPUT_BIT` | `Vertex` |
| `VK_PIPELINE_STAGE_VERTEX_SHADER_BIT` | `Vertex` |
| `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT` | `Fragment` |
| `VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT` | `Fragment` |
| `VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT` | `Fragment` |
| `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT` | `Fragment` |
| `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` | `Dispatch` |
| `VK_PIPELINE_STAGE_TRANSFER_BIT` | `Blit` |
| `VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT` | `Vertex` |
| `VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT` | `Mesh` |
| `VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT` | `Object` |
| `VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR` | `AccelerationStructure` |
| `VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR` | `Dispatch` |
| `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` | All stages (use for broad barriers, then refine) |

### Metal 3 → Metal 4

| Metal 3 Concept | Metal 4 Equivalent |
|---|---|
| `MTLFence` (`updateFence:afterStages:`) | Still exists (`afterEncoderStages:`), but queue barriers are simpler for most inter-pass sync. Fences offer more precise control over specific encoder ordering. |
| `memoryBarrierWithScope:afterStages:beforeStages:` | `barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:` (TBDR Fragment constraint applies). |
| `memoryBarrierWithScope:` on compute | `barrierAfterEncoderStages:Dispatch beforeEncoderStages:Dispatch visibilityOptions:Device` |
| `MTLBlitCommandEncoder` (inherently serialized) | Blits on compute encoder need explicit `StageBlit` barriers if interleaved with dispatches. |
| Render pass load/store actions | Unchanged in Metal 4. |

### OpenGL → Metal 4

OpenGL is mostly implicit — the driver handles most dependencies. When porting from a GL-based engine, you must **add** explicit barriers wherever the original code relied on implicit ordering (render to texture → bind → draw). `glMemoryBarrier()` maps to intra-pass or queue barriers; `glTextureBarrier()` maps to split render passes with `Fragment → Fragment` inter-pass barriers.

## Anti-Patterns / Common Mistakes

1. **Emitting barriers on the wrong encoder.** Producer barriers go on the outgoing encoder. Consumer barriers go on the incoming encoder. Swapping them produces no synchronization.

2. **Using `Fragment` in `afterEncoderStages` on a render encoder.** Invalid on TBDR — tile memory hasn't flushed. Split into separate render passes with inter-pass barriers.

3. **Forgetting `MTL4VisibilityOptionDevice` on write→read transitions.** Without it, the barrier is execution-only — cached writes may not be visible to subsequent reads.

4. **Over-barriering with `StageAll`.** Correct but kills performance. Use the state-to-stage mapping table to emit the narrowest barriers.

5. **Not flushing barriers at encoder type transitions.** When switching from render to compute (or vice versa), all pending deferred barriers must be emitted on the outgoing encoder before ending it.

6. **Treating split barriers as real barriers.** Begin-only means "record, don't emit." The actual barrier is emitted at end-only. Emitting at begin-only wastes GPU time.

7. **Ignoring render pass transition barriers.** Metal 4 requires explicit barriers between sequential render passes sharing attachments. Metal 3 handled this implicitly. Without them, TBDR shows stale tile data.

8. **Splitting encoders when an intra-pass barrier or queue barrier would do.** Each new encoder flushes tile memory on TBDR. A UAV→UAV transition within compute, or a producer/consumer pair across passes that don't conflict on attachments, doesn't need an encoder split — and shouldn't have one.

## Debugging Strategies

**Symptoms of missing barriers:**
- Random visual corruption that changes each frame (TBDR tile coherency)
- Stale data visible in some tiles but not others
- GPU crashes or hangs (under-barriering)
- Correct rendering but poor performance (over-barriering)

**Diagnostic steps:**
1. For **queue-level barriers** (inter-pass), start with overly broad barriers (`StageAll` → `StageAll` with `Device` visibility) to confirm the issue is sync-related. For **intra-pass barriers**, use the broadest stages valid for the encoder type — the validation layer will report an error if you use an unsupported stage.
2. If broad barriers fix it, narrow them systematically — remove one barrier at a time to find the minimum required set
3. Check encoder transitions — are pending barriers being flushed when switching from render to compute (or vice versa)?
4. Check render pass transitions — do sequential passes sharing attachments have producer/consumer barriers?
5. Request a GPU frame capture from the user — examine resource state at each draw call

**Common root causes:**
- Missing render pass transition barriers (TBDR tile coherency — stale data in random tiles)
- Missing flush at encoder type switch (compute → render transition drops pending barriers)
- Read-write barrier missing `MTL4VisibilityOptionDevice` (writes remain in cache)
- Consumer barrier on wrong encoder (emitted on outgoing instead of incoming)

## Good Practices

1. **Minimize encoder count.** Every encoder boundary on TBDR flushes tile memory and breaks command-buffer batching. The other practices below all support this principle:
   - Use intra-pass barriers (`barrierAfterEncoderStages:beforeEncoderStages:`) when stages allow — they keep work inside the same encoder.
   - Merge consecutive render encoders that share compatible attachments — keeps tile memory hot.
   - Avoid unnecessary compute ↔ render encoder transitions — group work by encoder type when possible.

   Profile encoder counts, not just barrier counts, when optimizing.

2. **Start coarse, refine.** Begin with broad barriers to get correct rendering. Then narrow the stage masks using the mapping table. Profile to find unnecessary barriers.

3. **Accumulate, don't emit immediately.** The deferred pattern (accumulate stages on each state-transition call, emit at encoder transitions) naturally batches barriers and handles the producer/consumer split.

4. **Queue barriers vs fences.** Queue barriers affect all encoders with matching stages — simpler for most inter-pass sync. Fences target specific encoders (update/wait pairs) — use when you need precise ordering between particular encoders.

### Performance Considerations

The patterns above produce correct synchronization. The principles below describe optimizations to consider once correctness is established. Each is selective — adopt based on profiling, not by default.

- **Group compute-class operations.** Metal 4 routes blit, dispatch, and acceleration structure operations through compute encoders. When these are sequential and have no inter-op dependency, they can share a single encoder rather than each starting its own. A naive port that creates a fresh encoder per operation type pays setup cost and tile-memory flush per transition; a unified compute encoder amortizes that cost.

- **Reorder commands to enable encoder merging.** Commands with no explicit ordering dependency — two compute dispatches reading from disjoint resources, a blit that doesn't depend on a preceding draw — can be reordered to group operations by encoder type. This reduces render↔compute transitions. Most impactful in command-recording engines that have visibility into the dependency graph between submissions.

- **Track resource access ranges, not just point-in-time transitions.** The deferred barrier pattern accumulates stages between encoder boundaries. A more aggressive approach tracks each resource's access pattern as ranges (first-access, last-access) and computes the minimum barrier set for the whole workload. This eliminates barriers between compatible accesses (consecutive reads, for example) that point-in-time tracking would emit anyway. The two approaches are mutually exclusive within a single subsystem — pick one and apply consistently.

- **Use pipeline reflection to skip syncs for unused resources.** A pipeline state object's reflection identifies which resources it actually accesses. If a resource is in your barrier-tracking set but the bound PSO's shader doesn't read or write it, no barrier is needed for that resource at that PSO's draw/dispatch. Engines that bind descriptor sets containing more resources than each shader uses can avoid significant unnecessary syncs by querying PSO reflection at PSO creation time.

- **Use color attachment mapping to merge render passes.** Metal 4 supports logical-to-physical color attachment mapping on render pipelines. Render passes that target different subsets of the same attachments can then be merged into a single encoder, with each pipeline writing only to its declared subset. Without this, naive ports start separate encoders for each subset combination — paying TBDR tile flush costs that could be avoided.

These optimizations are most impactful for engines with high draw counts, GPU-driven rendering, or aggressive multi-pass techniques. For straightforward bring-up, the practices above are sufficient.
