---
name: translating-to-metal4-api
description: ALWAYS use as the Metal 4 API reference and cross-API translation map when porting graphics code to Metal 4 — translating Metal 3, D3D12, or Vulkan calls, looking up any MTL4 / Metal 4 API or renamed type, finding the Metal 4 equivalent of a source-API call, or learning what the redesign removed or changed. Covers the Metal 3 → Metal 4 namespace/type table, removed/changed APIs (ParallelRenderEncoder, set*Bytes), Apple GPU TBDR (tile-based deferred rendering) architecture, and the D3D12 / Vulkan / Metal 3 → Metal 4 translation tables. Do NOT trigger for buffer/texture/heap/residency/storage-mode management — use `managing-metal4-resources`; barriers, fences, or events — use `managing-metal4-synchronization`; pipeline state creation, MTL4Compiler, or reflection — use `creating-metal4-shader-pipelines`; CAMetalLayer drawable presentation — use `presenting-metal-drawables`.
---

# Metal 4 API Reference

## Overview

Metal 4 is a ground-up API redesign — not a minor version bump. It introduces the `MTL4` prefix (ObjC protocols/classes) / `MTL4::` namespace (metal-cpp). All Metal 4 objects are created from `MTLDevice` / `MTL::Device`. Key removals: blit encoder, parallel render encoder, set*Bytes, managed storage mode, per-encoder resource binding, addCompletedHandler.

Metal 4 requires macOS 26+, iOS 26+. Feature detection:

```objc
// ObjC
if ([device supportsFamily:MTLGPUFamilyMetal4]) { /* Metal 4 path */ }
```
```cpp
// metal-cpp
if (device->supportsFamily(MTL::GPUFamilyMetal4)) { /* Metal 4 path */ }
```

Metal 4 can coexist with Metal 3 queues in the same app for incremental adoption.

## Apple GPU Architecture: TBDR

All Apple Silicon GPUs use Tile-Based Deferred Rendering (TBDR), which is fundamentally different from IMR (Immediate-Mode Rendering) used by desktop NVIDIA/AMD GPUs. TBDR affects render pass structure, storage modes, synchronization, and performance decisions throughout the Metal API surface. **Read [tbdr-architecture.md](references/tbdr-architecture.md) before working with any Metal code** — it covers how TBDR works, the bandwidth cost model, efficient and costly patterns, hard constraints, and how to reason about whether an engine's existing rendering pipeline will work well on Apple Silicon.

## References

**Read the relevant Metal SDK header before writing translation code** — the headers are the source of truth for property names, types, and method signatures.

- Metal SDK headers - `$(xcrun --show-sdk-path)/System/Library/Frameworks/Metal.framework/Headers/MTL4*.h`
- metal-cpp headers (if using C++) - Metal 4 C++ wrappers live in the `MTL4/` subdirectory (e.g., `MTL4CommandBuffer.hpp`, `MTL4ComputeCommandEncoder.hpp`)
- Apple documentation - [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api?language=objc)

## Design Patterns

### Namespace Translation Table

| Metal 3 | Metal 4 | Notes |
|---|---|---|
| `MTLCommandQueue` | `MTL4CommandQueue` | Created via `[device newMTL4CommandQueue]` |
| `MTLCommandBuffer` (transient, from queue) | `MTL4CommandBuffer` (reusable, from device) | Explicit begin/end lifecycle with allocator |
| `MTLBlitCommandEncoder` | **Removed** → `MTL4ComputeCommandEncoder` | All blit ops on unified compute encoder |
| `MTLRenderCommandEncoder` | `MTL4RenderCommandEncoder` | Argument table binding model |
| `MTLComputeCommandEncoder` | `MTL4ComputeCommandEncoder` | Unified: blit + dispatch + accel struct |
| `MTLAccelerationStructureCommandEncoder` | **Removed** → `MTL4ComputeCommandEncoder` | Merged into compute |
| `MTLParallelRenderCommandEncoder` | **Removed** | Use suspend/resume or multiple cmd buffers |
| N/A | `MTL4CommandAllocator` | Manages command buffer memory, pooled per frame |
| N/A | `MTL4ArgumentTable` / `MTL4ArgumentTableDescriptor` | Replaces all per-encoder resource binding |
| N/A | `MTL4Compiler` / `MTL4CompilerDescriptor` | Explicit pipeline compilation |
| N/A | `MTL4CompilerTask` / `MTL4CompilerTaskOptions` | Async compilation tasks |
| `MTLRenderPassDescriptor` | `MTL4RenderPassDescriptor` | Render pass with explicit width/height |
| `MTLRenderPipelineDescriptor` | `MTL4RenderPipelineDescriptor` | Uses function descriptors, not MTLFunction |
| `MTLComputePipelineDescriptor` | `MTL4ComputePipelineDescriptor` | Uses function descriptors |
| `MTLMeshRenderPipelineDescriptor` | `MTL4MeshRenderPipelineDescriptor` | Mesh shader pipelines |
| `MTLTileRenderPipelineDescriptor` | `MTL4TileRenderPipelineDescriptor` | Tile shader pipelines |
| N/A | `MTL4LibraryDescriptor` | Library from source via compiler |
| N/A | `MTL4LibraryFunctionDescriptor` | Describes shader function via library + name |
| N/A | `MTL4CounterHeap` / `MTL4CounterHeapDescriptor` | Replaces `MTLCounterSampleBuffer` |
| N/A | `MTL4CommitFeedback` / `MTL4CommitOptions` | GPU timing and error info (replaces addCompletedHandler) |
| N/A | `MTL4PipelineDataSetSerializer` | Pipeline caching |
| N/A | `MTL4Archive` | Binary pipeline cache |
| N/A | `MTL4AccelerationStructureDescriptor` | RT acceleration structure descriptors |
| N/A | `MTL4MachineLearningCommandEncoder` | CoreML inference in command buffers |
| `MTLResidencySet` | `MTLResidencySet` | Explicit residency management (same namespace) |
| N/A | `MTLTextureViewPool` | Lightweight texture views with contiguous resource IDs |

### Device Factory Methods

All Metal 4 objects are created from the device:

```objc
// ObjC
id<MTL4CommandQueue>     queue     = [device newMTL4CommandQueue];
id<MTL4CommandBuffer>    cmd       = [device newCommandBuffer];
id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
id<MTL4Compiler>         compiler  = [device newCompilerWithDescriptor:desc error:&error];
id<MTL4ArgumentTable>    table     = [device newArgumentTableWithDescriptor:desc error:&error];
id<MTL4CounterHeap>      heap      = [device newCounterHeapWithDescriptor:desc error:&error];
id<MTLResidencySet>      resSet    = [device newResidencySetWithDescriptor:desc error:&error];
```

```cpp
// metal-cpp
MTL4::CommandQueue*     queue     = device->newMTL4CommandQueue();
MTL4::CommandBuffer*    cmd       = device->newCommandBuffer();
MTL4::CommandAllocator* allocator = device->newCommandAllocator();
MTL4::Compiler*         compiler  = device->newCompiler(compDesc, &error);
MTL4::ArgumentTable*    table     = device->newArgumentTable(tableDesc, &error);
MTL4::CounterHeap*      heap      = device->newCounterHeap(heapDesc, &error);
MTL::ResidencySet*      resSet    = device->newResidencySet(resDesc, &error);
```

### APIs Removed from Metal 3

- **`set*Bytes`** — All variants (`setVertexBytes`, `setFragmentBytes`, `setComputeBytes`). All data must go through buffers. See [Replacing `set*Bytes`](#replacing-setbytes-transient-buffer-allocation) below for the recommended pattern.
- **`setVertexBuffer`/`setFragmentTexture`/etc.** — All per-encoder resource binding. Use `ArgumentTable`.
- **`StorageModeManaged`** — Removed. Switch buffers to `Shared` or `Private`, then drop the associated `didModifyRange:` calls (they only apply to managed-storage buffers and become invalid).
- **`addCompletedHandler`/`addScheduledHandler`** — Use `SharedEvent` signaling or `CommitOptions` feedback handlers.
- **`useResource`/`useHeap`** — Use `ResidencySet` instead.
- **`BlitCommandEncoder`** — All blit ops (copy, fill, resolve, mipmap generation, optimization) move to `ComputeCommandEncoder`.
- **`AccelerationStructureCommandEncoder`** — Acceleration structure builds move to `ComputeCommandEncoder`.
- **`ParallelRenderCommandEncoder`** — Use suspend/resume or multiple cmd buffers with batch commit.
- **`StoreActionOptions`** — Removed. This controlled custom sample position hints for MSAA depth storage on non-Apple Silicon hardware. Apple Silicon does not need this — ignore it. Store actions themselves (`Store`, `DontCare`, etc.) are unchanged and critical.
- **Tessellation on render encoder** — Use mesh shaders (`MeshRenderPipelineDescriptor`).

### Replacing `set*Bytes`: Transient Buffer Allocation

With `set*Bytes` removed, applications need to manage their own transient buffer allocations for small, short-lived data like per-draw constants and uniforms. The recommended approach is a per-frame bump allocator backed by a `Shared` storage mode buffer.

The transient buffer allocator hands out suballocations from a single large buffer. Each suballocation returns a GPU address that can be bound directly via the argument table. At the start of each frame (after the GPU has finished consuming the previous frame's data), reset the offset to zero.

```cpp
// metal-cpp
class TransientBufferAllocator {
    MTL::Buffer* _buffer;
    NS::UInteger _capacity;
    NS::UInteger _offset = 0;
public:
    TransientBufferAllocator(MTL::Device* device, NS::UInteger capacity)
        : _capacity(capacity) {
        _buffer = device->newBuffer(capacity, MTL::ResourceStorageModeShared);
    }

    // Write data and return a GPU address suitable for argument table binding
    MTL::GPUAddress write(const void* data, NS::UInteger size) {
        // Align to 16 bytes (suitable for constant data packed as float4)
        _offset = (_offset + 15) & ~15;
        assert(_offset + size <= _capacity &&
               "TransientBufferAllocator overflow — increase capacity");
        memcpy(static_cast<uint8_t*>(_buffer->contents()) + _offset, data, size);
        MTL::GPUAddress addr = _buffer->gpuAddress() + _offset;
        _offset += size;
        return addr;
    }

    // Reset at frame start after GPU is done with this frame's data
    void reset() { _offset = 0; }

    // The backing buffer — add to your ResidencySet
    MTL::Buffer* buffer() const { return _buffer; }
};
```

```cpp
// Usage per draw
PerDrawConstants constants = { mvpMatrix, normalMatrix, materialID };
MTL::GPUAddress addr = frameAllocator->write(&constants, sizeof(constants));
table->setAddress(addr, kConstantsBindingIndex);
```

**Key points:**
- Size the buffer for worst-case per-frame usage — overflowing silently corrupts data
- One transient buffer allocator per in-flight frame to avoid CPU/GPU data races
- Alignment depends on how shaders consume the data — check your shader's expected layout. 16-byte alignment is a safe default for constant data packed as `float4`
- Add the backing buffer to your `ResidencySet`
- For engines with many draws, pre-calculate total transient allocation size per frame to right-size the buffer

### Changed APIs

**Fences — scope changed:**
- Metal 3: `MTLFence` works across multiple command queues on the same device
- Metal 4: `MTLFence` only works within the **same command queue**
- For cross-queue sync, use `MTLEvent` when the dependency stays on the GPU. Use `MTLSharedEvent` only when the CPU also signals or waits — it carries CPU-side machinery that's wasted on pure GPU↔GPU sync.

**Events — moved to queue:**
- Metal 3: Events signaled/waited on command buffer (`encodeSignalEvent:value:`, `encodeWaitForEvent:value:`)
- Metal 4: Events signaled/waited on the queue (`[queue signalEvent:value:]`, `[queue waitForEvent:value:]`)

**Command buffer completion:**
- Metal 3: `[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) { ... }]`
- Metal 4: `MTL4CommitOptions` with feedback handler, or `MTLSharedEvent` signaling

**Render pass descriptor:**
- Metal 4 uses `MTL4RenderPassDescriptor` which supports explicit `renderTargetWidth` and `renderTargetHeight` (required only when no attachments are set — otherwise inferred)

```objc
// ObjC
MTL4RenderPassDescriptor* rpDesc = [[MTL4RenderPassDescriptor alloc] init];
rpDesc.renderTargetWidth = width;    // required if no attachments — otherwise inferred
rpDesc.renderTargetHeight = height;  // required if no attachments — otherwise inferred

// Color attachment — cache accessor to avoid repeated ARC traffic
MTLRenderPassColorAttachmentDescriptor* ca0 = rpDesc.colorAttachments[0];
ca0.texture = colorTexture;
ca0.loadAction = MTLLoadActionClear;
ca0.storeAction = MTLStoreActionStore;
ca0.clearColor = MTLClearColorMake(0, 0, 0, 1);

// Depth attachment
rpDesc.depthAttachment.texture = depthTexture;
rpDesc.depthAttachment.loadAction = MTLLoadActionClear;
rpDesc.depthAttachment.storeAction = MTLStoreActionDontCare;  // memoryless depth
rpDesc.depthAttachment.clearDepth = 1.0;

// Create encoder
id<MTL4RenderCommandEncoder> encoder = [cmd renderCommandEncoderWithDescriptor:rpDesc];
```

### Command Buffer Lifecycle

```objc
// ObjC — One-time setup
id<MTL4CommandBuffer> cmd = [device newCommandBuffer];       // reusable
id<MTL4CommandAllocator> alloc = [device newCommandAllocator]; // per in-flight frame

// Per-frame
[alloc reset];
[cmd beginCommandBufferWithAllocator:alloc];
// encode...
[cmd endCommandBuffer];
id<MTL4CommandBuffer> cmds[] = {cmd};
[queue commit:cmds count:1];
```

```cpp
// metal-cpp
MTL4::CommandBuffer* cmd = device->newCommandBuffer();
MTL4::CommandAllocator* alloc = device->newCommandAllocator();

alloc->reset();
cmd->beginCommandBuffer(alloc);
// encode...
cmd->endCommandBuffer();
MTL4::CommandBuffer* cmds[] = {cmd};
queue->commit(cmds, 1);
```

**Key points:**
- Command buffers are reusable — created once from `device`, not per-frame from `queue`
- Allocators manage backing memory — one per in-flight frame, reset at frame start
- `beginCommandBuffer` / `endCommandBuffer` is an explicit lifecycle (Metal 3 had no equivalent)
- Queue submission is batched via `commit:count:`
- Command buffers do NOT retain resources — you manage lifetimes

**Validation rules:**
- `beginCommandBuffer` with nil allocator → error
- Calling `beginCommandBuffer` twice on an open command buffer → error
- Using same allocator for two simultaneously open command buffers → error
- `endCommandBuffer` before `beginCommandBuffer` → error
- Allocator reset before command buffer is committed → error
- Empty command buffers (no encoders) can be committed successfully

### Resource Binding (Argument Tables)

```objc
// ObjC — Setup
MTL4ArgumentTableDescriptor* atDesc = [[MTL4ArgumentTableDescriptor alloc] init];
atDesc.maxBufferBindCount = <count>;       // must be > highest [[buffer(N)]] index used
atDesc.maxTextureBindCount = <count>;      // must be > highest [[texture(N)]] index used
atDesc.maxSamplerStateBindCount = <count>; // must be > highest [[sampler(N)]] index used
atDesc.supportAttributeStrides = YES;
atDesc.initializeBindings = YES;          // initialize unbound slots to null (deterministic reads)
id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:atDesc error:&error];
```

**Sizing:** Each count sets the maximum bind index you can use in that space — a shader using `[[buffer(20)]]` requires `maxBufferBindCount` of at least 21. The appropriate size depends on the set of shaders that will be used with the table during encoding. Since argument tables are reusable across encoders and frames, size them for the broadest set of shaders they'll encounter. There is a device maximum for each bind space — one option is to create a single argument table per encoding thread at the maximum size, which works with any shader.

**Render stage binding:** Render encoders accept shaders in multiple stages (vertex & fragment, or object & mesh & fragment), and each stage has independent bind spaces. You can bind the same argument table instance to all stages, or bind distinct argument tables per stage:

```objc
// Bind resources (all bound resources must be in a residency set)
[table setAddress:[buffer gpuAddress] atIndex:index];                    // buffer
[table setAddress:[buffer gpuAddress] stride:stride atIndex:index];      // buffer with stride
[table setTexture:[texture gpuResourceID] atIndex:index];                // texture
[table setSamplerState:[sampler gpuResourceID] atIndex:index];           // sampler

// Assign to encoder
[renderEncoder setArgumentTable:table atStages:MTLStageVertex | MTLStageFragment];
[computeEncoder setArgumentTable:table];  // no stage param
```

```cpp
// metal-cpp
MTL4::ArgumentTableDescriptor* desc = MTL4::ArgumentTableDescriptor::alloc()->init();
desc->setMaxBufferBindCount(count);       // must be > highest [[buffer(N)]] index used
desc->setMaxTextureBindCount(count);      // must be > highest [[texture(N)]] index used
desc->setSupportAttributeStrides(true);
desc->setInitializeBindings(true);        // initialize unbound slots to null (deterministic reads)
MTL4::ArgumentTable* table = device->newArgumentTable(desc, &error);

table->setAddress(buffer->gpuAddress(), index);
table->setTexture(texture->gpuResourceID(), index);
table->setSamplerState(sampler->gpuResourceID(), index);

// Render encoder
renderEncoder->setArgumentTable(table, MTL::RenderStageVertex | MTL::RenderStageFragment);

// Compute encoder
computeEncoder->setArgumentTable(table);
```

**Binding model — three spaces:**

Shaders declare bindings in three spaces: `[[buffer(N)]]`, `[[texture(N)]]`, `[[sampler(N)]]`. The argument table stores GPU references into three corresponding internal tables. Each space accepts specific reference types:

| Space | Accepts | Shader parameter type |
|---|---|---|
| Buffer | GPU address (`gpuAddress()`) | `constant T&`, `constant T*`, `device T&`, `device T*` — actual buffer bindings |
| Buffer | Resource ID (`gpuResourceID()`) | Acceleration structures, tensors, and other non-buffer objects bound to `[[buffer(N)]]` slots |
| Texture | Resource ID (`gpuResourceID()`) | All texture types |
| Sampler | Resource ID (`gpuResourceID()`) | All sampler states |

The fundamental distinction: **GPU addresses support pointer arithmetic** — you can offset into different regions of the same `MTLBuffer` on both CPU and GPU (e.g., `buffer->gpuAddress() + offset`). **Resource IDs identify whole objects** — you bind the entire object, and any sub-indexing happens through the typed shader code (e.g., tensor element access, plane indexing).

**Rule of thumb:** If the shader parameter is a pointer or reference to a buffer type, use `gpuAddress()`. For everything else, use `gpuResourceID()`.

**Key points:**
- Metal snapshots the table at each draw/dispatch call — you can mutate bindings between draws without recreating the table
- After the initial `setArgumentTable`, subsequent `setAddress`/`setTexture`/`setSamplerState` calls on the table are visible at draw time (the table is a live mutable reference)

**Validation rules:**
- Drawing without `setArgumentTable` when PSO requires bindings → error: "No argument table set for [stage] stage"
- Table too small for required bindings → error: "Argument table only supports N [type] bindings"
- Binding index not populated → error: "[type] binding at index N was never set"
- Null buffer address (`setAddress(0, index)`) → error: "Buffer binding at index N cannot be null"

### Draw Calls (GPU Address for Index/Indirect)

```objc
// ObjC — Indexed draw (index buffer must be in a residency set)
[encoder drawIndexedPrimitives:type
                    indexCount:indexCount
                     indexType:indexType
                   indexBuffer:[indexBuffer gpuAddress] + offset
             indexBufferLength:indexBufferLength];

// Indirect draw (indirect buffer must be in a residency set)
[encoder drawPrimitives:type indirectBuffer:[indirectBuffer gpuAddress]];
```

```cpp
// metal-cpp
encoder->drawIndexedPrimitives(type, indexCount, indexType,
    indexBuffer->gpuAddress() + offset, indexBufferLength);

encoder->drawPrimitives(type, indirectBuffer->gpuAddress());
```

**Important:** `indexBufferLength` must be the **full accessible range** from the fetch point, not just `indexCount * indexSize`. Metal 4 uses this for bounds-checking on vertex index values. See Anti-Patterns #9.

### Compute Dispatch

```objc
// ObjC — Direct dispatch
[computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                threadsPerThreadgroup:threadsPerThreadgroup];
[computeEncoder dispatchThreads:threadsPerGrid
           threadsPerThreadgroup:threadsPerThreadgroup];

// Indirect dispatch — GPU address
[computeEncoder dispatchThreadgroups:[indirectBuffer gpuAddress]
                threadsPerThreadgroup:threadsPerThreadgroup];
```

```cpp
// metal-cpp
computeEncoder->dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup);
computeEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);

// Indirect
computeEncoder->dispatchThreadgroups(indirectBuffer->gpuAddress(), threadsPerThreadgroup);
```

**Cross-API equivalents:**
- D3D12: `ID3D12GraphicsCommandList::Dispatch` / `ExecuteIndirect`
- Vulkan: `vkCmdDispatch` / `vkCmdDispatchIndirect`

### Blit Operations (Compute Encoder)

Metal 4 removes `BlitCommandEncoder`. All blit operations go through the unified `ComputeCommandEncoder`:

```objc
// ObjC
id<MTL4ComputeCommandEncoder> enc = [cmd computeCommandEncoder];
[enc copyFromBuffer:src sourceOffset:srcOff toBuffer:dst destinationOffset:dstOff size:size];
[enc copyFromBuffer:src sourceOffset:srcOff sourceBytesPerRow:bpr sourceBytesPerImage:bpi
          sourceSize:srcSize toTexture:tex destinationSlice:slice destinationLevel:level
   destinationOrigin:origin];
[enc generateMipmapsForTexture:texture];
[enc fillBuffer:buffer range:range value:value];
[enc endEncoding];
```

Additional operations on unified compute encoder:
- `optimizeContentsForCPUAccess:` / `optimizeContentsForGPUAccess:`
- `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`
- `optimizeIndirectCommandBuffer:withRange:`
- All acceleration structure operations (build, refit, compact, copy)

When interleaving blit and dispatch operations on the same compute encoder, explicit barriers between `StageBlit` and `StageDispatch` are needed if they share resources.

### Synchronization Quick Reference

| Scope | Mechanism | API |
|---|---|---|
| Within pass | Intrapass barrier | `barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:` |
| Between passes (coarse) | Queue barrier (producer) | `barrierAfterStages:beforeQueueStages:visibilityOptions:` |
| Between passes (coarse) | Queue barrier (consumer) | `barrierAfterQueueStages:beforeStages:visibilityOptions:` |
| Between passes (precise) | Fence | `updateFence:afterEncoderStages:` / `waitForFence:beforeEncoderStages:` |
| Cross-queue | Event | `[queue signalEvent:value:]` / `[queue waitForEvent:value:]` |
| Cross-device/CPU | SharedEvent | `[queue signalEvent:value:]` / `[event waitUntilSignaledValue:timeoutMS:]` |

**Always choose narrowest scope.** Queue barriers affect all encoders with matching stages; fences target specific encoders for more precise ordering. For full synchronization details, see the `managing-metal4-synchronization` skill.

### Drawable Presentation

Brief reference — for full presentation and frame pacing details, see the `presenting-metal-drawables` skill.

```objc
// ObjC — Metal 4 presentation flow
id<CAMetalDrawable> drawable = [layer nextDrawable];
[queue waitForDrawable:drawable];   // GPU-side wait for drawable availability
[queue commit:cmds count:count];    // submit GPU work
[queue signalDrawable:drawable];    // signal when GPU is done
[drawable present];                 // present to screen
```

```cpp
// metal-cpp
CA::MetalDrawable* drawable = layer->nextDrawable();
queue->wait(drawable);
queue->commit(cmds, count);
queue->signalDrawable(drawable);
drawable->present();
```

### GPU Queries and Device Capabilities

For timestamp queries, occlusion queries, pipeline statistics, and device capability querying, see [gpu-queries-and-capabilities.md](references/gpu-queries-and-capabilities.md). Covers:
- Timestamp queries with `MTL4CounterHeap` (create, write, resolve, convert to nanoseconds)
- Occlusion queries with visibility result buffers
- Pipeline statistics (not available on Metal — use Xcode profiler)
- Device capability querying (`supportsFamily:`, max buffer length, unified memory, timestamp support)
- Cross-API equivalents for D3D12 and Vulkan query types

### Suspend/Resume (Replaces ParallelRenderEncoder)

```objc
// ObjC
id<MTL4RenderCommandEncoder> enc0 = [cmd0 renderCommandEncoderWithDescriptor:rpDesc
                                                                     options:MTL4RenderEncoderOptionSuspending];
// encode draws...
[enc0 endEncoding];

id<MTL4RenderCommandEncoder> enc1 = [cmd1 renderCommandEncoderWithDescriptor:rpDesc
                                                                     options:MTL4RenderEncoderOptionResuming];
// encode more draws...
[enc1 endEncoding];

// Must commit all in order as single batch
id<MTL4CommandBuffer> cmds[] = {cmd0, cmd1};
[queue commit:cmds count:2];
```

**Rules:**
- Suspended pass MUST be immediately resumed in the next command buffer in the commit batch
- Cannot interleave other encoders between suspend and resume
- Resume without preceding suspend is an error
- Suspended and resumed render pass descriptors must match (render targets, width, height, sample counts)
- After a suspending encoder, no new encoders can be created in the same command buffer
- The total number of unique residency sets across all command buffers committed together, plus the command queue's residency sets, may not exceed 32 — otherwise the commit fails


## Cross-API Translation

For detailed D3D12→Metal 4, Vulkan→Metal 4, and Metal 3→Metal 4 translation tables, see [cross-api-translation.md](references/cross-api-translation.md). Covers concept-by-concept mappings for command infrastructure, resource binding, synchronization, pipeline states, and presentation.

## Anti-Patterns / Common Mistakes

1. **Using `set*Bytes` or per-encoder binding** — These don't exist in Metal 4. Use argument tables (see Argument Tables section above).
2. **Creating command buffers from queue** — Create from `device`, not queue.
3. **Forgetting `beginCommandBuffer`/`endCommandBuffer`** — Explicit lifecycle required. Missing either causes validation errors.
4. **Using fences across queues** — Metal 4 fences are same-queue only. Use events for cross-queue.
5. **Not managing resource lifetimes** — Command buffers don't retain resources. Use deferred destruction.
6. **Using `StorageModeManaged`** — Removed. Use Shared or Private.
7. **Calling `useResource`/`useHeap` on encoders** — Use ResidencySet instead.
8. **Passing `Buffer*` to draw calls** — Metal 4 uses `GPUAddress` for index/indirect buffers.
9. **Passing consumed range instead of accessible range for buffer lengths** — Metal 4 address-based APIs (e.g., `drawIndexedPrimitives:indexBufferLength:`) expect the full valid range the GPU may access from the given address, not just the bytes the current operation reads. Getting this wrong causes silent data clamping, not crashes or validation errors.
10. **Using same allocator for two open command buffers** — One allocator per simultaneously open command buffer. Reuse after commit + reset.
11. **Misaligned index buffer addresses.** Metal requires index buffer addresses to be aligned to the index type size (2 bytes for UInt16, 4 bytes for UInt32). Engines that pack multiple meshes into a shared geometry buffer can produce misaligned offsets — an odd number of UInt16 indices shifts subsequent meshes by 2 bytes, breaking 4-byte alignment assumptions. The symptom is degenerate or corrupted geometry with no validation error. Pad index data to 4-byte boundaries when sharing buffers. For general buffer alignment, query `MTLDevice` properties: `minimumLinearTextureAlignmentForPixelFormat:` for linear textures, `minimumTextureBufferAlignmentForPixelFormat:` for texture buffer views. See `MTL4RenderCommandEncoder.h` for the authoritative index buffer alignment specification.

## Debugging Strategies

### Validation

For Metal validation setup (API validation, shader validation, load/store-action diagnostics), see `using-metal-validation` or `man MetalValidation`.

### Metal HUD

Set these before launching the app (before any Metal device is created):

```
MTL_HUD_ENABLED=1                          # visual overlay: FPS, GPU time, memory, present mode
MTL_HUD_LOG_ENABLED=1                      # log per-frame stats to console (requires HUD enabled)
MTL_HUD_LOG_SHADER_ENABLED=1               # log shader compilation activities (requires HUD enabled)
MTL_HUD_ENCODER_TIMING_ENABLED=1           # per-encoder GPU time tracking (vertex/fragment/compute)
MTL_HUD_SHOW_VALUE_RANGE=1                 # show min/max/avg over 1200 frames
```

The HUD overlay shows: Metal device, resolution, present mode (direct vs composited), memory usage, Game Mode status, FPS, GPU time, and frame interval chart.

With logging enabled, the HUD writes per-frame statistics to the console once per second — the agent can parse this output to diagnose frame rate, GPU time, and memory issues without user interaction.

With shader compiler logging, the HUD emits signposts for each compiled shader with compilation time and cache status — useful for detecting runtime compilation hitches.

**Reference:** Apple documentation - [Monitoring your Metal app's graphics performance](https://developer.apple.com/documentation/xcode/monitoring-your-metal-apps-graphics-performance)

### What Validation Catches

- **Missing bindings:** Reports exact binding index and type: "Buffer binding at index N was never set"
- **Wrong API usage:** Reports removed API calls and suggests Metal 4 equivalents
- **Command buffer lifecycle errors:** Catches double-begin, missing begin/end, allocator reuse
- **Resource lifetime issues:** Reports access to released resources
- **Out-of-bounds GPU access:** Shader validation catches buffer overflows, nil texture reads
- **Non-resident resources:** Shader validation detects access to resources not in a residency set
- **Incorrect load/store actions:** Visual validation makes bad actions immediately obvious (fuchsia/checkerboard)

### Diagnostic References

- Apple documentation - [Validating your app's Metal API usage](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-api-usage)
- `man MetalValidation` on macOS for full environment variable reference

## Good Practices

- **Check the SDK headers directly** when unsure about an API. The Metal 4 headers (`MTL4*.h`) are well-documented with comments.
- **Use incremental adoption when porting from Metal 3** — Metal 3 and Metal 4 queues can coexist in the same app, allowing subsystems to be ported individually.
- **When porting from Metal 3, Metal 3 PSOs work on Metal 4 encoders** (and vice versa) — pipeline creation can be ported incrementally.
- **Ensure all Metal API struct members are sourced from the engine, not hardcoded.** Every value passed to Metal (pixel formats, sample counts, load/store actions, blend factors, vertex formats, etc.) should trace back to the engine's data — not be a hardcoded constant. If the engine provides a value, use it. If it doesn't, discuss with the user rather than guessing a default. Hardcoded values that happen to work for one sample will silently break on others.

### Performance Considerations

A naive 1:1 translation from D3D12, Vulkan, or Metal 3 to Metal 4 produces correct code but creates more encoders than necessary, leaving significant performance on the table. The cross-API translation reference (`references/cross-api-translation.md`) carries inline `Perf:` hints on rows where the naive mapping has a meaningful optimization.

For the full set of optimization principles — encoder count minimization, grouping compute-class operations, command reordering, color attachment mapping, resource access range tracking, and pipeline reflection-driven barrier elision — see `managing-metal4-synchronization` Performance Considerations. Most performance optimization work in a Metal 4 port is concentrated there because encoder boundaries are fundamentally a synchronization concern.

## Companion Skills

- **Resource management skill** (`managing-metal4-resources`) — for residency sets, storage modes, descriptor heaps
- **Synchronization skill** (`managing-metal4-synchronization`) — for barriers, fences, events, cross-API sync translation
- **Pipeline skill** (`creating-metal4-shader-pipelines`) — for shader compilation (MTL4Compiler), metallib loading, shader reflection, and pipeline state creation
- **Presentation skill** (`presenting-metal-drawables`) — for drawable lifecycle, frame pacing, vsync, CAMetalLayer
- **Metal shader converter skill** (`integrating-metal-shaderconverter-shaders`) — for binding model, argument buffers, descriptor tables
