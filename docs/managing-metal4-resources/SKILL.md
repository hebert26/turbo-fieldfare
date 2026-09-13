---
name: managing-metal4-resources
description: ALWAYS use when working with Metal 4 resources — buffer / texture / heap creation, storage modes, `MTLResidencySet` management, heap aliasing, purgeable state, texture view pools, GPU lossless compression, and deferred destruction. Trigger for MTLBuffer, MTLTexture, MTLHeap, MTLResidencySet, addAllocation, requestResidency, useResidencySet, addResidencySet, MTLStorageMode*, MTLPurgeableState, MTLTextureViewPool, MTLResourceViewPool, newTextureViewWithPixelFormat:, copyFromBuffer:toTexture:, "GPU fault on resource", "non-resident resource", heap aliasing. Do NOT trigger for TBDR architecture or Metal 3 → Metal 4 namespace tables — use `translating-to-metal4-api`. Barrier visibility / `MTL4VisibilityOption*` / render-pass transition barriers — use `managing-metal4-synchronization`. `CAMetalLayer.residencySet` — use `presenting-metal-drawables`. `NS::SharedPtr` / autorelease pool / ARC bridging — use `managing-metal-cpp-lifetimes`.
---

# Metal 4 Resources

## Overview

Metal 4 requires explicit residency management via `MTLResidencySet`, removes managed storage mode, and uses GPU addresses for resource binding. Command buffers do NOT retain resources — you must manage lifetimes. All Apple Silicon uses unified memory (CPU and GPU share the same physical memory pool) and Tile-Based Deferred Rendering (TBDR).

## Ownership

**Owns:**
- Buffer / texture / heap creation, storage-mode selection
- General `MTLResidencySet` semantics — kernel/userspace, attachment levels, validation rules, deferred destruction
- Heap allocation and aliasing patterns
- Purgeable state, texture view pools, view reinterpretation
- GPU lossless compression rules
- Deferred destruction pattern (frame-delayed release queue)

**Doesn't own:**
- TBDR architecture → `translating-to-metal4-api/references/tbdr-architecture.md`
- Drawable-owned `CAMetalLayer.residencySet` (read-only specialization) → `presenting-metal-drawables`
- Barrier visibility options for aliased resources (`MTL4VisibilityOptions`) → `managing-metal4-synchronization`
- Render-pass transition barriers → `managing-metal4-synchronization`
- `NS::SharedPtr` / autorelease pool / ARC bridging → `managing-metal-cpp-lifetimes`
- Metal 3 → Metal 4 prefix and namespace mapping → `translating-to-metal4-api`

## References

**Read the relevant Metal SDK header before writing resource code** — the headers are the source of truth for property names, types, and method signatures.

- Metal SDK headers - `$(xcrun --show-sdk-path)/System/Library/Frameworks/Metal.framework/Headers/` — focus on `MTLResource.h`, `MTLResidencySet.h`, `MTLAllocation.h`, `MTLHeap.h`, `MTLTexture.h`, `MTLBuffer.h`
- Apple documentation - [Resource fundamentals](https://developer.apple.com/documentation/metal/resource-fundamentals)
- Apple documentation - [Memory heaps](https://developer.apple.com/documentation/metal/memory-heaps)

## Design Patterns

| Working on… | Read |
|---|---|
| Buffer / texture creation, storage modes (Shared / Private / Memoryless), upload via staging buffer | `references/storage-and-upload.md` |
| Residency sets (creation, attachment levels, validation rules), deferred destruction | `references/residency.md` |
| Heaps + aliasing, purgeable memory, texture view pool, view reinterpretation | `references/heaps-views.md` |
| Apple GPU TBDR architecture | `translating-to-metal4-api/references/tbdr-architecture.md` |

### Apple GPU: TBDR Architecture

Tile-Based Deferred Rendering shapes nearly every storage-mode and render-pass decision on Apple Silicon — `StorageModeMemoryless` for tile-only intermediates, `Clear`/`DontCare` load/store actions on transient attachments, in-tile MSAA, and programmable blending all derive from it. The canonical TBDR reference for the suite lives in `translating-to-metal4-api/references/tbdr-architecture.md` — read it before making storage-mode or render-pass decisions. For the cross-tile fragment-output sync constraint, see `managing-metal4-synchronization`.

## Anti-Patterns / Common Mistakes

### Correctness — silent failures and GPU faults

1. **Using `StorageModeManaged`** — Removed in Metal 4. Use Shared + explicit copies for CPU→GPU sync.
2. **Forgetting residency** — GPU faults on non-resident resources. Add to set + commit BEFORE use.
3. **Removing from residency set or releasing resources while GPU is in-flight** — Command buffers don't retain resources, and either operation can cause GPU faults. The implementation may evict a resource removed from the residency set even if not yet deallocated. Use deferred destruction (frame-delayed release queue) to ensure the GPU is done before removing from the set or releasing.
4. **Exceeding residency set limits** — Queue supports up to 32 sets, each command buffer supports up to 32 sets (independent limits). Suspend/resume chains share one 32-set limit across the entire chain.
5. **Using `useResource`/`useHeap` on Metal 4 encoders** — These don't exist. Use residency sets.
6. **Concurrent residency set modification without synchronization** — `addAllocation`/`removeAllocation` from multiple threads requires synchronization (e.g., `@synchronized` in ObjC, mutex in C++). Deferred `commit()` at queue submit time.
7. **Binding resources with mismatched type or format for shader expectations** — If a shader expects a different type or format than the resource was created with (e.g., `texture2d_array` but resource is `MTLTextureTypeCube`, or buffer data interpreted with wrong stride/alignment), the result is silent failure (black textures, wrong data, garbled geometry). For textures, create a view with the correct type/format. For buffers, ensure the binding metadata matches the shader's expectation. GPU frame captures reveal the mismatch directly.

### Performance — works but leaves performance on the table

8. **Using `StorageModeShared` for textures that should be Private** — Prevents GPU-optimal tiled memory format. Use Private with staging buffer uploads for GPU-only textures.
9. **Using `Load`/`Store` actions on transient attachments** — Incurs unnecessary bandwidth on TBDR. Use `Clear`/`DontCare` for intermediates that don't need to persist. Enable load/store validation in Metal debug layer to catch this visually.
10. **Repopulating the residency set every frame** — Adding resources to the residency set based on per-frame usage is wasteful. Keep all required resources in a residency set persistently and add/remove only as resources are created/destroyed.
11. **Splitting render passes unnecessarily** — Each split incurs store actions on the ending pass + load actions on the starting pass. Merge passes where data dependencies allow it; use programmable blending for single-pass deferred.


## Good Practices

### GPU Bandwidth Compression

Apple GPUs apply transparent bandwidth compression to textures in both `StorageModeShared` and `StorageModePrivate`. Compression reduces bandwidth on load/store actions and spatially coherent shader reads/samples — it does not reduce worst-case memory occupation. The ratio depends on spatial similarity of stored color values; more similar neighboring pixels compress better. `StorageModePrivate` may use more aggressive forms (including lossy variants) than `Shared` since the layout is fully GPU-controlled.

**When compression is unavailable or doesn't help:**
- `MTLTextureUsageShaderWrite` disables lossless compression on GPU families prior to Universal Texture Compression support (M5+). On M5+, shader write is compatible but highly scattered write patterns may reduce the benefit.
- `MTLTextureUsagePixelFormatView` disables lossless compression for most ordinary pixel formats.
- Block-compressed formats (BCn, ASTC) already contain explicitly compressed data and are not additionally compressed on access — the bandwidth reduction story doesn't apply here.
- Worst-case random data has a negative compression ratio and is a candidate for opting out of lossless compression.
- Primarily spatially scattered shader reads may reduce benefits — profile to determine net gain or loss.

The bandwidth-savings benefit is most material on uncompressed render targets and uncompressed sampled textures with usage flags that don't disable compression. For block-compressed assets and resources that opt out via usage flags, it's a non-factor.

**Guidance:** Set `StorageModePrivate` for GPU-only resources, use the minimum required usage flags, and only add `ShaderWrite` or `pixelFormatView` when the shader actually needs those capabilities.

### General Practices

- **Always use compressed formats (ASTC or BCn)** for offline textures — 4-8x size reduction
- **Use Private for render targets** — may benefit from more aggressive (including lossy) GPU compression than Shared
- **Avoid unnecessary usage flags:** `ShaderWrite` and `pixelFormatView` can disable lossless compression
- **Prefer 16-bit formats** (`RGBA16Float`) over 128-bit (`RGBA32Float`) — 4x sample throughput
- **Use `optimizeContentsForGPUAccess:`** after CPU updates to shared textures
- **Use `Depth16Unorm`** when 32-bit depth not needed
- **Align render target sizes to 8-pixel boundaries** for optimal tile coverage on Apple GPUs
- **Use purgeable state** for cached resources that can be regenerated
- **Trace the resource binding chain when debugging.** When a resource produces wrong results or silent failures, trace its full path: resource creation (format, type, storage mode) → descriptor binding (correct entry type, address, metadata) → shader binding slot → shader code (how the shader reads it). A break at any point in this chain produces silent corruption. GPU frame captures are the most efficient tool for verifying each link.

