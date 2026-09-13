# Heaps, aliasing, purgeable memory, and texture views

Detail on `MTLHeap` allocation and aliasing, purgeable state for cached resources, the lightweight `MTLTextureViewPool`, and reinterpretation of textures and buffers via views. The SKILL.md body covers the policy (use heaps for temporal aliasing or grouped lifetimes; views for shader-vs-resource type/format mismatches); load this file when writing the actual heap setup, view pool population, or `newTextureViewWithPixelFormat:` calls.

## Resource Heaps and Aliasing

Use `MTLHeap` to sub-allocate buffers and textures from a single memory block. Heaps are recommended in two cases: **(1)** temporal memory reuse — aliasing the same memory for different resources whose lifetimes don't overlap, avoiding repeated kernel allocations; **(2)** grouping closely related resources that share the same lifetime and usage patterns into a single allocation.

### Heap types

| Type | When to use | Placement |
|---|---|---|
| `MTLHeapTypeAutomatic` (default) | Temporary, write-often resources. Optimal GPU-specific layout. | Metal picks offsets |
| `MTLHeapTypePlacement` | Persistent, write-rarely resources where the app controls aliasing or fragmentation directly | App specifies `offset:` per resource |
| `MTLHeapTypeSparse` (legacy) | Pre-Metal-4 sparse heaps (`MTLResourceStateCommandEncoder` mapping) | Sparse tile assignment by encoder |

Metal 4 placement-sparse resources (sparse buffers and textures whose tiles map and unmap on demand) use a **placement** heap, not a sparse heap. See "Placement sparse resources" below.

Adding a heap to a residency set makes the **entire heap** resident — no need to add individual sub-allocated resources.

### Automatic heap (Metal picks offsets)

```objc
MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
heapDesc.size = heapSize;
heapDesc.storageMode = MTLStorageModePrivate;
heapDesc.type = MTLHeapTypeAutomatic;  // default
id<MTLHeap> heap = [device newHeapWithDescriptor:heapDesc];

// Allocate resources from the heap — Metal manages placement
id<MTLBuffer> buffer = [heap newBufferWithLength:size options:MTLResourceStorageModePrivate];
id<MTLTexture> texture = [heap newTextureWithDescriptor:texDesc];
```

Automatic heaps are an allocator: when you destroy a resource and create another, Metal may reuse the freed memory.

### Placement heap (app controls offsets)

The app computes offsets and is responsible for non-overlapping placement (or for treating overlap as intentional aliasing). Resource storage mode and CPU cache mode must match the heap's. Query size and alignment per resource via `heapBufferSizeAndAlignWithLength:options:` and `heapTextureSizeAndAlignWithDescriptor:`:

```objc
MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
heapDesc.size = heapSize;
heapDesc.storageMode = MTLStorageModePrivate;
heapDesc.type = MTLHeapTypePlacement;
id<MTLHeap> heap = [device newHeapWithDescriptor:heapDesc];

// Query alignment requirements before placing each resource
MTLSizeAndAlign bufLayout = [device heapBufferSizeAndAlignWithLength:size
                                                             options:MTLResourceStorageModePrivate];
NSUInteger bufferOffset = 0;  // assumes start of heap is suitably aligned for this resource
id<MTLBuffer> buffer = [heap newBufferWithLength:size
                                         options:MTLResourceStorageModePrivate
                                          offset:bufferOffset];

MTLSizeAndAlign texLayout = [device heapTextureSizeAndAlignWithDescriptor:texDesc];
// Align the next offset up to texLayout.align — bufferOffset + bufLayout.size is the earliest free byte
NSUInteger textureOffset = (bufferOffset + bufLayout.size + texLayout.align - 1) & ~(texLayout.align - 1);
id<MTLTexture> texture = [heap newTextureWithDescriptor:texDesc offset:textureOffset];
```

**Aliasing.** Placement at the same offset range as another resource implicitly aliases the underlying memory (per `newBufferWithLength:options:offset:` documentation). Use this for resources with non-overlapping lifetimes — e.g., SSAO and Depth of Field buffers used at different stages of the frame can share the same offset range.

Issue a `MTL4VisibilityOptionDevice` barrier between the writer of the outgoing resource and the reader/writer of the incoming resource — `Device` flushes writes to the GPU coherence point, which is sufficient when only one of the aliased VAs is logically live at a time. The heavier `MTL4VisibilityOptionResourceAlias` is reserved for the case where multiple virtual addresses can *simultaneously* be backed by the same physical address — the canonical case is placement-sparse resources whose tiles share heap pages (see "Placement sparse resources" below). For ordinary placement-heap aliasing, do not use `ResourceAlias` — it may flush to the system coherence point on some hardware and is more expensive than needed. See `managing-metal4-synchronization`.

### Placement sparse resources

A *placement sparse* texture or buffer has no backing memory at creation. The app maps tiles from a placement heap on demand via `MTL4CommandQueue` — useful for very large textures (sparse virtual textures, large terrain heightmaps) where only a working set is resident at any time.

Validation rules (enforced by the Metal debug layer):
- The texture or buffer must be created with a non-zero `placementSparsePageSize` (`MTLSparsePageSize16`, `MTLSparsePageSize64`, or `MTLSparsePageSize256`).
- The heap must be `MTLHeapTypePlacement` with `maxCompatiblePlacementSparsePageSize` ≥ the resource's page size.
- The `heap` parameter to `updateTextureMappings:heap:operations:count:` may be `nil` only when *all* operations in the call are `MTLSparseTextureMappingModeUnmap`.
- After updating mappings, issue a barrier against stage `MTLStageResourceState` before any work that consumes the newly-mapped region.
- **Every backing heap must be in a residency set** before the sparse resource is consumed by shaders or by any `MTL4CommandQueue` sparse operation (`updateTextureMappings:`, `updateBufferMappings:`, `copyTextureMappingsFromTexture:`, etc.). Both the sparse resource and the backing heap must be made resident.

```objc
// 1. Create a placement sparse texture (no backing memory yet)
MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:largeWidth
                                                                                 height:largeHeight
                                                                              mipmapped:YES];
texDesc.storageMode = MTLStorageModePrivate;
texDesc.usage = MTLTextureUsageShaderRead;
texDesc.placementSparsePageSize = MTLSparsePageSize16;
id<MTLTexture> sparseTexture = [device newTextureWithDescriptor:texDesc];

// 2. Create a placement heap with at least the texture's sparse page size
MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
heapDesc.size = backingHeapSize;
heapDesc.storageMode = MTLStorageModePrivate;
heapDesc.type = MTLHeapTypePlacement;
heapDesc.maxCompatiblePlacementSparsePageSize = MTLSparsePageSize16;
id<MTLHeap> sparseBackingHeap = [device newHeapWithDescriptor:heapDesc];

// 3. Map a region of the sparse texture to tiles in the heap
//    textureRegion is in tiles (not pixels); heapOffset is in tiles.
MTL4UpdateSparseTextureMappingOperation op = {
    .mode = MTLSparseTextureMappingModeMap,
    .textureRegion = MTLRegionMake2D(tileX, tileY, tileWidth, tileHeight),
    .textureLevel = mipLevel,
    .textureSlice = 0,
    .heapOffset = heapTileOffset,
};
[queue updateTextureMappings:sparseTexture
                        heap:sparseBackingHeap
                  operations:&op
                       count:1];

// 4. Issue a barrier against MTLStageResourceState before any subsequent work
//    that reads or writes the newly-mapped region. See `managing-metal4-synchronization`.
```

To unmap, set `mode = MTLSparseTextureMappingModeUnmap`; the `heapOffset` is ignored. The `heap` parameter to `updateTextureMappings:heap:operations:count:` may be `nil` if every operation in the call is an unmap.

Sparse buffers use the parallel API `updateBufferMappings:heap:operations:count:` with `MTL4UpdateSparseBufferMappingOperation` (carries `bufferRange` in tiles instead of `textureRegion`).

**Use `MTL4VisibilityOptionResourceAlias`** when more than one sparse resource maps the same heap tile and both are accessed across a barrier — this is the case the option exists for. For a single sparse resource updating its own mappings, `MTL4VisibilityOptionDevice` on the post-update barrier is sufficient.

## Purgeable Memory

Mark idle cached resources as volatile to allow the system to reclaim memory under pressure:

```objc
// Mark idle cache as volatile (doesn't count toward memory footprint)
[texture setPurgeableState:MTLPurgeableStateVolatile];

// Before reuse: set back to non-volatile and check if data was purged
MTLPurgeableState state = [texture setPurgeableState:MTLPurgeableStateNonVolatile];
if (state == MTLPurgeableStateEmpty) {
    // Data was purged — regenerate texture
}
```

Useful for texture caches, streaming resources, or other data that can be regenerated on demand.

## Texture View Pool

Lightweight views with contiguous slot indices in a single `MTLTextureViewPool`. Each slot returns an `MTLResourceID`, not a full `MTLTexture` instance — cheaper to create and manage than `newTextureViewWithPixelFormat:` views, and the standard form for bindless / argument-table binding.

```objc
MTLResourceViewPoolDescriptor* desc = [[MTLResourceViewPoolDescriptor alloc] init];
desc.resourceViewCount = capacity;
id<MTLTextureViewPool> pool = [device newTextureViewPoolWithDescriptor:desc error:&error];

// Default view of an existing texture (no reinterpretation)
MTLResourceID viewID = [pool setTextureView:texture atIndex:N];

// Reinterpreted view via MTLTextureViewDescriptor — different format, type, level/slice range
MTLTextureViewDescriptor* viewDesc = [[MTLTextureViewDescriptor alloc] init];
viewDesc.pixelFormat = desiredFormat;
viewDesc.textureType = desiredType;
viewDesc.levelRange  = NSMakeRange(baseMip, mipCount);
viewDesc.sliceRange  = NSMakeRange(baseSlice, sliceCount);
MTLResourceID reinterpretID = [pool setTextureView:texture descriptor:viewDesc atIndex:M];

// Typed buffer view (raw buffer interpreted as a texture)
MTLResourceID bufferViewID = [pool setTextureViewFromBuffer:buffer
                                                 descriptor:texDesc
                                                     offset:byteOffset
                                                bytesPerRow:bytesPerRow
                                                    atIndex:K];
```

`MTLResourceID` is a single `uint64_t _impl`. Slot N's resource ID is `pool.baseResourceID._impl + N` — slot indices are contiguous, and arithmetic on the base is the standard way to compute IDs for bulk binding (e.g., populating a bindless array). The `MTLResourceID` returned by `setTextureView:…atIndex:` is equivalent.

Useful for format reinterpretation between formats with the same per-pixel bit width (e.g., sRGB ↔ linear pairing such as `BGRA8Unorm_sRGB` ↔ `BGRA8Unorm`, or `R32Float` viewed as `R32Uint` for integer atomics on float data), type reinterpretation (e.g., Cube → 2DArray for compute / shader sampling), bindless texture arrays, and typed buffer views. For Metal Shader Converter descriptor heap setup and `IRDescriptorTableEntry` encoding, see `integrating-metal-shaderconverter-shaders`.

## Resource Interpretation and Views

Shaders may expect to access a resource differently than how it was created — different type, format, or subrange. When the shader's expectation doesn't match the resource's original configuration, you need to provide a compatible view or binding.

**Textures: choose the view path based on how the view is consumed.** Both paths share the parent's allocation — neither allocates new memory.

| Consumer | Use | Reason |
|---|---|---|
| Argument table / descriptor heap (bindless) — `[table setTexture:resourceID atIndex:]` | `MTLTextureViewPool` `setTextureView:descriptor:atIndex:` | The argument table API takes `MTLResourceID`; pool views are cheaper to create and manage. |
| Render-pass attachment (`MTLRenderPassAttachmentDescriptor.texture` / `.resolveTexture`) | `[texture newTextureViewWithPixelFormat:textureType:levels:slices:]` | The attachment property is typed `id<MTLTexture>`; pool views return `MTLResourceID` and can't bind here. |
| `setTexture:` on a blit / compute encoder, MetalFX scaler inputs, anywhere the API takes `id<MTLTexture>` | `newTextureViewWithPixelFormat:` | Same reason — the API requires a full `MTLTexture` object. |

Shape of the view (per `MTLTextureViewDescriptor` properties or the equivalent `newTextureViewWithPixelFormat:textureType:levels:slices:` arguments):

| Shader Expects | Resource Created As | View Configuration |
|---|---|---|
| `texture2d_array` (per-face write) | `MTLTextureTypeCube` | `textureType = MTLTextureType2DArray`, `sliceRange = NSMakeRange(0, 6 * arrayLength)` |
| `texture2d` (single slice) | `MTLTextureType2DArray` | `textureType = MTLTextureType2D`, `sliceRange = NSMakeRange(N, 1)` |
| Different pixel format | Original format | `pixelFormat = desiredFormat` (must be compatible) |
| Single mip level (read-write) | Full mip chain | `levelRange = NSMakeRange(N, 1)` |

**Example: Cube-to-2DArray for per-face writes (IBL precomputation, cube map filtering).** Per-face writes are typically encoded as a render pass with the 2DArray view as an attachment — `MTLRenderPassAttachmentDescriptor.texture` is `id<MTLTexture>`, so this case requires `newTextureViewWithPixelFormat:`:

```objc
id<MTLTexture> cubeTexture = ...;  // MTLTextureTypeCube
id<MTLTexture> arrayView = [cubeTexture newTextureViewWithPixelFormat:cubeTexture.pixelFormat
                                                          textureType:MTLTextureType2DArray
                                                               levels:NSMakeRange(0, cubeTexture.mipmapLevelCount)
                                                               slices:NSMakeRange(0, 6 * cubeTexture.arrayLength)];
// Bind arrayView as the render-pass color attachment for the writing pass;
// bind the original cubeTexture for subsequent sampling.
```

If the per-face write is instead encoded on a compute encoder that consumes the view through an argument table, populate a slot in an `MTLTextureViewPool` and bind the returned `MTLResourceID` — same view shape, lighter binding path.

**Buffers:** Metal buffers are untyped (raw bytes). Interpretation happens at the binding or shader level — no explicit view API needed for raw access. The same buffer can be bound as structured data in one shader and as raw bytes in another. However, ensure:
- The GPU address and length you bind covers the correct range for the shader's interpretation.
- Alignment requirements match the shader's expected type (e.g., `float4` requires 16-byte alignment).
- When an engine's descriptor system stores typed metadata (format, stride) alongside the GPU address, the metadata must match the shader's expectation, not the buffer's original creation parameters.

For *typed* buffer views (a buffer accessed as a texture in a shader through an argument table), use `setTextureViewFromBuffer:descriptor:offset:bytesPerRow:atIndex:` on the texture view pool. If a typed buffer view needs to bind into an API that takes `id<MTLTexture>`, allocate a buffer-backed texture instead via `[device newTextureWithDescriptor:]` with `MTLTextureTypeTextureBuffer`.

**General principle:** When debugging black textures, wrong data, or silent shader failures, check whether the resource's type/format matches what the shader expects. GPU frame captures reveal these mismatches directly — the bound resource type will differ from the shader parameter type.

**Key points for texture views:**
- Views share the parent's allocation.
- After writing through a view, the original texture sees the same data.
- Create views during resource setup, not per-frame.
- Pool views are referenced by `MTLResourceID`; legacy `newTextureViewWithPixelFormat:` views are full `MTLTexture` instances with their own ResourceID distinct from the parent.
