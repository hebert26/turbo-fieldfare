# Storage modes, buffer creation, and texture upload

Detail on storage-mode selection, buffer creation patterns, texture descriptors, and uploading data into private resources via the compute encoder. The SKILL.md body covers the policy (which mode for which access pattern); load this file when writing the actual creation and upload code.

## Storage Mode Decision

| Access Pattern | Mode | Notes |
|---|---|---|
| CPU read/write (uniforms, dynamic VBs) | `StorageModeShared` | Zero-copy on Apple Silicon (unified memory). Textures still get GPU bandwidth compression similar to Private. |
| GPU-only (textures, static meshes, RTs) | `StorageModePrivate` | Best perf for GPU-only access. May use more aggressive GPU compression (including lossy variants) than Shared. |
| Tile-only intermediates (depth, MSAA, G-buf) | `StorageModeMemoryless` | Exists only in tile memory. 4K depth: 32MB → ~0. |
| ~~CPU+GPU sync~~ | ~~`StorageModeManaged`~~ | **Removed in Metal 4.** Remove all `didModifyRange:` calls and never use StorageModeManaged. |

Note: Metal 4 is Apple-Silicon-only. If the engine still ships a Metal 3 backend for Intel-based Macs (x86_64 CPU, integrated Intel or discrete AMD GPU), that backend keeps using Managed + `didModifyRange:` — don't strip those calls from non-Metal-4 code paths.

## Buffer Creation and Upload

```objc
// ObjC — Shared buffer (CPU-accessible)
id<MTLBuffer> buffer = [device newBufferWithLength:size
                                           options:MTLResourceStorageModeShared];
void* data = [buffer contents];  // direct CPU access

// Private buffer (GPU-only) — requires staging upload
id<MTLBuffer> privateBuffer = [device newBufferWithLength:size
                                                  options:MTLResourceStorageModePrivate];

// Upload to private buffer via staging buffer + compute encoder
id<MTLBuffer> stagingBuffer = [device newBufferWithLength:size
                                                  options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
memcpy([stagingBuffer contents], sourceData, size);

id<MTL4ComputeCommandEncoder> enc = [cmd computeCommandEncoder];
[enc copyFromBuffer:stagingBuffer sourceOffset:0
           toBuffer:privateBuffer destinationOffset:0 size:size];
[enc endEncoding];
```

**Note:** All Metal 4 copy operations live on `MTL4ComputeCommandEncoder` — see `translating-to-metal4-api` for the encoder-naming change. Buffers can also be allocated from an `MTLHeap` for aliasing — see `references/heaps-views.md`.

**`CPUCacheModeWriteCombined`** is a CPU mapping flag that avoids polluting the CPU cache when writing contiguous data. Use it only when the CPU will not read from the resource — ensure only `memcpy` to `.contents` (buffers) or `replaceRegion` (textures) operations are used. If it's unclear whether the CPU might read back, omitting `WriteCombined` is a safe default.

## Texture Creation

```objc
MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
texDesc.width = width;
texDesc.height = height;
texDesc.depth = depth;                          // 1 for 2D textures
texDesc.mipmapLevelCount = mipLevels;           // 1 if no mipmaps
texDesc.arrayLength = arraySize;                // 1 for non-array textures
texDesc.pixelFormat = pixelFormat;              // from engine's format mapping
texDesc.textureType = MTLTextureType2D;         // 2D, 2DArray, 3D, Cube, etc.
texDesc.storageMode = MTLStorageModePrivate;    // GPU-only (most textures)
texDesc.usage = MTLTextureUsageShaderRead;      // add ShaderWrite for read-write access, RenderTarget for RTs

// For render targets
texDesc.usage |= MTLTextureUsageRenderTarget;

// For tile-only intermediates (depth, MSAA — zero system memory)
texDesc.storageMode = MTLStorageModeMemoryless;

id<MTLTexture> texture = [device newTextureWithDescriptor:texDesc];
```

**Key considerations:**
- `storageMode` — Private for most GPU textures, Memoryless for tile-only intermediates, Shared only if CPU needs to read/write.
- `usage` — set the minimum required flags. Extra flags (especially `ShaderWrite` and `pixelFormatView`) can disable lossless compression — see Good Practices > GPU Bandwidth Compression in `SKILL.md`.
- `textureType` — must match shader expectations (2D, 2DArray, Cube, etc.). For mismatches, see `references/heaps-views.md` (Resource Interpretation and Views).
- Textures can also be allocated from an `MTLHeap` for aliasing — see `references/heaps-views.md`.

## Texture Upload (Buffer to Texture)

Upload texture data from a staging buffer to a private texture via compute encoder:

```objc
// ObjC — Copy staging buffer data into a private texture
id<MTL4ComputeCommandEncoder> enc = [cmd computeCommandEncoder];
[enc copyFromBuffer:stagingBuffer
       sourceOffset:srcOffset
  sourceBytesPerRow:bytesPerRow           // row pitch in bytes
sourceBytesPerImage:bytesPerImage         // slice pitch (for 3D/array textures)
         sourceSize:MTLSizeMake(width, height, depth)
          toTexture:texture
   destinationSlice:arrayLayer
   destinationLevel:mipLevel
  destinationOrigin:MTLOriginMake(0, 0, 0)];
// Repeat the copy for each (mip level, array slice / cube face) pair the texture has.
// Cube textures use slices 0–5 for +X, -X, +Y, -Y, +Z, -Z; cube arrays use 6 * arrayLength slices.
[enc endEncoding];
```

**Parameters:**
- `sourceBytesPerRow` — row pitch, not just `width * bytesPerPixel` (may include padding from the engine's resource loader).
- `sourceBytesPerImage` — slice pitch for 3D textures or array layers. For single 2D uploads, use `bytesPerRow * height`.
- `sourceSize` — dimensions of the mip level being uploaded (compute from base texture size: `max(1, baseWidth >> mipLevel)`).
- `destinationSlice` / `destinationLevel` — for array layers and mip levels respectively.
