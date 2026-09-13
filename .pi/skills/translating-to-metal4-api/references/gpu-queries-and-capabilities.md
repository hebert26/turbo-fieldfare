### GPU Queries

Metal 4 uses `MTL4CounterHeap` for GPU queries (replaces Metal 3's `MTLCounterSampleBuffer`). Different query types use different heap types and patterns.

#### Timestamp Queries

Used to measure GPU execution time of specific work spans.

```objc
// ObjC — Create timestamp counter heap
MTL4CounterHeapDescriptor* desc = [[MTL4CounterHeapDescriptor alloc] init];
desc.type = MTL4CounterHeapTypeTimestamp;
desc.count = queryCount * 2;  // begin + end pairs
id<MTL4CounterHeap> heap = [device newCounterHeapWithDescriptor:desc error:&error];

// Write begin timestamp (before work)
[renderEncoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                  afterStage:MTLRenderStageFragment
                                    intoHeap:heap
                                     atIndex:beginIndex];

// Write end timestamp (after work)
[renderEncoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                  afterStage:MTLRenderStageFragment
                                    intoHeap:heap
                                     atIndex:endIndex];

// Write from command buffer level (not tied to a specific encoder)
[cmd writeTimestampIntoHeap:heap atIndex:index];

// Resolve on CPU
NSData* data = [heap resolveCounterRange:NSMakeRange(beginIndex, 2)];
// Parse MTL4TimestampHeapEntry structs from data — each has a .timestamp field

// Convert to nanoseconds using mach_timebase_info
mach_timebase_info_data_t timebase;
mach_timebase_info(&timebase);
double nanos = (endTimestamp - beginTimestamp) * timebase.numer / timebase.denom;

// Invalidate entries for reuse
[heap invalidateCounterRange:NSMakeRange(beginIndex, 2)];
```

**Granularity options:**
- `MTL4TimestampGranularityPrecise` — higher precision timestamp (still constrained by TBDR architecture), may have small GPU performance cost
- `MTL4TimestampGranularityRelaxed` — approximate timestamp, no performance cost

**Cross-API equivalents:**
- D3D12: `ID3D12QueryHeap` with `D3D12_QUERY_HEAP_TYPE_TIMESTAMP` + `ResolveQueryData`
- Vulkan: `VkQueryPool` with `VK_QUERY_TYPE_TIMESTAMP` + `vkGetQueryPoolResults`

#### Occlusion Queries

Metal uses **visibility result buffers** for occlusion queries, not counter heaps. This is unchanged from Metal 3.

```objc
// ObjC — Create visibility result buffer
id<MTLBuffer> visBuffer = [device newBufferWithLength:queryCount * sizeof(uint64_t)
                                              options:MTLResourceStorageModeShared];

// Set on render pass descriptor
renderPassDesc.visibilityResultBuffer = visBuffer;

// In render encoder — set mode before drawing occluder
[encoder setVisibilityResultMode:MTLVisibilityResultModeCounting offset:queryIndex * sizeof(uint64_t)];
// draw...
[encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];

// Read results after GPU completion
uint64_t* results = (uint64_t*)[visBuffer contents];
uint64_t samplesPassed = results[queryIndex];  // 0 = fully occluded
```

**Cross-API equivalents:**
- D3D12: `ID3D12QueryHeap` with `D3D12_QUERY_HEAP_TYPE_OCCLUSION`
- Vulkan: `VkQueryPool` with `VK_QUERY_TYPE_OCCLUSION`

#### Pipeline Statistics Queries

Metal does not have a direct equivalent of D3D12/Vulkan pipeline statistics queries (`D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS`, `VK_QUERY_TYPE_PIPELINE_STATISTICS`). If the engine uses these, they should be disabled or stubbed for Metal.

GPU performance counters are available through Xcode's GPU profiler and the Metal System Trace instrument, but not as runtime queries.

### Device Capabilities

Query device capabilities to determine feature support and hardware limits. The agent should check these during discovery and early milestones to inform implementation decisions.

```objc
// ObjC — Feature family detection
BOOL supportsMetal4 = [device supportsFamily:MTLGPUFamilyMetal4];
BOOL supportsApple9  = [device supportsFamily:MTLGPUFamilyApple9];  // ray tracing, mesh shaders

// Device properties
NSString* name              = device.name;
uint64_t  maxBufferLength   = device.maxBufferLength;
BOOL      hasUnifiedMemory  = device.hasUnifiedMemory;  // always YES on Apple Silicon
NSUInteger maxThreadsPerTG  = device.maxThreadsPerThreadgroup.width;

// Argument table limits (check SDK headers for exact property names)
// Max buffer bindings, texture bindings, sampler bindings

// Timestamp counter heaps are supported on all Metal 4 devices —
// no separate probe needed beyond supportsMetal4.
```

**Key capabilities to query:**
- `supportsFamily:` — Metal 4 support, GPU family (Apple7, Apple8, Apple9+)
- `maxBufferLength` — maximum single buffer allocation size
- `hasUnifiedMemory` — always YES on Apple Silicon (shared CPU/GPU memory)
- `maxThreadsPerThreadgroup` — maximum compute threadgroup size
- `readWriteTextureSupport` — tier of read-write texture support
- Ray tracing support — hardware RT requires Apple9+, software RT available from Apple6+

**Cross-API equivalents:**
- D3D12: `ID3D12Device::CheckFeatureSupport` with various `D3D12_FEATURE` queries
- Vulkan: `vkGetPhysicalDeviceProperties`, `vkGetPhysicalDeviceFeatures`, `vkGetPhysicalDeviceProperties2`
