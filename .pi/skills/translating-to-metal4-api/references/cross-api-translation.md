## Cross-API Translation

**A note on performance.** The mappings below produce a working Metal 4 port. They are not always performance-optimal — Metal 4 offers consolidations (encoder collapsing, tile-based deferred rendering optimizations, indirect command buffers for GPU-driven workloads, async compute on dedicated queues) that a naive 1:1 translation will miss. A few rows below carry inline `Perf:` hints flagging the most common gaps. Treat the table as the bring-up reference and revisit performance separately once correctness is established.

**A note on residency.** Metal 4 requires all resources accessed at draw or dispatch time — including those bound through argument tables — to be in a residency set attached to the queue or command buffer. This is a behavior change from Metal 3 (which auto-tracked encoder-bound resources) and a new explicit requirement for ports from D3D12 and Vulkan. See `managing-metal4-resources` for the residency set model.

### D3D12 → Metal 4

#### Command infrastructure

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `ID3D12CommandQueue` | `MTL4CommandQueue` | |
| `ID3D12CommandAllocator` | `MTL4CommandAllocator` | Reset per frame, same concept |
| `ID3D12GraphicsCommandList` | `MTL4CommandBuffer` + encoders | Metal splits command list into typed encoders |
| `ID3D12Fence` | CPU↔GPU sync (CPU waits, completion callbacks): `MTLSharedEvent`. GPU-only cross-queue sync: `MTLEvent`. Same-queue sync: queue barriers | *Perf: choose the narrowest scope — `MTLSharedEvent` carries CPU-side machinery you don't need for GPU-only fences. `MTLEvent` is sufficient when no CPU readback or callback is required.* |

#### Resource creation

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `CreateCommittedResource` (buffer) | `[device newBufferWithLength:options:]` | |
| `CreateCommittedResource` (texture) | `[device newTextureWithDescriptor:]` | |
| `CreatePlacedResource` (buffer) | `[heap newBufferWithLength:options:offset:]` | Sub-allocated from `MTLHeap` |
| `CreatePlacedResource` (texture) | `[heap newTextureWithDescriptor:offset:]` | Sub-allocated from `MTLHeap` |
| `CreateHeap` | `[device newHeapWithDescriptor:]` | |
| `CreateConstantBufferView` / `CreateShaderResourceView` (buffer) / `CreateUnorderedAccessView` (buffer) | No view object — descriptor table or argument table holds GPU address | Size in metadata for typed views |
| `CreateShaderResourceView` (texture) / `CreateUnorderedAccessView` (texture) | Whole texture: no view. Subresource view: `[pool setTextureView:texture descriptor:viewDesc atIndex:]` on an `MTLTextureViewPool` for argument-table / descriptor-heap binding (returns `MTLResourceID`). When the view must bind into an API that takes `id<MTLTexture>` (e.g., a render-pass attachment), use `[texture newTextureViewWithPixelFormat:textureType:levels:slices:]`. | |
| `CreateSampler` | `[device newSamplerStateWithDescriptor:]` | |

#### Pipeline state

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `ID3D12PipelineState` | `MTLRenderPipelineState` or `MTLComputePipelineState` | Created via `MTL4Compiler` or legacy path |
| `ID3D12RootSignature` | Argument table layout | Metal shader converter can translate root signatures |
| `SetPipelineState` | `[encoder setRenderPipelineState:]` / `[encoder setComputePipelineState:]` | |
| `SetGraphicsRootSignature` / `SetComputeRootSignature` | Bind argument table via `[encoder setArgumentTable:]` | Layout is implicit in the argument table |

#### Descriptor binding

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `ID3D12DescriptorHeap` | `MTLTextureViewPool` (bindless texture indexing with contiguous resource IDs), `MTLBuffer` with descriptor entries, or `MTL4ArgumentTable` | Choice depends on heap contents and binding model |
| `SetDescriptorHeaps` | Bind argument table buffer / heap-backed `MTLBuffer` via argument table address | |
| `SetGraphicsRootDescriptorTable` | `[table setAddress:atIndex:]` | |
| `SetGraphicsRoot32BitConstants` | `[table setBytes:length:atIndex:]` | Inline constants |
| `SetGraphicsRootConstantBufferView` | `[table setAddress:atIndex:]` | GPU address of buffer |
| `SetGraphicsRootShaderResourceView` | `[table setAddress:atIndex:]` | GPU address of buffer |
| `SetGraphicsRootUnorderedAccessView` | `[table setAddress:atIndex:]` | GPU address of buffer |

#### Render passes / clears

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `OMSetRenderTargets` | Configure `MTL4RenderPassDescriptor` and start a new render encoder via `[cmd renderCommandEncoderWithDescriptor:]` | Render targets are pass-level, not encoder-state. *Perf: collapse consecutive render encoders when attachment configurations match — keeps tile memory hot on TBDR.* |
| `ClearRenderTargetView` | `MTLLoadActionClear` with `clearColor` on the color attachment | Pass-level load action; no per-call clear |
| `ClearDepthStencilView` | `MTLLoadActionClear` with `clearDepth` / `clearStencil` on depth/stencil attachment | |
| `ClearUnorderedAccessViewUint` / `ClearUnorderedAccessViewFloat` | Compute encoder fill (texture) or `[blit fillBuffer:range:value:]` (buffer) | |
| `DiscardResource` | `MTLStoreActionDontCare` on render pass attachment | Tile-memory only on TBDR — frees bandwidth |

#### Input Assembler / Output Merger state

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `IASetVertexBuffers` | `[table setAddress:atIndex:]` at vertex buffer slot | Slot offset depends on Metal shader converter convention (slot 6+ for converted shaders) |
| `IASetIndexBuffer` | Per-draw via `drawIndexedPrimitives:...indexBuffer:` | Index buffer not bound on encoder; passed per draw |
| `IASetPrimitiveTopology` | `MTLPrimitiveTopologyClass` on `MTL4RenderPipelineDescriptor` (class) + `MTLPrimitiveType` per draw (specific type) | Class is pipeline state; specific type is per-draw argument |
| `OMSetBlendFactor` | `[encoder setBlendColorRed:green:blue:alpha:]` | |
| `OMSetStencilRef` | `[encoder setStencilReferenceValue:]` (single) or `[encoder setStencilFrontReferenceValue:backReferenceValue:]` (per-side) | |

#### Rasterizer state (per-encoder)

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `RSSetViewports` | `[encoder setViewports:count:]` | |
| `RSSetScissorRects` | `[encoder setScissorRects:count:]` | |

#### Drawing

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `DrawInstanced` | `[encoder drawPrimitives:vertexStart:vertexCount:instanceCount:]` (with `:baseInstance:` variant) | |
| `DrawIndexedInstanced` | `[encoder drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:]` | |
| `Dispatch` | `[encoder dispatchThreadgroups:threadsPerThreadgroup:]` | |
| `ExecuteIndirect` (low command count) | `drawPrimitives:indirectBuffer:` looped per command | Simple, no extra setup |
| `ExecuteIndirect` (high command count, multiple state changes) | `MTLIndirectCommandBuffer` + `executeCommandsInBuffer:withRange:` | Higher integration cost — PSOs need `supportIndirectCommandBuffers`, full residency for ICB-referenced resources, command range and reset lifecycle to manage |

#### Synchronization

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `ResourceBarrier` (transition) | Queue barriers (producer/consumer) | See `managing-metal4-synchronization` skill |
| `ResourceBarrier` (UAV) | Intra-pass barrier with `VisibilityOptionDevice` | *Perf: stay within the same encoder — intra-pass barriers are cheap; ending an encoder to insert a barrier blows up encoder count and flushes tile memory on TBDR.* |
| `ResourceBarrier` (aliasing) | Barrier with `VisibilityOptionDevice` between writer of outgoing resource and reader/writer of incoming resource | `VisibilityOptionResourceAlias` is heavier and reserved for placement-sparse resources whose tiles share heap pages — see `managing-metal4-synchronization` |
| Split barriers (`BEGIN_ONLY` / `END_ONLY`) | Record on begin, emit on end | No direct Metal equivalent |

#### Copy / blit

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `CopyBufferRegion` / `CopyTextureRegion` | `[computeEncoder copyFromBuffer:...]` | Blit ops on compute encoder |
| `CopyResource` | Full-resource copy on compute encoder | |
| `ResolveSubresource` | `MTLStoreActionMultisampleResolve` on render pass attachment | Resolve happens at pass end, not as a separate command |

#### Queries

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `CreateQueryHeap` | `[device newCounterSampleBufferWithDescriptor:]` (timestamps) or visibility result buffer | |
| `BeginQuery` (`VISIBILITY_PREDICATE`) | `[encoder setVisibilityResultMode:MTLVisibilityResultModeBoolean offset:]` | |
| `EndQuery` (`VISIBILITY_PREDICATE`) | `[encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0]` | |
| `ResolveQueryData` | Read from visibility result buffer / counter sample buffer | |

#### Debug markers

| D3D12 | Metal 4 | Notes |
|---|---|---|
| `BeginEvent` (PIX) | `[encoder pushDebugGroup:]` (or on `MTL4CommandBuffer`) | |
| `EndEvent` (PIX) | `[encoder popDebugGroup]` | |
| `SetMarker` (PIX) | `[encoder insertDebugSignpost:]` | Zero-duration marker |

---

### Vulkan → Metal 4

#### Command infrastructure

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `VkQueue` | `MTL4CommandQueue` | |
| `VkCommandPool` | `MTL4CommandAllocator` | Reset per frame |
| `VkCommandBuffer` | `MTL4CommandBuffer` + encoders | Metal splits into typed encoders |
| `vkBeginCommandBuffer` / `vkEndCommandBuffer` | `[cmd beginCommandBufferWithAllocator:]` / `[cmd endCommandBuffer]` | |
| `VkFence` | `MTLSharedEvent` | CPU-GPU sync |
| `VkSemaphore` | `MTLEvent` (GPU-only) or `MTLSharedEvent` (if CPU also signals/waits) | Cross-queue sync. *Perf: `MTLEvent` is cheaper when no CPU interaction is needed.* |
| `VkEvent` | Producer/consumer barrier pair | |

#### Resource creation

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `vkCreateBuffer` + `vkAllocateMemory` + `vkBindBufferMemory` | `[device newBufferWithLength:options:]` | Allocation is implicit |
| `vkCreateImage` + `vkAllocateMemory` + `vkBindImageMemory` | `[device newTextureWithDescriptor:]` | Allocation is implicit |
| `vkCreateImageView` | `[pool setTextureView:texture descriptor:viewDesc atIndex:]` on an `MTLTextureViewPool` for argument-table / descriptor-heap binding (returns `MTLResourceID`). When the view must bind into an API that takes `id<MTLTexture>` (e.g., render-pass attachment), use `[texture newTextureViewWithPixelFormat:textureType:levels:slices:]`. | |
| `vkCreateBufferView` (typed buffer view) | `[pool setTextureViewFromBuffer:descriptor:offset:bytesPerRow:atIndex:]` on an `MTLTextureViewPool` for argument-table binding. For `id<MTLTexture>` consumers, allocate a buffer-backed texture: `[device newTextureWithDescriptor:]` with `MTLTextureTypeTextureBuffer`. | |
| `vkCreateSampler` | `[device newSamplerStateWithDescriptor:]` | |
| `vkAllocateMemory` (memory pool) | `[device newHeapWithDescriptor:]` | Heap-based sub-allocation |

#### Pipeline state

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `VkPipeline` | `MTLRenderPipelineState` or `MTLComputePipelineState` | |
| `VkPipelineLayout` | Argument table layout | |
| `vkCmdBindPipeline` | `[encoder setRenderPipelineState:]` / `setComputePipelineState:` | |

#### Descriptor binding

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `VkDescriptorSet` | `MTL4ArgumentTable` bindings | |
| `vkCmdBindDescriptorSets` | `[encoder setArgumentTable:]` | |
| `VK_DESCRIPTOR_TYPE_*` | Argument table `setAddress` / `setTexture` / `setSamplerState` | |
| `vkCmdPushConstants` | `[table setBytes:length:atIndex:]` | |
| `vkCmdBindVertexBuffers` | `[table setAddress:atIndex:]` at vertex buffer slot | |
| `vkCmdBindIndexBuffer` | Per-draw via `drawIndexedPrimitives:...indexBuffer:` | |

#### Render passes / clears

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `VkRenderPass` + subpasses | `MTL4RenderPassDescriptor` + load/store actions | No subpass equivalent — use programmable blending on TBDR |
| `vkCmdBeginRenderPass` / `vkCmdEndRenderPass` | `[cmd renderCommandEncoderWithDescriptor:]` / `[encoder endEncoding]` | *Perf: collapse consecutive render encoders when attachment configurations match — keeps tile memory hot on TBDR.* |
| `VkAttachmentDescription` (load/store) | `MTL4RenderPassColorAttachmentDescriptor` (`loadAction` / `storeAction`) | |
| Subpass dependencies | Load/store actions handle tile memory; programmable blending for read-after-write within tile | |
| `vkCmdNextSubpass` | No direct equivalent — use programmable blending or split into separate passes | *Perf: programmable blending preferred over splitting — keeps tile memory hot on TBDR.* |
| `vkCmdClearColorImage` | `MTLLoadActionClear` on render pass attachment, or compute fill outside a pass | |
| `vkCmdClearDepthStencilImage` | `MTLLoadActionClear` with depth/stencil clear | |
| `vkCmdFillBuffer` | `[blit fillBuffer:range:value:]` | |

#### Viewport / scissor

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `vkCmdSetViewport` | `[encoder setViewports:count:]` | |
| `vkCmdSetScissor` | `[encoder setScissorRects:count:]` | |

#### Drawing

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `vkCmdDraw` | `[encoder drawPrimitives:vertexStart:vertexCount:instanceCount:]` | |
| `vkCmdDrawIndexed` | `[encoder drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:]` | |
| `vkCmdDrawIndirect` | `[encoder drawPrimitives:indirectBuffer:]` (GPU address) | |
| `vkCmdDrawIndexedIndirect` | `[encoder drawIndexedPrimitives:...indirectBuffer:]` (GPU address) | |
| `vkCmdDispatch` | `[encoder dispatchThreadgroups:threadsPerThreadgroup:]` | |
| `vkCmdDispatchIndirect` | `[encoder dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:]` | |
| Multi-draw indirect (high count, multiple state changes) | `MTLIndirectCommandBuffer` + `executeCommandsInBuffer:withRange:` | Higher integration cost — see D3D12 ExecuteIndirect notes above |

#### Synchronization

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `vkCmdPipelineBarrier` | Queue barriers or intra-pass barriers | See `managing-metal4-synchronization` skill. *Perf: prefer intra-pass barriers when stages allow — they stay within the encoder. Queue barriers force encoder boundaries.* |

#### Copy / blit

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `vkCmdCopyBuffer` | `[computeEncoder copyFromBuffer:...]` | Blit ops on compute encoder |
| `vkCmdCopyImage` | `[computeEncoder copyFromTexture:...]` | |
| `vkCmdCopyBufferToImage` / `vkCmdCopyImageToBuffer` | `[computeEncoder copyFromBuffer:...toTexture:]` / inverse | |
| `vkCmdBlitImage` | Render to a render target with sampling — no direct arbitrary-blit equivalent | |
| `vkCmdResolveImage` | `MTLStoreActionMultisampleResolve` on render pass attachment | |

#### Queries

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `vkCmdBeginQuery` (`VK_QUERY_TYPE_OCCLUSION`) | `[encoder setVisibilityResultMode:MTLVisibilityResultModeBoolean offset:]` | |
| `vkCmdEndQuery` (occlusion) | `[encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0]` | |
| `vkCmdCopyQueryPoolResults` | Read from visibility result buffer / counter sample buffer | |

#### Debug markers

| Vulkan | Metal 4 | Notes |
|---|---|---|
| `vkCmdBeginDebugUtilsLabelEXT` / `vkCmdDebugMarkerBeginEXT` | `[encoder pushDebugGroup:]` | |
| `vkCmdEndDebugUtilsLabelEXT` / `vkCmdDebugMarkerEndEXT` | `[encoder popDebugGroup]` | |
| `vkCmdInsertDebugUtilsLabelEXT` / `vkCmdDebugMarkerInsertEXT` | `[encoder insertDebugSignpost:]` | |

---

### Metal 3 → Metal 4

| Metal 3 | Metal 4 | Notes |
|---|---|---|
| `[queue commandBuffer]` | `[device newCommandBuffer]` + allocator | Reusable, explicit lifecycle |
| `[cmd commit]` | `[queue commit:cmds count:]` | Batch submission |
| `[cmd presentDrawable:]` | `[queue signalDrawable:]` + `[drawable present]` | Queue-level presentation |
| `setVertexBuffer:offset:atIndex:` | `[table setAddress:atIndex:]` | Via argument table |
| `setFragmentTexture:atIndex:` | `[table setTexture:atIndex:]` | Via argument table |
| `[cmd blitCommandEncoder]` | `[cmd computeCommandEncoder]` | Blit ops on compute encoder |
| `[cmd parallelRenderCommandEncoder:]` | Suspend/resume or multiple cmd buffers | |
| `memoryBarrierWithScope:afterStages:beforeStages:` | `barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:` | Plus TBDR Fragment constraint |
| `useResource:usage:stages:` | `[residencySet addAllocation:]` | Residency set model |
| `addCompletedHandler:` | `MTLSharedEvent` or `MTL4CommitOptions` feedback | |
| `MTLFence` (cross-queue) | `MTLFence` (same-queue only) + `MTLEvent` (cross-queue) | Scope changed |
| `StorageModeManaged` | **Removed** — use `Shared` or `Private` | |
