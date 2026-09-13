# Residency sets and deferred destruction

Detail on `MTLResidencySet` — kernel/userspace semantics, attachment levels, validation rules — and the canonical deferred-destruction pattern that releases resources only after the GPU has finished referencing them. The SKILL.md body covers the policy (every GPU-accessed allocation must be in a committed residency set); load this file when wiring the actual creation, attachment, and frame-delayed release.

## Residency Sets

All resources must be in a residency set before GPU access. Replaces `useResource`/`useHeap`.

Residency sets are not thread safe. Adding and removing allocations should be protected by a mutex when multiple threads are involved.

Residency sets are mirrored objects — they have a userspace representation and a kernel-side representation. `addAllocation`/`removeAllocation` modify the userspace side only. `commit()` synchronizes the kernel side with the userspace side. This is a CPU-side operation that is relatively expensive, so batch adds and removes together and commit once rather than committing per operation.

When GPU workload is submitted (via `queue commit:`), the GPU snapshots which residency sets are associated with the workload — but not the contents of those sets. This means all required resources must already be added and committed in at least one associated residency set **before** the workload is submitted. For removal, the application must not commit the removal of the last reference to a resource from the associated residency sets until the GPU workload referencing that resource has completed.

```objc
// ObjC
MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
desc.initialCapacity = 128; // hint for initial allocation — set auto-grows
id<MTLResidencySet> resSet = [device newResidencySetWithDescriptor:desc error:&error];

// Add resources (MTLBuffer, MTLTexture, MTLHeap all conform to MTLAllocation;
// adding a heap makes its entire contents resident).
id<MTLAllocation> allocations[] = { buffer, texture, heap };
[resSet addAllocations:allocations count:3];
[resSet commit]; // sync adds to kernel — must happen before GPU workload submission

// Request persistent residency — call once after creating the set.
// Wires all currently committed resources and automatically wires
// future resources on subsequent commit calls.
[resSet requestResidency];

// Attach to queue (or per-command-buffer via [cmd useResidencySet:resSet])
[queue addResidencySet:resSet];
```

**`requestResidency` is a persistent state** — call it once after creating the set. It wires all currently committed resources and ensures future `commit` calls automatically wire newly added resources. No need to call it each time resources are added.

### Attachment Levels

```objc
// ObjC
// Queue-level (persistent, most efficient) — up to 32 per queue
[queue addResidencySet:resSet];  // auto-attached to all committed cmd buffers

// Command-buffer-level (per-frame) — up to 32 per cmd buffer
[cmd useResidencySet:resSet];    // between beginCommandBuffer and endCommandBuffer
```

**Prefer queue-level** for resources used every frame. Use command-buffer-level for varying resources.

### Implementation Patterns

| Pattern | Approach | Best For |
|---|---|---|
| **Global set** (recommended) | Single set on queue, add/remove as resources created/destroyed | Most engines |
| **Per-frame sets** | Sets per command buffer for fine control | Complex engines |
| **Hybrid** | Queue-level for persistent + cmd-buffer for transient | Engines needing fine-grained per-frame control |

### Validation Rules

- **Command queue:** up to 32 residency sets (via `addResidencySet`), independent of command buffer limits.
- **Each command buffer:** up to 32 residency sets (via `useResidencySet`), independent of the queue's limit.
- **Suspend/resume chains:** the entire sequence counts as a single command buffer, so all command buffers in the chain share one 32-set limit.
- `useResidencySet` must be called between `beginCommandBuffer` and `endCommandBuffer`.
- Memoryless textures cannot be added to residency sets (they exist only in tile memory).

## Deferred Resource Destruction

Resources must survive until the GPU finishes referencing them. The residency set removal (for resources that the GPU is referencing) must not be committed until the GPU workload has completed. Use frame-delayed destruction:

```
// On create: add to residency set under the same mutex used for removal
lock(residencySetMutex)
    residencySet.addAllocation(resource)
unlock(residencySetMutex)

// On destroy: queue resource for deferred release
deferredList[currentFrame % maxFramesInFlight].add(resource)

// On frame update: destroy resources from N frames ago (GPU no longer referencing)
lock(residencySetMutex)
    residencySet.removeAllocation(resource)
unlock(residencySetMutex)

// Batch all removals, then commit once
residencySet.commit()
resource = nil  // release the resource
```

**Ensure the GPU is done with the resource before removing from the residency set and committing.** Committing a removal while the GPU still references the resource can cause GPU faults — the kernel may evict the resource. The deferred destruction pattern (wait N frames matching `maxFramesInFlight`) ensures the GPU workload has completed before the removal is committed.
