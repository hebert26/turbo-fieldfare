# Apple Silicon and Metal implementation

Read for kernels, scheduling, attention, or memory ownership. Source map checked
2026-09-07. Confirm the current code before editing. This reference supplies
engineering constraints, not permission to start an optimization program.

## Understand the machine

CPU and GPU share physical memory on Apple Silicon. Unified memory is not extra
RAM added to installed RAM. Shared resources can avoid duplicate CPU/GPU payloads,
but accesses still need correct ordering. See Apple's
[Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/).

Know the target chip, GPU family, bandwidth, RAM, and supported Metal features.
Check device capabilities before adopting newer instructions or matrix facilities.
An API present in the SDK is not necessarily supported or efficient on every
customer chip. Preserve the existing hardware dispatch paths.

Distinguish computation, memory traffic, SSD latency, CPU submission, and
synchronization as bottlenecks. A busy GPU alone does not establish efficiency.
Occupancy depends on resources available to concurrent threads, and excessive
competition can hurt caches. See Apple's
[occupancy guidance](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy).
Use profiling only when the assignment and repository rules allow it.

## Trace dependencies

Start in Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift,
Infrastructure/Metal/MetalContext.swift, and the owning Kernels/ and Metal/
files under Sources/TurboFieldfare/. Trace each buffer's producer, consumer,
last use, and release.

Single-token decode and batched prefill need different kernels. Follow the
existing GEMV path for decode and matrix/tile path for prefill. Consider fusion,
packed-weight decoding, coalesced access, SIMD reductions, register pressure,
and threadgroup scratch against actual shapes and the expensive operation.
Preserve affine quantization, scales, biases, activations, and reductions.
Do not materialize dequantized model weights as Swift arrays.

CPU router readback is a real dependency before exact expert IDs become
available to disk I/O. Batch dispatches where dependencies permit. Overlap
independent shared-expert work and bounded reads where supported. A second
command queue or fewer waits is not automatically faster. Never remove a wait,
barrier, or completion check without proving ordering.

Track submission and completion separately. Cancellation, partial submission,
load failure, and unload must retain resources until GPU use ends. Never
overwrite a slot referenced by in-flight work. Asynchronous CPU tasks also
require bounded ownership and cancellation behavior.

## Keep the live set small

Choose storage modes for actual access. Preserve alignment for mapped and
bytesNoCopy buffers, bounds, strides, and lifetimes. Shared addressing does not
make memory free, guarantee residency, or remove file-cache traffic. Treat
recommended working-set values as guidance alongside observed pressure.

Reuse bounded scratch and compiled pipelines. Avoid per-token/per-layer
allocations, repeated array conversion, and full transcript rebuilding.
Keep JSON, tool policy, and app bookkeeping in CPU code. Metal is the required
inference compute path, not a reason to make simple bookkeeping a GPU workload.

Budget resident weights, per-layer expert slots, K/V, prefill scratch, vision,
transport queues, decoded images, thinking display, and the foreground app.
OS file caching also affects pressure. Distinguish reserved capacity, resident
memory, and physical footprint.

## Attention and correctness

Inspect Runtime/KVCache/KVCacheManager.swift, Runtime/Prefill/PrefillChunkScratch.swift,
and attention kernels. Preserve logical positions, sliding-ring wrap, chunk
boundaries, and full-attention history. Full-attention K and V receive different
normalization/position transforms even when their raw projection is shared.
They cannot simply share final storage.

For scheduling-only changes, preserve exact results. For arithmetic changes,
use existing appropriate numerical references and established tolerances, plus
useful complete outputs. Never loosen a tolerance after failure. Consult Apple's
[bandwidth guidance](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth)
when unnecessary reads or writes cause the measured bottleneck.
