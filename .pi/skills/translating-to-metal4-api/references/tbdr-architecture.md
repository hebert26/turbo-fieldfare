# Apple GPU Architecture: Tile-Based Deferred Rendering (TBDR)

All Apple Silicon GPUs use TBDR. This is fundamentally different from the Immediate-Mode Rendering (IMR) architecture used by desktop NVIDIA/AMD GPUs that D3D12 and Vulkan engines are typically designed for. Understanding TBDR is critical for making correct decisions throughout a Metal port — it affects render pass structure, storage modes, synchronization, and performance.

## How TBDR Works

Rendering happens in two phases per render pass:

1. **Tiling phase (vertex processing):** All vertex shaders for every draw call in the render pass run first. The GPU transforms vertices to screen space and bins triangles into screen-space tiles.

2. **Rendering phase (fragment processing):** All tiles execute concurrently and in no guaranteed order, each using fast on-chip tile memory. Within each tile, all fragment shaders for all draw calls targeting that tile execute against tile memory. Only the final result is written to system memory via the render pass's store actions.

### Tile Size

Tile size is not fixed — it is determined by the total tile memory available divided by the per-pixel memory cost of all render target attachments. Specifically: tile size in pixels is inversely proportional to the per-pixel attachment cost, computed as (sum of pixel format sizes of all non-MSAA attachments) + (sum of pixel format sizes of all MSAA attachments × MSAA sample count). The maximum imageblock allocation size for the GPU family (found in Apple's GPU [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)) defines the total tile memory budget. Threadgroup memory allocated for tile shaders within the pass is also partitioned from this budget, further reducing available tile pixels.

You can query or override tile dimensions via `tileWidth`/`tileHeight` on the render pass descriptor. The default chooses the largest tile size that fits within the imageblock limit. Setting tiles larger than what fits causes some attachments to spill to device memory, which generally degrades performance compared to using smaller tiles — avoid this unless profiling shows a net benefit.

### Concurrent Unordered Tiles

During the rendering phase, all in-flight tiles are executing concurrently and modifying their respective portions of the render targets simultaneously. Reading from render target contents outside your tile is a race condition — the data is actively being written by other tiles. This is why fragment output cannot be read from system memory mid-pass: not only has it not been stored yet, but neighboring tiles are concurrently producing their output. The only safe way to read pixel data within a render pass is through programmable blending (`[[color(N)]]`), which reads from the current tile's own tile memory.

## IMR vs TBDR Comparison

| Aspect | IMR (NVIDIA/AMD) | TBDR (Apple Silicon) |
|--------|-------------------|---------------------|
| Fragment execution | Immediately after vertex, per-draw | Deferred until all geometry is binned, per-tile |
| Intermediate storage | System memory (VRAM) | On-chip tile memory |
| Render pass boundary cost | Low | High — each boundary incurs store actions for the ending pass + load actions for the starting pass |
| Overdraw cost | Full bandwidth per overdrawn pixel | Reduced — hidden surface removal before fragment shading |
| MSAA cost | Full bandwidth for multisample storage + resolve | Near-free — multisample textures can be memoryless, resolve happens in tile memory |
| Depth/stencil intermediates | Always in VRAM | Can be memoryless (tile-only, zero system memory) |

## Bandwidth Cost Model

Understanding bandwidth costs enables reasoning about whether a given render pass structure is efficient. All costs are per render pass boundary, per attachment:

**Load actions** (each tile performs this when it starts):
- `Clear` — tile memory initialized to clear value. No system memory read. Cheap.
- `DontCare` — tile memory left uninitialized. No system memory read. Cheapest.
- `Load` — tile memory populated from system memory. Reads the full attachment. Expensive — for a 4K RGBA16Float attachment, this is ~64MB of bandwidth.

**Store actions** (each tile performs this when it ends):
- `Store` — tile memory written to system memory. Full attachment write. Expensive.
- `DontCare` — tile memory discarded. No system memory write. Cheapest.
- `MultisampleResolve` — MSAA resolve performed in tile memory, resolved result written to the resolve texture. The multisample data itself can be discarded (memoryless).
- `StoreAndMultisampleResolve` — both the multisample data and the resolved result are written.

**Load and store actions are set per attachment, not per pass.** Different attachments in the same render pass can have different actions. This enables optimizing by loading/storing only the attachments that need to persist, while using `DontCare` for transient intermediates.

**Each render pass boundary** implies the store actions of the ending pass and the load actions of the starting pass. A pass boundary between two passes sharing an attachment with `Store` then `Load` costs a full write + a full read of that attachment. For a 4K RGBA16Float target, that's ~128MB of bandwidth per boundary for that one attachment alone.

## Efficient TBDR Patterns

- **Fewer, larger render passes.** All draw calls within a pass share tile memory — no store/load between them. A single pass with 1000 draw calls is cheaper than 10 passes with 100 draws each, if the data dependencies allow it.

- **Memoryless intermediates.** Depth buffers, stencil buffers, MSAA multisample textures, and G-buffer attachments that are only consumed within the same render pass can use `StorageModeMemoryless`. They exist only in tile memory and consume zero system memory and zero bandwidth. Example: a 4K 32-bit depth buffer = ~32MB of bandwidth saved per frame by avoiding store.

- **In-tile MSAA.** The multisample render target can be memoryless (exists only in tile memory). Use `MultisampleResolve` as the store action — the resolve happens entirely in tile memory. The resolve target (non-multisampled) is the persistent output written to system memory. This makes MSAA nearly free on Apple Silicon.

- **Programmable blending.** Fragment shaders can read the current tile's pixel values directly from tile memory via `[[color(N)]]` at zero bandwidth cost. This enables techniques like single-pass deferred rendering: G-buffer attachments are written and consumed in the same pass using memoryless storage — they never touch system memory.

- **Tile shader threadgroup memory.** Threadgroup memory allocated for tile shaders persists for the tile's lifetime within the render pass. Each tile shader dispatch grid inherits the memory state left by the previous tile shader dispatch on the same tile — enabling cheap forward communication between tile shader invocations without system memory round-trips. Imageblocks (the in-tile representation of render target pixels) are a special case of this persistent tile memory.

- **Tile-aligned algorithms.** If the engine's rendering pipeline implements screen-space tiling (e.g., for tiled light culling or bucketing), aligning the algorithm's tile size so each algorithmic tile fits entirely within a single GPU tile allows the associated data structures to reside in tile memory. If algorithmic tiles straddle GPU tile boundaries, this promotion to tile memory is not possible.

## Costly TBDR Patterns

These patterns incur bandwidth costs. Whether the cost is justified depends on data dependencies in the engine's rendering pipeline — sometimes the cost is unavoidable, but understanding it enables informed trade-offs.

- **Many render pass boundaries sharing attachments.** Each boundary costs one store + one load per shared attachment. D3D12 and Vulkan engines designed for IMR GPUs often split work into many small passes (shadow, G-buffer, lighting, post-processing) — this is the most common source of unnecessary bandwidth cost when porting to TBDR. On TBDR, each split that stores and then loads the same attachment costs significant bandwidth. However, passes targeting non-overlapping sets of textures (e.g., shadow maps vs. scene color) cannot be merged regardless — they are inherently separate.

- **`Load` action on attachments that could be cleared or left uninitialized.** If a render pass overwrites the entire attachment, `Clear` or `DontCare` avoids reading stale data from system memory. `Load` is only necessary when prior contents must be preserved (e.g., incremental rendering, or a pass that draws into a subregion of an existing target).

- **`Store` action on attachments that are not read after the pass.** If an attachment is only consumed within the pass (e.g., an intermediate depth buffer), storing it wastes bandwidth. Use `DontCare` for store and consider `StorageModeMemoryless`.

- **Post-processing as many separate render passes.** Chains where each effect (blur, bloom, tone mapping) is its own pass incur a store + load per step. Effects that can operate per-tile (no data dependencies outside the tile) can potentially be merged into a single pass or converted to compute.

## TBDR Hard Constraints

- **Cannot read other tiles' render target data within a render pass.** All tiles execute concurrently — reading another tile's output is a race condition. The only way to read pixel data within a pass is via programmable blending (`[[color(N)]]`), which accesses the current tile's own memory. If the engine reads a render target as a texture while also rendering to it (e.g., screen-space reflections sampling the color buffer), the pass must be split. Reason about data dependencies in terms of in-tile vs. not-in-tile — per-pixel reasoning is unnecessarily pessimistic since all pixels within the current tile are accessible.

- **Cannot establish cross-tile fragment-to-fragment dependencies within a pass.** You cannot issue an intra-pass barrier that waits for fragment output from other tiles, because those tiles are executing concurrently. While certain barrier encodings are legal (e.g., `afterStages`/`beforeStages` which can snap to the nearest valid point, typically a pass boundary), they do not achieve mid-pass cross-tile synchronization — they provide a coarser guarantee. If you need fragment output from the full framebuffer, split into separate render passes with inter-pass barriers. This is called the **TBDR Fragment constraint** and is referenced throughout the synchronization skill.

- **Feedback loops without programmable blending.** Any technique that reads and writes the same attachment must either use programmable blending (tile memory path, current tile only) or be restructured into multiple passes.

## Reasoning About Existing Workloads

When evaluating whether an engine's rendering pipeline is efficient on TBDR:

1. **Map the full render pass graph.** Identify every render pass, its attachments, and the data dependencies between passes. Which attachments are shared across passes? Which are consumed only within a single pass?

2. **Classify data dependencies as in-tile or not.** For each dependency where a pass reads data it produced: does the read need data from outside the current tile? If yes, a pass boundary is required. If the read is tile-local, the work may be mergeable into a single pass using programmable blending or tile shaders.

3. **Identify memoryless candidates.** For each attachment: is it read after the render pass that produces it ends? If no, it can be `StorageModeMemoryless` with `DontCare` store action — zero system memory, zero bandwidth.

4. **Evaluate pass merging opportunities.** Passes that target the same (or mergeable) sets of attachments and whose data dependencies are all in-tile can potentially be merged. Post-processing effects that operate per-tile are prime candidates. Passes targeting entirely different texture sets (e.g., shadow passes vs. scene passes) cannot be merged.

5. **Check for IMR-specific patterns that hurt TBDR.** "Render pass splitting for GPU parallelism" is an IMR technique that is counterproductive on TBDR. "Early-Z pre-passes as separate render passes" may cost more in pass boundary bandwidth than they save in overdraw — Apple's hardware hidden surface removal already reduces overdraw within a pass. Evaluate these trade-offs with profiling.

6. **Estimate bandwidth costs.** For each pass boundary, sum the bandwidth: (store action cost per attachment) + (load action cost per attachment in the next pass). Use attachment dimensions × bytes-per-pixel × sample count as the per-action cost. This gives a concrete number to compare against the cost of alternative structures.
