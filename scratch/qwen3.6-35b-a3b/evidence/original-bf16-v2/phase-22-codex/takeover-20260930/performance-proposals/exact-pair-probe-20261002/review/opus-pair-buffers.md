Opus pair buffer review, exact-pair-probe-20261002 (read-only targeted delta against the rejected exact-block sibling; no builds, runs or shard reads)

Verdict: scoped PASS, no blocker. This covers MoE offsets, private rows, routed/shared order, scratch hazards, retention, clearing, and the state and rollback effects of this delta. Source checks (Astra) and the schedule, counters and protocol (Grok) are outside this review.

Hashes:
- All 19 entries in `frozen-hashes.json` match the files on disk. The four delta files are:
  - QwenMoE.swift `bc01449b…`
  - QwenOfficialSourceRunner.swift `82006005…`
  - QwenExactBlockProbe.swift `3d3cd53e…`
  - QwenExactBlockTypes.swift `88325002…`
- The other candidate files are byte-identical to the rejected sibling.
- My own diff of the two candidate trees agrees with `patch/vs-rejected.diff`.

**Offsets.**
- Pair rows use `token × hiddenSize × 4`, the same formula as the existing routed `rowOffset` (QwenMoE.swift:1134).
- Both `hiddenRows` and `outputRows` are bounds-checked at that offset (:1192–1195).
- Route weights keep `(token × 8 + rank) × 4` (:1137).

**Private tracked rows.**
- There are 4 rows of `hiddenSize × 4` bytes, `.storageModePrivate` and `.hazardTrackingModeTracked`.
- They are allocated and checked for alias, device and tracking before `exactBlockAdmission?.validate` (:1226) and before any encoder.
- Each row is fully overwritten by a blit before it is read.
- `encodeSharedBF16` and `encodeProjection` check only length and device, not storage mode, so private inputs are valid.

**Arithmetic and order.**
- Work is the sorted pair union, each entry expanded to its pair tokens in token order. That gives (expertID, tokenIndex) order, so each token's 8 contributions are added in ascending ID with their original rank weights, exactly as serial does.
- After all routed encoders, each token gets: a blit of its post-norm row and its finished routed row, then the unchanged single-token `encodeSharedBF16` (`tokenCount: 1` projections, the same `_bf16` pipelines it already uses at :620–621), then a blit back to its row.
- The epilogue therefore adds the shared branch to the complete routed sum, as serial's single command does. The blits copy bytes, so the inputs are bit-identical to the rejected path's CPU `floats(post)`/`floats(routed)` copies.
- The CPU combine (`residual + combined`) is unchanged.

**Hazards.**
- Every bound buffer must be tracked: the guard rejects `.untracked` for the existing retained set, and the rows are created tracked.
- Within one command, Metal orders encoders by whole-resource dependencies:
  - routed writes to `outputRows` before token a's input blit;
  - token a's output blit before token b's input blit;
  - token a's epilogue reads of `sharedOutput`/`sharedOutputGate` before token b's projections overwrite them (write after read).
- Read-only weights and mapped experts need no ordering.

**Retention, failure and cancellation.**
- The rows are appended to `retainedBuffers` before `lease.submit`.
- On success and on the partial-encoding/cancellation path (:1314, :1323), `settleGroupedCommand` retains them with the lease until completion, including completion handlers. Task cancellation only cancels the lease while the wait continues.
- A failure before submit calls `lease.cancel()`.

**Clearing.**
- Pair A (`[0,1]`) alone has `initializeOutput` and clears all 4 rows before its contributions.
- The guard `initializeOutput == (tokens == [0,1])` forbids pair B from clearing, so A's finished rows 0–1 survive.
- B's encoders touch only rows 2–3, and those rows hold zero when B starts.

**State and rollback.**
- Pair B maps only after A's submit returns from settle, so at most 16 slots are leased at once (`requested.count ≤ 16`, `work.count == 16`).
- The CPU reads `outputRows` only after both pair commands have settled.
- KV and linear state are untouched by this delta.
- Any throw still runs `try? lease.cancel()` and the prefill catch, which restores the block-boundary baseline, so A's transient rows are discarded.
- With `exactPairSharedNames == nil`, production grouped prefill keeps the old planner and the separate shared commands.

Non-blocking notes:
1. Rows are allocated for every pair command: 4 × 80 per block, 32 KiB at a time. This costs performance only; consider preallocating after the gate.
2. Whole-buffer tracking serializes the two tokens' shared branches. That is correct but conservative.
3. The old group planner's `if let exactBlock` branches inside the `else` path are now unreachable. They are harmless.
