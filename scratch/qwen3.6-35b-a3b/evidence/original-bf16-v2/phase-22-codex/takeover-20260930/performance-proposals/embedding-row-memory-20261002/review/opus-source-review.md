Opus source review, embedding-row-memory-20261002 (read-only at 526d35a; no builds, runs or payload reads)

Verdict: scoped approval for one bounded serial experiment. RESIDENCY=0 in both arms; only `TURBO_QWEN_PROTECTED_EMBEDDING_ROWS` differs. No blocker found. This is not production approval.

**Hashes.**
- `patch/versus-memory.diff` SHA256 is cc24b2d4…103b0, as stated.
- All 12 entries in `frozen-source-hashes.json` match the files on disk.
- The diff touches exactly three files: QwenOfficialSourceModel, QwenOfficialSourceRunner and QwenExactBlockProbe.

**Exact conversion.**
- The CPU row path computes `Float(bitPattern: UInt32(UInt16(littleEndian: w)) << 16)`.
- The GPU path, `qwen_bf16_embedding` via `qwenBF16Value`, computes `as_type<float>(uint(bits) << 16)` (`qwen_bf16.metal:16–18`, 20–41).
- Both are pure bit widenings of the same little-endian BF16 word, so they are bit-identical, including NaN and Inf payloads.
- Downstream, both produce a `[Float]` row.

**Protected read path:**
- **Admission.** Load uses `source.admitTensor` for the indexed shard: an owner-bound `TensorRange` with a retained FD. `LoadedTensorRegion` then requires BF16 storage, shape [vocab, 2048] and size = vocab × 4096.
- **Zero-byte read at the end offset.** `readPlan` (OfficialSourceHandle.swift:631–660) allows offset = size with count 0, so this runs the existing check path at admission.
- **Each row.**
  - Cancellation is checked before and after.
  - The token is bounds-checked (0 ≤ id < vocab).
  - The offset is checked (offset + 4096 ≤ `admittedByteCount`).
  - One 4096-byte `preadTensorRange` follows: its owner and boundary checks, a single syscall under the 4 MiB tile, and the original before-read, after-read and before-return checks.
  - A throw discards the temporary buffer.
- **Per call:** a 4 KiB word array and an 8 KiB float array. No table, Metal buffer or extra copy.

**Gates and unchanged paths:**
- **Off by default.** The new loader parameter defaults to false, and the effective value is `protectedEmbeddingRows && embeddingHooks.isKnownNone`. Any other request falls back to the unchanged resident path.
- **Default path unchanged.** It builds `entry` as [embedding] + [head], the same order and the same `consumed` accounting as before.
- **Runner guard** (runner :239–241): it refuses a protected model unless the runner's own `hooks.isKnownNone`. It throws after the expert reservation, which is a local released through ARC, as the existing comment states.
- **Output head:** still resident and still the only `entryWeights` tensor in row mode. `projection(entryWeights, headName)` is unchanged.
- **Image features:** `produceToken` and prefill still take `featureRow` before `embedding(token)`.
- **Other users:** `embedding(_:)` is the only user of `encodeEmbeddingFromImmutableIDs`.

**Proof of removal (scoped note, not a blocker).**
- `residentWeightBytes` is computed from `consumed`. In row mode `spec(embedding)` is never called, so the measured drop should be exactly 1,017,118,720 bytes.
- The phase probe's `afterModelLoad` Metal `currentAllocatedSize` should fall from 4,894,752,768 to about 3,877,634,048.
- The report field `removedResidentEmbeddingBytes` is a hard-coded constant, not evidence. The analysis must take both differences from the measured fields against the baseline run.

**Expected differences, not defects:**
- Per-cycle IO counters include one extra 4 KiB protected read per consumed token.
- Prefill gains 1,175 row reads, about 0.5–1.2 s estimated.
- `expertCacheQuota` capacity rises by the freed bytes. The cache still reserves exactly 40 × 16 slots, so allocation and LFU behaviour are unchanged.

**Run gate:**
- Exact match to the residency-off reference, `b433d22a…`:
  - all 29 IDs;
  - accepted route SHA `95a37710…`;
  - accepted raw-row SHA `212502f1…`;
  - linear-state, committed-K and committed-V SHAs;
  - positions 1203 with one pending token.
- `usesProtectedEmbeddingRows` is true.
- Measured removal as above.
- Report minimum free memory and the phase samples. Free memory ≥ 30% is the only resource pass; report the margin rather than claiming one.
