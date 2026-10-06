# Grok source review — embedding rows — 2026-10-02

Scoped approval for one serial experiment. No source blocker.

`patch/versus-memory.diff` SHA-256 is `cc24b2d41370dac67e1c4f19e083f63595ef09828727751db04c605386c103b0`. All 12 `frozen-source-hashes.json` files match the candidate bytes. Against the memory-probe sources, only these three files differ:

- `QwenOfficialSourceModel.swift`
- `QwenOfficialSourceRunner.swift`
- `QwenExactBlockProbe.swift`

Run the candidate once with `TURBO_QWEN_PROTECTED_EMBEDDING_ROWS=1` and `TURBO_QWEN_EXPERT_CACHE_RESIDENCY=0`. Every other flag stays equal to residency-off `run.py`. The saved baseline is `residency-off-001/run-20261002T194046.297918Z-1ce275b1ad5c472bb2e94ffb2fc6c967`, which already has residency `0` and does not read the embedding flag. Pass requires the same 29 output IDs, 1120 route and weight bits, raw logit rows, and final state as that reference, with free memory at or above 30%. No speed claim. No second flag.

## Conversion and row read

`protectedEmbeddingRow` widens each little-endian `UInt16` with `Float(bitPattern: UInt32(UInt16(littleEndian:)) << 16)`. That is the same expression as the existing CPU BF16 vector load in this model (`QwenOfficialSourceModel.swift` 288–289). The resident GPU embedding kernel does the same shift (`qwenBF16Value` in `qwen_bf16.metal`: `as_type<float>(uint(bits) << 16)`). Inf and NaN stay in the bits. A failed `preadTensorRange` throws before the function returns the row, so the partial `UInt16` buffer is discarded.

The row path admits `model.language_model.embed_tokens.weight` through `admitTensor`, checks BF16 storage, shape `[vocab, 2048]`, and `region.size == vocab * 4096`, then does the existing zero-byte read at the end of the range. Each token reads 4096 bytes through unchanged `preadTensorRange`. That call checks owner, range, and retained-file identity, and keeps before-read, after-read, before-return, and cancellation. The token id must be in `0 ..< vocabularySize`, hidden size must be 2048, and `offset = id * 4096` must leave a full 4096 bytes inside `admittedByteCount`. The comparison is ordered so a too-large offset does not subtract past zero.

Pinned vocabulary 248320 makes one row 4096 bytes and the table `248320 * 4096 = 1,017,118,720` bytes. Accepted vocabulary cannot overflow that multiply (`vocabularySize <= 1,048,576`).

## Hooks, head, and image rows

`isKnownNone` is true only for `QwenOfficialSourceTransactionHooks.none`. The public initializer sets it false, including an empty closure. `load` applies the row path only when the flag argument is true and `embeddingHooks.isKnownNone`. The default argument is false, so ordinary load stays on the resident table. A custom hook object also stays on the resident table. The probe passes `.none` into both `load` and the runner, and reads the flag only for exact `"1"`. The runner throws if a row-mode model is paired with any other hooks.

`lm_head.weight` is still passed to `QwenBF16Weights`. Decode still projects `model.headName`. An image feature row is assigned before `embedding(token)` in both single-token produce and grouped prefill, so those rows do not read the embedding table.

## What the report proves

The embedding spec is omitted, so `spec` does not add `vocab * hidden * 2` to `residentWeightBytes`. For this model that product is 1,017,118,720. The report prints `removedResidentEmbeddingBytes = 1_017_118_720` only when `usesProtectedEmbeddingRows` is true, and it also prints `residentWeightBytes`. The constant is that omitted product, not a second measurement of `currentAllocatedSize`.

`analyze.py` compares `afterModelLoad` Metal markers and records whether the difference equals 1,017,118,720. It does not fail the run when Metal rounding makes the difference unequal. Treat that Metal number as the observed allocation check. The logical table removal is the omitted spec plus `residentWeightBytes`.

## Stop

One child. If IDs, routes, raw rows, or final state differ, or free memory goes below 30%, stop. Do not turn residency on and do not add another source change in that run.
