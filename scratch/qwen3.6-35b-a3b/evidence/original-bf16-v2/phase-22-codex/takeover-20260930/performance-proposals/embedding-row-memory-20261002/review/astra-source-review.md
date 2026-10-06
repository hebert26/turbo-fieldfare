Astra embedding-row source review — frozen, 2026-10-02

Scoped PASS for the bounded serial memory comparison. No concrete blocker found. All12 candidate files match frozen-source-hashes.json SHA25667694a9d45fb3ae3f2b4772eee05a5ece3f01cd4326323b148f36c3a5c47af31. The actual patch matches cc24b2d41370dac67e1c4f19e083f63595ef09828727751db04c605386c103b0 and changes only model/runner/probe versus the memory baseline. Main owns wrappers and execution.

Reviewed file SHA256
- Model:1ce5b41193750056bfa57944d2808e409b3573dc2238bc206eade5cb96580c19
- Runner:305be6432de7d9aaac812a6cfb1819e8f1c472f8838ad2916350b3798946900a
- Probe:af0ee3b1efc1f2d8538070177403a3ba0b26ddf012e8c5f3d7e7298676a49401

Source and byte conversion
The model retains the actual OfficialSourceHandle.TensorRange from admitTensor, with its owner-bound retained descriptor. That token retains its source owner and closes its descriptor on deinitialization. The model owns both until release. Header storage must be BF16, shape must equal[vocab,2048], and byte count must be vocab×4096. The public loader validates architecture first, bounding vocab at1048576, so this size multiplication is safe. Each nonnegative Int32 token is range-checked before UInt64 conversion. Its4096-byte row offset is safe; subtraction-based remaining-length checks precede the protected reader's additional checked file/off_t arithmetic.
Each row invokes unchanged direct-buffer preadTensorRange into exactly2048 UInt16 words. Entry/loop/final cancellation, beforeRead/afterRead/beforeReturn validation, short-read/EINTR handling and failure propagation remain in that protected reader. A failing local destination is discarded, never returned. The initialization zero-length end read performs beforeReturn validation only; it is not a three-check payload read and should not be described as one. Full trusted-receipt/source validation at model publication and original later consumer boundaries remains intact. No reader implementation was modified.
UInt16(littleEndian:) followed by UInt32(word)<<16 and Float(bitPattern:) is the exact bit widening in qwen_bf16.metal qwenBF16Value (:16–17). No rounding, accumulation or conversion through Float(integer) is introduced. Finite-value behavior remains downstream as before. The caller receives an owned immutable Float row, not an alias into reusable I/O memory.

Eligibility and consumers
Default protectedEmbeddingRows=false preserves the resident table and original GPU embedding path. An explicit request with a supplied custom loader hook uses the resident path because isKnownNone is false, including an explicit no-op callback. A row-mode model rejects a custom-hook runner, preserving the fact that CPU reads cannot provide the removed embedding GPU submission/completion events. Genuine hooks.none is used by this isolated probe. Row-mode selection is frozen in the model, not a mutable environment check per token.
Ordinary produce and grouped prompt both call the same embedding(_:). Image feature rows still bypass that method. The language head remains a separate resident entry, as required by the untied source model. All full/linear/MoE/routing code is unchanged. This does not qualify the unexercised image path, custom-hook fallback or failure injections empirically.

Memory evidence requirements
Resident accounting excludes the embedding specification only in row mode, so the expected removed allocation is248320×2048×2=1,017,118,720 bytes (970MiB). No whole embedding table or FP32 cache is introduced. A call holds4096 bytes of words plus8192 bytes of Float output, with normal allocation overhead. The existing model and expert-cache reservation remain.
protectedEmbeddingRows reports the actual retained-range state. removedResidentEmbeddingBytes is a declared constant for the pinned geometry, not an independent measurement. Validate the claimed reduction using residentWeightBytes difference AND afterModelLoad/currentAllocatedSize markers with equal other flags. Physical footprint and system-free improvement remain separate empirical questions. No guaranteed3-point gain or speed claim follows.

Decision: clear for the fixed residency0 baseline-versus-row comparison, preserving30% floor and the complete baseline's IDs, raw rows, route bits and actual final-state digests. No test, build, model execution, payload read or source edit was performed by this reviewer.
