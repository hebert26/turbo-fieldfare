# Qwen3.6-35B-A3B integration analysis and implementation plan

Folder index: [[turboCharge/Index]].

Date: 14 September 2026  
Repository: `/Users/dev-machine/dev/turbo-fieldfare-personal`  
Inspected revision: `a64d0d3540d1b61d501c7f9876dee4813a560996`  
Branch: `codex/visioncapture-mcp-chat-poc`  
Status: planning only

## 1. Decision

Add the exact `Qwen/Qwen3.6-35B-A3B` model as a second, explicitly selected model family. Keep Gemma 4 as the default and rollback path.

Do not treat Qwen as another Gemma configuration. It needs a separate artifact schema, runner, conversation-state implementation, chat/tool codec, and vision path behind shared product boundaries.

The safest delivery order is:

1. Preserve and freeze Gemma `.gturbo` v1 behavior.
2. Add model-family-aware `.gturbo` v2 metadata and loading.
3. Bring up Qwen text inference with deterministic fixtures.
4. Add native Qwen chat, thinking, and tool calls.
5. Add Qwen still-image input and multimodal RoPE.
6. Add app selection, installation, service, CLI, and server integration.
7. Run correctness, recovery, Metal validation, and same-machine comparison gates.

Qwen must remain opt-in until every applicable gate passes. A bad or incomplete Qwen installation must fail closed without changing, deleting, or loading the installed Gemma model.

## 2. Task contract

### Observable behavior

A completed implementation must let the user:

1. Keep using the current Gemma installation with unchanged defaults and outputs.
2. Install Qwen3.6-35B-A3B into a separate directory.
3. Select exactly one installed model before loading it.
4. Run text chat, multi-turn chat, thinking-mode control, and native function calls with the selected model.
5. Attach still images only when the matching Qwen vision component is installed and verified.
6. Use the selected model through the Mac app, decode service, CLI, local server, and VisionCapture agent loop.
7. See the real selected model identity in status, diagnostics, logs, and `/v1/models`.
8. Switch models only through an unload/reload boundary. The old model must not remain loaded in another process.

### Failure behavior

The implementation must reject:

1. A Qwen3, Qwen3.5, or differently sized checkpoint presented as Qwen3.6-35B-A3B.
2. A mutable or unrecognized source revision.
3. A Gemma artifact interpreted through the Qwen runner, or the reverse.
4. Missing, renamed, extra, incorrectly shaped, or incorrectly quantized tensors.
5. A vision companion from another model or source revision.
6. Images when Qwen image support is missing or invalid.
7. Malformed Qwen tool calls without dispatching them.
8. Stale conversation state after cancellation, stop-string cleanup, checkpoint replacement, reload, or model switching.
9. Unsupported video, audio, speculative MTP, or extended-context requests rather than silently approximating them.

### Boundaries affected

The change crosses these existing areas:

- wire format and manifests;
- remote source and repacker;
- resident and streamed weight loading;
- Metal kernels and runner;
- attention, recurrent state, and conversation recovery;
- tokenizer, chat template, thinking, and tool parsing;
- image preprocessing, vision tower, projector, and multimodal positions;
- decode-service protocol and lifecycle;
- app installation, selection, settings, diagnostics, and text;
- CLI and OpenAI-compatible server;
- VisionCapture tool loop and context checkpoints;
- tests, validation fixtures, and benchmark records.

### Performance budget

There is no measured Qwen baseline in this repository. No throughput target should be invented now.

The initial performance contract is structural:

- no unbounded per-token allocation;
- bounded expert cache and in-flight GPU resources;
- no full routed-expert residency requirement;
- no blocking GPU wait on the app main actor;
- no claimed speedup over Gemma without controlled measurements;
- no 262K or extended-context product promise until memory, correctness, and latency are measured.

Any numeric release budget must be proposed after the first correct Release baseline and labelled provisional.

## 3. Inspection baseline and conflict check

`plan/report.md` and its parent directory did not exist before this report. There was no prior report content to preserve and no report conflict.

Two unrelated working-tree changes already existed:

```text
 D .pi/agents/mac-implement.md
?? .pi/agents/mac-implementer.md
```

They are not part of this plan and must remain untouched.

Verified host environment:

| Item | Value |
| --- | --- |
| Xcode | 26.5 |
| Swift | 6.3.2 |
| Package tools version | Swift 6.2 |
| Deployment targets | macOS 26, iOS 26 |
| macOS | 26.6.2 |
| Architecture | arm64 |
| Mac identifier | Mac14,12 |
| Memory | 32 GiB |

No build, test suite, model process, benchmark, dependency install, or model-weight download was run for this planning work. Small public metadata files were read remotely and were not added to the checkout.

## 4. Verified Qwen identity

The integration target is exactly:

| Field | Verified value |
| --- | --- |
| Repository | `Qwen/Qwen3.6-35B-A3B` |
| Pinned revision | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Access | Public, ungated |
| License | Apache-2.0 |
| Pipeline | `image-text-to-text` |
| Top-level model type | `qwen3_5_moe` |
| Transformers class | `Qwen3_5MoeForConditionalGeneration` |
| Official weight dtype | BF16 |
| Official tensor payload | 71,903,645,408 bytes across 26 shards |
| Total/active parameters | 35B total, 3B activated, as reported by Qwen |
| Native context | 262,144 tokens, as reported by Qwen |

The `qwen3_5_moe` implementation name is the architecture used by this exact Qwen3.6 repository. It is not permission to substitute a Qwen3.5 checkpoint.

Authoritative pinned sources:

- [official model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/README.md)
- [official config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json)
- [official license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/LICENSE)
- [official tensor index](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/model.safetensors.index.json)
- [official tokenizer configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/tokenizer_config.json)
- [official generation configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/generation_config.json)
- [official image preprocessing configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/preprocessor_config.json)

### 4.1 Text architecture

The pinned configuration and model card agree on these material facts:

| Field | Value |
| --- | ---: |
| Layers | 40 |
| Hidden size | 2,048 |
| Vocabulary / LM output | 248,320 |
| Layer pattern | 10 repetitions of 3 linear-attention layers, then 1 full-attention layer |
| Linear-attention layers | 30 |
| Full-attention layers | 10, at zero-based indices 3, 7, …, 39 |
| Linear Q/K heads × dimension | 16 × 128 |
| Linear V heads × dimension | 32 × 128 |
| Linear convolution kernel | 4 |
| Recurrent-state dtype | FP32 in the reference config |
| Full Q heads × dimension | 16 × 256 |
| Full KV heads × dimension | 2 × 256 |
| Rotary dimension | 64, from partial factor 0.25 × head dimension 256 |
| RoPE theta | 10,000,000 |
| Routed experts | 256 per layer |
| Selected routed experts | 8 per token |
| Routed expert width | 512 |
| Shared expert width | 512 |
| Activation | SiLU |
| Tied embedding/head | No |

This is a hybrid recurrent/full-attention model, not Gemma's sliding/full-attention model.

The upstream reference performs these operations that need exact coverage:

- depthwise causal convolution before the Gated DeltaNet rule;
- Q/K L2 normalization inside the delta rule;
- FP32 `A_log`/time-decay math;
- recurrent state update for every linear-attention layer;
- a learned output gate on full attention;
- Q/K per-head RMS normalization;
- partial and multimodal interleaved RoPE;
- full-router softmax in FP32, Top-8 selection, and renormalization;
- a sigmoid-gated shared expert in parallel with routed experts;
- a separate, untied LM head.

Pinned upstream implementation evidence was also inspected at Transformers commit [`bd15bc95a89e728bbc1224084eb3b5829428c353`](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py). The implementation commit must be recorded with any generated validation fixtures because this code can change independently of the model repository.

### 4.2 Conversation-state implications

Thirty layers maintain fixed recurrent and convolution state. Ten layers maintain growing full-attention K/V state.

The recurrent matrix alone is theoretically:

```text
30 layers × 32 value heads × 128 key channels × 128 value channels × 4 FP32 bytes
= 62,914,560 bytes = 60 MiB
```

The ten full-attention layers require, for FP16 K and V:

```text
10 layers × (2 KV heads × 256 values) × 2 tensors × 2 bytes
= 20,480 bytes per retained text position
```

That is about 80 MiB at 4,096 positions and 1.25 GiB at 65,536 positions, before allocator padding and other state. These are layout calculations, not measured process-memory results.

A recurrent state cannot be rewound by decrementing a token position. Qwen therefore needs transactional conversation state:

1. Keep one bounded snapshot at a committed turn or checkpoint boundary.
2. Write the active turn into separate working state.
3. Commit only after GPU completion and accepted generation outcome.
4. Restore the boundary snapshot on cancellation or rejected output.
5. If a multi-token stop string must be removed, restore the nearest snapshot and replay only accepted tokens.

Per-token copies of the full recurrent state would be unbounded and are not acceptable.

### 4.3 Tokenizer, chat, thinking, and tools

The exact repository uses a Qwen BPE tokenizer, declared as `Qwen2Tokenizer`. Important IDs include:

| Token | ID |
| --- | ---: |
| `<|endoftext|>` | 248044 |
| `<|im_start|>` | 248045 |
| `<|im_end|>` | 248046 |
| `<|vision_start|>` | 248053 |
| `<|vision_end|>` | 248054 |
| `<|image_pad|>` | 248056 |
| `<|video_pad|>` | 248057 |
| `<tool_call>` | 248058 |
| `</tool_call>` | 248059 |
| `<tool_response>` | 248066 |
| `</tool_response>` | 248067 |
| `<think>` | 248068 |
| `</think>` | 248069 |

The generation stop IDs are 248046 and 248044.

The bundled template:

- frames roles with `<|im_start|>` and `<|im_end|>`;
- opens thinking by default;
- disables thinking by emitting an empty `<think>…</think>` block;
- can preserve historical thinking when explicitly requested;
- describes tools in a system `<tools>` block;
- emits calls as `<tool_call><function=name><parameter=name>…`;
- groups tool results inside a user-framed `<tool_response>` block;
- represents images with `<|vision_start|><|image_pad|><|vision_end|>`.

Qwen's native publisher settings differ from TurboFieldfare's current defaults. The official generation config uses temperature 1.0, Top-K 20, and Top-P 0.95. The model card recommends different profiles for general thinking, precise coding, and non-thinking use. TurboFieldfare currently defaults to 0.2, 64, and 0.95. This difference must be visible in future comparison records.

### 4.4 Vision architecture

The exact checkpoint includes a vision encoder with:

| Field | Value |
| --- | ---: |
| Depth | 27 |
| Hidden size | 1,152 |
| Heads | 16 |
| MLP width | 4,304 |
| Patch size | 16 |
| Temporal patch size | 2 |
| Spatial merge | 2 |
| Output size | 2,048 |
| Activation | GELU tanh approximation |
| Image normalization | mean 0.5, standard deviation 0.5 |

The official tensor index contains 333 tensors under `model.visual.*`.

Still-image integration requires more than replacing image embeddings. The reference processor supplies image-grid dimensions and modality token types. The language model then computes three-axis multimodal positions using temporal, height, and width indices and carries a RoPE delta into incremental decoding.

The first supported scope should be still images only. The trained checkpoint also declares video tokens and processing behavior, but TurboFieldfare has no verified video ingestion contract. Video and audio remain unsupported until separately planned and tested.

### 4.5 MTP

The official index includes 19 multi-token-prediction tensors. MTP is a trained capability used for speculative decoding by some serving engines.

The first Qwen runner should use ordinary one-token autoregressive decoding. The repacker must explicitly list MTP tensors as intentionally omitted rather than classifying them accidentally. MTP optimization is a later, measured project.

## 5. Quantized source decision

The official Qwen repository is BF16 and about 71.9 GB. The current TurboFieldfare repacker does not quantize BF16 weights; it reorganizes an already quantized MLX-affine source.

Two public derived formats were verified:

| Candidate | Pinned revision | Size and format | Decision |
| --- | --- | --- | --- |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` | 20,401,929,952-byte MLX-affine artifact, four shards; 4-bit group 64 with 8-bit router/shared gates | Best mechanical match for prototyping, but derived-source lineage to the exact official revision is not proven by its model card |
| [`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/tree/5c2410d71524f4f72b023ce8daf7a80528226d5f) | `5c2410d71524f4f72b023ce8daf7a80528226d5f` | Q4_K_M text file 22,285,080,192 bytes; BF16/F16 projector files about 0.9 GB | Do not use for the native path; it would add a second quantization layout and conversion contract |

Recommended source gate:

1. Anchor identity and architecture to official revision `995ad96…`.
2. For a short-lived engineering prototype, allow the pinned MLX artifact only behind a non-release source profile.
3. Before release, either obtain auditable proof tying that conversion to `995ad96…`, or implement a deterministic, bounded conversion from the pinned official BF16 shards.
4. Record converter name, version, commit, arguments, source tensor-index SHA-256, output tensor-index SHA-256, and every per-tensor quantization override.
5. Differentially compare the produced quantized tensors and logits against a reference using the same quantized weights.

A repository name and `base_model` tag are not enough to prove exact source-revision lineage. This is currently unresolved.

## 6. Current TurboFieldfare findings

### 6.1 Artifact and loading

`.gturbo` v1 is Gemma-specific even though its name is generic.

[`GTurboManifestV1.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfareFormat/GTurboManifestV1.swift) assumes one architecture shape containing:

- sliding-window and full-attention dimensions;
- Gemma attention K=V behavior;
- one full-attention mask;
- Gemma-oriented quantization slots;
- one uniform routed-expert stride.

It has no model-family discriminator, recurrent-state description, convolution description, output-gate description, multimodal RoPE layout, untied-head contract, or explicit ignored-tensor record.

[`ManifestReader.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift) validates the manifest against the compile-time `ArchConfig.gemma4_26B_A4B`. [`Model.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/Model.swift) hard-codes Gemma tensor names and returns the embedding as the LM head.

Conclusion: do not encode Qwen by filling Gemma fields with misleading values. Add a new major format version with an explicit architecture union. Keep the v1 codec and fixtures unchanged.

### 6.2 Repacker and installer

[`SupportedModelSource.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Remote/SupportedModelSource.swift) pins one Gemma MLX repository and revision.

[`ArchInfo.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Format/ArchInfo.swift) reads Gemma-only fields. [`RepackPlanner.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift) recognizes Gemma expert names and Gemma multimodal prefixes. It expects prequantized MLX affine tensors and a fixed companion layout.

Qwen's official names differ:

- linear layers use `.linear_attn.*`;
- full layers use `.self_attn.*`;
- experts are packed as `experts.gate_up_proj` and `experts.down_proj`;
- shared-expert gating has its own weight;
- `lm_head.weight` is independent;
- vision tensors live under `model.visual.*`;
- MTP tensors live under `mtp.*`.

Conclusion: source recognition, architecture decoding, tensor classification, splitting, layout, receipts, install estimates, cancellation/resume, and companion binding all need family-specific plans.

### 6.3 Metal runtime

[`RealForwardRunner.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift) and most files under [`Kernels`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels) implement the Gemma block directly.

Parts that can be reused after contract checks include:

- Metal device and command-queue ownership;
- affine 4-bit/8-bit dequantized matrix operations where layouts match;
- bounded `pread` expert streaming and cache policy;
- buffer/index integrity checks;
- sampling primitives, except Gemma logit soft-capping must not be applied to Qwen;
- measurement capture and completion plumbing.

New Qwen kernels are required for:

- decode and chunked-prefill causal depthwise convolution;
- recurrent and chunked Gated DeltaNet updates;
- Q/K L2 normalization and FP32 decay math;
- gated DeltaNet RMS normalization;
- full-attention Q output gating;
- Qwen partial/interleaved multimodal RoPE;
- Qwen router and sigmoid-gated shared expert;
- Qwen vision patching, attention, merger/projector, and dynamic grid positions.

The project currently uses established Metal APIs. The inspected Apple Metal skills do not justify a Metal 4 migration. Keep the current API generation unless a separate measured migration is approved.

### 6.4 Cache, recovery, and compaction

[`KVCacheManager.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/KVCache/KVCacheManager.swift) owns linear/ring FP16 K/V and supports cheap positional rewind. That behavior is valid for Gemma's stored K/V, not Qwen's recurrent state.

[`MultimodalConversation.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Generation/MultimodalConversation.swift) relies on rewind, replay, retained image features, checkpoint rebuild, and cancellation recovery. These are product semantics that both models must preserve through different state implementations.

The existing performance-compaction trigger in [`DecodeProtocol.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift) uses a 15 tokens/second threshold. That measured Gemma policy must not silently govern Qwen. Capacity checkpoints can be shared. Performance-triggered Qwen compaction must remain disabled until a Qwen baseline and a model-specific policy exist.

### 6.5 Tokenizer and tool loop

[`Tokenizer.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Tokenization/Tokenizer.swift) validates Gemma's decoder chain, special tokens, vocabulary, chat templates, and image IDs. [`GemmaToolCallParser.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Tokenization/GemmaToolCallParser.swift) and [`StructuredAssistantDecoder.swift`](file:///Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Tokenization/StructuredAssistantDecoder.swift) parse Gemma channel syntax.

Qwen uses different BPE decoding, role framing, thought tags, call grammar, result framing, stop IDs, and image IDs. Reusing Gemma's parser would be incorrect and unsafe.

The host-side VisionCapture safety checks remain authoritative. Native Qwen tool syntax changes only model serialization. It must not widen MCP permissions, accept unlisted tools, bypass target locking, replay uncertain actions, or convert malformed output into an action.

### 6.6 Vision

The current image path is explicitly Gemma-specific:

- `Gemma4ImagePreprocessor` and geometry;
- Gemma image tokens 255999, 258880, and 258882;
- Gemma vision companion format and weights;
- Gemma prompt expansion and attention behavior.

Qwen requires a separate processor and companion. The companion must be bound to both the text artifact digest and official source revision. Existing encoded-image size, image-count, and context budgets can provide outer safety limits, but token and pixel budgets need Qwen-specific calculations.

### 6.7 App, service, CLI, and server

Current defaults and labels assume one model:

- checkout path `scratch/gemma4.gturbo`;
- fixed app install descriptor;
- Gemma labels in status and error messages;
- one runner construction path in `RealInferenceClient` and decode service;
- Gemma CLI help text;
- Gemma tool-schema adaptation in the server;
- hard-coded model identity in the OpenAI adapter.

The decode protocol sends a model path but returns too little architecture identity for robust display and state checks. Model selection should be based on the verified manifest at the path, not on a client-supplied family string.

## 7. Reuse, adapt, or add

| Area | Reuse unchanged | Adapt behind a family boundary | New Qwen work |
| --- | --- | --- | --- |
| Format | v1 decoder and Gemma fixtures | version dispatch, shared file/integrity types | v2 family union and Qwen schema |
| Remote transfer | range transfer, retry, locks, resume transaction | source catalog, estimates, receipts | Qwen allowlist and tensor audit |
| Expert I/O | bounded `pread`, cache policies, RDADVISE controls | stride/slot sizing and labels | Qwen packed gate/up/down mapping |
| Quantized GEMM/GEMV | affine kernels when shape/layout matches | explicit per-tensor quant descriptors | any missing Qwen shapes/epilogues |
| Sampling | RNG, Top-K/P, repetition plumbing | family stop IDs and no Qwen softcap | presence penalty only if product scope approves it |
| Conversation product logic | epoch, turn order, checkpoints, audit | abstract state transaction and model codec | recurrent/conv state and RoPE delta |
| Chat/tool host | tool definitions and permission enforcement | family serializer/parser | Qwen template and XML-like call grammar |
| Vision host | attachment staging, hashing, fail-closed gates | family preprocessing and token budget | Qwen tower, merger, M-RoPE |
| Decode service | one-process gate and cancellation transport | return loaded model descriptor | family runner factory |
| App | transcript and lifecycle semantics | model catalog, selection, labels, paths | Qwen install and vision UI state |
| CLI/server | transport and request validation | dynamic model identity and family codec | Qwen-specific compatibility tests |

## 8. Target design

### 8.1 Model identity

Introduce a closed, strongly typed identity at the format/runtime boundary, for example:

```swift
public enum ModelFamily: String, Codable, Sendable {
    case gemma4
    case qwen36Moe
}

public struct InstalledModelDescriptor: Codable, Equatable, Sendable {
    public let family: ModelFamily
    public let displayName: String
    public let modelID: String
    public let sourceRevision: String
    public let artifactVersion: String
    public let textArtifactDigest: String
    public let capabilities: ModelCapabilities
}
```

The manifest creates this descriptor after integrity validation. The app may display it but must not manufacture it.

Capabilities describe what this installed artifact and current runtime can perform. They are not copied blindly from the model card. For the first release they should say text, still images when the companion is present, function calls, and thinking control. Video, audio, MTP, and extended context remain false.

### 8.2 `.gturbo` v2

Keep `.gturbo` v1 byte-for-byte compatible. Add a v2 manifest with:

1. `family` and exact architecture-profile ID.
2. Official model ID, official revision, source tensor-index digest, converter identity, and conversion inputs.
3. A tagged architecture payload:
   - existing Gemma payload if Gemma is ever emitted as v2;
   - Qwen hybrid layer pattern, full-attention dimensions, linear-attention dimensions, convolution size, state dtype, RoPE/M-RoPE details, MoE rules, untied head, and vocabulary.
4. Per-tensor or named-group quantization descriptors rather than assuming five Gemma buckets.
5. An explicit resident-tensor allowlist and streamed-tensor layout.
6. An explicit ignored-source-tensor list with reasons, including MTP for the first runner.
7. A vision-companion contract containing family, source revision, processor profile, text-artifact digest, and vision-artifact digest.
8. Required runtime feature flags. Unknown required features fail loading.

Do not use a minor v1 change for required Qwen fields. Older v1 decoders would not understand the architecture even if JSON decoding succeeded.

### 8.3 Runtime composition

Select variation outside hot loops:

```text
Verified manifest
  → Model family factory
      → Gemma runner + Gemma state + Gemma codec + Gemma vision
      → Qwen runner + Qwen hybrid state + Qwen codec + Qwen vision
```

Use a small session-level interface for generation, reset, checkpoint, and diagnostics. Keep each per-layer/per-token loop concrete to avoid dynamic dispatch and speculative abstraction in GPU hot paths.

A shared `ConversationStateTransaction` contract should expose observable semantics, not K/V implementation details:

- begin from committed state;
- append prefill;
- advance decode;
- commit accepted state;
- rollback active turn;
- rebuild from a checkpoint prompt;
- reset and release pages/resources;
- report retained token count and state bytes.

Gemma can implement this with its current K/V rewind rules. Qwen implements it with full-attention K/V, recurrent/conv working state, a bounded boundary snapshot, and RoPE delta.

### 8.4 Qwen weight residency

Keep resident:

- embeddings and untied LM head;
- norms and small scalar parameters;
- attention and linear-attention projections;
- routers and shared experts;
- active conversation state;
- optional vision tower according to the chosen vision residency policy.

Stream routed experts per layer from separate, aligned files. Preserve gate/up/down grouping for one expert so a cache hit supplies the complete expert. Compute cache bytes from the actual v2 stride rather than assuming Gemma's size.

The first implementation should preserve the current cache policies and expose actual bytes in diagnostics. New caching algorithms need measurements and are outside initial correctness scope.

### 8.5 Qwen chat codec

Create a Qwen codec with:

1. structural validation of the pinned tokenizer JSON and decoder;
2. resolved-and-checked special IDs rather than unverified constants;
3. golden rendering against the pinned upstream template;
4. separate visible final text, hidden thought text, and structured calls;
5. an incremental parser that cannot leak a partial call as user-visible text;
6. exact handling of multiple calls and parameter bodies;
7. a strict byte/token bound;
8. no dispatch until the complete call closes and schema validation succeeds;
9. exact Qwen tool-result framing;
10. configurable `enable_thinking`, with `preserve_thinking` off initially unless a product test proves it is needed.

Do not rename the existing `GFTokenizer` until Qwen tests exist. First introduce a family-neutral interface at its call sites, then keep the working Gemma implementation intact behind it.

### 8.6 Qwen vision path

Implement a separate Qwen path:

1. Decode image metadata and enforce existing file safety limits.
2. Compute Qwen's dynamic resize/grid plan with overflow-checked arithmetic.
3. Normalize RGB channels to the pinned mean/std.
4. Build temporal patch pairs as required by the reference for still images.
5. Run the 27-layer vision tower and merger.
6. Verify that produced feature rows exactly equal expanded `<|image_pad|>` positions.
7. Produce modality token types and 3D position IDs.
8. Retain and advance the multimodal RoPE delta during decode.
9. Bind cached image features to source digest, processor profile, and conversation lineage.
10. Fail closed on missing grids, row mismatch, companion mismatch, or unsupported media.

The upstream processor's very large pixel bounds are not automatically safe product limits. Keep TurboFieldfare's bounded image admission until representative image quality and memory measurements justify a Qwen-specific change.

### 8.7 Product model selection

Add a model catalog with stable IDs and separate locations, for example:

```text
Gemma: scratch/gemma4.gturbo
Qwen:  scratch/qwen3.6-35b-a3b.gturbo
Qwen vision companion: adjacent, separately receipt-bound
```

Exact names are an implementation choice, but no Qwen operation may overwrite Gemma's directory.

Selection rules:

1. Gemma remains the default for existing settings.
2. A Qwen choice appears only when its text artifact is verified or installable.
3. Changing selection while loaded first ends or marks the current lineage outside model context, then unloads the service, then loads the new artifact.
4. Never run both models for an interactive comparison.
5. A failed Qwen load restores a clear unloaded state and leaves Gemma selectable.
6. Model-specific defaults are displayed, not silently applied during a retained conversation.
7. Diagnostics show model ID, source revision, format version, vision status, quantization profile, context state bytes, and expert-cache bytes.

## 9. Ordered implementation packages

Each package below should land only after its listed gate. Keep changes small enough that a failed package can be reverted without disturbing Gemma.

### Package 0 — Freeze evidence and source policy

1. Save the official config, tokenizer config, generation config, preprocessor config, license, and tensor-index digests as test metadata with source URLs and revision.
2. Record the pinned Transformers reference commit used for fixtures.
3. Choose the release quantization source path.
4. If using the MLX artifact for prototyping, mark it non-release until official-revision lineage is resolved.
5. Define the exact allowed and intentionally ignored tensor sets.
6. Document that Qwen3.5 and other Qwen3.6 sizes are incompatible.

Gate: a source-policy review can trace every future output byte to an immutable input or documented deterministic transformation.

### Package 1 — Format v2 and compatibility

1. Add v2 wire types in `TurboFieldfareFormat` without modifying v1 field meaning.
2. Add version dispatch that reads the header/version before decoding the family payload.
3. Add a Qwen architecture payload with every field needed by kernels and state.
4. Add conversion-provenance and ignored-tensor records.
5. Add text/vision binding fields.
6. Add overflow, path, duplicate-name, shape, quantization, and unknown-feature validation.
7. Keep all v1 compatibility fixtures and loader tests.
8. Add hostile v2 manifests for family mismatch, missing recurrent fields, wrong layer pattern, tied head, wrong vocabulary, and companion mismatch.

Gate: all format and compatibility tests pass; an existing Gemma artifact still loads through v1 unchanged.

### Package 2 — Family-aware source catalog and repack planning

1. Replace the single source constant with a closed catalog keyed by stable product model ID.
2. Keep Gemma's descriptor values unchanged.
3. Add the exact Qwen official identity and the separately approved quantized input identity.
4. Decode Qwen config fields without Gemma fallbacks.
5. Audit every source tensor name, dtype, shape, byte range, shard, and quantization override.
6. Split `gate_up_proj` into the execution layout only if the kernel requires it; otherwise preserve the source packing.
7. Plan one complete aligned blob per routed expert and layer.
8. Place untied LM head and recurrent parameters in the resident plan.
9. Split the Qwen vision tensors into a model-bound optional companion.
10. List MTP as intentionally omitted.
11. Compute installation and scratch-space requirements from plans, not constants.
12. Preserve transactional cancel/resume/discard behavior.

Gate: synthetic Qwen headers produce a deterministic plan; one-byte, missing, extra, shape, revision, and digest changes all fail before activation.

### Package 3 — Independent numerical fixtures

1. Create tiny deterministic Qwen configurations that retain the real operation order but use small dimensions.
2. Generate reference inputs, weights, intermediate values, recurrent states, router choices, and outputs with the pinned upstream implementation.
3. Record generator code/version and fixture digests.
4. Add a simple Swift reference for each primitive where practical.
5. Include empty input, one token, chunk boundaries, awkward dimensions, repeated decode, and state carry-over.
6. Establish tolerances from measured reference error before optimizing. Record per-operation absolute/relative criteria.
7. Add negative controls that catch a missing output gate, wrong Top-8 normalization, non-FP32 decay, wrong convolution history, and wrong M-RoPE axis.

Gate: fixtures are reproducible and independent enough not to duplicate the Metal implementation's mistakes.

### Package 4 — Qwen quantized primitives and MoE

1. Verify current affine kernels for Qwen matrix shapes and group-64 layout.
2. Add shape-specialized paths only when existing kernels do not meet correctness or measured performance.
3. Implement the 256-way FP32 router softmax and deterministic Top-8 tie behavior.
4. Renormalize selected weights exactly as the reference does.
5. Implement routed expert SiLU gate/up/down execution.
6. Implement the sigmoid-gated shared expert.
7. Generalize expert cache slot bytes and alignment from the manifest.
8. Verify cache hits, misses, eviction, repeated experts, short reads, cancellation, and safe GPU lifetime.

Gate: synthetic CPU/Metal MoE results meet the predeclared tolerances under Metal API and shader validation.

### Package 5 — Full attention and hybrid state

1. Implement Qwen Q/K/V projection shapes, including the doubled Q projection carrying the output gate.
2. Add Q/K RMS normalization and partial 64-dimension RoPE.
3. Add full-attention output gating.
4. Add ten-layer full K/V storage with overflow-checked context allocation.
5. Implement linear Q/K/V/Z/A/B projections and depthwise causal convolution.
6. Implement single-token Gated DeltaNet recurrence.
7. Implement chunked prefill with exactly carried convolution and recurrent state.
8. Keep FP32 state and decay operations where required by the reference.
9. Add command-buffer ordering and bounded working/committed state.
10. Ensure cancellation never frees or reuses GPU resources still in flight.

Gate: layer-level decode and multi-chunk prefill match reference fixtures, including state after the final token.

### Package 6 — Qwen text runner

1. Add a Qwen-specific concrete runner selected from the verified v2 family.
2. Implement embedding without Gemma's embedding scale unless the Qwen reference requires it.
3. Execute the 30 linear and 10 full layers in manifest-validated order.
4. Apply Qwen norms, MoE, residuals, and separate LM head.
5. Do not apply Gemma's final-logit soft cap.
6. Integrate sampling and Qwen stop IDs.
7. Journal accepted generated tokens for bounded replay.
8. Report family-specific state and expert I/O diagnostics.
9. Keep MTP disabled and absent from execution.

Gate: tiny-model greedy-token parity, multi-token parity, chunked-prefill parity, and repeated-decode state parity all pass.

### Package 7 — Transactional conversation recovery

1. Move generation call sites from direct `KVCacheManager` assumptions to the state-transaction boundary.
2. Keep the Gemma implementation behavior unchanged.
3. Add one Qwen committed snapshot and one working state.
4. Restore on user cancellation, generation error, malformed tool output, and rejected checkpoint.
5. Rebuild accepted content after a matched multi-token stop suffix.
6. Reset both recurrent and K/V state on new chat, unload, or lineage loss.
7. Make checkpoint replacement rebuild Qwen state from the replacement prompt.
8. Include multimodal RoPE delta and retained-image lineage in snapshot/restore.
9. Disable Qwen performance-triggered compaction until calibrated; retain capacity-triggered compaction.

Gate: injected failure after every suspension/GPU completion boundary leaves the next turn equal to a clean reference run.

### Package 8 — Qwen tokenizer, chat, thinking, and tool calls

1. Add a Qwen tokenizer loader tied to the installed artifact sidecars.
2. Validate decoder, vocabulary, stop IDs, chat template identity, and image/tool/thought markers.
3. Golden-test ordinary, system, multi-turn, image, tool, and post-tool prompts against the pinned template.
4. Add Qwen incremental detokenization.
5. Add a Qwen structured decoder for thoughts, final text, and calls.
6. Parse parameter bodies without dispatching incomplete or ambiguous values.
7. Validate names and arguments against the host's supplied tool schema.
8. Generate host-owned call IDs when the model format has none.
9. Frame tool results exactly as Qwen expects.
10. Keep thinking hidden/visible according to existing product policy and selected control.
11. Keep historical-thought preservation off for the first release; add it only with context, privacy, and quality tests.
12. Replace user-facing hard-coded “Gemma” error text with the verified selected display name where the text is truly model-neutral.

Gate: malformed-call fuzz cases dispatch nothing; native round trips match template fixtures; all existing Gemma chat/tool tests still pass.

### Package 9 — Qwen still-image support

1. Add a v2 Qwen vision companion writer, verifier, and model binding.
2. Add dynamic Qwen image geometry and preprocessing with bounded dimensions.
3. Implement vision patch embedding, 27 transformer blocks, and merger/projector.
4. Add family-specific image-token expansion.
5. Add modality token types, grid metadata, three-axis interleaved M-RoPE, and decode delta.
6. Check image-feature rows against placeholder rows before model execution.
7. Preserve multi-image order and tool-result image provenance.
8. Cover retained images across turns and checkpoint rebuild.
9. Refuse image requests when the companion is absent, invalid, mismatched, or unsupported by the current loaded artifact.
10. Keep video and audio rejected with clear errors.

Gate: small deterministic vision stages match references; real still-image outputs later pass differential and qualitative checks; text-only behavior is identical with no companion installed.

### Package 10 — Decode service and IPC

1. Return an `InstalledModelDescriptor` after load.
2. Bind conversation epoch and model identity together.
3. Reject a generate or checkpoint request created for a different loaded identity.
4. Keep one decode service and one loaded model at a time.
5. Carry model-specific thinking and capability options explicitly.
6. Preserve backward decoding for optional fields where safe.
7. Add protocol tests for old client defaults, Qwen descriptor round trips, stale model identity, reset, cancel, and service invalidation.
8. Keep app UI work on the main actor and model load/generation off it.

Gate: service lifecycle tests prove no second model load, stale conversation, or mismatched response can cross selection changes.

### Package 11 — CLI and local server

1. Let `--model` select family only through the verified manifest.
2. Make CLI help and errors model-neutral.
3. Resolve tokenizer, runner, and adjacent companion from the selected artifact.
4. Return the verified model ID from `/v1/models` and response metadata.
5. Choose the matching family tool-schema serializer/parser.
6. Preserve OpenAI request validation and loopback-only binding.
7. Reject unsupported Qwen video input rather than treating it as text-only.
8. Add server tests for text, images, tool calls, thinking separation, model identity, and missing vision.

Gate: model-free protocol suites pass and a later real-model smoke suite returns valid end-of-turn and tool-call responses.

### Package 12 — App installation and selection

1. Add a model catalog and stable selected-model setting.
2. Migrate existing users to Gemma without changing their model path or defaults.
3. Add separate Qwen text and image installation status.
4. Never delete, rename, or replace Gemma during Qwen install/discard.
5. Show real sizes calculated from pinned metadata and saved partial ranges.
6. Keep install cancellation resumable and activation atomic.
7. Require unload before selection changes.
8. Mark the visible transcript outside the new model context using existing reload semantics.
9. Reset KV/recurrent lineage, gauges, retained images, and agent checkpoint state at the correct boundaries.
10. Make the HUD and About/status surfaces display verified model identity and actual memory categories.
11. Keep current Gemma sampling defaults. Present any Qwen-native profile as an explicit future option, not a silent switch.
12. Update accessibility labels and keyboard behavior for the model picker.

Gate: app state tests cover migration, selection, failed load, switch during idle/running states, missing vision, cancel/resume, and Gemma rollback.

### Package 13 — Real-model qualification

This package requires separate user authorization because it downloads and runs model artifacts.

1. Check macOS, Swift, disk, memory pressure, completed artifacts, and the repository's model-process exclusion list.
2. Build Release once through the supported workflow.
3. Run one model process at a time.
4. Differentially compare short greedy text prompts against the approved reference using the same quantized weights.
5. Compare selected logits and recurrent/full-state checkpoints, not only decoded prose.
6. Run text, multi-turn, cancellation, stop cleanup, tool round-trip, malformed tool, still-image, and checkpoint-rebuild cases.
7. Run Metal API and shader validation on supported Apple silicon.
8. Record failures, skips, hardware, OS, toolchain, command, exit status, artifact digests, and result bundles.
9. Run a relevant Release build after tests.
10. Obtain the project's required independent test and review approval for the exact candidate.

Gate: no new compiler/concurrency diagnostics, no unexpected skips, all required suites execute, and independent review passes the exact candidate digest.

### Package 14 — Rollout

1. Keep Qwen behind an opt-in feature/catalog entry.
2. Ship Gemma as the default.
3. Add a one-action rollback to select Gemma and unload Qwen.
4. Collect correctness and resource evidence without uploading private prompts or images.
5. Promote Qwen only after the pre-registered comparison decision rule is met.
6. If Qwen fails safety, state, image, or tool-call gates, disable its catalog entry without changing Gemma artifacts.

## 10. Verification matrix

| Requirement | Model-free evidence | Real-model evidence |
| --- | --- | --- |
| Exact identity | manifest/source digest rejection tests | installed receipt and reported revision |
| Gemma preservation | all v1 fixtures and existing suites | existing Gemma smoke and benchmark rerun |
| Qwen text math | tiny reference fixtures and negative controls | logit/token differential |
| Hybrid state | synthetic chunk/decode/rollback tests | multi-turn clean-vs-recovered differential |
| Expert streaming | synthetic cache and short-read tests | trace proves routed experts are read on demand |
| Chat/thinking | pinned template and parser goldens | coherent end-of-turn output in both supported modes |
| Tool safety | malformed/fuzz output dispatches nothing | VisionCapture audit shows only valid host-approved calls |
| Vision | preprocessing/grid/M-RoPE fixtures | same-image reference comparison and product workflow |
| Cancellation | injected boundary failures | cancel during prefill/decode/tool turn, then clean continuation |
| Compaction | checkpoint fixture and state rebuild | long agent task before/after checkpoint |
| IPC/model switch | stale identity and epoch tests | app switches with one service/model process |
| Memory/lifetime | bounded allocation and command completion tests | Instruments/OS measurements on target Mac |
| Metal correctness | shader compile plus deterministic fixtures | API/shader validation with real workloads |
| Performance | no numeric claim | controlled Release measurements only |

Real GPU skips do not count as passes. Compilation does not count as behavioral execution.

## 11. Future comparison protocol

The comparison has two separate questions:

1. Does each model run correctly and efficiently under the repository's frozen generation cases?
2. Which model performs the VisionCapture iOS exploration job more reliably?

Do not combine these into one “tokens per second” ranking.

### 11.1 Recover the benchmark contract first

The current checkout refers to `docs/COMMUNITY_BENCHMARKS.md`, but that file and its prompts are absent at the inspected revision. A complete prior copy exists at local Git revision `66508fbe8117b0a307cf521c6f7d01b923c3c0b5`, including:

```text
docs/COMMUNITY_BENCHMARKS.md
docs/benchmark-prompts/real-generation-v1/short-explanation.json
docs/benchmark-prompts/real-generation-v1/medium-review.json
docs/benchmark-prompts/real-generation-v1/long-synthesis.json
```

Before any benchmark, restore or explicitly version those files through a reviewed repository change. Do not copy commands from memory.

The recovered protocol requires:

- Release CLI build;
- no other model process;
- machine, OS, Swift, revision, model manifest, and prompt digests;
- power connected and Low Power Mode off;
- app defaults: temperature 0.2, Top-K 64, Top-P 0.95;
- 4,096 context and up to 1,024 generated tokens;
- seeds 20260721, 20260722, and 20260723;
- one discarded warmup per case;
- three measured cases in fresh processes;
- `stop=endOfTurn` and manual rejection of looping, repeated, or incomplete answers;
- prompt/generated token counts in every result;
- disclosure of every protocol change.

### 11.2 Model-specific benchmark deviations

Pre-register these facts:

1. Qwen's publisher defaults are not 0.2/64/0.95.
2. Qwen thinking is on by default and can consume generated tokens not shown as final text.
3. Qwen and Gemma tokenize the same prompt differently.
4. Their generated outputs route different experts and usually have different token counts.
5. The historical guide says rows are directly comparable only when case, prompt tokens, generated tokens, settings, and stop reason match.

Therefore:

- run the frozen community cases for each model as product measurements;
- keep the guide's common sampling values for that track;
- state the Qwen-native-default deviation beside every Qwen result;
- do not claim a direct decode-speed winner when token counts differ;
- report time to first token, prefill time, decode time, total wall time, tokens, output quality, peak memory, expert bytes read, and cache behavior separately;
- optionally run a second Qwen-native profile, but label it a separate experiment and never merge it into the common-settings table.

### 11.3 VisionCapture task suite

Create a versioned `visioncapture-v1` suite before either model sees it.

For every case, freeze:

1. app source revision and built product digest;
2. Simulator runtime, device type, UDID allocation procedure, locale, scale, and accessibility settings;
3. initial Simulator snapshot and test-data seed;
4. app bundle identifier;
5. VisionCapture server/gateway revision and host skill text digest;
6. exact user goal and restrictions;
7. permitted tool definitions and schema digest;
8. expected app-owned screens, required outcomes, forbidden actions, and seeded defects;
9. timeout, maximum model turns, context, and sampling profile;
10. evidence fields used for scoring.

Include tasks that cover:

- reading screen text and state;
- choosing among similar controls;
- navigation across several screens;
- safe text entry and submission;
- image evidence where accessibility facts are insufficient;
- explicit “do not use/tap” restrictions;
- a refused or unavailable action;
- recovery after an inconclusive result without replay;
- one seeded functional defect and one clean path to test false reports;
- a long task that can trigger capacity checkpointing.

Do not score from the model's final prose alone. The source of truth is the host audit: exact MCP requests, returned target/session identity, dispatch proof, observations, screenshots, checkpoint ledger, and final app state.

### 11.4 Execution order

1. Verify both artifacts and record their complete digests.
2. Confirm adequate disk and acceptable `memory_pressure -Q`.
3. Confirm the required `pgrep` command prints nothing.
4. Build the exact Release candidate once.
5. Run only one app, CLI, server, decode service, or model test at a time.
6. Restore the frozen Simulator state before every run.
7. Use the same paired seed for both models.
8. Alternate model order by pair to reduce thermal/order bias: Gemma→Qwen, then Qwen→Gemma.
9. Start a new model conversation for every case.
10. Record one discarded warmup per model/workload class if the protocol requires it.
11. Keep power and energy mode fixed and record thermal or background-work deviations.
12. Stop and mark the pair invalid if target identity, app state, gateway version, or starting snapshot differs.

Choose and pre-register the number of paired repetitions before running. Five paired seeds per task is a reasonable initial proposal for stochastic behavior, but it is provisional and may be changed before data collection based on time and model-run cost. Never change it after seeing which model is ahead.

### 11.5 Scoring

Primary product outcome:

- proportion of tasks whose required final state and evidence are both satisfied.

Hard safety gate:

- no forbidden, unlisted, wrong-target, identity-mismatched, or replayed uncertain action reaches dispatch.

Secondary quality outcomes:

- valid tool-call proportion;
- malformed call count;
- repeated/no-progress action count;
- correct handling of host refusal;
- required checks completed;
- seeded defect recall;
- clean-path false-report count;
- factual precision of the final report against host evidence;
- number of compactions and successful continuation after them.

Resource outcomes:

- total task wall time;
- time to first token and first valid action;
- prefill/decode seconds and per-model tokens/second;
- prompt and generated tokens, including hidden thinking;
- expert bytes read, cache hits/misses, and RDADVISE mode;
- process peak and steady-state memory;
- text and vision residency bytes;
- stop reason, cancellation, and error counts.

Report paired per-task values, medians, ranges, and uncertainty. Do not hide failures inside an average.

### 11.6 Decision rule

Before model execution, the product owner and implementer must set numeric acceptance thresholds from a correct baseline. No universal threshold is asserted in this plan.

The release decision must at minimum require:

1. all correctness and fail-closed gates pass;
2. zero dispatched safety violations in the registered suite;
3. no material regression in target identity or recovery behavior;
4. Qwen completes the intended image-plus-tool workflow while routed experts are actually streamed;
5. any claimed task-success improvement is supported by paired results and uncertainty, not one example;
6. resource use is acceptable on the target 32 GiB Mac under the pre-registered budget;
7. Gemma still passes and remains selectable as rollback.

If these conditions are not met, keep Gemma as default and Qwen experimental or disabled. A higher publisher benchmark score is not an override.

## 12. Known unknowns

These items remain unknown until later authorized fixture work or model execution:

1. Whether the public MLX 4-bit artifact was converted from official revision `995ad96…` exactly.
2. The final `.gturbo` Qwen text and vision installed sizes after alignment and splitting.
3. Qwen quantized-logit error and the correct numerical tolerances for this runtime.
4. End-to-end Qwen speed, memory, energy, and expert-cache behavior on this Mac.
5. Whether current affine kernels are performant for every Qwen shape.
6. The best prefill chunk size and expert-cache slot count for Qwen.
7. The safe product image pixel/token budget with adequate screenshot quality.
8. Whether historical thinking preservation improves this VisionCapture workflow enough to justify its context and privacy cost.
9. Whether Qwen's native sampling profile or the current common profile performs better in the product.
10. A model-specific threshold for performance-triggered context compaction.
11. Correctness and usefulness beyond 64K context; 262K support is trained-model metadata, not a verified host capability.
12. Video, audio, MTP, and extended RoPE support; all are outside the initial host scope.

## 13. Completion checklist

Qwen is ready for an opt-in trial only when every box below is evidenced:

- [ ] Official identity and quantized lineage are immutable and auditable.
- [ ] `.gturbo` v1 Gemma compatibility passes unchanged.
- [ ] `.gturbo` v2 rejects wrong family, revision, tensor, quantization, and companion.
- [ ] Qwen text runner matches independent reference fixtures.
- [ ] Recurrent and full-attention state survives chunking, cancellation, and rebuild.
- [ ] Routed experts stream through bounded cache storage.
- [ ] Native Qwen chat, thinking, calls, and results round-trip correctly.
- [ ] Malformed output dispatches no tool.
- [ ] Still-image preprocessing, tower, placeholder count, and M-RoPE are correct.
- [ ] Missing or mismatched image support fails closed.
- [ ] Decode service, app, CLI, and server report the verified model identity.
- [ ] App model switching never loads both models or changes Gemma's installation.
- [ ] Model-free suites, Release builds, Metal validation, and real-model suites pass.
- [ ] Frozen community cases are recorded with all deviations.
- [ ] Paired VisionCapture outcomes are scored from host evidence.
- [ ] Independent test and architecture review pass the exact candidate.
- [ ] Gemma remains the default and tested rollback path.

The next action is Package 0: resolve the quantized source lineage and freeze the exact Qwen architecture, tokenizer, processor, and tensor contracts before writing production code.

## Subsequent owner decision and source-preparation record (2026-09-15)

The owner selected the official BF16 source rather than a community quantization path. The exact pinned snapshot `995ad96eacd98c81ed38be0c5b274b04031597b0` is prepared at `scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/`. The bundle was moved there by same-volume atomic rename after preparation; it remains source-only, not converted or runnable. This subsequent record does not rewrite the historical community-prototype analysis above.
