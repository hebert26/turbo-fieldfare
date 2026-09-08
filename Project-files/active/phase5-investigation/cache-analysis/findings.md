# Phase 5 cache and prefill evidence

Recommendation: retain the current expert selection, cache replacement, on-demand image residency, and supported prefill scheduling. Existing receipts establish substantial reuse, but do not justify a cache policy or Metal implementation change. Give the growing full-attention command-buffer group priority for the next precise timing measurement.

This is a read-only continuation of the completed expert audit. Source inspected: `/Users/dev-machine/dev/turbo-fieldfare-personal`, current dirty checkout at `ea02a4a3df1a81936eb539da38e373d7037c2624`. The parent records complete source/binary provenance. Current source explains the receipts, but no complete build-source manifest proves every current file produced their binaries. No app control, model run, build, test, profiler, experiment switch, or production edit occurred here.

## 1. Expert reuse: one new defensible bound

Saved decode GPU counts allow an aggregate inference that the earlier audit did not extract. At each of 30 layers per forward, current `produceToken` records one final routed-expert command buffer. It records one additional buffer when a cached-expert phase-1 split runs. That split requires both hits and misses. See [split condition](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1749), [count recording](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1464), and [final layer drain](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1896).

For `F` completed decode forwards and `R` routed expected buffers, base layer calls `B = 30F`, observed mixed splits `M = R − B`, and all-hit calls are at most `B − M`. This is a lower bound on mixed plans because an unsuccessful optional argument-buffer allocation can suppress a split. It cannot distinguish the unsplit all-hit, all-miss, or fallback mixed cases.

| Saved receipt | Complete output rows | Layer calls | Observed mixed splits | Mixed-plan share, at least | All-hit share, at most |
|---|---:|---:|---:|---:|---:|
| Journey 27 | 147 | 366,210 | 267,840 | 73.14% | 26.86% |
| Journey 28 checkpoint | 51 | 110,940 | 85,409 | 76.99% | 23.01% |
| Screenshot diagnostic, image-result step 2 only | 1 | 660 | 543 | 82.27% | 17.73% |

All used GPU groups have `valid_buffer_count == expected_buffer_count`. Attention and shared-expert counts each equal `30F`. Three error rows and one cancelled row from Journey 27, and two error rows from Journey 28, lack the counters and are excluded. Journey 27 and Journey 28 contain no attached image inputs. The diagnostic screenshot was explicitly requested, and its 22 decode forwards are a short sample.

These are layer-call bounds, not individual expert hit percentages. Actual per-layer IDs, exact hit/miss totals, miss bytes, and route changes across navigation, recovery, and tool-result turns are absent from the saved trace schema. `cached_prompt_tokens` measures attention-state reuse, not expert reuse. [AgentInferenceTrace.finish](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Diagnostics/AgentInferenceTrace.swift:133) serializes timing and token totals only. Current `ExpertCachePlan` has the required transient `experts`, `misses` (indices into experts, not expert IDs), and `hits` fields at [PreadExpertStreamer.swift:37](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift:37).

## 2. Screenshot release and refill

The source deliberately drains GPU work before releasing streamers under `on-demand`, then prepares the image. [VisionRuntime.encodeImage](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Vision/VisionRuntime.swift:354) and [Model.prepareExpertResidencyForVision](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/Model.swift:731) provide the boundaries. Recreated streamers start with empty expert slots and zero lifetime-use counts at [PreadExpertStreamer.swift:166](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift:166). Layer verification survives. [Model.openLayerLocked](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/Model.swift:270) allocates streamers lazily as prefill fetches experts again.

The existing transition result exposes released/remaining layer counts, allocated slot bytes, GPU drain duration, and release duration, but `AgentInferenceTrace` does not emit them. `diagnosticSlotScratchBytes` reports allocated slot capacity, not occupied expert count or physical resident pages. There is no read-byte accumulator in `readFull`. An estimated logical read volume is `sum(missCount × expertStride)` and must not be reported as SSD bytes because reads can be served by the operating system cache.

The parent's passive-memory extraction found the expected service-memory dip and return around the screenshot. It cannot isolate the released component, refill duration, or instantaneous peak. The image turn's decode counters start after prefill, so the 82.27% mixed-split bound demonstrates mixed reuse after refill, not refill cost itself. Source ordering for tool-result images is [MultimodalConversation.swift:591](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Generation/MultimodalConversation.swift:591), followed by `completeEncodedTurn` at line 604.

A later measurement should retain the existing transition result and sample the service before release, after release, during image preparation, and during prefill refill. It must measure memory overlap before considering keep-ready or bounded retention. Preserving LFU history alone saves no expert-weight read if the corresponding slots were released. Any policy benefit remains unmeasured.

## 4. All-cache-hit scheduling

[Model.fetchRoutedExperts(plan:)](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift:108) always enqueues a global dispatch and resumes a continuation, even for zero misses. The streamer then executes zero parallel read iterations, briefly locks the cache, and returns existing buffers. [routedExpertBuffers(for:)](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift:93) already supplies those views synchronously.

An all-hit shortcut could therefore apply to at most 8.06 calls per decode forward in Journey 27 and 6.90 in Journey 28, out of 30. The actual number can be zero. Dispatch, continuation, and per-all-hit call time are not saved. `totalIoNanos` includes dispatch/fetch overhead and view creation, while excluding the earlier cache-plan work and RDADVISE. See [fetch timer](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1823). Shared-expert work is already in flight, so reduced host latency might overlap GPU work and fail to shorten the complete forward. Recommendation: do not implement the shortcut yet. Measure zero-miss call count and wall cost first, preserving plan side effects and buffer lifetime rules.

## 5. Reuse inside prefill batches

All three saved datasets record `prefill_chunk_tokens: 128`. They provide no 128-versus-256 comparison and no unique-expert counts. Current selectable sizes are 32, 64, 128, 256 at [PrefillRuntimeConfig.swift:229](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Prefill/PrefillRuntimeConfig.swift:229).

[PrefillMoEGrouping.groupTokenExpertPairs](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Prefill/MoE/PrefillMoEGrouping.swift:91) retains all `8T` token/expert pairs, groups each unique expert once, and exposes `groups.count`, `perExpertCounts`, tile occupancy, and maximum pairs per expert. For a `T`-token chunk, unique experts `U` lie between 8 and `min(128,8T)`. A cold layer fetch needs `U × expertStride` logical bytes, not `8T × expertStride`, because repeated pairs share weights. Later chunks depend on retained slots and replacement order. These are calculations, not workload measurements.

[fetchBindingForTile](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift:240) already returns `plannedHits`, `plannedMissIndices`, and slot assignments. [encodeStreamedBatched](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift:387) returns the microbatch count, currently discarded at [RealForwardRunner.swift:1298](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1298). These are precise future observation points.

Two subtleties matter. LFU increments once per expert request, so grouped prefill counts an expert once per chunk even if many token rows use it. Decode counts it once per token. Also, current image turns use one scratch layout at least 280 tokens wide, while text-only scratch scales with the chosen chunk size. The claim that increasing 128 to 256 is universally memory-free is too broad. The KV ring and image-turn scratch already cover 280, but text-only scratch can grow. See [scratch selection](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:630) and [scratch byte formula](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Prefill/PrefillChunkScratch.swift:47). The source comment about a historical 11,612-token prompt reading 578 GB with no hits is not a current measurement receipt.

Historical [CACHE-03–08](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/summaries/03-expert-cache-prediction-and-layout.md:45) found memory-pressure and holdout failures for larger caches, trace-trained allocation, predictors, and layout changes. Historical [PF-03, PF-08–10](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/summaries/06-prefill.md:46) rejected deeper lookahead, argument-buffer rings, and extra overlap after weak or negative complete-run results. Existing grouped prefill should remain the baseline.

## 6. Metal priorities and limits

The timing agent's saved-run analysis identifies growth in the full-attention/router group while routed-expert time stays broadly stable. Current [command-buffer contents](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1651) include normalization, QKV projection, positional work, attention, output projection, post-attention setup, and the router. Its total is not an attention-only measurement. Router wait also includes prior queued work. Adding these GPU and host timing groups would double count overlap.

The exact full-attention observation path is [gAttention](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1579) → [Attention.encodeFull](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Attention/Attention.swift:196) → [encodeSplit](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Attention/Attention.swift:228) → `attention_decode_partial` and `attention_decode_combine` in [attention.metal](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Metal/Attention/attention.metal:135). Full decode uses 16 sequence chunks, 16 query heads, 2 KV heads, and head dimension 512. Its active sequence grows with conversation length. The TensorOps prefill path handles a different, multiple-query shape.

The next measurement should isolate the two full-attention stages from their surrounding projection/router work at several actual context lengths. Only then choose a shader candidate. Historical [KV-03–06](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/summaries/05-attention-and-kv-cache.md:50) rejected full-GQA variants on long-context scaling and MLX-style geometry on the shipping checkpoint's quality gate. Those candidates should not be revived from group timing alone. No measured path to 25 generated tokens/second follows from these receipts.

## Reproduction and limitations

Run from the task workspace:

```sh
python3 phase5-investigation/cache-analysis/extract_cache_bounds.py
```

Exit code 0. The extractor reads the timing agent's immutable snapshots and writes only this folder. [cache-bounds.json](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/cache-analysis/cache-bounds.json) records source paths, snapshot paths, byte counts, SHA-256 hashes, exact source lines, count checks, and excluded rows. [cache-bounds.csv](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/cache-analysis/cache-bounds.csv) is the per-output table. No malformed snapshot lines occurred.

Source review used `rg -n`, `rg --files`, and `nl -ba … | sed -n …` on the files linked above, plus the earlier expert report and experiment summaries. Searches of saved JSONL and current source found no persisted expert-ID, exact-hit/miss, or prefill-occupancy fields. Generic VisionCapture/CFNetwork `cache_hit` entries refer to other caches and were excluded.

These are navigation/diagnostic receipts, not community benchmark results. They use 64K maximum context, 32 expert slots, Thinking ON, prefill 128, and RDADVISE OFF. They have variable prompts and output counts, no matched warmup/three-case sequence, and tool-call endings. Current source and saved binary provenance are distinct. All-hit wall cost, true per-layer reuse, cache-refill duration, prefill batch comparisons, and individual Metal stage cost remain unmeasured.

Keep current expert behavior. Measure full-attention stages and screenshot release/refill before choosing a production change.
