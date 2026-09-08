# Phase 5: Gemma navigation, expert reuse and measured delay

> **Follow-on investigation, 7 September 2026:** [Google training, native format and current implementation](google-deep-investigation-20260907/report-source.md) and its [claim ledger](google-deep-investigation-20260907/claim-ledger.md) are saved beside this report. They include frozen-source parser findings, a native thinking-prefix mismatch and the limits of concise-packet evidence. Concurrent source edits are recorded separately and were not made by this investigation. The original Phase 5 measurements below retain their original scope.

7 September 2026 · Investigation only · Google documentation, current source, saved journeys and passive memory samples.

**Recommendation: keep the current expert cache and Metal implementation. First resolve the conversation-history contract, then measure selected-expert reuse and screenshot refill.** Saved evidence shows useful cache reuse. It also shows that growing full-attention work and time inside tool requests are substantial costs. No measured path to 25 generated tokens/second is established.

Navigation needs the experts selected for its actual tokens and screen state. Useful selected weights can remain in the existing cache, which already avoids reading weights on cache hits. The unanswered question is which layer/expert pairs recur often enough to retain within the memory budget. Published material does not disclose semantic expert assignments. Actual route traces are needed to answer that question.

## Understand the model before choosing changes

Primary sources below were checked on 7 September 2026. Findings about other models are explicitly identified. Public `main` artifacts can change.

| Documented finding and evidence type | Source and version | Consequence for this runtime/input | Uncertainty | Smallest useful measurement |
|---|---|---|---|---|
| **Training disclosure:** broad web, code and image pretraining, January 2025 cutoff. Posttraining follows Gemma 3 broadly and adds thinking. | [Gemma 4 technical report](https://arxiv.org/html/2607.02770v2#S2.SS4), §§2.4–3, v2, 24 July 2026 | Ground decisions in current screen and action evidence. | Exact 26B training recipe, iOS datasets and accessibility-tree exposure are undisclosed. | Classify observed navigation errors without assuming their training cause. |
| **Architecture:** 30 layers, eight selected from 128 experts per layer, plus the shared expert. Twenty-five attention layers use a local window and five use full context. | [Google 26B configuration](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/config.json), `main`, checked 7 September; [reference routing code](https://raw.githubusercontent.com/google-deepmind/gemma/main/gemma/gm/nn/gemma4/_moe.py), `MoE.__call__` / `_router` | Record layer plus expert ID. Retained copies can preserve all selected contributions. | No published navigation expert map or local reuse distribution. | Observe eight IDs, hits and misses at each layer/token. |
| **Capability claim:** screen/UI understanding and native function calling. **Benchmark evidence:** 26B InfographicVQA scores 89.3 with 1120 image tokens and 77.8 with 280. | [Model card](https://ai.google.dev/gemma/docs/core/model_card_4#core_capabilities), updated 30 July; [technical report](https://arxiv.org/html/2607.02770v2#A1), Tables 6/12 | Evaluate whether screenshots preserve the details needed for navigation. | These scores do not establish iOS-testing competence. | Read labels and selection state on saved ambiguous screens, then measure decision accuracy. |
| **Image guidance:** supported budgets are 70/140/280/560/1120. Higher detail costs more computation. Images before text are recommended. | [Model card](https://ai.google.dev/gemma/docs/core/model_card_4#best_practices), updated 30 July | Our current vision implementation caps pooled tokens at 280. Examine layout and detail before increasing image use. | Correct reading of three large labels does not prove small-text accuracy. | Record actual pooled tokens, errors, full turn time and memory for the same screens. |
| **Prompt contract:** native system/tool markers, thinking retained within a tool loop, removed before the next standard user turn. | [Prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4#managing_thought_context_between_turns), updated 3 June | Check the tokens retained in attention state, not just displayed text. | Effect of a local contract mismatch on navigation is unmeasured. | Compare retained history at one tool boundary and one new-user boundary. |
| **Tool contract:** the host validates and executes generated calls, then supplies their results. | [Function-calling guide](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4#full_function_calling_sequence), updated 4 June | VisionCapture remains the source for action outcomes. Concise inputs must retain usable choices, uncertainty and recovery rules. | A generated call or HTTP success does not prove a screen transition. | Correlate model decisions with existing canonical action receipts. |
| **Sampling guidance:** temperature 1.0, Top-P 0.95, Top-K 64. | [Model card](https://ai.google.dev/gemma/docs/core/model_card_4#best_practices), updated 30 July | Local temperature is 0.2. Keep it as the measured baseline. | Neither setting is proved best for this workflow. | Later compare successful-action rate and total latency on matched tasks. |
| **Other-model research:** Mixtral reports weak topic separation and stronger adjacent-token locality. MoE-Infinity studies trace-based caching. | [Mixtral](https://arxiv.org/html/2401.04088v1#S5), §5, 8 January 2024; [MoE-Infinity](https://arxiv.org/abs/2401.14361v3), v3, 12 March 2025 | Reuse-distance analysis and offline cache replay are sensible hypotheses. | Neither proves Gemma/Apple Silicon gains. | Replay unchanged selected routes against candidate capacities. |
| **Drafter limitation:** Google warns that 26B verification may activate extra experts and give no speedup for one request. | [Google MTP overview](https://ai.google.dev/gemma/docs/mtp/overview), updated 5 May 2026 | Keep the 26B drafter as a research lead only. | Extra memory, acceptance rate and verification cost are unknown here. | Establish present costs first. No drafter download or implementation. |

The report describes a dedicated roughly 550M-parameter vision encoder for 26B. Encoder-free 12B results and smaller-model encoder improvements do not transfer to it. Google's memory table uses different quantization, attention-state precision, context and modality assumptions from this installation. The earlier Gemma 3 recipe supplies background on distillation and reinforcement learning, not an exact Gemma 4 recipe. [Gemma 4 report, §§2.1/2.5](https://arxiv.org/html/2607.02770v2#S2.SS1), [Gemma 3 report, §§2.2/3](https://arxiv.org/html/2503.19786v1#S3).

### Installed input contract: two findings before optimization

**Old thoughts persist across a new user turn.** In current [MultimodalConversation.sendToolUser](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Generation/MultimodalConversation.swift:460), the existing conversation appends a user suffix to retained attention state. `completeEncodedTurn` retains committed generated token IDs, including thought tokens. Display filtering does not remove them from that state. The saved diagnostic's `Ready.` generation records 60 thinking tokens and 1,639 conversation tokens. The next distinct user request reuses all 1,639. This conflicts with Google's new-user-boundary guidance. Preservation inside the same tool loop is appropriate. Navigation impact and the cost of rebuilding history remain unmeasured. No fix was made.

**Image order needs a contract comparison.** [Tokenizer.encodeToolChat](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Tokenization/Tokenizer.swift:385) creates text then image parts. The installed template emits the response text before image placeholders. Google's [current template](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja), published 9 July, also does this for tool-result arrays despite the general image-first recommendation. Simply reordering the input array would not change that template's output. Compare valid native layouts before selecting a change.

The installed template SHA-256 is `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0`. It differs from Google's current template in thought gating and turn-continuation handling. This is artifact drift, not evidence that swapping templates is safe. The installed pack comes from pinned `mlx-community/gemma-4-26b-a4b-it-4bit` revision `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`, with 4-bit routed/shared weights and an 8-bit router. Its key architecture fields agree with Google's current 26B config. [Installed metadata](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/baseline/installed-template-metadata.json), [pinned source](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Remote/SupportedModelSource.swift:3).

## Measurement baseline and provenance

The investigated source is `/Users/dev-machine/dev/turbo-fieldfare-personal`, commit `ea02a4a3df1a81936eb539da38e373d7037c2624`, **plus 32 tracked modifications and 475 untracked files**. This task workspace instead starts at `614a8f69eda15b7f61ff2024b50de931871b3e30`. It was not used as the runtime source.

The baseline records 265 relevant source/build-script hashes and the complete tracked patch, SHA-256 `acdf3f6d1399219fe8685420f630b2c9bb9b56872a49502a1d1b8e97a93a046d`. All 14 installed Metal source resources match their current source counterparts. There is no complete historical build-source manifest, so binary identity does not prove every current Swift file produced it. The final check found all source hashes, the tracked patch and working-tree status unchanged. [Source inventory and command receipts](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/baseline/commands.json), [source hashes](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/baseline/source-files.json), [end check](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/baseline/end-check.json).

| Binary receipt | App SHA-256 | Decode-service SHA-256 |
|---|---|---|
| Journey 27 | `226a6c79d9204984557f5a0ae5c6eb2a51e2f404406e5b9b53ccbcc106178319` | `62ddec19992bc493f5759795d3f3aa4a57d71e8873b522266e78596763bcd924` |
| Screenshot diagnostic, Journey 28, and current installed files | `9a93e52fbe949be00d1269922b569bb718a89bd6ea9eac763a3795d43a3f51ef` | `a5bc244a8113d0874f9163a5783545c2c51c521476038fe08089502876c9bf4f` |

Hardware: M2 Pro Mac mini, 12 CPU cores, 32 GiB RAM, macOS 26.5.1 (25F80), Swift 6.3.2. Current checks show AC power, Low Power Mode off, and 129 GiB free disk. The installed text manifest hash is `1cb53c2423f05dfa673e5f0d9a3407aa355b8227b621f8f8e575830a2bb7fa91`; companion manifest hash is `6668c699d7a613d90c3e16924784808ae4372211b1758e89ee8115515fdaab77`. All 36 text-manifest file sizes match. Weight contents were not rehashed or copied.

Saved settings are 65,536 maximum context, 32 expert slots, Thinking ON, LFU, prefill 128, on-demand vision, RDADVISE off, temperature 0.2, Top-K 64 and Top-P 0.95. Current saved preferences agree. The highest completed context observed was **56,365**, not a filled 65,536-token context.

## 1. Actual expert reuse: aggregate evidence, identities still missing

Existing GPU buffer counts reveal mixed cache hits and misses. For `F` decode forwards, base layer calls `B = 30F`. Routed buffer count `R` includes an extra buffer for each successful cached/missing split, so `M = R − B` is a lower bound on mixed plans. All-hit calls are at most `B − M`. Optional buffer-allocation fallback prevents treating the mixed bound as an exact frequency. [Source condition](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1749).

| Population | Layer calls | Mixed plans, at least | All-hit plans, at most |
|---|---:|---:|---:|
| Journey 27, 147 outputs | 366,210 | 73.14% | 26.86% |
| Journey 28 checkpoint, 51 outputs | 110,940 | 76.99% | 23.01% |
| Image-result generation, 23 tokens | 660 | 82.27% | 17.73% |

Every used GPU group has complete valid/expected counts. These are **layer-call bounds, not individual-expert hit rates**. Neither navigation journey contains an image. Recovery and tool-result inputs are present, but the trace has no selected IDs, exact misses, read bytes or per-layer distribution. `cached_prompt_tokens` measures attention-state reuse. [Reproducible bounds and exclusions](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/cache-analysis/cache-bounds.json).

The precise future observation point is [ExpertCachePlan](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift:37), joined to layer/token/turn and decode versus prefill. Its `misses` values index the selected-expert array. Logical requested bytes can be calculated from misses, but cannot be called SSD bytes because macOS may serve reads from memory.

## 2. Screenshot cache release and refill: observed memory transition

The one explicitly requested screenshot produced a correct description of three bottom-navigation labels. Its complete user turn took **16.121 seconds**: 9.055 seconds of prefill, 2.991 seconds of generation, a 0.983-second screenshot-tool gap, and 3.092 seconds not separately attributed. The image-result generation itself took 11.309 seconds, including 6.483 seconds prefill and 1.770 seconds generation. The residual is not a measured image-encoder or refill duration. [Diagnostic receipt](/Users/dev-machine/dev/VisionOS/Project-files/active/cache-layout-recovery/evidence/gemma-journey27-20260907/chat-image-diagnostic/receipt.md:34), [exact step rows](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/journey-analysis/image_diagnostic.steps.csv).

Service footprint samples around the image input were **4,809 → 1,985 → 4,813 MiB** at 16:06:31, 16:06:42 and 16:06:52 London time. The whole 22-sample capture peaked at 4,826 MiB, after the answer. This supports a release/refill transition but cannot isolate its components or catch a brief peak. [Memory samples](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/memory-summary.json).

[Model.prepareExpertResidencyForVision](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/Model.swift:731) deliberately releases streamers after GPU work drains. Weights and LFU frequency history disappear, while layer verification survives. Existing transition/vision timing values are not persisted in the journey trace. **Keep on-demand residency.** Retention benefit and peak overlap are unmeasured. Keeping frequency metadata alone would not avoid the initial reads of released weights.

## 3. Memory at 64K/32 slots/Thinking ON

| Resource | Calculated capacity or measured footprint |
|---|---:|
| Expert slots, `3,358,720 × 32 × 30` bytes | **3.003 GiB capacity** |
| FP16 attention-state buffers at 65,536 maximum context | **1.499 GiB capacity** |
| Common weights file including index | 1.261 GiB on disk |
| All routed experts | 12.012 GiB on disk |
| Journey 27 sampled service maximum | **4,830 MiB footprint** |
| Journey 28 sampled service maximum | **4,815 MiB footprint** |
| Current idle service, three samples | **4,813 MiB footprint** |

The attention-state calculation uses 1,304-row local rings and separate K/V buffers: `25×1304×8×256×2×2 + 5×65536×2×512×2×2` bytes. Shared projection weights do not make post-normalization K/V values identical. [Allocation source](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/KVCache/KVCacheManager.swift:32), [calculations](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/baseline/capacity-calculations.json).

Current passive sampling ran **17:54:37–17:54:58 UTC**, observing the existing app PID 30726 and service PID 30791. Service CPU stayed 0.0%. `memory_pressure -Q` reported 40% system-wide free percentage before/after. Swap usage stayed at 11,343.50 MiB, with 44 system swap-in pages and zero new swap-out pages. Page size was 16 KiB. `top MEM` is process footprint, not all physically resident pages. `ps` reported service RSS of 9,360 KiB at the start, illustrating the distinction. These idle observations do not establish active headroom or attribute swap to this service. [Raw samples and exact commands](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/passive-memory/commands.json).

The saved Journey 27/28 capture windows also show system swap-outs, without process attribution. Ten-second sampling can miss peaks, and the trace omits filled expert-slot counts. The 64K setting was measured, but full 64K occupancy and screenshot peak headroom remain unmeasured.

## 4. All-hit scheduling: possible shortcut, benefit unmeasured

[Model.fetchRoutedExperts(plan:)](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift:108) dispatches asynchronously even with zero misses. A direct-buffer return is a precise candidate, but the saved bound permits at most 8.06 eligible calls per forward in Journey 27 and 6.90 in Journey 28. Actual frequency can be lower, including zero. Per-call scheduling time is unavailable and can overlap GPU work. **Do not implement this shortcut yet.** First measure zero-miss count and dispatch-to-return time while preserving plan side effects and buffer lifetimes.

## 5. Prefill batches: existing grouping already avoids duplicate loads

[PrefillMoEGrouping.groupTokenExpertPairs](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Prefill/MoE/PrefillMoEGrouping.swift:91) groups all `8T` selected token/expert pairs by their `U` unique experts. A cold layer needs `U` expert fetches rather than `8T`, with `8 ≤ U ≤ min(128,8T)`. This describes existing behavior. Actual unique counts and repeated reads across chunks are absent from these traces.

Supported batches are 32/64/128/256. Every saved sample used 128. LFU counts one grouped request per expert/chunk, so its frequency meaning differs from decode's per-token requests. A 256-token batch can grow text scratch even though image scratch and local rings already accommodate 280. Observe `groups.count`, `perExpertCounts`, tile hits/misses and returned microbatch counts in [PrefillGroupedRoutedMoE](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift:240) before comparing supported batches. **Keep 128 as the baseline.**

Historical [cache experiments](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/summaries/03-expert-cache-prediction-and-layout.md:45) found memory-pressure and unseen-workload failures. Historical [prefill experiments](/Users/dev-machine/dev/turbo-fieldfare-personal/docs/experiments/summaries/06-prefill.md:46) rejected deeper scheduling and extra overlap. Their results inform caution, not present-machine speed estimates.

## 6. Remaining delay: full attention and tool time matter

| Saved workload | Generated tokens / generation seconds | Generated tokens/s | Prefill seconds | Full recorded interval |
|---|---:|---:|---:|---:|
| Journey 27, 147 completed outputs | 12,354 / 1,062.733 | **11.625** | 965.730 | 3,970.883 s, interrupted |
| Journey 28, 51 completed outputs | 3,749 / 234.757 | **15.970** | 280.665 | 1,313.663 s, unfinished checkpoint |

Totals exclude error generations. Complete raw diagnostics and errors are retained in [immutable snapshots and derived rows](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/journey-analysis/trace-summary.json). Generated tokens include thinking and tool syntax. Journey 27 classified 3,134 thinking tokens plus 3,559 unknown-hidden-channel tokens, which must not all be relabelled as thinking. Journey 28 classified 2,266 thinking tokens. Separate thinking wall time is unavailable.

Journey 27's completed-output-to-next-input gaps total 1,841.224 seconds. Journey 28's total 775.004 seconds. These are host/tool intervals. One Journey 28 gap of 50.690 seconds contains 50.487 seconds of sequential VisionCapture inspect/describe/tap requests, leaving 0.203 seconds. This delay cannot be removed by an expert kernel change. Required action validation still applies. [Correlated server records](/Users/dev-machine/dev/turbo-fieldfare-personal/Project-files/active/phase5-investigation/journey-analysis/journey28-http-events.jsonl).

| Journey 27 completed context | Outputs | Generated tokens/s | Full-attention/router group, ms/forward | Routed experts, ms/forward | Fetch aggregate, ms/forward |
|---|---:|---:|---:|---:|---:|
| Below 8,192 | 27 | 17.323 | 6.789 | 7.025 | 19.225 |
| 8,192–32,767 | 59 | 12.772 | 27.279 | 7.333 | 17.552 |
| 32,768–49,151 | 41 | 10.190 | 51.478 | 7.056 | 15.718 |
| 49,152–56,365 | 20 | 8.857 | 65.456 | 6.912 | 15.315 |

These are observed populations with changing prompts, not a controlled context experiment. The growing full-attention group is the strongest next GPU measurement target. It includes projections, normalization and routing as well as attention. Host fetch/router-wait counters overlap GPU work and cannot be added as separate costs. At long context this group alone exceeded the proposed 40 ms/token total budget.

Exact candidate observation path: [RealForwardRunner gAttention](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1579) → [Attention.encodeFull / encodeSplit](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Attention/Attention.swift:196) → `attention_decode_partial` / `attention_decode_combine` in [attention.metal](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Metal/Attention/attention.metal:135). Measure those stages separately before selecting any shader change. Current evidence justifies **no specific Metal patch**.

## Commands, deviations and remaining work

All new files are confined to this `phase5-investigation` folder. Reproduction commands from this task workspace, each exit 0:

```sh
python3 phase5-investigation/capture_baseline.py
python3 phase5-investigation/passive_memory.py
python3 phase5-investigation/analyze_memory.py
python3 phase5-investigation/journey-analysis/analyze_saved_traces.py
python3 phase5-investigation/cache-analysis/extract_cache_bounds.py
```

The first two collect a fresh baseline if rerun. The journey extractor reuses its saved snapshots. Current process/memory reads needed broader sandbox access. Initial `pgrep` failed with exit 3, “Cannot get process list”; the broader read succeeded with exit 0 and confirmed the two existing PIDs. Initial sandboxed hardware `sysctl` exited 1; the passive collector succeeded. Full stdout/stderr and per-command exits are saved. Source searches used `rg`, with line reads using `nl`/`sed`.

Historical receipts record `./script/build_and_run.sh --verify`, exit 0, and `open -n --env TURBOFIELDFARE_AGENT_TRACE_PATH=/tmp/gemma-journey27-20260907.jsonl /Applications/TurboFieldfare.app` or the Journey 28 trace filename. Those commands were **not run here**. The existing app/service remained untouched. No build, test, new model process, profiler, experimental switch, deployment, commit, source edit, model duplication or proposal edit occurred.

These observations are **not community benchmark results**. They use app/tool workloads, the Application Support model path, 64K maximum context, changing prompts, unequal generations, different historical builds, tool-call stops, interrupted work and background captures. The initial diagnostic `Ready.` output overlapped a VisionCapture build. There were no frozen three-case prompts, discarded warmups, fixed-seed fresh processes or controlled repeat runs. Therefore no community timing footer exists for this investigation. Full app diagnostics replace it. Both checkouts lack `scratch/gemma4.gturbo`, and the existing model processes fail the fresh-run exclusivity precondition.

The next authorized measurement should first verify the new-user history boundary and native image layout, then record bounded per-layer routes and cache outcomes across text, recovery, tool-result and screenshot turns. Retain existing vision-transition timings, sample memory during release/refill, and isolate full-attention stages. Those measurements require a coordinated session after the owner finishes with the current model and a separately approved temporary observation build for counters that are not exported today. Device navigation, if needed, requires Terra coordination.

**The evidence report is complete. Per-layer reuse, exact refill cost, full-64K active memory, all-hit scheduling cost and batch comparisons remain open. Keep current runtime behavior until those measurements support a specific change.**
