# Saved journey measurements for Phase 5

The saved evidence supports an investigation of growing full-attention cost and long tool requests. It does not establish a useful fixed expert set, a 25 tokens/second path, or a measured benefit from retaining expert caches through screenshots.

## Evidence and baseline

- Journey27: archived `gemma-journey27-20260907/model-trace.jsonl`, 313 lines, 586,315 bytes, SHA-256 `e61087be0e4cd85047cffd8eafd90a9a3326bf1a236e609441200bdad4647f60`. Coordinator stopped it at step 150. This is an interrupted journey.
- Journey28: complete-record snapshot of `/tmp/gemma-journey28-20260907.jsonl` captured 2026-09-07 17:55:51 UTC, 110 lines, 206,761 bytes, SHA-256 `b95f096817f791533943c26c6e70fd71c6195318f2b41d07edca8eda087c02d1`. Last record is output 52 at Unix 1788796510.153469. The snapshot has no natural final answer or cancellation record. The last proposed call has no subsequent input in this snapshot. Do not label it a completed journey.
- Screenshot: Journey27's separate `chat-image-diagnostic/model-trace-with-observation.jsonl`, 11 lines, 30,117 bytes, SHA-256 `2d98ef9752998ced950e7d9fe4b4f9faad71394d6f6a16d1055747cb23357501`. Its five outputs cover three separate user requests. The long gaps between those requests are human idle time and are excluded from tool latency.
- All three trace settings specify 65,536 maximum context tokens, 32 expert slots, LFU, Thinking ON, 128-token prefill, prefill enabled, on-demand vision, RDADVISE off, temperature 0.20000000298023224, Top-K 64, Top-P 0.949999988079071, repetition penalty 1, maximum new tokens 65,536, and full-SHA256 verification. This is configured capacity. Highest measured completed context is 56,365 tokens in Journey27, 17,710 in Journey28, and 3,009 in the extended diagnostic.
- Journey27 `run-setup.json` records commit `ea02a4a3df1a81936eb539da38e373d7037c2624` plus existing changes, M2 Pro Mac mini/32 GB, macOS 26.5.1 (25F80), Swift 6.3.2. App/helper hashes are `226a6c79d9204984557f5a0ae5c6eb2a51e2f404406e5b9b53ccbcc106178319` / `62ddec19992bc493f5759795d3f3aa4a57d71e8873b522266e78596763bcd924`. Diagnostic and Journey28 receipts instead record `9a93e52fbe949be00d1269922b569bb718a89bd6ea9eac763a3795d43a3f51ef` / `a5bc244a8113d0874f9163a5783545c2c51c521476038fe08089502876c9bf4f`. These are different installed builds. These historical hashes are receipt evidence, not current-binary rehashes.
- Historical launch commands are `open -n --env TURBOFIELDFARE_AGENT_TRACE_PATH=/tmp/gemma-journey27-20260907.jsonl /Applications/TurboFieldfare.app` and the corresponding Journey28 filename. Deployment receipts record `./script/build_and_run.sh --verify`, exit 0. No launch, build, navigation, model execution, test, profiler, or production modification was performed for this extraction.

Existing receipt locations are under `/Users/dev-machine/dev/VisionOS/Project-files/active/cache-layout-recovery/evidence/`. Journey27 closure lines 7–20 and 38–40 establish its population and historical provenance. Diagnostic receipt lines 7–12, 16–24 and 34–48 establish its build and scope. Journey28's `run-setup.json` explicitly distinguishes directly read facts from coordinator-reported provenance.

## Generation and complete elapsed time

| Population | Completed outputs | Generated tokens | Decode seconds | Weighted tokens/second | Prefill seconds | Complete trace span |
|---|---:|---:|---:|---:|---:|---:|
| Journey27 | 147 | 12,354 | 1,062.733472 | 11.624740 | 965.730050 | 3,970.883138 s to Stop |
| Journey28 snapshot | 51 | 3,749 | 234.757341 | 15.969682 | 280.664904 | 1,313.662650 s to output 52 |

Weighted throughput is `sum(output.diagnostics.generated_tokens) / sum(output.diagnostics.decode_seconds)`. Progress events are excluded to avoid counting the same tokens twice. Error generations are separate: Journey27 has 232 tokens, 22.994068 decode seconds and 24.596847 prefill seconds across three errors. Journey28 has 120 tokens, 6.850970 decode seconds and 7.890457 prefill seconds across two errors. Cancelled Journey27 step 150 lacks decode/prefill/token diagnostics.

For Journey27, completed input-to-output intervals total 2,069.168445 s. Completed-output-to-next-input gaps total 1,841.224052 s across 147 gaps, median 8.476982 s, maximum 70.584870 s. Those gaps account for 46.37% of the trace duration. Decode including errors accounts for 27.34%, and prefill including errors for 24.94%. The balance includes generation bookkeeping, cancellation and short correction transitions. The trace gaps include host and tool execution. They are not SSD or GPU waits.

Journey28 has 775.004124 s across 50 completed-output-to-next-input gaps, median 8.240587 s and maximum 50.689892 s. Those gaps account for 59.00% of the bounded trace span. Different prompt lengths, different installed builds, an unfinished run and shorter context prevent a controlled Journey27/28 speed comparison.

Generated tokens include Thinking and tool syntax. Journey27's recorded completed `structured_progress` has 3,134 thinking tokens and 3,559 tokens classified as `unknown_hidden_channel_tokens`. Journey28 has 2,266 thinking tokens and zero unknown-hidden tokens. Do not relabel all unclassified tokens as thinking or interpret generated-token throughput as visible answer speed. No separate thinking wall-time counter exists in these traces. The tiny reported `time_to_first_token_seconds` excludes the much larger prefill/image preparation time evident from timestamps and must not be quoted as complete user-visible response latency.

## Actual screenshot turn

The two navigation journeys contain zero incoming images and zero screenshot proposals in these snapshots. Journey27 closure additionally reports zero screenshot requests in its correlated VisionCapture log. The explicit screenshot diagnostic is the sole image sample here.

Extended diagnostic lines 4–7, steps 1–2, cover one complete screenshot request and answer:

| Measured segment | Seconds |
|---|---:|
| User input to screenshot-call output | 3.828503 |
| Screenshot-call output to image tool-result input | 0.983032 |
| Image tool-result input to final answer | 11.309499 |
| Complete user turn | 16.121034 |
| Sum of prefill counters across both generations | 9.054826 |
| Sum of decode counters across both generations | 2.990724 |
| Remaining time after those counters and the tool gap | 3.092452 |

Step 1 generated 16 tokens / 1.220351 decode seconds = 13.110982 tokens/s. Step 2 generated 23 / 1.770373 = 12.991613 tokens/s, after 6.482741 s of prefill. Step 2 computed 381 new prompt tokens and reused 1,732 cached prompt tokens, finishing at context 2,135. Line 6 records one 359,042-byte image, SHA-256 `76fb9ad93af2dc04bd3f1396b901b7e2dc4e8a5a225be9100877e5bffe6f7199`. The output reports `endOfTurn` with no new call. Its actual image answer agrees with the receipt's separately observed screen.

The post-image generation has 3.056416 s between `elapsed_seconds` and its prefill+decode counters. This residual is not a measured vision-encoder duration or cache-refill duration. It can include image preparation, interprocess transfer, cache release/reconstruction and bookkeeping. The trace contains no cache-release timestamps, filled-slot counters, LFU history snapshots, per-layer hits/misses or read-byte series. Its I/O aggregate changes from 28.488797 to 38.570481 ms/forward between the short pre-image and post-image generations, but the 15 versus 22 forwards, different prompts and image work prevent attributing that difference specifically to cache refill.

Diagnostic receipt line 36 reports sampled charged helper memory peaking at 4,826 MiB and app memory 144 MiB over 22 ten-second samples. Those sparse samples cannot prove the transient encoding peak or absence of memory pressure. Parent investigation owns memory extraction. The initial `Ready.` generation overlapped a VisionCapture build and is explicitly unsuitable for performance comparison.

## Recovery and tool timing

Journeys record proposed calls separately from VisionCapture outcomes. A successful HTTP request or an emitted tap is not proof of a successful app transition. Journey27 has 39 tool inputs whose outcome is `inconclusive`, two `submitted_once_reobserved`, two `not_sent`, and one `failed_reobserved`. The repeated opaque-button attempts are not evidence of expert specialization or useful repeated navigation.

Useful bounded recovery examples in `journey27.snapshot.jsonl`:

- Malformed output 17, line 39, used 5.511113 s. The subsequent valid decision 18, line 41, used 4.544785 s, after a 0.006059 s correction gap.
- Malformed output 82, line 175, used 23.085461 s. Decision 83, line 177, used 17.716479 s, after 0.020790 s.
- Malformed output 85, line 181, used 19.089249 s. Decision 86, line 183, used 15.763087 s, after 0.015801 s.
- Input 59, line 126, carries `failed_reobserved` after the prior tap's 50.178276 s tool gap. The next decision takes 25.236676 s from input to output and proposes type. The trace's `mcp_failure` at line 125 contains the delivered tap contradiction. This is recovery evidence, not a new success claim.

Journey27's prior `mcp-latency-checkpoint.md` lines 7–13 already correlates steps 24/27/31 with VisionCapture request durations: 33.291375 s for profile typing, 28.174192 s for a tab tap, and 24.020940 s for a tag-picker tap. Its source contract review rejects simply deleting validation reads.

New read-only correlation of Journey28 demonstrates the same external delay pattern:

| Output step → next input | Complete gap | Sequential server durations | Sum / remainder |
|---|---:|---|---:|
| 29 → 30 | 47.005661 s | inspect 11.092, describe 3.232, inspect 7.700, cached tap 24.739 | 46.763 / 0.243 s |
| 36 → 37 | 47.910457 s | inspect 8.260, describe 11.421, type 5.696, inspect 7.995, describe 5.984, inspect 8.272 | 47.628 / 0.282 s |
| 44 → 45 | 45.353110 s | inspect 8.397, describe 11.468, type 5.755, inspect 8.102, describe 11.342 | 45.064 / 0.289 s |
| 50 → 51 | 50.689892 s | inspect 22.603, describe 11.460, tap 16.424 | 50.487 / 0.203 s |

Server source is `gemma-journey28-20260907/visioncapture-live.jsonl`, read SHA-256 `fdb48c9f3dcbbc04925324c240cb1255ab8f4a39076afae9228ac9d530fb9a9a`, 131,282,425 bytes. Step 50 starts/completions are original lines 120303/120917, 120924/121802 and 121808/123838. Step 29 spans 61319–65541, step 36 spans 80429–85439, and step 44 spans 104625–109468. `journey28-http-events.jsonl` retains original source-line numbers, timestamps and event messages. Durations use rounded server elapsed milliseconds. Gaps outside these requests include transport and host handling. No saved stage timing attributes the delay to any specific server subsystem.

## Remaining GPU timing

These are weighted by `output.diagnostics.runner.decode_forward_count`. All recorded GPU timing groups have `valid_buffer_count == expected_buffer_count` for every completed output in the three snapshots. This establishes counter completeness, not isolated kernel timing. Group names come directly from `gpu_completion_timing`.

| Journey27 completed context | Outputs | Generated tokens/s | Full attention+router ms/forward | Sliding attention+router | Routed experts | Shared experts | Fetch aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|
| <8,192 | 27 | 17.323 | 6.789 | 9.313 | 7.025 | 4.509 | 19.225 |
| 8,192–32,767 | 59 | 12.772 | 27.279 | 9.521 | 7.333 | 4.616 | 17.552 |
| 32,768–49,151 | 41 | 10.190 | 51.478 | 9.383 | 7.056 | 4.505 | 15.718 |
| ≥49,152, maximum 56,365 | 20 | 8.857 | 65.456 | 9.322 | 6.912 | 4.306 | 15.315 |

The observed long-context delay grows mainly in the full-attention/router group. Routed expert computation is nearly flat in these populations. This justifies making the current full-attention path the next measurement target if Metal work is later authorized. It does not yet select a specific kernel modification. Fetch and router-wait counters overlap other work and cannot be added to GPU timings as disjoint costs. At long context, even the full-attention group alone exceeds the proposed 40 ms/token total budget for 25 tokens/s in this saved run.

The cache/prefill subanalysis separately derives bounded hit/miss information from GPU command-buffer counts. It cannot reconstruct expert IDs, LFU histories, exact all-hit frequency, or per-prefill-chunk expert reuse. Those missing data remain necessary before changing cache or scheduling policy.

## Commands, limitations and next measurement

Primary extraction command, exit 0:

```sh
python3 phase5-investigation/journey-analysis/analyze_saved_traces.py > phase5-investigation/journey-analysis/analysis-command-output.json.txt
```

The script creates snapshots once, then reuses those exact files on subsequent executions. It emits per-step CSVs, grouped JSON measurements and source hashes. Original source-line numbers are retained. An initial exploratory one-liner exited 1 because some tool-result content appends prose after its first JSON object. The final extractor deliberately uses `JSONDecoder.raw_decode` for that field and successfully parses all trace records. No trace lines were excluded. Native-log extraction ignored two non-JSON log lines, including the filter banner, and retained 411 `HTTPExecute` records. The full log has 137 request starts and 135 completions, including events after the bounded model trace. Their difference alone does not prove lost requests.

These saved observations deviate from the community benchmark protocol: autonomous app/tool journeys, growing and unequal prompts, short generations, changed historical builds, installed Application Support model path, background captures, possible competing work, and one explicitly overlapping build. No warm/cold controlled CLI pair, system-pressure time series, true filled 65,536-token context, exclusive runtime, repeatability series, or independent present-source equivalence was established. They are workload measurements, not performance ceilings.

The exact missing measurements are per-layer expert selections/hits/misses/read bytes split by decode and prefill, cache destruction and reconstruction timing around images, instantaneous or sufficiently sampled service memory during that transition, explicit all-hit scheduling duration, and full-attention substage timings under comparable contexts. Safe saved-evidence work is complete. Preserve model routing and current runtime policies. Seek a bounded fresh observational run only after coordinating exclusive access to the existing model process.
