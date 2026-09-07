# Buffer-reuse comparison — 2026-09-06

The buffer-reuse candidate was slower in all three measured cases. All six runs per build exited 0 and ended with `endOfTurn`. The three measured outputs matched byte for byte (`cmp` exit 0, reported by the coordinator). The isolated source change has been reverted, preserving other existing edits. These observations do not establish that the change caused the slowdown.

## Measured cases

| Case | Prompt / generated tokens | Baseline decode / tokens per second | Buffer reuse decode / tokens per second |
|---|---:|---:|---:|
| Short explanation | 61 / 527 | 41.38 s / 12.736 | 52.98 s / 9.948 |
| Medium review | 430 / 665 | 53.77 s / 12.367 | 66.98 s / 9.929 |
| Long synthesis | 3,015 / 597 | 55.81 s / 10.697 | 70.34 s / 8.487 |

This is the frozen CLI generation protocol at 4,096 context tokens and 16 default expert-cache slots. It does not measure the 64K / 32-slot navigation workload or establish 25 tokens/s. Matching output bytes remove output-token differences from this comparison, but do not prove general correctness.

## Environment and limits

Mac mini Mac14,12, Apple M2 Pro, 12 CPU cores, 32 GB RAM; macOS 26.5.1 (25F80); Swift 6.3.2. Both captured system records name commit `ea02a4a3df1a81936eb539da38e373d7037c2624` with existing dirty and untracked work. AC power and Low Power Mode off were recorded.

Protocol deviations: the existing installed model at `/Users/dev-machine/Library/Application Support/TurboFieldfare/gemma4.gturbo` replaced `scratch/gemma4.gturbo`, and background applications stayed open. The coordinator observed WindowServer at 44% CPU in one snapshot. System memory free percentage reported through `memory_pressure` varied: the saved baseline system record shows 57%, while the coordinator reported another pre-model snapshot at 56%, the candidate record shows 52%, and a during-run snapshot was 41%. No concurrent model or build ran during the cases. `pmset` reported no thermal warning. These snapshots do not isolate a cause or rule out all environmental effects.

Baseline CLI SHA-256: `7b68e7bafa9bc46410457a2b318eabf99dae55e09ec822ca8db9e3b3b1fc0a49`.
Candidate CLI SHA-256: `8578cd8cf78d44c2588e3e4465a328acd8f0f826baf5e1166a2aef740c456b04`.
Both model manifest SHA-256 values: `1cb53c2423f05dfa673e5f0d9a3407aa355b8227b621f8f8e575830a2bb7fa91`.

The three frozen prompt hashes match in both [baseline system record](../../benchmark-results/2026-09-06-standard/system/system.txt) and [candidate system record](../../benchmark-results/2026-09-06-buffer-reuse/system/system.txt). Raw outputs and timing files remain in their adjacent `warmup/` and `measured/` folders.

## Commands and full footers

Release build command: `swift build -c release --product TurboFieldfareCLI`. For each build, run the three cases once as discarded warmups, then once in fresh measured processes. Case/seed pairs: `short-explanation:20260721`, `medium-review:20260722`, `long-synthesis:20260723`. Each case uses this command, with `case_id`, `seed`, `run_folder`, and `pass` substituted:

```sh
.build/release/TurboFieldfareCLI \
  --model "/Users/dev-machine/Library/Application Support/TurboFieldfare/gemma4.gturbo" \
  --messages-file "docs/benchmark-prompts/real-generation-v1/${case_id}.json" \
  --max-new 1024 --max-context 4096 \
  --temperature 0.2 --top-k 64 --top-p 0.95 --seed "$seed" \
  > "benchmark-results/${run_folder}/${pass}/${case_id}.stdout" \
  2> "benchmark-results/${run_folder}/${pass}/${case_id}.stderr"
```

Folders are `2026-09-06-standard` and `2026-09-06-buffer-reuse`; passes are `warmup` and `measured`. The coordinator confirmed the exact executable, flags, seeds, sequential fresh processes, no explicit expert-cache flag, and no additional environment or profiling controls. All twelve process exit codes were 0 according to the coordinator. Full stderr footers, in short/medium/long order:

```text
Baseline warmup
[stop=endOfTurn prefill=61tok new=527tok decode=40.51s tok/s=13.010]
[stop=endOfTurn prefill=430tok new=665tok decode=55.47s tok/s=11.988]
[stop=endOfTurn prefill=3015tok new=597tok decode=55.82s tok/s=10.694]
Baseline measured
[stop=endOfTurn prefill=61tok new=527tok decode=41.38s tok/s=12.736]
[stop=endOfTurn prefill=430tok new=665tok decode=53.77s tok/s=12.367]
[stop=endOfTurn prefill=3015tok new=597tok decode=55.81s tok/s=10.697]
Buffer reuse warmup
[stop=endOfTurn prefill=61tok new=527tok decode=47.17s tok/s=11.172]
[stop=endOfTurn prefill=430tok new=665tok decode=64.79s tok/s=10.264]
[stop=endOfTurn prefill=3015tok new=597tok decode=63.85s tok/s=9.350]
Buffer reuse measured
[stop=endOfTurn prefill=61tok new=527tok decode=52.98s tok/s=9.948]
[stop=endOfTurn prefill=430tok new=665tok decode=66.98s tok/s=9.929]
[stop=endOfTurn prefill=3015tok new=597tok decode=70.34s tok/s=8.487]
```

The original buffer path and accurate content-signature wording are now deployed, exit 0. Log: `/tmp/turbo-restored-buffer-and-wording-deploy-20260906.log`. Installed app SHA-256 is `cea60fe89f8941c94bb1c2875334b0a1fa6ec094cf4e6cb7947444cd4a76a048`, service `df6fc90698f04ccb1ba690059b11013380963a2e6ff76103127693524e967844`. This confirms deployment, not new performance or navigation acceptance.
