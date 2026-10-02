Opus memory result, 2026-10-02 (read-only; probe.stderr markers, memory.jsonl, process-system.jsonl, analysis JSON, live code; no runs, builds or edits)

**Decision.** Neither residency OFF nor no-cache is the next change. The next single exact change is to stop keeping the 1.017 GB BF16 embedding table resident and read each needed row on demand, with full checks. Keep residency and every other setting as in this run.

## Timeline (measured; relative seconds)

The level I recompute from vm_stat, (active + inactive + free + speculative) ÷ total, matches `memory_pressure` within about 1 point in every sample. That confirms the XNU formula.

| Phase | Level | Compressor occupied | Wired | Child footprint / RSS | Metal |
|---|---|---|---|---|---|
| 0 s | 71–72 | 1.75 | 6.81 | 0 / 0 | — |
| Load (4 s) | 46 | 9.46 | 7.28 | 4.71 / 1.10 | 4.56 GiB (4.894 GB) |
| Prefill 20–161 s | 30–39 (min 30; 23 of 71 samples ≤ 31) | 11.7 → 14.2 | flat 7.1–7.6 | 5.1 → 8.45 / 0.04–0.23 | +96 MiB per layer marker, to 8.72 GiB |
| Residency request (162–164 s) | 33 → 32 | 13.61 → 10.22 → 7.33 (170 s) | 7.35 → 10.81 → 14.48 | 8.5 / RSS 0.14 → 2.9 (170 s) | unchanged |
| Decode reads (164 s on) | 30, 34, 30, then 28 at the stop | | | | |

GiB unless marked.
- At load, the child's first 4.7 GiB went with +7.7 GiB of compressor and −9.6 GiB of anonymous pages.
- Wired plus compressor stays about 21 GiB across the residency request: 20.95 → 21.0–21.8.
- Over the run, `Compressions` rose by 90 GiB and `Decompressions` by 74 GiB.

## Interpretation

**Supported:**
- The level is pinned by non-available memory (wired + compressor), not by file cache. File-backed pages stay 4.2–6.9 GiB and count as free.
- Prefill already touched 30% with no residency at all. Residency only converts compressed pages into wired ones. Requesting it for the 3.75 GiB cache shrank the compressor by about 6.3 GiB while wired grew 7.1 GiB, for a net of about 0.8 GiB (about 2.5 points). That conversion indicates the cache was largely compressed before the request.
- So turning residency off can remove at most that late step. It cannot give margin in prefill, which sat at 30–31 for long stretches.
- With free pages at 0.05 GiB throughout, each GB of this process's demand removed should lower wired + compressor by up to about 1 GiB, about 3 points. This is an estimate.

**Counter limits:**
- Compressor pages cannot be attributed to a process.
- `ri_wired_size` stays 0 because GPU wiring is charged to the kernel.
- RSS excludes non-resident and GPU-mapped pages.
- The final footprint of 0 is an exited-process record. The lifetime maximum is 9.21 GB.
- vm_stat and memory_pressure are sampled at different instants: at 172 s they read 32% and 28%.
- The request returning does not prove residency has been reached.

**Hypothesis, not acted on here:** the 90 GiB of compression churn suggests the VM compresses and decompresses idle, incompressible BF16 model memory. That could add kernel CPU and GPU-start latency. Grok's area.

## Residency OFF correctness

Correct, and it is the default path. The residency set is opt-in (runner :138–141). Every MoE encoder binds the expert slot buffers with `setBuffer` and declares them with `useResource(..., .read)` (QwenMoE.swift 876/880, 894/901, 1186/1190, 1205/1211, 1330/1351). Metal therefore makes them resident for each command without the set. The set (ModelExpertIO.swift:685–757) only adds allocations, requests residency and attaches to the queue. Arithmetic, routes and checks are unchanged. It is simply not the decisive lever.

## Why not no-cache

`F_NOCACHE` targets the unified buffer cache, which is not the pool that moves here. It does not reduce the process's own 8.7 GiB, and its indirect effect is unproven.

## Next action: drop the resident embedding table

**Why:** the embedding table is 1.017 GB, 21% of the 4.89 GB dense load. It is touched one row per token, so it is idle and very likely sits compressed at about 1:1. Its only consumer is `QwenOfficialSourceRunner.embedding(_:)` (:908–923, via `encodeEmbeddingFromImmutableIDs`).

**Change (Sol), exact:**
- In `QwenOfficialSourceModel` (:350–354), make only `lm_head` resident.
- Keep an admitted `TensorRange` for `model.language_model.embed_tokens.weight`.
- `embedding(token)` reads bytes token × 4096 ..< +4096 through `preadTensorRange`, which keeps every before, after and before-return check plus the vocabulary bounds check. It then widens BF16 to FP32 bit for bit, as `Float(bitPattern: UInt32(b) << 16)`.
- No arithmetic, routing or cache change.
- The existing gates cover exactness: 29 IDs, 1,120 route and weight bits, raw rows, and final state bits.

**Expected (estimate):**
- after-load Metal about 3.88 GB;
- every phase about +3 points: prefill minimum about 33, decode about 31–33;
- prefill cost about +0.5 s for 1,175 row reads; decode cost negligible.

**Gate:** run the same instrumented serial child once, with residency unchanged. It passes if it completes with exact outputs and stays at or above 31%. If it still crosses 30%, report to Main and Hebert that 16 slots plus this context do not fit the 30% guard on this machine without a further reduction, before any recovery comparison. Do not run a flag sweep.
