Opus next lever after the pair-cost and metadata-cost results, 2026-10-02 (read-only at d6d7bc5; saved results, receipts and code only; no builds, tests, runs or weight reads)

Inputs: `exact-pair-cost-20261002/decision-001.json`, `exact-metadata-cost-20261002/decision-001.json`, this folder's `receipt.json`, and the stage analysis of the real-app trace `expert-read-app-trial-20261002/qwen-read-known-none-off-app-001-strict-analysis.stdout.json`. That trace took 20.950 s for 32 tokens (0.655 s per token). The current reference, 1.529820 tok/s, is 0.654 s per token, so the two match.

**Where a real-app token goes.** Measured values are per token, with 31 forwards over 32 tokens. Lines marked "est." are derived.

| Item | ms per token | Share |
|---|---|---|
| Expert map (`expert_map` span) | 288 | 44% |
| of which miss reads, est. (app019: 6,332 misses × 1.26 ms slope ÷ 32) | about 250 | 38% |
| GPU command waits | 209 | 32% |
| of which GPU busy | 109 | |
| of which submit to GPU start (4,434 commands) | 87 | 13% |
| CPU attention | 24 | 4% |
| Everything else | 133 | 20% |
| of which full-source scans, est. (82 per forward × 1.19 ms, from pair-cost serial: 0.396 s ÷ 333) | about 94 | 14% |

**Does the metadata path merit production work? Not as the next lever.**
- The parallel part is at most the per-entry work: 62% of a scan for opens alone (measured), or 74% counting opens, stats and closes. The ideal 4-lane ratio is therefore 0.44–0.54. That saves 44–52 ms per token, at most about 8%, or about 1.65 tok/s.
- The receipt's own gate (≤0.85) corresponds to about 14 ms per token (2%).
- Production would also need three things the receipt lists: caller-linked cancellation, an EMFILE admission or fallback, and Main's ruling on the changed order of observations between entries.
- There is a contention risk. In the four-lane trial, a hit check took 173 µs under concurrency, against 97 µs serial. That was a different path, so it is a warning, not evidence for this case.
- Let the frozen standalone run answer feasibility; it is cheap. Start production work only if 4 lanes reach ≤0.65 in both orders, which is about 33 ms or more per token.

**Larger lever: hide miss reads behind compute with a predicted next-layer expert set.**
- In `produceToken` and `moeStep` (QwenOfficialSourceRunner.swift:331–357 and 940–1000, map at :976), reads happen only inside `coordinator.map`. The read path is idle for the rest of each layer: about 9.5 ms per layer of non-map time, against about 6.5 ms of miss time.
- The model's own router for layer L+1 can rank experts before layer L+1's real routing. It can be applied to either input below. This uses existing weights; nothing is fitted to outputs.
  - **Early, P_a:** `residual-after-mixer` of layer L. The window opens after layer L's map, about 8 ms (est.).
  - **Late, P_b:** `residual-after-moe` of layer L. This misses only layer L+1's mixer term. The window is the L+1 mixer, router and pre-map scan, about 5 ms (est.).
- Routes and arithmetic do not change: only the time at which a pair is read changes.
- Ideal bounds, assuming perfect prediction and the same per-miss slope, are scenarios, not predictions: P_b about 190 ms per token, P_a about 250 ms. A recall R scales these.
- This is not a cache-size or replacement-policy change: the cache-bound analysis says "No prefetch". It is also not more read workers.
- Earlier notes rejected the idea only by argument ("guessed routes … can increase traffic"). Recall has never been measured.

**Next experiment: measure next-layer prediction recall, with no change to reads or arithmetic.**
- **Run:** an isolated runner probe, prefill of the same prompt, then 32 serial decode tokens.
- **Capture:** existing hooks record `residual-after-mixer` and `residual-after-moe` for each layer (about 21 MB), `observeRoute`, and the per-map hit and miss data. Record the layer windows from existing spans.
- **After decode, in the same process:** for each position and each layer L < 39, compute the top-8 of the router for L+1 applied to RMSNorm(h, postNorm for L+1), for h in {P_a, P_b}. Use the existing `projection` and `QwenMoE.route` calls on weights that are already resident.
- **Score each prediction against two things:**
  - the actual routes for L+1;
  - the 16-slot LFU state, which the replay already reproduces at 6,338 against 6,332 misses.
- **Report:**
  - recall over actual misses;
  - waste: pairs predicted but neither used nor resident;
  - S = Σ min(covered misses × 1.26 ms, window − waste × 1.26 ms) ÷ 32.
- **Exactness check:** token IDs equal the serial capture.
- **Cost:** one small probe patch (Sol) and one run of about 3–4 minutes (Main). Prefill takes about 110 s.

**Stop rule, fixed before the run:**
- If S < 65 ms per token for the better predictor (10%), drop prefetch. The next items are then the 87 ms per token of submit-to-GPU-start latency, and the metadata path if it meets the 0.65 bar.
- If S ≥ 130 ms per token (20%), stage an exact prefetch probe:
  - bounded staging of at most 16 pairs (96 MiB);
  - every per-read check before and after each read;
  - consumer `validateBoth`;
  - the post-map scan before GPU use;
  - outstanding prefetch reads drained before rollback or cancellation;
  - serial fallback when hooks are present.

  Astra must first rule on prefetch reads issued after layer L's post-map scan but before layer L+1's pre-map scan. Gate it on exact tokens and routes, at least 30% free memory, and a measured drop in map wall.
- Between 65 and 130 ms, Main decides.

No result here is a speed claim or a hardware limit. Even P_a's ideal bound gives about 2.5 tok/s.

Decision: do not start production work on parallel metadata now. Use its frozen standalone run only to learn whether parallel checks can reach the 0.65 bar. The strongest next exact lever is to overlap expert reads with compute, using the model's own next-layer router as the predictor. Measure its recall first, without changing behaviour.
