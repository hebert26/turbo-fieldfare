Opus CPU-hint next step, 2026-10-02 (read-only at f210c70; inputs: decision-001.json, run-20261002T143735Z-ad72b5f9/result.json rows, astra-cpu-result.md, astra-prefetch-admission-scope.md; no builds, runs or code)

Verdict: the one OFF/ON experiment is worth running, but expect about half the headline. The run's own data and the import design put the central gain near 45–65 ms/output, not 92. That is 6–9% of the 723 ms/output measured here. One unbalanced pair cannot resolve that reliably.

**Most likely critical flaw: the scenario charges the import cost per case, but the design pays it per imported pair on the demand critical path.**
- The scenario subtracts a = 1 ms per case. The reviewed import adds two fresh `validateBoth` calls for every covered pair: one before the copy and one after it. The original publication `validateBoth` is kept and is already in the demand cost.
- One `validateBoth` is two zero-byte checks, about 0.19 ms (app020: 96.9 µs per check, serial). The import also copies 6 MiB from quarantine into the slot. That copy is unmeasured; I estimate about 0.13 ms.
- So the import adds about 0.39–0.52 ms per pair, or 1.2–1.6 ms per case at 3.16 covered pairs per case. The scenario's a = 1 ms per case is below that.
- The scenario also uses q = 1.26 ms from an older capture. This run's own fit over 1,092 maps is 1.016 ± 0.039 ms per miss (intercept 2.79 ms). Each map has 8 consumers, so this slope is already miss cost minus hit cost, which is exactly the saving from turning a miss into a hit-like import.

Recomputed with the run's own formula and rows, replacing a with a per-covered-pair cost c:

| q (ms) | c = 0.39 | c = 0.52 |
|---|---|---|
| 1.016 (this run) | 62.0 | 46.6 |
| 1.26 | 83.8 | 68.3 |

All values are ms/output. Each `validateBoth` per imported pair is worth about 23 ms/output: 3,448 pairs × 0.19 ms ÷ 29.

So the gain probably sits at or below the fixed 65 ms gate. It is still positive, and still the largest measured exact candidate.

**Missing measurements the ON/OFF must report, or a null result cannot be explained:**
1. **Import cost on the critical path.** For each imported pair, separately: the before-copy check, the copy, the after-copy check and publication. Also report map wall per covered miss, ON against OFF.
2. **Contention from the prefetch's own checks.**
   - Prefetch reads keep their per-read before, after and before-return checks. That is about 6 per pair, 5,345 pairs including the 1,897 wasted.
   - They now run inside the window, at the same time as the runner's serial post-map and pre-map full scans (2 per case, about 1.19 ms each) and the hit checks.
   - Earlier traces showed per-read checks slowing several-fold under concurrency.
   - Report full-scan wall per call and hit-check wall per call for ON and OFF. A 20% slowdown of the scans alone costs about 19 ms/output.
3. **Order balance.** Run ABBA, not a single OFF/ON pair. In the same-build pair-cost run, the four serial arms varied by 5.5% (1.617–1.706 s), against an expected effect of 6–9%.

**One design question for Astra, not a request to weaken anything.** With known-none hooks, nothing runs between the after-copy check and the original publication `validateBoth`. Can that single publication check serve as the after-copy observation? That would recover about 23 ms/output. The copy could also be avoided by swapping buffer ownership instead of copying, if the slot layout allows it.

Memory is unchanged from Astra's note: the minimum free was exactly 30%, so the 48 MiB allocation must pass a fresh preflight first.

Decision: run the staged experiment with the three measurements above and ABBA order. Set the pass bar as a measured end-to-end decode gain in both orders, at least 30 ms/output, with exact IDs and routes. Treat a result under that bar as a no-gain finding, not a reason to tune.
