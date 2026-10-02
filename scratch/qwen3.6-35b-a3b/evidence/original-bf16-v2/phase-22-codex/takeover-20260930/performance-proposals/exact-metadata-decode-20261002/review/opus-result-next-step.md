Opus result and next step, exact-metadata-decode-20261002 (read-only at b43390c; post-run-audit-001.json, the comparison receipt and both raw result.json files; prior notes for context; no builds, runs or shard reads)

**Result, measured.** The run was not accepted: ON's validation mean was 1.134995 ms, above the preset 1.05 ms. The gate stays where it is, and the reverse pair was correctly not run.

| Measure | OFF | ON |
|---|---|---|
| Source validation per call (2,296 calls each) | 1.431 ms | 1.135 ms (ratio 0.793) |
| Source wall | 3.286 s | 2.606 s (−0.680 s) |
| Decode | 20.403 s | 19.366 s (−1.037 s) |
| Decode CPU | | +12.1% (system 23.29 → 26.28 s) |

- Outputs, routes, plans and safety match, and free memory stayed at 31% or above.
- **The validator explains 0.680 s, or 3.3%, of the decode change.** The other 0.357 s falls inside the variation between OFF arms: this OFF arm took 20.403 s, against 19.956 s for the prefetch experiment's OFF arm on the same request.
- **In-app lanes scale worse than standalone.** Standalone runs gave serial 1.156 ms and 4-lane 0.776 ms (ratio 0.69–0.73). In-app runs gave 1.431 and 1.135 ms. The cause is a hypothesis: kernel caches or metadata paths disturbed by the 33.6 GB of expert reads per decode.
- Even if accepted, this lever is worth about 3%. Stop sweeping metadata parameters.

**Zoom out (measured per output).** These figures are from the prefetch OFF arm on this request: 29 outputs, 19.956 s.

| Item | Per output | Share |
|---|---|---|
| Map | 280 ms | 41% |
| Full-source scans | 108 ms | 16% |
| Everything else (GPU, launch gaps, CPU attention, CPU work) | 300 ms | 43% |

- Inside the map, measured serial `validateBoth` takes 1.926 s: hit checks 0.780 s plus publication checks 1.146 s.
- Validation measured directly on the critical path is therefore 5.05 s, 25% of decode. That counts scans and `validateBoth`, but not per-read checks.
- The 6.2 s read phase also contains three fresh per-read checks per stream, on 10,674 streams. Their share of the read phase has never been isolated. The app019 bins could not separate it, and that note says a condition varying one term while holding the other fixed is needed.
- 20 tok/s means 50 ms per output, about 14× below today's 688 ms. No measured exact lever has shown more than about 5%:
  - four read lanes: map unchanged;
  - exact block: ×1.37 slower; pair block 0.94;
  - prefetch: +10%;
  - 4-lane metadata: −3%.
- Every route to a multiple must cut per-position cost. Per-position cost is dominated by expert reads and validation, and neither is amortized by blocks under the current contract.

**One direction: find out whether the expert read phase is limited by validation or by bytes.** It is the largest bucket and the one unexplained term. The answer decides which family of exact changes can still give a multiple.

**Experiment: standalone read-path attribution.**
- **Scope:** no model, GPU, inference or route change. Sol writes the command; Main runs it, because it reads payload ranges exactly as decode already does.
- **Replay:** this request's captured 1,120 plans (5,337 misses, 3,623 hits) with the production map structure:
  - per map, up to 8 pairs × 2 streams through `concurrentPerform`, joined;
  - then the serial hit and publication `validateBoth` calls in plan order;
  - pair-sized scratch buffers, no cache publication.
- **Three arms over the same sequence, run ABBA:**
  - **A, production:** unchanged `preadTensorRange` with every check, through the admitted TensorRanges.
  - **B, checks only:** the same `validateRetainedFile` sites and counts per stream, as zero-byte calls, with no payload.
  - **C, diagnostic payload only:** the same offsets and 4 MiB syscalls through separate read-only file handles, with no checks. It is never used for inference. Its bytes are hashed against arm A to prove identical data.
- **Report for each arm:**
  - read-phase wall per map and per miss;
  - user and system CPU;
  - process-attributed read bytes (`proc_pid_rusage`);
  - check counts;
  - minimum free memory.
- **Cost:** a small command reusing the metadata harness and the existing admissions. Each arm takes about 6–9 s; the whole run takes a few minutes.

**Pre-registered rule.** Let A, B and C be read-phase wall per miss.
- **Validation-bound: (A − C) ≥ 0.4 × A in both orders.**
  - Then validation-driven time is most of decode: scans, `validateBoth` and per-read checks together, which is at least 25% measured and plausibly about 50%.
  - The next step is an exact cheaper validation design: the same observations with fewer kernel metadata operations per byte read.
  - Hebert also gets the measured per-output validation budget for any contract decision. That decision is his, and nothing is assumed here.
  - Upside (hypothesis): up to about 2×. This is the precondition for any multi-token path.
- **Byte-bound: (A − C) ≤ 0.15 × A.**
  - Stop validation engineering on the read path.
  - The remaining exact levers are fewer expert bytes per output: cache memory under the 30% guard, or union sharing across positions.
  - Main should then put the measured bytes-per-output arithmetic to Hebert before more probes.
- **In between:** Main decides.

This is not a hardware-limit claim, and it is not a path to 20 tok/s by itself. It is the cheapest measurement that separates the two remaining families, and it repeats none of the rejected experiments: four lanes, larger cache, prefetch, block verification and metadata lanes.

Decision: do not accept or retune the metadata candidate. The next step is the three-arm, standalone read-path attribution, which splits the 280 ms per output map phase into validation and bytes.
