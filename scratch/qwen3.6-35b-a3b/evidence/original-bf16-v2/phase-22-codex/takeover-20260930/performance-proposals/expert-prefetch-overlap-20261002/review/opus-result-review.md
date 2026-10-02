Opus result review, expert-prefetch-overlap-20261002 (read-only at f210c70; post-run-audit-001.json, the comparison receipt, both raw result.json files, the candidate source and the metadata-parallel results; no builds, runs or payload reads)

**Rejection verified.**
- **Speed:** OFF decode 19.955851667 s, ON 22.023224875 s, ratio 1.1036, above the 0.95 first-pair gate. The receipt exits 2 with the error "ratio exceeds 0.95", so the reverse pair was correctly not run.
- **Exactness, recomputed from both raw results:** `outputIDs` (29), `routes` (1,120, including weight bits) and `plans` (1,120) are identical, with the same capture SHA.
- **Counts are consistent:**
  - 8,960 consumers, of which 5,337 are plan misses: 3,448 imported and 1,889 demand-read (3,778 streams).
  - 5,345 speculative pairs = 10,690 streams.
  - Unused bytes 1,897 × 6,291,456 = 11,934,892,032.
  - ON logical reads = demand + speculative.
- **Memory:** minimum free was 30% (wrapper) and 31% (internal) OFF, and 32% ON.

**Measured decomposition, ON minus OFF** (whole-run counters):

| Item | Change |
|---|---|
| Map | −2.487 s (7.263 → 5.043 ms per map) |
| Full-source scans | +2.926 s (1.361 → 2.635 ms per call, 2,296 calls each) |
| Hint | +0.342 s |
| Remaining decode outside map, scans and hint | +1.287 s (8.697 → 9.984 s, +14.8%) |
| Total | +2.067 s |

- Hit and publication checks did not change: 0.215 → 0.213 ms each.
- Before-copy checks were 0.206 ms per imported pair.
- Copies averaged 0.239 ms per stream. They run inside the existing demand `concurrentPerform` (candidate PreadExpertStreamer.swift:757–767), so the 1.645 s summed copy time overlaps other work and is not all on the critical path.

**Actual source locks** (code fact; mechanism is hypothesis):
- `OfficialSourceHandle.identityLock` (:112) covers only dictionary compare and insert in `remember` (:765–772). The code states that no filesystem I/O holds it (:110–111).
- `ioMeasurementLock` covers counters only.
- `revalidateSource` (QwenOfficialSourceModel.swift:482) takes no lock.
- The speculative `owner.read` calls `preadTensorRange` without any coordinator lock.
- The capture `Mutex` wraps only counter updates, not the measured bodies.

So the 1.27 ms added per scan is not a Swift lock wait.
- **Hypothesis:** contention below Swift (VFS and metadata work on the same source tree, CPU or memory) between the scan's 78 opens and 156 stats and the speculative streams' per-read checks (6 per pair).
- **Consistent with:** metadata-parallel's measured self-contention. The same work used +28.8% CPU at 2 lanes and +74.8% at 4.
- The candidate starts the speculative batch before the post-map scan (candidate runner :1006–1021). The first pair therefore always overlaps that scan.

**Why the prefetch line should stop, not be rescheduled** (estimates from measured counters and the CPU-screen rows):
- **Interference outweighs saving per pair.**
  - Interference per speculative pair: (2.926 + 1.287) ÷ 5,345 = 0.79 ms.
  - Net map saving per imported pair: 2.487 ÷ 3,448 = 0.72 ms.
  - Under linear scaling, even zero wasted pairs would lose: about +0.57 s including hints.
- **The queue fit only because it stretched its own windows.**
  - ON drain was only 0.154 s.
  - Rerunning the same queue in the screen's OFF windows (pair wall 1.63–1.9 ms) gives 1.8–2.8 s of drain.
- **A fence does not rescue it.** A fence means launching after the post-map scan and joining before the pre-map scan. Modelled, it keeps 76–82% of the imports and adds 0.58–0.67 s of join wait, for a net of about −0.24 to +0.08 s. That is break-even, not a 5% gain.
- Predictor tuning cannot fix a per-pair loss. Removing checks is out of scope.

**One next exact change: parallel receipt-entry checks inside the decode full scans (the exact-metadata-parallel candidate), limited to one `produce` operation.**
- **Why this one.** It attacks the same measured cost: OFF scans are 3.124 s, 15.7% of decode. In decode the scan runs alone: the map has finished, or not started, and no GPU command is pending on this path. The standalone 4-lane ratio, 0.69–0.73 in both orders, already includes its own self-contention.
- **Estimate (hypothesis):** 0.84–0.97 s saved per 29 outputs, 4.2–4.9% of OFF decode. Two lanes (0.80–0.82) would save about 3% for half the extra CPU.
- **Preserved:**
  - every open, stat and compare, both scans, the bindings, `remember`, and per-entry cancellation, via Astra's operation-owned latch (astra-cancellation-path.md);
  - an unchanged synchronous post-map-to-submit path;
  - serial fallback outside `produce`.
- **Main's open items:** the file-descriptor-pressure policy and the ruling on changed inter-entry observation order. Neither removes a check.
- **Measurement:** reuse this experiment's harness and counters, OFF against lanes, in ABBA order. Report full-source wall per call, decode wall, CPU time, exact IDs, routes and plans, and minimum free memory.
- **Suggested stop rule:**
  - Stop if per-call scan wall stays above 1.05 ms, or if decode does not improve by at least 3% in both orders.
  - Stop if any count, ID, route or plan differs.

Decision: the rejection stands. Retire expert prefetch under the current validator cost; do not tune or reschedule it. The next exact change worth staging is 4-lane parallel receipt-entry checks in decode full scans, with an expected gain of 4–5%. That is useful, but small, and it is no route to 20 tok/s.
