Scoped PASS: clear for Main’s isolated recall probe build/run. No concrete execution or scoring blocker found. This approves neither expert prefetch nor model/app speed claims.

Verified all 22 frozen entries. Manifest SHA256 ea3f667f60558c66852522e820b2a9382bd1318a02e759f28044471056426386. Forward patch 0fe8b1dc7783f4a96e2d57206fa215b9535a724ad693fe5f4e23714020b2f024. Key candidate hashes:
- Probe 265b0c734c4cfdd750a51daf9c8dd41937ff2d590991e57148bd394bda9648d3
- Capture 1c920a5a647776a3c4c5526b5c0a43d8803d1d4b5f9ec33e83240027f38806be
- Runner 6abe1808e3b0edaadf31b42c96d1b489cca71dbc67603b2e4d5fa18256babcd4
- ModelExpertIO e3c8385ff13c32c428f7acdeed488177736a6071cff3d00ab5ce43baeaaa0c3b
- run.py 2ee30996d952fb6b1c35c9c9a4a107e199c9bed270396e83f25d2ba853a54387

Capture is a concrete owned collector, not a transaction hook. hooks.none remains selected, so activation-hook presence does not force separated GPU stages or token-major prefill. It retains the already-CPU early/late residual arrays with Swift value semantics. Actual routes, arithmetic, expert work lists, source checks and map/publication/cancellation paths are not substituted. The prototype’s extra sampler source scan is absent in the frozen probe.

Storage is bounded to 32 positions and 40 layers, with early/late vectors only for layers 0–38, checked width <=4096, mutex-protected records, and no mutable GPU scratch pointer retained. Duplicate/missing/malformed in-bound records cause rejection. The driver requires exactly 1120 complete successful maps for 28 forwards; no silent dropped-record fallback exists. Container/timing overhead can perturb measurements and is disclosed.

Actual slot IDs are captured under the existing coordinator locks before plan mutates cache state. plan.misses is correctly converted from member indices into expert IDs. Successful runner mapEnd plus complete generation excludes failed plans. Map shapes, unique rank/slot sets and actual misses equal actual experts minus resident set are checked. Per-layer private caches have no intervening writer in this serial diagnostic, so target pre-plan residency is valid for both preceding-layer launch points. No approximate LFU replay enters the score.

The frozen 1175-input/29-output capture is hash pinned and includes generation settings and source identity. Actual greedy sampler outputs alone extend the runner history. Reference IDs only reject divergence. EOS naturally stops at sample29, remains unconsumed, and leaves 28 forwards at position1203. This provides 28*39=1092 cases per predictor, not 29*40. It is an isolated exact token stream check, not parser/tool/conversation qualification.

Posthoc prediction requires the collector drained and runner idle. It receives only prior residual and target-layer index, performs target postNorm RMS normalization, existing resident router projection/settlement and officialSourceCPU routing, with fresh source validation before and after. Prediction cost includes those operations. Actual future routes/misses are only scorer inputs. Early launch is current mapEnd, late launch is after-MoE timestamp, both ending at next mapStart. No cross-token layer39 prediction occurs.

Score uses predicted nonresident pairs, covered actual misses and wasted nonresident pairs correctly. Resident predicted hits earn no read-saving credit. Early/late totals remain separate, negative scores are retained, denominators use 29 outputs with separate per-forward reporting. q is declared old-slope sensitivity and additional validation a=0 is explicitly optimistic. The serialized-predictor/all-speculative-reads-drain formula is a scenario, not an empirical bound. Window ceilings include instrumentation effects. Posthoc predictor timing may differ from live placement; 65/130ms labels only prioritize further review. Missing windows prevent a complete prioritization result.

Wrapper preserves source/build/binary linkage, fixed flags, one model/runner, 16-slot cache, 12GiB budget, >=30% free-memory guard and owned-child deadline. The collector and JSON add bounded memory beyond vector bytes, so the observed free-memory minimum remains necessary. No prefetch correctness, physical SSD attribution, parser behavior or throughput qualification is implied.

No code changes, builds, tests or model runs were performed for this review. Clear for the frozen recall-only experiment.
