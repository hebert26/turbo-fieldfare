# Grok memory result — 2026-10-02

Decision: one serial child of this frozen probe with `TURBO_QWEN_EXPERT_CACHE_RESIDENCY` omitted or any value other than `"1"`. Keep every other flag, prompt 1175, context 8192, 16 slots, arithmetic, routes, and the 30% floor. No source edit. The runner already builds a nil residency owner unless that variable is exactly `"1"` (`QwenOfficialSourceRunner.swift` 144–148). `run.py` currently forces `"1"`. This is a resource discriminator, not a speed claim and not a prediction that the child will finish.

Reviewed run: `run-20261002T192037.312697Z-2c4d23ceb7624e69b4d5bcfddf4da3d4`. Analysis: valid evidence, no correctness result, no speed claim, stop “memory below 30% or unavailable”, launch free 72%, minimum 28%, child exit -15, wall 172.300 s, pid 81987. Last durable phase is `prefillDone`. No decode marker.

## Temporal evidence

Times are seconds after the first `memory.jsonl` sample (`startUptimeNanoseconds` 2120177168742208). Process/system rows are the nearest 2 s sample, not the marker instant. GiB is 2^30. Decimal GB is 10^9.

| Event | s | Metal bytes | Nearby s | Footprint | Compressor occupied | Wired | `memory_pressure -Q` |
|---|---:|---:|---:|---:|---:|---:|---:|
| First sample | 0.005 | — | 0.000 | 82 KiB | 1,876,410,368 (1.876 GB) | 7,315,046,400 | 72% |
| Load sample | — | afterModelLoad 4,894,752,768 | 4.005 | 5.058 GB | 10,160,144,384 (10.160 GB) | 7.817 GB | 46% |
| `residencyRequestBefore` | 161.950 | 9,359,032,320 | 162.002 | 9.07 GB | 13.61 GiB | 7.35 GiB | 33% |
| `residencyRequestAfter` | 163.402 | 9,359,032,320 | 164.001 | 9.17 GB | 10.22 GiB | 10.81 GiB | 32% |
| `prefillDone` | 165.788 | 9,330,065,408 | 166.004 | 9.15 GB | 9.29 GiB | 12.42 GiB | 30% |
| Still alive | — | — | 170.006 | 9.16 GB | 7.33 GiB | 14.48 GiB | 30% |
| Stop sample | 172.013 | — | 172.006 | 0 | 7.59 GiB | 13.46 GiB | 28% |

The request call returned in 1.452 s. Metal `currentAllocatedSize` did not change across it. `residencyAllocatedSizeBytes` went 0 → 4,026,531,840 (set accounting for 1280 buffers). Peak Metal at a marker is 9,359,163,392. The 16-slot cache was already allocated layer by layer with `residencyRequested` false; layer 38 held 3,925,868,544 cache bytes before the pin.

From the 162.002 sample to the 164.001 sample, system wired rose 226,734 pages (3,714,809,856 bytes) and compressor-occupied fell 222,180 pages (3,640,197,120 bytes). Decompressions in that interval: 237,993. That bracket contains the residency return. Wired kept rising after the return and after `prefillDone` (14.48 GiB at 170 s) while the child footprint stayed about 9.16–9.21 GB. Free was not monotone: 33, 32, 30, 34, 30, then 28. The guard fired about 6.225 s after `prefillDone`.

Early compressor growth is also measured. Occupied compressor went from 114,527 pages at 0 s to 620,126 pages at 4.005 s, while this child’s footprint was about 5.06 GB and wired rose only about 0.5 GB. That load-time compression is before any expert-layer marker and before `requestResidency`.

Lifetime maximum `ri_phys_footprint` is 9,212,255,600 bytes (9.212 GB). Highest live sample is 9,210,404,208 at 168.002 s with `ri_proc_exit_abstime` still 0. The 172.006 row is an exited process: footprint 0, exit abstime 50888380367451. Its resident size 3,153,625,088 is not a live allocation. `ri_wired_size` stayed 0 on every sample.

## Counter limits

`memory_pressure -Q` is active + inactive + free + speculative over physical pages. Wired pages and compressor-occupied pages are outside that sum. A near 1:1 move from compressor to wired barely moves the percentage: 33% → 32% across the 162–164 transfer. The later net, wired up more than compressor down through 170 s, is what takes the level to 30% and then 28%.

System `vm_stat` cannot name the process that gained the wired pages. This child’s footprint did not jump at 164 s, and `ri_resident_size` rose only about 0.1 GiB in that interval (about 0.05 → 0.15 GiB). GPU wiring is absent from `ri_wired_size`. Resident size also excludes much of the GPU mapping, so those process counters neither prove nor erase a residency pin. `residencyRequestAfter` means the call returned. Wired still climbed for several seconds after that.

The 172 s rows disagree because they are different instants around process death. `-Q` printed 28%. The same-second `vm_stat` recomputes near 32% after wired had already fallen and the footprint was 0. The guard input is the 28% reading. Analysis first-to-last footprint change of −81992 bytes is the exit row, not a live release.

File-backed pages stayed about 4.2–6.9 GiB and sit inside the available set. They are not the pool that moved at 164 s.

## Command-buffer resource use

Skipping the optional set leaves command-buffer resource use intact.

Slot buffers are `makeBuffer(..., options: .storageModeShared)` with default hazard tracking (`PreadExpertStreamer.swift` 628–631). The paired cache and the coordinator retain them. The residency object’s extra `retainedAllocations` exist only when the set exists.

Serial MoE encoders bind each slot and declare it for that command: `setBuffer(mapped.gateUp)` plus `useResource(mapped.gateUp, usage: .read)`, and the same pair for `mapped.down` (`QwenMoE.swift` 876–880 and 894–901). The grouped exact path does the same (1252–1256 and 1271–1277). Leases and completion handlers keep the coordinator until GPU completion. The set’s own `deinit` comment treats encoder hazard calls as still valid after detach.

`requestResidency` and `queue.addResidencySet` run only after all 40 layers register (`ModelExpertIO.swift` 737–742). They pin allocations that already exist. With the flag off, those calls are absent. `setBuffer`, `useResource`, source checks, routes, and the 16 slots stay.

## Why this one change

The only optional call inside the 162–164 s wire-up / compressor-down bracket is `requestResidency`. Metal size was already stable at 9,359,032,320 bytes. The existing no-cache candidate is `payload-nocache-staged`: `F_NOCACHE` on routed-expert file descriptors, a different 50-prompt/64-new harness, `executed: false`, and a wrapper default floor of 20%. That advice targets file-cache pages, which stayed inside the available set on this run. It does not skip the residency pin.

An embedding-table rewrite is a new read path. These samples do not isolate that table at 164 s. Keep it out of this child.

If the off child still crosses 30%, stop. Record the phase and the wire/compressor/footprint samples. Do not lower 30%, do not add no-cache or an embedding change to a retry, and do not start recovery.
