Opus repeat-memory diagnosis, 2026-10-02 (read-only at 298c08a; candidate source, both run receipts and memory.jsonl, earlier successful runs' memory.jsonl, live allocation code, and Apple XNU and system_cmds source; no runs, builds, shard reads or source edits)

## 1. What the guard actually measures (primary source)

`/usr/bin/memory_pressure -Q` prints `memorystatus_get_level()`, which is the kernel's `memorystatus_level` (system_cmds `memory_pressure.c`, `get_percent_free`; XNU `kern_memorystatus.c` `memorystatus_get_level`). On macOS, XNU sets:
- the available count to `AVAILABLE_NON_COMPRESSED_MEMORY = active + inactive + free + speculative` pages (`osfmk/vm/vm_page.h`, non-jetsam macOS branch; updated at `vm_pageout.c:3850`);
- `memorystatus_level = available * 100 / total_pages` (`vm_pageout.c:4738`).

Consequences:
- File cache (pageable, on the active and inactive queues) and ordinary anonymous memory already count as free. Reclaimable cache cannot lower this number.
- Only memory outside those queues lowers it: wired memory (including GPU/IOKit allocations) and compressor-held pages. A reading of 29–30% means about 70% of RAM is wired or compressed.
- This is real pressure, not cache. Lowering the floor would hide it.

## 2. Why the baseline fails now

**Measured:**
- **Same shape in every run.** Load drops the level 13–30 points in 4–6 s. The level then slides through prompt processing to a plateau of 30–35%, and decode stays there.
  - Successful runs (prefetch OFF/ON, metadata OFF/ON, CPU-hint) had minima of 30–32% from launches of 61–75%.
  - The two current aborts launched at 52% and 71% and reached 29% at 158 s and 150 s.
  - The causal run launched at 46% and aborted at 16 s.
  - All of them run within 1–3 points of the guard for about 100 s. An abort is one 2-second sample landing on 29.
- **Launch level does not protect.**
  - Run 1 (52%) reached 30% at 6 s, then stayed at 30–35% for 150 s while prompt processing kept allocating.
  - Run 2 (71%) fell 46 → 30% over 6–118 s.

  The end level converges near 30–33% whatever the launch level. That fits a system holding a pressure equilibrium by compressing or swapping other memory once this process's demand arrives. This interpretation is a hypothesis.
- **Child RSS stayed at 0.15–0.35 GB** during prompt processing while the level fell 16 points (5 GB). The process's large memory is GPU (IOKit) memory that `ps` RSS does not count.
- **In the last 6 s of run 2**, RSS rose from 43 MB to 3.0 GB (144–150 s). That is roughly where prompt processing ends: earlier prompt walls were 128.7–142.6 s, plus load. The phase is unmarked. Anonymous RSS counts as available, so this jump is a phase signal, not proven to be the cause.

**Supported cause (code fact plus timeline; the magnitude split is unmeasured):**
- The grouped prefill creates each layer's coordinator on first use (candidate runner, `expertCoordinator(layer:)` inside the layer loop).
- `QwenBF16PairedExpertCache.init` allocates all 16 gate/up and down shared `MTLBuffer` slots eagerly (`PreadExpertStreamer.swift:626–634`).
- With one 1,175-token chunk processed layer-major, the 3.75 GiB cache (11.7 points of 32 GiB) is therefore added gradually across the whole of prompt processing. That matches the slow slide.
- Dense BF16 residency, about 4.8 GB estimated (including the 1.017 GB embedding table and the 1.017 GB `lm_head`, `QwenBF16Weights.swift:118/:222`), matches the load step.
- The baseline itself fills the system to the guard. The recurrent-recovery code never ran, and it adds only about 1 MiB of saved rows. Its per-cycle 64 MiB CPU clone is anonymous memory, which counts as available.

**Not supported:**
- A residency-request step. The expert residency set is requested only at the 40th layer (`ModelExpertIO.swift:734–737`, flag `TURBO_QWEN_EXPERT_CACHE_RESIDENCY=1`), but no successful run shows a level step there.
- Final-snapshot blame. That 111 MiB copy happens after decode, which neither abort reached.

**Unknown:** run 2 lost 16 points during prompt processing, but the cache alone explains 11.7. The remaining 4–5 points (about 1.4 GB) could be:
- the process's own extra GPU memory (transient per-call buffers, pooled allocations);
- compressor growth of other processes, driven by about 58 GB of prompt-phase reads;
- kernel wired memory.

Nothing sampled so far separates these.

## 3. One next action: attribution sampling only, no code change

**Run:** one serial-only child, with the frozen binary, the same flags and guard, and no legacy or new arms. Add sampling to the wrapper only, every 2 s:
- **Child**, via `proc_pid_rusage(pid, RUSAGE_INFO_V4)`, unprivileged for your own process:
  - `ri_phys_footprint`, which includes IOKit and GPU memory;
  - `ri_wired_size`;
  - `ri_resident_size`;
  - `ri_diskio_bytesread`. Its rate marks the prompt and decode phases without markers in the driver.
- **System**, via `vm_stat`: free, active, inactive, speculative, wired, pages occupied by compressor, file-backed, swapouts. Recompute the level from these to confirm the formula above.
- **Optional:** `footprint <pid>` after load, mid-prompt and at the plateau, if it runs without elevation, to split IOAccelerator from malloc.

An abort at 29% still yields the attribution. Cost: about 2.5 minutes.

**Decision rule:**
- **Child footprint at the plateau above about 11 GB (the process explains the excess).** Fix the largest exact item:
  - the resident 1.017 GB embedding table, which needs only one row per token and could become per-token protected row reads with the same bytes;
  - or any pooled transient GPU buffers the footprint reveals.

  Each GB is about 3 points.
- **Child footprint about 9.5–10 GB, and compressor or wired memory outside the child growing by 3 GB or more during prompt processing.** The pressure is read-driven. Test the existing exact, flag-only `payload-nocache-staged` (`F_NOCACHE` on expert payload FDs) in the same serial sampler run.
- **Either case:** do not lower the 30% floor, context, cache16, prompt or outputs. Do not retry blindly.

## 4. Measuring the recovery candidate safely

The candidate's memory difference from serial is negligible. Run the three-arm comparison only after a serial-only sampler run holds a minimum of 33% or more, which is 3 points of margin. That needs the attribution above plus one exact footprint reduction. Admission by launch level alone (50% or 71%) has already failed twice.

Decision: the baseline fails because the run sits at the system's pressure plateau, about 70% of RAM wired or compressed, within 1–3 points of the guard. The next action is a wrapper-only attribution run of the serial arm, before any recovery measurement or code change.
