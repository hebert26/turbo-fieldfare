# 2 Qwen expert memory and disk handling

Source: https://chatgpt.com/space/page_72056a8fd1348191a7a9ce9597d5579b

Local snapshot: 4 October 2026. Source sequence: 0. Keep this copy uncommitted. The source Page remains authoritative.

[Back to the four priorities index](https://chatgpt.com/space/page_880397aba17881919107f5a8cb6818c0)

**Status: complete for the agreed scope. Do not restart this point without a specific reason.** Updated 3 October 2026. Results cover saved work through 2 October.

**Done:** Added bounded source reads, reusable expert caches, parallel reads, and measurements of file checks and memory use. An expert is one of the model’s calculation groups. The model still chooses its actual experts. File identity checks remain in place.

**Working well:** Qwen can run without holding the full roughly 71 GB weight payload in memory. Cache limits control retained expert data. Exact-output and routing comparisons protect the working behaviour. App memory figures alone do not describe all system memory pressure.

**Ideas already checked:** Larger caches did not establish a speed benefit in the earlier trials. Loading predicted experts early preserved output but made one complete decode comparison about 10% slower. A four-worker file-metadata check experiment improved first-order decode time by about 5%, but failed its fixed acceptance rule and increased CPU use. Neither candidate was accepted into the app.

**Why this stays closed:** The original read/cache work is implemented, and the rejected ideas have saved measurements. Do not restart generic cache tuning, repeat the same prediction trial, remove source checks, or restrict Qwen to hand-picked experts. A new idea must explain what differs from the rejected experiment and which measured cost it addresses.

**Reopen only for:** A new measured failure that blocks point 4, or a new request from Hebert. Current memory work is a bounded point 4 dependency, not a new point 2 programme. Keep that distinction explicit.

**Saved work:** `103bc8d` and `45f62e8` contain streaming and scheduling changes. `b43390c` saves the rejected expert-prefetch trial. `47df241` saves the rejected metadata trial.

## Evidence and resume rules

**Evidence for resuming work:** Repository: `/Users/dev-machine/dev/turbo-fieldfare-personal`.

Evidence root relative to the repository: `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-22-codex/takeover-20260930/`.

Read the [index resume rules](https://chatgpt.com/space/page_880397aba17881919107f5a8cb6818c0) before choosing work after chat compaction. Keep completed work, rejected ideas, and unproved results distinct. Any necessary return must be bounded and must not replace point 4 as the active priority.

## Notes and decisions

Add dated notes here. Record the evidence, decision, and next action. Keep the status above current.

