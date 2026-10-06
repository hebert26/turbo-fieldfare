# 3 Qwen CPU and GPU work

Source: https://chatgpt.com/space/page_71298161b6248191bf63419024bb461a

Local snapshot: 4 October 2026. Source sequence: 0. Keep this copy uncommitted. The source Page remains authoritative.

[Back to the four priorities index](https://chatgpt.com/space/page_880397aba17881919107f5a8cb6818c0)

**Status: complete for the agreed scope. Do not restart this point without a specific reason.** Updated 3 October 2026. Results cover saved work through 2 October.

**Done:** Reduced repeated preparation, reused packed attention data, and grouped independent GPU work. Some changes spread calculations across more GPU lanes while preserving the required arithmetic. Added timing measurements and visible request stages. Experimental kernels in the repository are not automatically enabled in the app.

**Working well:** A repeated app comparison used the same input and produced the same answer. Grouped prompt processing reduced average prompt time from 205.1 to 150.9 seconds, about 26%. The measured whole turn fell from 240.4 to 187.9 seconds, about 22%.

**Proof limit:** Output generation was slightly slower in that comparison. The gain was faster input processing. It was a short workload, not sustained 20-token-per-second proof. The grouped option remains explicitly controlled; do not describe every prototype as a production default.

**Why this stays closed:** The useful changes and their limits are saved. Do not restart broad CPU/GPU profiling or repeat the same kernel experiments after losing chat context. Preserve the measured gains. Code needed for grouped token verification belongs to point 4.

**Reopen only for:** A concrete regression in the accepted path, a measured dependency required by point 4, or a new request from Hebert. Use one bounded hypothesis and a direct comparison.

**Saved work:** `990c062` preserves key packing reuse. `9ff5554` preserves attention accumulation work. `db56686` preserves GPU experiments, measurements, and request progress. The repeated app result is in `context-64k-trial-20261001/qwen-grouped-linear-app-repeat-summary.json` under the evidence root below.

## Evidence and resume rules

**Evidence for resuming work:** Repository: `/Users/dev-machine/dev/turbo-fieldfare-personal`.

Evidence root relative to the repository: `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-22-codex/takeover-20260930/`.

Read the [index resume rules](https://chatgpt.com/space/page_880397aba17881919107f5a8cb6818c0) before choosing work after chat compaction. Keep completed work, rejected ideas, and unproved results distinct. Any necessary return must be bounded and must not replace point 4 as the active priority.

## Notes and decisions

Add dated notes here. Record the evidence, decision, and next action. Keep the status above current.

