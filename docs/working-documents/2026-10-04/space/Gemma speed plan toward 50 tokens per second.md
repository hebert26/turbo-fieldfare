# Gemma speed plan toward 50 tokens per second

Source: https://chatgpt.com/space/page_bf2a78f4031c8191b606f49150349eda

Local snapshot: 4 October 2026. Source sequence: 0. Keep this copy uncommitted. The source Page remains authoritative.

Improve the existing Gemma model in TurboFieldfare toward **50 accepted output tokens per second** on this Mac. This is a target, not a proven result. Use the lessons from Qwen and make useful use of unified memory: RAM shared by the CPU and GPU. Keep the Mac usable for other applications. Created 4 October 2026.

## Team and ownership

| Agent | Role |
| --- | --- |
| Astra, xhigh reasoning | Main agent. Own the plan, investigation, experiment choices, process control, result review and final acceptance. |
| Sol 6.1 | Code approved, bounded changes. Own explicitly assigned source files. Report changed paths and measured results to Astra. |

Astra delegates coding to Sol 6.1. Give each assignment exclusive file ownership. Both work in the same checkout and preserve other edits. Astra owns the single model-run queue. No two model runs or conflicting source edits at once.

## Scope and branch safety

Repository: `/Users/dev-machine/dev/turbo-fieldfare-personal`.

Use the **current branch and checkout**. Do not create a branch or worktree. Before editing, read `AGENTS.md` and `README.md`, record the branch, commit and changed paths, and verify that the previous work is saved. If the tree is clean, its existing commit is the starting checkpoint; do not create an empty commit. If there are new edits, identify ownership before committing them. Do not commit unrelated work.

Small experiments may stay uncommitted while measured. Keep one candidate at a time and retain its diff and results. Remove only the failed experiment's own edits. Never use a broad reset or clean. Commit each reviewed, stable improvement before the next major change. Do not push.

Qwen work remains paused. Preserve its source, evidence, installed behavior and recovery files. Do not import a whole Qwen experiment branch. Do not resume its goal, create timers, or start unrelated work.

## What success means

Use the existing Gemma model and format. Verify the exact model, pack, configuration and machine before reporting a baseline. Previous machine inspection reported an M2 Pro with 32 GB RAM; verify this again. Speed from a different machine or Qwen is not a Gemma baseline.

Count accepted output tokens only. Report prompt processing time, time to first token, steady generation speed, and complete request time. Separate cold start from warm runs. Include setup, rebuild and cleanup costs where they affect the user.

Aim for at least 50 tokens per second across repeated, representative responses in the actual app. A short peak, draft-token count or faster isolated calculation does not meet the target. Freeze the acceptance workload and counting rules before comparison. Preserve output quality, tool behavior and supported image behavior. If 50 is not yet achieved, report the best repeatable result and measured remaining cost. Do not change the target or claim completion.

## First work

1. Inspect the current Gemma path and Qwen findings. Identify which improvements already apply to shared code. Avoid duplicate work.

2. Make the required machine and model preflight checks. Measure an unchanged Gemma baseline with the repository's supported benchmark commands.

3. Show an early app run so Hebert can see the starting behavior. Do not spend the whole task on isolated probes.

4. Find the largest measured cost. Have Sol 6.1 make the smallest useful experiment. Measure it before expanding it.

5. Keep a successful change only after full-request comparison, correctness checks and Astra's review. Commit it and show the app improvement.

## Investigation order

**Memory and data movement.** Measure model data already in RAM, GPU-accessible data, repeated reads, cache misses, duplicate copies and waits. Test whether keeping frequently used weights available helps. Check whether more of the existing model can remain available without harmful memory pressure. A larger cache is a hypothesis, not an automatic improvement.

**GPU work.** Find idle gaps, repeated dispatches, CPU/GPU waits and expensive calculations. Consider more parallel GPU lanes, fewer waits, fused work and overlap where measurements support them. Preserve the required arithmetic and output behavior.

**Prompt and conversation reuse.** Check whether valid existing state is needlessly rebuilt. Keep model-specific context settings and correct invalidation. Do not shrink the real workload to improve the score.

**Multiple-token prediction.** Consider this only after Gemma support and the cost model are clear. It predicts and verifies several tokens together. Qwen's implementation is not a ready-made Gemma solution. Count accepted tokens and all draft, verification and recovery costs.

## Unified-memory policy

Use more RAM when it produces a measured speed gain. Do not impose the old low-memory target as a fixed limit. Measure memory pressure, swap growth, peak process memory, responsiveness and GPU/CPU use alongside speed. Leave room for macOS and other applications.

Increase memory use in bounded steps. Stop the owned experiment if the machine becomes unusable or sustained critical memory pressure appears. Keep a smaller working configuration available. A high utilisation figure alone does not prove failure; use actual system health. Do not kill unrelated apps or purge caches to force a pass.

## Lessons that must not be lost

| Qwen finding | Gemma implication |
| --- | --- |
| Keeping fixed weights available removed a large driver cost in a Qwen probe. | Inspect Gemma's data residency first. Prove its full-request effect before adopting it. |
| A much larger expert cache reduced logical reads but made whole requests slower. | Sweep useful memory sizes. Do not equate more cached data with speed. |
| Reusing valid prompt state improved one ordered Qwen comparison. | Check Gemma for repeated work. One comparison is not sustained proof. |
| Buffer allocation and copying were a small share in one Qwen audit. | Measure Gemma before spending time on buffer pools. |
| Some multiple-token substeps improved, while complete responses became slower. | Include every cost. Reject a local win that harms the user-visible request. |
| Later Qwen checkpoint work remains unproved. | Do not treat saved source or a review as a passing build or speed result. |

These are transfer ideas. They are not claims about Gemma performance. Record rejected ideas and the evidence needed to revisit them, so context compaction does not restart failed work.

## Evidence and review

Use the same model, prompt, context, seed and settings for comparisons. Run serially. Use repeated, balanced before/after runs with a useful output length; start with at least three pairs for a candidate worth keeping. Record failures and slower runs. Report the range and median, not only the best run.

Do not write unit tests for exploratory experiments. First establish whether the idea works. For a selected production change, run proportionate existing checks and add focused tests only where needed. Use `Scripts/test.sh` for package tests. Use SwiftFairy for relevant Swift guidance when available; an unavailable reference service is not a passed review.

Keep raw commands, exit codes, timing output, source commit/diff and tested input identity. Store logs and scripts in repository-owned locations. Keep this Page concise: current result, decisions, rejected ideas, next experiment and commit. No HTML, Safari formatting or broad documentation work is required.

## Run boundaries

Before every model run, follow the repository preflight: macOS 26+, Swift 6.2+, sufficient disk, acceptable `memory_pressure -Q`, a completed existing `scratch/gemma4.gturbo`, and no process matching `TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm`. Stop and report a failed preflight. Do not kill an app to bypass it. Run only one model process at a time.

Keep servers on `127.0.0.1`. Do not download, duplicate, replace or requantize the model. Do not prune selected experts or change the model to inflate speed. Images require the valid adjacent vision pack. Do not read or run Qwen original weights for this task. Preserve unrelated settings and installations.

## Source material

[Four priorities for Qwen speed](https://chatgpt.com/space/page_880397aba17881919107f5a8cb6818c0) and [Qwen 36-hour plan](https://chatgpt.com/space/page_401147ecde808191bdbac68ac69f54ea) contain the previous reasoning. They remain Qwen records, not Gemma acceptance.

Read the saved stop and commit receipts under `/Users/dev-machine/dev/turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/coordination/qwen-speed-36h-20261003/`, especially `safe-stop-handoff-20261004.json` and `safe-stop-commit-receipt-20261004.json`. Check source evidence before repeating any numerical claim.

Start with a brief status, the verified starting commit, the Gemma baseline plan and the first coding assignment. Continue toward a measured app improvement. Give short progress updates at useful milestones.

