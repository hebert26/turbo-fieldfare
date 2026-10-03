# Saved embedding source recovery

Checked HEAD: `526d35a090eb4b87e8b6410babd685d6bb5310e7`. The tracked tree was clean.

Candidate root: `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-22-codex/takeover-20260930/performance-proposals/embedding-row-memory-20261002`.
Every file in this candidate stage is untracked. `.gitignore:44` ignores `/scratch`.

## Source that is not committed

Only these three candidate Swift files have no exact file copy in the Git index:

- `candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenExactBlockProbe.swift`
- `candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenOfficialSourceModel.swift`
- `candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenOfficialSourceRunner.swift`

All nine other candidate source/package files match committed files under `baseline-memory-probe-20261002/candidate/`.
The full overlay differs from HEAD in exactly the twelve files listed in `frozen-source-hashes.json`. No extra source changes were found.
The three new files are also saved in `overlay/`. Those overlay copies match the candidate files.
`patch/forward.diff` covers all twelve source/package files. `patch/versus-memory.diff` covers only the three new files. All three patch files have no exact committed copy.
`prepare.py`, `run.py`, `preflight.py`, and `analyze.py` have no exact committed copy. `build.py` and `evidence_helpers.py` match committed copies.

Committed production source does not include protected embedding rows. Git history searches for `TURBO_QWEN_PROTECTED_EMBEDDING_ROWS` and `protectedEmbeddingRows` found no source commits.
No committed exact source set exists for this candidate. Its base source and nine unchanged trial files are committed. Saving the three new files or full forward patch can recover its source.

## Actual latest run

Only one candidate run exists: `run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5`.
Use its `receipt.json`. The top-level `receipt.json` and `source-handoff.json` describe the earlier pre-run state.
- Build receipt: exit 0. Overlay stable.
- Run receipt: wrapper exit 2. Child exit -15. Wall time 182.04904562514275 seconds.
- Stop reason: memory below 30% or unavailable. Observed minimum free memory: 29%. Launch free memory: 73%.
- No `result.json`. No complete correctness result. No speed claim.
- Actual after-load Metal allocation: baseline 4,894,752,768 bytes; candidate 3,877,634,048 bytes. Difference: 1,017,118,720 bytes.
- Last saved phase: `prefillDone`. That does not prove the active phase at stop.

All twelve frozen candidate hashes, all 38 frozen stage hashes, and all eight base file hashes match the saved text/source files.
All six raw run file hashes in the actual receipt match. No model payload or binary was read or hashed for this check.

## Explicit preservation list

The list below has 65 files (3,713,149 bytes). Paths are relative to the candidate root above.
It saves the complete frozen source stage and all raw run records. Existing scripts require the saved base and candidate copies at these paths. This is the smallest intact stage list that also preserves its build/source link records.
Force-add only these explicit files. Do not add the whole candidate directory.
Exclude `build/`, `overlay/`, `__pycache__/`, model payloads, and build product files. Build logs are not needed for this list.

```text
analyze.py
baseline-hashes.json
baseline/Package.swift
baseline/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift
baseline/Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift
baseline/Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift
baseline/Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift
baseline/Sources/TurboFieldfare/Runtime/Qwen/QwenOfficialSourceConversationState.swift
baseline/Sources/TurboFieldfare/Runtime/Qwen/QwenOfficialSourceModel.swift
baseline/Sources/TurboFieldfare/Runtime/Qwen/QwenOfficialSourceRunner.swift
build-command.json
build-receipt.json
build.py
candidate/Package.swift
candidate/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift
candidate/Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift
candidate/Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift
candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenBaselineMemoryPhaseProbe.swift
candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenExactBlockProbe.swift
candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenExactBlockTypes.swift
candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift
candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenOfficialSourceConversationState.swift
candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenOfficialSourceModel.swift
candidate/Sources/TurboFieldfare/Runtime/Qwen/QwenOfficialSourceRunner.swift
candidate/Sources/TurboFieldfareBaselineMemoryProbe/Command.swift
control-manifest.json
evidence_helpers.py
frozen-hashes.json
frozen-source-hashes.json
overlay-after-build-hashes.json
overlay-before-build-hashes.json
overlay-prepared-hashes.json
patch-checks.json
patch/forward.diff
patch/inverse.diff
patch/versus-memory.diff
preflight-20261002T195801.899959Z-b126b896-3b80-4462-a886-852782c41d55.json
preflight-plan.json
preflight.py
prepare.py
receipt.json
request.json
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/analysis-20261002T200202.831664Z-aa58502614bb4b56beb6462c3427d7e4.json
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/commit-after.stderr
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/commit-after.stdout
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/commit.stderr
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/commit.stdout
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/extra-process-guard.stderr
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/extra-process-guard.stdout
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/hardware.stderr
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/hardware.stdout
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/launch-memory.stderr
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/launch-memory.stdout
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/memory.jsonl
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/preflight.stderr
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/preflight.stdout
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/probe.stderr
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/probe.stdout
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/process-system.jsonl
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/receipt.json
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/sampler.stderr
run-20261002T195800.631823Z-da4cd4d8aa864a78b4377fdd86cfd6b5/sampler.stdout
run.py
source-handoff.json
wrapper-handoff.json
```

Optional review records: `review/astra-source-review.md`, `review/grok-source-review.md`, `review/opus-source-review.md`.

## Limits for the next run

The saved wrappers require HEAD 526d35a. A preservation commit changes HEAD. Keep these old records unchanged. A new trial must use a new frozen stage and its new source commit.
`prepare.py` also refuses an existing overlay. Do not rerun it in this old stage.
This report made no source edits, commits, builds, tests, model runs, worktrees, or model payload reads.
Only this report was written. Main owns the preservation commit and the next implementation.
