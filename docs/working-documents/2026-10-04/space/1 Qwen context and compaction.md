# 1 Qwen context and compaction

Source: https://chatgpt.com/space/page_1ee844c408c4819189ea686cc542c7b8

Local snapshot: 4 October 2026. Source sequence: 0. Keep this copy uncommitted. The source Page remains authoritative.

[Back to the four priorities index](https://chatgpt.com/space/page_880397aba17881919107f5a8cb6818c0)

**Status: complete for the agreed scope. Do not restart this point without a specific reason.** Updated 3 October 2026. Results cover saved work through 2 October.

**Done:** Qwen and Gemma now have separate saved context settings. Qwen supports the app’s 64K choice without replacing Gemma’s setting. Existing conversation recovery and compaction safeguards remain. Compaction shortens the conversation and rebuilds its working state.

**Working well:** The model’s context setting is no longer tied to the other model. The code can preserve the existing choices when older settings are loaded. This resolves the shared-setting problem that started point 1.

**Proof limit:** A 64K capacity setting is not a test with 64K occupied tokens. The saved app comparison contained about 1,087 tokens. Longer uninterrupted agent sessions and fewer rebuilds at a full 64K window remain unproved. These limits are not permission to restart point 1.

**Why this stays closed:** The separate-setting implementation exists in `MacAppSettings.swift`. Do not re-investigate whether Qwen needs its own setting, replace it with a shared setting, or repeat the original context-size discussion. Preserve Gemma’s behaviour. Broader context qualification is a separate future scope.

**Reopen only for:** A concrete failure in the existing setting, compaction, or recovery behaviour, or a new request from Hebert. Identify the failing case first. Make the smallest required fix and return to point 4. Do not claim that compaction has been removed or that full 64K performance has passed.

**Saved work:** `db56686` includes the model-specific context settings. Relevant code: `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift` and `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift`.

## Evidence and resume rules

**Evidence for resuming work:** Repository: `/Users/dev-machine/dev/turbo-fieldfare-personal`.

Evidence root relative to the repository: `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-22-codex/takeover-20260930/`.

Read the [index resume rules](https://chatgpt.com/space/page_880397aba17881919107f5a8cb6818c0) before choosing work after chat compaction. Keep completed work, rejected ideas, and unproved results distinct. Any necessary return must be bounded and must not replace point 4 as the active priority.

## Notes and decisions

Add dated notes here. Record the evidence, decision, and next action. Keep the status above current.

