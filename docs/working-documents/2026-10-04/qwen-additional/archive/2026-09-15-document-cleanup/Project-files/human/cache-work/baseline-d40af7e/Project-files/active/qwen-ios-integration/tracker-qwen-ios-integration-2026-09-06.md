---
title: "Add Qwen for iOS testing"
slug: qwen-ios-integration
implementation: ./implementation-qwen-ios-integration-2026-09-06.md
status: awaiting-approval
version: 1
approved_version: null
approved_by: null
approved_at: null
created_at: "2026-09-06"
updated_at: "2026-09-06"
current: null
next_action: "Review version 1 and the existing iOS integration dependency."
---

> Owner update, 2026-09-06: “okay, no do not create worktree yet”. The plan now lives in /Users/dev-machine/dev/turbo-fieldfare-personal/Project-files. The earlier planning worktree is being removed after preserving these documents. Earlier requirements to create or retain that worktree are superseded. Do not create another worktree until the owner requests it. The B1 discussion below concerns adopting the current uncommitted iOS implementation into any future implementation checkout. Implementation remains awaiting approval.


# Tracker: Add Qwen for iOS testing

Goal: Qwen completes the current iOS screenshot and accessibility test journeys through TurboFieldfare.

Details, scope, code, evidence: [implementation-qwen-ios-integration-2026-09-06.md](./implementation-qwen-ios-integration-2026-09-06.md)

## Key

| Mark | Meaning |
|---|---|
| `[ ]` | not started |
| `[~]` | in progress |
| `[x]` | done |
| `[!]` | blocked |
| `[-]` | skipped, reason in the implementation document |
| `[s]` | source-done: code accepted, its live or manual proof has not passed. Does not close a phase. |

## Now

- Phase `-` - task `-` - `unassigned`
- Blocked: `1` - coverage rows not passing: `54`
- Next: Review version 1 before implementation; phase 7 also needs the current iOS source snapshot.

## Phases

| # | Phase | Needs | Tasks | Unit tests | Status | Done |
|---|---|---|---|---|---|---|---|
| 1 | Qwen installs as a verified model pack | - | 0/4 | 4 of 4 not passing | `[ ]` | - |
| 2 | Qwen reads the exact chat and tool history | 1 | 0/3 | 1 of 1 not passing | `[ ]` | - |
| 3 | Qwen updates its recurrent memory on Metal | 1 | 0/4 | 5 of 5 not passing | `[ ]` | - |
| 4 | Qwen attends to earlier tokens on Metal | 1 | 0/3 | 4 of 4 not passing | `[ ]` | - |
| 5 | Qwen selects and streams the correct experts | 1 | 0/3 | 6 of 6 not passing | `[ ]` | - |
| 6 | Qwen generates text through the runtime | 2, 3, 4, 5 | 0/4 | 8 of 8 not passing | `[ ]` | - |
| 7 | Qwen restores a conversation after an interrupted turn | 6 | 0/3 | 4 of 4 not passing | `[!]` | - |
| 8 | Qwen understands screenshots in a conversation | 6, 7 | 0/5 | 9 of 9 not passing | `[ ]` | - |
| 9 | Existing clients load the selected model | 7, 8 | 0/3 | 8 of 8 not passing | `[ ]` | - |
| 10 | The current iOS navigation loop accepts Qwen decisions | 9 | 0/3 | 4 of 4 not passing | `[ ]` | - |
| 11 | Qwen completes the recorded iOS test journeys | 10 | 0/4 | 1 of 1 not passing | `[ ]` | - |

## Done-when rules

- **D1 tasks** - every task in the phase is `[x]` or `[-]` with a reason in the implementation document. `[~]`, `[!]` and `[s]` do not pass.
- **D2 acceptance** - every acceptance box in the phase is ticked.
- **D3 covered** - every row of the phase coverage table reads `<n>/<n> pass`, `no-unit-test - <proof>`, or `waived - <link>`. `missing`, `stale`, and any failing count block the phase.
- **D4 no blind spot** - no file this phase changed is missing from its coverage table. Run the command in `coverage-gate.md`. Every path it prints must appear in the table, as its own row or under a directory row that is a prefix of it.
- **D5 evidence** - the phase evidence path is written in the implementation document.

**A phase cannot be `[x]` while D3 or D4 fails.** Code this phase added, with no unit test covering it, blocks this phase.

## Phase 1 - Qwen installs as a verified model pack

Status: `[ ]` - Owner: `unassigned` - Needs: `-` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-1)

- [ ] 1.1 Add GTurboManifestV2.swift so Qwen layers stay explicit.
- [ ] 1.2 Pin Qwen36SourceAdapter.swift for reproducible downloads.
- [ ] 1.3 Map Qwen tensors in RepackPlanner.swift without requantizing.
- [ ] 1.4 Dispatch ModelLoader.swift by family to reject mixed packs.

**Acceptance**

- [ ] Qwen install verification rejects unsupported tensors before model loading.
- [ ] Gemma v1 files still load through the existing path.
- [ ] Cancel/resume preserves only verified Qwen ranges.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareFormat/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareRepack/Core/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Qwen36/Qwen36Model.swift` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 2 - Qwen reads the exact chat and tool history

Status: `[ ]` - Owner: `unassigned` - Needs: `1` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-2)

- [ ] 2.1 Add Qwen36Tokenizer.swift to preserve exact control tokens.
- [ ] 2.2 Add Qwen36ChatRenderer.swift for exact tool continuations.
- [ ] 2.3 Add Qwen36ToolCallParser.swift so partial calls cannot run.

**Acceptance**

- [ ] Qwen chat and tool fixtures match the pinned reference tokens.
- [ ] Incomplete model calls cannot become executable host actions.
- [ ] Gemma tokenization and tool decoding retain their existing behavior.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Tokenization/` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 3 - Qwen updates its recurrent memory on Metal

Status: `[ ]` - Owner: `unassigned` - Needs: `1` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-3)

- [ ] 3.1 Add qwen36_primitives.metal for exact Qwen normalization.
- [ ] 3.2 Add Qwen36CausalConv.swift to retain only valid history.
- [ ] 3.3 Add gated_deltanet.metal to match outputs and state.
- [ ] 3.4 Register Qwen shaders in MetalContext.swift for real execution.

**Acceptance**

- [ ] DeltaNet Metal results match the independent output and state fixtures.
- [ ] Sequence splits preserve convolution history and recurrent state.
- [ ] GPU failures do not mark a partial state as usable.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Metal/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Qwen36/Qwen36RecurrentBuffers.swift` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | none yet | `missing - not implemented` | - |
| `Package.swift` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 4 - Qwen attends to earlier tokens on Metal

Status: `[ ]` - Owner: `unassigned` - Needs: `1` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-4)

- [ ] 4.1 Add Qwen36Attention.swift for gated grouped attention.
- [ ] 4.2 Add Qwen36RoPE.swift to preserve text and image positions.
- [ ] 4.3 Add Qwen36AttentionCache.swift to bound valid KV rows.

**Acceptance**

- [ ] Qwen attention outputs match causal gated reference fixtures.
- [ ] Image rotary positions remain separate from actual token offsets.
- [ ] KV allocation scales only with the ten full-attention layers.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Metal/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Qwen36/Qwen36AttentionCache.swift` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 5 - Qwen selects and streams the correct experts

Status: `[ ]` - Owner: `unassigned` - Needs: `1` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-5)

- [ ] 5.1 Add Qwen36Router.swift to select eight of 256 experts.
- [ ] 5.2 Add Qwen36Experts.swift for SiLU and the shared gate.
- [ ] 5.3 Reuse DequantInt4GEMV.swift only for verified Qwen layouts.

**Acceptance**

- [ ] All 256 routed expert IDs are addressable with eight selected per token.
- [ ] Shared and routed outputs use the Qwen activation and gates.
- [ ] Expert slots stay valid until dependent GPU work completes.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Metal/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Kernels/Quant/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Qwen36/Qwen36ExpertIO.swift` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Infrastructure/Streaming/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 6 - Qwen generates text through the runtime

Status: `[ ]` - Owner: `unassigned` - Needs: `2, 3, 4, 5` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-6)

- [ ] 6.1 Add Qwen36ForwardRunner.swift with the exact layer order.
- [ ] 6.2 Add Qwen sampling to Sampler.swift without Gemma softcap.
- [ ] 6.3 Add Qwen36Prefill.swift for bounded prompt processing.
- [ ] 6.4 Add Qwen36ReferenceFixtures to expose numerical drift.

**Acceptance**

- [ ] Qwen logits come from the independent output head and correct layer sequence.
- [ ] Prompt chunk boundaries do not change final state or logits.
- [ ] Qwen sampling supports uncapped output without altering Gemma behavior.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Inference/ModelRuntimeFactory.swift` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Generation/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Metal/Sampling/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Kernels/Sampling/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Configuration/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Prefill/` | none yet | `missing - not implemented` | - |
| `Package.swift` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 7 - Qwen restores a conversation after an interrupted turn

Status: `[!]` - Owner: `unassigned` - Needs: `6` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-7)

- [!] 7.1 Add Qwen36StateCheckpoint.swift for atomic restoration.
- [ ] 7.2 Adapt MultimodalConversation.swift to recover hybrid state.
- [ ] 7.3 Reset Qwen state on New Chat and model replacement.

**Acceptance**

- [ ] Failure restores complete model state or explicitly invalidates the conversation.
- [ ] Tool-result retry never replays an already dispatched device action.
- [ ] New Chat starts from empty recurrent, image and host action state.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Generation/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareApp/Core/Inference/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareApp/Core/State/AppModel.swift` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

Blocked by: `B1 - reviewed current iOS integration is absent from this worktree.`

## Phase 8 - Qwen understands screenshots in a conversation

Status: `[ ]` - Owner: `unassigned` - Needs: `6, 7` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-8)

- [ ] 8.1 Add Qwen36VisionPack.swift to bind images to the text model.
- [ ] 8.2 Add Qwen36ImagePreprocessor.swift for exact screenshot pixels.
- [ ] 8.3 Add qwen36_vision.metal for the Qwen screenshot tower.
- [ ] 8.4 Add Qwen36VisionMerger.swift for ordered image embeddings.
- [ ] 8.5 Add Qwen36MultimodalRenderer.swift for tool-result images.

**Acceptance**

- [ ] Qwen screenshot embeddings and prompt positions match reference fixtures.
- [ ] Images in tool results use the selected Qwen processor.
- [ ] Missing or mismatched vision support returns an explicit unavailable error.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareFormat/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareRepack/Core/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Vision/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Kernels/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Metal/Qwen36/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Runtime/Generation/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Tokenization/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | none yet | `missing - not implemented` | - |
| `Package.swift` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 9 - Existing clients load the selected model

Status: `[ ]` - Owner: `unassigned` - Needs: `7, 8` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-9)

- [ ] 9.1 Add a Model picker in InspectorView.swift for explicit selection.
- [ ] 9.2 Route decode requests through ModelRuntimeFactory.swift.
- [ ] 9.3 Route CLI and server model loads through the same factory.

**Acceptance**

- [ ] The Model control uses separate settings and install descriptors per family.
- [ ] The Mac app retains a single decode-service model owner.
- [ ] CLI and loopback server dispatch the selected family without cache mixing.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareApp/Core/Configuration/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareApp/Core/State/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareApp/Core/Inference/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareDecodeProtocol/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareDecodeService/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareApp/Mac/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareCLI/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareServer/` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 10 - The current iOS navigation loop accepts Qwen decisions

Status: `[ ]` - Owner: `unassigned` - Needs: `9` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-10)

- [ ] 10.1 Use model capabilities in VisionCaptureToolLoop.swift.
- [ ] 10.2 Preserve screenshot and retry checks in the tool loop.
- [ ] 10.3 Freeze ios-journeys.json so both models face the same tests.

**Acceptance**

- [ ] Qwen uses only the current Simulator accessibility actions.
- [ ] Screenshot validation and host verification remain enforced.
- [ ] The comparison manifest specifies reproducible iOS starting states.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareApp/Core/Tools/` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareApp/Core/Inference/AppGenerationRequest.swift` | none yet | `missing - not implemented` | - |
| `Sources/TurboFieldfareApp/Core/State/AppModel.swift` | none yet | `missing - not implemented` | - |
| `Package.swift` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 11 - Qwen completes the recorded iOS test journeys

Status: `[ ]` - Owner: `unassigned` - Needs: `10` - Done: `-` - [details](./implementation-qwen-ios-integration-2026-09-06.md#phase-11)

- [ ] 11.1 Record Gemma runs against ios-journeys.json as the baseline.
- [ ] 11.2 Record Qwen runs against the same ios-journeys.json.
- [ ] 11.3 Write comparison.md so the owner can judge iOS reliability.
- [ ] 11.4 Clean up only recorded Qwen scratch to reclaim temporary disk.

**Acceptance**

- [ ] The owner can compare actual iOS outcomes, full task time and memory.
- [ ] No unsafe replay, wrong-device action or unsupported screenshot acceptance is hidden.
- [ ] Generated scratch is removed while evidence and this worktree remain.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `none - this phase changes no code` | none yet | `missing - not implemented` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Changelog

| Date | Phase | What happened |
|---|---|---|
| 2026-09-06 | - | Version 1 created in a separate worktree; awaiting approval. |
