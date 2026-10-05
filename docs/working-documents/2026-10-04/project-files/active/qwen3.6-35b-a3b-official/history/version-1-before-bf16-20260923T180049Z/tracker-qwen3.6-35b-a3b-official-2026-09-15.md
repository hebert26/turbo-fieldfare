---
title: "Qwen3.6-35B-A3B official integration"
slug: qwen3.6-35b-a3b-official-implementation-notes
implementation: ./implementation-qwen3.6-35b-a3b-official-2026-09-15.md
status: superseded
execution_state: paused
evidence_scope: historical-quantized-v1
version: 1
approved_version: 1
approved_by: "Hebert"
approved_at: "2026-09-15T11:53:55Z"
created_at: "2026-09-15"
updated_at: "2026-09-23"
current: "Paused by Hebert; existing Gemma streaming is the reference"
next_action: "Keep implementation paused. Reconcile Qwen with the verified Gemma streaming design before resuming."
---

# Tracker: Qwen3.6-35B-A3B official integration

Current requirement: preserve the official BF16 weights and use route-selected expert streaming. Implementation is paused by Hebert. The prior quantized plan is historical and its conversion is withdrawn. See the current requirement and verified Gemma explanation in the implementation document.

Goal: run the downloaded official Qwen model at its published BF16 precision using bounded expert streaming, with Gemma unchanged as default and rollback.

Details, scope, code, evidence: [implementation-qwen3.6-35b-a3b-official-2026-09-15.md](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md)

## Key
| Mark | Meaning |
|---|---|
| `[ ]` | not started |
| `[~]` | in progress |
| `[x]` | done |
| `[!]` | blocked |
| `[-]` | skipped, reason in the implementation document |
| `[s]` | source-done: code accepted, live or manual proof has not passed |

## Now
- Execution: paused by Hebert. No implementation, conversion, model run or VoiceOver test is authorized to resume.
- Owner: `Astra / GPT-6 Sol high / GPT-6 Luna xhigh`.
- Current reference: Gemma keeps shared working data and bounded expert slots; routing selects experts per layer and token.
- Historical version-1 evidence: `19/25` phases, `111/145` tasks, `245/293` coverage rows accepted.
- Unit test coverage rows not passing: `48` (historical version 1).
- Blocked: `0` historical task marks; execution is paused by the owner.
- Next: finish the original-precision streaming plan, naming the loader, expert-access, GPU calculation and memory changes before implementation resumes.
- Voice: spoken replies and macOS VoiceOver are both off.
- Document reviewed: 2026-09-23. Current decisions are recorded; the replacement implementation plan is still unfinished.

## Phases

Historical version-1 phase table. These marks do not qualify original BF16 execution.
| # | Phase | Needs | Tasks | Unit tests | Status | Done |
|---|---|---|---|---|---|---|
| 1 | The exact official checkpoint is recognized | - | 4/4 | all pass | `[x]` | 2026-09-15 |
| 2 | A v2 Qwen manifest loads without changing v1 | 1 | 6/6 | all pass | `[x]` | 2026-09-15 |
| 3 | Tiny Qwen execution fixtures are reproducible | 1 | 5/5 | all pass | `[x]` | 2026-09-15 |
| 4 | Tiny BF16 ranges convert deterministically | 1 | 5/5 | all pass | `[x]` | 2026-09-15 |
| 5 | A local official snapshot validates offline | 1 | 5/5 | all pass | `[x]` | 2026-09-15 |
| 6 | A complete Qwen pack is sized before writing | 2, 4, 5 | 6/6 | all pass | `[x]` | 2026-09-15 |
| 7 | A tiny transformed pack resumes byte-identically | 2, 4, 6 | 6/6 | all pass | `[x]` | 2026-09-15 |
| 8 | Qwen Metal bindings have one checked contract | 2 | 4/4 | all pass | `[x]` | 2026-09-16 |
| 9 | Qwen full attention matches the tiny oracle | 2, 3, 8 | 5/5 | all pass | `[x]` | 2026-09-16 |
| 10 | Qwen linear attention matches the tiny oracle | 2, 3, 8 | 5/5 | all pass | `[x]` | 2026-09-16 |
| 11 | Qwen MoE matches the tiny oracle | 2, 3, 8 | 5/5 | all pass | `[x]` | 2026-09-16 |
| 12 | A tiny Qwen runner emits the expected tokens | 2, 3, 9, 10, 11 | 6/6 | 16 of 16 pass · D4 pass | `[x]` | 2026-09-16 |
| 13 | Qwen turns commit or roll back atomically | 12 | 6/6 | 12 of 12 pass · D4 pass | `[x]` | 2026-09-16 |
| 14 | Qwen prompts match the pinned chat template | 1 | 4/4 | 5 of 5 pass · D4 pass | `[x]` | 2026-09-17 |
| 15 | Incomplete Qwen tool output dispatches nothing | 14 | 5/5 | 7 of 7 pass · D4 pass | `[x]` | 2026-09-17 |
| 16 | Qwen still images produce matching token rows | 2, 3, 8, 12, 13 | 8/8 | 27 of 27 satisfied · D4 pass | `[x]` | 2026-09-18 |
| 17 | Decode service binds responses to loaded identity | 12, 13, 15 | 6/6 | 18 of 18 satisfied · D4 pass | `[x]` | 2026-09-22 |
| 18 | CLI runs verified Qwen requests | 12, 15, 16 | 5/5 | 9 of 9 satisfied · D4 pass | `[x]` | 2026-09-22 |
| 19 | Loopback server serves verified Qwen chat | 12, 15, 16 | 5/5 | 12 of 12 satisfied · D4 pass | `[x]` | 2026-09-22 |
| 20 | The app selects Qwen without moving Gemma | 6, 12, 15, 16, 17 | 7/9 | 43/50 coverage accepted; 7 pending | `[~]` | - |
| 21 | Authorized conversion writes Qwen beside Gemma | 2, 4, 6, 7 | 3/6 | 14/17 coverage accepted; 3 pending | `[~]` | - |
| 22 | Real Qwen text behavior matches the quantized reference | 12, 13, 15, 21 | 0/8 | 11 of 12 not passing · D4 fallback | `[ ]` | - |
| 23 | Real Qwen screenshots preserve verified workflow truth | 16, 20, 22 | 0/7 | 9 of 9 not passing · D4 fallback | `[ ]` | - |
| 24 | A controlled Gemma-Qwen comparison is recorded | 22, 23 | 0/7 | 9 of 9 not passing · D4 fallback | `[ ]` | - |
| 25 | Opt-in Qwen can roll back to Gemma | 18, 19, 20, 22, 23, 24 | 0/7 | 8 of 8 not passing · D4 fallback | `[ ]` | - |

## Done-when rules
Every phase closes on the same five lines. Written in full once here. Repeated short inside each phase. Copy them word for word.

- **D1 tasks** - every task in the phase is `[x]` or `[-]` with a reason in the implementation document. `[~]`, `[!]` and `[s]` do not pass.
- **D2 acceptance** - every acceptance box in the phase is ticked.
- **D3 covered** - every row of the phase coverage table reads `<n>/<n> pass`, `no-unit-test - <proof>`, or `waived - <link>`. `missing`, `stale`, and any failing count block the phase.
- **D4 no blind spot** - no file this phase changed is missing from its coverage table. Run the command in `coverage-gate.md`. Every path it prints must appear in the table, as its own row or under a directory row that is a prefix of it.
- **D5 evidence** - the phase evidence path is written in the implementation document.

**A phase cannot be `[x]` while D3 or D4 fails.** Code this phase added, with no unit test covering it, blocks this phase.

D3 proves a named test file exists and really ran and passed. It does not prove the test exercises that code well. The implementation coverage plan states the observable behavior each test must prove.

## Phase 1 - The exact official checkpoint is recognized

Status: `[x]` - Owner: `identity` - Needs: `-` - Done: `2026-09-15` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-1)

- [x] 1.1 Record the metadata-only official identity fixture  2026-09-15
- [x] 1.2 Implement immutable Qwen source identity validation  2026-09-15
- [x] 1.3 Map all official tensor names without loading weights  2026-09-15
- [x] 1.4 Add independent identity and tensor-map rejection tests  2026-09-15

**Acceptance**
- [x] The exact pinned identity is accepted from metadata without reading tensor payloads.
- [x] Wrong identity, sidecar digest, tensor name, or MTP count is rejected before planning.
- [x] Gemma source constants and prepared Qwen shards remain unchanged.

**Unit test coverage**

Base: `ae138d53c244f32985b7a98b0293b20361c499e4`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | `13/13 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` | `13/13 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Format/Fixtures/Qwen36OfficialMetadata.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | `13/13 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | `13/13 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` | `13/13 pass` | 2026-09-15 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 2 - A v2 Qwen manifest loads without changing v1

Status: `[x]` - Owner: `format` - Needs: `1` - Done: `2026-09-15` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-2)

- [x] 2.1 Define the v2 header and closed architecture union  2026-09-15
- [x] 2.2 Define the v2 Qwen vision binding contract  2026-09-15
- [x] 2.3 Add the verified installed-model descriptor  2026-09-15
- [x] 2.4 Route runtime loading through format-owned dispatch  2026-09-15
- [x] 2.5 Add hostile v2 and descriptor tests  2026-09-15
- [x] 2.6 Re-run the frozen v1 compatibility fixture unchanged  2026-09-15

**Acceptance**
- [x] A valid Qwen v2 text manifest returns a verified Qwen descriptor.
- [x] Invalid family, architecture, provenance, quantization, or vision binding fails closed.
- [x] The frozen v1 compatibility fixture remains byte-identical and passes.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareFormat/GTurboFormatV2.swift` | `Tests/TurboFieldfareFormat/GTurboFormatV2Tests.swift` | `5/5 pass` | 2026-09-15 |
| `Sources/TurboFieldfareFormat/GTurboVisionFormatV2.swift` | `Tests/TurboFieldfareFormat/GTurboVisionFormatV2Tests.swift` | `4/4 pass` | 2026-09-15 |
| `Sources/TurboFieldfareFormat/InstalledModelDescriptor.swift` | `Tests/TurboFieldfareFormat/InstalledModelDescriptorTests.swift` | `3/3 pass` | 2026-09-15 |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift` | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestReaderQwenV2Tests.swift` | `2/2 pass` | 2026-09-15 |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestReaderQwenV2Tests.swift` | `2/2 pass` | 2026-09-15 |
| `Tests/TurboFieldfareFormat/GTurboFormatV2Tests.swift` | `Tests/TurboFieldfareFormat/GTurboFormatV2Tests.swift` | `5/5 pass` | 2026-09-15 |
| `Tests/TurboFieldfareFormat/GTurboVisionFormatV2Tests.swift` | `Tests/TurboFieldfareFormat/GTurboVisionFormatV2Tests.swift` | `4/4 pass` | 2026-09-15 |
| `Tests/TurboFieldfareFormat/InstalledModelDescriptorTests.swift` | `Tests/TurboFieldfareFormat/InstalledModelDescriptorTests.swift` | `3/3 pass` | 2026-09-15 |
| `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestReaderQwenV2Tests.swift` | self | `2/2 pass` | 2026-09-15 |
| `Tests/TurboFieldfareFormatCompatibility/GTurboFormatCompatibilityTests.swift` | `Tests/TurboFieldfareFormatCompatibility/GTurboFormatCompatibilityTests.swift` | `3/3 pass` | 2026-09-15 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 3 - Tiny Qwen execution fixtures are reproducible

Status: `[x]` - Owner: `fixtures` - Needs: `1` - Done: `2026-09-15` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-3)

- [x] 3.1 Write the pinned tiny-fixture generator  2026-09-15
- [x] 3.2 Generate full and linear attention fixture sections  2026-09-15
- [x] 3.3 Generate MoE, greedy-text, and vision fixture sections  2026-09-15
- [x] 3.4 Add fixture digest and schema tests  2026-09-15
- [x] 3.5 Record generator environment and negative controls  2026-09-15

**Acceptance**
- [x] The fixture is bound to pinned model and Transformers revisions.
- [x] It covers full attention, DeltaNet, MoE, greedy text, and actual tiny vision tower/merger with intermediates.
- [x] Regeneration is deterministic and negative controls detect five plausible defects.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Package.swift` | `Tests/TurboFieldfare/Core/QwenFixtures/QwenFixtureDigestTests.swift` | `6/6 pass` | 2026-09-15 |
| `scratch/qwen3.6-35b-a3b/fixture-generator/` | evidence-only | `no-unit-test - scratch/qwen3.6-35b-a3b/evidence/phase-3/fixture-generation.md` | 2026-09-15 |
| `Tests/TurboFieldfare/Core/QwenFixtures/qwen36-tiny-fixtures.json` | `Tests/TurboFieldfare/Core/QwenFixtures/QwenFixtureDigestTests.swift` | `6/6 pass` | 2026-09-15 |
| `Tests/TurboFieldfare/Core/QwenFixtures/QwenFixtureDigestTests.swift` | self | `6/6 pass` | 2026-09-15 |
| `scratch/qwen3.6-35b-a3b/evidence/phase-3/fixture-generation.md` | evidence-only | `no-unit-test - scratch/qwen3.6-35b-a3b/evidence/phase-3/fixture-generation.md` | 2026-09-15 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 4 - Tiny BF16 ranges convert deterministically

Status: `[x]` - Owner: `quant` - Needs: `1` - Done: `2026-09-15` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-4)

- [x] 4.1 Define the official BF16 affine policy - 2026-09-15
- [x] 4.2 Implement bounded BF16 group quantization - 2026-09-15
- [x] 4.3 Add a tiled transform reader - 2026-09-15
- [x] 4.4 Add policy and quantizer golden tests - 2026-09-15
- [x] 4.5 Add transform boundary and cancellation tests - 2026-09-15

**Acceptance**
- [x] The policy explicitly covers every allowed Qwen tensor class.
- [x] Tiny BF16 inputs produce stable packed values, scales, and biases.
- [x] Chunk size, cancellation, and resume do not change final bytes.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
|`Sources/TurboFieldfareRepack/Core/Quantization/BF16AffineQuantizationPolicy.swift`|`Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineQuantizationPolicyTests.swift`|`4/4 pass`|2026-09-15|
|`Sources/TurboFieldfareRepack/Core/Quantization/StreamingBF16AffineQuantizer.swift`|`Tests/TurboFieldfareRepack/Core/Quantization/StreamingBF16AffineQuantizerTests.swift`|`5/5 pass`|2026-09-15|
| `Sources/TurboFieldfareRepack/Core/Quantization/BF16AffineTransformReader.swift` | `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineTransformReaderTests.swift` | `5/5 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineQuantizationPolicyTests.swift` | self | `4/4 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Quantization/StreamingBF16AffineQuantizerTests.swift` | self | `5/5 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineTransformReaderTests.swift` | self | `5/5 pass` | 2026-09-15 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 5 - A local official snapshot validates offline

Status: `[x]` - Owner: `ingest` - Needs: `1` - Done: `2026-09-15` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-5)

- [x] 5.1 Replace the singleton with a closed source catalog  2026-09-15
- [x] 5.2 Implement no-follow local snapshot loading  2026-09-15
- [x] 5.3 Audit local index and header agreement  2026-09-15
- [x] 5.4 Add catalog and local-loader tests  2026-09-15
- [x] 5.5 Prove existing remote Gemma loading is unchanged  2026-09-15

**Acceptance**
- [x] A valid tiny local official-shaped snapshot validates without network access.
- [x] Every wrong file, digest, dtype, shape, range, or shard mapping fails before output creation.
- [x] The Gemma remote source descriptor and path remain unchanged.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Remote/SupportedModelSource.swift` | `Tests/TurboFieldfareRepack/Core/Remote/ModelSourceCatalogTests.swift` | `8/8 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Remote/ModelSourceCatalog.swift` | `Tests/TurboFieldfareRepack/Core/Remote/ModelSourceCatalogTests.swift` | `8/8 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift` | `Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift` | `32/32 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Format/IndexLoader.swift` | `Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift` | `32/32 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift` | `Tests/TurboFieldfareRepack/Core/Format/SafetensorsHostileHeaderTests.swift` | `40/40 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Remote/ModelSourceCatalogTests.swift` | self | `8/8 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift` | self | `32/32 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Remote/RemotePayloadCopyTests+Installation.swift` | self | `33/33 pass` | 2026-09-15 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 6 - A complete Qwen pack is sized before writing

Status: `[x]` - Owner: `planner` - Needs: `2, 4, 5` - Done: `2026-09-15` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-6)

- [x] 6.1 Decode exact Qwen architecture fields for planning  2026-09-15
- [x] 6.2 Plan resident and hybrid-state tensors  2026-09-15
- [x] 6.3 Plan routed experts and optional vision companion  2026-09-15
- [x] 6.4 Integrate family dispatch into the existing planner  2026-09-15
- [x] 6.5 Add exhaustive tiny Qwen planning tests  2026-09-15
- [x] 6.6 Re-run range-plan compatibility tests  2026-09-15

**Acceptance**
- [x] All 1,045 tensors are resident, streamed, vision, or one of 19 explicit MTP omissions.
- [x] Destination, scratch, alignment, and expert stride are known before a file is opened.
- [x] Repeated planning yields the same fingerprint and leaves Gemma plans unchanged.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Format/ArchInfo.swift` | `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` | `6/6 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Planning/QwenRepackPlanner.swift` | `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` | `6/6 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift` | `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` | `6/6 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` | self | `6/6 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Planning/RangeCopyPlannerTests.swift` | self | `4/4 pass` | 2026-09-15 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 7 - A tiny transformed pack resumes byte-identically

Status: `[x]` - Owner: `writer` - Needs: `2, 4, 6` - Done: `2026-09-15` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-7)

- [x] 7.1 Add transformed range output to WriterCore  2026-09-15
- [x] 7.2 Route Qwen resident writing through the transform  2026-09-15
- [x] 7.3 Bind checkpoints to source, policy, and plan digests  2026-09-15
- [x] 7.4 Preflight and audit transformed artifacts  2026-09-15
- [x] 7.5 Add resume, preflight, and audit tests  2026-09-15
- [x] 7.6 Re-run existing resume and disk-space suites  2026-09-15

**Acceptance**
- [x] Clean and resumed tiny Qwen conversions have identical directory digests.
- [x] Capacity failure, cancellation, corruption, and fingerprint changes fail without activation.
- [x] Existing Gemma writer, checkpoint, and disk behavior remains intact.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Writing/WriterCore.swift` | `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | `3/3 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Writing/TransformedTensorWriter.swift` | `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | `3/3 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Writing/ResidentWriter.swift` | `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | `8/8 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Remote/RemoteInstallCheckpoint.swift` | `Tests/TurboFieldfareRepack/Core/Remote/RemoteInstallCheckpointTests.swift` | `6/6 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/System/DiskSpaceChecker.swift` | `Tests/TurboFieldfareRepack/Core/System/DiskSpaceCheckerTests.swift` | `3/3 pass` | 2026-09-15 |
| `Sources/TurboFieldfareRepack/Core/Verification/RepackAudit.swift` | `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | `8/8 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | self | `3/3 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | self | `8/8 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/Remote/RemoteInstallCheckpointTests.swift` | self | `6/6 pass` | 2026-09-15 |
| `Tests/TurboFieldfareRepack/Core/System/DiskSpaceCheckerTests.swift` | self | `3/3 pass` | 2026-09-15 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 8 - Qwen Metal bindings have one checked contract

Status: `[x]` - Owner: `metal` - Needs: `2` - Done: `2026-09-16` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-8)

- [x] 8.1 Define host-side Qwen binding and layout types  2026-09-16
- [x] 8.2 Define matching common MSL declarations  2026-09-16
- [x] 8.3 Register the common module in one MetalContext edit  2026-09-16
- [x] 8.4 Add host layout and registration tests  2026-09-16

**Acceptance**
- [x] Host and shader layouts agree on every binding, offset, stride, and scalar width.
- [x] The common MSL module compiles without a placeholder compute pipeline.
- [x] Existing Metal resource loading remains unchanged.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen/QwenMetalContracts.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `7/7 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_common.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `7/7 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `7/7 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | self | `7/7 pass` | 2026-09-16 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 9 - Qwen full attention matches the tiny oracle

Status: `[x]` - Owner: `attention` - Needs: `2, 3, 8` - Done: `2026-09-16` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-9)

- [x] 9.1 Implement Qwen partial RoPE and Q/K normalization  2026-09-16
- [x] 9.2 Implement Qwen full-attention output gating  2026-09-16
- [x] 9.3 Implement ten-layer Qwen full KV storage  2026-09-16
- [x] 9.4 Add concrete full-attention Metal kernels  2026-09-16
- [x] 9.5 Add full-attention and KV numerical tests  2026-09-16

**Acceptance**
- [x] CPU-reference one-token and chunked Qwen full attention match P3 outputs and intermediates.
- [x] Ten full-attention KV layers preserve state across repeated decode.
- [x] Q/K norm, partial-RoPE, and sigmoid-gate Metal pipelines execute on supported hardware.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | `6/6 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenFullAttentionKVTests.swift` | `6/6 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_full_attention.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | `6/6 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `7/7 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | self | `6/6 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenFullAttentionKVTests.swift` | self | `6/6 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | self | `7/7 pass` | 2026-09-16 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 10 - Qwen linear attention matches the tiny oracle

Status: `[x]` - Owner: `deltanet` - Needs: `2, 3, 8` - Done: `2026-09-16` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-10)

- [x] 10.1 Implement bounded causal-convolution state  2026-09-16
- [x] 10.2 Implement one-token Gated DeltaNet update  2026-09-16
- [x] 10.3 Implement chunked prefill with exact state carry  2026-09-16
- [x] 10.4 Add concrete convolution and DeltaNet kernels  2026-09-16
- [x] 10.5 Add state, decode, and chunk numerical tests  2026-09-16

**Acceptance**
- [x] One-token Qwen convolution and DeltaNet outputs match P3.
- [x] Every chunk partition produces the same final output and FP32 state.
- [x] Cancellation does not recycle resources still in GPU use.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | `7/7 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift` | `7/7 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_linear_attention.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift` | `7/7 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `7/7 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | `7/7 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift` | `7/7 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `7/7 pass` | 2026-09-16 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 11 - Qwen MoE matches the tiny oracle

Status: `[x]` - Owner: `moe` - Needs: `2, 3, 8` - Done: `2026-09-16` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-11)

- [x] 11.1 Generalize manifest-driven expert layout  2026-09-16
- [x] 11.2 Implement deterministic Qwen Top-8 routing  2026-09-16
- [x] 11.3 Implement routed and sigmoid-gated shared experts  2026-09-16
- [x] 11.4 Add concrete Qwen MoE kernels  2026-09-16
- [x] 11.5 Add MoE math, paging, and lifetime tests  2026-09-16

**Acceptance**
- [x] Router choices, weights, routed output, and shared output match P3.
- [x] Expert layouts and cache lifetimes derive from validated v2 metadata.
- [x] Gemma expert streaming and layout remain unchanged.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`; authoritative candidate D4 is in the implementation evidence.

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Infrastructure/ModelIO/PackedExpertsLayout.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | `14/14 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | `14/14 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | `14/14 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_moe.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | `14/14 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `8/8 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | self | `14/14 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | self | `8/8 pass` | 2026-09-16 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 12 - A tiny Qwen runner emits the expected tokens

Status: `[x]` - Owner: `runner` - Needs: `2, 3, 9, 10, 11` - Done: `2026-09-16` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-12)

- [x] 12.1 Define the session-level family runtime boundary  2026-09-16
- [x] 12.2 Map Qwen resident and streamed weights  2026-09-16
- [x] 12.3 Implement concrete Qwen prefill and decode loops  2026-09-16
- [x] 12.4 Apply Qwen stop IDs and explicit sampling inputs  2026-09-16
- [x] 12.5 Add factory, loader, and runner tests  2026-09-16
- [x] 12.6 Re-run Gemma model and runner regression suites  2026-09-16

**Acceptance**
- [x] Metadata-only admission selects Gemma v1 or official Qwen v2 without reading payloads; runtime loading validates the declared payload before constructing a runner.
- [x] A complete synthetic four-layer Qwen text fixture covers normalized intermediates, final norm, untied-head logits, cached decode, and greedy IDs against an independent oracle.
- [x] Qwen uses explicit norms, an untied head, raw logits, Qwen stop IDs, and caller sampling; no Gemma scale, softcap, or MTP path is used.
- [x] Existing Gemma loader, sampling, generation, and stop behavior remain unchanged in meaning.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`; exact candidate aggregate and source D4 are recorded in the implementation evidence.

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | `7/7 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Inference/Model.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `11/11 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `11/11 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | `16/16 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Generation/Sampler.swift` | `Tests/TurboFieldfare/Core/Runtime/Generation/SamplerTests.swift` | `8/8 pass` | 2026-09-16 |
| `Package.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `11/11 pass` | 2026-09-16 |
| `scratch/qwen3.6-35b-a3b/fixture-generator/generate_qwen36_text_model_fixture.py` | none | `no-unit-test - scratch/qwen3.6-35b-a3b/evidence/phase-12/oracle` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/QwenFixtures/qwen36-tiny-text-model-fixtures.json` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `11/11 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | `7/7 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `11/11 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | `16/16 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/PackedExpertsLayout.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `11/11 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | `11/11 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | `11/11 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ModelLoaderTests.swift` (as-is) | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ModelLoaderTests.swift` | `27/27 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests.swift` (as-is) | `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests.swift` | `20/20 pass` | 2026-09-16 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 13 - Qwen turns commit or roll back atomically

Status: `[x]` - Owner: `state` - Needs: `12` - Done: `2026-09-16` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-13)

- [x] 13.1 Define family-neutral state transaction semantics  2026-09-16
- [x] 13.2 Implement Qwen committed and working state  2026-09-16
- [x] 13.3 Integrate transactions into conversation recovery  2026-09-16
- [x] 13.4 Apply Qwen stop and compaction policy in the app session  2026-09-16
- [x] 13.5 Add transaction and injected-boundary tests  2026-09-16
- [x] 13.6 Re-run existing Gemma recovery and stopping tests  2026-09-16

**Acceptance**
- [x] Soft Stop commits accepted Qwen tokens and all matching state.
- [x] Hard task cancellation and errors restore the committed text boundary.
- [x] Hidden stop suffix and checkpoint rebuild match a clean replay.
- [x] Qwen ignores the uncalibrated Gemma performance threshold.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`; correction2 candidate and source D4 are recorded in the implementation evidence.

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Generation/LogitProducer.swift` (unchanged regression input) | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | `16/16 pass` | 2026-09-16 |
|`Sources/TurboFieldfare/Runtime/Generation/ConversationStateTransaction.swift`|`Tests/TurboFieldfare/Core/Runtime/Generation/ConversationStateTransactionTests.swift`|`5/5 pass`|2026-09-16|
| `Sources/TurboFieldfare/Runtime/Generation/MultimodalConversation.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenConversationStateTests.swift` | `12/12 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenConversationStateTests.swift` | `12/12 pass` | 2026-09-16 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | `16/16 pass` | 2026-09-16 |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | `8/8 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Generation/ConversationStateTransactionTests.swift` | self | `5/5 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenConversationStateTests.swift` | self | `12/12 pass` | 2026-09-16 |
| `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | self | `21/21 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Generation/MultimodalConversationKVRecoveryTests.swift` (unchanged regression input) | self | `5/5 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Stopping.swift` (unchanged regression input) | self | `7/7 pass` | 2026-09-16 |
| `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Cancellation.swift` (unchanged regression input) | self | `2/2 pass` | 2026-09-16 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 14 - Qwen prompts match the pinned chat template

Status: `[x]` - Owner: `chat` - Needs: `1` - Done: `2026-09-17` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-14)

- [x] 14.1 Define model-neutral chat message and codec types  2026-09-17
- [x] 14.2 Load the verified Qwen tokenizer sidecars  2026-09-17
- [x] 14.3 Implement exact Qwen chat rendering  2026-09-17
- [x] 14.4 Add tokenizer and template golden tests  2026-09-17

**Acceptance**
- [x] Qwen encode/decode and incremental detokenization match pinned sidecars.
- [x] All supported prompt forms match pinned template token IDs.
- [x] Existing `GFTokenizer` and Gemma chat behavior remain unchanged.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`; accepted candidate aggregate: `de5b7863d40820aad311287a9711acc626706bfa53cda4e03cff50a34414829a`.

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Tokenization/ModelChatCodec.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | `21/21 pass` | 2026-09-17 |
| `Sources/TurboFieldfare/Tokenization/QwenTokenizer.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenTokenizerTests.swift` | `25/25 pass` | 2026-09-17 |
| `Sources/TurboFieldfare/Tokenization/QwenChatCodec.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | `21/21 pass` | 2026-09-17 |
| `Tests/TurboFieldfare/Core/Tokenization/QwenTokenizerTests.swift` | self | `25/25 pass` | 2026-09-17 |
| `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | self | `21/21 pass` | 2026-09-17 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

**Phase 14 evidence**

<a id="phase-14-evidence"></a>

The accepted engineering candidate is HEAD `5770260510935cd32b82c4008ff06ae6458539ff` with frozen five-path aggregate `de5b7863d40820aad311287a9711acc626706bfa53cda4e03cff50a34414829a`. Independent approvals `#119` and `#120` approved verification `#117`; Main accepted this exact candidate in event `#132`.

The original failed candidate remains explicit historical evidence only: aggregate `0129f5510eb16b043196acff3443c5707efa9de9365a7900c5874b91cf72e64d`, archived at `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/rejected-candidate-0129f551/`, with receipt `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/rejected-candidate-0129f551-identities.txt`. Its prior failures and all earlier Phase 14 evidence remain preserved.

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-17 | Qwen tokenizer selector | PASS; 25 tests in 1 suite, zero skip events; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-tokenizer.log` |
| 2026-09-17 | Qwen chat-template selector | PASS; 21 tests in 1 suite, zero skip events; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-chat-template.log` |
| 2026-09-17 | Combined `TokenizerTests` selector | PASS; 56 tests in 2 suites (Gemma plus Qwen), zero skip events; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-combined-tokenizer-selector.log` |
| 2026-09-17 | Combined `ChatTemplateTests` selector | PASS; 30 tests in 2 suites (Gemma plus Qwen), zero skip events; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-combined-chat-template-selector.log` |
| 2026-09-17 | Debug build `swift build --target TurboFieldfare` | PASS; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-build-debug.log` |
| 2026-09-17 | Release build `swift build -c release --target TurboFieldfare` | PASS; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-build-release.log` |
| 2026-09-17 | Source D4 and preservation rechecks | PASS; unchanged 83-path source/status union, unchanged 36 historical evidence files, and preserved rejected archive aggregate. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/candidate-source-d4.txt`; `d4-comparison.txt`; `historical-evidence-preservation.txt`; `archive-preservation-check.txt` |
| 2026-09-17 | Independent pinned references | PASS; trim runs and 47-value boundary JSON runs are byte-identical. The four finite `String(Double)` mismatches justified the smallest private notation adapter. The 100,000-finite-value domain comparison is sampled, not exhaustive. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/trim-chat-reference-receipt.txt`; `double-json-reference-receipt.txt`; `double-format-comparison.txt`; `double-formatter-domain-probe-receipt.txt` |

The accepted test boundary uses isolated regular copies for the authentic sidecar fixtures and exercises the full hostile admission matrix. Gemma behavior, v1 artifacts, pinned assets, and old failures remain unchanged. This phase does not claim authentic 35B inference, GPU execution, image payload validation, conversion, product readiness, or performance readiness.

P15 is pending and ready to begin; no P15 implementation is included in this acceptance.

## Phase 15 - Incomplete Qwen tool output dispatches nothing

Status: `[x]` - Owner: `tools` - Needs: `14` - Done: `2026-09-17` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-15)

- [x] 15.1 Implement incremental Qwen structured decoding
- [x] 15.2 Parse and validate Qwen tool grammar
- [x] 15.3 Select structured decoding through the codec boundary
- [x] 15.4 Add split-token and malformed-output tests
- [x] 15.5 Re-run Gemma parser and host truth tests unchanged

**Acceptance**
- [x] Complete Qwen calls become schema-valid host proposals with host IDs.
- [x] Every incomplete, malformed, unknown, or ambiguous call dispatches nothing.
- [x] Gemma parsing and VisionCapture host truth remain unchanged.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff` - accepted five-path aggregate `9fd6d7f6bb45dbeb15e12bdd2e4f76fca6bfa6999aea1bb4c6b667556c28e84a` - D4 pass

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Tokenization/QwenStructuredAssistantDecoder.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift` | `16/16 pass` | 2026-09-17 |
| `Sources/TurboFieldfare/Tokenization/QwenToolCallParser.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenToolCallParserTests.swift` | `20/20 pass` | 2026-09-17 |
| `Sources/TurboFieldfare/Tokenization/StructuredAssistantDecoder.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift`; `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | `25/25 pass` (16 Qwen decoder + 9 Gemma) | 2026-09-17 |
| `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift` | self | `16/16 pass` | 2026-09-17 |
| `Tests/TurboFieldfare/Core/Tokenization/QwenToolCallParserTests.swift` | self | `20/20 pass` | 2026-09-17 |
| `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | self | `44/44 pass` in 4 suites (19 OpenAI request-validation + 9 Gemma; other suites are non-additive supporting coverage) | 2026-09-17 |
| `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` | self | `11/11 pass` | 2026-09-17 |

The coverage-row total advances by seven, not by summing test cases. The 25 wrapper checks reuse the 16-test decoder suite and 9-test Gemma suite, and the OpenAI selector's 44 cases include its named 19-case OpenAI and 9-case Gemma suites plus supporting suites. These overlapping execution counts are evidence for rows, not additive program coverage rows.

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

**Phase 15 evidence**

The accepted engineering candidate is HEAD `5770260510935cd32b82c4008ff06ae6458539ff` with frozen five-path aggregate `9fd6d7f6bb45dbeb15e12bdd2e4f76fca6bfa6999aea1bb4c6b667556c28e84a`. Independent test signoff is recorded in events `#172` and `#179`; verification submission `#176` was approved by both final verifiers in `#177` and `#178`; Main accepted the same identity in `#189`.

Authoritative final receipts are under `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/`: the full Qwen selector passed 36 tests in 2 suites, the actual released-call codec round trip passed 1 test in 1 suite, all five frozen regressions passed, Debug and Release builds passed, and D4, preservation, preflight, hygiene, and fixture-cleanup checks passed. The real precision-rounding negative control remains preserved in the parent Phase 15 evidence directory.

Publication is finish-only and all-or-none: no call or host ID is released early. EOS is terminal and non-publishing with one permitted tail flush. The aggregate retained-call budget is 256 KiB and non-string JSON nesting is capped at 128 containers. The supported schema subset fails closed; integer-versus-decimal storage is deliberately narrow, canonically equivalent duplicate keys are rejected, and the progress surface remains Gemma-oriented. This is an additive parsing seam, not Qwen model inference, tool execution, product integration, GPU behavior, performance, or product-readiness evidence.

The first focused round-trip receipt is preserved as a test-design failure caused by parameter-order assumptions and selection of the instructional example frame; its corrected receipt passes. The real decimal precision negative control and the earlier stale-linked-test-bundle OpenAI crash remain labelled as failures with successful correction receipts. Flash self-reported one prohibited early literal `echo` command in events `#78` and `#146`; it produced no technical check beyond the literal, is not retroactively authorized, and is not a claim of a complete session audit. Final reviewers inspected the receipts rather than rerunning commands.

## Phase 16 - Qwen still images produce matching token rows

Status: `[x]` - Owner: `vision` - Needs: `2, 3, 8, 12, 13` - Done: `2026-09-18` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-16)

- [x] 16.1 Implement Qwen image geometry and preprocessing  2026-09-18
- [x] 16.2 Validate and map the Qwen vision companion  2026-09-18
- [x] 16.3 Implement the 27-layer Qwen vision tower and merger  2026-09-18
- [x] 16.4 Implement pad expansion and multimodal positions  2026-09-18
- [x] 16.5 Add image lineage to the Qwen state snapshot  2026-09-18
- [x] 16.6 Add concrete Qwen vision and M-RoPE kernels  2026-09-18
- [x] 16.7 Add preprocessing, pack, tower, position, and state tests  2026-09-18
- [x] 16.8 Re-run existing Gemma multimodal regression suites  2026-09-18

**Acceptance**
- [x] Bounded Qwen preprocessing produces pinned dynamic grids.
- [x] The matching companion produces one 2,048-wide feature row per expanded image pad.
- [x] Three-axis interleaved M-RoPE and decode delta match P3.
- [x] Image lineage rolls back and rebuilds with the rest of Qwen state.
- [x] Missing/invalid vision and video fail explicitly; Gemma vision remains unchanged.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff`; exact accepted 25-path aggregate: `e0ab778b26b857e08d0b466fdd2086eb10ad999cbed6bde00b37771cd0d19a89`.

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `8/8 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | `6/6 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_full_attention.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | `6/6 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_vision.metal` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionRuntimeTests.swift`; `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `11/11 + 8/8 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenConversationStateTests.swift` | `18/18 + 12/12 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | `16/16 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenImagePreprocessor.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenImagePreprocessorTests.swift` | `6/6 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenMultimodalPositions.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalPositionTests.swift` | `5/5 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionConfig.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenImagePreprocessorTests.swift` | `6/6 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionRuntimeTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalDecoderOracleTests.swift` | `11/11 + 4/4 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionWeightStoreTests.swift` | `9/9 pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Vision/MultimodalPrefillInput.swift` | `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPrefillInputTests.swift` | `5/5 shared/mixed-family pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Vision/MultimodalPromptRenderer.swift` | `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPromptRendererTests.swift` | `3/3 shared/mixed-family pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Vision/VisionWeightStore.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionWeightStoreTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Vision/VisionWeightStoreTests.swift` | `9/9 Qwen + 3/3 v1 pass` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | self | `8/8 pass` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTestArchitecture.swift` | `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPromptRendererTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPrefillInputTests.swift` | `3/3 + 5/5 pass through executed callers` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenImagePreprocessorTests.swift` | self | `6/6 pass` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalDecoderOracleTests.swift` | self | `4/4 pass; 4/4 with Metal API + GPU validation` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalDecoderReference.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalDecoderOracleTests.swift` | `4/4 pass through executed oracle` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalPositionTests.swift` | self | `5/5 pass` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` | self | `18/18 final-candidate pass; 18/18 with Metal API + GPU validation` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionRuntimeTests.swift` | self | `11/11 pass; 11/11 with Metal API + GPU validation` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionWeightStoreTests.swift` | self | `9/9 pass` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPrefillInputTests.swift` | self | `5/5 shared/mixed-family pass` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPromptRendererTests.swift` | self | `3/3 shared/mixed-family pass` | 2026-09-18 |
| `Sources/TurboFieldfare/Runtime/Vision/VisionRuntime.swift` (planned, byte-unchanged; regression-only) | `QwenVisionRuntimeTests` 11/11 separately; `frozen-multimodalprefillregressiontests.log` 7/7 in 2 suites; `frozen-multimodalsuffixprefilltests.log` 9/9 in 2 suites | `no-unit-test - unchanged path; named adjacent regressions passed, never direct changed-source execution` | 2026-09-18 |
| `Tests/TurboFieldfare/Core/Runtime/Vision/VisionWeightStoreTests.swift` (planned, byte-unchanged; regression-only) | self | `3/3 synthetic v1 pass` | 2026-09-18 |

The 27 rows are the accepted 25-path P16 candidate plus the two explicitly planned-but-unchanged regression rows. The mixed result is 26 executed-pass rows plus one allowed `no-unit-test` proof row, so the phase and program totals use **satisfied**, not an assertion that every row was an executed test. Coverage arithmetic is: prior pending `107 = 19 + 9 + 7 + 8 + 21 + 7 + 10 + 9 + 9 + 8`; closing the original 19 P16 rows leaves `107 - 19 = 88` pending; adding the eight omitted P16 paths changes `218 + 8 = 226` total; therefore `226 - 88 = 138` satisfied and `138 + 88 = 226`.

Only the transaction-correction vision-state `18/18`, scalar-state `12/12`, Metal-validation vision-state `18/18`, Debug build, and Release build receipts were refreshed on the exact final identity. Every other count above is a historical Phase 16 root receipt applied only because `transaction-correction/preservation-comparison-final.txt` proves all 23 other candidate inputs byte-identical; those receipts were not rerun after the correction. The unchanged `VisionRuntime.swift` proof uses separately recorded Qwen runtime 11/11 plus the named adjacent Gemma regression receipts `frozen-multimodalprefillregressiontests.log` (7/7 in 2 suites) and `frozen-multimodalsuffixprefilltests.log` (9/9 in 2 suites); none is represented as direct execution of that unchanged source row.

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 17 - Decode service binds responses to loaded identity

Status: `[x]` - Owner: `Astra / Sol / Luna` - Needs: `12, 13, 15` - Done: `2026-09-22` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-17)

- [x] 17.1 Add descriptor and model identity to decode protocol  2026-09-22
- [x] 17.2 Load one verified family in the decode service  2026-09-22
- [x] 17.3 Bind app inference requests to identity plus epoch  2026-09-22
- [x] 17.4 Add protocol round-trip and compatibility tests  2026-09-22
- [x] 17.5 Add service load, cancel, reset, and stale-response tests  2026-09-22
- [x] 17.6 Re-run existing service gate and response tests  2026-09-22

**Acceptance**
- [x] Ready reports the descriptor produced by verified model loading.
- [x] Model identity and conversation epoch jointly scope every stateful request and response.
- [x] Failed load, reset, cancel, unload, and reconnect cannot leak stale state.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Package.swift` | none | `no-unit-test - actual DecodeService product link and nine target builds pass` | 2026-09-22 |
| `Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | `7/7 pass` | 2026-09-22 |
| `Sources/TurboFieldfareDecodeService/Entry.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | `15/15 pass` | 2026-09-22 |
| `Sources/TurboFieldfareDecodeService/DecodeCommandQueue.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | `15/15 pass` | 2026-09-22 |
| `Sources/TurboFieldfareDecodeService/DecodeServiceSession.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | `15/15 pass` | 2026-09-22 |
| `Sources/TurboFieldfareDecodeService/DecodeServiceOutbox.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `8/8 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Inference/AppInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | `7/7 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Inference/DecodeServiceInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | `26/26 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Inference/DecodeServiceResponseRouter.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceResponseMatchingTests.swift` | `5/5 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | `7/7 pass` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | `7/7 pass` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | `7/7 pass` | 2026-09-22 |
| `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | `15/15 pass` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | `26/26 pass` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceConnectionInvalidationTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceConnectionInvalidationTests.swift` | `12/12 pass` | 2026-09-22 |
| `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `8/8 pass` | 2026-09-22 |
| `Tests/TurboFieldfareDecodeService/DecodeConversationGateTests.swift` | `Tests/TurboFieldfareDecodeService/DecodeConversationGateTests.swift` | `10/10 pass` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceResponseMatchingTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceResponseMatchingTests.swift` | `5/5 pass` | 2026-09-22 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 18 - CLI runs verified Qwen requests

Status: `[x]` - Owner: `Sol / Luna` - Needs: `12, 15, 16` - Done: `2026-09-22` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-18)

- [x] 18.1 Add explicit Qwen-capable CLI options and help  2026-09-22
- [x] 18.2 Select family runtime and codec in CLI Run  2026-09-22
- [x] 18.3 Add CLI argument and help tests  2026-09-22
- [x] 18.4 Add tiny CLI run routing tests  2026-09-22
- [x] 18.5 Re-run existing CLI argument and image-order suites  2026-09-22

**Acceptance**
- [x] CLI text, chat, thinking, tools, and still images route through verified family components.
- [x] CLI reports loaded identity and rejects missing vision or video explicitly.
- [x] Existing Gemma CLI defaults and behavior remain unchanged.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff` with phase-start input snapshots

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareCLI/Args.swift` | `Tests/TurboFieldfare/Core/CLI/CLIQwenArgumentsTests.swift` | `5/5 pass` | 2026-09-22 |
| `Sources/TurboFieldfareCLI/Run.swift` | `Tests/TurboFieldfare/Core/CLI/CLIQwenRunTests.swift` | `9/9 pass` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | `10/10 pass` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | `10/10 pass` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/CLI/CLIQwenArgumentsTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIQwenArgumentsTests.swift` | `5/5 pass` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/CLI/CLIQwenRunTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIQwenRunTests.swift` | `9/9 pass` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/CLI/CLIArgumentsTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIArgumentsTests.swift` | `17/17 pass` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/CLI/CLIImageOrderTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIImageOrderTests.swift` | `1/1 pass` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/CLI/CLIPromptInputTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIPromptInputTests.swift` | `7/7 pass; 1 expected Apple7-only hardware skip` | 2026-09-22 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 19 - Loopback server serves verified Qwen chat

Status: `[x]` - Owner: `Sol / Luna` - Needs: `12, 15, 16` - Done: `2026-09-22` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-19)

- [x] 19.1 Derive server model identity from the descriptor
- [x] 19.2 Select family runtime and codec in server inference
- [x] 19.3 Expose verified model data in API responses
- [x] 19.4 Add server identity and inference tests
- [x] 19.5 Re-run HTTP, prompt-cache, and validation suites

**Acceptance**
- [x] Server identity comes from the verified descriptor or rejects an incompatible assertion.
- [x] Text, tools, thinking, images, and streaming use the selected family codec.
- [x] Prompt caches never cross identity/template/image lineage.
- [x] The server still binds only 127.0.0.1 and rejects video.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff` with phase-start server snapshots

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareServer/Core/ServerArguments.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift; Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | `11/11 identity + 44/44 validation-related pass` | 2026-09-22 |
| `Sources/TurboFieldfareServer/Core/ServerInference.swift` | `Tests/TurboFieldfareServer/ServerQwenInferenceTests.swift` | `7/7 inference pass` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift; Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | `11/11 identity + 10/10 runtime pass` | 2026-09-22 |
| `Sources/TurboFieldfareServer/Core/OpenAIModels.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | `11/11 identity pass` | 2026-09-22 |
| `Sources/TurboFieldfareServer/Command/main.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | `server build + 11/11/11 identity pass` | 2026-09-22 |
| `Sources/TurboFieldfareServer/Core/HTTPServer.swift` | `Tests/TurboFieldfareServer/HTTPServerTests.swift; Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | `19/19 HTTP + 11/11/11 identity pass` | 2026-09-22 |
| `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | `11/11 pass` | 2026-09-22 |
| `Tests/TurboFieldfareServer/ServerQwenInferenceTests.swift` | `Tests/TurboFieldfareServer/ServerQwenInferenceTests.swift` | `7/7 pass` | 2026-09-22 |
| `Tests/TurboFieldfareServer/HTTPServerTests.swift` | `Tests/TurboFieldfareServer/HTTPServerTests.swift` | `19/19 pass` | 2026-09-22 |
| `Tests/TurboFieldfareServer/ServerPromptCacheTests.swift` | `Tests/TurboFieldfareServer/ServerPromptCacheTests.swift` | `16/16 pass` | 2026-09-22 |
| `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | `44/44 across 4 suites pass` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | `10/10 pass` | 2026-09-22 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document


## Phase 20 - The app selects Qwen without moving Gemma

Status: `[~]` - Owner: `app` - Needs: `6, 12, 15, 16, 17` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-20)

- [x] 20.1 Add stable app model catalog and separate locations  2026-09-22
- [x] 20.2 Persist selection with backward-compatible settings  2026-09-22
- [x] 20.3 Bind app lifecycle to model identity and epoch  2026-09-22
- [x] 20.4 Route installation and vision status by selection  2026-09-22
- [~] 20.5 Add a visible accessible model picker
- [~] 20.6 Report selected identity and family diagnostics
- [x] 20.7 Route ordinary Qwen app and Agent turns through retained family generation  2026-09-22
- [x] 20.8 Add catalog, lifecycle, install, picker, and Agent tests  2026-09-22
- [x] 20.9 Re-run existing app lifecycle and VisionCapture truth tests  2026-09-22

**Acceptance**
- [x] Settings without selection still choose Gemma.
- [x] Picker selection unloads the old runtime before verified load and starts a new epoch.
- [x] Qwen text/vision install state and paths remain separate from Gemma.
- [ ] Failed Qwen load leaves Gemma selectable and no stale identity visible.
- [x] Agent Mode uses the selected codec while host verified evidence stays authoritative.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff` with phase-start snapshots

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareApp/Core/Installation/AppModelCatalog.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift` | `pass - bounded unit selectors 6/6; 5/5` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallDescriptor.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | `pass - bounded unit selectors 6/6; 8/8` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift` | `pass - bounded unit selectors 6/6; 5/5` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift` | `Tests/TurboFieldfareApp/Core/Configuration/AppModelSelectionSettingsTests.swift`; `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift` | `pass - bounded unit selectors 4/4; 17/17` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/State/AppModel.swift` | `Tests/TurboFieldfareApp/Core/State/AppModelSelectionTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelLoadPhaseOrderTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelConversationTests.swift` | `pass - bounded unit selectors 9/9; 5/5; 12/12` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Installation/RepackModelInstallerClient.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallTests.swift` | `pass - bounded unit selectors 8/8; 14/14` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Installation/RepackVisionPackInstallerClient.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | `pass - bounded unit selectors 8/8` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Mac/App/RootView.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | `pending - direct UI acceptance; helper selectors 6/6 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Mac/App/TurboFieldfareMacApp.swift` | native UI evidence; existing AppModelSelectionTests | `pending - clean build and keyboard rollback proved; available-Qwen/VoiceOver open` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | `pending - direct UI acceptance; helper selectors 6/6; 8/8 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallationProbeTests.swift` | `pending - direct UI acceptance; helper selectors 6/6; 5/5 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Tools/VisionCaptureToolLoop.swift` | `Tests/TurboFieldfareApp/Core/Tools/AppAgentCodecRoutingTests.swift`; `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` | `pass - bounded unit selectors 4/4; 11/11` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift` | `pass - bounded unit selectors 6/6` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Configuration/AppModelSelectionSettingsTests.swift` | `Tests/TurboFieldfareApp/Core/Configuration/AppModelSelectionSettingsTests.swift` | `pass - bounded unit selectors 4/4` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/State/AppModelSelectionTests.swift` | `Tests/TurboFieldfareApp/Core/State/AppModelSelectionTests.swift` | `pass - bounded unit selectors 9/9` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | `pass - bounded unit selectors 8/8` | 2026-09-22 |
| `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | `pass - bounded unit selectors 6/6` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Tools/AppAgentCodecRoutingTests.swift` | `Tests/TurboFieldfareApp/Core/Tools/AppAgentCodecRoutingTests.swift` | `pass - bounded unit selectors 4/4` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift` | `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift` | `pass - bounded unit selectors 17/17` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/State/AppModelLoadPhaseOrderTests.swift` | `Tests/TurboFieldfareApp/Core/State/AppModelLoadPhaseOrderTests.swift` | `pass - bounded unit selectors 5/5` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/State/AppGenerationRunIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/State/AppGenerationRunIdentityTests.swift` | `pass - bounded unit selectors 1/1` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` | `pass - bounded unit selectors 11/11` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | `pass - bounded unit selectors 13/13; 27/27` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift` | `pass - bounded unit selectors 13/13` | 2026-09-22 |
| `Sources/TurboFieldfare/Tokenization/QwenChatCodec.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift`; `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | `pass - bounded unit selectors 13/13; 21/21` | 2026-09-22 |
| `Sources/TurboFieldfare/Tokenization/QwenStructuredAssistantDecoder.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift`; `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift` | `pass - bounded unit selectors 13/13; 16/16` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift` | `pass - bounded unit selectors 27/27; 13/13` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallationProbeTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | `pass - bounded unit selectors 5/5; 8/8` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Mac/Components/ModelStatusBadge.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | `pending - direct UI acceptance; helper selectors 6/6 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Mac/Generation/OutputPaneView.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelTests.swift` | `pending - direct UI acceptance; helper selectors 6/6; 26/26 pass` | 2026-09-22 |
| `Sources/TurboFieldfareApp/MacPresentation/InstructionTranscriptDocumentController.swift` | `Tests/TurboFieldfareApp/Core/State/AppModelTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelConversationTests.swift` | `pending - direct UI acceptance; helper selectors 26/26; 12/12 pass` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift` | `pass - bounded unit selectors 13/13` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | `pass - bounded unit selectors 27/27` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallationProbeTests.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallationProbeTests.swift` | `pass - bounded unit selectors 5/5` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Installation/AppVisionPackInstallationProbe.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | `pass - bounded unit selectors 8/8` | 2026-09-22 |
| `Sources/TurboFieldfareApp/MacPresentation/AppModelIdentityPresentation.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | `pass - bounded unit selectors 6/6` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Vision/MultimodalPromptRenderer.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift`; `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | `pass - bounded unit selectors 13/13; 21/21` | 2026-09-22 |
| `Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 5/5; 27/27; 9/9` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Inference/Model.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 5/5; 27/27; 9/9` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 5/5; 27/27; 9/9` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 5/5; 27/27; 9/9` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Inference/AppInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 5/5; 27/27; 9/9` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Diagnostics/AppDiagnostics.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 5/5; 27/27; 9/9` | 2026-09-22 |
| `Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 5/5; 9/9` | 2026-09-22 |
| `Sources/TurboFieldfareDecodeService/DecodeServiceOutbox.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift` | `pass - bounded unit selectors 9/9; 5/5` | 2026-09-22 |
| `Sources/TurboFieldfareDecodeService/Entry.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift` | `pass - bounded unit selectors 9/9; 5/5` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Inference/DecodeServiceInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 5/5; 27/27; 9/9` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift` | `pass - bounded unit selectors 5/5` | 2026-09-22 |
| `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `pass - bounded unit selectors 9/9` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | `pass - bounded unit selectors 26/26` | 2026-09-22 |

Evidence: `coordination/p20-acceptance-reconciliation.md` and its dated keyboard addendum map all 50 rows to passing logs and UI limits. Tasks 20.1–20.4 and 20.7–20.9 are accepted within their stated unit scope. The split regression receipts satisfy 20.9 because the requirement is four nonzero suites, not one combined command. The bounded VoiceOver attempt reached first-use setup but did not verify model identification or selection; receipt: `execution/luna6-voiceover-20260922T192634Z/`. Phase 20 remains open for 20.5–20.6.

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document


## Phase 21 - Authorized conversion writes Qwen beside Gemma

Status: `[~]` - Owner: `operator` - Needs: `2, 4, 6, 7` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-21)

- [x] 21.1 Add explicit local official-source CLI mode
- [x] 21.2 Connect CLI mode to plan, transform, audit, and receipt
- [x] 21.3 Add local CLI and workflow unit tests
- [ ] 21.4 Record current capacity and process preflight
- [ ] 21.5 Run one authorized official conversion with resume evidence
- [ ] 21.6 Verify receipts, output digests, and no-Gemma mutation

**Acceptance**
- [ ] Authorized conversion uses only the canonical pinned official source.
- [ ] Preflight uses current capacity and stops before writing when insufficient.
- [ ] A complete v2 text artifact and matching vision companion publish only after audit.
- [ ] Resume is byte-identical and no Gemma path or official source file changes.

**Unit test coverage**

Base: `5770260510935cd32b82c4008ff06ae6458539ff` with phase-start snapshots

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Command/main.swift` | `Tests/TurboFieldfareRepack/Core/Command/QwenLocalRepackCLITests.swift` | `17/17 pass - tiny fixtures` | 2026-09-22 |
| `Sources/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepacker.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | `13/13 pass - tiny fixtures` | 2026-09-22 |
| `Tests/TurboFieldfareRepack/Core/Command/QwenLocalRepackCLITests.swift` | self | `17/17 pass - tiny fixtures` | 2026-09-22 |
| `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | self | `13/13 pass - tiny fixtures` | 2026-09-22 |
| `scratch/qwen3.6-35b-a3b/evidence/phase-21/preflight.txt` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-21/conversion.log` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-21/verification.md` (PROPOSED) | none yet | `missing - phase not started` | - |
| `Sources/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifier.swift` | `Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift` | `7/7 pass - tiny fixtures` | 2026-09-22 |
| `Sources/TurboFieldfareRepack/Core/Writing/TransformedTensorWriter.swift` | `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | `15/15 pass - tiny fixtures` | 2026-09-22 |
| `Sources/TurboFieldfareRepack/Core/Verification/VerifiedInstallReceiptWriter.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | `13/13 pass - tiny fixtures` | 2026-09-22 |
| `Sources/TurboFieldfareRepack/Core/Verification/RepackAudit.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | `13/13 pass - tiny fixtures` | 2026-09-22 |
| `Sources/TurboFieldfareRepack/Core/System/DiskSpaceChecker.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | `13/13 pass - tiny fixtures` | 2026-09-22 |
| `Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift` | self | `7/7 pass - tiny fixtures` | 2026-09-22 |
| `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | self | `15/15 pass - tiny fixtures` | 2026-09-22 |
| `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | self | `10/10 pass - tiny fixtures` | 2026-09-22 |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/VerifiedInstallReceipt.swift` | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/QwenReceiptPathBindingTests.swift` | `3/3 pass - tiny fixtures` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/QwenReceiptPathBindingTests.swift` | self | `3/3 pass - tiny fixtures` | 2026-09-22 |


**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document


## Phase 22 - Real Qwen text behavior matches the quantized reference

Status: `[ ]` - Owner: `text-verify` - Needs: `12, 13, 15, 21` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-22)

- [~] 22.1 Add separate real-text integration test cases
- [ ] 22.2 Preflight the authorized real-text session
- [ ] 22.3 Build the affected release products once
- [ ] 22.4 Compare real logits and greedy tokens to the same weights
- [ ] 22.5 Verify soft Stop, hard cancel, suffix, and rebuild
- [ ] 22.6 Verify real thinking and tool output behavior
- [ ] 22.7 Run Metal API and shader validation on text kernels
- [ ] 22.8 Obtain independent text evidence review

**Acceptance**
- [ ] Release products build with no new attributable diagnostics.
- [ ] Real logits/tokens match the independent same-quantized-weight reference.
- [ ] Stop, cancel, suffix removal, and rebuild preserve transactional semantics.
- [ ] Thinking/tools behave exactly and malformed output dispatches nothing.
- [ ] Text kernels pass Metal API/shader validation on supported hardware.
- [ ] Independent review passes the exact candidate.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Scripts/qwen36_quantized_reference.py` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealTextParityTests.swift` | `missing - Swift compiled, same-pack parity not run` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealTextParityTests.swift` | self | `missing - compiled, opt-in test not run` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealStateRecoveryTests.swift` | self | `missing - compiled, opt-in tests not run` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealToolCodecTests.swift` | self | `missing - compiled, opt-in tests not run` | 2026-09-22 |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/preflight.txt` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/release-build.log` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/text-parity.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/state-recovery.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/thinking-tools.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/metal-validation.log` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/review.txt` (PROPOSED) | none yet | `missing - phase not started` | - |
| `Tests/Python/test_qwen36_quantized_reference.py` | self | `16/16 pass - helper checks only, no model` | 2026-09-22 |


**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 23 - Real Qwen screenshots preserve verified workflow truth

Status: `[ ]` - Owner: `vision-verify` - Needs: `16, 20, 22` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-23)

- [ ] 23.1 Add separate real-image integration test cases
- [ ] 23.2 Authorize and record target plus tool permissions
- [ ] 23.3 Compare real image features and positions
- [ ] 23.4 Verify invalid companion and unsupported video failures
- [ ] 23.5 Verify image checkpoint and app attachment lifecycle
- [ ] 23.6 Run the scoped VisionCapture QA workflow once
- [ ] 23.7 Obtain independent vision and workflow review

**Acceptance**
- [ ] Real screenshot grids, features, pad rows, and M-RoPE meet fixed reference criteria.
- [ ] Invalid/missing companions and video fail closed without damaging canonical artifacts.
- [ ] Image state survives Stop/rebuild and rejects changed lineage.
- [ ] VisionCapture conclusions come only from authorized host-verified evidence.
- [ ] Independent review passes the exact candidate and artifacts.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenRealVisionIntegrationTests.swift` | self | `pending - compiled, authentic execution gated` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionFailClosedIntegrationTests.swift` | self | `pending - compiled, authentic execution gated` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenRealImageStateTests.swift` | self | `pending - compiled, authentic execution gated` | 2026-09-22 |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/authorization.md` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/vision-parity.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/fail-closed.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/image-state.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/visioncapture-qa.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/review.txt` (PROPOSED) | none yet | `missing - phase not started` | - |
| `Sources/TurboFieldfare/Runtime/Vision/Preprocessing/VisionImageSource.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppImageAttachmentStoreTests.swift` | `pass - 4/4 store and 4/4 batch tests; 36/36 attachment regressions` | 2026-09-22 |
| `Sources/TurboFieldfareApp/Core/Inference/AppImageAttachment.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppImageAttachmentStoreTests.swift` | `pass - 4/4 store and 4/4 batch tests; 36/36 attachment regressions` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/Inference/AppImageAttachmentStoreTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppImageAttachmentStoreTests.swift` | `pass - 4/4 store and 4/4 batch tests; 36/36 attachment regressions` | 2026-09-22 |
| `Tests/TurboFieldfareApp/Core/State/AppImageAddConcurrencyTests.swift` | `Tests/TurboFieldfareApp/Core/State/AppImageAddConcurrencyTests.swift` | `pass - 4/4 store and 4/4 batch tests; 36/36 attachment regressions` | 2026-09-22 |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` | `pass - 19/19 tiny retained-image tests` | 2026-09-22 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` | `pass - 19/19 tiny retained-image tests` | 2026-09-22 |
| `Scripts/qwen36_quantized_vision_reference.py` | `Tests/Python/test_qwen36_quantized_vision_reference.py` | `pending - 16/16 synthetic units pass; authentic numerical execution gated` | 2026-09-22 |
| `Tests/Python/test_qwen36_quantized_vision_reference.py` | self | `pass - 16/16 synthetic standard-library tests` | 2026-09-22 |


**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document



## Phase 24 - A controlled Gemma-Qwen comparison is recorded

Status: `[ ]` - Owner: `comparison` - Needs: `22, 23` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-24)

Preparation only, 2026-09-22: exact historical prompts, a visibly versioned guide, and a twelve-run comparison draft are staged in Phase 24 evidence. Current CLI differences are explicit and GPT-6 Luna owns the bounded validation review. Live restoration and all seven tasks remain pending Phases 22 and 23. No benchmark has run.

- [ ] 24.1 Restore and review the historical benchmark guide
- [ ] 24.2 Restore the three frozen benchmark prompts
- [ ] 24.3 Pre-register comparison cases and deviations
- [ ] 24.4 Record same-hardware environment and baselines
- [ ] 24.5 Run paired community text cases in alternating order
- [ ] 24.6 Run paired VisionCapture quality cases
- [ ] 24.7 Publish comparison with uncertainty and no speed promise

**Acceptance**
- [ ] The historical guide and three prompts are restored with provenance.
- [ ] Cases, metrics, validity rules, and Qwen deviations are fixed before runs.
- [ ] Paired runs use one machine and one model process in alternating order.
- [ ] Every result includes full timing/error, tokens, settings, stop, and validity.
- [ ] Conclusions report uncertainty and make no unmeasured speed promise.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `docs/COMMUNITY_BENCHMARKS.md` (PROPOSED) | none yet | `missing - phase not started` | - |
| `docs/benchmark-prompts/real-generation-v1/short-explanation.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `docs/benchmark-prompts/real-generation-v1/medium-review.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `docs/benchmark-prompts/real-generation-v1/long-synthesis.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/protocol.md` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/environment.txt` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/text-results.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/qa-results.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/comparison.md` (PROPOSED) | none yet | `missing - phase not started` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 25 - Opt-in Qwen can roll back to Gemma

Status: `[ ]` - Owner: `release-verify` - Needs: `18, 19, 20, 22, 23, 24` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-25)

- [ ] 25.1 Freeze exact candidate and evidence index
- [ ] 25.2 Run final independent behavioral verification
- [ ] 25.3 Run final independent architecture review
- [ ] 25.4 Stage and install the authorized app candidate
- [ ] 25.5 Verify opt-in picker and cross-surface identity
- [ ] 25.6 Exercise one-action Gemma rollback
- [ ] 25.7 Clean up only dispensable artifacts this work item created

**Acceptance**
- [ ] Final test and architecture reviewers pass the exact frozen candidate.
- [ ] Authorized installed app launches with Gemma default and Qwen opt-in.
- [ ] App, CLI, and server report the same verified Qwen identity.
- [ ] One visible action rolls back to verified Gemma without artifact mutation.
- [ ] Only work-item-owned dispensable temporary artifacts are removed.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `none - this phase changes no production or test code` | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/candidate.md` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/test-review.txt` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/architecture-review.txt` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/install.log` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/opt-in-launch.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/rollback.json` (PROPOSED) | none yet | `missing - phase not started` | - |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/cleanup.txt` (PROPOSED) | none yet | `missing - phase not started` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Changelog
| Date | Phase | What happened |
|---|---|---|
| 2026-09-15 | - | Version 1 rewritten with concrete tasks, real paths, dependencies, and coverage; awaiting approval. |
| 2026-09-15 | - | The historical creation note above predates the recorded approval; version 1 approval is recorded, document status is in-progress, and Phase 2 is ready but not started. |
| 2026-09-15 | 2 | Phase 2 started at base `5770260510935cd32b82c4008ff06ae6458539ff`; task 2.1 is in progress. |
| 2026-09-15 | 2 | Candidate `b833f068321d148c2370858736736ad829a2e3cb900161e0c7e40ee78bcf6241` was rejected at `#90`: decoded vision lacked bound text identity. |
| 2026-09-15 | 2 | Fix: descriptor carries the text-manifest SHA-256 and Codable requires companion equality; regression `#105` passed. |
| 2026-09-15 | 2 | Fresh builds and suites pass for candidate `5ddb0e82337fb5716005c3584c0716bb0f954737177921ffb8f49265e3e6b597`; both approvals remain pending. |
| 2026-09-15 | 2 | Correction: Terra #119 and Grok #121 approved verification #117; Phase 2 closed; Phase 3 ready, not started. |
| 2026-09-15 | 3 | Phase 3 closed: deterministic tiny text and vision fixtures plus 6/6 tests pass. |
| 2026-09-15 | 3 | Correction: the prior close was premature; Phase 3 remains in review until both current exact-candidate verifiers approve. |
| 2026-09-15 | 3 | Correction: Terra #140 and Grok #142 formally approved verification #139; D1-D5 complete. Phase 3 is now closed; Phase 4 is ready and not started. |
| 2026-09-15 | 4 | Phase 4 started at base `5770260510935cd32b82c4008ff06ae6458539ff`; production API is frozen and independent tests are in progress. |
| 2026-09-15 | 4 | Terra #135 and Grok #137 approved submission #133; Phase 4 closed; Phase 5 is ready but not started. |
| 2026-09-15 | 5 | Closed after #51/#53/#57; 8/8 coverage rows pass; D4 PASS (23 inherited + 7 P5). Phase 6 ready. |
| 2026-09-15 | 6 | Closed after verification #93, Grok #95/#98, and Terra #96; 6/6 coverage rows pass; D4 PASS (30 inherited + 4 P6). Phase 7 ready. |
| 2026-09-15 | 7 | Closed after #57/#59/#62/#64/#65; 10/10 coverage rows pass; D4 PASS (34 inherited + 9 P7). Phase 8 ready. |
| 2026-09-16 | 8 | Closed after #74/#75/#76/#81; 4/4 coverage rows pass; D4 PASS (43 inherited + 4 P8). Phase 9 ready. |
| 2026-09-16 | 9 | Closed after the exact candidate checks; 7/7 coverage rows pass; project D4 PASS (45 inherited + 7 P9). Phases 10 and 11 are ready, not started. |
| 2026-09-16 | 10 | Closed: 7/7 rows pass; D4 has 57 files (50 inherited + 7 P10) and 12 directory markers. State, DeltaNet, Registry ran 7 each; P11 ready. |
| 2026-09-16 | 11 | Closed on reviewed candidate; 5/5 tasks, 3/3 acceptance, 7/7 coverage, D1-D5 pass. See Phase 11 evidence for identity and approvals; P12 follows. |
| 2026-09-16 | 12 | Stage A oracle/runner refinement recorded; P12 remains in progress with 13 pending rows. Stage B requires Main's explicit release. |
| 2026-09-16 | 12 | Correction: #73 released Stage B after Stage A review/install; #92/#93 found a CPU-only runner; #104 rejected the waiver. Hybrid correction next. |
| 2026-09-16 | 12 | Closed on final2 exact candidate: 6/6 tasks, 4/4 acceptance, 16/16 coverage rows, and D1-D5 pass; P13 is next and P13–25 remain pending. |
| 2026-09-16 | 13 | Closed on correction2 exact candidate: 6/6 tasks, 4/4 acceptance, 12/12 coverage rows, and D1-D5 pass; P14 planning is underway and implementation is held. |

| 2026-09-22 | 18 | Phase-start baseline recorded. Sol drafts CLI changes outside compiled inputs while Luna completes Phase 17 tests. No Phase 18 behavior is accepted yet. |

| 2026-09-22 | 17 | Main accepted the final sixteen-input candidate after nine builds, two source reviews, 165 focused tests and native process cleanup proof. Phase 18 production and tests now proceed in parallel. |

| 2026-09-22 | 19 | Server phase baseline and staged design recorded. Two required startup/HTTP source paths added to coverage before editing. P18 test inputs remain frozen. |

| 2026-09-22 | App reload enables ten agent slots. P18 remains open for a test-timeout repair. P19 coverage accounts for the necessary companion-validation helper before live promotion. |

| 2026-09-22 | P20 retained runtime/catalog/lifecycle and P21 complete local workflow enter staged implementation with disjoint Sol/Luna ownership. Baselines and all additional source/test coverage rows recorded before live promotion; real conversion remains separately authorized. |

| 2026-09-22 | P18 accepted after final build, 49 passes plus one expected hardware skip, independent timeout repair and focused 10-test rerun. All 645 final input hashes match. P19 promotion/build released; P20/P21 remain staged. |

| 2026-09-22 | Server product build passed with six promoted files and unchanged Gemma server session. P20 adds the necessary family-aware vision-installation probe to coverage before live promotion. |

| 2026-09-22 | 19 | Accepted corrected companion admission, server build and 113 focused passing tests. Six production paths and three test paths frozen; only one existing tiny-fixture expectation changed after server regression runs. App/conversion remain staged. |
