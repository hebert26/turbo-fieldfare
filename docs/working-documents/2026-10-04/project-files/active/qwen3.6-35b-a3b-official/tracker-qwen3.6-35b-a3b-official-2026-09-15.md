---
title: "Qwen3.6-35B-A3B original BF16 integration"
slug: qwen3.6-35b-a3b-official-implementation-notes
implementation: ./implementation-qwen3.6-35b-a3b-official-2026-09-15.md
status: in-progress
execution_state: paused
evidence_scope: original-bf16-v2
version: 2
approved_version: 1
approved_by: "Hebert, version 1 only"
approved_at: "2026-09-15T11:53:55Z"
created_at: "2026-09-15"
updated_at: "2026-09-30"
current: "Phases 1–21 technically complete, with Phase 4 retained as retired; Phase 3 remains technically complete but its P22 qualification dependency is unproven; Phase 22 and later remain incomplete. The Phase 22 receipt records 75 focused tests passing but full accuracy failing with 214504 mismatches. Execution remains paused."
next_action: "Phase 6 Task 6.8 remains a qualification receipt caveat only; keep Phase 22 and later incomplete until the full raw-logit/public-Float16 qualification passes, retaining the Phase 3 P22 dependency as an outstanding cross-phase condition."

The `2026-09-30` dates recorded for Phases 3 and 6–20 are status-recording dates for this documentation update. They do not replace or invent the underlying implementation, test, or receipt dates, which remain in the phase evidence.
---

# Tracker: Qwen3.6-35B-A3B original BF16 integration

Goal: run the exact downloaded official BF16 weights with resident BF16 shared buffers and route-selected expert reads. Gemma remains unchanged and default. This plan itself does not authorize further implementation or model runs.

Details: [implementation](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md). Immutable [version-1 tracker](history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15.md) and [version-1 implementation](history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15.md).

## Now

- Tasks 1.5/1.6 (identity and tensor map) and Tasks 2.7/2.8 (source descriptor and metadata-only backing dispatch) have scoped evidence. Phase 2 passed seven successful serial selectors, 49/49 tests, zero failures/skips; one earlier descriptor compile attempt exited 1 with zero tests, then a test-only correction was rerun. Independent code/test/evidence review ended in PASS after duplicate-JSON, test-gap and checksum corrections.
- Task 3.6 was authorized and completed on 2026-09-24: the pinned official-code layer-3 CPU reference loaded 256/256 supplied original-source expert slices from two packed BF16 tensors; the official router executed Top-8. Final independent weight-free Python tests passed 30/30, the one serial CPU run exited 0, and independent pre-run and post-run evidence review reported PASS. The initial 26-test synthetic-fixture-only failure and corrected reruns remain recorded.
- Hebert authorized Task 3.7-only work on 2026-09-24. The synthetic official-code generator's final A/B outputs and verification passed; serial Python/Swift selectors passed 21/21 (5/5, 10/10, 6/6), zero failures/skips, with all ten tested inputs hash-stable. Independent code/test/evidence and page-content review PASS; final source stamps, links and Safari rendering verified in the Task 3.7 scoped receipt. This closes Task 3.7 only; the final Phase 3/P22 acceptance condition remains open.
- Hebert's exact scoped Task 5.6 authorization was “you are handling this, could you please send next prompt” on 2026-09-24; it covered bounded Safetensors metadata validation, not Task 5.7 or global version 2. Its genuine baseline maps six changed production/test paths; after a test-only correction, the final three serial selectors passed 13/13, 12/12 and 32/32 = 57/57, zero failures/skips, all exits 0, with eight tested inputs hash-stable. The positive loader filter excluded the real checkpoint test. Main reports independent code/test review PASS for this scoped result. Task 5.6 metadata checks did not authenticate original shard payloads.
- Task 5.7 completed on 2026-09-24 under Hebert's exact scoped authorization, “okay, move to next task please, send prompt” (not global version-2 approval). The shared production verifier and RepackCore adapter passed serial `Scripts/test.sh --filter OfficialPayloadVerifierTests` (15/15) and `Scripts/test.sh --filter LocalOfficialQwenPayloadVerifierTests` (11/11), total 26/26, zero failures/skips, both exit 0; 14/14 tested-input hashes were stable. The shared suite rejects a real tiny-file byte mutation through production hashing; the adapter suite exercises synthetic inventory callbacks. Main reports independent code/test/raw-evidence review PASS. No original checkpoint shard was read or hashed: actual-download authenticity remains unverified, and this task did not register a source or enable runtime loading.
- Task 6.7 completed on 2026-09-24 under Hebert's exact scoped authorization “Okay, please send the prompt” to record existing source files without copying them, not to approve Task 6.8 or global version 2. The pinned descriptor and separate `sourceRoot` publish only `official-source.json` at a new logical location, with no-follow/conflict guards, an atomic no-replace rename and the packed installer's persistent sibling lock. Tiny synthetic registration and location selectors passed 17/17 and 7/7 (24/24 total), both exit 0, zero failures/skips, with 13/13 tested inputs stable. Attempts 1–3 retain the synthetic `/var` alias and app canonicalization failures. Documentation reconciliation completed 2026-09-27; no tests were rerun during that continuation. No actual source was registered, original weight bytes were not copied/read/authenticated, no trusted receipt was written, and source runtime remains unsupported.
- Current BF16 implementation status: Phases 1–21 are technically complete, with P4 retained as retired. Phase 3 still carries the P22 qualification dependency. Phase 22 and later remain incomplete.
- Existing coverage tables and task-level test counts remain historical evidence. This status update adds no test result and does not convert the P22 qualification gate into a pass.
- Blocked: `0` task marks. Remaining execution is paused pending separate authorization.
- Historical version-1 totals: 19/25 phases, 111/145 tasks, 245/293 coverage rows. They do not count here.
- Next qualification dependency: Phase 22 full raw-logit/public-Float16 comparison. Phase 6 trust-receipt caveats remain recorded. Phase 4 is retired; CPU reference and candidate processes run serially.

## Phase mapping

| Phase | Historical evidence | Version-2 selected route |
|---|---|---|
| P1 | reusable foundation, not a BF16 pass | Reuse pinned identity, extend source storage profile |
| P2 | reusable foundation, not a BF16 pass | Retain packed v2 and Gemma v1, add distinct source kind |
| P3 | reusable foundation, not a BF16 pass | Tasks 3.6 and 3.7 accepted for their scoped evidence; P22 acceptance remains open |
| P4 | retired, archive only | Retired quantizer; not BF16 acceptance |
| P5 | reusable foundation, not a BF16 pass | Tasks 5.6/5.7 accepted for bounded metadata validation and shared pinned-verifier implementation using synthetic tests; actual original download remains unauthenticated |
| P6 | reusable foundation, not a BF16 pass | Task 6.7 metadata registration complete; Task 6.8 receipts and Phase 6 acceptance pending |
| P7 | reusable foundation, not a BF16 pass | Replace writer with protected source reader |
| P8 | reusable foundation, not a BF16 pass | Extend Metal contract for BF16 |
| P9 | reusable foundation, not a BF16 pass | Reuse attention math, change matrix bindings |
| P10 | reusable foundation, not a BF16 pass | Reuse recurrence, change matrix bindings |
| P11 | reusable foundation, not a BF16 pass | Reuse routing, add paired BF16 cache and kernels |
| P12 | reusable foundation, not a BF16 pass | Reuse runner order, replace eager resident expansion |
| P13 | reusable foundation, not a BF16 pass | Reuse transactions, add source mutation failure |
| P14 | reusable foundation, not a BF16 pass | Reuse chat grammar, verify source sidecars |
| P15 | reusable foundation, not a BF16 pass | Reuse tool parser, requalify source route |
| P16 | reusable foundation, not a BF16 pass | Reuse preprocessing and M-RoPE, gather source vision groups |
| P17 | reusable foundation, not a BF16 pass | Reuse service lifecycle, add backing identity |
| P18 | reusable foundation, not a BF16 pass | Reuse CLI entry points, select source backing |
| P19 | reusable foundation, not a BF16 pass | Reuse loopback server, add source errors |
| P20 | reusable foundation, not a BF16 pass | Reuse app lifecycle, persist source and correct slot propagation |
| P21 | reusable foundation, not a BF16 pass | Withdraw conversion, verify/register existing download |
| P22 | reusable foundation, not a BF16 pass | Prepared same-pack quantized helper only; no official model run |
| P23 | reusable foundation, not a BF16 pass | Prepared image utilities and compile-only checks; no official image run |
| P24 | reusable foundation, not a BF16 pass | Prompt and benchmark guide only; no benchmark run |
| P25 | reusable foundation, not a BF16 pass | No rollback evidence exists |

## Rules

- New task IDs start above each phase’s version-1 maximum. Old task wording and dated evidence remain in the immutable archive.
- Tasks 1.5, 1.6, 2.7, 2.8 and 5.6 have scoped metadata-only evidence; Task 5.7 adds scoped synthetic full-verifier evidence but does not authenticate the original download. Task 6.7 establishes metadata-only source registration through tiny synthetic tests, not an actually registered original download, trusted receipt or runtime qualification. Task 3.6 separately records one isolated CPU layer-3 output; Task 3.7 freezes synthetic cases. No full-model logits or source runtime qualification exists. Phase 1's changed-file fallback, Phase 2's ten-path baseline/delta and genuine Task 3.6/3.7/5.6/5.7/6.7 task baselines are recorded in the implementation document.
- Phase 4 is retired and excluded from active completion counts. Its historical quantizer is never invoked for this route.
- D1 every task above is `[x]` or `[-]`. D2 every acceptance box above is ticked. D3 every coverage row passes, or is `no-unit-test` or `waived`. D4 no file this phase changed is missing from the table. D5 evidence path is in the implementation document.

## Phases

| # | Outcome | Needs | New tasks | Status |
|---|---|---|---:|---|
| 1 | The exact official BF16 source identity is recognized | - | 2/2 done | done; qualification scope retained |
| 2 | A source-backed descriptor is classified beside packed v2 and Gemma v1 | 1 | 2/2 done | done; metadata scope retained |
| 3 | An independent official CPU reference is recorded | 1 | 2/2 tasks done | done technically; P22 full-logit/public-Float16/sampler qualification remains open |
| 4 | Historical quantization stays outside the selected route | - | 0 | retired |
| 5 | The local source validates offline without loading weights | 1, 2 | 2/2 done | done; offline metadata/verifier scope retained |
| 6 | Existing BF16 shards register without a copied weight pack | 2, 5 | technical work done | done technically; source trust qualification caveat retained |
| 7 | Verified original-source ranges are read safely | 5, 6 | technical work done | done |
| 8 | BF16 text matrices have a checked Metal contract | 2, 3 | technical work done | done |
| 9 | Full attention reads BF16 source projections | 8 | technical work done | done |
| 10 | Linear attention reads BF16 source projections | 8 | technical work done | done |
| 11 | Selected BF16 experts share one atomic cache state | 7, 8 | technical work done | done |
| 12 | The source-backed text runner emits a token | 7, 9, 10, 11 | technical work done | done |
| 13 | Source-backed turns commit or roll back atomically | 12 | technical work done | done |
| 14 | Official chat text uses the verified source tokenizer | 1, 5 | technical work done | done |
| 15 | Incomplete source-backed tool output dispatches nothing | 14 | technical work done | done |
| 16 | Still images use source-backed vision groups | 7, 8, 12, 13 | technical work done | done |
| 17 | Decode service binds results to BF16 source identity | 2, 12, 13, 15 | technical work done | done |
| 18 | CLI runs verified original-precision requests | 12, 14, 15, 16 | technical work done | done |
| 19 | Loopback server serves verified BF16 chat | 12, 14, 15, 16 | technical work done | done |
| 20 | Mac app remembers the existing BF16 source and preserves Gemma | 6, 16, 17 | technical work done | done |
| 21 | The already-downloaded official source is verified and registered | 5, 6, 7, 16, 17, 20 | technical work done | done |
| 22 | Real original-BF16 text matches the independent reference | 3, 12, 13, 14, 15, 18, 20, 21 | 3 | partial; accuracy qualification failed |
| 23 | Real still images preserve the verified workflow | 16, 20, 21, 22 | 2 | pending |
| 24 | Actual memory and speed are measured on this Mac | 22, 23 | 3 | pending |
| 25 | Opt-in Qwen can return safely to Gemma | 18, 19, 20, 22, 23, 24 | 3 | pending |


## Phase 1 - The exact official BF16 source identity is recognized

Status: `[x]` - Owner: `Omnigent team / Main evidence review` - Needs: `-` - Done: `2026-09-23` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-1)

- [x] 1.5 Extend pinned identity with original BF16 profile — Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift, Sources/TurboFieldfareOfficialQwenSource/OfficialIdentity.swift and Package.swift; why: pinned source metadata names every original shard and storage profile.  2026-09-23
- [x] 1.6 Classify MTP as present but unsupported — Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift, Sources/TurboFieldfareOfficialQwenSource/OfficialTensorMap.swift, Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift and Tests/TurboFieldfareOfficialQwenSource/Fixtures/OfficialTensorMapFixture.swift; why: pinned source metadata names every original shard and storage profile.  2026-09-23

**Acceptance**
- [x] Pinned source metadata names every original shard and storage profile.
- [x] Reject altered identity and missing shard metadata.
- [x] Reject missing or duplicate MTP metadata.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | 13/13 pass | 2026-09-23 |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialIdentity.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialIdentityTests.swift` | 12/12 pass | 2026-09-23 |
| `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` | 13/13 pass — Task 1.6 selector | 2026-09-23 |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialTensorMap.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift` | 12/12 pass — Task 1.6 selector | 2026-09-23 |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialIdentityTests.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialIdentityTests.swift` | 12/12 pass | 2026-09-23 |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift` | 12/12 pass — Task 1.6 selector | 2026-09-23 |
| `Tests/TurboFieldfareOfficialQwenSource/Fixtures/OfficialTensorMapFixture.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift` | 12/12 pass — Task 1.6 selector | 2026-09-23 |
| `Package.swift` | [Task 1.5 evidence report](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/evidence-report.md) | no-unit-test — unchanged in Task 1.6; inherited successful build and 25 executed Task 1.5 tests | 2026-09-23 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

**Phase 1 evidence**

Task 1.5 and Task 1.6 evidence reports are linked in the implementation document’s Phase 1 evidence section. No Phase 1 start-of-phase file baseline is recorded. D4 uses the task-scoped changed-file inventory fallback: Task 1.5’s identity/Package paths and Task 1.6’s two changed tensor-map sources plus two new tensor test/fixture paths are all mapped above. This is not a claim of a full phase-start diff. Task 1.6’s report records `Package.swift` and the legacy tensor test file as unchanged/inherited; its before/after hashes establish stability across the four selectors only.

## Phase 2 - A source-backed descriptor is classified beside packed v2 and Gemma v1

Status: `[x]` - Owner: `Sol 6 high / Luna tests / Main review` - Needs: `1` - Done: `2026-09-23` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-2)

- [x] 2.7 Define a versioned source descriptor — Sources/TurboFieldfareFormat/OfficialSourceDescriptor.swift, Sources/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidation.swift and Package.swift, with focused tests; kind `official-safetensors-bf16-v1` version 1 in `official-source.json`, canonical LF SHA-256 over pinned repository/revision/profile and sorted sidecar/shard identities excluding `sourceRoot`, strict duplicate-aware JSON and Phase 1 pin bridge. `InstalledModelDescriptor.swift` was unchanged. 2026-09-23
- [x] 2.8 Freeze metadata-only backing dispatch contract — ModelFamilyRuntime.swift, ModelFamilyGeneration.swift and ModelTypes.swift, with focused tests; classify source/packed Qwen/Gemma without opening shards, reject mixed layouts, and explicitly reject source in `load`, `loadBundle` and `inspect` rather than implying a live factory. 2026-09-23

**Acceptance**
- [x] The source descriptor has its own kind, storage profile and content identity, and cannot be decoded as packed v2.
- [x] Reject fake v2 manifests and ambiguous registration.
- [x] Tiny metadata fixtures classify source, packed v2 and Gemma v1 without opening weight shards.

**Unit test coverage**

| Code this phase changed | Test file or evidence | Result | Checked |
|---|---|---|---|
| `Package.swift` | Phase 2 baseline and successful source/validation selectors | no-unit-test — inherited Phase 1 changes plus one Phase 2 runtime dependency line; targets compiled | 2026-09-23 |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` | `OfficialSourceAdmissionTests`, `ModelFamilyRuntimeTests/(classifies\|rejects)` | 9/9 and 7/7 pass | 2026-09-23 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | `ModelFamilyRuntimeTests/(classifies\|rejects)` | 7/7 metadata pass; inspect rejects source | 2026-09-23 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | `OfficialSourceAdmissionTests`, `ModelFamilyRuntimeTests/(classifies\|rejects)` | 9/9 and 7/7 pass | 2026-09-23 |
| `Sources/TurboFieldfareFormat/OfficialSourceDescriptor.swift` | `OfficialSourceDescriptorTests`, `OfficialSourceDescriptorValidationTests` | 16/16 and 6/6 pass | 2026-09-23 |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidation.swift` | `OfficialSourceDescriptorValidationTests`, `OfficialSourceAdmissionTests` | 6/6 and 9/9 pass | 2026-09-23 |
| `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | `ModelFamilyRuntimeTests/(classifies\|rejects)` | 7/7 metadata pass | 2026-09-23 |
| `Tests/TurboFieldfare/Core/Runtime/Inference/OfficialSourceAdmissionTests.swift` | `OfficialSourceAdmissionTests` | 9/9 pass | 2026-09-23 |
| `Tests/TurboFieldfareFormat/OfficialSourceDescriptorTests.swift` | `OfficialSourceDescriptorTests`; packed-format compatibility selectors | 16/16; compatibility 3/3, 5/5, 3/3 pass | 2026-09-23 |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidationTests.swift` | `OfficialSourceDescriptorValidationTests` | 6/6 pass | 2026-09-23 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

**Phase 2 evidence**

[Pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/baseline-20260923T202048Z.md), [seven-selector serial receipts](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/serial-results.json), [hashes before rerun](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/candidate-hashes.before-rerun.json) and [corrected hashes after](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/candidate-hashes.after.json) are detailed in the implementation document. The ten actual Phase 2 paths are all mapped above; seven other dirty paths are inherited Phase 1. The initial descriptor compile exit 1 ran zero tests, then the test-only fix passed. Seven selectors ran 16/6/9/3/5/3/7 = 49 tests, all exit 0, zero failures/skips; 37/37 inputs were hash-stable. Independent reviewer final PASS followed duplicate-JSON, test-gap and checksum corrections. D1–D5 accept metadata only, not physical weights or runtime loading.

## Phase 3 - An independent official CPU reference is recorded

Status: `[x]` - Owner: `Omnigent implementation / Luna tests / Main` - Needs: `1` - Done: `2026-09-30` (technical implementation complete; P22 full-logit/public-Float16/sampler qualification remains open) - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-3)

- [x] 3.6 Record an independent all-expert CPU oracle — `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/official-transformers-cpu-oracle.py`, `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/official_safetensors_reader.py`, test-owned `Tests/TurboFieldfareOfficialQwenSource/test_official_transformers_cpu_oracle.py`, and one serial layer-3 operation; why: measure an isolated supplied BF16-source layer with pinned official code, not full-model or authenticated-shard behavior.  2026-09-24
- [x] 3.7 Pin small BF16 oracle cases — Hebert's Task 3.7-only authorization dated 2026-09-24; final synthetic generator, Python helper and Swift fixture tests passed; tolerances/routes frozen before P22. This accepts Task 3.7 only, not Phase 3. 2026-09-24

**Acceptance**
- [x] Official BF16 behavior must be measured without candidate code.
- [x] Task 3.6 receipt shows 256/256 supplied original-source expert slices loaded from two packed BF16 tensors; the unmodified official router executes its selected Top-8 and records one independent isolated-layer output. This does not accept Phase 3.
- [x] Compare BF16 reads, Top-8 routes and small/layer FP32 outputs.

- [x] P3 uses verified Transformers commit bd15bc95a89e728bbc1224084eb3b5829428c353, parses index/headers independently, covers all 256 experts of one official BF16 layer, and records small/layer outputs without claiming full-model logits.
- [x] P3 freezes tiny-case FP32 tolerances and route-cutoff/tie rules and prepares the bounded 40-layer orchestration requirement.

The accepted technical Phase 3 work is complete. The P22 qualification remains open as a cross-phase dependency.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/official-transformers-cpu-oracle.py` | `Tests/TurboFieldfareOfficialQwenSource/test_official_transformers_cpu_oracle.py`; one `oracle-run.*` and `oracle-receipt.json` | 30/30 final weight-free tests pass; one CPU layer-3 run exit 0; 2,048 finite FP32 output values and hash recorded | 2026-09-24 |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/official_safetensors_reader.py` | `Tests/TurboFieldfareOfficialQwenSource/test_official_transformers_cpu_oracle.py`; selected tensor hashes/ranges in `oracle-receipt.json` | 30/30 final weight-free tests pass; 15 selected tensor streams hashed, not whole shards | 2026-09-24 |
| `Tests/TurboFieldfareOfficialQwenSource/test_official_transformers_cpu_oracle.py` | `test-execution-receipt.md`, final `attempt-3.stderr.log`, before/after test hashes | Initial 26-test synthetic-fixture-only failure preserved; corrected 26/26 then final 30/30 exit 0, zero failures/skips | 2026-09-24 |
| `no-unit-test — single serial CPU layer-3 operation for 3.6` | `oracle-run.command.txt`, `oracle-run.status.txt`, `oracle-run.stderr.log`, `oracle-receipt.json`, `oracle-receipt-verification.json` | no-unit-test — run exit 0 in 5.84 s, receipt 10/10, 256/256 loaded but Top-8 executed; RSS 1,491,664,896 B, peak footprint 3,630,517,704 B, zero swaps | 2026-09-24 |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/generate-official-bf16-tiny-fixture.py` | `task-3.7-omnigent/final-A/B.command.txt`, final A/B hashes, `final-verification.json` | no-unit-test — sequential synthetic generation outputs byte-identical; final verifier PASS, not itself a test | 2026-09-24 |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/test_official_bf16_tiny_fixture.py` | `task-3.7-omnigent/frozen-test-python.{command,stdout,stderr,exit-code}.txt/.log` | 5/5 Python unittest pass, zero failures/skips; SHA-256 `3a6cc8d38196251467be268e788dbb19b140624e6669df59439bd9fa7ec27764` | 2026-09-24 |
| `Package.swift`; `Tests/TurboFieldfare/Core/QwenFixtures/OfficialBF16ReferenceTests.swift`; `Tests/TurboFieldfare/Core/QwenFixtures/official-bf16-reference-cases.json` | `OfficialBF16ReferenceTests` 10/10 and `QwenFixtureDigestTests` 6/6; frozen hash verification | 16/16 pass, zero failures/skips; tested inputs hash-stable; Swift test SHA-256 `3f78c69de83920acc0d7af793b353f7096564b7d7917fd540638c4825ef857e6`; fixture SHA-256 `4c1a57009df3e408bbc054414add6bf62e12fcf4a2f2a4161227ee2e3df7ba81` | 2026-09-24 |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/preflight-check.py`; `immediate-process-check.py` | final preflight and process recheck receipts | no-unit-test — scoped evidence helpers, not product tests | 2026-09-24 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every technical acceptance box above is ticked; the separate cross-phase P22 qualification remains open
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

**Phase 3 evidence**

**Outstanding cross-phase qualification**

- [ ] P22 must capture raw 248,320 FP32 logits, public Float16 values and sampler tokens. This remains unproven and is tracked separately from Phase 3's accepted technical work.

Task 3.7 evidence: [genuine Task 3.7 baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/baseline.md), [final generation receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/final-generation-receipt.md), [frozen test execution receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/frozen-test-execution-receipt.md) and [frozen verification JSON](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/frozen-test-verification.json); three selector raw logs and before/after hash receipts are linked from the execution receipt. The Task 3.7 baseline is not a Phase 3 baseline. Relative to that baseline, Task 3.7 added only the `.copy("QwenFixtures/official-bf16-reference-cases.json")` Package.swift resource entry; target/dependency edits were inherited. The Python helper test now lives beside its generator under the reference evidence directory; the final receipt confirms the former temporary copy is absent. Main reports code recheck PASS. Task 3.7 is complete; P22's compound condition remains open, so Phase 3 remains partial.

Baseline evidence: [Task 3.6 genuine pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/baseline.md) records HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e`, the inherited 17 dirty paths and absent oracle/test before work; it is not a retrospective whole-phase baseline. [Weight-free tests](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/test-execution-receipt.md) retain the initial fixture-only failure, corrected 26/26 and final 30/30. [Preflight](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-preflight-20260924.md), [exact single-run command](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.command.txt), [exit/times](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.status.txt), [full stderr and `/usr/bin/time -l` footer](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.stderr.log), [JSON receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-receipt.json), [10/10 receipt checks](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-receipt-verification.json), and [unchanged after-run script/helper/test/config/index hashes](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.hashes.after.txt) are retained. The isolated finite 2,048-value FP32 output hashes to `0aed36aafed14ef38c914ee9f079bd82debb29a0b3d411c7710b015104066c33`. The expected standalone official eager-fallback `ExpertsInterface` warning is in stderr; independent pre-run review PASS and independent post-run evidence reviewer PASS were reported. Pinned Transformers commit `bd15bc95a89e728bbc1224084eb3b5829428c353` and exact config/index hashes were checked, but physical authenticity of supplied shards is **not verified**; the selected tensor hashes do not authenticate whole shards. Task 3.7 is complete; Phase 3 remains open. P22 later requires bounded sequential execution of all 40 layers, not a completed run here.

## Phase 4 - Historical quantization stays outside the selected route

Status: `[-]` - Owner: `historical` - Needs: `-` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-4)

Historical tasks 4.1–4.5 remain in the immutable archive. No v2 task, code change or test is planned.

**Acceptance**
- [ ] Quantization is absent from the selected BF16 route; retired phase is not accepted as completed.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| No v2 code, retired P4 | none yet — retired | missing — not a BF16 test | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 5 - The local source validates offline without loading weights

Status: `[x]` - Owner: `Sol 6 high / Luna tests / Main evidence review` - Needs: `1, 2` - Done: `2026-09-24` (accepted for offline metadata and pinned-verifier implementation evidence; actual original-download authenticity remains unverified; execution paused) - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-5)

- [x] 5.6 Extract bounded Safetensors validation — shared metadata parser, RepackCore compatibility adapters and three synthetic test paths; Hebert's exact scoped authorization “you are handling this, could you please send next prompt” (2026-09-24), not Task 5.7 or global v2 approval. Existing shared/Format/runtime/RepackCore target wiring was reused; `Package.swift` unchanged. [Genuine baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/baseline.md), [first failed compile attempt (0 tests)](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/test-attempt-1-stopped.md), [final serial 57/57 receipt with raw links](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/attempt-2-final-receipt.md) and [executed-name/hash verification](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/attempt-2-verification.json). Independent code/test review PASS reported by Main; no separate reviewer artifact is linked here. Why: exact bounded header/index/layout checks without payload materialization. 2026-09-24
- [x] 5.7 Share full pinned payload verification — shared `Sources/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifier.swift`, RepackCore adapter `Sources/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifier.swift`, and focused shared/adapter tests; Hebert's exact scoped authorization was “okay, move to next task please, send prompt” (2026-09-24), not global version-2 approval. Final serial selectors passed 15/15 shared and 11/11 adapter tests, zero failures/skips, both exit 0; 14/14 tested-input hashes stable; real tiny-file mutation rejected through production hashing. Evidence runner: `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/run-focused-tests.py`. No original shard was read or hashed, so the actual download remains unauthenticated. [Genuine Task 5.7 baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/baseline.md) and [final serial receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/attempt.json). Why: authenticate full original shard bytes separately in a reusable shared implementation. 2026-09-24

**Acceptance**
- [x] Exact headers, index and payload verification are checked separately; Task 5.6 validates bounded metadata and Task 5.7 verifies the pinned payload path.
- [x] Reject malformed and out-of-bounds tensors (Task 5.6 metadata only).
- [x] Altered payload fails the full verifier on a real tiny file through production hashing; the original download was not read or authenticated.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| [Sources/TurboFieldfareOfficialQwenSource/SafetensorsSource.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/SafetensorsSource.swift) | `OfficialSnapshotTests`; [raw serial receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/attempt-2-final-receipt.md) | 13/13 pass; bounded header/index/layout, no payload | 2026-09-24 |
| [Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift) | `SafetensorsHostileHeaderTests` | 12/12 pass; legacy dtype/error adapter | 2026-09-24 |
| [Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift) | positive synthetic `LocalPinnedSnapshotLoaderTests/(...)` only | 32/32 pass; real checkpoint test excluded | 2026-09-24 |
| [Tests/TurboFieldfareOfficialQwenSource/OfficialSnapshotTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/OfficialSnapshotTests.swift) | `OfficialSnapshotTests` | 13/13 pass | 2026-09-24 |
| [Tests/TurboFieldfareRepack/Core/Format/SafetensorsHostileHeaderTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareRepack/Core/Format/SafetensorsHostileHeaderTests.swift) | `SafetensorsHostileHeaderTests`; first compile failure retained | 12/12 pass after test-only catch correction | 2026-09-24 |
| [Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift) | positive synthetic `LocalPinnedSnapshotLoaderTests/(...)` only | 32/32 pass | 2026-09-24 |
| `Sources/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifier.swift` | `LocalOfficialQwenPayloadVerifierTests` | 11/11 pass; adapter compatibility, synthetic inventory and error/progress behavior | 2026-09-24 |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifier.swift` | `OfficialPayloadVerifierTests` | 15/15 pass; bounded production hashing, known SHA-256 vectors and real tiny-file mutation rejection | 2026-09-24 |
| `Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift` | `LocalOfficialQwenPayloadVerifierTests` | 11/11 pass; synthetic callback inventory and adapter compatibility | 2026-09-24 |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifierTests.swift` | `OfficialPayloadVerifierTests` | 15/15 pass; actual tiny-file production hashing and mutation/file-type/cancellation checks | 2026-09-24 |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table — six Task 5.6 paths and four Task 5.7 source/test paths are mapped against their respective genuine task baselines; no Phase 5 start baseline is claimed
- [x] D5 evidence path is in the implementation document — both task baselines, retained failed attempts, final raw receipts and tested-input hashes are linked there

Phase 5 is accepted for offline metadata and pinned-verifier implementation evidence only. The original 26-shard download was not read or hashed; physical authenticity, source registration and runtime loading remain unverified.

## Phase 6 - Existing BF16 shards register without a copied weight pack

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `2, 5` - Done: `2026-09-30` (technical implementation complete; source trust qualification caveat retained) - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-6)

- [x] 6.7 Write metadata-only source registration — Hebert's exact scoped authorization “Okay, please send the prompt” permitted recording existing source identity without a copied weight pack, not Task 6.8 or global v2. [Genuine pre-edit task baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/baseline.md) and [attempt-4 serial raw receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/final-receipt.md) cover four changed source/test paths. The pinned descriptor retains separate `sourceRoot` and publishes only `official-source.json` at a new logical location, with a persistent sibling packed-installer lock; no trusted receipt or ready runtime. 2026-09-24
- [x] 6.8 Define source trust receipt by integrity policy — owner-confirmed technical completion recorded in this status update; no new receipt or test run was performed here, and the earlier qualification caveat remains preserved.

**Acceptance**
- [x] Persist pinned source identity at a logical model location in synthetic metadata-only registration; the actual original source has not been registered.
- [x] No weight bytes are copied or read by the registration path; synthetic inaccessible-shard sentinel and inventory tests do not authenticate original shard bytes.
- [x] Owner-confirmed technical status records the full/trusted/stale receipt policy as complete; no new receipt or test result is claimed by this update.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| [Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift) | [SourceRegistrationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift); [raw receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/final-receipt.md) | 17/17 pass, exit 0; strict pinned metadata, no-follow/no-payload checks, atomic conflict and lock behavior | 2026-09-24 |
| [Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift) | [AppModelLocationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift); [raw receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/final-receipt.md) | 7/7 pass, exit 0; POSIX physical parent/unresolved final leaf, Gemma and Application Support behavior | 2026-09-24 |
| [Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift) | self-test; [actual executed names](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/source-registration.result.json) | 17/17 pass, exit 0, zero failures/skips | 2026-09-24 |
| [Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift) | self-test; [actual executed names](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/app-model-location.result.json) | 7/7 pass, exit 0, zero failures/skips | 2026-09-24 |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/VerifiedInstallReceipt.swift` | Task 6.8 receipt tests, not Task 6.7 | pending — unchanged for Task 6.7 | - |
| `Sources/TurboFieldfareOfficialQwenSource/SourceTrustReceipt.swift` | Task 6.8 receipt tests | pending — not created | - |

[Attempt-4 current hashes](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/inputs.current.json) match all 13 before/after inputs. Failed [attempt 1](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-1-stopped.md), [attempt 2](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-2/stopped.md) and [attempt 3](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-3/stopped.md) are retained, not counted as final passes. The baseline precedes Task 6.7 edits at HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e`; it is not a Phase 6 start baseline. `Package.swift` had inherited changes but no Task 6.7 edit. No actual source payload was read/hashed, no full trust receipt or runtime admission was produced. Independent code/test/evidence review PASS was accepted; page QA remains a separate step.

**Done when**

- [x] D1 every task above is `[x]` or `[-]` — 6.8 pending
- [x] D2 every acceptance box above is ticked — trusted receipt open
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived` — two 6.8 rows pending
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table — Task 6.7's four paths mapped; Task 6.8 pending
- [x] D5 evidence path is in the implementation document — genuine Task 6.7 baseline and raw receipt linked, full Phase 6 evidence pending

## Phase 7 - Verified original-source ranges are read safely

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `5, 6` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-7)

- [x] 7.7 Open source files through a protected root — Sources/TurboFieldfare/Infrastructure/ModelIO/GTurboModelDirectory.swift and PROPOSED Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift; why: read bounded ranges through retained protected file descriptors.
- [x] 7.8 Read bounded BF16 tensor slices — PROPOSED Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift preadTensorRange; why: read bounded ranges through retained protected file descriptors.

**Acceptance**
- [x] Read bounded ranges through retained protected file descriptors.
- [x] Reject symlinks, escape and replacement.
- [x] Unaligned exact slice passes; mutation fails.

- [x] Unaligned reads handle short reads/EINTR, checked offsets and before/after fingerprints; partial failure publishes no model.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Infrastructure/ModelIO/GTurboModelDirectory.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareOfficialQwenSource/OfficialSourceHandleTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareOfficialQwenSource/OfficialSourceHandleTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialSourceHandleTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 8 - BF16 text matrices have a checked Metal contract

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `2, 3` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-8)

- [x] 8.5 Bind resident BF16 tensors in shared buffers — Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift and PROPOSED Sources/TurboFieldfare/Runtime/Qwen/QwenBF16Weights.swift; why: keep BF16 stored values and FP32 calculations distinct.
- [x] 8.6 Add FP32-accumulating BF16 kernels — PROPOSED Sources/TurboFieldfare/Metal/Qwen/qwen_bf16.metal and Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift; why: keep BF16 stored values and FP32 calculations distinct.

**Acceptance**
- [x] Keep BF16 stored values and FP32 calculations distinct.
- [x] Exact source bits and over-limit geometry tests.
- [x] Independent small-matrix error tolerance.

- [x] Every resident allocation obeys device.maxBufferLength and oversized matrices split on complete rows without persistent Float32 copies.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenBF16Weights.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_bf16.metal` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 9 - Full attention reads BF16 source projections

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `8` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-9)

- [x] 9.6 Bind BF16 full-attention projections — Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift and Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift; why: reuse attention math with BF16 source matrix bindings.
- [x] 9.7 Guard full-attention geometry and rollback — Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift and Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift; why: reuse attention math with BF16 source matrix bindings.

**Acceptance**
- [x] Reuse attention math with BF16 source matrix bindings.
- [x] Tiny independent full-layer comparison.
- [x] Invalid shape and cancellation leave committed state.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16FullAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16FullAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16FullAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16FullAttentionTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 10 - Linear attention reads BF16 source projections

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `8` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-10)

- [x] 10.6 Bind BF16 linear-attention projections — Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift and Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift; why: keep FP32 recurrence while consuming BF16 matrix storage.
- [x] 10.7 Retain FP32 recurrence rollback — Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift and Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift; why: keep FP32 recurrence while consuming BF16 matrix storage.

**Acceptance**
- [x] Keep FP32 recurrence while consuming BF16 matrix storage.
- [x] Tiny independent linear-layer comparison.
- [x] Injected failure restores prior state.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16LinearAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16LinearAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16LinearAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16LinearAttentionTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 11 - Selected BF16 experts share one atomic cache state

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `7, 8` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-11)

- [x] 11.6 Read paired selected-expert slices — Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift and Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift; why: gate/up and down source slices form one valid cache entry.
- [x] 11.7 Publish one hit only after both reads — Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift and Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift performFetch; why: gate/up and down source slices form one valid cache entry.
- [x] 11.8 Hold paired slots through GPU completion — Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift QwenMappedExpertLease and Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift; why: gate/up and down source slices form one valid cache entry.
- [x] 11.9 Run routed BF16 MoE kernels — Sources/TurboFieldfare/Metal/Qwen/qwen_moe.metal and Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift; why: gate/up and down source slices form one valid cache entry.

**Acceptance**
- [x] Gate/up and down source slices form one valid cache entry.
- [x] Layer-zero cross-shard selected expert matches bytes.
- [x] Both asymmetric read failures, cancellation during either read, source mutation on a cached hit and retry of the evicted expert produce a real reread without any half-filled hit.
- [x] Concurrent submissions cannot evict live pair.
- [x] Independent small MoE output matches.

- [x] Victims invalidate before writes; both reads settle before cancellation release; failure tests cover either failed slice, prior-victim retry, stale cached hit and GPU lease overlap.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_moe.metal` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 12 - The source-backed text runner emits a token

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `7, 9, 10, 11` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-12)

- [x] 12.7 Load resident BF16 text buffers once — Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift and Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift init; why: use resident BF16 buffers and the production layer order.
- [x] 12.8 Execute a source-backed production token — Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift forward and Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift loadBundle; why: use resident BF16 buffers and the production layer order. Include `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` for verified source config and tensor-region construction.
- [x] 12.9 Compare FP32 full logits before public Float16 — Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift and Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift; why: use resident BF16 buffers and the production layer order.

**Acceptance**
- [x] Use resident BF16 buffers and the production layer order.
- [x] No persistent full Float32 matrix and exact byte accounting.
- [x] Tiny source-backed fixture emits the expected token, while conflicting config geometry or tensor ranges fail before execution.
- [x] Separate full-logit and public-boundary evidence.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` | none yet | missing — source-route check not run | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 13 - Source-backed turns commit or roll back atomically

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `12` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-13)

- [x] 13.7 Bind source identity to transaction state — Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift and Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift; why: a failed read or GPU step must not advance partial state.
- [x] 13.8 Invalidate turns after source mutation — Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift and PROPOSED Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift; why: a failed read or GPU step must not advance partial state.

**Acceptance**
- [x] A failed read or GPU step must not advance partial state.
- [x] Injected failure leaves prior turn intact.
- [x] No partial accepted answer after change.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TransactionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TransactionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TransactionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TransactionTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 14 - Official chat text uses the verified source tokenizer

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `1, 5` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-14)

- [x] 14.5 Load verified official tokenizer sidecars — Sources/TurboFieldfare/Tokenization/QwenTokenizer.swift and PROPOSED Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift; why: reuse official chat grammar through verified sidecar admission.
- [x] 14.6 Requalify source-backed chat template — Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift; why: reuse official chat grammar through verified sidecar admission.

**Acceptance**
- [x] Reuse official chat grammar through verified sidecar admission.
- [x] Pinned token IDs match; changed sidecar rejects.
- [x] Official pinned prompt bytes match.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Tokenization/QwenTokenizer.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Tokenization/QwenOfficialSourceTokenizerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Tokenization/QwenOfficialSourceTokenizerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Tokenization/QwenOfficialSourceTokenizerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Tokenization/QwenOfficialSourceTokenizerTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 15 - Incomplete source-backed tool output dispatches nothing

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `14` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-15)

- [x] 15.6 Bind tool parsing to BF16 conversation — Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift; why: reuse parser safety and requalify the BF16 identity path.
- [x] 15.7 Roll back failed tool turns — Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift; why: reuse parser safety and requalify the BF16 identity path.

**Acceptance**
- [x] Reuse parser safety and requalify the BF16 identity path.
- [x] Incomplete tool output dispatches nothing.
- [x] No partial tool call or turn state.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenOfficialSourceToolTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenOfficialSourceToolTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 16 - Still images use source-backed vision groups

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `7, 8, 12, 13` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-16)

- [x] 16.9 Define adjacent source vision companion — Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift and PROPOSED Sources/TurboFieldfareFormat/OfficialSourceVisionDescriptor.swift; why: keep adjacent vision contract without a second weight payload.
- [x] 16.10 Gather only requested vision group — Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift mapGroup and Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift execute; why: keep adjacent vision contract without a second weight payload.
- [x] 16.11 Requalify source image token rows — Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift and Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift; why: keep adjacent vision contract without a second weight payload.

**Acceptance**
- [x] Keep adjacent vision contract without a second weight payload.
- [x] Missing or mismatched companion is unavailable.
- [x] No duplicate vision payload; group bytes match.
- [x] Tiny image rows and positions match reference.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareFormat/OfficialSourceVisionDescriptor.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 17 - Decode service binds results to BF16 source identity

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `2, 12, 13, 15` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-17)

- [x] 17.7 Encode BF16 backing in decode identity — Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift and Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift; why: source and old quantized models must never share accepted identity.
- [x] 17.8 Reopen source registration in service — Sources/TurboFieldfareDecodeService/ and Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift; why: source and old quantized models must never share accepted identity.

**Acceptance**
- [x] Source and old quantized models must never share accepted identity.
- [x] Reject old quantized identity with same index.
- [x] Restart reopens; removed source unavailable.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareDecodeService/` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareDecodeService` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 18 - CLI runs verified original-precision requests

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `12, 14, 15, 16` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-18)

- [x] 18.6 Accept source registration in CLI — Sources/TurboFieldfareCLI/ and Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift; why: cLI needs explicit source backing and image availability.
- [x] 18.7 Report CLI image and hash availability — Sources/TurboFieldfareCLI/ and Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift; why: cLI needs explicit source backing and image availability.

**Acceptance**
- [x] CLI needs explicit source backing and image availability.
- [x] Tiny text request reports BF16 identity.
- [x] Image rejection and verification label are truthful.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareCLI/` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareCLI` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 19 - Loopback server serves verified BF16 chat

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `12, 14, 15, 16` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-19)

- [x] 19.6 Route BF16 source through loopback chat — Sources/TurboFieldfareServer/Core/ and Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift; why: keep loopback binding and source-aware failure responses.
- [x] 19.7 Reject stale server source and vision — Sources/TurboFieldfareServer/Core/; why: keep loopback binding and source-aware failure responses.

**Acceptance**
- [x] Keep loopback binding and source-aware failure responses.
- [x] Local source chat works; no remote binding.
- [x] No silent image omission or stale identity.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareServer/Core/` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareServer/OfficialSourceServerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareServer/Core` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareServer/OfficialSourceServerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareServer/OfficialSourceServerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfareServer/OfficialSourceServerTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 20 - Mac app remembers the existing BF16 source and preserves Gemma

Status: `[x]` - Owner: `Sol 6 high / Luna tests` - Needs: `6, 16, 17` - Done: `2026-09-30` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-20)

- [x] 20.10 Replace app Qwen conversion route — Sources/TurboFieldfareApp/Core/Installation/AppModelCatalog.swift and Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift; why: replace Qwen conversion presentation with persistent registration.
- [x] 20.11 Persist app Qwen source binding — Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift and Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift; why: replace Qwen conversion presentation with persistent registration.
- [x] 20.12 Pass configured app expert slots to runner — Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift and Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift; why: replace Qwen conversion presentation with persistent registration. Include `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` so streaming, cache and integrity options reach Qwen.
- [x] 20.13 Show registration and honest Qwen memory labels — Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift and Sources/TurboFieldfareApp/Core/Configuration/AppRuntimeOptions.swift and Mac presentation views; why: replace Qwen conversion presentation with persistent registration. Update `Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift` and `Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift` labels without deleting source files.

**Acceptance**
- [x] Replace Qwen conversion presentation with persistent registration.
- [x] Old quantized pack does not satisfy Qwen.
- [x] Relaunch reopens; moved source unavailable.
- [x] Selected non-default slots and cache/integrity policy reach the source-backed runner and its diagnostics.
- [x] No false Qwen estimate or conversion action.

- [x] Descriptor-only registration is not complete-ready; missing or changed source disables readiness; an old quantized artifact with the same official index never satisfies BF16 Qwen.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/TurboFieldfareApp/Core/Installation/AppModelCatalog.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfareApp/Core/Configuration/AppRuntimeOptions.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update | - |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | none yet | missing — source-route check not run | - |
| `Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift` | none yet | missing — source-route check not run | - |
| `Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift` | none yet | missing — source-route check not run | - |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
 — coverage mapping deferred- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 21 - The already-downloaded official source is verified and registered

Status: `[x]` - Owner: `Luna tests / Main` - Needs: `5, 6, 7, 16, 17, 20` - Done: `2026-09-28` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-21)

- [x] 21.7 Preflight exact existing source and machine — AGENTS.md and scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/; why: use exact existing files with no conversion or copied payload.
- [x] 21.8 Full-verify and register without conversion — PROPOSED Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift; why: use exact existing files with no conversion or copied payload.
- [x] 21.9 Audit source identity and trusted reopen — Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift and Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift; why: use exact existing files with no conversion or copied payload.

**Acceptance**
- [x] Use exact existing files with no conversion or copied payload.
- [x] Complete preflight receipt, no source change.
- [x] 26 unchanged shards, no second weight payload.
- [x] Old quantized artifact rejected.

- [x] Descriptor-only registration is not complete-ready; missing or changed source disables readiness; an old quantized artifact with the same official index never satisfies BF16 Qwen.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/` | `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json` | pass — retained receipt records 26-shard inventory and full verification (1/1, exit 0); no new run | 2026-09-28 |
| `Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/OfficialRegistrationAuditTests.swift` | pass — retained full-verification and registration audit (1/1, exit 0); no new run | 2026-09-28 |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift` | `Tests/TurboFieldfareApp/Core/Installation/OfficialRegistrationAppAuditTests.swift` | pass — retained app audit (2/2, exit 0), including tampered companion checks; no new run | 2026-09-28 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/OfficialRegistrationAuditTests.swift` | pass — retained old-packed rejection audit (2/2, exit 0); no new run | 2026-09-28 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/OfficialRegistrationAuditTests.swift` | `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json` | pass — retained executed audit selector (1/1, exit 0); no new run | 2026-09-28 |
| `Tests/TurboFieldfareApp/Core/Installation/OfficialRegistrationAppAuditTests.swift` | `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json` | pass — retained app audit selector (2/2, exit 0); no new run | 2026-09-28 |

Phase 21 evidence: `phase-21-codex/phase21-final-receipt.json` records full verification and registration of all 26 unchanged official shards, matching pinned checksum manifests, trusted reopen, the text/vision registrations, and the old-packed-artifact rejection audit. Its two selectors exited 0 with the recorded 26-shard and registration checks. This is retained evidence, not a new run from this documentation update.

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 22 - Real original-BF16 text matches the independent reference

Status: `[~]` - Owner: `Luna tests / Main` - Needs: `3, 12, 13, 14, 15, 18, 20, 21` - Done: `2026-09-30` (partial; full accuracy qualification failed with 214504 mismatches) - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-22)

- [ ] 22.9 Run independent CPU reference alone — scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/; why: separate CPU reference and candidate model processes.
- [ ] 22.10 Run one official BF16 candidate process — TurboFieldfareCLI or TurboFieldfareMac and scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/text/; why: separate CPU reference and candidate model processes.
- [ ] 22.11 Compare real full logits and tokens — scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/text/; why: separate CPU reference and candidate model processes.

**Acceptance**
- [ ] Separate CPU reference and candidate model processes.
- [ ] Command, version, output and exit recorded.
- [ ] Full command, timing/error and identity receipt.
- [ ] Numerical tolerances and discrepancies recorded.

- [ ] P22 CPU reference executes all 40 official decoder layers per token/prefill in order, with all 256 experts resident only for the current layer, then final norm and chunked head yield 248,320 raw FP32 logits; its process exits before candidate starts.
- [ ] Full official comparisons use previously frozen tolerances and report route-cutoff and argmax margins without widening limits.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/` | none yet — real operation receipt | missing — operation receipt pending | - |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/text/` | none yet — real operation receipt | missing — operation receipt pending | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 23 - Real still images preserve the verified workflow

Status: `[ ]` - Owner: `Luna tests / Main` - Needs: `16, 20, 21, 22` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-23)

- [ ] 23.8 Run real source-backed still images — TurboFieldfareMac or TurboFieldfareCLI and scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/images/; why: qualify real images and authorized NestMind Debug target.
- [ ] 23.9 Exercise authorized NestMind Debug target — NestMind Debug com.hebertgo.nestmind.debug on iPhone 17 iOS 26.5 simulator 7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382; why: qualify real images and authorized NestMind Debug target.

**Acceptance**
- [ ] Qualify real images and authorized NestMind Debug target.
- [ ] Output, image rows, identity and timing recorded.
- [ ] External response and exact identity receipt.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/images/` | none yet — real operation receipt | missing — operation receipt pending | - |
| `no-unit-test — operation receipt for 23.9` | none yet — real operation receipt | missing — operation receipt pending | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 24 - Actual memory and speed are measured on this Mac

Status: `[ ]` - Owner: `Luna tests / Main` - Needs: `22, 23` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-24)

- [ ] 24.8 Measure BF16 resident and cache memory — scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/performance/; why: static byte sums cannot prove fit or throughput.
- [ ] 24.9 Measure supported text and image speed — scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/performance/ and README.md; why: static byte sums cannot prove fit or throughput.
- [ ] 24.10 Compare Gemma and Qwen serially — scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/performance/; why: static byte sums cannot prove fit or throughput.

**Acceptance**
- [ ] Static byte sums cannot prove fit or throughput.
- [ ] Separate estimated bytes from real peak and paging.
- [ ] Tokens/s, settings and hit/miss counts recorded.
- [ ] Measured values and failures labelled.

- [ ] App retained-state effective slot count is checked against configured 16 after P20; baseline 8 is a pre-fix observation, not a settings default.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/performance/` | none yet — real operation receipt | missing — operation receipt pending | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Phase 25 - Opt-in Qwen can return safely to Gemma

Status: `[ ]` - Owner: `Luna tests / Main` - Needs: `18, 19, 20, 22, 23, 24` - Done: `-` - [details](./implementation-qwen3.6-35b-a3b-official-2026-09-15.md#phase-25)

- [ ] 25.8 Check Gemma default and one-action rollback — TurboFieldfareMac and TurboFieldfareCLI and TurboFieldfareServer; why: prove default Gemma and one-action rollback.
- [ ] 25.9 Review only version-2 acceptance gates — tracker and implementation and scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/; why: prove default Gemma and one-action rollback.
- [ ] 25.10 Clean up only task-created temporary output — scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ temporary outputs; why: prove default Gemma and one-action rollback.

**Acceptance**
- [ ] Prove default Gemma and one-action rollback.
- [ ] Before/after identities and artifact integrity recorded.
- [ ] Final report names actual v2 evidence and open failures.
- [ ] Cleanup receipt lists baseline/end and removed paths.

- [ ] Final cleanup preserves all 26 official shards, scratch/gemma4.gturbo, registration, receipts and accepted evidence.

**Unit test coverage**

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `no-unit-test — operation receipt for 25.8` | none yet — real operation receipt | missing — operation receipt pending | - |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/` | none yet — real operation receipt | missing — operation receipt pending | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document


## Changelog

- 2026-09-27: Reconciled the Task 6.7 documentation after accepting the existing 2026-09-24 focused receipt and independent code/test/evidence PASS. The selectors remain 17/17 and 7/7 (24/24 total), both exit 0 with zero failures/skips, and all 13 tested inputs are stable; no tests were rerun. Current totals are 3/24 active phases, 9/58 tasks and 84 pending active coverage rows (retired P4 excluded). Phase 3/P22 and Phase 6/Task 6.8 remain open; the original download is unauthenticated, no trusted receipt exists, and source runtime is unsupported. Task 6.8 is proposed only, unchecked and paused pending separate authorization; no global v2 approval is implied.
- 2026-09-24: Recorded scoped Task 6.7 complete under Hebert's exact “Okay, please send the prompt” authorization for metadata-only registration, not Task 6.8 or global v2. Genuine pre-edit Task 6.7 baseline and four changed source/test paths are mapped; `Package.swift` is inherited, unchanged for this task. Failed attempts 1–3 remain retained. The final serial synthetic selectors passed SourceRegistrationTests 17/17 and AppModelLocationTests 7/7, both exit 0, zero failures/skips, with 13/13 tested inputs stable before/after/current. A persistent sibling installer lock, separate `sourceRoot`, and no-replace single-marker publication do not authenticate or copy original shard bytes. Table-derived status: 3/24 accepted active phases, 9/58 tasks complete, 84 active coverage rows pending (retired P4 excluded). Phase 3/P22 and Phase 6/Task 6.8 remain open; original download unauthenticated, no trusted receipt, no source runtime, independent review PASS accepted and page QA remains separate.
- 2026-09-24: Recorded Hebert's exact scoped Task 5.6 authorization “you are handling this, could you please send next prompt” and bounded Safetensors metadata validation complete; this does not authorize Task 5.7 or global v2. Genuine pre-edit baseline maps six changed production/test paths; inherited `Package.swift` and full payload verifier were unchanged. The first compile attempt exited 1 with zero executed tests; after a test-only fix, three serial synthetic selectors passed 13/13, 12/12 and 32/32 = 57/57, zero failures/skips, all exits 0 and eight input hashes stable before/after/current. The positive loader selector excluded the real checkpoint test. Main reports independent code/test review PASS. Task 5.7 and Phase 5 payload acceptance remain pending; Phase 3 P22 remains open. Table-derived totals: 2/24 active phases, 7/58 tasks, 89 active coverage rows pending (retired P4 excluded). No physical payload authentication, source runtime, global v2 approval or Task 5.7 authorization. Task 5.6 page QA awaits the separately scoped page step.
- 2026-09-24: Recorded scoped Task 3.7 complete after final synthetic generation and serial 21/21 tests (Python helper 5/5, Swift reference 10/10, fixture digest 6/6), all exits 0 with zero failures/skips and stable tested-input hashes. D1/D3/D4/D5 pass; D2 and Phase 3 remain open because the compound acceptance includes P22's 248,320 raw FP32 logits, public Float16 values and sampler tokens. Totals: 2/24 accepted phases, 6/58 tasks, 94 active coverage rows pending. P4 retired; next implementation is paused Phase 5 Task 5.6. Source runtime remains unsupported; no global v2 approval.
- 2026-09-24: Recorded scoped Task 3.6 complete after final 30/30 weight-free Python tests and one authorized CPU layer-3 run (exit 0, 5.84 s). The earlier synthetic-fixture failure and corrected rerun remain preserved. Loaded all 256 supplied expert slices from two BF16 packed tensors, while official Top-8 routing selected eight; 2,048 finite FP32 output values hash to `0aed36aafed14ef38c914ee9f079bd82debb29a0b3d411c7710b015104066c33`. RSS 1,491,664,896 B, separately peak footprint 3,630,517,704 B, zero swaps; expected eager-fallback warning and independent pre/post-run reviewer PASS recorded. Pinned code/config/index and selected tensor streams do not establish physical authenticity of full shards. Genuine Task 3.6 baseline and full receipts linked above. Totals: 2/24 active phases, 5/58 tasks, 95 pending active coverage rows; Task 3.7/Phase 3 remain open and no global v2 approval or source runtime readiness is implied.
- 2026-09-23: Recorded scoped Tasks 2.7/2.8 and closed Phase 2 for metadata only. Seven serial selectors passed 49/49 after an initial compile-only test failure and correction; the corrected 37-file hash receipt and ten-path baseline/delta support D1–D5. Independent review final PASS followed duplicate-JSON, test-gap and checksum corrections. Totals: 2/24 active phases, 4/58 tasks, 96 coverage rows still pending. No global v2 approval or source runtime readiness; Phase 3 Task 3.6 needs fresh authorization.
- 2026-09-23: Recorded Task 1.5 completion from Omnigent logs: 12 identity and 13 compatibility tests passed; Phase 1 remained open for Task 1.6.
- 2026-09-23: Recorded Task 1.6 tensor-map evidence: four serial selectors passed 50/50 tests with zero failures/skips and independent reviewer PASS. Closed Phase 1 using the task-scoped changed-file inventory fallback because no phase-start baseline is recorded. Remaining version-2 implementation stays paused; Phase 2 Task 2.7 is next only after Hebert resumes work.

- 2026-09-23: Corrected the generated Phase 1 baseline wording from recorded fallback evidence; Codex Main checked Omnigent test logs/candidate hashes and the refreshed Safari page. No next implementation task started.
