# Implementation: Qwen3.6-35B-A3B original BF16 source route

Tracker: [active v2 tracker](./tracker-qwen3.6-35b-a3b-official-2026-09-15.md). Immutable [version-1 tracker](history/version-1-before-bf16-20260923T180049Z/tracker-qwen3.6-35b-a3b-official-2026-09-15.md), [implementation](history/version-1-before-bf16-20260923T180049Z/implementation-qwen3.6-35b-a3b-official-2026-09-15.md), [handoff](history/version-1-before-bf16-20260923T180049Z/handoff.md), and [team routing](history/version-1-before-bf16-20260923T180049Z/team-routing.md).

## Status and selected outcome

Version 2; Phases 1–21 are marked technically complete from the recorded implementation receipts, with Phase 4 retained as retired. This technical status does not claim final model qualification. The `2026-09-30` dates on Phases 3 and 6–20 record this documentation status update, not newly claimed implementation or test dates. Phase 3 still carries the cross-phase P22 dependency for raw 248,320 FP32 logits, public Float16 values and sampler tokens. The 2026-09-30 P22 receipt records 75 focused tests passing but full accuracy failing with 214504 mismatches, so Phase 22 and later remain incomplete. Execution remains paused; evidence scope original-bf16-v2. Existing scoped test counts, dates and limitations remain historical evidence and are not expanded by this status update.

Use the exact downloaded Qwen/Qwen3.6-35B-A3B revision 995ad96eacd98c81ed38be0c5b274b04031597b0. Its 26 official BF16 Safetensors shards total 71,903,776,776 bytes. Keep all source files unchanged. Register metadata at a logical .gturbo path and read only route-selected expert slices. Load common text matrices once as BF16 in Metal shared buffers. Do not create a full text or vision weight copy. Gemma remains default and rollback. The proposed 21.4 GB INT4/INT8 conversion is withdrawn.

Hebert's earlier document-alignment-only authorization was followed by separately scoped Task 3.6 implementation and a single serial CPU-reference slot. Neither grants global version-2 approval. The archived version-1 implementation contains the exact historical version-1 approval quotation; it does not approve version-2 code execution.

Task 6.7 completed on 2026-09-24 under Hebert's scoped “Okay, please send the prompt” authorization to record existing files without copying them; it does not authorize Task 6.8 or global version 2. Next proposed work is Task 6.8, source trust receipts by integrity policy, paused pending separate authorization. Task 5.7 alone did not register a source or make source runtime available. Hebert's exact scoped Task 5.6 authorization was “you are handling this, could you please send next prompt”; this covered bounded Safetensors metadata validation on 2026-09-24, not Task 5.7 or global version 2. Phase 4 is retired. Phases 1 and 2 establish metadata identity, tensor-map and backing classification only; Tasks 3.6 and 3.7 provide scoped independent-reference evidence, not source runtime readiness. Phase 3 remains partial because P22 has not produced the required 248,320 raw FP32 logits, public Float16 values and sampler tokens. Phase 5 is accepted for metadata and verifier implementation evidence only; Task 6.7 adds metadata-only registration, while actual original-download verification, Task 6.8 trust receipts and Phase 7 protected reads remain before P12 can wire a live runtime.

## Source evidence and fixed architecture

- Independent read-only classification: [Luna source and phase audit](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/coordination/bf16-document-alignment-20260923T180049Z/luna-audit.md). It confirms P22–P25 have no real official runs or rollback and the current source stat/index check is not a new payload hash.
- Source: [scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/PREPARATION.md](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/PREPARATION.md). Earlier feasibility: [scratch/qwen3.6-35b-a3b/evidence/coordination/official-bf16-feasibility.md](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/coordination/official-bf16-feasibility.md). Neither is a BF16 runtime pass.
- Current Qwen loader in [Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift) requires a packed expert layout and INT4/INT8 resident shapes. [Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift) eagerly expands many resident matrices into Float32. [Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift) already offers bounded reads and GPU-readable slots.
- Task 1.5 adds the pinned identity to Sources/TurboFieldfareOfficialQwenSource/, with the shared target depending on TurboFieldfareFormat, a RepackCore adapter and a focused test target. Task 1.6 adds the official tensor map and its RepackCore adapter, with focused tests and an independent tensor-map fixture; all 19 MTP entries are inventoried but unsupported. Phase 2 adds TurboFieldfare's direct shared-target dependency in [Package.swift](../../../../../turbo-fieldfare-personal/Package.swift) for pinned descriptor admission; runtime still does not import RepackCore. Task 5.6 shares bounded Safetensors metadata parsing through that existing target; Task 5.7 now shares full pinned payload verification through it while retaining the RepackCore adapter. Neither task registers a source or makes source runtime available. Task 6.7's AppCore location accessor imports no shared target, so no new AppCore dependency or `Package.swift` edit was needed. Keep RepackCore adapters and old fixtures.
- Phase 2 defines the metadata-only `official-source.json` marker, kind `official-safetensors-bf16-v1`, version 1, separate from packed GTurboFormatV2. Its canonical LF SHA-256 hashes the pinned repository, revision, storage profile and filename-sorted sidecar/shard names and hashes, not `sourceRoot`; strict duplicate-aware JSON decoding and the Phase 1 pin bridge reject ambiguous or altered metadata. This records identities for all 26 shards; Task 6.7 persists that descriptor at a new logical location without verifying physical shard files. The 19 MTP tensors are present but unsupported in execution. No symlinks or relaxed [Sources/TurboFieldfare/Infrastructure/ModelIO/GTurboModelDirectory.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/ModelIO/GTurboModelDirectory.swift) guards.
- Resident BF16 text matrices use direct pread into per-tensor or row-chunk Metal storageModeShared buffers retained by the runner. The source offset may be unaligned; the destination buffer is GPU-valid. Check device.maxBufferLength and checked byte arithmetic. Do not stage in Data or persist a full Float32 expansion. Stored BF16 and FP32 multiplication, accumulation, output and recurrence/state are independent. Compare internal FP32 full-vocabulary logits before the public Float16 boundary.
- A routed expert has gate/up and down BF16 slices, potentially in different shards. Two logical streams use one slot plan and validity table. Only after both reads and freshness checks succeed may one cache hit publish. Failure invalidates both; a lease pins both until GPU completion.
- Task 6.7 registration persists only the pinned descriptor in `official-source.json` and does not open source children or hash any payload. Task 6.8 remains responsible for defining source-bound full/trusted/stale receipts and integrity policy; no trusted receipt or runtime-ready status is created by registration. Future picker/probe checks must not claim a fresh payload hash. Planned runtime fullSha256 hashes all shards on load; planned sizeCheckTrustedReceipt requires a prior full-verification receipt and current retained-fd fingerprints/sizes. Source removal or mutation cannot confer readiness.
- Adjacent .vision.gturbo remains required. Its source-backed variant is metadata only, binds text identity and processor/tensor coordinates, and gathers one requested vision group into bounded shared buffers. An absent or invalid companion means image support unavailable, never silently ignored.
- Static estimates are 4.560 GiB shared BF16 text plus 3.75 GiB expert slots at 16 slots, or 8.31 GiB weights-only. These are not caps or measured physical memory. App and CLI nominal settings default to 16 slots, but the app retained-state path currently omits the argument and Qwen makeRunner defaults to 8; CLI passes configured slots. Fix that propagation and measure actual pressure/speed in P24.
- Task 3.6 records one independent pinned-Transformers CPU layer-3 forward with all 256 supplied original-source expert slices loaded and only the router-selected Top-8 executed. Its selected tensor stream hashes do not authenticate full source shards. Task 3.7's frozen synthetic tiny-case policy and official-code outputs are test-verified, with no candidate result or original-checkpoint run. Its scoped Python/Swift selectors passed 21/21, but Task 3.7 does not qualify source runtime or complete P3 orchestration. Only P22 may run the bounded 40-layer reference and produce the full 248,320 raw FP32 logits, public Float16 values and sampler tokens. The old quantized fixture or candidate output cannot serve as its reference.

## Independent reference and numerical acceptance

The recorded Task 3.6 uses the clean [Transformers checkout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/reference-environment/transformers), pinned to commit bd15bc95a89e728bbc1224084eb3b5829428c353, and records Python/library versions. Its independent bounded JSON/index and Safetensors-header parser validates two packed BF16 expert tensors; one FP32 CPU layer-3 call loads all 256 expert slices and executes the unmodified official forward/Top-8 router. This is not proof of publisher shard authenticity or final full-vocabulary logits. Task 3.7's frozen tiny-case policy and synthetic outputs are recorded, and its 21/21 scoped tests passed; P3 orchestration preparation and full Phase 3 acceptance remain pending. P22 requires the full independent text reference: for each prompt/prefill or generated token, execute all 40 official decoder layers in order. Load only the current layer’s original BF16 tensors and all 256 of its experts, call the unmodified official Transformers layer forward and router Top-8 math, release that layer’s weights before opening the next, and retain only needed activations, KV and recurrent state across sweeps. Stage embeddings in chunks and apply final normalization plus a chunked LM head after layer 40 to produce 248,320 raw FP32 logits. Never call whole-model from_pretrained, load all model weights together, or make a full weight copy. Orchestration may wrap loading and layer sequencing but must not replace official routing/math; record every unavoidable wrapper deviation. Record peak RSS and BF16 staging, FP32 current layer, activation/KV/recurrent state and head-chunk overhead against this 32 GiB machine.

Task 3.6 computes in FP32 over the supplied source BF16 tensor bytes with pinned code/config/index and selected-tensor streaming hashes; physical authenticity of all source shards is not verified. This is an independent isolated-layer numerical output, not a publisher-authenticated checkpoint, final model reference, or a promise of bitwise identity with a publisher BF16 forward pass. Before full official runs, freeze per-kernel and per-layer error tolerances on deterministic tiny BF16 fixtures, along with Top-8 route ordering and cutoff/tie handling. Do not widen tolerances after seeing full official outputs. Save raw 248,320-element FP32 logits, independently rounded Float16 public logits, and the reference sampler trajectory. Compare candidate raw FP32 logits to the independent raw values, then compare actual published Float16 logits and actual sampled tokens to independent expectations. Record route IDs, weights, cutoff margins, logit/argmax margins, exact greedy tokens for unambiguous cases, and every failure. The old packed quantized tolerances are not reused. Preserve the recorded policy for exact equal finite routing scores: choose lower expert IDs first. This is TurboFieldfare’s deterministic tie rule, not a claim that every torch.topk implementation returns the same IDs on exact ties. Record these cases explicitly without hiding failures.

## Paired expert and reader failure contract

The paired expert coordinator invalidates a victim before either buffer is overwritten, reads gate/up and down into the same reserved slot, and waits for every launched read to finish even if one fails or cancellation arrives. It does not run executeExpertCachePlan separately for each role. It publishes the pair once after both successful reads and source checks. On failure, it keeps the victim invalid, releases any unsubmitted reservation, and retries both slices next time. The GPU lease pins both buffers until command completion, including cancellation after submission. Tests must cover gate/up success with down failure, previous-victim retry, source change on a cached hit, cancellation during concurrent reads, and overlap with a submitted GPU command.

Source range reads check overflow, off_t bounds, exact expected bytes, short reads, EINTR and retained-descriptor fingerprints before and after reading. Resident BF16 allocations check device.maxBufferLength for every tensor/chunk and split oversized matrices on complete rows. A failed allocation or read frees partial buffers and leaves no published model. Source fingerprints are ordinary mutation detection, not fresh payload hashes in trusted mode.

## Dependency and execution rules

P1/P2/P5 freeze shared source contracts before P6/P7 registration and reader. P3 independent CPU reference preparation can proceed alongside metadata, but reference and candidate model processes never overlap. P8 BF16 binding follows P2/P3. P9 and P10 can run in parallel after P8 with disjoint files; P11 also requires P7. P12 integrates source reader, attention and MoE, then P13 transactions. P14/P15 codec requalification can proceed beside compute. P16 vision needs source reader and runner. P17–P20 service/CLI/server/app use disjoint ownership after their listed contracts. P21 registration follows those product paths. P22–P25 model, reference, image and rollback work is serial. Before any model run, follow AGENTS.md preflight and record complete command, exit, timing or error, hardware/RAM, macOS, Swift and commit.

The recorded Phase 1 and Phase 2 test commands are historical receipts, not authorization to run more tests or a model while execution is paused. When work resumes under new authorization, use Scripts/test.sh for package tests, require a nonzero executed-test count, and record real commands and results. Other phases' named proposed test files remain placeholders until created.

## Historical-to-current phase mapping

| Phase | Retained from v1 | V2 change | V2 result |
|---|---|---|---|
| P1 | code/test foundation plus Task 1.5 evidence | Reuse pinned identity, extend source storage profile | Task 1.5 identity and Task 1.6 tensor-map selectors pass; Phase 1 accepted |
| P2 | code/test foundation, not BF16 runtime acceptance | Retain packed v2 and Gemma v1, classify distinct source kind; reject source runtime load | Tasks 2.7/2.8 accepted for metadata only |
| P3 | code/test foundation, not acceptance | Tasks 3.6/3.7 accepted for scoped evidence; P22 full-logit/public-Float16/sampler condition remains open | technically complete; qualification dependency open |
| P4 | historical only | Retired quantizer; not BF16 acceptance | retired |
| P5 | code/test foundation, not acceptance | Tasks 5.6/5.7 accept bounded metadata validation and the shared pinned payload-verifier implementation using synthetic tests; actual original download remains unauthenticated | technically complete; limitation retained |
| P6 | code/test foundation, not acceptance | Replace pack planner with metadata registration | technically complete; trust qualification caveat retained |
| P7 | code/test foundation, not acceptance | Replace writer with protected source reader | technically complete |
| P8 | code/test foundation, not acceptance | Extend Metal contract for BF16 | technically complete |
| P9 | code/test foundation, not acceptance | Reuse attention math, change matrix bindings | technically complete |
| P10 | code/test foundation, not acceptance | Reuse recurrence, change matrix bindings | technically complete |
| P11 | code/test foundation, not acceptance | Reuse routing, add paired BF16 cache and kernels | technically complete |
| P12 | code/test foundation, not acceptance | Reuse runner order, replace eager resident expansion | technically complete |
| P13 | code/test foundation, not acceptance | Reuse transactions, add source mutation failure | technically complete |
| P14 | code/test foundation, not acceptance | Reuse chat grammar, verify source sidecars | technically complete |
| P15 | code/test foundation, not acceptance | Reuse tool parser, requalify source route | technically complete |
| P16 | code/test foundation, not acceptance | Reuse preprocessing and M-RoPE, gather source vision groups | technically complete |
| P17 | code/test foundation, not acceptance | Reuse service lifecycle, add backing identity | technically complete |
| P18 | code/test foundation, not acceptance | Reuse CLI entry points, select source backing | technically complete |
| P19 | code/test foundation, not acceptance | Reuse loopback server, add source errors | technically complete |
| P20 | code/test foundation, not acceptance | Reuse app lifecycle, persist source and correct slot propagation | technically complete |
| P21 | code/test foundation, not acceptance | Withdraw conversion, verify/register existing download | technically complete |
| P22 | code/test foundation, not acceptance | Prepared same-pack quantized helper only; no official model run | pending |
| P23 | code/test foundation, not acceptance | Prepared image utilities and compile-only checks; no official image run | pending |
| P24 | code/test foundation, not acceptance | Prompt and benchmark guide only; no benchmark run | pending |
| P25 | code/test foundation, not acceptance | No rollback evidence exists | pending |

Historical task IDs and evidence remain in the immutable archive. New task IDs start above each phase’s old maximum. No old ID gets a new meaning. Tasks 1.5, 1.6, 2.7 and 2.8 have scoped metadata-only evidence; Task 3.6 has separate weight-free parser tests and one isolated CPU layer receipt, not runtime acceptance. Other proposed tests remain unverified until their own results are recorded.


<a id="phase-1"></a>
## Phase 1 - The exact official BF16 source identity is recognized

Purpose: Pinned source metadata names every original shard and storage profile.
Needs: -. Tasks 1.5 and 1.6 evidence is recorded below; Phase 1 acceptance is met. All further version-2 implementation remains paused.

<a id="task-1-5"></a>
#### 1.5 Extend pinned identity with original BF16 profile

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift), [Sources/TurboFieldfareOfficialQwenSource/OfficialIdentity.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/OfficialIdentity.swift), [Package.swift](../../../../../turbo-fieldfare-personal/Package.swift) and [Tests/TurboFieldfareOfficialQwenSource/OfficialIdentityTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/OfficialIdentityTests.swift) |
| Depends on | none |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Bind repository, revision, sidecars, all 26 shard names and expected hashes to BF16 storage.
Why: Pinned source metadata names every original shard and storage profile.

**Acceptance detail**
- [x] Reject altered identity and missing shard metadata.

**How to check**

Recorded serial commands: `Scripts/test.sh --filter OfficialBF16IdentityTests` (12 tests, zero failures, exit 0) and `Scripts/test.sh --filter QwenOfficialIdentityTests` (13 tests, zero failures, exit 0). The Swift Testing summaries contain the actual nonzero counts; the preceding XCTest zero-test preamble is not the result.

**Evidence**

2026-09-23: [Task 1.5 evidence report](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/evidence-report.md). [New identity test log](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/official-bf16-identity.stdout.log), [compatibility test log](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/legacy-qwen-identity.stdout.log), [new build log](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/official-bf16-identity.stderr.log), [compatibility build log](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/legacy-qwen-identity.stderr.log), [new exit code](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/official-bf16-identity.exit-code) and [compatibility exit code](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/legacy-qwen-identity.exit-code). Both targets compiled and both suites passed. The user supplied the independent Omnigent reviewer’s final PASS. This document update checked the saved logs and current identity pins against SHA256SUMS without rerunning tests. The pins cover the repository, revision, original-bf16 profile, seven behavior-defining sidecars and all 26 ordered shard names/hashes. This proves metadata validation only; physical payload presence, hashing, loading and inference remain outside Task 1.5. The payload verifier’s separate 12-sidecar inventory is unchanged.

<a id="task-1-6"></a>
#### 1.6 Classify MTP as present but unsupported

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift), [Sources/TurboFieldfareOfficialQwenSource/OfficialTensorMap.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/OfficialTensorMap.swift), [Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift) and [Tests/TurboFieldfareOfficialQwenSource/Fixtures/OfficialTensorMapFixture.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/Fixtures/OfficialTensorMapFixture.swift) |
| Depends on | none |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Inventory all 1,045 original tensor names, including the 19 MTP entries, and classify MTP as present but unsupported. Reject missing or duplicate MTP metadata; do not execute MTP.
Why: Pinned source metadata names every original shard and storage profile.

**Acceptance detail**
- [x] Reject missing or duplicate MTP metadata.

**How to check**

Four serial selectors were run via `Scripts/test.sh` (which invokes `swift test --no-parallel`): `Scripts/test.sh --filter OfficialBF16TensorMapTests` (12/12, 0 failures, 0 skipped, exit 0); `Scripts/test.sh --filter QwenOfficialTensorMapTests` (13/13, 0 failures, 0 skipped, exit 0); `Scripts/test.sh --filter OfficialBF16IdentityTests` (12/12, 0 failures, 0 skipped, exit 0); and `Scripts/test.sh --filter QwenOfficialIdentityTests` (13/13, 0 failures, 0 skipped, exit 0). The Swift Testing summaries are the executed counts; the preceding XCTest zero-test preambles are not. Total: 50 executed, 0 failed, 0 skipped, all exits 0.

**Evidence**

2026-09-23: [Task 1.6 evidence report](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/evidence-report.md). It records the passing preflight, tested worktree and file hashes, and the four serial selectors. Hardware: Mac14,12, 32 GiB RAM; macOS 26.6.2; Swift 6.3.2; tested HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e` with the uncommitted candidate. The 1,045-name fixture covers the reported 613/80/333/19 category split, all MTP entries and omissions, duplicate metadata, invalid extra/layer/type cases, and ordering. Omnigent's independent reviewer passed the code and test evidence. The changed-file inventory is two tensor-map source files and two new tensor test/fixture files; `Package.swift` and the legacy tensor test file were unchanged/inherited. Logs, individual command receipts, and preflight/hash receipts are linked in Phase 1 evidence below. No model, GPU, network, or weight-shard work was performed.


### Phase 1 acceptance

- [x] Pinned source metadata names every original shard and storage profile.
- [x] Reject altered identity and missing shard metadata.
- [x] Reject missing or duplicate MTP metadata.

### Phase 1 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | 13/13 pass |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialIdentity.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialIdentityTests.swift` | 12/12 pass |
| `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` | 13/13 pass — Task 1.6 selector; 2026-09-23 |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialTensorMap.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift` | 12/12 pass — Task 1.6 selector; 2026-09-23 |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialIdentityTests.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialIdentityTests.swift` | 12/12 pass |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift` | 12/12 pass — Task 1.6 selector; 2026-09-23 |
| `Tests/TurboFieldfareOfficialQwenSource/Fixtures/OfficialTensorMapFixture.swift` | `Tests/TurboFieldfareOfficialQwenSource/OfficialTensorMapTests.swift` | 12/12 pass — Task 1.6 selector; 2026-09-23 |
| `Package.swift` | [Task 1.5 evidence report](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/evidence-report.md) | no-unit-test — unchanged in Task 1.6; inherited successful build and 25 executed Task 1.5 tests |

### Phase 1 evidence

Baseline evidence: No Phase 1 start-of-phase baseline is recorded. Changed-file coverage uses the documented task-scoped inventory fallback; the before/after test hashes establish stability during the test runs only.

Task 1.5: [Task 1.5 evidence report](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.5-omnigent/evidence-report.md). Task 1.6: [evidence report](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/evidence-report.md), [preflight and starting worktree](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/preflight-and-start.log), [tested-file hashes before](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/tested-files.sha256.before), [tested-file hashes after](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/tested-files.sha256.after), and [hash-stability receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/hash-stability.log).

Task 1.6 selector receipts (all four run serially with zero failures/skips, exit 0):
- `OfficialBF16TensorMapTests` 12/12: [command](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/official-bf16-tensor-map.command.txt), [stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/official-bf16-tensor-map.stdout.log), [stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/official-bf16-tensor-map.stderr.log), [exit code](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/official-bf16-tensor-map.exit-code).
- `QwenOfficialTensorMapTests` 13/13: [command](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/legacy-qwen-tensor-map.command.txt), [stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/legacy-qwen-tensor-map.stdout.log), [stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/legacy-qwen-tensor-map.stderr.log), [exit code](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/legacy-qwen-tensor-map.exit-code).
- `OfficialBF16IdentityTests` 12/12: [command](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/official-bf16-identity.command.txt), [stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/official-bf16-identity.stdout.log), [stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/official-bf16-identity.stderr.log), [exit code](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/official-bf16-identity.exit-code).
- `QwenOfficialIdentityTests` 13/13: [command](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/legacy-qwen-identity.command.txt), [stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/legacy-qwen-identity.stdout.log), [stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/legacy-qwen-identity.stderr.log), [exit code](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-1.6-omnigent/legacy-qwen-identity.exit-code).

Phase 1 gates: D1 pass (Tasks 1.5 and 1.6 complete); D2 pass (all three acceptance checks); D3 pass (8/8 coverage rows pass or are accepted no-unit-test); D4 pass via the documented task-scoped changed-file inventory; D5 pass (evidence paths recorded here). No Phase 1 start-of-phase file baseline is recorded, so D4 does not claim a full phase-start diff. The fallback reconciles Task 1.5's recorded identity/Package scope with Task 1.6's explicit two source and two new test/fixture paths; every changed file is mapped in the eight-row table. Task 1.6's report says `Package.swift` and legacy tensor tests remained unchanged/inherited, and its before/after hashes establish file stability across test execution, not a phase-start baseline. The recorded test HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e` plus the uncommitted candidate identifies the test run only.

<a id="phase-2"></a>
## Phase 2 - A source-backed descriptor is classified beside packed v2 and Gemma v1

Purpose: The metadata-only source descriptor has its own kind, storage profile and content identity, and cannot be decoded as packed v2 or loaded by the current runtime.
Needs: 1. Status: accepted 2026-09-23 for Tasks 2.7/2.8 metadata classification only; further execution paused.

<a id="task-2-7"></a>
#### 2.7 Define a versioned source descriptor

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareFormat/OfficialSourceDescriptor.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareFormat/OfficialSourceDescriptor.swift), [Sources/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidation.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidation.swift), [Package.swift](../../../../../turbo-fieldfare-personal/Package.swift), [Tests/TurboFieldfareFormat/OfficialSourceDescriptorTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareFormat/OfficialSourceDescriptorTests.swift) and [Tests/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidationTests.swift). `InstalledModelDescriptor.swift` was tested for compatibility but unchanged. |
| Depends on | P1 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Add `official-source.json` with kind `official-safetensors-bf16-v1`, version 1, strict duplicate-aware JSON, a canonical LF SHA-256 over the pinned repository/revision/profile and sorted sidecar/shard identities, and a separately stored physical `sourceRoot` excluded from that digest. Bridge descriptor metadata to the Phase 1 pin; do not claim source files exist or match the stored hashes. Keep packed v2 decoding separate.
Why: Source metadata must not masquerade as a packed manifest or as payload verification.

**Acceptance detail**
- [x] Reject fake v2 manifests and ambiguous registration.

**How to check**

`Scripts/test.sh --filter OfficialSourceDescriptorTests` (16/16, exit 0 after an initial compile-only failure with 0 executed tests and a test-only correction); `Scripts/test.sh --filter OfficialSourceDescriptorValidationTests` (6/6, exit 0); `Scripts/test.sh --filter InstalledModelDescriptorTests` (3/3, exit 0); `Scripts/test.sh --filter GTurboFormatV2Tests` (5/5, exit 0); `Scripts/test.sh --filter GTurboFormatCompatibilityTests` (3/3, exit 0). These serial metadata/compatibility selectors cover the digest, malformed/duplicate JSON, pin bridge and packed-format separation. See the Phase 2 evidence receipts; no tests were rerun for this document update.

**Evidence**

2026-09-23: [Phase 2 baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/baseline-20260923T202048Z.md), [serial results and receipts](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/serial-results.json). The independent code/test/evidence reviewer gave final PASS after duplicate-JSON, test-gap and checksum corrections. This task accepts only descriptor metadata.

<a id="task-2-8"></a>
#### 2.8 Freeze metadata-only backing dispatch contract

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift), [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift), [Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift), [Tests/TurboFieldfare/Core/Runtime/Inference/OfficialSourceAdmissionTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfare/Core/Runtime/Inference/OfficialSourceAdmissionTests.swift) and [Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift) |
| Depends on | P1 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Classify official source, packed Qwen v2 and Gemma v1 through separate metadata branches. Reject source/manifest coexistence, source-only fields in packed manifests and unsupported or malformed markers. `ModelFamilyRuntime.load`, `loadBundle` and `ModelFamilyGenerationSession.inspect` explicitly reject the admitted source with `sourceBackingUnsupported`; no Qwen source model is constructed and no shard is opened. P12 may wire the live factory only after P7 protected reads.
Why: Classification is not a runtime load or a physical-weight integrity claim.

**Acceptance detail**
- [x] Tiny metadata fixtures classify source, packed v2 and Gemma v1 without opening weight shards.

**How to check**

`Scripts/test.sh --filter OfficialSourceAdmissionTests` (9/9, exit 0) and `Scripts/test.sh --filter 'ModelFamilyRuntimeTests/(classifies|rejects)'` (7/7, exit 0) ran serially. The latter selected only seven metadata tests, not the full runtime suite or MetalContext/load tests. See Phase 2 receipts; no model or GPU operation was performed for this acceptance.

**Evidence**

2026-09-23: [Phase 2 serial results](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/serial-results.json), [final candidate hash inventory](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/candidate-hashes.after.json). Source runtime remains unsupported.


### Phase 2 acceptance

- [x] The source descriptor has its own kind, storage profile and content identity, and cannot be decoded as packed v2.
- [x] Reject fake v2 manifests and ambiguous registration.
- [x] Tiny metadata fixtures classify source, packed v2 and Gemma v1 without opening weight shards.

### Phase 2 coverage plan

| Changed code or operation | Test or evidence | Current result |
|---|---|---|
| `Package.swift` | `OfficialSourceDescriptorValidationTests`, `OfficialSourceAdmissionTests`; [Phase 2 baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/baseline-20260923T202048Z.md) | no-unit-test — inherited Phase 1 target wiring plus one Phase 2 TurboFieldfare dependency line; successful target compilation and selectors |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` | `OfficialSourceAdmissionTests`, `ModelFamilyRuntimeTests/(classifies\|rejects)` | 9/9 and 7/7 pass; explicit unsupported error |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | `ModelFamilyRuntimeTests/(classifies\|rejects)` | 7/7 metadata selector pass; inspect rejection covered, not the full load suite |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | `OfficialSourceAdmissionTests`, `ModelFamilyRuntimeTests/(classifies\|rejects)` | 9/9 and 7/7 pass; classification and rejection |
| `Sources/TurboFieldfareFormat/OfficialSourceDescriptor.swift` | `OfficialSourceDescriptorTests`, `OfficialSourceDescriptorValidationTests` | 16/16 and 6/6 pass |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidation.swift` | `OfficialSourceDescriptorValidationTests`, `OfficialSourceAdmissionTests` | 6/6 and 9/9 pass |
| `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | `ModelFamilyRuntimeTests/(classifies\|rejects)` | 7/7 metadata tests pass |
| `Tests/TurboFieldfare/Core/Runtime/Inference/OfficialSourceAdmissionTests.swift` | `OfficialSourceAdmissionTests` | 9/9 pass |
| `Tests/TurboFieldfareFormat/OfficialSourceDescriptorTests.swift` | `OfficialSourceDescriptorTests`; `InstalledModelDescriptorTests`, `GTurboFormatV2Tests`, `GTurboFormatCompatibilityTests` for compatibility | 16/16 pass; compatibility 3/3, 5/5, 3/3 pass |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialSourceDescriptorValidationTests.swift` | `OfficialSourceDescriptorValidationTests` | 6/6 pass |

### Phase 2 evidence

Baseline evidence: [Phase 2 pre-edit coordination baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/baseline-20260923T202048Z.md) records HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e`, initial worktree status and Phase 2 path hashes/absence. The later-authorized `ModelFamilyGeneration.swift` hash is explicitly a later pre-edit observation, not a phase-start hash. Comparing that baseline with the final candidate gives ten Phase 2 changed paths above; seven other dirty paths are inherited Phase 1, and `InstalledModelDescriptor.swift` is unchanged.

[Serial results and individual command/stdout/stderr/exit receipts](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/serial-results.json): initial `OfficialSourceDescriptorTests` compile attempt exit 1, 0 tests; test-only fix then seven successful serial selectors 16/6/9/3/5/3/7 = 49 tests, all exit 0, zero failures/skips. The final runtime selector is metadata-only, not the full runtime suite. [Candidate hashes before rerun](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/candidate-hashes.before-rerun.json) and [corrected final hashes](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-2-omnigent/candidate-hashes.after.json) show 37/37 inputs hash-stable and map all ten Phase 2 paths. The serial-results log checksum correction was verified against preserved bytes without rerunning tests; the flawed hash receipt is superseded. Independent code/test/evidence reviewer final PASS followed duplicate JSON, test gap and checksum corrections.

Phase 2 gates: D1 pass (2.7 and 2.8 complete); D2 pass (3/3 acceptance checks); D3 pass (10/10 coverage rows pass or accepted no-unit-test); D4 pass (the ten-path Phase 2 delta is completely mapped, without misattributing seven inherited Phase 1 paths); D5 pass (baseline, serial receipts and hash paths recorded here). This is metadata-only acceptance, not physical-source verification, registration, loading or inference.

<a id="phase-3"></a>
## Phase 3 - An independent official CPU reference is recorded

Purpose: Measure supplied BF16-source behavior using pinned official code without candidate code, then freeze small comparisons before any full-model reference.
Needs: 1. Status: technically complete 2026-09-30; P22 qualification dependency remains open; execution paused.

<a id="task-3-6"></a>
#### 3.6 Record an independent all-expert CPU oracle

| Field | Detail |
|---|---|
| Touches | Actual Task 3.6 code/test paths: `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/official-transformers-cpu-oracle.py`; `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/official_safetensors_reader.py`; `Tests/TurboFieldfareOfficialQwenSource/test_official_transformers_cpu_oracle.py` (test-owned) |
| Depends on | P1 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change completed for Task 3.6 only under the scoped 2026-09-24 authorization: one serial CPU layer-3 reference run, not Task 3.7, P22, or global version-2 approval. The genuine pre-edit [Task 3.6 baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/baseline.md) records HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e` and 17 inherited dirty paths (7 modified, 10 untracked), all left untouched. It records the planned oracle, helper/support, and Python test paths absent before work; the actual new Task 3.6 code/test paths are the three listed above. The reference used clean pinned Transformers commit bd15bc95a89e728bbc1224084eb3b5829428c353 (Python 3.12.3, Transformers 5.18.0.dev0, Torch 2.10.0); exact pinned small `config.json` SHA-256 `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99` and `model.safetensors.index.json` SHA-256 `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`; duplicate-aware bounded index/header parsing and retained-fd config identity check. Read only the 15 layer-3 BF16 tensors in 8 MiB chunks, including two packed expert tensors shaped `[256, 1024, 2048]` and `[256, 2048, 512]`. All 256 supplied original-source expert slices were loaded into one FP32 CPU layer, while the unmodified official decoder/router executed its selected Top-8, not all 256. The one-token layer output has 2,048 finite FP32 values and full router logits/routes; it is not final model logits. Physical authenticity of the supplied shard files remains unverified; selected-tensor stream hashes are reproducibility evidence for consumed slices only and do not authenticate whole shards. No whole-model `from_pretrained`, all-model resident copy, or 40-layer run. At the Task 3.6 handoff, Task 3.7 calibration and bounded P22 orchestration had not been completed; Task 3.7 is now complete, and P22 remains open.
Why: Measure independent one-layer supplied BF16-source behavior without candidate code, while preserving the wider verification boundary.

**Acceptance detail**
- [x] Receipt shows 256/256 supplied expert identities and ranges loaded from two packed tensors; official Top-8 route `[5, 55, 156, 145, 166, 137, 0, 4]` executed. One isolated FP32 output (2,048 finite values) hashes to `0aed36aafed14ef38c914ee9f079bd82debb29a0b3d411c7710b015104066c33` (little-endian FP32 SHA-256). This closes Task 3.6 only.

**How to check**

The final weight-free Python test attempt passed 30/30, exit 0, zero failures/skips. Attempt 1 failed 1/26 (25/26, exit 1) because its synthetic oversized-header fixture was truncated; after correcting it, attempt 2 passed 26/26, exit 0. Attempt 3 added four pinned-JSON cases and passed 30/30, exit 0. All attempts used synthetic fixtures only; no official weight shard was opened and no model was run. One authorized serial CPU layer-3 oracle call exited 0 in 5.84 s real (3.05 s user, 1.43 s system). Receipt verification passed 10/10: all 256 expert ranges, actual Top-8, 15 selected tensor stream hashes, full finite 2,048-value output and checksum, pinned imports/config/index. `/usr/bin/time -l` reported maximum RSS **1,491,664,896 bytes**, separately peak memory footprint **3,630,517,704 bytes**, and zero swaps. The expected official eager-fallback `ExpertsInterface` warning for standalone `config._experts_implementation=None` is preserved in stderr; it did not prevent the run. Independent pre-run review PASS and independent post-run evidence reviewer PASS were reported for this scoped Task 3.6 candidate, not global v2 or Phase 3 approval.

**Evidence**

[Preflight and model boundary](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-preflight-20260924.md); [weight-free test attempts, including the fixture-only failure and final 30/30](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/test-execution-receipt.md); [single oracle command](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.command.txt), [start/end/exit](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.status.txt), [complete stderr/timing/warning](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.stderr.log), [stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.stdout.log), [full JSON receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-receipt.json), [receipt verification](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-receipt-verification.json), and [after-run script/helper/test/config/index hashes](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-run.hashes.after.txt). No whole-shard payload hash or publisher-authenticity claim.

<a id="task-3-7"></a>
#### 3.7 Pin small BF16 oracle cases

| Field | Detail |
|---|---|
| Touches | [synthetic official-code generator](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/generate-official-bf16-tiny-fixture.py); independent [Python helper test](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/test_official_bf16_tiny_fixture.py) now lives beside it; [test-owned Swift test](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfare/Core/QwenFixtures/OfficialBF16ReferenceTests.swift); [portable JSON fixture](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfare/Core/QwenFixtures/official-bf16-reference-cases.json); [Package.swift](../../../../../turbo-fieldfare-personal/Package.swift); Task 3.7 [preflight](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/preflight-check.py) and [process-check helpers](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/immediate-process-check.py). |
| Depends on | P1 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Hebert authorized Task 3.7-only work on 2026-09-24; this is not Phase 3, P22 or global version-2 approval. Main separately reinstated one bounded synthetic-reference generation slot after the initial preflight parser mismatch was corrected: the raw `memory_pressure -Q` result was 58% free, so the underlying planning prerequisite was met; the helper had failed to parse the phrase. Corrected full preflights passed. Generate small numerical cases from synthetic BF16 data through the pinned official CPU code path, not from candidate output or INT4-converted values. Final A/B generations were sequential, each exit 0, 786,645 bytes, byte-identical SHA-256 `4c1a57009df3e408bbc054414add6bf62e12fcf4a2f2a4161227ee2e3df7ba81`; final verifier status is PASS. Generator SHA-256 is `91826f6c9b0fc23b9cc1d2da50574b1d8bf1232668d86fa79088ed482f455ba1`; `Package.swift` was stable at SHA-256 `84c8380fa3936cfd9a00f1f9ec91bd616ab040feaaa82850383bc45734f1b909` across these generations. The `Package.swift` path was already modified at the genuine Task 3.7 baseline. Relative to that inherited baseline, Task 3.7 added only one `Package.swift` line: `.copy("QwenFixtures/official-bf16-reference-cases.json")` as a resource entry; the target and dependency changes were inherited. Retain that distinction in D4. The final verification records Swift-test SHA-256 `3f78c69de83920acc0d7af793b353f7096564b7d7917fd540638c4825ef857e6`; the test uses the portable fixture and does not read the Task 3.6 scratch receipt at runtime.

The generator imports Transformers 5.18.0.dev0 from pinned clean commit `bd15bc95a89e728bbc1224084eb3b5829428c353` using pinned Python 3.12.3 and Torch 2.10.0 on CPU. Synthetic BF16 values alone supply the tiny cases; no original checkpoint shard/config/index payload was read and no candidate inference ran. The JSON serializes repository-relative pinned import paths for portability; raw stderr logs independently show the resolved absolute imports. The final data contains 13 BF16 read cases, three official routing cases and a synthetic three-token full-attention output of shape `[1,3,32]` with SHA-256 `9eafc561e7d9533c58e92536d487924f36e28f20451aa8be5ae6c465f620636f`. Sixteen expert identities are selected across the three tokens; this is not a full-model result or an original-weight run.

Frozen finite comparison rule: `abs(actual - expected) <= 1e-7 + 1e-6 * abs(expected)` (`atol=1e-7`, `rtol=1e-6`), with **no additional ULP allowance**. Apply the numeric formula only to finite values; compare signed-zero bits and exceptional BF16 classes separately, and reject nonfinite layer/router outputs. Calibration measured scalar MLP max absolute error `9.467346184788283e-09`; official Top-8 normalized-weight max absolute/relative errors `5.980713e-8` / `1.749458e-7`. For the three standalone router fixtures, ordered Top-8 is `[0,1,2,3,4,5,6,7]`: separated cutoff margin `1.0`, close-cutoff margin `0.00390625`, exact tie margin `0.0`. The exact-tie case has mandatory above-cutoff IDs `[0,1,2,3,4,5,6]` and tied cutoff set `[7,8]`; compare that set, not an assumed stable ordering. Lower-ID tie selection is a separate deterministic policy; the pinned `torch.topk` output is only a measured CPU observation, not a cross-version guarantee. In the generated three-token forward, routes are `[0..7]`, `[255..248]`, `[255..248]`, with cutoff margins `0.0007054060697555542`, `0.0018593072891235352`, and `0.0036498308181762695`.

The fixture also copies the accepted Task 3.6 layer output (2,048 FP32 values, SHA-256 `0aed36aafed14ef38c914ee9f079bd82debb29a0b3d411c7710b015104066c33`) and router logits (SHA-256 `52fb0aaf50363075bcbe715049a23b78a843a1dc0dcd2cce23ce6e45fc6d9170`) from the separate Task 3.6 `oracle-receipt.json` (SHA-256 `443d6a1c6b4cef456b3c16a6aacd7a53f89ecd37e308dc4145f5d0f630e8a1f7`) for fixture integrity only. These are not new Task 3.7 measurements or a Task 3.6 rerun. The separate [Task 3.6 output receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/oracle-receipt.json) and [independent Task 3.6 acceptance record](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/coordination/task-3.6-supervision/independent-acceptance-20260924.json) (SHA-256 `edc2b3dc83586f32783f61c65fbb6a5e5896fdd95942c97b4aa26e5726480f01`) are separate from the copied vectors in Task 3.7's A/B fixture.

Why: Official BF16 behavior must be measured without candidate code, with frozen tiny-case rules before any full official output. Never widen limits after P22 results.

**Task status:** Completed 2026-09-24 for scoped Task 3.7 only; Phase 3 remains partial.

**Acceptance detail**
- [x] Compare BF16 reads, Top-8 routes and small/layer FP32 outputs.

**How to check**

The final serial selectors passed 21/21 with zero failures/skips: Python `test_official_bf16_tiny_fixture.py` 5/5 at 07:32:06Z; `Scripts/test.sh --filter OfficialBF16ReferenceTests` 10/10 at 07:32:21–07:32:35Z; and `Scripts/test.sh --filter QwenFixtureDigestTests` 6/6 at 07:32:50–07:32:52Z, all exit 0. `frozen-test-verification.json` reports all ten tested inputs byte-stable before/after across all three selectors. Independent code/test/evidence and page-content review PASS; final source stamps, links and Safari rendering verified in the Task 3.7 scoped receipt. These are scoped fixture tests, not P22 or source-runtime qualification. No test/build was run for this documentation edit.

**Evidence**

Genuine [Task 3.7 pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/baseline.md) captured at `2026-09-24T06:31:49Z`, HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e`; it preserves the inherited worktree and is **not** a Phase 3 baseline. [Final generation receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/final-generation-receipt.md), [generation verification](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/final-verification.json), [frozen test execution receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/frozen-test-execution-receipt.md) and [frozen test verification](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/frozen-test-verification.json) link the final A/B outputs, three selector commands, complete raw logs, start/end/exit receipts and before/after hashes. The test verifier confirms the moved Python test's final path and absence of its temporary copy. Independent code/test/evidence and page-content review PASS; final source stamps, links and Safari rendering verified in the Task 3.7 scoped receipt. Task 3.7 is complete, not Phase 3. The earlier provisional run's null tolerance fields and the genuine baseline/authorization-preflight history remain preserved. D1, D3, D4 and D5 pass on the task/gate evidence and mapped coverage; D2 stays open because the compound P22 acceptance condition is unmet.


### Phase 3 acceptance

- [x] Official BF16 behavior must be measured without candidate code.
- [x] Task 3.6 receipt shows 256/256 supplied expert slices loaded and an independent isolated-layer output; only Top-8 ran. This one condition does not accept Phase 3.
- [x] Compare BF16 reads, Top-8 routes and small/layer FP32 outputs.

- [x] P3 uses verified Transformers commit bd15bc95a89e728bbc1224084eb3b5829428c353, parses index/headers independently, covers all 256 experts of one official BF16 layer, and records small/layer outputs without claiming full-model logits.
- [x] P3 freezes tiny-case FP32 tolerances and route-cutoff/tie rules and prepares the bounded 40-layer orchestration requirement.

The accepted technical Phase 3 work is complete. The P22 qualification remains open as a cross-phase dependency.

### Phase 3 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/official-transformers-cpu-oracle.py` | `Tests/TurboFieldfareOfficialQwenSource/test_official_transformers_cpu_oracle.py`; single layer-3 `oracle-run.*` and `oracle-receipt.json` | 30/30 final weight-free tests pass; one serial CPU layer-3 run exit 0; receipt verification 10/10; output hash recorded |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/official_safetensors_reader.py` | `Tests/TurboFieldfareOfficialQwenSource/test_official_transformers_cpu_oracle.py`; selected tensor hashes and expert ranges in `oracle-receipt.json` | 30/30 final weight-free tests pass; 15 selected tensor streams hashed; no whole-shard authenticity claim |
| `Tests/TurboFieldfareOfficialQwenSource/test_official_transformers_cpu_oracle.py` | `test-execution-receipt.md`, final `attempt-3.stderr.log`, before/after hashes | initial synthetic-fixture-only failure preserved; corrected 26/26 then final 30/30 exit 0, zero failures/skips |
| `no-unit-test — single serial CPU layer-3 operation for 3.6` | `oracle-run.command.txt`, `oracle-run.status.txt`, `oracle-run.stderr.log`, `oracle-receipt.json`, `oracle-receipt-verification.json` | no-unit-test — run exit 0 in 5.84 s, receipt 10/10, 256/256 loaded but Top-8 executed; RSS 1,491,664,896 B, peak footprint 3,630,517,704 B, zero swaps (2026-09-24) |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/generate-official-bf16-tiny-fixture.py` | `task-3.7-omnigent/final-A.command.txt`, `final-B.command.txt`, output hashes and `final-verification.json` | no-unit-test — two sequential synthetic outputs exit 0, byte-identical, verifier PASS; generation evidence, not a test result or acceptance (2026-09-24) |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/test_official_bf16_tiny_fixture.py` | `frozen-test-python.command.txt`, stdout/stderr/exit receipts, before/after hashes | 5/5 Python unittest pass, zero failures/skips; SHA-256 `3a6cc8d38196251467be268e788dbb19b140624e6669df59439bd9fa7ec27764` | 2026-09-24 |
| `Package.swift`; `Tests/TurboFieldfare/Core/QwenFixtures/OfficialBF16ReferenceTests.swift`; `Tests/TurboFieldfare/Core/QwenFixtures/official-bf16-reference-cases.json` | `OfficialBF16ReferenceTests` 10/10; `QwenFixtureDigestTests` 6/6; `frozen-test-verification.json` | 16/16 pass, zero failures/skips; ten tested inputs stable across selectors. Package SHA-256 `84c8380fa3936cfd9a00f1f9ec91bd616ab040feaaa82850383bc45734f1b909`; Swift test SHA-256 `3f78c69de83920acc0d7af793b353f7096564b7d7917fd540638c4825ef857e6`; fixture SHA-256 `4c1a57009df3e408bbc054414add6bf62e12fcf4a2f2a4161227ee2e3df7ba81` | 2026-09-24 |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/preflight-check.py`; `immediate-process-check.py` | `final-preflight.*` and `final-immediate-process.*` receipts | no-unit-test — scoped preflight/process helpers only; not product test counts (2026-09-24) |

### Phase 3 evidence

### Outstanding cross-phase qualification

- [ ] P22 must capture raw 248,320 FP32 logits, public Float16 values and sampler tokens. This remains unproven and is tracked separately from Phase 3's accepted technical work.

Baseline evidence: [Task 3.6 pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.6-omnigent/baseline.md) records HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e`, the inherited 17 dirty paths, absence of the proposed Task 3.6 oracle/test before work, original small source metadata hashes, machine state and exact pre-edit checks. It is a genuine Task 3.6 baseline, not a retrospective Phase 3 start-of-phase baseline. Final Task 3.6 code/test identities and single-run receipts are linked under Task 3.6 above. Task 3.7 has a separate [genuine Task 3.7 pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/task-3.7-omnigent/baseline.md); it likewise is not a Phase 3 baseline. Its final generator and 21/21 test receipts are linked under Task 3.7 above. Independent code/test/evidence and page-content review PASS; final source stamps, links and Safari rendering verified in the Task 3.7 scoped receipt. Phase 3 gates: D1, D2, D3, D4 and D5 pass for the technical work; the separate P22 qualification remains open. P22 must later sequence all 40 layers with only one original-BF16 layer and its 256 expert slices resident at a time, retain required activations/KV/recurrent state, and use chunked embeddings/final norm/head for 248,320 raw FP32 logits, plus public Float16 values and sampler tokens. No P22 run is recorded here. At this dated Phase 3 handoff, Phase 5 Task 5.7 was next; that status is superseded by its completed scoped result. Phase 6 Task 6.7 was later proposed and is now complete for its scoped metadata-only registration result; the current proposed next task is Phase 6 Task 6.8, source trust receipts by integrity policy, not started and paused pending separate authorization. Phase 4 is retired.

<a id="phase-4"></a>
## Phase 4 - Historical quantization stays outside the selected route

Purpose: Version-1 quantizer work is retired and cannot prove BF16.
Needs: -. Status: retired historical phase.

Original tasks 4.1–4.5, quantizer code, tests and receipts remain in the version-1 archive. They prove only transformed/quantized behavior. There is no selected-route P4 task, test or BF16 acceptance. P6/P7/P21 do not depend on P4.


### Phase 4 acceptance

- [ ] Retired from selected route; not counted as a completed BF16 phase.

### Phase 4 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| No v2 code, retired P4 | none yet — retired | missing, not a BF16 test |

### Phase 4 evidence

none yet — v1 evidence is in archive; v2 P4 is retired.

<a id="phase-5"></a>
## Phase 5 - The local source validates offline without loading weights

Purpose: Exact headers, index and payload trust must be checked separately.
Needs: 1, 2. Status: accepted (Tasks 5.6 and 5.7 completed 2026-09-24 for metadata and verifier behavior); actual original-download authenticity remains unverified; further execution paused.

<a id="task-5-6"></a>
#### 5.6 Extract bounded Safetensors validation

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareOfficialQwenSource/SafetensorsSource.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/SafetensorsSource.swift); [Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift); [Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift); [Tests/TurboFieldfareOfficialQwenSource/OfficialSnapshotTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/OfficialSnapshotTests.swift); [Tests/TurboFieldfareRepack/Core/Format/SafetensorsHostileHeaderTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareRepack/Core/Format/SafetensorsHostileHeaderTests.swift); [Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift). `Package.swift` was inherited and unchanged for Task 5.6. |
| Depends on | P1, P2 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change completed 2026-09-24 under Hebert's exact Task 5.6-only authorization, “you are handling this, could you please send next prompt” (not Task 5.7 or global version-2 approval): reuse the existing shared source target, focused test target, Format dependency and runtime/RepackCore dependencies; **do not add targets or change `Package.swift`**. `OfficialSafetensorsSource` owns a public metadata-only `parseHeader(path:fileSize:headerBytes:)`, `readHeader(path:fileSize:readAt:)`, strict `parseIndex(_:)` and `validateShard(_:weightMap:shardName:)` API, with independent value descriptors and typed errors. The injectable read-at-offset seam requests only the 8-byte prefix and at most 16 MiB of header; index input is capped at 4 MiB. Duplicate and escaped-equivalent JSON keys, boolean/fractional/truncated dimensions or offsets, unknown dtype, checked shape/byte arithmetic, file bounds, overlapping ranges and leading/internal/trailing payload gaps are rejected. Zero-length tensors may occur only at the current contiguous boundary. The generic parser recognizes U32, BF16, F16 and F32; the RepackCore `Safetensors` adapter maps results/errors to existing `SourceTensor`/`RepackError`. The local loader delegates index and shard agreement while retaining BF16-only Qwen policy, bounded sidecar digests, safe leaf names, no-follow/nonblocking regular-file opens, missing/unreferenced-shard rejection and official identity/MTP checks. Header/index validation never materializes or authenticates weight payload bytes. The full-shard verifier was untouched by Task 5.6 and was extracted into the shared target in Task 5.7 below.
Why: Share the bounded checks used by RepackCore and future runtime source admission without changing `.gturbo` v1, Gemma defaults or source-runtime-unsupported behavior.

**Acceptance detail**
- [x] Reject malformed, overlapping, gapped and out-of-bounds tensors using only bounded metadata reads; preserve local and installer compatibility. This completes Task 5.6 only.

**How to check**

The first `Scripts/test.sh --filter OfficialSnapshotTests` attempt failed compilation (exit 1, **0 executed**) in a test-only nonexhaustive catch at `SafetensorsHostileHeaderTests.swift:155`; its raw logs remain preserved. After the test-owner correction, three positive selectors ran serially through `Scripts/test.sh`: `OfficialSnapshotTests` 13/13, `SafetensorsHostileHeaderTests` 12/12, and the explicit 32-name synthetic `LocalPinnedSnapshotLoaderTests/(...)` filter 32/32; **57/57 total**, zero failures/skips, all three exits 0. The positive filter excluded `canonicalOfficialSnapshotValidatesWithoutPayloadReads`; no original shard, model or reference process was run. Eight tested inputs (six changed paths, inherited `Package.swift`, `Scripts/test.sh`) matched before, after and current hashes. The first passing selector's receipt-wrapper regex briefly misread Swift Testing's `in 1 suite` wording, but its saved package exit is 0 and raw output reports 13/13; it was not rerun. Main reports independent code/test review **PASS** for this scoped Task 5.6 candidate; no separate reviewer artifact is linked here. These are metadata-only synthetic tests, not payload verification or a source runtime pass.

**Evidence**

2026-09-24: [Genuine Task 5.6 pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/baseline.md) records HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e`, inherited worktree paths, hashes of existing files and absence of the new shared/test paths before edits. [Attempt 1 compile failure and raw-log links](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/test-attempt-1-stopped.md) remain a failed attempt, not a pass. [Attempt 2 final serial receipt with three commands, raw stdout/stderr, exits and before/after hashes](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/attempt-2-final-receipt.md), [selected-name and hash verification](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/attempt-2-verification.json) and [current eight-input hashes](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/attempt-2-inputs.current.sha256) establish the final tested identity. No physical source-shard authentication is claimed.

<a id="task-5-7"></a>
#### 5.7 Share full pinned payload verification

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifier.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifier.swift), [Sources/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifier.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifier.swift), [Tests/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifierTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifierTests.swift) and [Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift) |
| Depends on | P1, P2 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change completed 2026-09-24 under Hebert's scoped authorization, “okay, move to next task please, send prompt” (not global version-2 approval): move production full-payload verification into the existing shared `TurboFieldfareOfficialQwenSource` target and retain the RepackCore adapter. The shared API verifies against the immutable official checksum pins; the adapter preserves RepackCore compatibility. Focused tests exercise real tiny-file hashing and byte-mutation rejection through the production hashing path, alongside synthetic callback coverage for the pinned inventory and adapter behavior. No `Package.swift` change or registration/runtime work was made.
Why: Exact headers, index and payload verification are separate; a successful tiny-fixture verifier test is not a claim that the downloaded checkpoint has been authenticated.

**Acceptance detail**
- [x] Altered payload fails full verifier on a real tiny file through the production hashing path.

**How to check**

Two serial package selectors passed: `Scripts/test.sh --filter OfficialPayloadVerifierTests` (15/15) and `Scripts/test.sh --filter LocalOfficialQwenPayloadVerifierTests` (11/11), total 26/26, zero failures/skips, both exit 0. The shared suite includes independent empty/`abc` SHA-256 vectors, bounded production hashing and actual tiny-file mutation rejection; the adapter suite covers synthetic inventory policy and compatibility. Final evidence records 14/14 tested inputs hash-stable. Earlier attempts are retained: runner preflight `CompletedProcess.pid` error before tests; a production compile failure (exit 1, zero tests); then the shared 15-test selector passed while the runner's summary parser stopped before adapter; the corrected final runner completed both selectors.

**Evidence**

2026-09-24: [Genuine Task 5.7 pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/baseline.md) records HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e`, inherited staged/unstaged/untracked state and affected-file hashes/absences. It is a Task 5.7 baseline, not a Phase 5 start-of-phase baseline. [Final authoritative serial receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/attempt.json), [shared selector stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/shared.stdout.log), [shared selector stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/shared.stderr.log), [adapter selector stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/adapter.stdout.log), [adapter selector stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/adapter.stderr.log), [14-input hashes before](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/inputs.before.json) and [after](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/inputs.current.json) record commands, exits and frozen candidate hashes. [Preflight-runner failure](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T123038Z-ac7d865f/attempt.json), [compile-failure receipt and raw stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T123117Z-8970df67/attempt.json), [intermediate shared-pass/parser-stop receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124337Z-1bd95f58/attempt.json) and [full raw compile stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T123117Z-8970df67/shared.stderr.log) preserve the failed attempts, not passes. Main reports independent code/test/raw-evidence review PASS for the final candidate and receipt. Only generated tiny files, inline manifest data and synthetic callbacks were used: no original checkpoint shard was read or hashed, so physical authenticity of the 71,903,776,776-byte download remains unverified.


### Phase 5 acceptance

- [x] Exact headers, index and payload verification are checked separately; Task 5.6 covers bounded metadata, while Task 5.7 covers the pinned full-payload verifier path.
- [x] Reject malformed and out-of-bounds tensors (Task 5.6 metadata validation only).
- [x] Altered payload fails the full verifier on an actual tiny file through production hashing. The real original download has not been read or authenticated.

### Phase 5 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| [Sources/TurboFieldfareOfficialQwenSource/SafetensorsSource.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/SafetensorsSource.swift) | `OfficialSnapshotTests`; [attempt-2 receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/attempt-2-final-receipt.md) | 13/13 pass; bounded read seam and strict metadata/layout; 2026-09-24 |
| [Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift) | `SafetensorsHostileHeaderTests` | 12/12 pass; legacy error and generic-dtype adapter; 2026-09-24 |
| [Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift) | synthetic positive `LocalPinnedSnapshotLoaderTests/(...)` only | 32/32 pass; no canonical checkpoint selection; 2026-09-24 |
| [Tests/TurboFieldfareOfficialQwenSource/OfficialSnapshotTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/OfficialSnapshotTests.swift) | `OfficialSnapshotTests` | 13/13 pass; 2026-09-24 |
| [Tests/TurboFieldfareRepack/Core/Format/SafetensorsHostileHeaderTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareRepack/Core/Format/SafetensorsHostileHeaderTests.swift) | `SafetensorsHostileHeaderTests`; failed attempt preserved | 12/12 pass after test-only compile correction; 2026-09-24 |
| [Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift) | synthetic positive `LocalPinnedSnapshotLoaderTests/(...)` only | 32/32 pass; 2026-09-24 |
| [Sources/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifier.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifier.swift) | `LocalOfficialQwenPayloadVerifierTests` | 11/11 pass; compatibility adapter, pinned inventory policy and error/progress propagation; 2026-09-24 |
| [Sources/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifier.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifier.swift) | `OfficialPayloadVerifierTests`; [final serial receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/attempt.json) | 15/15 pass; production bounded hashing, known SHA-256 vectors, actual tiny-file byte-mutation rejection and mutation/cancellation/file-type checks; 2026-09-24 |
| [Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift) | `LocalOfficialQwenPayloadVerifierTests` | 11/11 pass; synthetic callback inventory and adapter compatibility; 2026-09-24 |
| [Tests/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifierTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/OfficialPayloadVerifierTests.swift) | `OfficialPayloadVerifierTests` | 15/15 pass; actual tiny-file production hash path and independent vectors; 2026-09-24 |

### Phase 5 evidence

Task baselines: [Genuine Task 5.6 pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/baseline.md) and [genuine Task 5.7 pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/baseline.md). The Task 5.7 baseline records HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e` and inherited worktree state before its edits; it is not a Phase 5 start baseline. D4 reconciles the six Task 5.6 paths against the Task 5.6 baseline and the four Task 5.7 source/test paths against the Task 5.7 baseline; inherited `Package.swift` is not attributed to either task. The task-owned evidence runner [run-focused-tests.py](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/run-focused-tests.py), SHA-256 `155b19325f5d6fbde67659b38eb33eddac6cedddf05f1ec939ed77416724c544`, is listed separately from the four source/test paths and ten D3 coverage rows; its hash is included in the final 14-input snapshots.

Task 5.6: [baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/baseline.md), [failed attempt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/test-attempt-1-stopped.md), [57/57 final serial receipt and raw links](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.6-omnigent/attempt-2-final-receipt.md). Task 5.7: [genuine pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/baseline.md), [authoritative final serial receipt, commands, outcomes and approved hashes](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/attempt.json), [shared stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/shared.stdout.log) / [shared stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/shared.stderr.log) / [adapter stdout](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/adapter.stdout.log) / [adapter stderr](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/adapter.stderr.log), [before/after tested-input hash records](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/inputs.before.json) / [current](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124634Z-de450184/inputs.current.json), [compile failure (exit 1, zero tests)](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T123117Z-8970df67/attempt.json), [intermediate shared pass then runner-parser stop](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T124337Z-1bd95f58/attempt.json), and [initial runner preflight failure](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-5.7-omnigent/focused-attempt-20260924T123038Z-ac7d865f/attempt.json). Main reports independent code/test/raw-evidence review PASS for the final candidate and receipt. These receipts prove implementation behavior on generated fixtures only; the original checkpoint shards were not read or hashed.

Phase 5 gates: D1 pass (Tasks 5.6 and 5.7 complete); D2 pass (all three acceptance checks met); D3 pass (all ten mapped coverage rows pass); D4 pass (all six Task 5.6 and four Task 5.7 source/test paths mapped against their respective genuine task baselines; the Task 5.7 evidence runner is separately inventoried and hash-snapshotted, not counted as a source/test coverage row; no Phase 5 start baseline is claimed); D5 pass (task baselines, failed attempts, final raw commands/results and tested-input hashes linked here). Phase 5 is accepted for the offline metadata and pinned-verifier implementation; actual original-download authenticity, registration and runtime remain unverified/pending.

<a id="phase-6"></a>
## Phase 6 - Existing BF16 shards register without a copied weight pack

Purpose: Persist source identity at the logical model location without copying weights.
Needs: 2, 5. Status: technically complete 2026-09-30; source trust qualification caveat retained; execution paused.

<a id="task-6-7"></a>
#### 6.7 Write metadata-only source registration

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift), [Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift), [Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift), [Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift) |
| Depends on | Accepted P2 and P5 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change completed 2026-09-24 under [Hebert's scoped Task 6.7 assignment](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/coordination/task-6.7-supervision/assignment.txt), with exact authorization “Okay, please send the prompt”, to record existing source files without copying them. The shared `OfficialSourceRegistration.register(markerData:at:)` strictly decodes the bounded Phase 2 marker, validates the Phase 1 pinned inventory, requires an existing no-follow source directory without opening its children, and publishes only `official-source.json` at a new logical `.gturbo` location. Its `inspect(at:)` reads and revalidates that single marker. The descriptor persists `sourceRoot` separately from its canonical content SHA-256. No `Package.swift` change was made for Task 6.7; its earlier modifications were inherited. App location preserves Gemma/packed resolution, canonicalizes the checkout parent with POSIX `realpath`, leaves the final Qwen leaf unresolved for symlink rejection, and supports the fresh app-owned Application Support parent.

Conflict and failure handling reject an existing destination, source/destination ancestry (including Apple case/Unicode equivalents), nonregular/symlink paths and packed partial/resume state. Registration holds the same sibling `.install.lock` and nonblocking exclusive `flock` as the packed installer; the bounded lock sidecar may remain after an attempt. It syncs a task-owned single-marker stage, then atomically renames without replacement; prepublication failure/cancellation cleans only owned staging. A parent-sync failure after rename is explicitly `publishedDurabilityUnknown`, not an unpublished failure. None of this attests to physical source payloads. No trusted full-verification receipt, ready runtime, app picker, or live source installation is produced; Task 6.8 and later product wiring remain pending.

**Acceptance detail**
- [x] Metadata-only registration persists pinned source identity at a distinct logical path (2026-09-24).
- [x] No weight bytes are copied or read by the registration path (2026-09-24); synthetic inventory and inaccessible-shard sentinel tests exercise this boundary, not the original 26-shard download.

**How to check**

The final serial selectors were `Scripts/test.sh --filter SourceRegistrationTests` and `Scripts/test.sh --filter AppModelLocationTests`. The [attempt-4 raw receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/final-receipt.md) links exact commands, UTC start/end, full stdout/stderr, executed names, exits and hash maps: **17/17 + 7/7 = 24/24**, each selector exit 0, zero failures/skips, and all 13 tested inputs stable before, after each selector and currently. These test results are dated 2026-09-24. No tests were rerun during the 2026-09-27 documentation continuation. Earlier [attempt 1](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-1-stopped.md), [attempt 2](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-2/stopped.md) and [attempt 3](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-3/stopped.md) retain the `/var` synthetic-root and app-location alias failures; they are not final-candidate passes. No model, original shard or broad workflow selector ran.

**Evidence**

[Task 6.7 genuine pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/baseline.md) was captured at 2026-09-24 14:09:35 UTC before the first Task 6.7 edit at HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e`; it records inherited dirty paths, absence of both new registration source/test files, input hashes and machine inventory. It is **not** a Phase 6 start baseline. [Attempt-4 frozen input map](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/inputs.before.json), [final current hashes](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/inputs.current.json) and the [final focused receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/final-receipt.md) identify the tested candidate. Independent code/test/evidence review PASS was accepted before this documentation continuation. This Markdown reconciliation completed 2026-09-27 without rerunning tests; it does not accept Phase 6 as a whole or global version 2.

<a id="task-6-8"></a>
#### 6.8 Define source trust receipt by integrity policy

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Infrastructure/ModelIO/VerifiedInstallReceipt.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/ModelIO/VerifiedInstallReceipt.swift) and PROPOSED Sources/TurboFieldfareOfficialQwenSource/SourceTrustReceipt.swift |
| Depends on | P2, P5 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Full SHA checks all 26 whole shard files, headers and payloads, on load; trusted reopen checks prior verified receipt and current fingerprints; picker probes are cheap.
Why: Persist source identity at the logical model location.

**Acceptance detail**
- [x] Owner-confirmed technical status records full/trusted/stale receipt semantics as complete; no new receipt or test result is claimed by this update.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 6 acceptance

- [x] Persist pinned source identity at the logical model location in a tiny synthetic registration (Task 6.7, 2026-09-24); no actual source registration or runtime readiness claimed.
- [x] No weight bytes are copied or opened by Task 6.7 registration; original-source payload authenticity remains unverified.
- [x] Owner-confirmed technical status records the full/trusted/stale receipt policy as complete; no new receipt or test result is claimed by this update.

### Phase 6 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| [Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift) | [SourceRegistrationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift) / [raw final receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/final-receipt.md) | pass 2026-09-24 — 17/17, exit 0; metadata-only publication, lock, conflict/cancellation and no-payload sentinel |
| [Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift) | [AppModelLocationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift) / [raw final receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/final-receipt.md) | pass 2026-09-24 — 7/7, exit 0; canonical parent/unresolved leaf, Gemma and Application Support compatibility |
| [Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift) | self-test / [actual executed names](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/source-registration.result.json) | pass 2026-09-24 — 17/17, exit 0, zero failures/skips |
| [Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift](../../../../../turbo-fieldfare-personal/Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift) | self-test / [actual executed names](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/app-model-location.result.json) | pass 2026-09-24 — 7/7, exit 0, zero failures/skips |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/VerifiedInstallReceipt.swift` | Task 6.8 trust-receipt tests, not Task 6.7 | pending — unchanged for Task 6.7; no trusted receipt |
| `Sources/TurboFieldfareOfficialQwenSource/SourceTrustReceipt.swift` | Task 6.8 trust-receipt tests | pending — not created |

### Phase 6 evidence

Baseline evidence: [genuine Task 6.7 pre-edit baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/baseline.md), captured before the first Task 6.7 edit at HEAD `d6ecc3fc81db5c23b58816cfd32d64b7b371f61e` with inherited state and the two proposed new paths absent; no Phase 6 start baseline is claimed.
Task 6.7 evidence: the [genuine task baseline](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/baseline.md), [attempt-4 final receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/final-receipt.md), and [current 13-input hash map](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/task-6.7-omnigent/focused-attempt-4/inputs.current.json) identify the candidate. Its four Task 6.7 source/test paths are `Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift`, `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift`, `Tests/TurboFieldfareOfficialQwenSource/SourceRegistrationTests.swift` and `Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift`; the table above maps each path. Tests remain dated 2026-09-24 (17/17 and 7/7; 24/24 total, both exits 0, zero failures/skips). Documentation reconciliation completed 2026-09-27; no tests were rerun. Failed earlier attempts remain linked above. Task 6.8 is marked technically complete by owner status; no new receipt or test result is claimed here, and original-source payload authenticity remains unverified.

<a id="phase-7"></a>
## Phase 7 - Verified original-source ranges are read safely

Purpose: Read bounded ranges through retained protected file descriptors.
Needs: 5, 6. Status: technically complete 2026-09-30, execution paused.

<a id="task-7-7"></a>
#### 7.7 Open source files through a protected root

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Infrastructure/ModelIO/GTurboModelDirectory.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/ModelIO/GTurboModelDirectory.swift) and PROPOSED Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift |
| Depends on | P5, P6 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use openat and O_NOFOLLOW on the registered source root and retain descriptors.
Why: Read bounded ranges through retained protected file descriptors.

**Acceptance detail**
- [x] Reject symlinks, escape and replacement.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-7-8"></a>
#### 7.8 Read bounded BF16 tensor slices

| Field | Detail |
|---|---|
| Touches | PROPOSED Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift preadTensorRange |
| Depends on | P5, P6 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Check overflow, off_t bounds, exact expected bytes, short reads and EINTR; compare retained-fd device/inode/size/mtime/ctime before and after reads, and release partial buffers on failure.
Why: Read bounded ranges through retained protected file descriptors.

**Acceptance detail**
- [x] Unaligned exact slice passes; mutation fails.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 7 acceptance

- [x] Read bounded ranges through retained protected file descriptors.
- [x] Reject symlinks, escape and replacement.
- [x] Unaligned exact slice passes; mutation fails.

- [x] Unaligned reads handle short reads/EINTR, checked offsets and before/after fingerprints; partial failure publishes no model.

### Phase 7 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Infrastructure/ModelIO/GTurboModelDirectory.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareOfficialQwenSource/OfficialSourceHandleTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareOfficialQwenSource/OfficialSourceHandleTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfareOfficialQwenSource/OfficialSourceHandleTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 7 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-8"></a>
## Phase 8 - BF16 text matrices have a checked Metal contract

Purpose: Keep BF16 stored values and FP32 calculations distinct.
Needs: 2, 3. Status: technically complete 2026-09-30, execution paused.

<a id="task-8-5"></a>
#### 8.5 Bind resident BF16 tensors in shared buffers

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift) and PROPOSED Sources/TurboFieldfare/Runtime/Qwen/QwenBF16Weights.swift |
| Depends on | P2, P3 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Direct pread into per-tensor or complete-row-chunk Metal shared buffers, bounded by device.maxBufferLength and checked tensor offsets, without Data staging or Float32 expansion; free partial allocations on failure.
Why: Keep BF16 stored values and FP32 calculations distinct.

**Acceptance detail**
- [x] Exact source bits and over-limit geometry tests.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-8-6"></a>
#### 8.6 Add FP32-accumulating BF16 kernels

| Field | Detail |
|---|---|
| Touches | PROPOSED Sources/TurboFieldfare/Metal/Qwen/qwen_bf16.metal and [Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift) |
| Depends on | P2, P3 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Cover embedding, head, router, dense and shared projections while retaining packed affine kernels.
Why: Keep BF16 stored values and FP32 calculations distinct.

**Acceptance detail**
- [x] Independent small-matrix error tolerance.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 8 acceptance

- [x] Keep BF16 stored values and FP32 calculations distinct.
- [x] Exact source bits and over-limit geometry tests.
- [x] Independent small-matrix error tolerance.

- [x] Every resident allocation obeys device.maxBufferLength and oversized matrices split on complete rows without persistent Float32 copies.

### Phase 8 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenBF16Weights.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_bf16.metal` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16WeightsTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 8 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-9"></a>
## Phase 9 - Full attention reads BF16 source projections

Purpose: Reuse attention math with BF16 source matrix bindings.
Needs: 8. Status: technically complete 2026-09-30, execution paused.

<a id="task-9-6"></a>
#### 9.6 Bind BF16 full-attention projections

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift) and [Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift) |
| Depends on | P8 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use BF16 Q/K/V/output matrices with existing RoPE, norms and gating.
Why: Reuse attention math with BF16 source matrix bindings.

**Acceptance detail**
- [x] Tiny independent full-layer comparison.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-9-7"></a>
#### 9.7 Guard full-attention geometry and rollback

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift) and [Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift) |
| Depends on | P8 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Validate row chunks, head geometry and KV commit boundaries.
Why: Reuse attention math with BF16 source matrix bindings.

**Acceptance detail**
- [x] Invalid shape and cancellation leave committed state.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 9 acceptance

- [x] Reuse attention math with BF16 source matrix bindings.
- [x] Tiny independent full-layer comparison.
- [x] Invalid shape and cancellation leave committed state.

### Phase 9 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16FullAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16FullAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16FullAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16FullAttentionTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 9 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-10"></a>
## Phase 10 - Linear attention reads BF16 source projections

Purpose: Keep FP32 recurrence while consuming BF16 matrix storage.
Needs: 8. Status: technically complete 2026-09-30, execution paused.

<a id="task-10-6"></a>
#### 10.6 Bind BF16 linear-attention projections

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift) and [Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift) |
| Depends on | P8 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use BF16 QKV, z, b, a and output matrices while retaining small FP32 vectors.
Why: Keep FP32 recurrence while consuming BF16 matrix storage.

**Acceptance detail**
- [x] Tiny independent linear-layer comparison.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-10-7"></a>
#### 10.7 Retain FP32 recurrence rollback

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift) and [Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift) |
| Depends on | P8 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Keep recurrent and convolution state FP32 and commit atomically.
Why: Keep FP32 recurrence while consuming BF16 matrix storage.

**Acceptance detail**
- [x] Injected failure restores prior state.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 10 acceptance

- [x] Keep FP32 recurrence while consuming BF16 matrix storage.
- [x] Tiny independent linear-layer comparison.
- [x] Injected failure restores prior state.

### Phase 10 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16LinearAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16LinearAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16LinearAttentionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16LinearAttentionTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 10 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-11"></a>
## Phase 11 - Selected BF16 experts share one atomic cache state

Purpose: Gate/up and down source slices form one valid cache entry.
Needs: 7, 8. Status: technically complete 2026-09-30, execution paused.

<a id="task-11-6"></a>
#### 11.6 Read paired selected-expert slices

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift) and [Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift) |
| Depends on | P7, P8 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use two streams per layer for BF16 gate/up and down, possibly in different shards.
Why: Gate/up and down source slices form one valid cache entry.

**Acceptance detail**
- [x] Layer-zero cross-shard selected expert matches bytes.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-11-7"></a>
#### 11.7 Publish one hit only after both reads

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift) and [Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift) performFetch |
| Depends on | P7, P8 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use one cache-validity table, invalidate any victim before writes, never executeExpertCachePlan separately by role, wait for both launched reads after error/cancellation, and publish only after both succeed. On second-read failure invalidate the pair and release unsubmitted reservation.
Why: Gate/up and down source slices form one valid cache entry.

**Acceptance detail**
- [x] Gate/up-success/down-failure and down-success/gate-up-failure both invalidate the pair; previous-victim retry and changed-source cached hit force complete reread, never a half-filled hit.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-11-8"></a>
#### 11.8 Hold paired slots through GPU completion

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift) QwenMappedExpertLease and [Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift) |
| Depends on | P7, P8 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Pin both buffers until GPU command completion and resolve cancellation after submission without reusing slots; cancellation during concurrent reads waits for both reads to settle.
Why: Gate/up and down source slices form one valid cache entry.

**Acceptance detail**
- [x] Cancellation on either read waits for both reads to stop before release; concurrent submissions cannot evict a live pair.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-11-9"></a>
#### 11.9 Run routed BF16 MoE kernels

| Field | Detail |
|---|---|
| Touches | Sources/TurboFieldfare/Metal/Qwen/qwen_moe.metal and [Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift) |
| Depends on | P7, P8 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Read BF16 gate/up and down with FP32 accumulation and preserve Top-8 routing.
Why: Gate/up and down source slices form one valid cache entry.

**Acceptance detail**
- [x] Independent small MoE output matches.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 11 acceptance

- [x] Gate/up and down source slices form one valid cache entry.
- [x] Layer-zero cross-shard selected expert matches bytes.
- [x] Both asymmetric read failures, cancellation during either read, source mutation on a cached hit and retry of the evicted expert produce a real reread without any half-filled hit.
- [x] Concurrent submissions cannot evict live pair.
- [x] Independent small MoE output matches.

- [x] Victims invalidate before writes; both reads settle before cancellation release; failure tests cover either failed slice, prior-victim retry, stale cached hit and GPU lease overlap.

### Phase 11 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_moe.metal` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16ExpertCacheTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 11 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-12"></a>
## Phase 12 - The source-backed text runner emits a token

Purpose: Use resident BF16 buffers and the production layer order.
Needs: 7, 9, 10, 11. Status: technically complete 2026-09-30, execution paused.

<a id="task-12-7"></a>
#### 12.7 Load resident BF16 text buffers once

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift) and [Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift) init |
| Depends on | P7, P9, P10, P11 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Replace source-path eager decodedFloat32 matrices with retained BF16 Metal buffers.
Why: Use resident BF16 buffers and the production layer order.

**Acceptance detail**
- [x] No persistent full Float32 matrix and exact byte accounting.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-12-8"></a>
#### 12.8 Execute a source-backed production token

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift) forward and [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift) loadBundle and [Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift) QwenArchConfig / LoadedTensorRegion |
| Depends on | P7, P9, P10, P11 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Connect BF16 attention and paired MoE in the existing layer order. Construct QwenArchConfig and tensor regions from verified official config/index metadata using proposed source-specific initializers in ModelTypes.swift. Do not fabricate a GTurboQwenArchitectureV2 or packed tensor record to reuse its initializer. Reject inconsistent geometry, shapes and ranges before constructing the production runner.
Why: Use resident BF16 buffers and the production layer order.

**Acceptance detail**
- [x] Tiny source-backed fixture emits the expected token, while conflicting config geometry or tensor ranges fail before execution.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-12-9"></a>
#### 12.9 Compare FP32 full logits before public Float16

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift) and [Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift) |
| Depends on | P7, P9, P10, P11 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Keep full-vocabulary FP32 logits for oracle comparison and established Float16 public boundary.
Why: Use resident BF16 buffers and the production layer order.

**Acceptance detail**
- [x] Separate full-logit and public-boundary evidence.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 12 acceptance

- [x] Use resident BF16 buffers and the production layer order.
- [x] No persistent full Float32 matrix and exact byte accounting.
- [x] Tiny source-backed fixture emits the expected token, while conflicting config geometry or tensor ranges fail before execution.
- [x] Separate full-logit and public-boundary evidence.

### Phase 12 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TextRunnerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 12 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-13"></a>
## Phase 13 - Source-backed turns commit or roll back atomically

Purpose: A failed read or GPU step must not advance partial state.
Needs: 12. Status: technically complete 2026-09-30, execution paused.

<a id="task-13-7"></a>
#### 13.7 Bind source identity to transaction state

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift) and [Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift) |
| Depends on | P12 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Commit token, KV, convolution, recurrence and positions as one source-bound operation.
Why: A failed read or GPU step must not advance partial state.

**Acceptance detail**
- [x] Injected failure leaves prior turn intact.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-13-8"></a>
#### 13.8 Invalidate turns after source mutation

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift) and PROPOSED Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift |
| Depends on | P12 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Check retained source descriptors at turn boundaries and expert reads.
Why: A failed read or GPU step must not advance partial state.

**Acceptance detail**
- [x] No partial accepted answer after change.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 13 acceptance

- [x] A failed read or GPU step must not advance partial state.
- [x] Injected failure leaves prior turn intact.
- [x] No partial accepted answer after change.

### Phase 13 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TransactionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TransactionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TransactionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenBF16TransactionTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 13 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-14"></a>
## Phase 14 - Official chat text uses the verified source tokenizer

Purpose: Reuse official chat grammar through verified sidecar admission.
Needs: 1, 5. Status: technically complete 2026-09-30, execution paused.

<a id="task-14-5"></a>
#### 14.5 Load verified official tokenizer sidecars

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Tokenization/QwenTokenizer.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Tokenization/QwenTokenizer.swift) and PROPOSED Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift |
| Depends on | P1, P5 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Promote decoder behind verified source admission, not fixture-only bypass.
Why: Reuse official chat grammar through verified sidecar admission.

**Acceptance detail**
- [x] Pinned token IDs match; changed sidecar rejects.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-14-6"></a>
#### 14.6 Requalify source-backed chat template

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift) |
| Depends on | P1, P5 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Reuse chat grammar with BF16 source identity in prompt and result.
Why: Reuse official chat grammar through verified sidecar admission.

**Acceptance detail**
- [x] Official pinned prompt bytes match.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 14 acceptance

- [x] Reuse official chat grammar through verified sidecar admission.
- [x] Pinned token IDs match; changed sidecar rejects.
- [x] Official pinned prompt bytes match.

### Phase 14 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Tokenization/QwenTokenizer.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Tokenization/QwenOfficialSourceTokenizerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareOfficialQwenSource/OfficialSourceHandle.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Tokenization/QwenOfficialSourceTokenizerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Tokenization/QwenOfficialSourceTokenizerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Tokenization/QwenOfficialSourceTokenizerTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 14 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-15"></a>
## Phase 15 - Incomplete source-backed tool output dispatches nothing

Purpose: Reuse parser safety and requalify the BF16 identity path.
Needs: 14. Status: technically complete 2026-09-30, execution paused.

<a id="task-15-6"></a>
#### 15.6 Bind tool parsing to BF16 conversation

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift) |
| Depends on | P14 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Carry source identity through complete tool output.
Why: Reuse parser safety and requalify the BF16 identity path.

**Acceptance detail**
- [x] Incomplete tool output dispatches nothing.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-15-7"></a>
#### 15.7 Roll back failed tool turns

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift) |
| Depends on | P14 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Reject malformed/interrupted tool output without state commit.
Why: Reuse parser safety and requalify the BF16 identity path.

**Acceptance detail**
- [x] No partial tool call or turn state.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 15 acceptance

- [x] Reuse parser safety and requalify the BF16 identity path.
- [x] Incomplete tool output dispatches nothing.
- [x] No partial tool call or turn state.

### Phase 15 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/QwenOfficialSourceToolTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenOfficialSourceToolTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 15 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-16"></a>
## Phase 16 - Still images use source-backed vision groups

Purpose: Keep adjacent vision contract without a second weight payload.
Needs: 7, 8, 12, 13. Status: technically complete 2026-09-30, execution paused.

<a id="task-16-9"></a>
#### 16.9 Define adjacent source vision companion

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift) and PROPOSED Sources/TurboFieldfareFormat/OfficialSourceVisionDescriptor.swift |
| Depends on | P7, P8, P12, P13 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Metadata-only .vision.gturbo binds text digest, processor and required vision tensors.
Why: Keep adjacent vision contract without a second weight payload.

**Acceptance detail**
- [x] Missing or mismatched companion is unavailable.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-16-10"></a>
#### 16.10 Gather only requested vision group

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift) mapGroup and [Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift) execute |
| Depends on | P7, P8, P12, P13 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Read one patch, block or merger group into bounded shared buffer and hold GPU lease.
Why: Keep adjacent vision contract without a second weight payload.

**Acceptance detail**
- [x] No duplicate vision payload; group bytes match.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-16-11"></a>
#### 16.11 Requalify source image token rows

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift) and [Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift) |
| Depends on | P7, P8, P12, P13 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Reuse preprocessing and M-RoPE with BF16 group reader.
Why: Keep adjacent vision contract without a second weight payload.

**Acceptance detail**
- [x] Tiny image rows and positions match reference.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 16 acceptance

- [x] Keep adjacent vision contract without a second weight payload.
- [x] Missing or mismatched companion is unavailable.
- [x] No duplicate vision payload; group bytes match.
- [x] Tiny image rows and positions match reference.

### Phase 16 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareFormat/OfficialSourceVisionDescriptor.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenOfficialSourceVisionTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 16 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-17"></a>
## Phase 17 - Decode service binds results to BF16 source identity

Purpose: Source and old quantized models must never share accepted identity.
Needs: 2, 12, 13, 15. Status: technically complete 2026-09-30, execution paused.

<a id="task-17-7"></a>
#### 17.7 Encode BF16 backing in decode identity

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift) and [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift) |
| Depends on | P2, P12, P13, P15 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Carry source kind/content digest instead of fake v2 quantization fields.
Why: Source and old quantized models must never share accepted identity.

**Acceptance detail**
- [x] Reject old quantized identity with same index.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-17-8"></a>
#### 17.8 Reopen source registration in service

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareDecodeService/](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareDecodeService) and [Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift) |
| Depends on | P2, P12, P13, P15 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Load logical .gturbo registration and return source identity/epoch.
Why: Source and old quantized models must never share accepted identity.

**Acceptance detail**
- [x] Restart reopens; removed source unavailable.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 17 acceptance

- [x] Source and old quantized models must never share accepted identity.
- [x] Reject old quantized identity with same index.
- [x] Restart reopens; removed source unavailable.

### Phase 17 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareDecodeService/` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareDecodeService` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfareDecodeService/OfficialSourceIdentityTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 17 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-18"></a>
## Phase 18 - CLI runs verified original-precision requests

Purpose: CLI needs explicit source backing and image availability.
Needs: 12, 14, 15, 16. Status: technically complete 2026-09-30, execution paused.

<a id="task-18-6"></a>
#### 18.6 Accept source registration in CLI

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareCLI/](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareCLI) and [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift) |
| Depends on | P12, P14, P15, P16 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use source-aware family load and integrity choice.
Why: CLI needs explicit source backing and image availability.

**Acceptance detail**
- [x] Tiny text request reports BF16 identity.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-18-7"></a>
#### 18.7 Report CLI image and hash availability

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareCLI/](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareCLI) and [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift) |
| Depends on | P12, P14, P15, P16 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Distinguish full hash, trusted receipt and missing vision companion.
Why: CLI needs explicit source backing and image availability.

**Acceptance detail**
- [x] Image rejection and verification label are truthful.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 18 acceptance

- [x] CLI needs explicit source backing and image availability.
- [x] Tiny text request reports BF16 identity.
- [x] Image rejection and verification label are truthful.

### Phase 18 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfareCLI/` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareCLI` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfare/Core/CLI/OfficialSourceCLITests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 18 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-19"></a>
## Phase 19 - Loopback server serves verified BF16 chat

Purpose: Keep loopback binding and source-aware failure responses.
Needs: 12, 14, 15, 16. Status: technically complete 2026-09-30, execution paused.

<a id="task-19-6"></a>
#### 19.6 Route BF16 source through loopback chat

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareServer/Core/](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareServer/Core) and [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift) |
| Depends on | P12, P14, P15, P16 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Keep 127.0.0.1 binding and one loaded model.
Why: Keep loopback binding and source-aware failure responses.

**Acceptance detail**
- [x] Local source chat works; no remote binding.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-19-7"></a>
#### 19.7 Reject stale server source and vision

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareServer/Core/](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareServer/Core) |
| Depends on | P12, P14, P15, P16 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Fail visibly for changed source or invalid companion.
Why: Keep loopback binding and source-aware failure responses.

**Acceptance detail**
- [x] No silent image omission or stale identity.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 19 acceptance

- [x] Keep loopback binding and source-aware failure responses.
- [x] Local source chat works; no remote binding.
- [x] No silent image omission or stale identity.

### Phase 19 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfareServer/Core/` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareServer/OfficialSourceServerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareServer/Core` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareServer/OfficialSourceServerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareServer/OfficialSourceServerTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfareServer/OfficialSourceServerTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 19 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-20"></a>
## Phase 20 - Mac app remembers the existing BF16 source and preserves Gemma

Purpose: Replace Qwen conversion presentation with persistent registration.
Needs: 6, 16, 17. Status: technically complete 2026-09-30, execution paused.

<a id="task-20-10"></a>
#### 20.10 Replace app Qwen conversion route

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareApp/Core/Installation/AppModelCatalog.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Installation/AppModelCatalog.swift) and [Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift) |
| Depends on | P6, P16, P17 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Require BF16 source profile for Qwen readiness; retain Gemma behavior.
Why: Replace Qwen conversion presentation with persistent registration.

**Acceptance detail**
- [x] Old quantized pack does not satisfy Qwen.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-20-11"></a>
#### 20.11 Persist app Qwen source binding

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift) and [Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift) |
| Depends on | P6, P16, P17 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Store source root/registration link across relaunch and retain .vision.gturbo adjacency.
Why: Replace Qwen conversion presentation with persistent registration.

**Acceptance detail**
- [x] Relaunch reopens; moved source unavailable.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-20-12"></a>
#### 20.12 Pass configured app expert slots to runner

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift) and [Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift) and [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift) load/loadBundle |
| Depends on | P6, P16, P17 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Pass AppRuntimeOptions.expertCacheSlots through RealInferenceClient and QwenConversationState to makeRunner, including a non-default value. The current retained-state runner falls back to 8 despite the nominal 16-slot setting. In ModelFamilyRuntime.load/loadBundle, propagate streamingMode, expertCachePolicy and integrityPolicy into the source-backed Qwen factory. The current Qwen branch omits those options. Retain existing CLI and Gemma semantics.
Why: The selected settings must control actual cache allocation and verification behavior.

**Acceptance detail**
- [x] Selected non-default slots and cache/integrity policy reach the source-backed runner and its diagnostics.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.

<a id="task-20-13"></a>
#### 20.13 Show registration and honest Qwen memory labels

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift) and [Sources/TurboFieldfareApp/Core/Configuration/AppRuntimeOptions.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Configuration/AppRuntimeOptions.swift) and [Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift) conversion labels/buttons and [Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift) source picker/status |
| Depends on | P6, P16, P17 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Replace Qwen conversion/discard text in ModelInstallView with source verification/registration wording and show its state in InspectorView. Cancelling or discarding registration may remove only task-owned metadata, never original source shards or Gemma. Replace Gemma-specific KV estimates with the Qwen FP32-state calculation or explicitly unavailable estimates, and label total memory fit unmeasured.
Why: Installation controls must describe the operation they perform and memory labels must describe this model.

**Acceptance detail**
- [x] No false Qwen estimate or conversion action.

**How to check**

Owner-confirmed technical completion; task-level evidence mapping is deferred in this update and no new test run is claimed.

**Evidence**

existing phase receipt is authoritative; no new run was performed for this update.


### Phase 20 acceptance

- [x] Replace Qwen conversion presentation with persistent registration.
- [x] Old quantized pack does not satisfy Qwen.
- [x] Relaunch reopens; moved source unavailable.
- [x] Selected non-default slots and cache/integrity policy reach the source-backed runner and its diagnostics.
- [x] No false Qwen estimate or conversion action.

- [x] Descriptor-only registration is not complete-ready; missing or changed source disables readiness; an old quantized artifact with the same official index never satisfies BF16 Qwen.

### Phase 20 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `Sources/TurboFieldfareApp/Core/Installation/AppModelCatalog.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Core/Configuration/AppContextLengthOption.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Core/Configuration/AppRuntimeOptions.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift` | coverage mapping deferred — existing phase receipt remains authoritative self-test | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |
| `Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift` | coverage mapping deferred — existing phase receipt remains authoritative Tests/TurboFieldfareApp/Core/OfficialSourceInstallationTests.swift | deferred mapping — owner-confirmed technical completion; no new test count claimed in this update |

### Phase 20 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-21"></a>
## Phase 21 - The already-downloaded official source is verified and registered

Purpose: Use exact existing files with no conversion or copied payload.
Needs: 5, 6, 7, 16, 17, 20. Status: technically complete 2026-09-28, execution paused.

<a id="task-21-7"></a>
#### 21.7 Preflight exact existing source and machine

| Field | Detail |
|---|---|
| Touches | AGENTS.md and [scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0) |
| Depends on | P5, P6, P7, P16, P17, P20 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Check source, OS/Swift, disk, memory pressure and no competing model process.
Why: Use exact existing files with no conversion or copied payload.

**Acceptance detail**
- [x] Complete preflight receipt, no source change.

**How to check**

See the retained [Phase 21 final receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json), which records the executed verification, registration, app audit and old-packed rejection selectors.

**Evidence**

See the retained Phase 21 final receipt linked above; no new run was performed for this documentation update.

<a id="task-21-8"></a>
#### 21.8 Full-verify and register without conversion

| Field | Detail |
|---|---|
| Touches | PROPOSED Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift |
| Depends on | P5, P6, P7, P16, P17, P20 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Hash all official shards for trust and write text/vision metadata and receipt only.
Why: Use exact existing files with no conversion or copied payload.

**Acceptance detail**
- [x] 26 unchanged shards, no second weight payload.

**How to check**

See the retained [Phase 21 final receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json), which records the executed verification, registration, app audit and old-packed rejection selectors.

**Evidence**

See the retained Phase 21 final receipt linked above; no new run was performed for this documentation update.

<a id="task-21-9"></a>
#### 21.9 Audit source identity and trusted reopen

| Field | Detail |
|---|---|
| Touches | [Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift) and [Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift](../../../../../turbo-fieldfare-personal/Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift) |
| Depends on | P5, P6, P7, P16, P17, P20 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Check descriptor, receipt, fingerprints and profile across reopen.
Why: Use exact existing files with no conversion or copied payload.

**Acceptance detail**
- [x] Old quantized artifact rejected.

**How to check**

See the retained [Phase 21 final receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json), which records the executed verification, registration, app audit and old-packed rejection selectors.

**Evidence**

See the retained Phase 21 final receipt linked above; no new run was performed for this documentation update.


### Phase 21 acceptance

- [x] Use exact existing files with no conversion or copied payload.
- [x] Complete preflight receipt, no source change.
- [x] 26 unchanged shards, no second weight payload.
- [x] Old quantized artifact rejected.

- [x] Descriptor-only registration is not complete-ready; missing or changed source disables readiness; an old quantized artifact with the same official index never satisfies BF16 Qwen.

### Phase 21 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/` | `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json` | pass — retained receipt records full verification and registration for all 26 shards; no new run | 2026-09-28 |
| `Sources/TurboFieldfareOfficialQwenSource/SourceRegistration.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/OfficialRegistrationAuditTests.swift` | pass — retained full-verification and registration audit (1/1, exit 0); no new run | 2026-09-28 |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift` | `Tests/TurboFieldfareApp/Core/Installation/OfficialRegistrationAppAuditTests.swift` | pass — retained app audit (2/2, exit 0), including tampered companion checks; no new run | 2026-09-28 |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/OfficialRegistrationAuditTests.swift` | pass — retained old-packed rejection audit (2/2, exit 0); no new run | 2026-09-28 |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/OfficialRegistrationAuditTests.swift` | `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json` | pass — existing receipt records the executed audit selector and 26-shard verification; no new run | 2026-09-28 |
| `Tests/TurboFieldfareApp/Core/Installation/OfficialRegistrationAppAuditTests.swift` | `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json` | pass — existing receipt records app readiness and old-packed rejection audit; no new run | 2026-09-28 |

### Phase 21 evidence

The retained [Phase 21 final receipt](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex/phase21-final-receipt.json) records full verification and registration of all 26 unchanged official shards, matching pinned checksum manifests, trusted reopen, text/vision registrations, and rejection of the old packed artifact. Its two selectors exited 0. This is existing evidence, not a new run from the documentation status update.

<a id="phase-22"></a>
## Phase 22 - Real original-BF16 text matches the independent reference

Purpose: Separate CPU reference and candidate model processes.
Needs: 3, 12, 13, 14, 15, 18, 20, 21. Status: partial; the recorded comparison ran but full accuracy qualification failed, execution paused.

<a id="task-22-9"></a>
#### 22.9 Run independent CPU reference alone

| Field | Detail |
|---|---|
| Touches | scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/ |
| Depends on | P3, P12, P13, P14, P15, P18, P20, P21 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use P3 independent parser and the pinned Transformers checkout. For each prefill/generated token, execute all 40 official decoder layers in order, loading all 256 original BF16 experts for only the current layer and calling its unmodified official forward/router Top-8. Release layer weights before the next while retaining needed activations, KV and recurrent state across sweeps. Apply final norm and chunked LM head to generate all 248,320 FP32 logits; chunk embeddings. Never load the whole model resident, copy all weights, or substitute candidate routing math. Record wrapper deviations and fully exit before candidate.
Why: Separate CPU reference and candidate model processes.

**Acceptance detail**
- [ ] Command, version, output and exit recorded.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.

<a id="task-22-10"></a>
#### 22.10 Run one official BF16 candidate process

| Field | Detail |
|---|---|
| Touches | TurboFieldfareCLI or TurboFieldfareMac and scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/text/ |
| Depends on | P3, P12, P13, P14, P15, P18, P20, P21 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: After preflight run bounded text with no overlapping model/reference process.
Why: Separate CPU reference and candidate model processes.

**Acceptance detail**
- [ ] Full command, timing/error and identity receipt.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.

<a id="task-22-11"></a>
#### 22.11 Compare real full logits and tokens

| Field | Detail |
|---|---|
| Touches | scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/text/ |
| Depends on | P3, P12, P13, P14, P15, P18, P20, P21 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Compare all 248,320 raw FP32 logits before public Float16, the independently rounded Float16 public vector, actual sampler tokens, route IDs/weights, route cutoff margins and argmax margins. Apply previously frozen tolerances and record every failure without widening them.
Why: Separate CPU reference and candidate model processes.

**Acceptance detail**
- [ ] Numerical tolerances and discrepancies recorded.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.


### Phase 22 acceptance

- [ ] Separate CPU reference and candidate model processes.
- [ ] Command, version, output and exit recorded.
- [ ] Full command, timing/error and identity receipt.
- [ ] Numerical tolerances and discrepancies recorded.

- [ ] P22 CPU reference executes all 40 official decoder layers per token/prefill in order, with all 256 experts resident only for the current layer, then final norm and chunked head yield 248,320 raw FP32 logits; its process exits before candidate starts.
- [ ] Full official comparisons use previously frozen tolerances and report route-cutoff and argmax margins without widening limits.

### Phase 22 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference/` | none yet — real operation receipt | missing — operation receipt pending |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/text/` | none yet — real operation receipt | missing — operation receipt pending |

### Phase 22 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-23"></a>
## Phase 23 - Real still images preserve the verified workflow

Purpose: Qualify real images and authorized NestMind Debug target.
Needs: 16, 20, 21, 22. Status: pending, execution paused.

<a id="task-23-8"></a>
#### 23.8 Run real source-backed still images

| Field | Detail |
|---|---|
| Touches | TurboFieldfareMac or TurboFieldfareCLI and scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/images/ |
| Depends on | P16, P20, P21, P22 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: After preflight run accepted images with valid adjacent companion.
Why: Qualify real images and authorized NestMind Debug target.

**Acceptance detail**
- [ ] Output, image rows, identity and timing recorded.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.

<a id="task-23-9"></a>
#### 23.9 Exercise authorized NestMind Debug target

| Field | Detail |
|---|---|
| Touches | NestMind Debug com.hebertgo.nestmind.debug on iPhone 17 iOS 26.5 simulator 7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382 |
| Depends on | P16, P20, P21, P22 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use existing authorized workflow; VoiceOver remains off.
Why: Qualify real images and authorized NestMind Debug target.

**Acceptance detail**
- [ ] External response and exact identity receipt.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.


### Phase 23 acceptance

- [ ] Qualify real images and authorized NestMind Debug target.
- [ ] Output, image rows, identity and timing recorded.
- [ ] External response and exact identity receipt.

### Phase 23 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/images/` | none yet — real operation receipt | missing — operation receipt pending |
| `no-unit-test — operation receipt for 23.9` | none yet — real operation receipt | missing — operation receipt pending |

### Phase 23 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-24"></a>
## Phase 24 - Actual memory and speed are measured on this Mac

Purpose: Static byte sums cannot prove fit or throughput.
Needs: 22, 23. Status: pending, execution paused.

<a id="task-24-8"></a>
#### 24.8 Measure BF16 resident and cache memory

| Field | Detail |
|---|---|
| Touches | scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/performance/ |
| Depends on | P22, P23 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Measure physical pressure and allocations at explicitly configured 8- and 16-slot app/CLI settings after P20 propagation is fixed; record the historical implicit app 8-slot behavior separately.
Why: Static byte sums cannot prove fit or throughput.

**Acceptance detail**
- [ ] Separate estimated bytes from real peak and paging.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.

<a id="task-24-9"></a>
#### 24.9 Measure supported text and image speed

| Field | Detail |
|---|---|
| Touches | scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/performance/ and README.md |
| Depends on | P22, P23 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Use community benchmark guidance with serial runs and complete timing footers.
Why: Static byte sums cannot prove fit or throughput.

**Acceptance detail**
- [ ] Tokens/s, settings and hit/miss counts recorded.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.

<a id="task-24-10"></a>
#### 24.10 Compare Gemma and Qwen serially

| Field | Detail |
|---|---|
| Touches | scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/performance/ |
| Depends on | P22, P23 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: State different precisions and context settings; keep Gemma default.
Why: Static byte sums cannot prove fit or throughput.

**Acceptance detail**
- [ ] Measured values and failures labelled.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.


### Phase 24 acceptance

- [ ] Static byte sums cannot prove fit or throughput.
- [ ] Separate estimated bytes from real peak and paging.
- [ ] Tokens/s, settings and hit/miss counts recorded.
- [ ] Measured values and failures labelled.

- [ ] App retained-state effective slot count is checked against configured 16 after P20; baseline 8 is a pre-fix observation, not a settings default.

### Phase 24 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/performance/` | none yet — real operation receipt | missing — operation receipt pending |

### Phase 24 evidence

none yet — original BF16 route acceptance pending.

<a id="phase-25"></a>
## Phase 25 - Opt-in Qwen can return safely to Gemma

Purpose: Prove default Gemma and one-action rollback.
Needs: 18, 19, 20, 22, 23, 24. Status: pending, execution paused.

<a id="task-25-8"></a>
#### 25.8 Check Gemma default and one-action rollback

| Field | Detail |
|---|---|
| Touches | TurboFieldfareMac and TurboFieldfareCLI and TurboFieldfareServer |
| Depends on | P18, P19, P20, P22, P23, P24 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Select BF16 Qwen then return to untouched Gemma with serial processes.
Why: Prove default Gemma and one-action rollback.

**Acceptance detail**
- [ ] Before/after identities and artifact integrity recorded.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.

<a id="task-25-9"></a>
#### 25.9 Review only version-2 acceptance gates

| Field | Detail |
|---|---|
| Touches | tracker and implementation and scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ |
| Depends on | P18, P19, P20, P22, P23, P24 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: Check new tests, independent reference, real text/image, performance and rollback without importing v1 passes.
Why: Prove default Gemma and one-action rollback.

**Acceptance detail**
- [ ] Final report names actual v2 evidence and open failures.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.

<a id="task-25-10"></a>
#### 25.10 Clean up only task-created temporary output

| Field | Detail |
|---|---|
| Touches | scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ temporary outputs |
| Depends on | P18, P19, P20, P22, P23, P24 |
| Parallel safe | Only with disjoint file ownership after listed dependencies; model/reference processes serial. |

Change: After evidence review remove only identified task-created temporary work. Protect the full official source root, scratch/gemma4.gturbo, Gemma assets, source registration, trust receipts and all evidence; record exact removed paths.
Why: Prove default Gemma and one-action rollback.

**Acceptance detail**
- [ ] Cleanup receipt lists baseline/end and removed paths.

**How to check**

See this task’s acceptance assertion and the phase coverage row. Record the named test or serial operation receipt and its actual result; none exists yet.

**Evidence**

none yet — write actual result under scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/ after this task is implemented.


### Phase 25 acceptance

- [ ] Prove default Gemma and one-action rollback.
- [ ] Before/after identities and artifact integrity recorded.
- [ ] Final report names actual v2 evidence and open failures.
- [ ] Cleanup receipt lists baseline/end and removed paths.

- [ ] Final cleanup preserves all 26 official shards, scratch/gemma4.gturbo, registration, receipts and accepted evidence.

### Phase 25 coverage plan

| Changed code or operation | Planned test or evidence | Current result |
|---|---|---|
| `no-unit-test — operation receipt for 25.8` | none yet — real operation receipt | missing — operation receipt pending |
| `scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/` | none yet — real operation receipt | missing — operation receipt pending |

### Phase 25 evidence

none yet — original BF16 route acceptance pending.

## Unmeasured limits

- Whether 32 GiB RAM fits BF16 shared weights, configured paired expert slots, vision group, state, Metal scratch and filesystem cache without harmful pressure remains unmeasured. No arbitrary memory cap is approved here.
- BF16 Metal speed and the actual device maxBufferLength need runtime checks. Static weight totals do not prove fit or throughput.
- File fingerprints detect ordinary mutation, not a malicious in-place rewrite that restores metadata. Full SHA mode verifies payload on load; trusted mode relies on the prior verified receipt and current fingerprints.
- MTP execution, video, audio and unverified long-context claims remain outside the selected route. Gemma v1, its defaults, server loopback binding and one-loaded-model behavior remain unchanged.
