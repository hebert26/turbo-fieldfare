# Implementation: Qwen3.6-35B-A3B official integration

## Current requirement — implementation paused, 2026-09-23

Hebert clarified that the already-downloaded approximately 72 GB official model must retain its published BF16 precision. The size on disk is not a requirement to load the whole model into RAM: keep the necessary shared working data in memory and read selected experts from disk as needed. The previous 21.4 GB conversion would quantize selected weights to INT4/INT8 and does not satisfy this requirement. **Do not execute the old quantized preflight/conversion or ask for approval for it.** No conversion has run; the original 26 shards remain present and both proposed Qwen outputs are absent.

Implementation is paused at Hebert’s explicit request. Only the requested read-only Gemma study, status explanations and document updates proceeded. The required outcome is original BF16 precision with selected experts streamed from disk. The replacement plan for reading the source files, storing expert data and bounding memory use has not yet been completed. A read-only code audit identified quantized assumptions in the Qwen loader, packed-expert layout, expert mapping and Metal projections. Resident-weight allocations also need measurement; the disk footprint alone proves neither fit nor failure on this Mac. Detailed affected tasks and acceptance evidence must be revised before their old qualification claims are applied to this path.

The version-1 plan and receipts below are preserved as historical evidence for the quantized candidate. Their 19/25 phase, 111/145 task and 245/293 coverage totals are not completion claims for the newly clarified full-precision requirement. Reusable behavior and tests must be distinguished from precision-dependent work during reassessment. Astra coordinates, GPT-6 Sol high owns production code, and GPT-6 Luna xhigh owns tests.

NestMind Debug (`com.hebertgo.nestmind.debug`) on iPhone 17 / iOS 26.5 (`7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382`) is now the authorized workflow target, as recorded in Phase 23 `authorization.md`. The user stopped VoiceOver testing; keep VoiceOver off and do not restart those tests without a new explicit request. Main’s spoken replies are also off, as requested on 2026-09-23.

Read-only findings: `scratch/qwen3.6-35b-a3b/evidence/coordination/official-bf16-feasibility.md`. No source/test changes, model operation or new runtime qualification accompanies this scope correction.

### What preserving precision means

Precision is the amount of numerical detail kept in the model’s stored numbers, called weights. The downloaded Qwen weights use BF16, the publisher’s original numerical format. Some existing Qwen code expects weights rounded into smaller INT4/INT8 formats. That code must change to read and calculate with the original weights. The 72 GB source download has not been changed. Keeping its original precision does not mean loading all of it into RAM.

The code review identified three areas to revise: the loader’s accepted weight formats, the description of each expert’s bytes and the GPU calculations that use those bytes. Memory allocation also needs review because the current runner expands many shared weights into larger arrays. The bounded expert reader can be reused as a starting point, but original-precision execution, output correctness and actual memory use remain unverified. See the [read-only BF16 findings](../../../../../turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/coordination/official-bf16-feasibility.md).

### What is next when implementation resumes

Finish the replacement plan first: identify reusable work, name the exact loader, expert-access, calculation and memory changes, and define the evidence needed to accept them. Preserve the existing source and avoid an unapproved full-model duplicate. The old quantized conversion is not a next step.

Then GPT-6 Sol with high reasoning implements the agreed changes, and GPT-6 Luna with xhigh reasoning writes and runs the tests. Independent code and test preparation may run in parallel with separate file ownership, up to 10 agents including Main. Builds, tests and model operations keep their existing serial execution rule. Qualification must cover real Qwen output and measured memory use before the authorized NestMind Debug workflow and final Gemma rollback checks.

This document reflects the decisions and evidence known on 2026-09-23. It is not a completed replacement implementation plan, and updating it does not resume implementation.

### Verified Gemma streaming reference

The current production code confirms the intended architecture. For each generated token, all 30 layers run in order. At each layer, attention and a separate routing calculation use the shared working data; the router selects 8 of that layer’s 128 experts. Those selections are recomputed for each layer and token. The runtime reuses a selected expert already in its cache. A miss reads only that expert’s bytes from the layer file into a bounded slot. The production default is 16 slots per opened layer, with least-used entries replaced first and recency breaking ties. The shared expert can compute while disk reads complete, then the routed and shared outputs are combined.

Memory also holds touched pages of the common weight file, attention’s key/value history, shared-expert weights, temporary calculation buffers and cached routed experts. The common file is mapped without copying it into Swift arrays. File size, mapped address space, allocated cache capacity and physical RAM usage are different measurements. This design does not require all routed experts to stay in RAM or a fresh disk read on every cache hit. Prompt processing uses chunks and differs from the one-token generation loop.

Current source evidence: `README.md:324`; `Sources/TurboFieldfare/Runtime/Inference/Model.swift:27`; `Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift:1584`; `Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift:219`; `Sources/TurboFieldfare/Runtime/Inference/KVCacheManager.swift:31`. GPT-6 Sol traced these paths read-only. No model or memory benchmark ran for this explanation. Gemma’s current source is already quantized; its repacker preserves those source values. For Qwen, preserving the downloaded source means retaining BF16. Expert streaming and weight precision are separate choices.

### Current decisions and limits

- Hebert stopped VoiceOver testing. It is off; test apps and temporary settings were restored. Later attempts verified startup only, not VoiceOver model selection. No passing accessibility claim is made.
- Hebert selected NestMind Debug on the pictured iPhone 17 / iOS 26.5. The exact bundle and simulator identity are recorded in Phase 23 `authorization.md`; do not ask for the target again.
- The original Qwen download is present: 26 shard files totaling 71,903,776,776 bytes. The proposed text and vision output directories are absent. This was a metadata check, not a new full-file integrity verification.
- The 21.4 GB conversion is withdrawn. Its existing code/tests and historical receipts prove only their stated quantized behavior. They do not prove a BF16 runtime works.
- The next Qwen implementation plan is not finished. No code changes, performance guarantee, new memory measurement or full-precision acceptance are claimed. Implementation remains paused.

### Historical version-1 plan and evidence

Everything below records the earlier quantized candidate and its development history. Commands and acceptance marks are retained for traceability, not as instructions to execute while paused. Historical statements such as “pending approval” or “running tests” describe their dated checkpoint; the current status above supersedes them.

### Codex coordination after takeover

Astra now coordinates this approved plan in Codex. GPT-6 Sol (`gpt-6-sol`) writes production code with high reasoning, per Hebert’s update on 2026-09-22. GPT-6 Luna (`gpt-6-luna`) executes tests and writes unit tests with xhigh reasoning, per Hebert’s update on 2026-09-22. Pi is stopped. Historical Pi review commands below identify the earlier workflow and must not be executed; use bounded independent Codex review of the same candidate and evidence. Builds, tests, app runs, conversion and model/reference execution remain sequential.

Tracker: [tracker-qwen3.6-35b-a3b-official-2026-09-15.md](./tracker-qwen3.6-35b-a3b-official-2026-09-15.md)
Date: 2026-09-15

---

## Historical approved scope - version 1

**Approved 2026-09-15T11:53:55Z. Owner said: "I want you to handle the full implementation on this. I want all the phases done, all of them. So by the end of the completion of the document, we should have a new version of TurboCharge, running, up and running, and of course it should work." (this session).**

Version 1 approves the full 25-phase implementation. Phase 1 is closed. Later model/data operations remain subject to their phase-specific authorization and repository preflight.

Add the exact official `Qwen/Qwen3.6-35B-A3B` revision `995ad96eacd98c81ed38be0c5b274b04031597b0` as an opt-in second family by deterministically converting the prepared BF16 source, implementing its text/state/chat/tool/vision behavior, and integrating verified identity through repacker, runtime, service, CLI, server, and Mac app. Preserve `.gturbo` v1, Gemma defaults/artifacts/behavior, loopback-only server binding, and one-loaded-model lifecycle. Version 1 covers the full 25-phase implementation. It does not itself authorize operational model/data actions; each such phase still requires its stated authorization and repository preflight.

### What changes

1. A format-owned `.gturbo` v2 family contract and verified installed-model descriptor are added while v1 bytes stay frozen.
2. The official local BF16 snapshot is validated, explicitly quantized, planned, written, resumed, audited, and receipt-bound without using community Qwen weights.
3. Concrete Qwen full attention, causal convolution, Gated DeltaNet, MoE, text runner, transactional state, tokenizer/chat/tools, and still-image/M-RoPE paths are implemented and independently checked.
4. Decode service, CLI, loopback server, and Mac app use verified family identity; Gemma remains default and one-action rollback.
5. Full conversion, model runs, comparison, installation, and process-affecting work remain future authorized operations with one model-related process at a time.

### What does not change

- `.gturbo` v1 field meaning, frozen fixtures, Gemma model path, source descriptor, runtime behavior, sampling defaults, and rollback status.
- The prepared official weights, receipts, and integrity evidence under `scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/`.
- VisionCapture host permissions, returned-identity checks, verified verdict, target locking, and no-replay rules.
- Server binding to `127.0.0.1`; video, audio, MTP execution, and unverified long-context claims remain outside initial scope.

### Authoritative background

- [Prepared official source](file:///Users/dev-machine/dev/turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/PREPARATION.md) — canonical local preparation record.
- [Architectural report](file:///Users/dev-machine/Documents/Idea%20Home/turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report.md) — archived background only; this tracker owns executable task state.
- [Repository instructions](file:///Users/dev-machine/dev/turbo-fieldfare-personal/AGENTS.md) — model-run, process, storage, and reporting constraints.

### Intended code boundaries

| Area | Existing files to adapt | Proposed additions |
|---|---|---|
| Format/load | `Sources/TurboFieldfareFormat/GTurbo*V1.swift`; `ManifestReader.swift`; `ModelTypes.swift` | v2 dispatch, Qwen text/vision schema, verified descriptor |
| Repacker | `SupportedModelSource.swift`; `IndexLoader.swift`; `RepackPlanner.swift`; writer/checkpoint/audit files | official identity/tensor map, local loader, BF16 policy/quantizer, Qwen planner/workflow |
| Runtime | `Model.swift`; `LogitProducer.swift`; `MultimodalConversation.swift`; `RealInferenceClient.swift` | family factory, Qwen model/runner, full/linear attention, MoE, complete transaction state |
| Token/vision | current Gemma tokenizer/parser/vision dispatch points | Qwen tokenizer/chat/tools, processor/tower/merger/M-RoPE, v2 companion loader |
| Products | decode protocol/service, CLI core, server core, app settings/state/views/tools | verified identity/epoch routing, model catalog, accessible picker |
| Evidence | current SwiftPM targets in `Package.swift` | proposed focused tests plus future evidence under repository `scratch/` |

Every intended file is listed again in its phase coverage plan. `PROPOSED` means the path does not exist yet. Test owners write only test paths; production owners do not coedit them.

---

## Problem

The official snapshot is prepared and verified but unconverted and unrunnable. Current format, loader, runner, tokenizer, vision, installer, app, CLI, and server are concrete Gemma implementations. The existing repacker copies an already quantized MLX layout; its runtime quantization helper is explicitly fixture-only.

Qwen is not a larger Gemma configuration. It combines ten full-attention layers with thirty causal-convolution/Gated-DeltaNet layers, FP32 recurrent state, Q/K normalization, output gating, 256-way Top-8 MoE plus shared expert, untied head, a different tokenizer/chat/tool grammar, dynamic vision grids, and three-axis interleaved M-RoPE. State recovery therefore has to transact KV, convolution, recurrence, tokens, and positions together.

This plan separates independent file ownership and numerical gates so format, fixtures, quantization, ingestion, and chat can progress in parallel after identity. No whole phase starts until every phase in its `Needs` cell has closed D1–D5.

## Topological waves

| Wave | Phases that may start together after prior needs close |
|---|---|
| 0 | P1 exact identity |
| 1 | P2 format, P3 fixtures, P4 quantizer, P5 local ingestion, P14 chat template |
| 2 | P6 planner, P8 Metal contracts, P15 tool decoder |
| 3 | P7 writer plus P9 full attention, P10 linear attention, P11 MoE |
| 4 | P12 tiny runner; P21 conversion becomes dependency-ready after P7 but remains separately authorized |
| 5 | P13 text transaction state |
| 6 | P16 vision and P17 decode service |
| 7 | P18 CLI, P19 server, P20 app |
| 8 | P22 real text after P21 artifact |
| 9 | P23 real still images |
| 10 | P24 controlled comparison |
| 11 | P25 installed opt-in and rollback |

P9/P10/P11 have disjoint Swift/MSL/test ownership and may run in parallel after P2/P3/P8. P18/P19/P20 may run in parallel after their listed contracts. P21–P25 never overlap conversion, app, CLI, server, reference, or model-using processes; readiness does not waive the one-process rule.

## Shared-file ownership transfers

| File | First owner | Later owner | Required handoff |
|---|---|---|---|
| `ManifestReader.swift`, `ModelTypes.swift` | P2 | runtime consumers read only | P2 closes before P12/P16/P17 |
| `MetalContext.swift` and `Package.swift` resource contract | P8 | P9/P10/P11/P16 use registered contract | one P8 integration owner; later phases add separate shader bodies |
| writer/checkpoint/disk/audit files | P7 | P21 workflow consumes only | P7 closes before P21 |
| `Model.swift` | P12 | product phases consume only | P12 owns Qwen load/factory integration |
| `LogitProducer.swift`, `MultimodalConversation.swift`, `RealInferenceClient.swift` | P13 | P16 extends only Qwen image state file | P13 closes before P16 |
| `QwenConversationState.swift` | P13 text state | P16 image lineage | serial transfer documented by task 16.5 |
| `VisionRuntime.swift`, `MultimodalPrefillInput.swift`, `MultimodalPromptRenderer.swift` | P16 | product phases consume only | P16 closes before P18/P19/P20 |
| `DecodeProtocol.swift`, `Entry.swift`, app inference clients | P17 | P20 lifecycle consumes | P17 closes before P20 |
| `Run.swift` | P18 | final verification read only | no shared writer |
| server core files | P19 | final verification read only | no shared writer |
| `AppModel.swift`, settings, picker views, `VisionCaptureToolLoop.swift` | P20 | final verification read only | P20 is sole app integration writer |

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Wrong source or silent tensor omission | high | P1/P5 identity and exhaustive classification precede planning; P6 accounts for every tensor. |
| Nondeterministic or incorrect quantization | high | P4 fixes byte arithmetic/goldens before P6/P7; P22 compares the same quantized weights. |
| Recurrent state restored as if it were KV | high | P10 owns explicit state; P13 transactions compare next-turn state with clean replay. |
| Placeholder GPU success | high | P8 has declarations only; P9/P10/P11/P16 require concrete pipelines and numerical GPU execution. |
| Gemma regression or artifact mutation | high | v1 fixtures, catalog/default/path freezes, separate destinations, before/after digests, and final rollback. |
| Stale identity crosses process/product boundaries | high | descriptor plus epoch binds service, app, CLI, server, caches, and diagnostics. |
| Unsafe tool action inferred from model output | high | malformed output produces no call; VisionCapture host evidence and permissions remain authoritative. |
| Disk/process interference | high | current preflight at each operation, one model-related process, no killing or cleanup outside owned authorization. |
| Unsupported performance claim | medium | no Qwen threshold until P24 controlled evidence; validation timing excluded. |

## Owners

Roles below are duties, not assigned people or active agents.

| Duty | Exclusive responsibility |
|---|---|
| `*-code`, `format-code`, `metal-interface` | Production source only; never writes the tests named for that task. |
| `test`, `compat-test`, `fixture-test` | Test/fixture paths only; derives behavior from requirements and does not change production scope. |
| `metal` task owners | One `.metal` body or common-contract path at a time; concrete numerical proof stays with the owning phase. |
| `integration` / product duty | Shared call-site integration only after producer contracts close; no concurrent writer to the same file. |
| `operator` / `*-operator` | Future authorized commands only, after AGENTS.md preflight, one model-related process at a time. |
| `reviewer` | Fresh read-only exact-candidate review; no edits or delegated review. |
| `cleanup-owner` | Only dispensable temporary artifacts provably created by this work item; protected list is absolute. |

Authoring inventory: `25` phases, `145` tasks, and `218` actual tracker coverage rows. At that earlier authoring checkpoint, the board was `15/25` phases accepted, `77/145` tasks done, `111/218` coverage rows passing, and `107` pending. These are historical checkpoint figures, not the latest totals. Reproduce the phase and task counts with `grep -c '^## Phase ' tracker-*.md` and `grep -cE '^- \[.\] [0-9]+\.[0-9]+ ' tracker-*.md`; pending coverage is `grep -c '^| .*missing - phase not started' tracker-*.md`. The authoring generator also asserted every literal existing path resolves and every numeric task dependency ID exists.

---

## Phase 1 - The exact official checkpoint is recognized

**When this is done:** the exact official checkpoint is recognized

Needs: `nothing`
Base commit: `ae138d53c244f32985b7a98b0293b20361c499e4`
Disk baseline: `2026-09-15T11:53:55Z — / has 56 GiB free; existing worktrees and pre-existing TurboFieldfareMac/DecodeService processes were preserved; metadata-only test clarification applied.`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-1/`

### Current state

The canonical prepared bundle is `scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/`. `PREPARATION.md` records 26 shards and completed publisher/local digest checks; `STRUCTURAL_VERIFICATION.json` records 1,045 BF16 tensors, while `model.safetensors.index.json` is the authority for the 19 `mtp.*` names. At the Phase 1 baseline, `Sources/TurboFieldfareRepack/Core/Remote/SupportedModelSource.swift:3-51` knew only the pinned community Gemma source.

### Target state

A small source contract recognizes only `Qwen/Qwen3.6-35B-A3B` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, binds the seven behavior-defining sidecar digests, and classifies every indexed tensor name. Tests use metadata-only fixtures; no shard is copied or opened as a model.

### Tasks

#### 1.1 Record the metadata-only official identity fixture

| | |
|---|---|
| Duty | `fixture-test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Format/Fixtures/Qwen36OfficialMetadata.swift` |
| Depends on | `-` |
| Parallel safe | `yes` |
| Deliverable | A Swift fixture records the repository, revision, sidecar SHA-256 values, 1,045-name classification expectation, and 19 explicit MTP omissions. |

Copy only values already recorded in `PREPARATION.md`, `SHA256SUMS`, `STRUCTURAL_VERIFICATION.json`, `config.json`, and the index. Do not copy shard payloads or infer a quantization policy.

**Acceptance detail**
- [x] The fixture names the exact repository and 40-character revision.
- [x] Every count and digest cites the canonical preparation file that produced it.

**How to check**
```sh
Scripts/test.sh --filter QwenOfficialIdentityTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-1/qwen-official-filter.log` — the frozen candidate filter passed with 26 test functions in 2 suites and 62 cases; exit 0.

#### 1.2 Implement immutable Qwen source identity validation

| | |
|---|---|
| Duty | `code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift` |
| Depends on | `1.1` |
| Parallel safe | `yes` |
| Deliverable | `QwenOfficialIdentity` accepts the pinned sidecars and rejects any repository, revision, digest, architecture, tokenizer, or processor mismatch. |

Model identity is independent of the later quantization policy. Keep constants closed and product-stable; do not add a community Qwen fallback.

**Acceptance detail**
- [x] The exact pinned identity validates from metadata alone.
- [x] A one-byte digest or one-character revision change fails before planning.

**How to check**
```sh
Scripts/test.sh --filter QwenOfficialIdentityTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-1/qwen-official-filter.log` — the frozen candidate filter passed with 26 test functions in 2 suites and 62 cases; exit 0.

#### 1.3 Map all official tensor names without loading weights

| | |
|---|---|
| Duty | `code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift` |
| Depends on | `1.1` |
| Parallel safe | `yes` |
| Deliverable | A closed classifier assigns each indexed name to text-resident, routed-expert, vision, or intentionally omitted MTP. |

Derive the legal prefixes and layer bounds from the official index and config. Reject unknown names instead of silently dropping them.

**Acceptance detail**
- [x] All 1,045 indexed names receive exactly one classification.
- [x] Exactly 19 `mtp.*` names are intentional omissions and unknown names fail.

**How to check**
```sh
Scripts/test.sh --filter QwenOfficialTensorMapTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-1/qwen-official-filter.log` — the frozen candidate filter passed with 26 test functions in 2 suites and 62 cases; exit 0.

#### 1.4 Add independent identity and tensor-map rejection tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift`; `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` |
| Depends on | `1.1` |
| Parallel safe | `yes` |
| Deliverable | Tests independently exercise accepted metadata and mutations of every identity and classification boundary. |

Write tests only. Include wrong Qwen size, `qwen3_5_moe` architecture-name confusion, missing/extra tensor, duplicate class, wrong dtype count, and changed sidecar digest.

**Acceptance detail**
- [x] Both identity and tensor-map suites execute with a nonzero Swift Testing count.
- [x] Each plausible wrong checkpoint fails before any output path is created.

**How to check**
```sh
Scripts/test.sh --filter QwenOfficial
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-1/qwen-official-filter.log` — the frozen candidate filter passed with 26 test functions in 2 suites and 62 cases; exit 0.

### Phase 1 coverage plan

The frozen Phase 1 candidate changed exactly five paths. Each path has an executable test mapping. Each single-suite coverage row reports 13/13 passing test functions; the combined filter evidence reports 26 test functions in 2 suites and 62 cases.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | Exact metadata acceptance and single-field identity rejection. | `Scripts/test.sh --filter QwenOfficial` |
| `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` | Exhaustive, exclusive tensor classification and rejection of unknown names. | `Scripts/test.sh --filter QwenOfficial` |
| `Tests/TurboFieldfareRepack/Core/Format/Fixtures/Qwen36OfficialMetadata.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | Metadata-only fixture digests, counts, architecture names, and MTP split. | `Scripts/test.sh --filter QwenOfficial` |
| `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift` | The identity suite executes and rejects mutated repository, revision, digest, and config fields. | `Scripts/test.sh --filter QwenOfficial` |
| `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` | `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift` | The tensor-map suite executes and rejects missing, extra, duplicate, misplaced, and wrong-dtype descriptors. | `Scripts/test.sh --filter QwenOfficial` |

### Phase 1 evidence

<a id="phase-1-evidence"></a>

Candidate identity: base commit `ae138d53c244f32985b7a98b0293b20361c499e4`; frozen candidate digest `c28d3d3d9745933bce917ea46e3e65abeb927b4cd11bd015861ea252c4245f9b`.

Code-gate provenance: team workflow `workflow-mac-engineering-standard-mu2lgj8p-vvxl9x` recorded Grok approval `#77` and full approval `#81` on verification `#75`. This documentation closeout does not rerun code review.

GPU, model-run, and planner gates are N/A for Phase 1; planner integration is deferred to Phase 6.

The generic structural checker’s D4 result is structural-checker PASS (“no source files changed”) because its script watches a VisionCapture-only pathspec. It is not authoritative for this project’s D4. The authoritative project-specific D4 uses the full `PATHS=(Sources Tests Package.swift)` four-line union and the `git status --porcelain` guard; it reports the union count and any extra paths instead of hiding them.

The Phase 1 changed-source inventory is exactly these five candidate paths; existing dirty `.pi` edits are excluded:

- `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialIdentity.swift`
- `Sources/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMap.swift`
- `Tests/TurboFieldfareRepack/Core/Format/Fixtures/Qwen36OfficialMetadata.swift`
- `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialIdentityTests.swift`
- `Tests/TurboFieldfareRepack/Core/Format/QwenOfficialTensorMapTests.swift`

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-15 | `Scripts/test.sh --filter QwenOfficial` | 26 test functions, 2 suites, 62 cases passed; exit 0; model-free with no GPU, shard, or header read. | `scratch/qwen3.6-35b-a3b/evidence/phase-1/qwen-official-filter.log` |
| 2026-09-15 | `Scripts/test.sh --filter SafetensorsHostileHeaderTests\|RangeCopyPlannerTests\|RepackCLITests\|DiskSpaceCheckerTests\|InstallProgressReporterTests\|GTurboDirectoryAccessTests` | 32 tests in 6 suites passed; exit 0; model-free with no GPU, shard, or header read. | `scratch/qwen3.6-35b-a3b/evidence/phase-1/baseline-six-suites.log` |
| 2026-09-15 | `Scripts/test.sh --filter GTurboFormatCompatibilityTests` | 3 tests in 1 suite passed; exit 0; frozen v1 compatibility remained passing. | `scratch/qwen3.6-35b-a3b/evidence/phase-1/gemma-format-compat.log` |
| 2026-09-15 | `swift build -c release --target TurboFieldfareRepackCore` | Production build complete; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-1/release-repack-core.log` |
| 2026-09-15 | `shasum -a 256` over the five candidate paths | All five recorded hashes match the frozen candidate; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-1/candidate-file-hashes.txt` |
| 2026-09-15 | Project-specific D4 four-line union with `PATHS=(Sources Tests Package.swift)` and `git status --porcelain` guard | Exactly the five paths above, no extras; union and guard exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-1/d4-source-inventory.log` |

The prior tester unnecessarily performed a read-only rehash of the full 38-file `SHA256SUMS` set. That violated the no-shard-read instruction, made no mutations, is excluded from acceptance evidence, and must not be repeated.

## Phase 2 - A v2 Qwen manifest loads without changing v1

**When this is done:** a v2 Qwen manifest loads without changing v1

Needs: `1`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-15T13:10:54Z — / had 59 GiB free; memory_pressure reported 39% free; four existing worktrees and the pre-existing installed TurboFieldfareMac/DecodeService processes were preserved.`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-2/`

### Current state

`Sources/TurboFieldfareFormat/GTurboFormatV1.swift:3-44`, `GTurboManifestV1.swift:93-395`, and `GTurboVisionFormatV1.swift:3-288` are strict Gemma v1 contracts. `Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift:75-447` duplicates the v1 wire schema and validates against caller-supplied `ArchConfig` in `ModelTypes.swift:4-71`.

### Target state

A format-owned dispatcher reads the major version first. V1 takes the existing byte-compatible path. V2 carries a closed family union, the exact Qwen architecture, conversion provenance, per-group quantization, ignored tensors, text/vision binding, and a descriptor created only after validation. Runtime types consume that descriptor rather than duplicating wire structs.

### Tasks

#### 2.1 Define the v2 header and closed architecture union

| | |
|---|---|
| Duty | `format-code` |
| Touches | `Sources/TurboFieldfareFormat/GTurboFormatV2.swift` |
| Depends on | `1.4` |
| Parallel safe | `yes` |
| Deliverable | V2 wire types encode Gemma or Qwen explicitly and reject unknown families or required features. |

Include Qwen layer schedule, recurrent/convolution fields, full-attention dimensions, FP32 state, partial/interleaved RoPE, MoE rules, untied head, provenance, quantization groups, and ignored tensors.

**Acceptance detail**
- [x] No Qwen field is represented by a misleading Gemma field.
- [x] Unknown major versions, families, and required features fail closed.

**How to check**
```sh
Scripts/test.sh --filter GTurboFormatV2Tests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-fresh-test-GTurboFormatV2Tests.log` — 5 format tests in 1 suite passed; exit 0.

#### 2.2 Define the v2 Qwen vision binding contract

| | |
|---|---|
| Duty | `format-code` |
| Touches | `Sources/TurboFieldfareFormat/GTurboVisionFormatV2.swift` |
| Depends on | `1.4` |
| Parallel safe | `yes` |
| Deliverable | Qwen vision metadata binds family, official revision, processor profile, text manifest digest, and vision payload digest. |

Keep `gemma4_vision_companion` v1 untouched. Model video as unsupported rather than treating it as an image or text request.

**Acceptance detail**
- [x] A Qwen companion cannot decode as Gemma vision v1.
- [x] Wrong family, revision, processor hash, or text digest fails structural validation.

**How to check**
```sh
Scripts/test.sh --filter GTurboVisionFormatV2Tests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-fresh-test-GTurboVisionFormatV2Tests.log` — 4 vision tests in 1 suite passed; exact text-digest mismatch is exercised by `InstalledModelDescriptorTests`; exit 0.

#### 2.3 Add the verified installed-model descriptor

| | |
|---|---|
| Duty | `format-code` |
| Touches | `Sources/TurboFieldfareFormat/InstalledModelDescriptor.swift` |
| Depends on | `2.1, 2.2` |
| Parallel safe | `yes` |
| Deliverable | One Codable descriptor exposes validated family, model ID, revision, format, vision status, and quantization profile. |

The descriptor is an output of validation. Do not provide an initializer that lets app or server call sites manufacture verified identity from display strings.

**Acceptance detail**
- [x] Descriptor round trips preserve all identity fields.
- [x] It cannot claim verified vision without a bound companion descriptor.

**How to check**
```sh
Scripts/test.sh --filter InstalledModelDescriptorTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-fresh-test-InstalledModelDescriptorTests.log` — 3 descriptor tests in 1 suite passed, including the text/vision digest negative control; exit 0.

#### 2.4 Route runtime loading through format-owned dispatch

| | |
|---|---|
| Duty | `runtime-code` |
| Touches | `Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift`; `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` |
| Depends on | `2.1, 2.2, 2.3` |
| Parallel safe | `no` |
| Deliverable | `ManifestReader` consumes the format dispatcher and returns family-specific runtime configuration plus the verified descriptor. |

Remove new-schema duplication while preserving the public behavior used by existing v1 callers and toy configs.

**Acceptance detail**
- [x] Existing v1 loader tests still use their current Gemma expectations.
- [x] Qwen v2 derives identity from bytes, not a caller-supplied model label.

**How to check**
```sh
Scripts/test.sh --filter ManifestReaderQwenV2Tests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-fresh-test-ManifestReaderQwenV2Tests.log` — 2 Qwen reader tests in 1 suite passed; the separate v1 reader regression ran 16 tests; both exit 0.

#### 2.5 Add hostile v2 and descriptor tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareFormat/GTurboFormatV2Tests.swift`; `Tests/TurboFieldfareFormat/GTurboVisionFormatV2Tests.swift`; `Tests/TurboFieldfareFormat/InstalledModelDescriptorTests.swift`; `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestReaderQwenV2Tests.swift` |
| Depends on | `2.1, 2.2, 2.3` |
| Parallel safe | `yes` |
| Deliverable | New tests mutate family, schedule, recurrent fields, vocabulary, tied head, quantization, ignored tensors, path ranges, and companion binding. |

Test ownership is limited to the four Phase 2 test files. Use synthetic manifests small enough for fast execution.

**Acceptance detail**
- [x] Every Phase 2 suite executes with a nonzero count.
- [x] Each mutation fails at the narrow validator that owns the contract.

**How to check**
```sh
Scripts/test.sh --filter GTurboFormatV2Tests
Scripts/test.sh --filter GTurboVisionFormatV2Tests
Scripts/test.sh --filter InstalledModelDescriptorTests
Scripts/test.sh --filter ManifestReaderQwenV2Tests
```

**Evidence:** The corrected focused logs under `scratch/qwen3.6-35b-a3b/evidence/phase-2/` record 5 format, 4 vision, 3 descriptor, and 2 Qwen reader tests; each exits 0.

#### 2.6 Re-run the frozen v1 compatibility fixture unchanged

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfareFormatCompatibility/GTurboFormatCompatibilityTests.swift` |
| Depends on | `2.4, 2.5` |
| Parallel safe | `no` |
| Deliverable | The existing compatibility suite remains the rollback gate and receives only any assertion needed to prove dispatch chose v1. |

Do not regenerate or edit `Tests/TurboFieldfareFormatCompatibility/Fixtures/v1/`.

**Acceptance detail**
- [x] All existing v1 fixture bytes decode to their current values.
- [x] The v2 work changes no v1 fixture digest or semantic field.

**How to check**
```sh
Scripts/test.sh --filter GTurboFormatCompatibilityTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-fresh-test-GTurboFormatCompatibilityTests.log` — 3 frozen compatibility tests and their fixed hashes passed; exit 0.

### Phase 2 coverage plan

The final table maps every Phase 2 path to the focused suite that executed it. The authoritative D4 inventory reports nine Phase 2 paths, five inherited Phase 1 paths, and no unexplained extras.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareFormat/GTurboFormatV2.swift` | `Tests/TurboFieldfareFormat/GTurboFormatV2Tests.swift` | Closed Gemma/Qwen dispatch plus hostile family, feature, architecture, provenance, quantization, omission, path, and range validation. | `Scripts/test.sh --filter GTurboFormatV2Tests` |
| `Sources/TurboFieldfareFormat/GTurboVisionFormatV2.swift` | `Tests/TurboFieldfareFormat/GTurboVisionFormatV2Tests.swift` | Separate Qwen vision dispatch and family, revision, processor, payload, and video rejection. | `Scripts/test.sh --filter GTurboVisionFormatV2Tests` |
| `Sources/TurboFieldfareFormat/InstalledModelDescriptor.swift` | `Tests/TurboFieldfareFormat/InstalledModelDescriptorTests.swift` | Codable identity/quantization preservation, forged-label rejection, and exact text/vision digest binding. | `Scripts/test.sh --filter InstalledModelDescriptorTests` |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift` | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestReaderQwenV2Tests.swift` | Verified Qwen load derives identity from bytes while the v1 adapter preserves established errors. | `Scripts/test.sh --filter ManifestReaderQwenV2Tests` |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestReaderQwenV2Tests.swift` | Qwen runtime fields retain Qwen meaning and do not occupy Gemma `ArchConfig` slots. | `Scripts/test.sh --filter ManifestReaderQwenV2Tests` |
| `Tests/TurboFieldfareFormat/GTurboFormatV2Tests.swift` | `Tests/TurboFieldfareFormat/GTurboFormatV2Tests.swift` | The suite executes with a nonzero count and zero failures. | `Scripts/test.sh --filter GTurboFormatV2Tests` |
| `Tests/TurboFieldfareFormat/GTurboVisionFormatV2Tests.swift` | `Tests/TurboFieldfareFormat/GTurboVisionFormatV2Tests.swift` | The suite executes with a nonzero count and zero failures. | `Scripts/test.sh --filter GTurboVisionFormatV2Tests` |
| `Tests/TurboFieldfareFormat/InstalledModelDescriptorTests.swift` | `Tests/TurboFieldfareFormat/InstalledModelDescriptorTests.swift` | The suite executes with a nonzero count and zero failures. | `Scripts/test.sh --filter InstalledModelDescriptorTests` |
| `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestReaderQwenV2Tests.swift` | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ManifestReaderQwenV2Tests.swift` | The suite executes with a nonzero count and zero failures. | `Scripts/test.sh --filter ManifestReaderQwenV2Tests` |
| `Tests/TurboFieldfareFormatCompatibility/GTurboFormatCompatibilityTests.swift` | `Tests/TurboFieldfareFormatCompatibility/GTurboFormatCompatibilityTests.swift` | Existing frozen v1 text/vision bytes, hashes, constants, and semantics remain unchanged. | `Scripts/test.sh --filter GTurboFormatCompatibilityTests` |

### Phase 2 evidence

<a id="phase-2-evidence"></a>

Candidate identity: base commit `5770260510935cd32b82c4008ff06ae6458539ff`; frozen code/test candidate digest `5ddb0e82337fb5716005c3584c0716bb0f954737177921ffb8f49265e3e6b597`. `candidate-file-hashes.txt` covers all nine Phase 2 inputs and the five inherited, frozen Phase 1 inputs. The current-file recheck matches. Superseded candidate `b833f068321d148c2370858736736ad829a2e3cb900161e0c7e40ee78bcf6241` was rejected at verifier event `#90` and is not submitted for approval.

The correction makes the descriptor carry the SHA-256 of the exact text-manifest bytes. Codable validation now requires any verified vision companion's compatible-text digest to equal that identity. This proves structural binding consistency; it does not prove that later on-disk payload files match their declared digests.

The environment was Xcode 26.5 (17F42), macOS SDK 26.5, Swift 6.3.2 in Swift 6.2 package mode, macOS 26 deployment, arm64 macOS 26.6.2, and Apple M2 Pro. Phase 2 parses synthetic metadata only; GPU, model execution, conversion, performance, and Metal validation are not applicable. The pre-existing installed app and decode-service processes were not touched.

The generic tracker checker watches a VisionCapture-only pathspec, so its D4 statement is not authoritative here. Project-specific D4 fixed `BASE=5770260510935cd32b82c4008ff06ae6458539ff`, used `PATHS=(Sources Tests Package.swift)`, captured committed/unstaged/staged/untracked streams separately, their sorted union, and the status guard. It reports 14 paths: the five unchanged inherited Phase 1 files plus exactly nine covered Phase 2 files, with zero unexplained extras.

Independent mac-test event `#105` reports 3/3 descriptor tests passing for the corrected API, including the hostile Codable companion-digest mutation that detects the rejected implementation. Phase 2 closeout date: `2026-09-15`. Workflow `workflow-mac-engineering-complex-mu2ooosx-6hxiuv` records Terra approval `#119` and Grok approval `#121`; both approved verification `#117` for exact candidate `5ddb0e82337fb5716005c3584c0716bb0f954737177921ffb8f49265e3e6b597`. Phase 2 is closed; Phase 3 is ready but not started.

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-15 | Environment, phase base, status, safe code hashes, disk/worktree/process/memory capture | Exit 0; no shard or `SHA256SUMS` read. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/environment-and-start-identity.log` |
| 2026-09-15 | Baseline Debug and Release builds for `TurboFieldfareFormat` and `TurboFieldfare` | All four builds passed; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/baseline-{debug,release}-*.log` |
| 2026-09-15 | Baseline v1 format/vision/reader/compatibility filters | 31 tests in 4 suites passed; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/baseline-v1-reader-suites.log` |
| 2026-09-15 | Corrected candidate Debug and Release affected-target builds in isolated scratch path | Both configurations of `TurboFieldfareFormat` and `TurboFieldfare` passed without attributable diagnostics; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-fresh-{debug,release}-*.log` |
| 2026-09-15 | Four corrected focused suites in the same isolated scratch path | 5 format, 4 vision, 3 descriptor, and 2 reader tests passed; each filter nonzero; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-fresh-test-*.log` |
| 2026-09-15 | Corrected frozen v1 compatibility plus three existing regression suites | 3 compatibility, 6 manifest codec, 6 v1 vision, and 16 reader tests passed; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-fresh-test-*.log` |
| 2026-09-15 | Independent hostile Codable regression | 3 descriptor tests passed, including forged companion-digest rejection; exit 0; team event `#105`. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/tester-regression-InstalledModelDescriptorTests.log` |
| 2026-09-15 | Default-path incremental reader attempts after the public value-layout change | SwiftPM testing helper signaled 11 twice before executing a nonzero count. Both failures are retained; the isolated scratch rebuild then executed 2/2 successfully. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/correction-test-ManifestReaderQwenV2Tests{,-retry}.log` |
| 2026-09-15 | Candidate file hashes and current-input digest recheck | Recomputed hashes for all 14 current input files match the frozen digest; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/{candidate-file-hashes.txt,candidate-hash-recheck.log}` |
| 2026-09-15 | Authoritative four-stream D4 and status guard | 5 inherited P1 + 9 P2 paths; zero extras; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/d4-source-inventory.log` |
| 2026-09-15 | HTML generation through the disclosed lowercase scratch mirror | 25 phases and 10/145 completed tasks rendered, then copied to canonical `Project-files/human`; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/html-generation.log` |
| 2026-09-15 | Generic tracker checker on the lowercase mirror | All structural checks passed; exit 0. Its VisionCapture-only D4 line is non-authoritative and superseded by the project D4 row above. | `scratch/qwen3.6-35b-a3b/evidence/phase-2/tracker-checker.log` |

---

## Phase 3 - Tiny Qwen execution fixtures are reproducible

**When this is done:** tiny Qwen execution fixtures are reproducible

Needs: `1`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `62 GiB available (66,054,324,224 bytes), memory 63% free, zero forbidden model processes, three pre-existing secondary worktrees plus this checkout; full capture in phase-start-baseline.log`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-3/`

### Current state

Phase 3 implementation and checks are complete. Terra #140 and Grok #142 formally approved verification #139 for the exact preapproval candidate all-artifact digest `0db70b9179c882f06e906e745703a9c0cae5f4f1b36c1524e06d71df607b7781`; D1-D5 are complete. The repository has a frozen tiny Qwen numerical oracle, its pinned offline CPU generator, resource registration, and an independent digest/schema/integrity suite. Existing Gemma behavior and the inherited Phase 1/2 inputs are unchanged.

### Target state

Phase 3 is closed. The deterministic tiny configuration preserves the three-linear/one-full schedule and emits full-attention, linear-attention, MoE, greedy-text, actual depth-one vision-tower/merger, and text M-RoPE inputs, intermediates, final values, state, tolerances, and digests. Runtime owners can consume the fixture but cannot silently regenerate its oracle.

**P3 runner-proof boundary:** the frozen P3 `greedyText` bytes remain valid reduced-graph and component evidence, but they do not prove a complete normalized, stateful runner. The source schedule at `schedule_forward` lines 427–439 omits decoder input-layernorm, post-attention-layernorm, and final `model.norm`; `greedy_fixture` lines 442–457 recomputes growing full sequences rather than performing cached incremental decode. This limitation was independently confirmed for Phase 12 planning. P3 bytes, results, tolerances, and approvals remain unchanged; Phase 12 adds a separate complete text-model oracle rather than relabeling the old greedy fixture.

Candidate identity: base commit `5770260510935cd32b82c4008ff06ae6458539ff`. The external `candidate-identity.txt` records one digest of the frozen path-sorted `candidate-file-hashes.txt`, covering the behavioral inputs, submitted evidence, and all three canonical documents without a self-reference. The reviewed preapproval all-artifact digest covers the then-current canonical documents only; this postapproval closeout records a separate document-only hash for the updated canonical documents and does not claim that the updated documents were part of the reviewed digest. The fixture file itself is 5,732,242 bytes with SHA-256 `61999aed7cf049debf980f3482af7ae91b70dcf25a3fcebdd613255c0f824dca`.

### Tasks

#### 3.1 Write the pinned tiny-fixture generator

| | |
|---|---|
| Duty | `fixture-owner` |
| Touches | `scratch/qwen3.6-35b-a3b/fixture-generator/generate_qwen36_fixtures.py` (**PROPOSED**) |
| Depends on | `1.4` |
| Parallel safe | `yes` |
| Deliverable | A standalone generator records official revision, Transformers commit, arguments, deterministic seed, tensor layout, and output digests. |

Use a tiny architecture that retains the 3-linear/1-full repeating order, output gate, convolution history, FP32 DeltaNet state, Top-8 normalization, untied head, and interleaved M-RoPE.

**Acceptance detail**
- [x] Two successful invocations with identical inputs produced the same exact 5,732,242 fixture bytes and digest.
- [x] Metadata names both immutable upstream revisions and every tolerance source.

**How to check**
```sh
env PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HUB_KERNELS=0 scratch/qwen3.6-35b-a3b/reference-environment/venv/bin/python scratch/qwen3.6-35b-a3b/fixture-generator/generate_qwen36_fixtures.py --output <temporary-output> --negative-controls-output <temporary-control-output>
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-3/generator-run-2.log`, `generator-run-3.log`, `determinism-and-freeze.log`, and `fixture-generation.md`.

#### 3.2 Generate full and linear attention fixture sections

| | |
|---|---|
| Duty | `fixture-owner` |
| Touches | `Tests/TurboFieldfare/Core/QwenFixtures/qwen36-tiny-fixtures.json` (**PROPOSED**) |
| Depends on | `3.1` |
| Parallel safe | `no` |
| Deliverable | The fixture contains projections, normalized Q/K, partial RoPE, output gate, convolution history, recurrence, chunk boundaries, outputs, and final state. |

Include empty, one-token, awkward-length, repeated-decode, and split-prefill cases. Store explicit shapes and dtypes beside values.

**Acceptance detail**
- [x] The fixture exposes full-attention Q/K normalization, partial RoPE and gate values plus DeltaNet projections, convolution, recurrence, chunks, outputs, and final state.
- [x] Chunked, split-prefill, and token-at-a-time outputs and final recurrent states agree within `7.450580596923828e-09`.

**How to check**
```sh
Scripts/test.sh --filter QwenFixtureDigestTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-3/fixture-generation.md`, `swift-fixture-tests.log`, and `lead-fixture-tests.log`.

#### 3.3 Generate MoE, greedy-text, and vision fixture sections

| | |
|---|---|
| Duty | `fixture-owner` |
| Touches | `Tests/TurboFieldfare/Core/QwenFixtures/qwen36-tiny-fixtures.json` (**PROPOSED**) |
| Depends on | `3.1` |
| Parallel safe | `no` |
| Deliverable | The same fixture adds router logits, Top-8 choices, shared gate, layer outputs, greedy tokens, image grids, pad rows, and M-RoPE positions. |

Include deterministic router ties and an axis-asymmetric image grid so swapped axes cannot pass.

**Acceptance detail**
- [x] Expected routed experts and weights, including the literal pinned CPU Top-8 tie `[0,1,2,3,4,5,6,7]`, are explicit and not recomputed by production logic.
- [x] Grid `[1,4,6]` distinguishes height and width in text M-RoPE; the actual axial vision oracle carries explicit block-major `(t,h,w)` rows through 24 tower outputs and six ordered `[2048]` merger rows.

**How to check**
```sh
Scripts/test.sh --filter QwenFixtureDigestTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-3/fixture-generation.md`, `negative-controls-run-2.json`, and `determinism-and-freeze.log`.

#### 3.4 Add fixture digest and schema tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/QwenFixtures/QwenFixtureDigestTests.swift` (**PROPOSED**) |
| Depends on | `3.2, 3.3` |
| Parallel safe | `yes` |
| Deliverable | Tests verify fixture schema, source identities, exact digest, finite tolerances, tensor shapes, and cross-section references. |

Do not import production Qwen kernels. This suite protects the oracle from silent regeneration.

**Acceptance detail**
- [x] The suite executes six Swift Testing cases with zero failures and zero skips.
- [x] Hostile byte, source-revision, and missing-intermediate mutations are rejected; the exact byte digest also rejects any changed frozen value.

**How to check**
```sh
Scripts/test.sh --filter QwenFixtureDigestTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-3/swift-fixture-tests.log` and `lead-fixture-tests.log`.

#### 3.5 Record generator environment and negative controls

| | |
|---|---|
| Duty | `fixture-review` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-3/fixture-generation.md` (**PROPOSED**) |
| Depends on | `3.4` |
| Parallel safe | `no` |
| Deliverable | Evidence records exact interpreter/packages, command, output digest, and negative controls for missing gate, wrong Top-8 normalization, FP16 decay, stale convolution history, and swapped M-RoPE axes. |

This task does not install dependencies. If the pinned environment is unavailable, record the block instead of generating substitute numbers.

**Acceptance detail**
- [x] Every negative control changes its named expected output.
- [x] No tolerance was widened; four deltas exceed their relevant declared absolute tolerance, while FP16 decay changes the FP32 recurrent-state oracle by about `6.20e-06` but remains within `3e-05`, so it is recorded as sensitivity rather than overclaimed as a future tolerance-test rejection.

**How to check**
```sh
Scripts/test.sh --filter QwenFixtureDigestTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-3/fixture-generation.md` and `negative-controls-run-2.json`.

### Phase 3 coverage plan

Every Phase 3 behavioral path has its own row. The generator and evidence are deliberately ignored under `/scratch`; their exact hashes and file inventory are recorded separately in `ignored-scratch-audit.log`.

| Code this phase changes | Test file that must cover it | What the test or evidence proves | Command |
|---|---|---|---|
| `Package.swift` | `Tests/TurboFieldfare/Core/QwenFixtures/QwenFixtureDigestTests.swift` | `Bundle.module` finds and reads the copied exact fixture resource. | `Scripts/test.sh --filter QwenFixtureDigestTests` |
| `scratch/qwen3.6-35b-a3b/fixture-generator/` | evidence-only | Two independent successful invocations are byte-identical and five controlled mutations change named outputs. | pinned offline generator command; see evidence |
| `Tests/TurboFieldfare/Core/QwenFixtures/qwen36-tiny-fixtures.json` | `Tests/TurboFieldfare/Core/QwenFixtures/QwenFixtureDigestTests.swift` | Exact byte count/digest, schema, revisions, tensors, tolerances, cross-references, Top-8 tie, vision ordering, and control records are checked. | `Scripts/test.sh --filter QwenFixtureDigestTests` |
| `Tests/TurboFieldfare/Core/QwenFixtures/QwenFixtureDigestTests.swift` | self | The independent suite executes with a nonzero count and zero failures/skips. | `Scripts/test.sh --filter QwenFixtureDigestTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-3/fixture-generation.md` | evidence-only | Commands, environment, first failed attempt, successful runs, controls, tests, and D4 are recorded without hiding limitations. | evidence review |

### Phase 3 evidence

<a id="phase-3-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-15 | Separately authorized repository-local reference-environment provisioning: exact Transformers checkout, isolated CPU dependency environment, source pin, package freeze, pip check, and offline Qwen class imports | PASS for the prerequisite environment; exit 0 for checkout, install, pip check, and smoke. No model, tensor operation, generation, GPU command, weight, or numerical fixture was used during setup. | `scratch/qwen3.6-35b-a3b/evidence/phase-3/reference-environment/setup-summary.md` and linked logs |
| 2026-09-15 | Complete preflights before both successful pinned offline CPU/eager reference invocations | PASS; exits 0, zero forbidden processes, optional kernels absent, source/venv pinned. | `reference-run-preflight-2.log`; `reference-run-preflight-3.log` |
| 2026-09-15 | Two serial tiny text and actual vision-tower/merger generations | PASS; two identical offline CPU/eager generations exited 0; both exact outputs SHA-256 `61999aed7cf049debf980f3482af7ae91b70dcf25a3fcebdd613255c0f824dca`. The actual vision oracle carries staged tower rows `[24,16]` and six ordered merger rows `[6,2048]`. The earlier path-resolution attempt exited 1 before import/construction and is retained. | `generator-run-1.log`; `generator-run-2.log`; `generator-run-3.log`; `determinism-and-freeze.log` |
| 2026-09-15 | Five named semantic controls | PASS for named-output-change criterion; no tolerance changes. Four deltas exceed tolerance; FP16-decay delta is within tolerance and is not overclaimed. | `negative-controls-run-2.json`; `fixture-generation.md` |
| 2026-09-15 | Independent and lead Debug fixture filters | PASS; each final run exit 0 with 6 Swift Testing cases, zero failures/skips. Initial independent test-source compile errors were corrected; its overwritten log limitation is disclosed. | `swift-fixture-tests.log`; `lead-fixture-tests.log`; `fixture-generation.md` |
| 2026-09-15 | Release filtered package build | Build PASS, exit 0; fixture copied. The Release filter printed no Swift Testing execution section, so this is compilation evidence only. Pre-existing unrelated warnings remain visible. | `release-fixture-tests.log` |
| 2026-09-15 | Project D4 and ignored scratch audit | PASS; 17-path union is 14 unchanged inherited paths plus exactly three Phase 3 repository paths. Ignored generator/evidence inputs are inventoried and hashed separately. | `d4-summary.log`; `ignored-scratch-audit.log` |
| 2026-09-15 | Candidate freeze | Phase 3 behavioral-input digest `b228211529a3999141432ad2c21d3589c517cab6419449bf6ea1e8be7b241d8f`. | `candidate-identity.txt`; `candidate-file-hashes.txt` |

Formal approvals: Terra #140 and Grok #142 approved verification #139 for the exact preapproval candidate all-artifact digest `0db70b9179c882f06e906e745703a9c0cae5f4f1b36c1524e06d71df607b7781`. Phase 4 is ready and not started.

---

## Phase 4 - Tiny BF16 ranges convert deterministically

**When this is done:** tiny BF16 ranges convert deterministically

Needs: `1`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-15 phase start — 61 GiB available on /System/Volumes/Data; memory_pressure reported 61% free; four pre-existing worktrees and 93 simulator-device records were preserved. The forbidden-process inventory matched only its own shell command text; no model process was started or stopped.`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-4/`

Phase-start identity: the 16 inherited uncommitted `Sources`, `Tests`, and `Package.swift` paths hashed to `9c7bbad30057c2d54d9c334a58e031c5ca55890ababff5bb87e70c65d560f4f9` before Phase 4 source creation. Full environment, worktree, process, safe-path, and simulator inventories are in `phase-start-environment.log` and `phase-start-simulator-inventory.log`.

### Current state

Phase 4 implementation and model-free checks are complete; Terra #135 and Grok #137 approved Phase 4 submission #133 in workflow `workflow-mac-engineering-complex-mu2tpf6b-ao3jc2`. The reviewed aggregate is `90e546dcba0a815043197554f2305a792a5e55f4096ab60150d51f3621b5bbde`; the postapproval document revision is identified separately by its three document hashes. The production policy, bounded group quantizer, and positional transform reader compile in Debug and Release. No official shard, model, GPU, or runtime fixture quantizer was used.

The approved provisional policy keeps each frozen P2 category uniform: embedding/output head/router/linear attention use affine INT8; attention/shared/routed experts use affine INT4; normalization remains BF16; recurrent state is an explicit non-tensor FP32 profile. Vision remains BF16 with nil category, and all 19 exact MTP tensors are omitted with nil category. This is a deterministic storage policy, not a real-model accuracy claim.

The numeric contract decodes little-endian BF16 exactly, rounds BF16 scale/bias and affine coordinates to nearest-even, uses unsigned codes, stores even INT4 values in the low nibble, canonicalizes zero, preserves BF16 subnormal decoding, rejects non-finite or unrepresentable arithmetic, and reports the observed plus conservative declared dequantization error. Groups are 1...64 elements; an odd INT4 tail leaves its unused high nibble zero.

The reader requires an explicit last-dimension row width, so row tails never merge. Ordered no-follow positional reads join arbitrary byte splits. The checkpoint binds source identity, row width, bit width, committed bytes/elements/groups and advances only after atomic sink success. Scratch payload is bounded by the configured tile plus 128 source-group bytes plus 640 fixed quantizer bytes, independent of tensor length; allocator bookkeeping and peak RSS were not measured.

### Tasks

#### 4.1 Define the official BF16 affine policy

| | |
|---|---|
| Duty | `quant-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Quantization/BF16AffineQuantizationPolicy.swift` |
| Depends on | `1.4` |
| Parallel safe | `yes` |
| Deliverable | A closed policy selects retained BF16, affine INT4, or affine INT8 by exact tensor group and records every override for v2 provenance. |

Do not infer policy from absent `quantization` config: the official config is BF16. Reject a tensor without an explicit policy match.

**Acceptance detail**
- [x] Every P1 tensor class has one explicit storage policy or intentional omission.
- [x] The policy digest changes when any group rule or override changes.

**How to check**
```sh
Scripts/test.sh --filter BF16AffineQuantizationPolicyTests
```

**Evidence:** Independent authorship and assessment at team events `#102` and `#105`; lead-executed full log `tester/sol-executed-final-policy-filter.log` records 4 tests in 1 suite, zero failures, exit 0.

#### 4.2 Implement bounded BF16 group quantization

| | |
|---|---|
| Duty | `quant-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Quantization/StreamingBF16AffineQuantizer.swift` |
| Depends on | `4.1` |
| Parallel safe | `yes` |
| Deliverable | The quantizer consumes bounded BF16 groups and emits packed values plus BF16 scale/bias using the policy arithmetic. |

Specify ties, signed zero, subnormals, constant ranges, tail groups, integer overflow, and NaN/infinity behavior. Never call the runtime fixture helper.

**Acceptance detail**
- [x] Identical groups produce identical bytes across repeated calls.
- [x] Scratch usage is fixed by configured tile/group sizes, not tensor length.

**How to check**
```sh
Scripts/test.sh --filter StreamingBF16AffineQuantizerTests
```

**Evidence:** Independent authorship and assessment at team events `#93` and `#105`; lead-executed full log `tester/sol-executed-final-quantizer-filter.log` records 5 tests in 1 suite, zero failures, exit 0.

#### 4.3 Add a tiled transform reader

| | |
|---|---|
| Duty | `quant-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Quantization/BF16AffineTransformReader.swift` |
| Depends on | `4.2` |
| Parallel safe | `no` |
| Deliverable | A reader joins shard byte ranges into complete quantization groups without unbounded buffering or lost tail elements. |

Use positional reads and checked arithmetic. Cancellation must leave the caller at a checkpointable group boundary.

**Acceptance detail**
- [x] Different legal tile sizes yield byte-identical output.
- [x] Short reads, misalignment, overflow, and mid-group cancellation fail explicitly.

**How to check**
```sh
Scripts/test.sh --filter BF16AffineTransformReaderTests
```

**Evidence:** The cross-row defect was found before final freeze at event `#111`; the independent 2x65 regression and resume matrix passed at `#119/#120`. Lead-executed full log `tester/sol-executed-final-reader-filter.log` records 5 tests in 1 suite, zero failures, exit 0. The independent fixture-error failure is preserved in `tester/sol-executed-reader-row-regression-attempt-1-failure.log`.

#### 4.4 Add policy and quantizer golden tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineQuantizationPolicyTests.swift`; `Tests/TurboFieldfareRepack/Core/Quantization/StreamingBF16AffineQuantizerTests.swift` |
| Depends on | `4.1` |
| Parallel safe | `yes` |
| Deliverable | Tests own hand-computed tiny vectors for INT4/INT8 packing, BF16 scale/bias, constant/tail groups, special floats, and policy overrides. |

Do not derive expected packed values by calling production quantization code.

**Acceptance detail**
- [x] Both suites execute with nonzero counts.
- [x] Golden bytes and the declared dequantization error are stable.

**How to check**
```sh
Scripts/test.sh --filter BF16Affine
```

**Evidence:** Terra independently authored and assessed the suites at events `#93`, `#102`, and `#105`. Lead-executed authoritative logs are `tester/sol-executed-final-{policy,quantizer}-filter.log`.

#### 4.5 Add transform boundary and cancellation tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineTransformReaderTests.swift` |
| Depends on | `4.3` |
| Parallel safe | `yes` |
| Deliverable | Tests split the same logical input at every byte/group boundary and inject short reads and cancellation. |

Use tiny in-memory or temporary-file data only; never read the official shards.

**Acceptance detail**
- [x] The suite executes with a nonzero count.
- [x] Clean and resumed tiny transforms have identical bytes, scales, biases, and terminal checkpoints.

**How to check**
```sh
Scripts/test.sh --filter BF16AffineTransformReaderTests
```

**Evidence:** Terra independently assessed the corrected row/tile/cancellation/resume matrix at `#119/#120`; lead-executed authoritative log is `tester/sol-executed-final-reader-filter.log`.

### Phase 4 coverage plan

Every Phase 4 source and test path has a direct behavioral row. The project-specific D4 union contains exactly these six Phase 4 paths plus 17 unchanged inherited P1-P3 paths.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Quantization/BF16AffineQuantizationPolicy.swift` | `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineQuantizationPolicyTests.swift` | All 1,045 exact P1 tensors have P2-legal storage/category metadata; state, vision, and MTP are explicit; a real canonical rule mutation changes provenance. | `Scripts/test.sh --filter BF16AffineQuantizationPolicyTests` |
| `Sources/TurboFieldfareRepack/Core/Quantization/StreamingBF16AffineQuantizer.swift` | `Tests/TurboFieldfareRepack/Core/Quantization/StreamingBF16AffineQuantizerTests.swift` | Hand-computed INT4/INT8 bytes, BF16 parameters, ties, zero, subnormal, constants, tails, determinism, error bound, and hostile failures. | `Scripts/test.sh --filter StreamingBF16AffineQuantizerTests` |
| `Sources/TurboFieldfareRepack/Core/Quantization/BF16AffineTransformReader.swift` | `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineTransformReaderTests.swift` | Discontiguous/every-byte tiles, 2x65 row tails, scratch payload bound, every accepted resume boundary, short reads, sink failure, overflow, invalid checkpoint, and cancellation. | `Scripts/test.sh --filter BF16AffineTransformReaderTests` |
| `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineQuantizationPolicyTests.swift` | self | The independently authored suite executes 4 tests with zero failures/skips. | `Scripts/test.sh --filter BF16AffineQuantizationPolicyTests` |
| `Tests/TurboFieldfareRepack/Core/Quantization/StreamingBF16AffineQuantizerTests.swift` | self | The independently authored suite executes 5 tests with zero failures/skips. | `Scripts/test.sh --filter StreamingBF16AffineQuantizerTests` |
| `Tests/TurboFieldfareRepack/Core/Quantization/BF16AffineTransformReaderTests.swift` | self | The independently authored suite executes 5 tests with zero failures/skips. | `Scripts/test.sh --filter BF16AffineTransformReaderTests` |

### Phase 4 evidence

<a id="phase-4-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-15 | Phase-start environment, disk, memory, process, worktree, simulator, and safe dirty-input identity | PASS; base `5770260510935cd32b82c4008ff06ae6458539ff`, 61 GiB free, 61% memory free, inherited safe-input digest `9c7bbad30057c2d54d9c334a58e031c5ca55890ababff5bb87e70c65d560f4f9`; no model action. | `phase-start-environment.log`; `phase-start-simulator-inventory.log` |
| 2026-09-15 | Baseline Debug and Release `TurboFieldfareRepackCore` builds | PASS, both exit 0. | `baseline-{debug,release}-repack-core.log` |
| 2026-09-15 | Final Debug and Release `TurboFieldfareRepackCore` builds | PASS, both exit 0 with no attributable diagnostics. | `lead/final-{debug,release}-repack-core.log` |
| 2026-09-15 | Three final named Phase 4 filters, authored/assessed independently and rerun by Sol for full repository logs | PASS; policy 4, quantizer 5, reader 5 tests; 3 suites; zero failures/skips; each exit 0. | `tester/sol-executed-final-{policy,quantizer,reader}-filter.log`; Terra events `#105/#120` |
| 2026-09-15 | Final focused P1 map, P2 format, range-plan, and checkpoint regressions | PASS; 26 tests in 4 suites, zero failures/skips, exit 0. | `lead/final-focused-regressions.log` |
| 2026-09-15 | Boundedness and row/cancellation proof | PASS for fixed payload contract: tile + 128-byte BF16 group + 640-byte quantizer payload; 2x65 rows emit 64/1/64/1 and resume at 64/65/129/130 without duplicates. Allocator overhead and peak RSS were not measured. | `tester/sol-executed-final-reader-filter.log`; Terra events `#109/#120` |
| 2026-09-15 | Preserved initial failures | Reader source first failed to compile due cancellation API spelling; independent row test first failed because its fixture used raw subnormal bits rather than BF16 1/2. Both were corrected without overwriting evidence. | `api-freeze-debug-repack-core.log`; `tester/sol-executed-reader-row-regression-attempt-1-failure.log` |
| 2026-09-15 | Project-specific four-stream D4 with `PATHS=(Sources Tests Package.swift)` and status guard | PASS; 23-path union = 17 inherited P1-P3 + exactly 6 Phase 4 paths; no unexplained Phase 4 extras. | `lead/final-d4-source-inventory.log` |
| 2026-09-15 | Behavioral candidate freeze | Six Phase 4 production/test inputs hash to `785a5b41706248f069ec66d161de323346e9be11d67c7ee19a2286c13bafd844`; the current recheck passed before the reviewed approval. | `lead/behavioral-file-hashes.txt`; `lead/behavioral-subdigest.txt` |
| 2026-09-15 | Model/GPU/performance qualification | N/A for tiny synthetic CPU Phase 4. No model, shard, GPU, download, conversion, or performance run; the storage allocation remains provisional. | This evidence table and phase-start inventory. |

Formal approvals: Terra #135 and Grok #137 approved Phase 4 submission #133 in workflow `workflow-mac-engineering-complex-mu2tpf6b-ao3jc2` for reviewed aggregate `90e546dcba0a815043197554f2305a792a5e55f4096ab60150d51f3621b5bbde`. The approval covers the reviewed code/evidence candidate, not the later document-only revision. Nonblocking review notes: no direct Int32-overflow test and no direct policy-through-v2-validator call were added.

---

## Phase 5 - A local official snapshot validates offline

**When this is done:** a local official snapshot validates offline

Needs: `1`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-15T16:32:41Z — 61 GiB available on /System/Volumes/Data; macOS 26.6.2, Xcode 26.5, Swift 6.3.2, arm64 Apple M2 Pro, 32 GB RAM; no prohibited model processes. Existing worktrees were preserved. Full capture: `scratch/qwen3.6-35b-a3b/evidence/phase-5/phase-start.txt`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-5/`
Phase-start identity: the baseline was captured before Phase 5 source edits; HEAD `5770260510935cd32b82c4008ff06ae6458539ff`; dirty-input digest `3e6aab80ffe432b61c1bf8613434642e347e32b7624dffdada2258936dded8e5`.
Final candidate: eight-path digest `f834885a94cd5db9e766d7375d8f1c87086df28218df8064719c7d03daaccb1a`; evidence manifest digest `69d206f711ccd5f2b6e92f372e3cd402209e5d69674ebcae31556a775505cce3`; code submission #51, formal Grok #53, full verdict #57 in workflow `workflow-mac-engineering-standard-mu2vzbyt-kr2s6s`.

### Current state

The final candidate adds a closed source catalog and no-follow local snapshot validation. `IndexLoader.swift` was inspected and is unchanged in the final candidate. `Package.swift` is unchanged relative to the Phase 5 start, but remains inherited dirty from Phase 3. The canonical audit validates 26 shards, 1,045 BF16 tensors, and 19 explicit MTP omissions without payload reads.

### Target state

A closed source catalog retains Gemma values and adds the exact official Qwen source. A local loader validates regular files without following leaf symlinks, bounded JSON/header reads, sidecar digests, 26 referenced shards, BF16 dtypes, range/shape arithmetic, and exact index/header agreement. It does not download or quantize.

### Tasks

#### 5.1 Replace the singleton with a closed source catalog

| | |
|---|---|
| Duty | `ingest-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Remote/SupportedModelSource.swift`; `Sources/TurboFieldfareRepack/Core/Remote/ModelSourceCatalog.swift` |
| Depends on | `1.4` |
| Parallel safe | `no` |
| Deliverable | Catalog lookup uses stable product identity and preserves the existing Gemma repository, revision, files, and estimates exactly. |

Qwen points only to official pinned metadata and local-source validation; do not add community quantization as a release source.

**Acceptance detail**
- [x] Gemma catalog values compare equal to their pre-change values.
- [x] Only the exact Qwen repository/revision resolves to the Qwen entry.

**How to check**
```sh
Scripts/test.sh --filter ModelSourceCatalogTests
```

**Evidence:** PASS; 8 tests in 1 suite, zero failures, exit 0. `final-model-source-catalog-tests.log`. Code approval: submission #51; formal Grok #53/full verdict #57.

#### 5.2 Implement no-follow local snapshot loading

| | |
|---|---|
| Duty | `ingest-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift` |
| Depends on | `5.1` |
| Parallel safe | `yes` |
| Deliverable | The loader opens sidecars and shard headers from a caller-provided directory using bounded, no-follow file access. |

Validate the index against P1 identity and tensor map. The missing quantization slot in official `config.json` must not trigger a Gemma/MLX fallback.

**Acceptance detail**
- [x] A valid synthetic local snapshot yields one immutable metadata object.
- [x] Symlinks, special files, missing sidecars, extra shards, and changed digests fail.

**How to check**
```sh
Scripts/test.sh --filter LocalPinnedSnapshotLoaderTests
```

**Evidence:** PASS; 32 tests in 1 suite, zero failures, exit 0. The separate 1-test canonical audit is included in the 32-test count and is not double-counted. `final-local-pinned-snapshot-loader-tests.log`; `final-canonical-snapshot-audit.log`.

#### 5.3 Audit local index and header agreement

| | |
|---|---|
| Duty | `ingest-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Format/IndexLoader.swift`; `Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift` |
| Depends on | `5.2` |
| Parallel safe | `no` |
| Deliverable | Extend local parsing to require exact tensor names, BF16 dtype, shape-derived byte count, non-overlap, in-file bounds, and referenced shard membership. |

Keep remote Gemma parsing behavior unchanged. Use checked arithmetic already established in hostile header parsing.

**Acceptance detail**
- [x] Index/header name, shape, dtype, and shard assignments agree exactly.
- [x] A short header or one-byte range disagreement is reported before planning.

**How to check**
```sh
Scripts/test.sh --filter LocalPinnedSnapshotLoaderTests
```

**Evidence:** PASS; local-loader and hostile-header filters passed. `IndexLoader.swift` was inspected and unchanged. `final-local-pinned-snapshot-loader-tests.log`; `final-safetensors-hostile-header-tests.log`.

#### 5.4 Add catalog and local-loader tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Remote/ModelSourceCatalogTests.swift`; `Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift` |
| Depends on | `5.1` |
| Parallel safe | `yes` |
| Deliverable | Tests build tiny local safetensors containers and mutate identity, file type, index, headers, dtype, shapes, ranges, and shard membership. |

Test paths contain only tiny generated data. They must not reference the full official shard payloads.

**Acceptance detail**
- [x] Both suites execute with nonzero counts.
- [x] Every rejection occurs before creating a destination or network session.

**How to check**
```sh
Scripts/test.sh --filter LocalPinnedSnapshotLoaderTests
```

**Evidence:** PASS; catalog 8 tests and local loader 32 tests, each in 1 suite, zero failures, exit 0. The canonical audit is included in the local-loader count. `final-model-source-catalog-tests.log`; `final-local-pinned-snapshot-loader-tests.log`; `final-canonical-snapshot-audit.log`.

#### 5.5 Prove existing remote Gemma loading is unchanged

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Remote/RemotePayloadCopyTests+Installation.swift` |
| Depends on | `5.1, 5.3, 5.4` |
| Parallel safe | `no` |
| Deliverable | Re-run the existing remote installation test after the catalog handoff and add no Qwen expectation to it. |

This task owns only the existing test file if a catalog assertion is necessary.

**Acceptance detail**
- [x] The existing remote Gemma source still plans and copies through its current path.
- [x] No local Qwen choice changes Gemma authentication, retry, or URL rules.

**How to check**
```sh
Scripts/test.sh --filter RemotePayloadCopyTests
```

**Evidence:** PASS; 33 tests in 3 suites, zero failures, exit 0. `final-remote-payload-copy-tests.log`.

### Phase 5 coverage plan

Every intended file has its own row. The final rows name the executed suites and counts; `IndexLoader.swift` is retained as an inspected, unchanged Phase 5 boundary, while `Package.swift` remains inherited from earlier phases.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Remote/SupportedModelSource.swift` | `Tests/TurboFieldfareRepack/Core/Remote/ModelSourceCatalogTests.swift` | Gemma values remain exact and only the pinned official Qwen identity resolves. | `Scripts/test.sh --filter ModelSourceCatalogTests` |
| `Sources/TurboFieldfareRepack/Core/Remote/ModelSourceCatalog.swift` | `Tests/TurboFieldfareRepack/Core/Remote/ModelSourceCatalogTests.swift` | Stable catalog identity lookup and absence of remote Qwen download fields. | `Scripts/test.sh --filter ModelSourceCatalogTests` |
| `Sources/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoader.swift` | `Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift` | No-follow local sidecars/shards, digests, identity, ranges, dtypes, shapes, mappings, and no-output-on-failure validation. | `Scripts/test.sh --filter LocalPinnedSnapshotLoaderTests` |
| `Sources/TurboFieldfareRepack/Core/Format/IndexLoader.swift` | `Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift` | Exact index/header agreement is exercised; the source file is inspected and unchanged in the final candidate. | `Scripts/test.sh --filter LocalPinnedSnapshotLoaderTests` |
| `Sources/TurboFieldfareRepack/Core/Format/Safetensors.swift` | `Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift`; `Tests/TurboFieldfareRepack/Core/Format/SafetensorsHostileHeaderTests.swift` | Local validation plus hostile-header rejection; 32 local tests and 8 hostile-header tests, with the canonical audit included in the local count. | `Scripts/test.sh --filter LocalPinnedSnapshotLoaderTests\|SafetensorsHostileHeaderTests` |
| `Tests/TurboFieldfareRepack/Core/Remote/ModelSourceCatalogTests.swift` | self | 8 tests in 1 suite, zero failures. | `Scripts/test.sh --filter ModelSourceCatalogTests` |
| `Tests/TurboFieldfareRepack/Core/Local/LocalPinnedSnapshotLoaderTests.swift` | self | 32 tests in 1 suite, zero failures; canonical audit included. | `Scripts/test.sh --filter LocalPinnedSnapshotLoaderTests` |
| `Tests/TurboFieldfareRepack/Core/Remote/RemotePayloadCopyTests+Installation.swift` | self | 33 tests in 3 suites, zero failures; existing remote Gemma behavior remains covered. | `Scripts/test.sh --filter RemotePayloadCopyTests` |

### Phase 5 evidence

<a id="phase-5-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-15 | Final candidate identity and approval | PASS; submission #51, formal Grok #53, and full verdict #57 in workflow `workflow-mac-engineering-standard-mu2vzbyt-kr2s6s`. Exact eight-path candidate digest `f834885a94cd5db9e766d7375d8f1c87086df28218df8064719c7d03daaccb1a`; evidence manifest digest `69d206f711ccd5f2b6e92f372e3cd402209e5d69674ebcae31556a775505cce3`. Approval covers the eight P5 source/test inputs and specified evidence, not this later document revision. | `final-candidate-p5-manifest.txt`; `final-results.txt` |
| 2026-09-15 | Model source catalog filter | PASS; 8 tests in 1 suite, zero failures, exit 0. | `final-model-source-catalog-tests.log` |
| 2026-09-15 | Local pinned snapshot loader filter | PASS; 32 tests in 1 suite, zero failures, exit 0. The separate 1-test canonical audit is included in this 32-test count and is not double-counted. | `final-local-pinned-snapshot-loader-tests.log`; `final-canonical-snapshot-audit.log` |
| 2026-09-15 | Hostile safetensors header filter | PASS; 8 tests in 1 suite, zero failures, exit 0. | `final-safetensors-hostile-header-tests.log` |
| 2026-09-15 | Existing remote Gemma filter | PASS; 33 tests in 3 suites, zero failures, exit 0. | `final-remote-payload-copy-tests.log` |
| 2026-09-15 | Debug and Release `TurboFieldfareRepackCore` builds | PASS; both builds completed, exit 0. | `final-build-debug-repack-core.log`; `final-build-release-repack-core.log` |
| 2026-09-15 | Canonical snapshot structural audit | PASS; 26 shards, 1,045 BF16 tensors, and 19 MTP omissions; header/index agreement and sidecar SHA checks only. This does not prove payload integrity, model behavior, GPU behavior, or performance. | `final-canonical-snapshot-audit.log`; `final-candidate-p5-manifest.txt` |
| 2026-09-15 | Project-specific four-stream D4 and status guard | PASS; 30-path union at base `5770260510935cd32b82c4008ff06ae6458539ff` = 23 inherited paths plus 7 changed P5 paths. `IndexLoader.swift` is absent because it is unchanged; `Package.swift` is inherited dirty from an earlier phase. | `documentation/final-d4-source-inventory.log` |
| 2026-09-15 | Preserved initial malformed-index failure | The initial malformed-index rejection surfaced a Foundation JSON error; the final candidate maps it to the expected `RepackError.indexJsonInvalid`. | `initial-malformed-index-failure.txt` |

---

## Phase 6 - A complete Qwen pack is sized before writing

**When this is done:** a complete Qwen pack is sized before writing

Needs: `2, 4, 5`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-15T17:24:39Z — 61 GiB available on /System/Volumes/Data; memory free 71%; macOS 26.6.2, Xcode 26.5/SDK 26.5, Swift 6.3.2, SwiftPM tools 6.2, arm64 Apple M2 Pro, 32 GiB RAM; no prohibited model processes. Existing worktrees were preserved. Full capture: `scratch/qwen3.6-35b-a3b/evidence/phase-6/phase-start.txt`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-6/`
Phase-start identity: the baseline was captured before Phase 6 source edits; HEAD `5770260510935cd32b82c4008ff06ae6458539ff`; dirty-input manifest `ce0ecff7d5b820942a18594c83c7dd3d5ae501b79d299b2c339407f92244adc4`.
Final candidate: digest `ccf5c9dabd91059587e1a8c27d91508a1c4d76adbd7ca81975ccff61fb9310bb`; four-file manifest `008c0357f7d3a22dec4bbc478a6e9a3d68fce153c24506f717eacc1692e0526c`; evidence manifest `cd15e467c8d29c9464f6bceaf765993deef506011116863375a5f8807224db2e`. Verification #93 was approved by Grok #95/full verdict #98 and Terra #96 in workflow `workflow-mac-engineering-complex-mu2xntc9-xamj3h`.

### Current state

The final planner candidate requires the official Qwen architecture, assigns the validated tensor categories, and dispatches through the existing planner without changing Gemma semantics. The corrected INT4 `[2,65]` layout records 66 values (not 65), scales at offset 66 with size 8, bias at offset 74 with size 8, and total size 82. This is planner-only evidence: writer, conversion, model, GPU, and performance gates are N/A; source payload integrity remains unverified.

### Target state

A Qwen planner consumes the validated local snapshot, v2 schema, and explicit quantization policy. It assigns all 1,045 tensors, plans resident/streamed/vision regions, records 19 MTP omissions, computes aligned destination and scratch sizes before writing, and emits a stable plan fingerprint.

### Tasks

#### 6.1 Decode exact Qwen architecture fields for planning

| | |
|---|---|
| Duty | `planner-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Format/ArchInfo.swift` |
| Depends on | `2.6, 4.5, 5.5` |
| Parallel safe | `yes` |
| Deliverable | `ArchInfo` gains a family branch that requires official Qwen fields without Gemma defaults. |

Reject a wrong 40-layer schedule, hidden/head dimensions, convolution width, state dtype, expert counts, untied head, or vision config.

**Acceptance detail**
- [x] The official metadata yields the exact architecture contract.
- [x] Deleting any required Qwen field fails instead of substituting Gemma values.

**How to check**
```sh
Scripts/test.sh --filter QwenRepackPlannerTests
```

**Evidence:** PASS; `QwenRepackPlannerTests` ran 6 tests in 1 suite with 0 failed and 0 skipped, exit 0. `qwen-planner-tests-v2.log`.

#### 6.2 Plan resident and hybrid-state tensors

| | |
|---|---|
| Duty | `planner-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Planning/QwenRepackPlanner.swift` |
| Depends on | `6.1` |
| Parallel safe | `yes` |
| Deliverable | The Qwen planner places embeddings, untied LM head, norms, full-attention weights, linear-attention weights, recurrent parameters, routers, and shared experts. |

Use execution position and aligned transformed sizes from policy; do not copy source BF16 sizes into destination metadata.

**Acceptance detail**
- [x] Every text-resident source tensor appears once in the plan.
- [x] Ten full and 30 linear layers appear in validated execution order.

**How to check**
```sh
Scripts/test.sh --filter QwenRepackPlannerTests
```

**Evidence:** PASS; the final planner filter passed 6 tests in 1 suite. INT4 `[2,65]` sizing is 66 values, scales offset 66/size 8, bias offset 74/size 8, total 82. `qwen-planner-tests-v2.log`.

#### 6.3 Plan routed experts and optional vision companion

| | |
|---|---|
| Duty | `planner-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Planning/QwenRepackPlanner.swift` |
| Depends on | `6.1` |
| Parallel safe | `no` |
| Deliverable | Add one complete packed expert blob per layer and a separate model-bound vision companion plan. |

Preserve source `gate_up_proj` packing unless an implemented kernel contract requires a split. Compute expert stride and companion sizes rather than importing Gemma constants.

**Acceptance detail**
- [x] All 256 experts per layer have equal validated destination stride.
- [x] Vision tensors appear only in the companion plan and MTP only in omissions.

**How to check**
```sh
Scripts/test.sh --filter QwenRepackPlannerTests
```

**Evidence:** PASS; the final planner filter passed 6 tests in 1 suite with 0 failed and 0 skipped, exit 0. `qwen-planner-tests-v2.log`.

#### 6.4 Integrate family dispatch into the existing planner

| | |
|---|---|
| Duty | `planner-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift` |
| Depends on | `6.2, 6.3` |
| Parallel safe | `no` |
| Deliverable | Route verified Qwen metadata to `QwenRepackPlanner` and retain the existing Gemma planning branch unchanged. |

The caller receives complete required bytes, scratch bytes, artifact files, tensor records, omissions, and fingerprint before any writer is opened.

**Acceptance detail**
- [x] Qwen planning is deterministic across absolute output roots.
- [x] Gemma range-plan fingerprints retain current behavior.

**How to check**
```sh
Scripts/test.sh --filter QwenRepackPlannerTests
```

**Evidence:** PASS; the planner filter passed 6 tests, affected P2/P4/P5 regressions passed 72 tests in 6 suites, and diff-check passed. `qwen-planner-tests-v2.log`; `affected-regressions.log`; `diff-check-v2.log`.

#### 6.5 Add exhaustive tiny Qwen planning tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` |
| Depends on | `6.1` |
| Parallel safe | `yes` |
| Deliverable | Tests use synthetic headers covering all categories and mutate one name, dtype, shape, policy, revision, digest, expert count, layer type, and MTP omission. |

Assert exact planned offsets, aligned sizes, stride, output files, and fingerprint on a tiny plan.

**Acceptance detail**
- [x] The suite executes with a nonzero count.
- [x] No source tensor is unaccounted for or classified twice.

**How to check**
```sh
Scripts/test.sh --filter QwenRepackPlannerTests
```

**Evidence:** PASS; 6 tests in 1 suite, 0 failed and 0 skipped, exit 0. `qwen-planner-tests-v2.log`.

#### 6.6 Re-run range-plan compatibility tests

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Planning/RangeCopyPlannerTests.swift` |
| Depends on | `6.4, 6.5` |
| Parallel safe | `no` |
| Deliverable | Preserve existing absolute-root-independent fingerprints and overlap/path rejection tests. |

Change this file only if the family dispatch adds a new explicit assertion; do not rewrite Gemma expected plans.

**Acceptance detail**
- [x] The existing suite executes with a nonzero count.
- [x] Gemma plan semantics do not change.

**How to check**
```sh
Scripts/test.sh --filter RangeCopyPlannerTests
```

**Evidence:** PASS; `RangeCopyPlannerTests` ran 4 tests in 1 suite with 0 failed and 0 skipped, exit 0. The existing test file was inspected and unchanged. `range-copy-tests.log`; `affected-regressions.log`.

### Phase 6 coverage plan

Every intended file has its own row. The final rows name the executed planner or compatibility suites; `RangeCopyPlannerTests.swift` is retained as an inspected, unchanged boundary.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Format/ArchInfo.swift` | `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` | Required Qwen architecture fields reject missing or mutated values instead of using Gemma defaults. | `Scripts/test.sh --filter QwenRepackPlannerTests` |
| `Sources/TurboFieldfareRepack/Core/Planning/QwenRepackPlanner.swift` | `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` | All tensor categories, hybrid layer order, aligned sizes, expert stride, omissions, and fingerprint are checked. | `Scripts/test.sh --filter QwenRepackPlannerTests` |
| `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift` | `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` | Verified Qwen dispatch and unchanged Gemma planning semantics are covered by the planner suite. | `Scripts/test.sh --filter QwenRepackPlannerTests` |
| `Tests/TurboFieldfareRepack/Core/Planning/QwenRepackPlannerTests.swift` | self | 6 tests in 1 suite, zero failures/skips. | `Scripts/test.sh --filter QwenRepackPlannerTests` |
| `Tests/TurboFieldfareRepack/Core/Planning/RangeCopyPlannerTests.swift` | self | 4 tests in 1 suite, zero failures/skips; existing file inspected and unchanged. | `Scripts/test.sh --filter RangeCopyPlannerTests` |

### Phase 6 evidence

<a id="phase-6-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-15 | Final candidate identity and approvals | PASS; candidate digest `ccf5c9dabd91059587e1a8c27d91508a1c4d76adbd7ca81975ccff61fb9310bb`, four-file manifest `008c0357f7d3a22dec4bbc478a6e9a3d68fce153c24506f717eacc1692e0526c`, evidence manifest `cd15e467c8d29c9464f6bceaf765993deef506011116863375a5f8807224db2e`. Verification #93 was approved by Grok #95/full verdict #98 and Terra #96 in workflow `workflow-mac-engineering-complex-mu2xntc9-xamj3h`. | `candidate-identity.txt`; `candidate-files.sha256`; `evidence-manifest.sha256` |
| 2026-09-15 | Phase-start environment and baseline | PASS; base `5770260510935cd32b82c4008ff06ae6458539ff`, dirty-input manifest `ce0ecff7d5b820942a18594c83c7dd3d5ae501b79d299b2c339407f92244adc4`, 61 GiB free, 71% memory free, no prohibited model processes. | `phase-start.txt` |
| 2026-09-15 | Qwen planner filter | PASS; 6 tests in 1 suite, 0 failed, 0 skipped, exit 0. | `qwen-planner-tests-v2.log` |
| 2026-09-15 | Range-plan compatibility filter | PASS; 4 tests in 1 suite, 0 failed, 0 skipped, exit 0. `RangeCopyPlannerTests.swift` was inspected and unchanged. | `range-copy-tests.log` |
| 2026-09-15 | Affected P2/P4/P5 regressions | PASS; 72 tests in 6 suites, 0 failed, exit 0. Canonical local snapshot validation read verified metadata only and no payload. | `affected-regressions.log` |
| 2026-09-15 | Debug and Release `TurboFieldfareRepackCore` builds | PASS; both builds completed, exit 0. | `debug-repack-core-build.log`; `release-repack-core-build.log` |
| 2026-09-15 | Diff check | PASS; no candidate source/test input changed after the final evidence freeze. | `diff-check-v2.log` |
| 2026-09-15 | INT4 layout sizing correction | PASS; `[2,65]` records 66 values, scales offset 66/size 8, bias offset 74/size 8, total 82; the prior 65-value interpretation is rejected. | `qwen-planner-tests-v2.log`; `verification-summary.txt` |
| 2026-09-15 | Project-specific four-stream D4 and status guard | PASS; 34-path union = 30 inherited paths plus 4 changed P6 paths; status guard matched the union and missing count was 0. `RangeCopyPlannerTests.swift` was inspected unchanged. The initial D4 harness failure was repaired without code or evidence changes. | `d4-four-stream.txt`; `d4-four-stream.exit.txt`; `d4-four-stream-initial-failure.txt` |
| 2026-09-15 | Nonblocking review limitations | Production `snapshotDirectory` planning entry was not executed. `layout.json` sizing uses `JSONSerialization`; it is not proof of writer or v2 publication. Writer, conversion, model, GPU, and performance qualification remain N/A; source payload remains unverified. | `verification-summary.txt` |

---

## Phase 7 - A tiny transformed pack resumes byte-identically

**When this is done:** a tiny transformed pack resumes byte-identically

Needs: `2, 4, 6`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-15T18:19:37Z — 61 GiB available on /System/Volumes/Data; memory free 70%; macOS 26.6.2, Xcode 26.5/SDK 26.5, Swift 6.3.2, SwiftPM tools 6.2, arm64 Apple M2 Pro, 32 GiB RAM; no prohibited model processes. Full capture: `scratch/qwen3.6-35b-a3b/evidence/phase-7/phase-start-environment.txt`.
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-7/source-integrity-correction/`
Phase-start identity: baseline HEAD `5770260510935cd32b82c4008ff06ae6458539ff`; the historical unsafe candidate digest was `50058fffdf2ad96799f023eb5acb7964822cdd83915bd0be3faac89faaeeea42`.
Final corrected candidate: subdigest `eeb809a3534feadc426473ef241c5a3eb680e441a7c2edb25eeae1af8466a1f0`; evidence manifest `4ccc1f9ae7827ee955c6527bd59c529b2e792dc8e8d8ebb6b7812ea309d8e89f`. Verification #57 was approved by Terra #59 and Grok #62/full PASS #65; implementation #64 completed correction workflow `workflow-mac-engineering-complex-mu3206zw-bbigj4`.

### Current state

The corrected candidate replaces per-group transform ledgers with one O(1)-in-group-count cursor and source/destination rolling chains. Read-only exact-prefix replay validates consumed bytes before any payload read-write open and again before publication; old or missing transform progress refuses. Gemma schema-1 range-copy fields and matching remain unchanged. Production routed I/O is exercised on tiny test data, not authentic Qwen payload qualification or a published synthetic Qwen pack.

Historical result: the initial unsafe candidate `50058fffdf2ad96799f023eb5acb7964822cdd83915bd0be3faac89faaeeea42` was approved in `workflow-mac-engineering-complex-mu2zrf1l-mmia1n`; Main found routed mutation defect #116. The closed workflow refused reopening; correction workflow `workflow-mac-engineering-complex-mu3206zw-bbigj4` supersedes it. The baseline negative control remains preserved.

### Target state

A transform-aware writer consumes P6 plans, checkpoints complete quantization groups, preflights full destination plus scratch/headroom, resumes without changing bytes, audits every declared region and digest, and publishes only a complete tiny synthetic v2 pack.

### Tasks

#### 7.1 Add transformed range output to WriterCore

| | |
|---|---|
| Duty | `writer-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Writing/WriterCore.swift`; `Sources/TurboFieldfareRepack/Core/Writing/TransformedTensorWriter.swift` |
| Depends on | `2.6, 4.5, 6.6` |
| Parallel safe | `no` |
| Deliverable | `WriterCore` accepts a transform source that writes packed values and scale/bias regions at checked offsets while retaining raw-copy behavior. |

Keep the existing 512 KiB tile path for Gemma. Qwen commits checkpoints only after a complete deterministic group range reaches disk.

**Acceptance detail**
- [x] Tiny transformed regions match P4 golden bytes.
- [x] Short read, write failure, or cancellation never marks an incomplete group complete.

**How to check**
```sh
Scripts/test.sh --filter TransformedTensorWriterTests
```

**Evidence:** PASS; `TransformedTensorWriterTests` ran 3 tests in 1 suite, exit 0. The corrected cursor commits only after values, scale/bias, and sync complete. `test-TransformedTensorWriterTests-final.log`.

#### 7.2 Route Qwen resident writing through the transform

| | |
|---|---|
| Duty | `writer-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Writing/ResidentWriter.swift` |
| Depends on | `7.1` |
| Parallel safe | `no` |
| Deliverable | `ResidentWriter` dispatches Qwen planned regions to the transform writer and keeps its existing Gemma raw-copy branch. |

Retain bounded scratch allocation and checked offsets; no full tensor may be materialized in memory.

**Acceptance detail**
- [x] Resident metadata matches actual transformed region sizes.
- [x] Gemma resident byte copying is unchanged.

**How to check**
```sh
Scripts/test.sh --filter QwenTransformResumeTests
```

**Evidence:** PASS; `QwenTransformResumeTests` ran 8 tests in 1 suite, exit 0, including corrected routed production-path tiny-input restore-before-write behavior. Gemma/remote compatibility ran 33 tests across 3 suites. `test-QwenTransformResumeTests-final.log`; `test-RemotePayloadCopyTests-final.log`.

#### 7.3 Bind checkpoints to source, policy, and plan digests

| | |
|---|---|
| Duty | `writer-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Remote/RemoteInstallCheckpoint.swift` |
| Depends on | `7.1` |
| Parallel safe | `yes` |
| Deliverable | Checkpoint identity includes official source digest, converter version, quantization-policy digest, v2 plan fingerprint, and completed transformed group ranges. |

A changed source byte, policy, plan, or destination invalidates only owned resumable output and never overwrites a completed artifact.

**Acceptance detail**
- [x] A matching interruption resumes at the next complete group.
- [x] Any fingerprint component change refuses resume before writing.

**How to check**
```sh
Scripts/test.sh --filter RemoteInstallCheckpointTests
```

**Evidence:** PASS; `RemoteInstallCheckpointTests` ran 6 tests in 1 suite, exit 0. Old or missing transform progress refuses; compact cursor and domain-separated chains bind source, policy, plan, and destination state. `test-RemoteInstallCheckpointTests-final.log`.

#### 7.4 Preflight and audit transformed artifacts

| | |
|---|---|
| Duty | `writer-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/System/DiskSpaceChecker.swift`; `Sources/TurboFieldfareRepack/Core/Verification/RepackAudit.swift` |
| Depends on | `7.1, 7.3` |
| Parallel safe | `no` |
| Deliverable | Preflight uses P6 destination/scratch requirements and protected headroom; audit covers every v2 tensor region, omission, file size, and digest before activation. |

Do not reuse the historic free-space number. Measure available bytes at the authorized run and fail before output growth when insufficient.

**Acceptance detail**
- [x] Insufficient capacity creates no destination payload.
- [x] Audit rejects missing, extra, overlapping, corrupt, or unaudited regions.

**How to check**
```sh
Scripts/test.sh --filter QwenTransformResumeTests
Scripts/test.sh --filter DiskSpaceCheckerTests
```

**Evidence:** PASS; `DiskSpaceCheckerTests` ran 3 tests in 1 suite and Qwen resume/audit coverage ran 8 tests in 1 suite, all exit 0. Prefix replay runs before payload read-write access and before publication. `test-DiskSpaceCheckerTests-final.log`; `test-QwenTransformResumeTests-final.log`.

#### 7.5 Add resume, preflight, and audit tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift`; `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` |
| Depends on | `7.1` |
| Parallel safe | `yes` |
| Deliverable | Tests inject cancellation after each tiny group, short reads/writes, changed fingerprints, insufficient capacity, and audit corruption. |

Compare clean and resumed directory digests. Use temporary tiny inputs only.

**Acceptance detail**
- [x] Both suites execute with nonzero counts.
- [x] Every interruption point converges to the clean digest or a fail-closed partial.

**How to check**
```sh
Scripts/test.sh --filter TransformedTensorWriterTests
Scripts/test.sh --filter QwenTransform
```

**Evidence:** PASS; transformed writer 3/3 and Qwen resume 8/8. The overlapping `QwenTransform` filter is 8/8 and is not additive. The preserved unsafe baseline negative control ran 1 test with 2 issues and exit 1 before correction. `test-TransformedTensorWriterTests-final.log`; `test-QwenTransformResumeTests-final.log`; `baseline-negative-control.log`.

#### 7.6 Re-run existing resume and disk-space suites

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Remote/RemoteInstallCheckpointTests.swift`; `Tests/TurboFieldfareRepack/Core/System/DiskSpaceCheckerTests.swift` |
| Depends on | `7.2, 7.3, 7.4, 7.5` |
| Parallel safe | `no` |
| Deliverable | Existing compact checkpoint, overflow, damaged range, and authoritative disk assessment behavior remains intact. |

This owner changes only existing test assertions required by the generalized types.

**Acceptance detail**
- [x] Both existing suites execute with nonzero counts.
- [x] Gemma cancellation/resume and disk-space behavior does not regress.

**How to check**
```sh
Scripts/test.sh --filter RemoteInstallCheckpointTests
Scripts/test.sh --filter DiskSpaceCheckerTests
```

**Evidence:** PASS; checkpoint 6/6, disk 3/3, and Gemma/remote 33/33 across 3 suites. Release product retains an inherited unused-result warning at `Command/main.swift:273`; no correction-file diagnostic was introduced. `test-RemoteInstallCheckpointTests-final.log`; `test-DiskSpaceCheckerTests-final.log`; `test-RemotePayloadCopyTests-final.log`.

### Phase 7 coverage plan

Every intended file has its own row. The final rows name the executed transformed-writer, Qwen-resume, checkpoint, and disk suites. The overlapping `QwenTransform` filter is evidence only and is not double-counted.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Core/Writing/WriterCore.swift` | `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | Transformed bytes, group commit ordering, short writes, and cancellation. | `Scripts/test.sh --filter TransformedTensorWriterTests` |
| `Sources/TurboFieldfareRepack/Core/Writing/TransformedTensorWriter.swift` | `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | Tiny transformed output and incomplete-group refusal. | `Scripts/test.sh --filter TransformedTensorWriterTests` |
| `Sources/TurboFieldfareRepack/Core/Writing/ResidentWriter.swift` | `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | Routed Qwen transform path and unchanged Gemma branch. | `Scripts/test.sh --filter QwenTransformResumeTests` |
| `Sources/TurboFieldfareRepack/Core/Remote/RemoteInstallCheckpoint.swift` | `Tests/TurboFieldfareRepack/Core/Remote/RemoteInstallCheckpointTests.swift` | Compact cursor identity, legacy refusal, and matching resume. | `Scripts/test.sh --filter RemoteInstallCheckpointTests` |
| `Sources/TurboFieldfareRepack/Core/System/DiskSpaceChecker.swift` | `Tests/TurboFieldfareRepack/Core/System/DiskSpaceCheckerTests.swift` | No-destination capacity failure and preserved disk policy. | `Scripts/test.sh --filter DiskSpaceCheckerTests` |
| `Sources/TurboFieldfareRepack/Core/Verification/RepackAudit.swift` | `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | Missing, extra, corrupt, and unaudited transformed-region refusal. | `Scripts/test.sh --filter QwenTransformResumeTests` |
| `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | self | 3 tests in 1 suite, zero failures. | `Scripts/test.sh --filter TransformedTensorWriterTests` |
| `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | self | 8 tests in 1 suite, zero failures; routed restore-before-write coverage included. | `Scripts/test.sh --filter QwenTransformResumeTests` |
| `Tests/TurboFieldfareRepack/Core/Remote/RemoteInstallCheckpointTests.swift` | self | 6 tests in 1 suite, zero failures. | `Scripts/test.sh --filter RemoteInstallCheckpointTests` |
| `Tests/TurboFieldfareRepack/Core/System/DiskSpaceCheckerTests.swift` | self | 3 tests in 1 suite, zero failures. | `Scripts/test.sh --filter DiskSpaceCheckerTests` |

### Phase 7 evidence

<a id="phase-7-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-15 | Historical unsafe candidate and corrected approval | The initial candidate `50058fffdf2ad96799f023eb5acb7964822cdd83915bd0be3faac89faaeeea42` was approved in workflow `workflow-mac-engineering-complex-mu2zrf1l-mmia1n`; Main found routed mutation defect #116. Correction workflow `workflow-mac-engineering-complex-mu3206zw-bbigj4` completed with verification #57, Terra #59, Grok #62/full PASS #65, and implementation #64. The unsafe result remains preserved and explicitly superseded. | `source-integrity-correction/verification-summary.txt`; `source-integrity-correction/baseline-negative-control.log` |
| 2026-09-15 | Final corrected candidate identity | PASS; candidate subdigest `eeb809a3534feadc426473ef241c5a3eb680e441a7c2edb25eeae1af8466a1f0`; evidence manifest `4ccc1f9ae7827ee955c6527bd59c529b2e792dc8e8d8ebb6b7812ea309d8e89f`; nine candidate files individually match. | `source-integrity-correction/candidate-evidence-identity.txt`; `source-integrity-correction/candidate-files.sha256`; `source-integrity-correction/candidate-source-recheck.log` |
| 2026-09-15 | Corrected routed transform behavior | PASS; compact O(1) cursor and domain-separated source/destination chains replay exact saved prefixes read-only before payload read-write opens and before publication. Old or missing transform progress refuses. | `source-integrity-correction/verification-summary.txt` |
| 2026-09-15 | Final focused filters | PASS; transformed writer 3/3, Qwen resume 8/8, checkpoint 6/6, disk 3/3, and Gemma/remote 33/33 across 3 suites. The overlapping QwenTransform filter is 8/8 and is not additive. | `source-integrity-correction/test-TransformedTensorWriterTests-final.log`; `source-integrity-correction/test-QwenTransformResumeTests-final.log`; `source-integrity-correction/test-RemoteInstallCheckpointTests-final.log`; `source-integrity-correction/test-DiskSpaceCheckerTests-final.log`; `source-integrity-correction/test-RemotePayloadCopyTests-final.log` |
| 2026-09-15 | Debug and Release core/product builds | PASS; all four builds completed, exit 0. Release product retains an inherited unused-result warning at `Command/main.swift:273`; no correction-file diagnostic was introduced. | `source-integrity-correction/build-debug-core-final.log`; `source-integrity-correction/build-release-core-final.log`; `source-integrity-correction/build-debug-product-final.log`; `source-integrity-correction/build-release-product-final.log` |
| 2026-09-15 | Diff and D4 gates | PASS; `git diff --check` passed. Final four-stream union/status is 43 with exact match, 34 paths inherited from P1-P6 and 9 P7 candidate paths; six paths are the correction delta; missing count 0. | `source-integrity-correction/diff-check-final.log`; `source-integrity-correction/d4-four-stream-final.log`; `source-integrity-correction/candidate-files.sha256` |
| 2026-09-15 | Preserved negative control and nonblocking limits | The unsafe baseline negative control remains 1 test, 2 issues, exit 1. No dedicated same-run restore-before-publish test was added; shared `validateProgress` was inspected. Prefix replay performs O(committed-bytes) I/O. No official shard payload, conversion, model, GPU, app/server run, benchmark, or published synthetic Qwen pack was performed. | `source-integrity-correction/baseline-negative-control.log`; `source-integrity-correction/verification-summary.txt` |

---

## Phase 8 - Qwen Metal bindings have one checked contract

**When this is done:** qwen Metal bindings have one checked contract

Needs: `2`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-16T06:48:38Z — 61 GiB available on /System/Volumes/Data; memory free 55%; macOS 26.6.2, Xcode 26.5/SDK 26.5, Swift 6.3.2, SwiftPM tools 6.2, arm64 Mac14,12, 32 GiB RAM; no prohibited model processes. Full capture: `scratch/qwen3.6-35b-a3b/evidence/phase-8/phase-start.stdout`.
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-8/`
Phase-start identity: baseline HEAD `5770260510935cd32b82c4008ff06ae6458539ff`; the final candidate has four files and subdigest `862ae63096cba87c90cc804380d74d2ba6beb52bfcad4b9e1037653ee0e35b7f`. Inherited Phase 7 candidate identity remains `eeb809a3534feadc426473ef241c5a3eb680e441a7c2edb25eeae1af8466a1f0`.
Evidence manifest: `30a64bd061fa2c7320efaff6854f2b9d512953c6394f4d29687d2fe7ec993c68`. Team workflow `workflow-mac-engineering-complex-mu34za8y-igmjrs` recorded proposal #74, approvals #75/#76 (full), and lead completion #81.

### Current state

Phase 8 is complete. The final candidate centralizes checked Swift/MSL ABI bindings, scalar widths, alignment, strides, typed dimension/feature/32-bit-address failures, declaration-only `qwen_common`, and ordered exact-once registration in `MetalContext`. `Package.swift` remains unchanged by this phase. Focused independent coverage is 7/7 with zero failures; the unchanged `MetalContext` regression is 1/1. Debug and Release target builds, source/diagnostic/order checks, and D4 pass.


The first proposal #69 contained incorrect narrative AIR hashes; reviewer #70 caught the mismatch. Corrected proposal #74 binds the actual direct AIR SHA-256 `1969deb0c9aad11ee67abf23146e18875ca9e1d8486b39fa0b98118faefa2a91` and combined AIR SHA-256 `8a2103a3273cc9e2c76b6f374eccebbd40ad985fed9631693746ec56e4373d71`. The earlier missing-component failures, the superseded `a2edda47ad21779a1e429d680070490e8b39a53233db2dd1eb408298b81de537`, and the successful compile with four inherited warnings remain preserved rather than rewritten.

### Target state

One owner defines shared Swift/MSL binding indices, aligned structs, feature requirements, and registers a common Qwen shader module. P8 contains no success-stub kernel. Concrete pipelines and numerical execution first appear in P9, P10, P11, and P16.

### Tasks

#### 8.1 Define host-side Qwen binding and layout types

| | |
|---|---|
| Duty | `metal-interface` |
| Touches | `Sources/TurboFieldfare/Kernels/Qwen/QwenMetalContracts.swift` |
| Depends on | `2.6` |
| Parallel safe | `yes` |
| Deliverable | Swift types centralize buffer indices, scalar widths, alignment, strides, checked dimension conversion, and feature checks used by all Qwen kernels. |

Do not expose Gemma layout constants or add per-token protocol dispatch.

**Acceptance detail**
- [x] Swift layout assertions cover every shared struct and binding index.
- [x] Unsupported dimensions or device features produce typed errors.

**How to check**
```sh
Scripts/test.sh --filter QwenMetalContractTests
```

**Evidence:** PASS; independent `QwenMetalContractTests` ran 7 tests in 1 suite with zero failures. ABI fields, bindings, affine layout remainders, invalid/unsupported/unaddressable dimensions, and first-missing-feature errors are covered. `focused-tests-final.stdout`; `focused-tests-final.exit`.

#### 8.2 Define matching common MSL declarations

| | |
|---|---|
| Duty | `metal-interface` |
| Touches | `Sources/TurboFieldfare/Metal/Qwen/qwen_common.metal` |
| Depends on | `8.1` |
| Parallel safe | `no` |
| Deliverable | The common module contains shared structs, constants, and helpers only; it does not export a no-op compute kernel. |

Keep shader integer widths and alignment exactly matched to the Swift contract.

**Acceptance detail**
- [x] The shared production library compiles the common declarations.
- [x] Host layout tests detect any changed offset, stride, or index.

**How to check**
```sh
Scripts/test.sh --filter QwenMetalContractTests
```

**Evidence:** PASS; declaration-only `qwen_common` compiled to nonempty AIR, exit 0, SHA-256 `1969deb0c9aad11ee67abf23146e18875ca9e1d8486b39fa0b98118faefa2a91`. The combined production library also compiled nonempty, exit 0, SHA-256 `8a2103a3273cc9e2c76b6f374eccebbd40ad985fed9631693746ec56e4373d71`; host declaration and no-kernel checks passed. `offline-msl/qwen-common-metal-final.stdout`; `offline-msl/shared-production-metal-final.stdout`; `final-diagnostics-scan.txt`.

#### 8.3 Register the common module in one MetalContext edit

| | |
|---|---|
| Duty | `metal-integration` |
| Touches | `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` |
| Depends on | `8.2` |
| Parallel safe | `no` |
| Deliverable | Add `qwen_common` to the shared module map and preserve current module order and private-library caching semantics. |

`Package.swift` already copies the complete `Sources/TurboFieldfare/Metal` directory; verify that fact and do not edit the package unless the resource boundary changes.

**Acceptance detail**
- [x] Missing Qwen common source reports `missingShaderResource`.
- [x] Existing Metal modules remain registered once.

**How to check**
```sh
Scripts/test.sh --filter QwenMetalContractTests
```

**Evidence:** PASS; focused coverage verified missing-resource mapping and ordered exact-once `qwen_common` registration after existing modules. The unchanged device-creation regression ran 1 test in 1 suite with zero failures. `focused-tests-final.stdout`; `metal-context-regression.stdout`; `offline-msl/ordered-inputs.sha256`.

#### 8.4 Add host layout and registration tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` |
| Depends on | `8.1` |
| Parallel safe | `yes` |
| Deliverable | Tests compare Swift field offsets/strides with declared contract constants and verify resource registration without creating fake pipelines. |

Actual GPU pipeline creation and numerical checks remain mandatory in P9, P10, P11, and P16.

**Acceptance detail**
- [x] The suite executes with a nonzero count.
- [x] No test treats a missing concrete kernel as a successful GPU path.

**How to check**
```sh
Scripts/test.sh --filter QwenMetalContractTests
```

**Evidence:** PASS; the independent suite ran 7 tests in 1 suite with zero failures. Source scan reports zero kernel entries in declaration-only `qwen_common`; direct and combined AIR compilation are compiler acceptance only. `focused-tests-final.stdout`; `final-diagnostics-scan.txt`; `offline-msl/qwen-common-metal-final.stdout`.

### Phase 8 coverage plan

Every final Phase 8 candidate file has its own row. The four rows map to the independent 7/7 contract suite; the separate unchanged MetalContext regression is recorded in the evidence table.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen/QwenMetalContracts.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | ABI fields, binding indices, widths, alignment, strides, and typed failures. | `Scripts/test.sh --filter QwenMetalContractTests` |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_common.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | Matching declarations, no kernel entry, and direct/combined AIR compilation. | `Scripts/test.sh --filter QwenMetalContractTests` |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | Missing-resource mapping and ordered exact-once registration; regression separately 1/1. | `Scripts/test.sh --filter QwenMetalContractTests` |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | self | 7 tests in 1 suite, zero failures. | `Scripts/test.sh --filter QwenMetalContractTests` |

### Phase 8 evidence

<a id="phase-8-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-16 | Candidate identity and approvals | PASS; four-file candidate subdigest `862ae63096cba87c90cc804380d74d2ba6beb52bfcad4b9e1037653ee0e35b7f`, evidence manifest `30a64bd061fa2c7320efaff6854f2b9d512953c6394f4d29687d2fe7ec993c68`. Proposal #74, Terra/Grok approvals #75/#76 (full), and lead completion #81 are recorded for the exact candidate. | `candidate-evidence-identity.txt`; `candidate-files.sha256`; `candidate.diff`; `verification-summary.txt` |
| 2026-09-16 | Swift/MSL contract and focused tests | PASS; 7 tests in 1 suite, zero failures. Typed dimension, feature, and 32-bit-address failures; ABI/layout/index checks; declaration-only no-kernel check; and missing-resource behavior passed. | `focused-tests-final.stdout`; `focused-tests-final.exit`; `verification-summary.txt` |
| 2026-09-16 | Direct and combined MSL compilation | PASS; direct and combined AIR were nonempty and exit 0. Actual SHA-256 values are `1969deb0c9aad11ee67abf23146e18875ca9e1d8486b39fa0b98118faefa2a91` and `8a2103a3273cc9e2c76b6f374eccebbd40ad985fed9631693746ec56e4373d71`. This is compiler acceptance only, not runtime GPU/model/performance evidence. | `offline-msl/qwen_common.air`; `offline-msl/shared-production-combined.air`; `verification-submission-correction.txt` |
| 2026-09-16 | MetalContext regression and source/order gates | PASS; unchanged device creation 1/1; source contract, kernel-entry, module-order, and exact-once checks passed. Combined compilation has 10 inherited warnings before Qwen at generated line 4531; direct Qwen diagnostics are zero and no Qwen diagnostic is attributed. | `metal-context-regression.stdout`; `final-diagnostics-scan.txt`; `offline-msl/ordered-inputs.sha256` |
| 2026-09-16 | Debug/Release target builds and diff | PASS; Debug and Release target builds exit 0; candidate diff check exit 0. `Package.swift` is inherited dirty but unchanged by the four-file Phase 8 candidate. | `debug-build-final.stdout`; `release-build-final.stdout`; `candidate-diff.exit`; `candidate-diff.stderr` |
| 2026-09-16 | Authoritative D4 | PASS; 47-path union/status match = 43 inherited plus 4 Phase 8 paths, with zero missing or unclassified paths. | `d4-four-stream-final.stdout`; `candidate-files.sha256` |
| 2026-09-16 | Toolchain authorization and preserved failures | User expressly authorized Apple MetalToolchain installation. Official export workflow installed build `17F42`, identifier `com.apple.dt.toolchain.Metal.32023.883`, without switching Xcode. The repository-owned export bundle is recorded; Apple-managed asset/cache/cryptex paths remain outside the repository. Subsequent import exit 70 (`already installed`), original missing-component failures, and superseded candidate `a2edda47ad21779a1e429d680070490e8b39a53233db2dd1eb408298b81de537` with four inherited warnings remain preserved. | `toolchain-install/`; `offline-msl/post-install-compile-proof-final.*`; `verification-submission-correction.txt` |
| 2026-09-16 | P7 inheritance and postapproval document identity | P7 nine-file identity `eeb809a3534feadc426473ef241c5a3eb680e441a7c2edb25eeae1af8466a1f0` matches. Canonical docs were archived before this edit; their final postapproval document-only hashes are recorded separately from the reviewed code candidate. | `p7-artifact-identity-recheck.*`; `documentation/pre-final-doc-snapshot/SHA256SUMS.txt`; `documentation/final-document-hashes.txt` |
| 2026-09-16 | Scope limits | No runtime GPU execution, model construction/run, official payload/shard access, download, conversion, publication, benchmark, or performance claim was performed. P9 owns concrete pipelines and numerical GPU checks. | `verification-summary.txt`; `final-diagnostics-scan.txt` |

---

## Phase 9 - Qwen full attention matches the tiny oracle

**When this is done:** qwen full attention matches the tiny oracle

Needs: `2, 3, 8`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-16T08:09:48Z — macOS 26.6.2 on arm64 Apple M2 Pro; / had 60 GiB free; memory free was 59%; the prepared pack was present; no prohibited processes were running; existing dirty and untracked repository work was preserved.`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-9/`

### Current state

At phase start, the shared Gemma attention, RMSNorm, and RoPE primitives were not Qwen's contract. Qwen required doubled Q/gate projection ordering, Q/K normalization, partial RoPE, grouped-query attention, and ten full-attention KV layers with completion-safe lifetime rules.

### Target state

The candidate now provides a CPU reference for Qwen's ordered full-attention math and ten-layer KV state. It also executes concrete Q/K norm plus partial-RoPE and sigmoid-gate Metal primitives with validation enabled. The QK, softmax, value, and layout portions remain the disclosed CPU reference; this phase does not claim full-GPU attention, an authentic full Qwen model, conversion, inference, or performance.

### Tasks

#### 9.1 Implement Qwen partial RoPE and Q/K normalization

| | |
|---|---|
| Duty | `attention-code` |
| Touches | `Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift` |
| Depends on | `2.6, 3.5, 8.4` |
| Parallel safe | `yes` |
| Deliverable | Encode Q/K RMS normalization and rotate exactly 64 dimensions from head dimension 256 using theta 10,000,000. |

Keep position arithmetic overflow-checked and separate from Gemma `RoPE` constants.

**Acceptance detail**
- [x] Prefill and decode intermediates match P3 tolerances.
- [x] A negative control rotating the wrong span fails.

**How to check**
```sh
Scripts/test.sh --filter QwenFullAttentionTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-full-attention-test.txt` — 6 Swift Testing tests passed with no skip; frozen P3 intermediates and outputs, official geometry, negative controls, and validated real GPU preprocessing primitives executed; exit 0.

#### 9.2 Implement Qwen full-attention output gating

| | |
|---|---|
| Duty | `attention-code` |
| Touches | `Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift` |
| Depends on | `9.1` |
| Parallel safe | `no` |
| Deliverable | Consume the doubled Q projection as query plus output gate, apply attention, gate the result, then project output in reference order. |

Do not apply Gemma logit softcap or assume equal Q and KV head counts.

**Acceptance detail**
- [x] One-token and awkward-length outputs match the oracle.
- [x] Removing or moving the output gate breaks the negative control.

**How to check**
```sh
Scripts/test.sh --filter QwenFullAttentionTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-full-attention-test.txt` — the 6-test suite passed with output-gate ordering, one-token/length-5 coverage, and gate/softcap/GQA negative controls; exit 0.

#### 9.3 Implement ten-layer Qwen full KV storage

| | |
|---|---|
| Duty | `attention-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift` |
| Depends on | `9.1` |
| Parallel safe | `yes` |
| Deliverable | Allocate and advance only the ten manifest-declared full-attention layers with checked context capacity and explicit last-use lifetime. |

Expose snapshot/restore primitives needed by P13 without claiming recurrent state is KV.

**Acceptance detail**
- [x] Repeated decode preserves layer separation and positions.
- [x] Overflow, out-of-order write, and early reuse fail before memory corruption.

**How to check**
```sh
Scripts/test.sh --filter QwenFullAttentionKVTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-kv-test.txt` — 6 Swift Testing tests passed with no skip; ten selected layers, distinct K/V state, split/repeated decode, snapshot/restore lineage, failure completion, cancellation, and early-reuse checks passed; exit 0.

#### 9.4 Add concrete full-attention Metal kernels

| | |
|---|---|
| Duty | `attention-metal` |
| Touches | `Sources/TurboFieldfare/Metal/Qwen/qwen_full_attention.metal`; `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` |
| Depends on | `9.1, 9.2, 8.4` |
| Parallel safe | `no` |
| Deliverable | Implement and register the concrete Q/K norm, partial-RoPE, and sigmoid-gate pipelines required by the Qwen attention component. |

Query pipeline/device limits; bounds-check awkward token/head shapes; do not add a no-op fallback.

**Acceptance detail**
- [x] The required Q/K norm, partial-RoPE, and gate pipelines are created and execute on supported hardware.
- [x] Awkward lengths and final partial threadgroups remain in bounds.

**How to check**
```sh
Scripts/test.sh --filter QwenFullAttentionTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-full-attention-test.txt` and `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-composed-msl-compile.txt` — real validated GPU primitives passed at awkward counts; composed `qwen_common.metal` then `qwen_full_attention.metal` compiled with Apple metal 32023.883 and zero diagnostics; exit 0.

#### 9.5 Add full-attention and KV numerical tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenFullAttentionKVTests.swift`; `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` |
| Depends on | `9.1, 9.3` |
| Parallel safe | `yes` |
| Deliverable | Tests compare CPU/oracle intermediates, outputs, repeated decode, prefill splits, KV bounds, and lifetime behavior using declared tolerances. |

Run actual Metal on supported hardware; a GPU skip is not a pass for this phase.

**Acceptance detail**
- [x] All three named suites execute with nonzero counts and no unexpected skips.
- [x] Missing Q/K norm, gate, partial RoPE, or a KV layer is detected.

**How to check**
```sh
Scripts/test.sh --filter QwenFullAttention
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-full-attention-test.txt`, `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-kv-test.txt`, `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-metal-contract-test.txt`, and `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-combined-test.txt` — full attention 6/6, KV 6/6, registry 7/7, and the combined filter 1/1 passed; the combined test overlaps the full-attention suite and is not additive.

### Phase 9 coverage plan

The exact Phase 9 candidate changed seven paths: four production paths and three test paths. The inherited paths visible in the repository-wide status inventory belong to earlier phases and are preserved, not silently assigned to Phase 9.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | CPU-reference order, Q/K norm, partial RoPE, gate, intermediates, output, and negative controls. | `Scripts/test.sh --filter QwenFullAttentionTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenFullAttentionKV.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenFullAttentionKVTests.swift` | Ten-layer selection, distinct K/V, positions, snapshot lineage, completion leases, cancellation, and refusal cases. | `Scripts/test.sh --filter QwenFullAttentionKVTests` |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_full_attention.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | Real validated Q/K norm, partial-RoPE, and sigmoid-gate pipelines handle awkward counts. | `Scripts/test.sh --filter QwenFullAttentionTests` |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | Existing module registry plus the appended full-attention module remains ordered and single-registered. | `Scripts/test.sh --filter QwenMetalContractTests` |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | The independent numerical suite executes 6 tests with no skips and checks P3 and official geometry. | `Scripts/test.sh --filter QwenFullAttentionTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenFullAttentionKVTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenFullAttentionKVTests.swift` | The independent KV suite executes 6 tests with no skips and checks state and GPU lifetime. | `Scripts/test.sh --filter QwenFullAttentionKVTests` |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | The registry suite executes 7 tests with no skips and retains P8 assertions. | `Scripts/test.sh --filter QwenMetalContractTests` |

### Phase 9 evidence

<a id="phase-9-evidence"></a>

Candidate identity: HEAD `5770260510935cd32b82c4008ff06ae6458539ff`; seven-path candidate manifest digest `dd0e76f9119d7818e7666eb95c079c347079b5cbe18be5bcaed6cad49dd45e45`; exact candidate file hashes are in `scratch/qwen3.6-35b-a3b/evidence/phase-9/lead/final-candidate-sha256.txt` and the recheck is in `final-candidate-recheck.txt`.

Coverage counts are deliberately non-additive: the combined filter is 1 test and is included in the full-attention 6; the focused counts are attention 6, KV 6, registry 7, and the exact Gemma AttentionTests regression 11, for 30 unique focused tests. The saved focused logs report zero failures and zero skips; the SwiftPM XCTest harness's selected-tests preamble executed zero XCTest tests before Swift Testing discovered the named suites.

The independent frozen P3 oracle covers doubled per-head Q/raw-gate split, Q/K (1+weight) RMSNorm, partial rotate-half RoPE, contiguous GQA, headDim^-0.5 causal FP32 softmax, V accumulation, transpose/reshape, sigmoid gating, and o_proj. Frozen P3 geometry is 2 Q heads, 1 KV head, head dimension 16, rotary span 12; it covers one- and five-token intermediates and final output, split-at-3 prefill, and token decode at absolute and relative tolerance `1e-5`. The official analytic probe is 16 Q heads, 2 KV heads, head dimension 256, rotary span 64, and theta `1e7`; dimensions 64 through 255 remain unchanged. Actual GPU qualification is limited to Q/K normalization plus partial RoPE and sigmoid gating. QK, FP32 softmax, V, and layout remain CPU-reference behavior; no full-GPU attention or authentic Qwen model qualification is claimed.

KV evidence records exactly ten mask-selected full-attention layers with distinct K/V buffers, one-shot read/write leases, completion-gated publication, branch-safe snapshots, cancellation-safe abandon, and bounded capacity. The snapshot represents full-attention KV positions and lineage only; it does not claim recurrent or convolution state. Deterministic abandon/cancel, nonpublication, and same-range reacquisition follow accepted plans `149/155`; no failed GPU command was manufactured, and GPU-error completion mapping was read-reviewed. No production code blocks for GPU completion, and no performance budget or speed claim was measured.

The saved engineering approval is teamworkflow `teamworkflow-mac-engineering-complex-mu3sj9na-t7t4mc` revision 5: proposal `165`, approvals `166` and `168`, and full verdict `169`. Terra's medium tester gap after three partial rounds was completed by Luna's xhigh tester; no acceptance waiver was used. The saved lead checks are: `final-preflight.txt` PASS exit 0 (macOS 26.6.2, Swift 6.3.2 arm64, Xcode 26.5/SDK 26.5, M2 Pro with Metal 4, 60 GiB free, populated pack, no prohibited process); `final-combined-test.txt` PASS exit 0; `final-full-attention-test.txt` 6/6 PASS exit 0 with Metal API and GPU validation; `final-kv-test.txt` 6/6 PASS exit 0 with validation; `final-metal-contract-test.txt` 7/7 PASS exit 0; Debug and Release builds PASS; and `final-composed-msl-compile.txt` PASS exit 0 with zero diagnostics and exact common-before-attention source order.

The project-specific four-stream D4 inventory is preserved in `final-d4-status-hashes.txt`: 52 total paths were observed, classified as 45 inherited paths plus the seven exact Phase 9 paths (four production and three tests), with no extra Phase 9 path; status and hash recheck exit 0. The generic checker still reports its VisionCapture-only D4 as “no source files changed”; that fallback is not authoritative for this SwiftPM project. The lead's scoped identity record also contains an obsolete `Sources/TurboFieldfare/Runtime/KVCacheManager.swift` hash attempt; that path is absent, was marked nonblocking because all seven scoped hashes and the recheck are correct, and is not reported as a passing hash.

Failure history remains preserved: `resume-preflight.txt` exit 17 was a self-match from its literal process regex and `resume-preflight-corrected.txt` plus `final-preflight.txt` pass; `metal-tool-help.txt` exit 74 was a broken pipe from piping help to `head` and the corrected full capture passes; `final-gemma-attention-regression.txt` selected the intended Gemma tests but also selected `VisionAttentionTests` and failed its `37760 > 32768` threadgroup-memory validation assertion. The exact intended `final-gemma-attention-exact.txt` rerun passes 11/11. Because no saved baseline proves that Vision validation failure predated Phase 9, it is documented as an observed out-of-scope validation failure, not as proven preexisting; no Vision fix or rerun was performed.

The first independent test helper hard-coded three projection rows and failed its first compile/run; Luna corrected the helper to derive the output row count, after which the independent handoff passed. No captured old-tester log exists; provenance is the agent reports/journals `#160/#164`, not a manufactured log. Metal Toolchain 17F42 was already installed and authorized in P8; no new installation occurred in P9. No official Qwen payload was read, converted, run, downloaded, or benchmarked. The separate postapproval document-only identity is recorded in `scratch/qwen3.6-35b-a3b/evidence/phase-9/documentation/final-document-hashes.txt` after generation; the pre-edit canonical documents are archived in `scratch/qwen3.6-35b-a3b/evidence/phase-9/documentation/pre-final-doc-snapshot/`.

## Phase 10 - Qwen linear attention matches the tiny oracle

**When this is done:** qwen linear attention matches the tiny oracle

Needs: `2, 3, 8`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-16T11:13:58Z — macOS 26.6.2, Xcode 26.5/SDK 26.5, Swift 6.3.2, arm64 Apple M2 Pro with 19 GPU cores and Metal 4; 60 GiB available, memory free 53%, no prohibited processes. The resolved Gemma pack and its pre-existing symlinked installation were preserved; no payload contents or payload hashes were read.`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-10/`

### Current state

Phase 10 is closed. The candidate adds explicit thirty-layer linear-attention state with width-four causal-convolution history and FP32 recurrent matrices, a CPU reference-order Gated DeltaNet implementation, and four concrete Metal kernels: layout, causal convolution, FP32 recurrence, and gated RMSNorm. It also binds the linear module into `MetalContext` without changing Gemma paths.

### Target state

One-token and chunked Qwen linear-attention calculations match the pinned P3 oracle at its declared tolerances. Convolution history and recurrence remain explicit state across chunk partitions. Committed state is separate from staging, and submitted staging remains retained until actual GPU completion before it is committed or discarded.

### Tasks

#### 10.1 Implement bounded causal-convolution state

| | |
|---|---|
| Duty | `deltanet-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift` |
| Depends on | `2.6, 3.5, 8.4` |
| Parallel safe | `yes` |
| Deliverable | Own per-layer convolution history of width four and FP32 DeltaNet recurrent matrices with checked dimensions and explicit clone/restore operations. |

Do not represent recurrent state as a token position or `KVCacheManager` rewind.

**Acceptance detail**
- [x] Thirty official linear-layer states initialize deterministically and remain disjoint.
- [x] Clone/restore captures both convolution history and FP32 recurrence, including every element.

**How to check**
```sh
Scripts/test.sh --filter QwenLinearAttentionStateTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-10/lead/state-tests.log` — 7/7 tests pass with no failures or skips. The suite checks official 30-of-40 layer selection, disjoint state, full history/matrix clone/restore/reset, and pending-use protection.

#### 10.2 Implement one-token Gated DeltaNet update

| | |
|---|---|
| Duty | `deltanet-code` |
| Touches | `Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift` |
| Depends on | `10.1` |
| Parallel safe | `yes` |
| Deliverable | Project Q/K/V/Z/A/B, update depthwise causal convolution, compute FP32 decay/update, normalize, gate, and output-project in reference order. |

Keep decay and recurrent accumulation FP32 even when activations are BF16/FP16.

**Acceptance detail**
- [x] One-token intermediates and final state match P3 at output FP32 absolute/relative tolerance `1e-5` and state FP32 absolute/relative tolerance `2e-5`.
- [x] An independent 64-token Float16(`logDecay`) mutant changes both output and final state by more than `1e-4`; production remains checked at the declared P3 tolerances.

**How to check**
```sh
Scripts/test.sh --filter QwenGatedDeltaNetTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-10/lead/deltanet-tests.log` — 7/7 tests pass with no failures or skips. P3 causal-convolution, recurrence, and direct-weight RMSNorm/SiLU-gate references match; the independent Float16 decay negative control separates output and state by more than `1e-4`.

#### 10.3 Implement chunked prefill with exact state carry

| | |
|---|---|
| Duty | `deltanet-code` |
| Touches | `Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift` |
| Depends on | `10.2` |
| Parallel safe | `no` |
| Deliverable | Add a bounded chunk path that produces the same output and final convolution/recurrent state as token-at-a-time reference execution. |

Exercise chunks shorter than, equal to, and crossing the convolution width.

**Acceptance detail**
- [x] Every tested chunk partition ends at the same reference output and FP32 convolution/recurrent state.
- [x] Submitted cancellation retains staging and admission until actual completion, then discards staging without changing committed bytes.

**How to check**
```sh
Scripts/test.sh --filter QwenGatedDeltaNetTests
```

**Evidence:** `state-tests.log` and `deltanet-tests.log` record five-token token-at-a-time, chunked, split-prefill, repeated-decode, and awkward-length checks. The shared-event cancellation test blocks a submitted command on an actual GPU event, proves snapshot/restore is refused while it is in use, then proves committed history and matrix bytes are unchanged after cancellation and completion.

#### 10.4 Add concrete convolution and DeltaNet kernels

| | |
|---|---|
| Duty | `deltanet-metal` |
| Touches | `Sources/TurboFieldfare/Metal/Qwen/qwen_linear_attention.metal`; `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` |
| Depends on | `10.1, 10.2, 8.4` |
| Parallel safe | `no` |
| Deliverable | Implement and register bounded layout, depthwise convolution, FP32 recurrent update/decay, and gated normalization kernels required by the host component. |

All threads participating in barriers must remain converged at partial boundaries. No missing pipeline may silently fall back to success.

**Acceptance detail**
- [x] Layout, causal-convolution, FP32-recurrence, and gated-RMSNorm pipelines create and execute on supported hardware.
- [x] Awkward token length 5, heads, and tails stay in bounds and match the independent CPU results.

**How to check**
```sh
Scripts/test.sh --filter QwenGatedDeltaNetTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-10/lead/deltanet-tests.log` — all four production pipelines execute on the actual GPU for awkward length 5 and match independent CPU results at the declared tolerances. `deltanet-metal-validation.log` repeats the same 7/7 suite with Metal API and shader validation enabled before device creation.

#### 10.5 Add state, decode, and chunk numerical tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift`; `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift`; `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` |
| Depends on | `10.1, 10.2` |
| Parallel safe | `yes` |
| Deliverable | Tests cover zero/one token, convolution boundaries, many chunk partitions, repeated decode, cancellation after submission, and final FP32 state. |

Use P3 oracle values and independent CPU checks; a skipped GPU run does not close the phase.

**Acceptance detail**
- [x] State, DeltaNet, and registry suites each execute seven tests with no failures or unexpected skips.
- [x] Wrong history, reduced decay precision, projection/update order, gate, or final state is detected by the focused checks.

**How to check**
```sh
Scripts/test.sh --filter QwenGatedDeltaNet
```

**Evidence:** `state-tests.log`, `deltanet-tests.log`, `deltanet-metal-validation.log`, and `metal-contract-tests.log` record State 7/7, ordinary DeltaNet 7/7, validation DeltaNet 7/7, and Registry 7/7. The ordinary and validation DeltaNet runs are repeated validation of the same seven tests, not additive coverage. The test-list log confirms nonzero discovery for all three suites.

### Phase 10 coverage plan

The exact Phase 10 candidate changed seven paths: four production paths and three test paths. The inherited paths visible in the full repository status inventory belong to earlier phases and are preserved, not silently assigned to Phase 10.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | Official layer isolation, full history/matrix clone/restore/reset, and completion-safe state transactions. | `Scripts/test.sh --filter QwenLinearAttentionStateTests` |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenGatedDeltaNet.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift` | P3 convolution/recurrence/gated-norm order, chunk parity, negative control, and final FP32 state. | `Scripts/test.sh --filter QwenGatedDeltaNet` |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_linear_attention.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift` | Actual GPU execution of all four kernels at awkward length with independent CPU comparison and validation rerun. | `Scripts/test.sh --filter QwenGatedDeltaNet` |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | Linear module registration remains ordered, discoverable, and exact-once. | `Scripts/test.sh --filter QwenMetalContractTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | The state suite executes 7 tests with zero failures or skips. | `Scripts/test.sh --filter QwenLinearAttentionStateTests` |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenGatedDeltaNetTests.swift` | The DeltaNet suite executes 7 tests with zero failures or skips. | `Scripts/test.sh --filter QwenGatedDeltaNet` |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | The registry suite executes 7 tests with zero failures or skips. | `Scripts/test.sh --filter QwenMetalContractTests` |

### Phase 10 evidence

<a id="phase-10-evidence"></a>

Candidate identity: HEAD `5770260510935cd32b82c4008ff06ae6458539ff` plus the seven exact Phase 10 paths in `scratch/qwen3.6-35b-a3b/evidence/phase-10/lead/candidate-sha256.txt`. Candidate digest: `f45bca995585fc5b0ab35d6643ac451660b516214684de511fdc19324cee9336`.

The seven candidate paths are four production files (`QwenLinearAttentionState.swift`, `QwenGatedDeltaNet.swift`, `qwen_linear_attention.metal`, and `MetalContext.swift`) and three test files (the state, DeltaNet, and Metal-contract suites). The lead four-stream D4 artifact retains 69 path entries: 57 file paths plus 12 untracked directory markers. The file-path union is 57 paths: 50 inherited files plus these seven exact Phase 10 files, with no extra Phase 10 file. The generic task-tracker checker is not authoritative for this SwiftPM repository because its D4 pathspec watches only `VisionCapture`; it reports its own “no source files changed” result separately. The authoritative `Sources`, `Tests`, and `Package.swift` array-based four-stream check is recorded in `source-d4-four-stream-union.txt`, with the status guard in `source-status.txt`.

State, ordinary DeltaNet, and registry discovery each report 7 tests, for 21 unique tests. The validation DeltaNet run repeats the ordinary seven and is not added to that count. No test failed or skipped. The declared P3 output tolerance is FP32 absolute/relative `1e-5`; the declared recurrent-state tolerance is FP32 absolute/relative `2e-5`. The observed P3 agreement of `7.450580596923828e-09` is reported as an observation, not a budget. The fixture's approximately `6.20e-6` Float16-decay sensitivity is recorded as sensitivity, not a tolerance rejection; the separate 64-token independent Float16(`logDecay`) mutant changes both output and state by more than `1e-4`, while production remains checked at the declared tolerances.

State ownership is transactional: reservation copies committed history and recurrent matrices into staging; pre-submit abort/cancel releases staging; a lock-linearized sole submit marks the update submitted and commits the command buffer; submitted cancellation retains staging and admission until completion and then discards it. The shared-event blocked GPU test proves that clone/restore is refused during actual use, cancellation does not corrupt old committed bytes, and the layer can be reacquired after completion.

The final Debug build exits 0 with no new warning after the initial generic `setBytes` pointer warning was corrected to `withUnsafeBytes`. The final Release build exits 0 and retains two unrelated existing server warnings from `HTTPServer.swift` and `ServerInference.swift`. The initial warning and the initial unsupported non-following symlink probe are retained in `debug-build-initial.log` and `preflight.md`; the withdrawn empty/incomplete-pack conclusion is superseded by the resolved-path preflight. Initial test corrections (the `encodeWaitForEvent` API spelling and the independent mutant's `[Float]` inference) remain represented by agent-report provenance where no original correction log was saved; no passing result is attributed to those absent logs.

Runtime `MetalContext` composes the actual MSL source and the validation run executes the four kernels on the supported GPU. No separate offline AIR compile is claimed. No official Qwen payload or model was read, hashed, downloaded, converted, constructed, run, or benchmarked. No performance budget or authentic full-model qualification is claimed; Phase 10 is correctness-first and single-thread-per-value-head for the recurrent kernel.

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-16 | Resolved-path repository preflight | PASS; exit 0 at 11:13:58Z. The existing Gemma pack resolved to its regular directory with the recorded manifest, receipt, payload, tokenizer, and 30 expert files; no payload contents or hashes were read. | `lead/preflight.md` |
| 2026-09-16 | Final Debug and Release builds | PASS; both exit 0. Debug has no new warning; Release retains two unrelated existing server warnings. | `lead/debug-build-final.log`; `lead/release-build.log` |
| 2026-09-16 | Test discovery | PASS; State 7, DeltaNet 7, and Registry 7 were discovered with nonzero counts. | `lead/test-list.log` |
| 2026-09-16 | State behavior and lifetime tests | PASS; 7/7 with zero failures and skips, including official layer isolation and shared-event cancellation. | `lead/state-tests.log` |
| 2026-09-16 | Ordinary DeltaNet tests and real GPU execution | PASS; 7/7 with zero failures and skips, including P3 CPU comparisons and all four awkward-length GPU pipelines. | `lead/deltanet-tests.log` |
| 2026-09-16 | DeltaNet Metal validation rerun | PASS; 7/7 with zero failures and skips; API and shader validation were enabled before device creation. This repeats, rather than adds to, ordinary DeltaNet coverage. | `lead/deltanet-metal-validation.log` |
| 2026-09-16 | Metal registry regression | PASS; 7/7 with zero failures and skips. | `lead/metal-contract-tests.log` |
| 2026-09-16 | Diff check | PASS; whitespace/diff check exit 0. | `lead/diff-check.log` |
| 2026-09-16 | Authoritative D4/status/hash checks | PASS; 57 file paths are 50 inherited plus seven exact Phase 10 paths; the lead artifact also retains 12 untracked directory markers, for 69 entries. All candidate hashes and the aggregate digest match. | `lead/source-d4-four-stream-union.txt`; `lead/source-status.txt`; `lead/candidate-sha256.txt`; `lead/candidate-digest.txt` |
| 2026-09-16 | Initial diagnostics and corrections | Preserved: initial Debug build exposed one new `setBytes` pointer warning, fixed before freeze; initial non-following symlink probe and withdrawn empty-pack claim are corrected in the preflight record; initial test API/type corrections retain agent-report provenance where no original log exists. | `lead/debug-build-initial.log`; `lead/preflight.md` |

## Phase 11 - Qwen MoE matches the tiny oracle

**When this is done:** qwen MoE matches the tiny oracle

Needs: `2, 3, 8`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-16T13:22:03Z — 62,227,772 KB available; macOS 26.6.2 arm64 Apple silicon; prohibited process count zero; memory free 51%; the required Gemma install was present.`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-11/`

### Current state

The Gemma packed-expert stride and streaming path were not sufficient for Qwen's manifest-driven 256-way routing, Top-8 normalization, packed gate/up/down tensors, and sigmoid-gated shared expert. Phase 11 therefore had to preserve v1/Gemma decoding while adding a strict v2 Qwen path and a bounded completion-owned streamed-expert lifetime.

### Target state

A concrete Qwen MoE path validates writer-emitted v2 metadata, performs deterministic finite-FP32 routing, executes routed and shared experts, and retains mapped resources until actual GPU completion. The phase must prove the behavior against the frozen P3 oracle without claiming authentic payload or full-model qualification.

### Tasks

#### 11.1 Generalize manifest-driven expert layout

| | |
|---|---|
| Duty | `moe-code` |
| Touches | `Sources/TurboFieldfare/Infrastructure/ModelIO/PackedExpertsLayout.swift`; `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift` |
| Depends on | `2.6, 3.5, 8.4` |
| Parallel safe | `no` |
| Deliverable | Use verified per-layer expert count, stride, offsets, and affine descriptors rather than Gemma constants. |

Preserve existing Gemma layout initializers and labels. Reject inconsistent strides before mapping. The final path has strict separate writer-v2 Qwen layout decode, official 40-by-256 metadata, tiny paths, and cross-binding rejection before mapping.

**Acceptance detail**
- [x] Qwen 256-expert layouts validate from v2 metadata.
- [x] Gemma packed layouts retain their current byte calculations.

**How to check**
```sh
Scripts/test.sh --filter QwenMoETests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-ordinary.log`, `scratch/qwen3.6-35b-a3b/evidence/phase-11/lead/correction-v1-packed-layout-regression.log`, and `scratch/qwen3.6-35b-a3b/evidence/phase-11/lead/correction-model-runtime-schema-regression.log` — the final targeted suite passed 14/14; packed-layout regression passed 4/4; schema validation passed 8/8; all exited 0 with no skips.

#### 11.2 Implement deterministic Qwen Top-8 routing

| | |
|---|---|
| Duty | `moe-code` |
| Touches | `Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift` |
| Depends on | `11.1` |
| Parallel safe | `yes` |
| Deliverable | Compute finite FP32 routing probabilities, sort them descending, break exact equal-probability ties by ascending expert ID, choose eight unique IDs, and renormalize their weights. This is TurboFieldfare's deterministic tie policy, not a guarantee of universal `torch.topk` tie parity; frozen P3 reference outputs, fixtures, and tolerances remain unchanged. |

Expose chosen IDs and weights for diagnostics/tests without adding abstraction inside per-expert hot loops.

**Acceptance detail**
- [x] Router IDs and normalized weights match P3 fixtures.
- [x] Tie, NaN, and repeated-expert cases have explicit behavior.

**How to check**
```sh
Scripts/test.sh --filter QwenMoETests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-ordinary.log` and `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-metal-validation.log` each show the same 14-test suite, including the pinned routing and independent-generator oracle cases, with 14/14 pass, zero failures, and zero skips. The exact owner choice is recorded below and normalized as `qwen-phase11-moe #35`.

#### 11.3 Implement routed and sigmoid-gated shared experts

| | |
|---|---|
| Duty | `moe-code` |
| Touches | `Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift` |
| Depends on | `11.2` |
| Parallel safe | `no` |
| Deliverable | Execute packed routed gate/up/down projections and combine the shared expert through its sigmoid gate in reference order. |

Reuse affine kernels only after shape/layout checks; retain streamed mappings until the command buffer completes. Real synthetic files exercise mapping misses, cache hits, cross-token reuse, eviction with distinct on-disk bytes, short-read recovery, and within-token duplicate rejection. There are no frozen separate routed-only/shared-only tensors; independent branch tests cover those components, while the combined output is compared to the frozen P3 record.

**Acceptance detail**
- [x] Routed/shared branch behavior is independently tested, and the combined output matches P3.
- [x] Short reads, eviction, and early reuse fail without stale output.

**How to check**
```sh
Scripts/test.sh --filter QwenMoETests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-ordinary.log` — CPU routed/shared tests, negative controls, and actual evaluation passed. `scratch/qwen3.6-35b-a3b/evidence/phase-11/lead/correction-pread-expert-streamer-regression.log` passed 16/16 for the reused streamer contract. `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-metal-validation.log` passed the internally submitted packed-routed/resident-shared GPU path; the independent packed dequant comparison is provisional at abs `2e-4` / rel `2e-3`, distinct from the P3 CPU tolerance `1e-5`.

#### 11.4 Add concrete Qwen MoE kernels

| | |
|---|---|
| Duty | `moe-metal` |
| Touches | `Sources/TurboFieldfare/Metal/Qwen/qwen_moe.metal`; `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` |
| Depends on | `11.2, 11.3, 8.4` |
| Parallel safe | `no` |
| Deliverable | Implement Qwen-specific routing selection, normalization, packed offsets, shared gating, or epilogues not satisfied by validated existing kernels. |

No hard-coded slot count or chip name. Query limits and bound partial threadgroups. Runtime composes the real MSL source and creates the concrete routing, packed routed, resident shared, and epilogue pipelines. Debug/Release and validation reuse the unchanged production MSL inputs; no separate fabricated AIR compile is claimed.

**Acceptance detail**
- [x] Required pipelines create and execute on supported hardware.
- [x] The same selected experts and weights reach the host diagnostics.

**How to check**
```sh
Scripts/test.sh --filter QwenMoETests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-ordinary.log` and `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-metal-validation.log` both ran the concrete composed-MSL pipelines; the latter reports Metal API Validation and Metal GPU Validation enabled. `scratch/qwen3.6-35b-a3b/evidence/phase-11/lead/correction-qwen-metal-contracts.log` passed 8/8, and the final source D4 artifact records the two shared contract paths already inherited in the 62-path union.

#### 11.5 Add MoE math, paging, and lifetime tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift`; `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` |
| Depends on | `11.1` |
| Parallel safe | `yes` |
| Deliverable | Tests cover ties, Top-8 renormalization, shared gate, packed offsets, cache hit/miss/eviction, repeated experts, short reads, cancellation, and GPU lifetime. |

Compare against P3 and an independent CPU calculation. A GPU skip is not a pass. `submitExperts` internally creates, transitions, encodes, and commits its command buffer before returning it. Prevalidation or command-buffer creation errors remain unsubmitted and are releasable by cancel/deinit; after-transition encoding errors cancel/discard the result but still commit for completion cleanup. Post-submit cancellation retains pins until actual completion and never reports success. Real shared-event tests prove reuse is blocked before completion and released later after completion. `lease.experts` remains module-visible residual state; no normal streamed Qwen operation accepts an external command buffer.

**Acceptance detail**
- [x] The suite executes with a nonzero count and no unexpected skip.
- [x] Wrong Top-8 normalization, missing shared gate, or stale cache data fails.

**How to check**
```sh
Scripts/test.sh --filter QwenMoETests
```

**Evidence:** The final ordinary run is 14/14 and the separate API/GPU-validation run is also 14/14 for the same 14 cases, with no failures and no skips. The new independent-generator oracle reconstructs Float32 input and generator phase 3–9 parameters, derives router logits independently, calls actual `QwenMoE.evaluate` for all three frozen tokens, and matches shapes, exact IDs, normalized weights, shared gate, and combined output at abs/rel `1e-5`. The fixture (`61999aed…`) and generator (`45104427…`) are unchanged.

### Phase 11 coverage plan

The final candidate has seven exact Phase 11 paths. The authoritative project D4 union/status check has 62 file paths: 57 inherited paths plus five newly dirty Phase 11 paths, with two additional Phase 11-touched paths already present in the inherited union. The correction changed no additional path and changed only `QwenMoETests.swift` relative to production candidate `38780ccb`; the other six candidate paths are byte-identical to that candidate. The mirrored tracker D4 is not the source D4; the project `Sources`, `Tests`, and `Package.swift` four-stream artifact is authoritative.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Infrastructure/ModelIO/PackedExpertsLayout.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | Strict Qwen v2 decode and Gemma/v1 byte-calculation regression; reused packed-layout 4/4 and final Qwen suite 14/14. | `Scripts/test.sh --filter QwenMoETests` |
| `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | Synthetic file mapping, misses, reuse, eviction, short-read cleanup, and completion-owned lifetime; reused streamer 16/16 and final Qwen suite 14/14. | `Scripts/test.sh --filter QwenMoETests` |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenMoE.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | Independent FP32 routing/evaluation oracle, negative controls, and CPU/GPU behavior; final ordinary and validation runs each 14/14 over the same cases. | `Scripts/test.sh --filter QwenMoETests` |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_moe.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | Concrete composed-MSL routing, packed-routed, resident-shared, and epilogue pipelines execute on the supported GPU with validation available. | `Scripts/test.sh --filter QwenMoETests` |
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | Qwen module registration and concrete pipeline lookup remain wired through the production context; reused registry/contract 8/8. | `Scripts/test.sh --filter QwenMoETests` |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMoETests.swift` | The changed suite executes 14/14 with no unexpected skips; the independent generator oracle is named and runs. | `Scripts/test.sh --filter QwenMoETests` |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | Existing production contract suite remains 8/8; it is included in the fresh test-target build and exact candidate D4. | `Scripts/test.sh --filter QwenMetalContractTests` |

### Phase 11 evidence

<a id="phase-11-evidence"></a>

Phase 11 is closed for candidate `ff2f4899b380480f7a58d93e69a729f1deb559a8fdb782ee4b7616654a29cd64` at HEAD `5770260510935cd32b82c4008ff06ae6458539ff`. The linked correction team accepted the technical proof: plan review `qwen-phase11-oracle-correction #10` approved at `#11`; full initial review at `#16`; independent tester contribution at `#22` and release at `#24`; verification at `#26`; Terra approval `#28`; Grok full approval `#29`; formal approval `#30`. Main accepted that technical proof. The old verification summary's pending-review sentence predates approvals `#28/#29/#30` and is not used as the current status.

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-16 | Final candidate identity and mutable-input audit | PASS; HEAD, seven candidate hashes, candidate digest, unchanged fixture/generator/Package.swift/Scripts/test.sh identities, and pinned Transformers tree are recorded. | `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/final-candidate-sha256.txt`; `mutable-input-identities.txt`; `evidence-audit.txt` |
| 2026-09-16 | Fresh ordinary Qwen MoE suite | PASS; 14/14, zero failures, zero skips; the independent-generator oracle executed. | `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-ordinary.log` |
| 2026-09-16 | Separate API/GPU-validation Qwen MoE suite | PASS; the same 14 cases, 14/14, zero failures, zero skips; Metal API Validation and Metal GPU Validation reported enabled. Do not count this as 28 unique tests. | `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/qwen-moe-metal-validation.log` |
| 2026-09-16 | Reused exact unchanged-input regressions | PASS; unique total 56 including final 14: Registry 8, PackedExpertsLayout 4, ModelRuntimeSchema 8, PreadExpertStreamer 16, StreamingKernelIntegration 1, MoEFusedFFN 1, SharedExpertInt4 1, SharedExpertInt8 1, RouterTopK 2. Do not add historical 13/12/8 or validation repeats. | `scratch/qwen3.6-35b-a3b/evidence/phase-11/lead/correction-verification-summary.txt` and the `correction-*` logs in that directory |
| 2026-09-16 | Debug and Release builds plus composed-MSL checks | PASS; fresh test target compiled; Debug/Release and real composed-MSL pipeline inputs are explicitly reused unchanged; two inherited server Release warnings remain. No separate AIR compile is claimed. | `scratch/qwen3.6-35b-a3b/evidence/phase-11/lead/correction-debug-build-final.log`; `correction-release-build-final.log`; `correction-qwen-moe-ordinary.log`; `correction-qwen-moe-metal-validation.log` |
| 2026-09-16 | Authoritative source D4 and status guard | PASS; exact 62-path union/status match, 57 inherited plus five newly dirty Phase 11 paths and two additional already-inherited Phase 11-touched paths, yielding seven candidate paths. | `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/final-source-d4-four-stream-union.txt`; `evidence-audit.txt` |
| 2026-09-16 | Final diff check | PASS; `git diff --check -- Sources Tests Package.swift` exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-11/oracle-coverage-correction/diff-check.log` |

The independent oracle reconstructs Float32 input and generator phase 3–9 parameters and calls the actual `QwenMoE.evaluate` for all three frozen tokens. It anchors shapes and inputs/logits, then matches exact IDs, normalized weights, shared gate, and combined output at abs/rel `1e-5`. Frozen separate routed-only/shared-only tensors do not exist, so the branch claim is preserved by independent branch tests, not a direct frozen-branch comparison. This describes the evidence without waiving the acceptance behavior.

Review history remains explicit. The original `qwen-phase11-moe` plan `#38`, Terra `#39`, Grok full `#40`, formal `#41`, and Main release `#42` are historical review events, not sufficient closeout evidence. Candidate `324e663a853ebbcf9302d2ea5f7bf80886a7db43f96c52e9b856c9fffea69ca2` was rejected by Terra `#67`, Grok `#71`, and Main `#69` because optional `trackLastUse` permitted encode → cancel/slot reuse → late commit. Interim auto-tracking was also superseded because an abandoned uncommitted command buffer could leak admission. Production candidate `38780ccb2d9836ee6ecf8c25d3591be26c1192dd1f9fbb715bc59f9c5b90e0a1` was verified at `#87` and approved at `#89/#90/#91`; its evidence digest was `509223c12779d1866c541e95fcd985b6a26811130b628ac9845857bac4b65a16`. Main still refused Phase 11 closeout because Grok `#90` admitted the frozen MoE output/gate were unused but incorrectly called that nonblocking. A separate Terra-high informal read-only adjudication confirmed missing acceptance proof, not a math defect; formal escalation was refused on the completed team, so the linked correction above supplied the needed proof. The original approval is not reused as sufficient evidence, and the Terra-high adjudication is not called formal.

Delivered behavior includes strict separate writer-v2 layout decoding, official 40-by-256 metadata and tiny paths with cross-binding rejection before mapping, preserved Gemma/v1 behavior, real synthetic mapping/cache miss/reuse/eviction files with distinct bytes and short-read recovery, within-token duplicate rejection with valid cross-token reuse, completion-owned internal submission, CPU routed/shared evaluation, real GPU route and packed-routed plus resident-shared/epilogue execution, and provisional independent dequant tolerances distinct from P3 CPU tolerance. `submitExperts` creates/transitions/encodes/commits before returning; prevalidation errors remain unsubmitted and releasable, while post-transition encode errors cancel/discard and still commit for cleanup. Post-submit cancellation pins slots until actual completion and never succeeds. `lease.experts` is module-visible residual state, not a normal external command-buffer operation.

No authentic Qwen payload/model run, conversion, publication, official-payload installation, benchmark, or performance proof is claimed. At the Phase 11 closeout, hybrid runner integration belonged to Phase 12; Phase 12 now records the final2 closure. The documentation closeout archive, canonical-copy, HTML generation, checker results, and post-document hashes are under `scratch/qwen3.6-35b-a3b/evidence/phase-11/documentation/`.

## Phase 12 - A tiny Qwen runner emits the expected tokens

**When this is done:** a complete synthetic Qwen text model has a verified, independently checked normalized prefill and cached decode path; no authentic-model qualification is implied.

Needs: `2, 3, 9, 10, 11`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `Final2 preflight recorded macOS 26.6.2, Xcode 26.5, Swift 6.3.2, 59 GiB filesystem available, 53% memory free, and no prohibited process. This documentation closeout ran no environment command.`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-12/`

### Current state

The Phase 12 source candidate is closed on the exact 13-file aggregate `869272236bf18564fbd0ef99e330d112a5e39de45009b680ae71aa96a7434a26` at baseline/current HEAD `5770260510935cd32b82c4008ff06ae6458539ff`. `final2-candidate-identities.txt` is the authoritative ordered identity list. The candidate contains the family admission boundary, Qwen byte-backed model and runner, sampling integration, exact synthetic fixture, and focused tests. The source D4 boundary is the four-stream union over `Sources`, `Tests`, and `Package.swift`: committed 0, unstaged 19, staged 0, and untracked 52, for a status-matching union of 71 paths. The candidate is 4 inherited paths plus 9 current-added paths; those 9 are path additions to the dirty inventory, not 9 newly created files.

The 16 coverage rows are deliberately not a 13-file count. They include the ignored Stage A generator, explicit rows for the inherited `PackedExpertsLayout.swift`, `QwenLinearAttentionState.swift`, and `QwenLinearAttentionStateTests.swift` paths authorized in this correction, and unchanged compatibility test paths. The authoritative 13-file candidate remains the list in `final2-candidate-identities.txt`; coverage rows are proof mappings and may include ignored evidence or unchanged compatibility paths. The lowercase `project-files` mirror remains only the established builder/checker workaround. Its documentation checker no-base/D4 result is disclosed separately and is not the authoritative source D4.

Historical holds are retained as historical, not current blockers. The original qwen-phase12-text-runner sequence recorded Luna tester acceptance/release (#234, #236–#242), Sol verification (#244), Terra full/formal (#243/#245), Grok full/formal (#246/#248, with the formal tool text truncated to `P` and the complete PASS in #246), source claims #249–#257, and owner completion #259; tester PASS was test completion, not approval duty. Main #73 released Stage B after the Stage A artifact review/install. Terra #92 and Grok #93 then found that the initial runner was CPU-only and not a hybrid acceptance; Main #104 rejected both a CPU-only waiver and an unmeasured tolerance-conflict assumption. Main #137 later released the bounded hybrid correction after the P9/P10/P11 integration plan. The P9 GPU Q/K normalization, RoPE, and gate, P10 four GPU linear operations with authoritative convolution evidence, and P11 mapped `submitExperts`/shared GPU path are the accepted bounded components; the attention core remains CPU.

The correction also preserves the narrow reader correction from Main #187/#192: original data remains unchanged, default production schema requires the strict nine writer-v2 keys, and explicit synthetic schema requires those nine keys plus overflow-free `totalSize == biasesOffset + biasesSize`; unknown, missing, malformed, unequal, and overflowing fields reject. The P10 completion-race correction (#193–#206, with success dependency await at #228) requires cancellation-insensitive locked check/register, clear/publish before waiter extraction, exactly-once resume outside the lock, and await on success and both rollback paths. No `Task.yield`, spin, sleep, trap, forced restore, or ignored failed restore is accepted; failed restore is structured and nonreusable. Main #230 invalidated old final runs after source changes; final2 is the fresh exact candidate.

### Target state

Phase 12 is closed. The complete synthetic four-layer fixture constructs an actual normalized `Qwen3_5MoeForCausalLM` reference with an untied head, decodes the serialized BF16/INT4/INT8 virtual-file bytes before reference execution, and exercises the concrete Qwen runner over full-attention, convolution, recurrent, MoE, final-norm, cache, state, and raw-logit boundaries. Production family admission remains metadata-only until runtime file/hash/record/layout validation completes before runner construction. The 4×32 synthetic fixture cannot become an official verified Qwen `.gturbo`; positive authentic 35B payload execution remains outside Phase 12.

The fixed prompt lengths are `[1, 4, 7]`; the non-greedy stream is `[1, 4, 7, 2, 14]`; one-shot, split, multichunk, and token-at-a-time cached execution agree within the frozen value budget `1e-5` and state budget `2e-5`, with exact greedy IDs `[1, 12, 6]`. The observed one-shot maxima are hidden `7.1525574e-7`, logits `4.1723254e-7`, state `9.536743e-7`; token-at-a-time maxima are hidden `7.748604e-7`, logits `4.7683716e-7`, state `9.536743e-7`. These are observations, not new budgets. Input-norm, post-attention-norm, final-norm, and reset-cache controls produce the frozen rejecting deltas; no tolerance or fixture expectation was changed.

### Verification boundary

Final2 verification used 13 literal filter commands, all serial and nonempty, reporting 148 unique selected tests across 14 suites. The counts were ModelFamilyRuntimeTests 7, QwenTextModelTests 11, QwenTextRunnerTests 16, ModelLoaderTests 27, RawCompletionLoopTests 20, SamplerTests 11, QwenFullAttentionTests 6, QwenGatedDeltaNetTests 7, QwenMoETests 14, QwenFullAttentionKVTests 6, QwenLinearAttentionStateTests 11, PackedExpertsLayoutTests 4, and ModelRuntimeSchemaValidationTests 8. The Metal-validation QwenTextRunnerTests run executed 16 tests already included in the 148; it is overlapping validation evidence, not 16 additional tests, and 164 unique tests is rejected. No unfiltered suite was authorized or claimed.

`SamplerTests` used the authorized literal `Scripts/test.sh --filter SamplerTests`. Swift Testing applied substring selection, so 8 core SamplerTests and 3 AppMemorySamplerTests were selected. The three AppMemory tests were not inspected before execution; retrospective inspection found two injected process-footprint cases and one current-process sample/peak case, with no model, payload, or network operation. This is an explicit process deviation, not pre-execution inspection or retroactive authorization.

Debug and Release TurboFieldfare builds passed with no new compiler or concurrency diagnostics, and `git diff --check` exited 0. Metal API and GPU validation were enabled for QwenTextRunnerTests; 16/16 passed with no validation error. The exact candidate byte recheck found 13 files, aggregate match, and zero mismatches. The frozen oracle JSON digest is `e07372907e09f7deab72abd64f417b6f514953a098ca51d53845dc832ae6b1ec`; the approved generator digest is `97a05adb478241a66b54a6a0326558804f4597dd96498799f8b8cdef42c12620`; the frozen schema selector is `ModelRuntimeSchemaValidationTests`, not the empty selector used in the withdrawn history.

No authentic official 35B payload was loaded, hashed, executed, or qualified. No app installation, benchmark, performance result, new kernel migration, official tiny descriptor, fixture-tolerance change, or format-default weakening is claimed. The server remains loopback-only. Existing unrelated warnings remain historical and are not attributed to the corrected candidate.

### Tasks

#### 12.1 Define the session-level family runtime boundary

| | |
|---|---|
| Duty | `runner-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` |
| Depends on | `2.6, 3.5, 9.5, 10.5, 11.5` |
| Parallel safe | `yes` |
| Deliverable | Metadata-only family admission and a separate payload-validating runtime loader boundary. |

Classification reads validated manifest metadata and returns `gemmaV1` or official Qwen v2 `LoadedModelManifest`. Payload/hash/record/layout validation remains in `Runtime.load` before runner construction.

**Acceptance detail**
- [x] A verified v1 descriptor selects only Gemma, and a verified Qwen v2 manifest selects only Qwen.
- [x] Classification performs no payload, GPU, model, or runner work; hostile family metadata fails closed.

**How to check**
```sh
Scripts/test.sh --filter ModelFamilyRuntimeTests
```

**Evidence:** `final2-filters/ModelFamilyRuntimeTests.log` — 7 tests in 1 suite passed; exit 0. The exact filter was nonempty and the final candidate identity is recorded in `final2-candidate-identities.txt`.

#### 12.2 Map Qwen resident and streamed weights

| | |
|---|---|
| Duty | `runner-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Inference/Model.swift`; `Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift` |
| Depends on | `12.1` |
| Parallel safe | `no` |
| Deliverable | Map validated Qwen records through the private byte-backed mapper, preserving unchanged v1 loading and separately mapping the untied head. |

The exact JSON artifact is self-contained: tests extract and hash its serialized virtual files in a disposable directory, use no scratch generator, and reject missing, size, hash, duplicate, wrong-family, record, and layout failures before runner creation. The untied head remains distinct from embeddings.

**Acceptance detail**
- [x] Exact serialized fixture files are extracted and hashed in an isolated disposable directory; no scratch generator dependency is used by tests.
- [x] The untied head is distinct from embeddings, and every hostile file/record/layout mutation fails before runner creation.

**How to check**
```sh
Scripts/test.sh --filter QwenTextModelTests
```

**Evidence:** `final2-filters/QwenTextModelTests.log` — 11 tests in 1 suite passed; exit 0. The byte recheck and aggregate identity have zero mismatches.

#### 12.3 Implement concrete Qwen prefill and decode loops

| | |
|---|---|
| Duty | `runner-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` |
| Depends on | `12.2` |
| Parallel safe | `no` |
| Deliverable | Execute the complete synthetic four-layer decoder with explicit norms, attention/DeltaNet, MoE, residuals, final norm, untied head, and cache/state boundaries. |

The runner covers prompts `[1, 4, 7]`, stream `[1, 4, 7, 2, 14]`, all cache partitions, normalized intermediates, final hidden, and raw logits. Full KV, convolution, and recurrent state use explicit layout adapters. Missing input/post/final norms and stale or cleared cache controls fail the declared budgets rather than merely producing output; no reference-cache path is mislabeled as full recomputation.

**Acceptance detail**
- [x] One-shot, split, multichunk, and token-by-token cached outputs and states match the independently decoded serialized fixture within FP32 abs/rel `1e-5` for values and `2e-5` for state; IDs match exactly.
- [x] Missing input-layernorm, post-attention-layernorm, final norm, or stale/cleared cache controls fail the declared budget.

**How to check**
```sh
Scripts/test.sh --filter QwenTextRunnerTests
```

**Evidence:** `final2-filters/QwenTextRunnerTests.log` — 16 tests in 1 suite passed; exit 0. Frozen observed maxima are hidden `7.748604e-7`, logits `4.7683716e-7`, and state `9.536743e-7` for token-at-a-time; exact IDs and four controls passed. Metal validation is recorded separately.

#### 12.4 Apply Qwen stop IDs and explicit sampling inputs

| | |
|---|---|
| Duty | `runner-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Generation/Sampler.swift` |
| Depends on | `12.3` |
| Parallel safe | `no` |
| Deliverable | Pass family stop IDs and caller-selected sampling settings without applying Gemma transforms to Qwen logits or changing Gemma defaults. |

Gemma softcap `30` remains explicit; Qwen consumes raw logits without that softcap. Family stop IDs and caller sampling are explicit and not silently merged.

**Acceptance detail**
- [x] Qwen greedy sampling consumes raw un-softcapped logits and its explicit family stop IDs.
- [x] Gemma scale, softcap, defaults, and stop behavior remain unchanged in meaning.

**How to check**
```sh
Scripts/test.sh --filter QwenTextRunnerTests
```

**Evidence:** `final2-filters/QwenTextRunnerTests.log` — 16 tests in 1 suite passed; `final2-filters/SamplerTests.log` reports 11 tests across 2 suites: 8 core `SamplerTests` and 3 `AppMemorySamplerTests`, all nonempty and passing. The sampler substring-selection deviation is recorded above; no model or network operation occurred in the three retrospectively identified AppMemory tests.

#### 12.5 Add factory, loader, and runner tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` |
| Depends on | `12.1` |
| Parallel safe | `yes` |
| Deliverable | Add the focused family, model, and runner suites plus the exact reviewed JSON artifact. |

The three suites use the self-contained JSON artifact, not ignored scratch. They do not re-quantize, re-encode, inject direct Float values, import generator expectations, or create a parallel toy runner. Negative controls include wrong family, tied head, layer order, absent norms, stale cache, raw-vs-softcap, cache mode, and altered serialized bytes.

**Acceptance detail**
- [x] All three proposed suites execute with nonzero counts, no unexpected skips, and zero failures under the scoped filters.
- [x] Negative controls fail within the declared budget for wrong family, tied head, layer order, absent norms, stale cache, raw-vs-softcap, cache mode, and altered serialized bytes; exact bytes and hashes are checked from the JSON artifact.

**How to check**
```sh
Scripts/test.sh --filter 'ModelFamilyRuntimeTests|QwenTextModelTests|QwenTextRunnerTests'
```

**Evidence:** The three exact logs report 7, 11, and 16 tests, each in one suite, all exit 0. Across all 13 literal filters, 148 unique selected tests ran across 14 suites; the overlapping validation run is not added to that count.

#### 12.6 Re-run Gemma model and runner regression suites

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ModelLoaderTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests.swift` (as-is) |
| Depends on | `12.2, 12.3, 12.4, 12.5` |
| Parallel safe | `no` |
| Deliverable | Run the scoped existing Gemma loader, sampler, and raw-completion regressions after Stage B integration; preserve current Gemma behavior. |

Only the named filters were run, serially and nonempty; no unfiltered `Scripts/test.sh` run was authorized. The P9 broad-Vision failure remains historical/out of scope and is not recast as proven pre-existing.

**Acceptance detail**
- [x] Existing `ModelLoaderTests`, `SamplerTests`, and `RawCompletionLoopTests` execute with nonzero counts, no unexpected skips, and zero failures.
- [x] Gemma v1 load, generation, sampling, stop output, and resource behavior remain unchanged in meaning.

**How to check**
```sh
Scripts/test.sh --filter 'ModelLoaderTests|SamplerTests|RawCompletionLoopTests'
```

**Evidence:** `ModelLoaderTests` reports 27 tests in 1 suite, the literal `SamplerTests` filter reports 11 tests across 2 suites (8 core `SamplerTests` and 3 `AppMemorySamplerTests`), and `RawCompletionLoopTests` reports 20 tests in 1 suite; each exact filter exited 0. Compatibility paths remain unchanged test paths, not claimed modified files.

### Phase 12 ownership and artifact contract

The final2 source candidate and tests are frozen on the exact 13-file identity list. The JSON's virtual files are the only Phase 12 resource: there is no separate binary resource, manifest, receipt, official-install claim, or authentic-payload qualification. The ignored generator and its provenance evidence are audited outside the `Sources`, `Tests`, and `Package.swift` D4 union. The three added test paths, the inherited correction paths, and unchanged compatibility paths are all identified explicitly in the coverage table; no unlisted source or test path is claimed.

### Phase 12 coverage plan

The table has 16 proof rows. It is intentionally broader than the 13-file candidate: the ignored generator has evidence-only proof, three inherited/correction paths are explicit, and the two compatibility rows are unchanged test paths. The authoritative project D4 is the final2 four-stream source union, not the lowercase documentation mirror. Every executed proof is linked to a real final2 log or audit record.

| Code this phase changed | Test file that covers it | What the test or evidence proves | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | Metadata-only Gemma v1/Qwen v2 classification, wrong-family rejection, and no payload/GPU/model/runner work. | `Scripts/test.sh --filter ModelFamilyRuntimeTests` |
| `Sources/TurboFieldfare/Runtime/Inference/Model.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | Qwen byte-backed mapping plus unchanged v1 Gemma loader behavior. | `Scripts/test.sh --filter QwenTextModelTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextModel.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | Exact JSON byte/hash checks, validated mapping, untied head, and hostile load failures. | `Scripts/test.sh --filter QwenTextModelTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | Complete normalized schedule, raw logits, cache/state boundaries, awkward sizes, and exact IDs against an independently decoded-byte oracle. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `Sources/TurboFieldfare/Runtime/Generation/Sampler.swift` | `Tests/TurboFieldfare/Core/Runtime/Generation/SamplerTests.swift` | Core `SamplerTests` coverage is 8/8; the literal `--filter SamplerTests` also selected 3 `AppMemorySamplerTests`, for 11 tests across 2 suites. The three AppMemory tests are retrospective-only. | `Scripts/test.sh --filter SamplerTests` |
| `Package.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `QwenTextModelTests` runs 11/11 and proves the reviewed JSON bundle resource is registered, loadable through the package boundary, and identity-checked. | `Scripts/test.sh --filter QwenTextModelTests` |
| `scratch/qwen3.6-35b-a3b/fixture-generator/generate_qwen36_text_model_fixture.py` | evidence-only | Pinned generator identity, two byte-identical generations, independent artifact audit, and no scratch dependency in tests. | `final2-candidate-identities.txt` and `correction-requirement-map.txt` |
| `Tests/TurboFieldfare/Core/QwenFixtures/qwen36-tiny-text-model-fixtures.json` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | Exact relative paths/base64 bytes, byte counts, hashes, aggregate identity, virtual pack records, and oracle values. | `Scripts/test.sh --filter QwenTextModelTests` |
| `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | Nonzero execution, hostile metadata admission, and no production payload access. | `Scripts/test.sh --filter ModelFamilyRuntimeTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | Nonzero execution, exact JSON byte/hash checks, validated mapping, untied head, and hostile load failures. | `Scripts/test.sh --filter QwenTextModelTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | Nonzero execution, normalized intermediates/final norm/raw logits, cache partitions, state boundaries, exact IDs, and negative controls; no GPU skip is accepted as a pass. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/PackedExpertsLayout.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextModelTests.swift` | Primary changed-reader proof: exact-byte, `fixtureRedundantEnd`, 10-key, and `totalSize` checks run in `QwenTextModelTests` (11/11). Production regressions remain `PackedExpertsLayoutTests` 4/4, `ModelRuntimeSchemaValidationTests` 8/8, and `QwenMoETests` 14/14. | `Scripts/test.sh --filter QwenTextModelTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenLinearAttentionState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | Recurrent/convolution state shape, carry, reset, and ownership behavior remain covered. | `Scripts/test.sh --filter QwenLinearAttentionStateTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenLinearAttentionStateTests.swift` | The state suite executes 11 tests with zero failures. | `Scripts/test.sh --filter QwenLinearAttentionStateTests` |
| `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ModelLoaderTests.swift` (as-is) | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/ModelLoaderTests.swift` | Existing Gemma v1 loader behavior remains unchanged; this path was executed, not modified by Phase 12. | `Scripts/test.sh --filter ModelLoaderTests` |
| `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests.swift` (as-is) | `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests.swift` | Existing Gemma raw completion, stopping, cancellation, and resource behavior remains unchanged; this path was executed, not modified by Phase 12. | `Scripts/test.sh --filter RawCompletionLoopTests` |

### Phase 12 evidence

<a id="phase-12-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-16 | Final2 preflight, corrected Debug/Release builds, and focused technical gates | PASS; macOS 26.6.2, Xcode 26.5, Swift 6.3.2, arm64, 59 GiB free, 53% memory free, no prohibited process; Debug/Release builds and `git diff --check` exit 0 with no new diagnostics. | `scratch/qwen3.6-35b-a3b/evidence/phase-12/stage-b/final2-verification-summary.txt`; `final2-debug-build.log`; `final2-release-build.log`; `final2-git-diff-check.log` |
| 2026-09-16 | Thirteen serial literal exact-filter commands | PASS; all nonempty and exit 0. Counts 7, 11, 16, 27, 20, 11, 6, 7, 14, 6, 11, 4, and 8; 148 unique selected tests across 14 suites. | `scratch/qwen3.6-35b-a3b/evidence/phase-12/stage-b/final2-filters/summary.txt`; per-filter logs |
| 2026-09-16 | Metal API and GPU validation for QwenTextRunnerTests | PASS; validation enabled, 16/16 passed, no validation error. This overlaps QwenTextRunnerTests and is not an additional 16 unique tests. | `scratch/qwen3.6-35b-a3b/evidence/phase-12/stage-b/final2-qwen-text-runner-metal-validation.log` |
| 2026-09-16 | Exact candidate identity and serialized-byte recheck | PASS; candidate aggregate `869272236bf18564fbd0ef99e330d112a5e39de45009b680ae71aa96a7434a26`, 13 files, zero mismatches; oracle JSON `e07372907e09f7deab72abd64f417b6f514953a098ca51d53845dc832ae6b1ec`; generator `97a05adb478241a66b54a6a0326558804f4597dd96498799f8b8cdef42c12620`. | `scratch/qwen3.6-35b-a3b/evidence/phase-12/stage-b/final2-candidate-identities.txt`; `final2-candidate-byte-recheck.txt` |
| 2026-09-16 | Source D4 four-stream union and candidate classification | PASS; committed 0, unstaged 19, staged 0, untracked 52, union/status 71; 62 inherited plus 9 current-added paths; candidate 4 inherited plus 9 current-added paths. Nine is a dirty-inventory addition count, not a newly-created-file count. | `scratch/qwen3.6-35b-a3b/evidence/phase-12/stage-b/final2-source-d4-four-stream-union.txt`; `final2-closure-audit-receipt-v2.txt` |
| 2026-09-16 | Final2 closure audit and digest receipt | PASS; read-only identity, D4, filter-count, sampler-deviation, and HEAD guards all passed. The failed first audit attempt is retained as non-authoritative; it ran no product command and is not a product failure. | `scratch/qwen3.6-35b-a3b/evidence/phase-12/stage-b/final2-closure-audit-receipt-v2.txt`; `final2-closure-digests.txt`; `final2-closure-supplement.txt` |
| 2026-09-16 | Linked evidence-only supplement and Main acceptance | PASS; verification #26, Terra full/formal #31/#33, Grok full/formal #30/#32, claims #35–#38, owner #39, and Main #40 accepted unchanged exact engineering evidence. This documentation closeout did not rerun build, test, model, reference, performance, environment, or payload operations. | `scratch/qwen3.6-35b-a3b/evidence/phase-12/stage-b/final2-closure-supplement.txt` |
| 2026-09-16 | Historical Stage A artifact and Stage B planning entries | Historical only; synthetic CPU reference generation ran in the earlier authorized Stage A work, while the initial 7/7 runner test completion was not hybrid approval. Those holds are superseded by the final2 evidence above. | `scratch/qwen3.6-35b-a3b/evidence/phase-12/oracle/stage-a-summary.txt`; `scratch/qwen3.6-35b-a3b/evidence/phase-12/stage-b-status-documentation-recovery/verification-summary.txt` |

## Phase 13 - Qwen turns commit or roll back atomically

**When this is done:** qwen turns commit or roll back atomically

Needs: `12`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-16T17:03:11Z` — the phase preflight recorded macOS 26.6.2, Xcode 26.5, Swift 6.3.2, arm64, 59 GiB available on `/`, the existing worktrees, and Apple M2 Pro Metal 4. It preserved the inherited dirty source/test inputs and found no prohibited model or app process.
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-13/`

### Current state

P12 supplied the concrete Qwen runner and its full-attention KV, convolution, and recurrent state boundaries. P13 added the family-neutral transaction, Qwen committed/working aggregate, conversation facade, session/client prepared-token route, and app-boundary tests. The aggregate carries text tokens, full KV, convolution history, DeltaNet recurrence, RoPE/replay metadata, a pending accepted-but-unconsumed token, and valid FP16 logits scratch.

The correction2 candidate is the accepted engineering candidate `f9f620fd3d60d41bcb1d5becee5e0896c8eabfb18dd3b0a2d10a04d002d4b8c3` at HEAD `5770260510935cd32b82c4008ff06ae6458539ff`. It supersedes candidate `dfba0178ec563d127acfa675732644cc155821d6ab22b78010684214b443fc56`; verification #148 was rejected for missing declared gates and app acceptance proof, not for a changed product requirement. Correction2 was verified at #180, independently tested at #183; a tool-stage rejection at #184 was recovered by tester finishes #189/#197 and fresh identical-input verification #198. Terra #199 and Grok #202 approved, Sol #205 completed, and Main #206 accepted. No new technical runs were added for that workflow refresh.

### Target state

A Qwen turn begins from one committed boundary and writes one working state. Soft Stop commits accepted tokens and their matching state; hard task cancellation and errors roll back the whole aggregate. A matched hidden stop suffix is removed from visible tokens and state through bounded replay. Checkpoint rebuild preserves nonzero same-epoch lineage, restores producer state after failure, and materializes pending tokens exactly once. Capacity-triggered rebuild remains enabled, while Qwen does not use the uncalibrated Gemma sustained-slow threshold. Internal progress observation is awaited outside locks before stream publication and precommit; there is no fire-and-forget observer or throwing work after commit.

String, tool, and image Qwen routes explicitly remain unavailable until P14 and later phases. No authentic 35B payload, official payload execution, conversion, benchmark, performance result, app installation, or application release is claimed. Gemma behavior and artifacts remain retained.

### Tasks

#### 13.1 Define family-neutral state transaction semantics

| | |
|---|---|
| Duty | `state-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Generation/LogitProducer.swift`; `Sources/TurboFieldfare/Runtime/Generation/ConversationStateTransaction.swift` |
| Depends on | `12.6` |
| Parallel safe | `no` |
| Deliverable | The contract exposes begin, prefill, advance, commit, rollback, suffix removal, checkpoint rebuild, reset, retained tokens, and state bytes without naming KV internals. |

Gemma retains its existing adapter behavior; Qwen uses the concrete transaction. CPU task cancellation does not release submitted GPU resources.

**Acceptance detail**
- [x] Soft Stop, hard cancel, error, suffix trim, and reset have distinct observable semantics.
- [x] The interface adds no dynamic dispatch inside token/layer loops.

**How to check**
```sh
Scripts/test.sh --filter ConversationStateTransactionTests
```

**Evidence:** `c2-test-ConversationStateTransactionTests.log` — 5 tests in 1 suite passed; exit 0. The exact candidate and command receipt are in `correction2-candidate-identities.txt` and `c2-exact-command-receipts.txt`.

#### 13.2 Implement Qwen committed and working state

| | |
|---|---|
| Duty | `state-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` |
| Depends on | `13.1` |
| Parallel safe | `yes` |
| Deliverable | Hold one committed boundary and one working state containing full KV, convolution history, DeltaNet recurrence, token journal, and text RoPE delta. |

The implementation retains valid FP16 logits scratch and coordinates submitted GPU ownership before rollback or reuse. Snapshot/replay recovery preserves all text-state components instead of only a token position.

**Acceptance detail**
- [x] Rollback restores every text state component, not only token position.
- [x] Commit makes accepted generated tokens the next turn baseline.

**How to check**
```sh
Scripts/test.sh --filter QwenConversationStateTests
```

**Evidence:** `c2-test-QwenConversationStateTests.log` — 12 tests in 1 suite passed; exit 0. Coverage includes exact-once pending-token consumption, complete aggregate rollback, nonzero failed-checkpoint recovery, hidden suffix replay, stale-handle rejection, and submitted-GPU-owner reuse.

#### 13.3 Integrate transactions into conversation recovery

| | |
|---|---|
| Duty | `state-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Generation/MultimodalConversation.swift` |
| Depends on | `13.1, 13.2` |
| Parallel safe | `no` |
| Deliverable | Replace direct Qwen rewind assumptions with transaction calls while retaining existing Gemma `MultimodalConversationKVRecovery` behavior. |

Matched multi-token stop suffixes are removed from visible tokens and Qwen state through bounded replay from the committed boundary. The facade also carries the accepted-but-unconsumed token across a turn boundary exactly once.

**Acceptance detail**
- [x] Soft Stop commits accepted tokens; task cancellation and errors rollback.
- [x] Hidden stop removal leaves the next turn equal to a clean reference.

**How to check**
```sh
Scripts/test.sh --filter QwenConversationStateTests
```

**Evidence:** `c2-test-QwenConversationStateTests.log` — 12/12 pass; `facadeSoftStopCommitsAcceptedPendingTokenAndNextTurnConsumesItOnce`, `hiddenSuffixRemovesPendingThenConsumedTokenByReplay`, and rollback/reuse cases passed. The existing Gemma recovery suite also passed 5/5.

#### 13.4 Apply Qwen stop and compaction policy in the app session

| | |
|---|---|
| Duty | `state-code` |
| Touches | `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` |
| Depends on | `13.3` |
| Parallel safe | `no` |
| Deliverable | Route stop/cancel/reset/checkpoint calls to the family transaction and report family state bytes. |

Capacity rebuild remains available for Qwen. The Gemma sustained-slow performance trigger is not applied to Qwen, and Gemma's `15`-token/s meaning is unchanged.

**Acceptance detail**
- [x] Qwen sustained-slow evidence never triggers the Gemma threshold.
- [x] Capacity rebuild resets and replays all Qwen text state.

**How to check**
```sh
Scripts/test.sh --filter RealInferenceClientQwenRoutingTests
```

**Evidence:** `c2-test-RealInferenceClientQwenRoutingTests.log` — 8 tests in 1 suite passed; exit 0. It covers mutated-checkpoint cancellation, active logical bytes during hard cancellation, gated predecode cancellation, postdecode soft Stop commit, capacity-versus-sustained-slow policy, and clean session reuse.

#### 13.5 Add transaction and injected-boundary tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Runtime/Generation/ConversationStateTransactionTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenConversationStateTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` |
| Depends on | `13.1` |
| Parallel safe | `yes` |
| Deliverable | Tests inject failure before/after prefill, decode, GPU completion, stop match, commit, checkpoint rebuild, and reset. |

The dedicated transaction and Qwen state suites compare continuation against clean replay. The app suite exercises the prepared-token session/client boundary and shared generation registry, including cancellation after real replay mutation and cancellation before decode.

**Acceptance detail**
- [x] Both dedicated state suites execute with nonzero counts.
- [x] Every injected failure leaves the next turn equal to the clean reference.

**How to check**
```sh
Scripts/test.sh --filter QwenConversationState
```

**Evidence:** `c2-test-ConversationStateTransactionTests.log` reports 5/5 and `c2-test-QwenConversationStateTests.log` reports 12/12. `c2-test-RealInferenceClientStateTests.log` reports 21 tests across 2 suites, with the 8 routing cases overlapping the explicit routing run; all exit 0.

#### 13.6 Re-run existing Gemma recovery and stopping tests

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfare/Core/Runtime/Generation/MultimodalConversationKVRecoveryTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Stopping.swift`; `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Cancellation.swift` |
| Depends on | `13.3, 13.4, 13.5` |
| Parallel safe | `no` |
| Deliverable | Run current positional rewind, hidden stop, soft stopping, and cancellation suites after transaction integration. |

The existing Gemma recovery suite remains separate from the Qwen transaction implementation. The two RawCompletionLoop extension files execute as the registered `RawCompletionLoopTests` suite; their stop and cancellation behavior remains unchanged in meaning.

**Acceptance detail**
- [x] Both existing suites execute across three files with nonzero counts.
- [x] Gemma soft Stop and hard cancellation retain their current distinction.

**How to check**
```sh
Scripts/test.sh --filter MultimodalConversationKVRecoveryTests
```

**Evidence:** `c2-test-MultimodalConversationKVRecoveryTests.log` reports 5/5. The exact `RawCompletionLoopTests` filter in `c2-test-RawCompletionLoopTests.log` reports 20/20, including stopping and cancellation cases; both exit 0.

### Phase 13 coverage plan

The correction2 aggregate contains 12 scope paths: 8 actually edited files and 4 frozen, unchanged regression inputs. `QwenTextRunner.swift` is the intentional inherited P12 path changed by P13. The four frozen inputs remain listed because the phase acceptance explicitly required their compatibility proof. The direct mappings below follow the declarations in the named source/test files; the app routing suite is declared inside `RealInferenceClientStateTests.swift`, so its exact selector is recorded as 8/8 rather than inventing a second test path.

| Code this phase changed or regression input | Test file that covers it | What the test or evidence proves | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Runtime/Generation/LogitProducer.swift` (unchanged regression input) | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | Producer reset/commit and cancellation boundaries are exercised by the concrete Qwen runner tests. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `Sources/TurboFieldfare/Runtime/Generation/ConversationStateTransaction.swift` | `Tests/TurboFieldfare/Core/Runtime/Generation/ConversationStateTransactionTests.swift` | Begin, commit, rollback, stale/overlapping handles, reset, and replay metrics. | `Scripts/test.sh --filter ConversationStateTransactionTests` |
| `Sources/TurboFieldfare/Runtime/Generation/MultimodalConversation.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenConversationStateTests.swift` | Facade soft Stop, exact-once pending-token handoff, hidden suffix replay, and clean next-turn continuation. | `Scripts/test.sh --filter QwenConversationStateTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenConversationStateTests.swift` | Full KV/convolution/recurrent aggregate rollback, stale-handle rejection, submitted-GPU ownership, capacity rebuild, and nonzero failed-checkpoint recovery. | `Scripts/test.sh --filter QwenConversationStateTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | Existing runner producer transaction publication, FP16 scratch, and cancellation/reuse behavior after P13 integration. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | Prepared session/facade routing, logical committed bytes, registry ownership, injected predecode/replay cancellation, soft Stop commit, and Qwen policy. | `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests` |
| `Tests/TurboFieldfare/Core/Runtime/Generation/ConversationStateTransactionTests.swift` | self | Five nonzero transaction tests pass with zero failures. | `Scripts/test.sh --filter ConversationStateTransactionTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenConversationStateTests.swift` | self | Twelve nonzero Qwen aggregate/replay tests pass with zero failures. | `Scripts/test.sh --filter QwenConversationStateTests` |
| `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | self | The source selector executes 21 cases across the base and routing suites; its 8 routing cases overlap the explicit routing selector. | `Scripts/test.sh --filter RealInferenceClientStateTests` |
| `Tests/TurboFieldfare/Core/Runtime/Generation/MultimodalConversationKVRecoveryTests.swift` (unchanged regression input) | self | Existing Gemma positional rewind and hidden-stop recovery pass 5/5. | `Scripts/test.sh --filter MultimodalConversationKVRecoveryTests` |
| `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Stopping.swift` (unchanged regression input) | `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Stopping.swift` | Registered RawCompletionLoop stopping, hidden-token, and soft-stop cases execute in the suite. | `Scripts/test.sh --filter RawCompletionLoopTests` |
| `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Cancellation.swift` (unchanged regression input) | `Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Cancellation.swift` | Registered RawCompletionLoop cancellation and stop-vs-cancel cases execute in the suite. | `Scripts/test.sh --filter RawCompletionLoopTests` |

### Phase 13 evidence

<a id="phase-13-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-16 | Correction2 exact candidate and byte recheck | PASS; aggregate `f9f620fd3d60d41bcb1d5becee5e0896c8eabfb18dd3b0a2d10a04d002d4b8c3`, 12-file list, and post-gate byte match; baseline/current HEAD `5770260510935cd32b82c4008ff06ae6458539ff`. | `scratch/qwen3.6-35b-a3b/evidence/phase-13/correction2-candidate-identities.txt`; `c2-final-byte-recheck.log` |
| 2026-09-16 | Debug and Release target builds | PASS; TurboFieldfare and TurboFieldfareAppCore each built in Debug and Release with exit 0 and no diagnostics in the frozen logs. | `c2-debug-TurboFieldfare-build.log`; `c2-debug-TurboFieldfareAppCore-build.log`; `c2-release-TurboFieldfare-build.log`; `c2-release-TurboFieldfareAppCore-build.log`; `c2-exact-command-receipts.txt` |
| 2026-09-16 | Ordinary focused tests | PASS; 103 unique cases in 10 suites, all nonempty, zero failures, zero skips. `RealInferenceClientStateTests` selected 21 cases across 2 suites; its 8 routing cases are counted once. | `c2-test-*.log`; `c2-exact-command-receipts.txt` |
| 2026-09-16 | Metal API and shader validation | PASS; 49 unique validation-enabled cases, including 13 host-only cases, on the Apple GPU; all nonempty, zero failures, zero skips. `MTL_DEBUG_LAYER=1` and `MTL_SHADER_VALIDATION=1` were set before device creation; the source-selector run overlaps 8 routing cases. This is validation evidence, not 49 distinct GPU-exercising tests. | `c2-metal-*.log`; `c2-exact-command-receipts.txt` |
| 2026-09-16 | Authoritative source D4 and status guard | PASS; four-stream `Sources`/`Tests`/`Package.swift` union has committed 0, unstaged 22, staged 0, untracked 56, union/status 78; 71 inherited paths plus 7 newly dirty paths, with zero missing. This engineering D4 is separate from the documentation checker. | `c2-d4-four-stream-union.txt` |
| 2026-09-16 | P12 preservation and diff checks | PASS; 12 untouched P12 candidate files retain exact hashes; QwenTextRunner is the intentional P13 change. Tracked whitespace check exited 0 and untracked trailing-whitespace grep found no matches. | `c2-p12-preservation.log`; `c2-diff-whitespace-check.log` |
| 2026-09-16 | Application boundary proofs | PASS; prepared-token route, checkpoint cancellation after replay mutation, active logical-byte/hard-cancel behavior, predecode hard cancellation, postdecode soft Stop, capacity policy, and clean reuse all passed 8/8. | `c2-test-RealInferenceClientQwenRoutingTests.log`; `c2-metal-RealInferenceClientQwenRoutingTests.log` |
| 2026-09-16 | Review and acceptance trail | PASS; correction2 verification #180, tester #183, recovered fresh identical-input verification #198, Terra #199, Grok #202, Sol #205, and Main #206 accepted the exact candidate. Superseded verification #148 and candidate aggregate remain disclosed above. | `c2-final-verification-summary.txt`; `correction2-candidate-identities.txt` |
| 2026-09-16 | Scope limitation | OPEN; Qwen string/tool/image codec work awaits P14 and later phases. No authentic 35B payload, network, conversion, benchmark, performance, app-install, or application-release proof is claimed. | `c2-final-verification-summary.txt` |

## Phase 14 - Qwen prompts match the pinned chat template

**When this is done:** qwen prompts match the pinned chat template

Needs: `1`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Candidate: frozen five-path aggregate `de5b7863d40820aad311287a9711acc626706bfa53cda4e03cff50a34414829a`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/`
Status: accepted `2026-09-17`; four of four tasks, three of three acceptance items, five of five coverage rows, and D1–D5 pass.

### Current state

Phase 14 is accepted for the tokenizer and chat-codec boundary. `QwenTokenizer.load(from:)` admits one verified Qwen v2 model directory through bounded manifest bytes and descriptor-relative, no-follow sidecar reads. The separate internal `loadOfficialSidecar(from:)` seam uses isolated regular copies for tokenizer-only tests and does not masquerade as installed admission. Qwen trim behavior matches Python scalar `str.strip`, the LF-only helpers remain unchanged, and the private JSON notation adapter addresses the four demonstrated finite `String(Double)` mismatches.

The accepted candidate is HEAD `5770260510935cd32b82c4008ff06ae6458539ff`. Independent approvals `#119` and `#120` approved verification `#117`; Main accepted the exact candidate in event `#132`. The original failed candidate is preserved at aggregate `0129f5510eb16b043196acff3443c5707efa9de9365a7900c5874b91cf72e64d` under `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/rejected-candidate-0129f551/`, with receipt `rejected-candidate-0129f551-identities.txt`. Earlier failures and evidence remain historical and unchanged.

### Target state

Delivered P14 behavior is limited to model-neutral message/codec types, verified Qwen tokenizer admission and incremental decoding, exact supported Qwen chat rendering, isolated authentic sidecar fixtures, and independent tokenizer/template tests. Existing `GFTokenizer` and Gemma chat behavior remain unchanged in meaning. The accepted boundary does not include authentic 35B inference, GPU execution, image payloads, conversion, application/service/CLI/server qualification, performance, or product readiness.

### Tasks

#### 14.1 Define model-neutral chat message and codec types

| | |
|---|---|
| Duty | `chat-code` |
| Touches | `Sources/TurboFieldfare/Tokenization/ModelChatCodec.swift` |
| Depends on | `1.4` |
| Parallel safe | `yes` |
| Deliverable | Create narrow message, content-part, tool-definition, historical-call, prompt-render, detokenization, and special-token capabilities used by product call sites. |

Design from existing `GFTokenizer` call sites. Keep Gemma implementation behavior behind an adapter; do not rename it yet.

**Acceptance detail**
- [x] The contract represents text, image, tool call/result, and thinking state without family tokens.
- [x] Clients request behavior rather than access the underlying tokenizer object.

**How to check**
```sh
Scripts/test.sh --filter QwenChatTemplateTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-chat-template.log` — 21/21 Qwen chat-template tests passed; exit 0.

#### 14.2 Load the verified Qwen tokenizer sidecars

| | |
|---|---|
| Duty | `chat-code` |
| Touches | `Sources/TurboFieldfare/Tokenization/QwenTokenizer.swift` |
| Depends on | `14.1` |
| Parallel safe | `yes` |
| Deliverable | Load vocabulary, merges, tokenizer config, chat template, decoder rules, and special IDs from the verified model directory. |

Require pinned sidecar digests and reject video/audio use in initial scope. Do not fetch tokenizer assets.

**Acceptance detail**
- [x] Encode/decode and incremental detokenization match pinned tokenizer fixtures.
- [x] Wrong/missing sidecars or special IDs fail before prompt rendering.

**How to check**
```sh
Scripts/test.sh --filter QwenTokenizerTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-tokenizer.log` — 25/25 Qwen tokenizer tests passed; exit 0.

#### 14.3 Implement exact Qwen chat rendering

| | |
|---|---|
| Duty | `chat-code` |
| Touches | `Sources/TurboFieldfare/Tokenization/QwenChatCodec.swift` |
| Depends on | `14.1, 14.2` |
| Parallel safe | `no` |
| Deliverable | Reproduce the pinned template for system/user/assistant/tool roles, multi-step tools, images, generation prompt, and thinking on/off. |

Compile the supported template behavior into Swift; do not execute remote code or a mutable template at runtime.

**Acceptance detail**
- [x] Golden token IDs match pinned ordinary, tool, image, and post-tool prompts.
- [x] Thinking disabled emits the exact empty thought framing required by the template.

**How to check**
```sh
Scripts/test.sh --filter QwenChatTemplateTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-chat-template.log` — 21/21 Qwen chat-template tests passed; exit 0.

#### 14.4 Add tokenizer and template golden tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Tokenization/QwenTokenizerTests.swift`; `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` |
| Depends on | `14.1` |
| Parallel safe | `yes` |
| Deliverable | Tests cover Unicode/BPE boundaries, incremental bytes, every role, thinking modes, image markers, multiple tools, consecutive tool results, and invalid message order. |

Expected prompts/token IDs come from the pinned template evidence, not production rendering code.

**Acceptance detail**
- [x] Both suites execute with nonzero counts.
- [x] Any role, marker, newline, thought, or tool-result framing drift fails.

**How to check**
```sh
Scripts/test.sh --filter QwenChat
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-tokenizer.log`; `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-chat-template.log`; `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-combined-tokenizer-selector.log`; `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-combined-chat-template-selector.log` — focused and combined selectors passed with zero skip events; exit 0.

### Phase 14 coverage plan

The P14 scope covers exactly five paths. The correction edited four files; `Sources/TurboFieldfare/Tokenization/ModelChatCodec.swift` was in scope but unchanged by the correction. The focused Qwen selectors report 25 tokenizer cases and 21 chat-template cases, each with zero skip events. The intentionally broad selectors report 56 cases in two suites for `TokenizerTests` and 30 cases in two suites for `ChatTemplateTests`; those combined counts are not presented as Qwen-only counts.

| P14 scope path | Test file that covers it | What the test or evidence proves | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Tokenization/ModelChatCodec.swift` (scope path; unchanged by correction) | `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | Model-neutral message/content/tool shapes and unchanged Gemma adapter delegation. | `Scripts/test.sh --filter QwenChatTemplateTests` |
| `Sources/TurboFieldfare/Tokenization/QwenTokenizer.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenTokenizerTests.swift` | Pinned metadata/BPE/decode behavior, sidecar parsing, verified installed admission, bounded no-follow reads, typed asset errors, and hostile matrix. | `Scripts/test.sh --filter QwenTokenizerTests` |
| `Sources/TurboFieldfare/Tokenization/QwenChatCodec.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | Exact role/thinking/tool/image rendering, Python scalar trimming, newline-helper preservation, JSON number parity, and nonfinite rejection. | `Scripts/test.sh --filter QwenChatTemplateTests` |
| `Tests/TurboFieldfare/Core/Tokenization/QwenTokenizerTests.swift` | self | The Qwen tokenizer suite executes 25 tests in one suite, zero skips. | `Scripts/test.sh --filter QwenTokenizerTests` |
| `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | self | The Qwen chat suite executes 21 tests in one suite, zero skips. | `Scripts/test.sh --filter QwenChatTemplateTests` |

### Phase 14 evidence

<a id="phase-14-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-16 | Read-only planning/reference preparation | Historical only; original result: Pending; no production or test implementation is written or released. Main journal 52 releases only Sol's scratch-owned isolated tokenizer/template oracle stage in the existing pinned offline environment; no model, weights, GPU, inference, or downloads. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/stage-a-baseline.txt` |
| 2026-09-17 | Qwen tokenizer selector | PASS; 25 tests in 1 suite, zero skip events; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-tokenizer.log` |
| 2026-09-17 | Qwen chat-template selector | PASS; 21 tests in 1 suite, zero skip events; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-qwen-chat-template.log` |
| 2026-09-17 | Combined `TokenizerTests` selector | PASS; 56 tests in 2 suites (Gemma plus Qwen), zero skip events; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-combined-tokenizer-selector.log` |
| 2026-09-17 | Combined `ChatTemplateTests` selector | PASS; 30 tests in 2 suites (Gemma plus Qwen), zero skip events; exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-test-combined-chat-template-selector.log` |
| 2026-09-17 | Debug build | PASS; `swift build --target TurboFieldfare`, exit 0; no candidate diagnostics. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-build-debug.log` |
| 2026-09-17 | Release build | PASS; `swift build -c release --target TurboFieldfare`, exit 0; no candidate diagnostics. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/correction-2-build-release.log` |
| 2026-09-17 | D4 and preservation checks | PASS; source/status union remains 83 paths, historical evidence remains 36 files, and rejected archive aggregate remains `0129f551...`. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/candidate-source-d4.txt`; `d4-comparison.txt`; `historical-evidence-preservation.txt`; `archive-preservation-check.txt` |
| 2026-09-17 | Independent reference checks | PASS; trim and 47-value JSON references are byte-identical across two runs. Four finite mismatches justified a private notation-only adapter. The 100,000-finite-value comparison is explicitly sampled, not exhaustive. | `scratch/qwen3.6-35b-a3b/evidence/phase-14/correction/trim-chat-reference-receipt.txt`; `double-json-reference-receipt.txt`; `double-format-comparison.txt`; `double-formatter-domain-probe-receipt.txt` |

The accepted fixture policy uses isolated copied sidecars, not symlinks, hard links, or official source inodes. The hostile matrix includes malformed, oversize, nonregular, symlink, missing, tampered, wrong-family, and admission-boundary cases. The existing Gemma-v2 success assertion precedes Qwen-only family rejection, so this matrix is non-tautological.

A process-attribution deviation is recorded only in the closeout evidence: navigator #119 performed genuine hash/status/JSON checks but also ran unauthorized shell commands and wrote `/tmp` scratch files despite a no-command assignment. Main inspected the original session tool records at lines 300–361 and found no source, test, asset, build, test, model, or reference-generation edits/runs in that sequence. Technical acceptance rests on the unchanged candidate and Sol's actual receipts; this is not perfect workflow compliance and is not retroactive authorization.

P15 is pending and ready to begin; no P15 implementation is included here.

---

## Phase 15 - Incomplete Qwen tool output dispatches nothing

**When this is done:** incomplete Qwen tool output dispatches nothing

Needs: `14`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `2026-09-17T18:29:52Z — macOS 26.6.2, arm64 Mac14,12 with 32 GiB RAM; 61 GiB free, 48% memory free, no matching prohibited model process; no model run authorized or performed.`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/`

### Current state at phase start (historical)

`StructuredAssistantDecoder.swift:89-422` and `GemmaToolCallParser.swift:14-405` parse Gemma-specific output. Qwen emits thought tags and XML-like `<tool_call><function=...><parameter=...>` blocks with no model-supplied call ID. Host VisionCapture validation already owns permissions and returned truth.

### Target state

A family-selected Qwen structured decoder incrementally separates thinking, visible text, complete calls, and malformed output. Only complete schema-valid calls become host call proposals; the host generates IDs. VisionCapture’s verified results and replay/permission rules remain authoritative and unchanged.

### Delivered state

The target state is accepted on HEAD `5770260510935cd32b82c4008ff06ae6458539ff` with frozen five-path aggregate `9fd6d7f6bb45dbeb15e12bdd2e4f76fca6bfa6999aea1bb4c6b667556c28e84a`. Publication is finish-only and all-or-none, and host IDs are allocated only at successful release. EOS publishes nothing and permits one terminal tail flush. A 256 KiB aggregate retained-call budget and a 128-container non-string JSON depth limit bound parsing before Foundation decoding. The schema subset fails closed; decimal/integer storage remains deliberately narrow, canonically equivalent duplicate keys are rejected, and progress reporting remains Gemma-oriented.

This is an additive family-selected parsing seam. It does not establish authentic Qwen inference, tool execution, product integration, GPU behavior, performance, or product readiness.

All applicable Phase 15 commands require the current `AGENTS.md` preflight, serial execution, and already-local assets. Classified tokenizer-asset selectors run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; these commands never authorize downloads.

### Tasks

#### 15.1 Implement incremental Qwen structured decoding

| | |
|---|---|
| Duty | `tool-code` |
| Touches | `Sources/TurboFieldfare/Tokenization/QwenStructuredAssistantDecoder.swift` |
| Depends on | `14.4` |
| Parallel safe | `yes` |
| Deliverable | Incrementally classify thought text, final text, tool-call boundaries, stop tokens, and unfinished output across arbitrary token splits. |

Do not expose incomplete tags or parameter bodies as executable calls.

**Acceptance detail**
- [x] Every prefix has deterministic visible/thought/call state.
- [x] EOF inside any tag yields malformed output and zero executable calls.

**How to check**
```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser|StructuredAssistantDecoder)Tests'
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-qwen-two-suites-frozen.log` — 36 tests in 2 suites passed, including 16 decoder tests; exit 0 with no skip or warning marker.

#### 15.2 Parse and validate Qwen tool grammar

| | |
|---|---|
| Duty | `tool-code` |
| Touches | `Sources/TurboFieldfare/Tokenization/QwenToolCallParser.swift` |
| Depends on | `15.1` |
| Parallel safe | `yes` |
| Deliverable | Parse nested function/parameter framing, preserve multiline values, reject duplicates/ambiguity, validate allowed names/schema, and assign host-owned call IDs only after completion. |

Never repair malformed model text into an action. Unknown tools and invalid arguments remain non-executable evidence.

**Acceptance detail**
- [x] Complete allowed calls round trip through Qwen chat result framing.
- [x] Malformed, unknown, duplicate, and partial calls produce no executable proposal.

**How to check**
```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter releasedCallRoundTripsThroughQwenAssistantAndToolResultFraming
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser|StructuredAssistantDecoder)Tests'
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-roundtrip-focused-final.log`; `test-qwen-two-suites-frozen.log` — an actually released host-ID call round-tripped through unchanged QwenChatCodec assistant/tool-result framing and reparsed with equal raw-string and non-string arguments; the focused proof passed 1/1 and the parser suite passed 20/20.

#### 15.3 Select structured decoding through the codec boundary

| | |
|---|---|
| Duty | `tool-code` |
| Touches | `Sources/TurboFieldfare/Tokenization/StructuredAssistantDecoder.swift` |
| Depends on | `15.1, 15.2` |
| Parallel safe | `no` |
| Deliverable | Add family dispatch that keeps the existing Gemma decoder/parser intact and chooses Qwen only from a verified descriptor. |

Host tool definitions remain the allowlist; model family cannot widen permissions.

**Acceptance detail**
- [x] Gemma outputs select the existing decoder.
- [x] A descriptor/output family mismatch fails instead of trying both grammars.

**How to check**
```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser|StructuredAssistantDecoder)Tests'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter GemmaToolCallTests
```

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-qwen-two-suites-frozen.log`; `test-gemma-tool-call-frozen.log` — 16 Qwen decoder tests and 9 unchanged Gemma tests passed, including descriptor mismatch without grammar fallback.

#### 15.4 Add split-token and malformed-output tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift`; `Tests/TurboFieldfare/Core/Tokenization/QwenToolCallParserTests.swift` |
| Depends on | `15.1` |
| Parallel safe | `yes` |
| Deliverable | Tests split every marker position, cover multiple calls/parameters, thought-to-call transitions, stop before close, invalid schema, unknown tool, duplicate parameter, and hostile tag text. |

Assert proposals, not private parser call order, and assert zero executable calls for every malformed case.

**Acceptance detail**
- [x] Both suites execute with nonzero counts.
- [x] Every incomplete or ambiguous case produces zero executable host calls.

**How to check**
```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser|StructuredAssistantDecoder)Tests'
```

The earlier parser-only `Scripts/test.sh --filter QwenTool` run remains historical evidence, not the combined phase selector.

**Evidence:** `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-qwen-two-suites-frozen.log` — 36 tests in both named suites passed with nonzero counts and no skip marker.

#### 15.5 Re-run Gemma parser and host truth tests unchanged

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfareServer/OpenAIValidationTests.swift`; `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` |
| Depends on | `15.3, 15.4` |
| Parallel safe | `no` |
| Deliverable | Run existing Gemma structured-output and VisionCapture returned-identity suites after codec dispatch is added. |

Do not change `VisionCaptureToolLoop.swift` in this phase and do not weaken host verified-result assertions.

**Acceptance detail**
- [x] Existing Gemma call parsing executes with a nonzero count.
- [x] VisionCapture remains the source of truth for returned identity and verification.

**How to check**
```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter GemmaToolCallTests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter OpenAIValidationTests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter MultimodalConversationCanonicalizationTests
Scripts/test.sh --filter StructuredOutputDiagnosticsTests
Scripts/test.sh --filter VisionCaptureReturnedIdentityTests
```

**Evidence:** The five frozen regression receipts under `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/` are `test-gemma-tool-call-frozen.log` (9/9), `test-openai-validation-frozen.log` (44/44 in 4 suites), `test-multimodal-canonicalization-frozen.log` (11/11), `test-structured-output-diagnostics-frozen.log` (5/5), and `test-vision-capture-identity-frozen.log` (11/11).

### Phase 15 coverage plan

Every changed or intentionally re-run path has its own row. The seven rows are tracker coverage rows; test-case counts overlap and must not be added together. In particular, the wrapper's 25 checks reuse 16 decoder and 9 Gemma tests, while the 44-case OpenAI selector contains 19 OpenAI request-validation cases, 9 Gemma cases, and supporting suites.

| Code this phase changes | Test file that covers it | What the test or evidence proves | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Tokenization/QwenStructuredAssistantDecoder.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift` | Finish-only all-or-none release, release-time host IDs, EOS plus one tail, arbitrary splits, malformed-tail rollback, aggregate-size and depth limits, and family mismatch; 16/16 pass. | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser\|StructuredAssistantDecoder)Tests'` |
| `Sources/TurboFieldfare/Tokenization/QwenToolCallParser.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenToolCallParserTests.swift` | Exact grammar and raw strings, recursive supported schema, fail-closed unknown/duplicate/unsupported/deep/precision cases, and typed failures; 20/20 pass. | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser\|StructuredAssistantDecoder)Tests'` |
| `Sources/TurboFieldfare/Tokenization/StructuredAssistantDecoder.swift` | `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift`; `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | Verified-descriptor family selection, no fallback, and unchanged descriptor-free Gemma behavior; 25/25 pass from 16 Qwen decoder plus 9 Gemma cases. | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser\|StructuredAssistantDecoder)Tests'`; `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter GemmaToolCallTests` |
| `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift` | self | The decoder suite executes 16/16, including a real released-call round trip through unchanged QwenChatCodec assistant/tool-result framing and reparse equality. | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser\|StructuredAssistantDecoder)Tests'` |
| `Tests/TurboFieldfare/Core/Tokenization/QwenToolCallParserTests.swift` | self | The parser suite executes 20/20 with no skips. | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter 'Qwen(ToolCallParser\|StructuredAssistantDecoder)Tests'` |
| `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` (unchanged) | self | The unchanged file executes inside 44 passing tests across 4 suites; the named OpenAI suite has 19 cases and the included Gemma suite has 9. | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Scripts/test.sh --filter OpenAIValidationTests` |
| `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` (unchanged) | self | VisionCapture returned identity and host verification remain authoritative; 11/11 pass. | `Scripts/test.sh --filter VisionCaptureReturnedIdentityTests` |

The historical parser-only `Scripts/test.sh --filter QwenTool` run remains preserved but is not the combined new-suite acceptance command.

### Phase 15 evidence

<a id="phase-15-evidence"></a>

Accepted identity: HEAD `5770260510935cd32b82c4008ff06ae6458539ff`, frozen five-path aggregate `9fd6d7f6bb45dbeb15e12bdd2e4f76fca6bfa6999aea1bb4c6b667556c28e84a`. Independent test signoff is `#172` and `#179`; verification submission `#176` was approved by `#177` and `#178`; Main accepted the same identity in `#189`.

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-17 | Final offline environment and asset preflight | PASS; macOS/Swift/capacity/memory, no prohibited process, completed Gemma pack/cache, pinned Qwen sidecars, and offline variables recorded against the accepted aggregate. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/preflight-final.txt` |
| 2026-09-17 | Released-call QwenChatCodec round trip | PASS; 1 test in 1 suite, exit 0. A real `finish()` release with host ID renders raw multiline Unicode and canonical non-string JSON through unchanged assistant/tool-result framing, omits the ID on wire, retains it in host metadata, and reparses to equal arguments. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-roundtrip-focused-final.log` |
| 2026-09-17 | Combined Qwen decoder/parser selector | PASS; 36 tests in 2 suites, zero skip or warning markers, exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-qwen-two-suites-frozen.log` |
| 2026-09-17 | Frozen Gemma tool-call regression | PASS; 9/9, exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-gemma-tool-call-frozen.log` |
| 2026-09-17 | Frozen OpenAI validation regression | PASS; 44 tests in 4 suites, exit 0 with no signal line. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-openai-validation-frozen.log` |
| 2026-09-17 | Frozen multimodal canonicalization regression | PASS; 11/11, exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-multimodal-canonicalization-frozen.log` |
| 2026-09-17 | Frozen structured diagnostics regression | PASS; 5/5, exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-structured-output-diagnostics-frozen.log` |
| 2026-09-17 | Frozen VisionCapture identity regression | PASS; 11/11, exit 0. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-vision-capture-identity-frozen.log` |
| 2026-09-17 | Debug and Release `TurboFieldfare` builds | PASS; both exit 0 with no candidate diagnostics. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/implementation-build-debug-frozen.log`; `implementation-build-release-frozen.log` |
| 2026-09-17 | D4, preservation, hygiene, and UUID sidecar-copy cleanup | PASS; P14's 83-path union plus exactly five P15 paths, accepted P14 aggregate and frozen inputs unchanged, 36 historical evidence files unchanged, no added concurrency escape hatch, and no copied-sidecar directory remaining. | `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/candidate-source-d4.txt`; `d4-comparison.txt`; `preservation-identities.txt`; `historical-evidence-preservation.txt`; `text-hygiene-concurrency-cleanup.txt` |

The real decimal-rounding negative control is preserved at `scratch/qwen3.6-35b-a3b/evidence/phase-15/negative-control-decimal-precision-rounding.log`: it demonstrated that a precision-losing decimal was silently widened before the fix. The parent directory also preserves the parser-risk negative control and the stale-linked-test-bundle OpenAI crash plus its clean rebuilt-bundle recovery. None is relabelled as a pass.

The first focused round-trip receipt at `scratch/qwen3.6-35b-a3b/evidence/phase-15/roundtrip-correction/test-roundtrip-focused.log` is an exit-1 test-design failure: sorted parameters invalidated an order assumption and extraction selected the instructional example tool call. The corrected focused and combined receipts pass. Flash self-reported one prohibited early literal `echo` command in events `#78` and `#146`; it produced no technical check beyond that literal, is not retroactively authorized, and does not establish a complete session audit. Final reviewers inspected receipts; they did not rerun the commands.

The accepted behavior is deliberately bounded: finish-only release with zero premature IDs, EOS plus one non-publishing tail, a 256 KiB aggregate retained-call budget, a 128-container non-string JSON depth limit, and a fail-closed schema subset. Integer-versus-decimal storage is narrower than broad JSON Schema equivalence, canonically equivalent duplicate keys are rejected, and progress reporting remains Gemma-oriented. No authentic Qwen model inference, tool execution, app/service/CLI/server integration, GPU behavior, performance, deployment, or product readiness is claimed.

---

## Phase 16 - Qwen still images produce matching token rows

**When this is done:** qwen still images produce matching token rows

Needs: `2, 3, 8, 12, 13`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff`
Disk baseline: `preflight-final.txt — macOS 26.6.2 (25G83), arm64, Swift 6.3.2, Xcode 26.5, 37 GiB available, 79% memory free; process and required-pack gates passed`
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-16/`; exact correction identity and final receipts are under `transaction-correction/`; docs closeout evidence is under `docs-closeout/`.

### Current state

`VisionConfig.swift:3-33`, `Gemma4ImageGeometry.swift:3-76`, `Gemma4ImagePreprocessor.swift:18-239`, `VisionRuntime.swift:93-902`, and `MultimodalPromptRenderer.swift:18-121` hard-code Gemma geometry, tower names, dimensions, and token IDs. `MultimodalPrefillInput.swift:14-97` provides useful feature/span validation.

### Target state

A separate Qwen path validates the v2 companion, plans bounded dynamic grids, preprocesses pixels, executes the 27-layer tower/merger, expands image pads, supplies temporal/height/width interleaved M-RoPE, and serially extends P13 snapshots with image lineage. Video is rejected explicitly.

### Tasks

#### 16.1 Implement Qwen image geometry and preprocessing

| | |
|---|---|
| Duty | `vision-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenImagePreprocessor.swift`; `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionConfig.swift` |
| Depends on | `2.6, 3.5, 8.4, 12.6, 13.6` |
| Parallel safe | `yes` |
| Deliverable | Compute overflow-checked dynamic width/height grids for patch 16, temporal patch 2, merge 2, configured pixel bounds, mean 0.5, and std 0.5. |

Keep existing encoded-byte/image-count admission as an outer bound; do not adopt the publisher maximum as a product budget without measurement.

**Acceptance detail**
- [x] Portrait, landscape, tiny, and awkward inputs produce pinned grids.
- [x] Overflow, zero size, excessive decoded bytes, and video fail before allocation.

**How to check**
```sh
Scripts/test.sh --filter QwenImagePreprocessorTests
```

**Evidence:** Historical frozen `frozen-qwenimagepreprocessortests.log/.status`, 6/6 PASS; valid for the final candidate through `transaction-correction/preservation-comparison-final.txt` (23/23 unchanged inputs), not a post-correction rerun.

#### 16.2 Validate and map the Qwen vision companion

| | |
|---|---|
| Duty | `vision-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Vision/VisionWeightStore.swift`; `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift` |
| Depends on | `16.1` |
| Parallel safe | `yes` |
| Deliverable | Dispatch companion loading by v2 family and bind model revision, processor profile, text manifest digest, receipt, files, and tensor regions before mapping. |

Keep existing v1 store and sidecar tolerance unchanged. Missing or invalid Qwen vision means image support unavailable, never text-only image acceptance.

**Acceptance detail**
- [x] Only the matching companion maps.
- [x] Wrong family/revision/digest/profile, missing receipt, corrupt size, and unsupported region fail closed.

**How to check**
```sh
Scripts/test.sh --filter QwenVisionWeightStoreTests
```

**Evidence:** Historical frozen `frozen-qwenvisionweightstoretests.log/.status`, 9/9 PASS; exact 333-tensor, 893,142,496-byte official-layout synthetic companion lifecycle. It is preserved-input evidence, not authentic companion inference.

#### 16.3 Implement the 27-layer Qwen vision tower and merger

| | |
|---|---|
| Duty | `vision-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Vision/VisionRuntime.swift` (planned but byte-unchanged); `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift` |
| Depends on | `16.1, 16.2` |
| Parallel safe | `no` |
| Deliverable | Create a concrete Qwen tower/merger selected outside its layer loop, using verified names/shapes and bounded in-flight command buffers. |

Reuse validated vision linear/attention mechanisms where layout matches; do not force Qwen weights through Gemma projector names or fixed tokens.

**Acceptance detail**
- [x] Tiny tower intermediates and merged 2,048-wide rows match P3.
- [x] Cancellation retains resources until submitted GPU work completes.

**How to check**
```sh
Scripts/test.sh --filter QwenVisionRuntimeTests
```

**Evidence:** Historical frozen `frozen-qwenvisionruntimetests.log/.status` and `frozen-metal-validation-qwenvisionruntimetests.log/.status`, 11/11 PASS with Metal API and GPU validation. Numerical proof is bounded to the frozen P3 reduced fixture, reduced ordered 27-layer proof, and official-layout synthetic lifecycle.

#### 16.4 Implement pad expansion and multimodal positions

| | |
|---|---|
| Duty | `vision-code` |
| Touches | `Sources/TurboFieldfare/Runtime/Vision/MultimodalPromptRenderer.swift`; `Sources/TurboFieldfare/Runtime/Vision/MultimodalPrefillInput.swift`; `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenMultimodalPositions.swift` |
| Depends on | `16.1, 16.3` |
| Parallel safe | `no` |
| Deliverable | Expand each Qwen image marker to its dynamic merged-row count, validate feature rows, emit modality/grid metadata, and compute interleaved temporal/height/width M-RoPE plus decode delta. |

Never reuse Gemma token IDs or fixed 256-row assumptions. Multiple images preserve prompt order.

**Acceptance detail**
- [x] Every image pad span has exactly one matching feature row per token.
- [x] An asymmetric-grid fixture detects any M-RoPE axis swap.

**How to check**
```sh
Scripts/test.sh --filter QwenMultimodalPositionTests
```

**Evidence:** Historical frozen `frozen-qwenmultimodalpositiontests.log/.status`, 5/5 host-only PASS; no Metal-validation claim is made for this host-only selector.

#### 16.5 Add image lineage to the Qwen state snapshot

| | |
|---|---|
| Duty | `state-handoff` |
| Touches | `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` |
| Depends on | `16.4` |
| Parallel safe | `no` |
| Deliverable | After P13 text state is stable, extend the committed/working snapshot with image digest, processor profile, grid metadata, retained-feature lineage, and M-RoPE delta. |

This is the only P16 write to the P13-owned file; ownership transfers serially after task 13.6.

**Acceptance detail**
- [x] Rollback/rebuild restores matching image lineage and positions.
- [x] A changed image or processor profile invalidates retained features.

**How to check**
```sh
Scripts/test.sh --filter QwenVisionConversationStateTests
```

**Evidence:** Exact-final-candidate `transaction-correction/frozen-vision-state-tests.log/.status`, 18/18 PASS, and `frozen-metal-validation-vision-state-tests.log/.status`, 18/18 PASS with Metal API and GPU validation. These are actor-seam state proofs, not end-to-end product image inference.

#### 16.6 Add concrete Qwen vision and M-RoPE kernels

| | |
|---|---|
| Duty | `vision-metal` |
| Touches | `Sources/TurboFieldfare/Metal/Qwen/qwen_vision.metal` |
| Depends on | `16.3, 16.4, 8.4` |
| Parallel safe | `no` |
| Deliverable | Implement Qwen-specific patch/position, merger, and three-axis interleaved M-RoPE kernels not covered by validated existing vision kernels. |

Query device limits, bound partial grids, and keep barrier participation converged.

**Acceptance detail**
- [x] Every required pipeline creates and executes on supported hardware.
- [x] Awkward grids remain in bounds and match P3 tolerances.

**How to check**
```sh
Scripts/test.sh --filter QwenVisionRuntimeTests
```

**Evidence:** Historical frozen `frozen-qwenvisionruntimetests.log/.status` and `frozen-metal-validation-qwenvisionruntimetests.log/.status`, 11/11 PASS on the supported Apple GPU, plus `frozen-qwenmetalcontracttests.log/.status`, 8/8 PASS.

#### 16.7 Add preprocessing, pack, tower, position, and state tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenImagePreprocessorTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionWeightStoreTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionRuntimeTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalPositionTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` |
| Depends on | `16.1` |
| Parallel safe | `yes` |
| Deliverable | Tests cover geometry, corruption/binding, tiny tower math, multiple images, exact row counts, axis-asymmetric M-RoPE, decode delta, rollback, rebuild, missing pack, and video refusal. |

GPU suites must run on supported hardware; unexpected skips block the phase.

**Acceptance detail**
- [x] All five suites execute with nonzero counts.
- [x] Wrong grid, row count, companion, axis, lineage, or resource lifetime fails.

**How to check**
```sh
Scripts/test.sh --filter QwenVision
```

**Evidence:** Phase 16 root frozen receipts passed 6/6 preprocessing, 9/9 store, 11/11 runtime, 5/5 positions, and 7/7 original state tests. The exact final identity refreshed state proof passed 18/18 in `transaction-correction/`; the other suite receipts remain historical-by-preservation.

#### 16.8 Re-run existing Gemma multimodal regression suites

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPromptRendererTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPrefillInputTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Vision/VisionWeightStoreTests.swift` |
| Depends on | `16.2, 16.3, 16.4, 16.7` |
| Parallel safe | `no` |
| Deliverable | Run current Gemma prompt token, feature/span, and companion-binding tests after family dispatch. |

Do not alter Gemma IDs, geometry, companion v1, or expected feature rows.

**Acceptance detail**
- [x] All three existing suites execute with nonzero counts.
- [x] Gemma image behavior remains unchanged.

**How to check**
```sh
Scripts/test.sh --filter MultimodalPromptRendererTests
Scripts/test.sh --filter MultimodalPrefillInputTests
Scripts/test.sh --filter VisionWeightStoreTests
```

**Evidence:** Historical frozen shared/mixed-family receipts passed renderer 3/3 and prefill input 5/5; both suites include Qwen coverage alongside Gemma preservation assertions. Synthetic Gemma v1 store passed 3/3; the broad store selector's 12 tests include 9 Qwen tests and are not counted as 12 Gemma tests. Adjacent Gemma-only regression receipts also passed: `frozen-multimodalprefillregressiontests.log` 7/7 in 2 suites and `frozen-multimodalsuffixprefilltests.log` 9/9 in 2 suites.

### Phase 16 coverage plan

The table records the same 27-row set as the tracker: the exact accepted 25-path candidate plus two explicitly planned-but-unchanged regression rows. A selector in this table identifies the historical command that produced the named receipt; it does not claim a post-correction rerun unless the proof says exact-final-candidate.

| Code this phase changes | Test file that covers it | What the test or evidence proves | Command |
|---|---|---|---|
| `Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | Historical-by-preservation 8/8 proves exact-once Qwen vision registration and shader resolution. | `Scripts/test.sh --filter QwenMetalContractTests` |
| `Sources/TurboFieldfare/Kernels/Qwen/QwenFullAttention.swift` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift` | Historical-by-preservation 6/6 proves reference order, causal weights, partial RoPE, rejection, and awkward real-GPU preprocessing. | `Scripts/test.sh --filter QwenFullAttentionTests` |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_full_attention.metal` | `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenFullAttentionTests.swift`; `QwenMetalContractTests.swift` | Historical-by-preservation 6/6 plus 8/8 covers execution and ABI/registration. | `Scripts/test.sh --filter QwenFullAttentionTests` |
| `Sources/TurboFieldfare/Metal/Qwen/qwen_vision.metal` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionRuntimeTests.swift`; `QwenMetalContractTests.swift` | Historical-by-preservation 11/11 plus 8/8 covers pipeline execution, awkward bounds, ABI, and registration. | `Scripts/test.sh --filter QwenVisionRuntimeTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift`; `QwenConversationStateTests.swift` | Exact-final-candidate 18/18 vision-state and 12/12 scalar-state prove replay, lineage, quota, overflow, busy cancellation, rollback, and scalar preservation. | `Scripts/test.sh --filter QwenVisionConversationStateTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTextRunnerTests.swift` | Historical-by-preservation 16/16 covers scalar/hybrid runner behavior. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenImagePreprocessor.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenImagePreprocessorTests.swift` | Historical-by-preservation 6/6 covers official still duplication, grids, block order, quota, and invalid media. | `Scripts/test.sh --filter QwenImagePreprocessorTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenMultimodalPositions.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalPositionTests.swift` | Historical-by-preservation 5/5 covers three axes, asymmetric order, multiple images, malformed spans, and quota. | `Scripts/test.sh --filter QwenMultimodalPositionTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionConfig.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenImagePreprocessorTests.swift` | Historical-by-preservation 6/6 covers geometry and admission limits. | `Scripts/test.sh --filter QwenImagePreprocessorTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionRuntime.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionRuntimeTests.swift`; `QwenMultimodalDecoderOracleTests.swift` | Historical-by-preservation 11/11 and 4/4 cover the reduced 27-layer tower/merger and vector M-RoPE decoder oracle. | `Scripts/test.sh --filter QwenVisionRuntimeTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/Vision/QwenVisionWeightStore.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionWeightStoreTests.swift` | Historical-by-preservation 9/9 covers the exact 333-tensor, byte-bound synthetic companion contract. | `Scripts/test.sh --filter QwenVisionWeightStoreTests` |
| `Sources/TurboFieldfare/Runtime/Vision/MultimodalPrefillInput.swift` | `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPrefillInputTests.swift` | Historical-by-preservation 5/5 shared/mixed-family suite covers Gemma preservation, Qwen widths, quota, and span validation. | `Scripts/test.sh --filter MultimodalPrefillInputTests` |
| `Sources/TurboFieldfare/Runtime/Vision/MultimodalPromptRenderer.swift` | `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPromptRendererTests.swift` | Historical-by-preservation 3/3 shared/mixed-family suite covers Gemma preservation and Qwen wrapper/pad order. | `Scripts/test.sh --filter MultimodalPromptRendererTests` |
| `Sources/TurboFieldfare/Runtime/Vision/VisionWeightStore.swift` | `QwenVisionWeightStoreTests.swift`; `VisionWeightStoreTests.swift` | Historical-by-preservation 9/9 Qwen plus 3/3 synthetic v1 proves dispatch without changing v1 behavior. | `Scripts/test.sh --filter VisionWeightStoreTests` |
| `Tests/TurboFieldfare/Core/Kernels/Qwen/QwenMetalContractTests.swift` | self | Historical-by-preservation 8/8 executed. | `Scripts/test.sh --filter QwenMetalContractTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTestArchitecture.swift` | `MultimodalPromptRendererTests.swift`; `MultimodalPrefillInputTests.swift` | Historical-by-preservation 3/3 plus 5/5 executed callers use `QwenTestArchitecture.qwen36`. | `Scripts/test.sh --filter MultimodalPromptRendererTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenImagePreprocessorTests.swift` | self | Historical-by-preservation 6/6 executed. | `Scripts/test.sh --filter QwenImagePreprocessorTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalDecoderOracleTests.swift` | self | Historical-by-preservation 4/4, including a 4/4 Metal API + GPU validation receipt. | `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 Scripts/test.sh --filter QwenMultimodalDecoderOracleTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalDecoderReference.swift` | `QwenMultimodalDecoderOracleTests.swift` | Historical-by-preservation 4/4 executed oracle covers the independent reference fixture. | `Scripts/test.sh --filter QwenMultimodalDecoderOracleTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenMultimodalPositionTests.swift` | self | Historical-by-preservation 5/5 host-only execution; Metal validation is not claimed. | `Scripts/test.sh --filter QwenMultimodalPositionTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` | self | Exact-final-candidate 18/18, plus 18/18 with Metal API + GPU validation. | `Scripts/test.sh --filter QwenVisionConversationStateTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionRuntimeTests.swift` | self | Historical-by-preservation 11/11, plus 11/11 with Metal API + GPU validation. | `Scripts/test.sh --filter QwenVisionRuntimeTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionWeightStoreTests.swift` | self | Historical-by-preservation 9/9 executed. | `Scripts/test.sh --filter QwenVisionWeightStoreTests` |
| `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPrefillInputTests.swift` | self | Historical-by-preservation 5/5 shared/mixed-family suite executed. | `Scripts/test.sh --filter MultimodalPrefillInputTests` |
| `Tests/TurboFieldfare/Core/Runtime/Vision/MultimodalPromptRendererTests.swift` | self | Historical-by-preservation 3/3 shared/mixed-family suite executed. | `Scripts/test.sh --filter MultimodalPromptRendererTests` |
| `Sources/TurboFieldfare/Runtime/Vision/VisionRuntime.swift` (planned, byte-unchanged; regression-only) | `QwenVisionRuntimeTests` 11/11 separately; `frozen-multimodalprefillregressiontests.log` 7/7 in 2 suites; `frozen-multimodalsuffixprefilltests.log` 9/9 in 2 suites | `no-unit-test -` the path did not change; named adjacent Gemma regressions passed, never as direct changed-source execution. | `Scripts/test.sh --filter QwenVisionRuntimeTests` |
| `Tests/TurboFieldfare/Core/Runtime/Vision/VisionWeightStoreTests.swift` (planned, byte-unchanged; regression-only) | self | Historical-by-preservation 3/3 synthetic v1 executed; the broad selector's other 9 tests are Qwen tests. | `Scripts/test.sh --filter VisionWeightStoreTests` |

### Phase 16 evidence

<a id="phase-16-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-17 | Frozen P16 focused suites and Metal validation on the original held 25-path candidate | PASS at recorded nonzero counts: preprocessing 6/6, store 9/9, runtime 11/11, positions 5/5, state 7/7, decoder oracle 4/4, metal contract 8/8, full attention 6/6, text runner 16/16, shared/mixed-family renderer 3/3 and prefill 5/5, v1 store 3/3, adjacent Gemma prefill regression 7/7 in 2 suites, and adjacent Gemma suffix regression 9/9 in 2 suites; Debug/Release PASS | `scratch/qwen3.6-35b-a3b/evidence/phase-16/verification-manifest.txt`; `frozen-*.log/.status`, including `frozen-multimodalprefillregressiontests.log` and `frozen-multimodalsuffixprefilltests.log` |
| 2026-09-18 | Exact-final-candidate transaction correction gates | PASS: vision state 18/18, scalar state 12/12, Metal-validation vision state 18/18 with API + GPU banners, Debug and Release builds; HEAD `5770260510935cd32b82c4008ff06ae6458539ff`, aggregate `e0ab778b26b857e08d0b466fdd2086eb10ad999cbed6bde00b37771cd0d19a89` | `scratch/qwen3.6-35b-a3b/evidence/phase-16/transaction-correction/final-verification-packet.txt`; `verification-manifest-final.txt`; `acceptance-mapping.txt` |
| 2026-09-18 | Preservation and D4 identity | PASS: 23/23 unchanged P16 inputs; full 107-path four-stream union; 25 candidate paths present | `transaction-correction/preservation-comparison-final.txt`; `transaction-correction/d4-four-stream-final.log`; `d4-p15-to-p16-comparison.txt` |
| 2026-09-18 | Canonical documentation closeout checks | Pending Main editorial acceptance; docs-only checks do not rerun engineering | `scratch/qwen3.6-35b-a3b/evidence/phase-16/docs-closeout/` |

Main accepted the P16 engineering candidate in correction-team event `#216`; independent approvals are correction-team events `#203` and `#211` on submission `#201`. This documentation task records that acceptance and does not grant product acceptance.

The final exact-candidate receipts are only the transaction-correction vision-state 18/18, scalar-state 12/12, Metal-validation vision-state 18/18, Debug build, and Release build. All other passing counts above are historical Phase 16 root receipts carried forward solely because the 23 corresponding inputs are byte-identical. The unchanged `VisionRuntime.swift` proof keeps Qwen runtime 11/11 separate and uses `frozen-multimodalprefillregressiontests.log` (7/7 in 2 suites) plus `frozen-multimodalsuffixprefilltests.log` (9/9 in 2 suites) only as named adjacent Gemma regressions, never direct changed-source execution. The original held aggregate `7c333170af9ab81d0c81c14c6b7b6744f816c4877bd9c76196cde3048853a6b7`, baseline/corrective failures, later failed tranches, and invalid concurrent-edit scalar build remain preserved rather than relabelled.

The unauthorized unfiltered full-package run remains **FAIL**: 1,767 tests in 248 suites, 3 issues, and 2 skips. One P16 `qwen_vision` module expectation was narrowly fixed and its focused selector passed 8/8. Two unresolved failures are attributed by source/artifact inspection to existing settings behavior and installed-pack path binding; pre-P16 execution is not established. Whole-run authentic-model access remains unverified. The two unauthorized Sol command sets and the prior Pro self-reported prohibited read-only `cat` through bash (no exact command/session, no claimed writes/builds/tests, and not a full audit) remain disclosed.

The numerical proof is limited to the frozen P3 reduced fixture, reduced ordered 27-layer proof, and official-layout synthetic lifecycle. Image state proofs are actor-seam checks, not end-to-end product inference. Authentic companion conversion and inference remain Phase 21 work; product routing remains Phase 18/20 work. Gemma defaults, `.gturbo` v1, and host authority are unchanged.

---

## Phase 17 - Decode service binds responses to loaded identity

**When this is done:** decode service binds responses to loaded identity

Needs: `12, 13, 15`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff` — recorded at Codex takeover; the original phase-start baseline was not recorded in this document. D4 uses explicit candidate accounting and preservation manifests.
Disk baseline: takeover preflight recorded macOS 26.6.2, Swift 6.3.2, M2 Pro with 32 GiB RAM, about 80 GiB free disk, and 55% free memory. Fresh process and resource receipts accompany each execution; no new worktree or simulator was created.
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-17/`

### Current state

Main accepted Phase 17 on 2026-09-22 after nine target builds, the actual DecodeService product link, two independent production-source reviews, and all thirteen focused selectors passed. The matrix contains 165 tests across 14 suites, with zero failures and no skips. The ready-write lifecycle suite passed three consecutive runs after its byte assertion was corrected to capture the actual transmitted JSON frame.

The final sixteen-input candidate manifest SHA256 is `fdf8725f633834c3be1822a384d52b801314bab92261943864bb3d99768b1d01`. The accepted snapshot, complete test footers, source preservation counts and Main acceptance receipt are in `scratch/qwen3.6-35b-a3b/evidence/phase-17/execution/codex-accepted-20260922T120651Z/`. At acceptance, all 301 production inputs matched the successful builds, 623 inputs outside P17 were preserved, and all 25 accepted P16 inputs were unchanged.

Takeover corrections cover the compiler ownership annotation, immutable queued admission, startup handshake proof, libbsm linking, and the real ready-write publication/cleanup boundary. Execution also corrected test ownership of transferred handles, missing temporary model directories, one inconsistent Gemma ready response, and a JSON-order-dependent byte comparison. Original failed and interrupted runs remain preserved. The passing evidence is tied to each final unchanged source/test file, with each test-only correction rerun in its affected suite.

This acceptance proves the identity and lifecycle component boundary. It does not qualify authentic Qwen weights, ordinary app Qwen generation, performance or the working application. Those remain in the approved later phases.

### Target state

The service returns the descriptor produced by validated load, admits one loaded family, binds conversation epoch plus descriptor identity to generation/checkpoint traffic, and rejects stale responses or mismatched requests across unload/load. Cancellation preserves P13 semantics.

### Tasks

#### 17.1 Add descriptor and model identity to decode protocol

| | |
|---|---|
| Duty | `service-protocol` |
| Touches | `Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift` |
| Depends on | `12.6, 13.6, 15.5` |
| Parallel safe | `yes` |
| Deliverable | Extend load-ready, generation, checkpoint, diagnostics, and unload events with backward-decodable optional descriptor/model identity fields where compatibility requires. |

The service, not the caller, supplies verified identity after load.

**Acceptance detail**
- [x] Existing Gemma messages decode with their current defaults.
- [x] Qwen descriptor round trips without losing revision, family, format, or vision state.

**How to check**
```sh
Scripts/test.sh --filter DecodeProtocolQwenIdentityTests
```

**Evidence:** Protocol identity: 7/7 passed. `scratch/qwen3.6-35b-a3b/evidence/phase-17/execution/codex-accepted-20260922T120651Z/focused-test-results.json`.

#### 17.2 Load one verified family in the decode service

| | |
|---|---|
| Duty | `service-code` |
| Touches | `Sources/TurboFieldfareDecodeService/Entry.swift` |
| Depends on | `17.1` |
| Parallel safe | `no` |
| Deliverable | Create the family runtime/codec from validated manifest, return its descriptor, and keep one loaded model directory at a time. |

Failed load leaves a clear unloaded state. Unload releases family state before another load is admitted.

**Acceptance detail**
- [x] The ready event contains the loader-produced descriptor.
- [x] A second load, wrong descriptor assertion, or failed load cannot retain mixed state.

**How to check**
```sh
Scripts/test.sh --filter DecodeServiceQwenLifecycleTests
```

**Evidence:** Service lifecycle: 15/15 passed, including three consecutive final runs. `scratch/qwen3.6-35b-a3b/evidence/phase-17/execution/codex-accepted-20260922T120651Z/focused-test-results.json`.

#### 17.3 Bind app inference requests to identity plus epoch

| | |
|---|---|
| Duty | `service-client` |
| Touches | `Sources/TurboFieldfareApp/Core/Inference/AppInferenceClient.swift`; `Sources/TurboFieldfareApp/Core/Inference/DecodeServiceInferenceClient.swift` |
| Depends on | `17.1, 17.2` |
| Parallel safe | `no` |
| Deliverable | Carry loaded identity with each conversation epoch and reject responses from a prior model/epoch/generation. |

Preserve current scoped cancellation transport and reset behavior.

**Acceptance detail**
- [x] A stale Qwen or Gemma response never reaches the active transcript.
- [x] Reset, cancel, unload, and reconnect clear the bound identity consistently.

**How to check**
```sh
Scripts/test.sh --filter DecodeServiceModelIdentityTests
```

**Evidence:** Client identity: 26/26 passed; connection invalidation: 12/12 passed, including real child exit/cancellation. `scratch/qwen3.6-35b-a3b/evidence/phase-17/execution/codex-accepted-20260922T120651Z/focused-test-results.json`.

#### 17.4 Add protocol round-trip and compatibility tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` (**PROPOSED**) |
| Depends on | `17.1` |
| Parallel safe | `yes` |
| Deliverable | Tests decode old payloads, round-trip Qwen descriptors, reject asserted identity mismatches, and preserve request IDs/epochs. |

Test only protocol values; service process behavior belongs to task 17.5.

**Acceptance detail**
- [x] The suite executes with a nonzero count.
- [x] Old Gemma payloads remain decodable and new identity cannot be forged by omission.

**How to check**
```sh
Scripts/test.sh --filter DecodeProtocolQwenIdentityTests
```

**Evidence:** Protocol identity: 7/7 passed with legacy payload and tiny runtime readiness coverage. `scratch/qwen3.6-35b-a3b/evidence/phase-17/execution/codex-accepted-20260922T120651Z/focused-test-results.json`.

#### 17.5 Add service load, cancel, reset, and stale-response tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` (**PROPOSED**); `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` (**PROPOSED**) |
| Depends on | `17.1` |
| Parallel safe | `yes` |
| Deliverable | Tests use tiny fake runtimes to cover one-load admission, failed load, unload/reload, cancellation, reset, service invalidation, stale epoch, and stale model identity. |

No real model process is needed for these behavioral tests.

**Acceptance detail**
- [x] Both suites execute with nonzero counts.
- [x] No response or state crosses model identity or epoch changes.

**How to check**
```sh
Scripts/test.sh --filter DecodeServiceQwen
```

**Evidence:** Lifecycle: 15/15; client identity: 26/26; connection invalidation: 12/12 passed. `scratch/qwen3.6-35b-a3b/evidence/phase-17/execution/codex-accepted-20260922T120651Z/focused-test-results.json`.

#### 17.6 Re-run existing service gate and response tests

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfareDecodeService/DecodeConversationGateTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceResponseMatchingTests.swift` |
| Depends on | `17.2, 17.3, 17.4, 17.5` |
| Parallel safe | `no` |
| Deliverable | Run current one-process conversation gate and app response matching after identity binding. |

Do not weaken existing generation ID, turn order, or socket lifecycle assertions.

**Acceptance detail**
- [x] Both existing suites execute with nonzero counts.
- [x] Gemma lifecycle and stale-response handling remain unchanged.

**How to check**
```sh
Scripts/test.sh --filter DecodeConversationGateTests
```

**Evidence:** Conversation gate: 10/10; response matching: 5/5 passed. `scratch/qwen3.6-35b-a3b/evidence/phase-17/execution/codex-accepted-20260922T120651Z/focused-test-results.json`.

### Phase 17 coverage plan

The candidate includes ten production source files, five test files, and Package.swift. Two unchanged regression suites remain explicit coverage rows. Tests use tiny fixtures and controlled non-model children. Live model/service inference is outside this phase's evidence.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Package.swift` | none; build configuration | Actual DecodeService product linking resolves both audit-token symbols from libbsm. | `swift build --configuration debug --product TurboFieldfareDecodeService` |
| `Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | Legacy messages and exact verified identity round trip. | `Scripts/test.sh --filter DecodeProtocolQwenIdentityTests` |
| `Sources/TurboFieldfareDecodeService/Entry.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | Production lifecycle admission, ready-write handoff and ambiguous-write cleanup. | `Scripts/test.sh --filter DecodeServiceQwenLifecycleTests` |
| `Sources/TurboFieldfareDecodeService/DecodeCommandQueue.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | Refused queued requests cannot acquire a successor lease after identifier reuse. | `Scripts/test.sh --filter DecodeServiceQwenLifecycleTests` |
| `Sources/TurboFieldfareDecodeService/DecodeServiceSession.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | One atomic lifecycle transition, exact binding and cleanup ownership. | `Scripts/test.sh --filter DecodeServiceQwenLifecycleTests` |
| `Sources/TurboFieldfareDecodeService/DecodeServiceOutbox.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | Terminal, stamp and measurement ordering. | `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Sources/TurboFieldfareApp/Core/Inference/AppInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | Readiness carries the verified runtime identity. | `Scripts/test.sh --filter DecodeProtocolQwenIdentityTests` |
| `Sources/TurboFieldfareApp/Core/Inference/DecodeServiceInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | Startup rejection, bound lifecycle joining and stale response isolation. | `Scripts/test.sh --filter DecodeServiceModelIdentityTests` |
| `Sources/TurboFieldfareApp/Core/Inference/DecodeServiceResponseRouter.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceResponseMatchingTests.swift` | Generation matching and buffered response behavior. | `Scripts/test.sh --filter DecodeServiceResponseMatchingTests` |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | Tiny bundle installation, replacement refusal and unload joins. | `Scripts/test.sh --filter DecodeProtocolQwenIdentityTests` |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyRuntime.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyRuntimeTests.swift` | Validated family selection preserves Gemma and rejects unsupported families. | `Scripts/test.sh --filter ModelFamilyRuntimeTests` |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeProtocolQwenIdentityTests.swift` | Named suite executes with nonzero tests, zero failures and no unexpected skips. | `Scripts/test.sh --filter DecodeProtocolQwenIdentityTests` |
| `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceQwenLifecycleTests.swift` | Named suite executes with nonzero tests, zero failures and no unexpected skips. | `Scripts/test.sh --filter DecodeServiceQwenLifecycleTests` |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | Named suite executes with nonzero tests, zero failures and no unexpected skips. | `Scripts/test.sh --filter DecodeServiceModelIdentityTests` |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceConnectionInvalidationTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceConnectionInvalidationTests.swift` | Named suite executes with nonzero tests, zero failures and no unexpected skips. | `Scripts/test.sh --filter DecodeServiceConnectionInvalidationTests` |
| `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | Named suite executes with nonzero tests, zero failures and no unexpected skips. | `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Tests/TurboFieldfareDecodeService/DecodeConversationGateTests.swift` | `Tests/TurboFieldfareDecodeService/DecodeConversationGateTests.swift` | Named suite executes with nonzero tests, zero failures and no unexpected skips. | `Scripts/test.sh --filter DecodeConversationGateTests` |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceResponseMatchingTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceResponseMatchingTests.swift` | Named suite executes with nonzero tests, zero failures and no unexpected skips. | `Scripts/test.sh --filter DecodeServiceResponseMatchingTests` |

### Phase 17 evidence

<a id="phase-17-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-22 | Takeover candidate preflight and target builds | Original compiler error preserved; corrected source then passed all nine target builds | `execution/codex-stage1-20260922T103126Z/`; `execution/codex-correction-stage1-20260922T105308Z/` |
| 2026-09-22 | First focused test invocation | Link failure before tests; remaining twelve selectors were not run | `execution/codex-tests-20260922T120310/` |
| 2026-09-22 | Required system library link, actual DecodeService product link, nine target builds | Product link and all nine target builds passed; source and package hashes preserved | `execution/codex-package-link-20260922T110457Z/` |
| 2026-09-22 | Sixteen-input candidate capture and preservation | All prior fifteen inputs unchanged; 623 other takeover inputs and 25 Phase 16 inputs preserved | `execution/codex-final16-candidate-20260922T110743Z/` |
| 2026-09-22 | Initial independent source review | No demonstrated source defect; review identified the ready-write test gap subsequently closed below | `execution/codex-ready-final-candidate-20260922T112020Z/independent-review-1.txt`; `independent-review-2.txt` |

| 2026-09-22 | Production ready-write boundary and nine target builds | All nine builds passed; actual writer path tests cover ready visibility, pending teardown, partial/full write failure and cleanup ordering | `execution/codex-ready-boundary-20260922T111250Z/` |
| 2026-09-22 | Final focused matrix and native process observer checks | 165 tests, 14 suites, 13 selectors passed; lifecycle suite passed three final runs; no failures/skips | `execution/codex-ready-tests-fixed-20260922T112919Z/correction-20260922T113646Z/final-selector-matrix.csv`; `execution/codex-accepted-20260922T120651Z/focused-test-results.json` |
| 2026-09-22 | Main final candidate review, preservation and acceptance | Sixteen inputs captured; 301 production, 623 outside-P17 and 25 P16 identities verified; process gate passed | `execution/codex-accepted-20260922T120651Z/` |

All paths in this table are relative to `scratch/qwen3.6-35b-a3b/evidence/phase-17/`. Earlier candidates are historical. The accepted snapshot and per-selector evidence above bind the final P17 source and test paths. Test-only fixture corrections did not change the production build inputs. No authentic Qwen inference or performance claim follows from these component tests.

---

## Phase 18 - CLI runs verified Qwen requests

**When this is done:** CLI runs verified Qwen requests

Needs: `12, 15, 16`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff` — recorded before working-source changes; baseline snapshots preserve the dirty inputs.
Disk baseline: 78 GiB available on 2026-09-22. Existing devices, worktrees, processes and source snapshots are recorded in `scratch/qwen3.6-35b-a3b/evidence/phase-18/baseline/`. No new worktree or simulator was created.
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-18/`

### Accepted implementation evidence

Main accepted P18 on 2026-09-22. The final CLI product build passed. Six focused selectors contain 49 passes and one existing Apple7-only hardware skip on this M2+ machine. All 645 compiled inputs matched the corrected final test manifest. Tiny runtime fixtures prove actual Qwen generation and codec/renderer composition; authentic-weight inference, performance and complete companion qualification remain later work.

Independent Luna review found that the test timeout could wait forever on an unstructured generation task. Luna repaired owned cancellation and gate cleanup using a one-shot race and strengthened cancellation-at-EOS to assert zero published events. Main reviewed the fix. Its ten-test selector then passed with exit 0 in 27.013 seconds and all compiled inputs unchanged. The earlier 50-pass aggregate was a counting error, corrected to 49 passes plus one skip.

Final snapshot and acceptance receipt: `execution/codex-accepted-20260922T1329/`. Final compiled-input manifest SHA256: `fae673eb93a27ccd91209269cf71d8c2bee9dcbfce6b7eaae8f5bf8561ab5dc7`.

### Target state

CLI argument validation stays model-independent, then verified load selects the family runtime and codec. Text, messages, thinking, tools, still images, streaming, identity output, help, and errors reflect the loaded descriptor. CLI does not wait for decode-service integration.

### CLI contract recorded on 2026-09-22

`--thinking auto|on|off` defaults to `auto`, preserving each family's existing thinking behavior. `--show-model-identity` prints loaded identity on stderr and distinguishes verified Qwen metadata from legacy Gemma v1 metadata. `--tools-file` accepts an OpenAI-shaped array of function definitions. The CLI returns validated tool calls but never executes them. `--video` is recognized and rejected before any model or tokenizer load.

`--messages-file` retains existing text and image content and adds the existing family-neutral message fields for reasoning, tool calls and tool results. Unsupported video/audio parts and invalid option combinations fail before load. Sampling defaults remain unchanged.

`Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` supplies a narrow public session interface over the existing verified bundle, family codec and generation components. Low-level Qwen internals stay private. The CLI has an injectable session boundary for routing tests, and tiny runtime fixtures exercise the production session itself. This interface is necessary for task 18.2 because the current Qwen runner and vision internals cannot be called from the CLI target.

### Review decisions recorded on 2026-09-22

Verified metadata admission selects the runtime family. Gemma continues through the original CLI parser and generation implementation so its defaults and behavior remain intact. The new session handles Qwen. Both input parsing and the shared image hardware gate precede metadata inspection, preserving early-error behavior.

Main review required full loaded-identity equality before reread image metadata can be used, model-directory Gemma tokenizer loading in the reusable facade, explicit image residency, remaining-context response clamping, and a busy guard across asynchronous generation. Complete tool calls are published only after actual model EOS, parser completion, and a final cancellation check. Tiny-fixture hooks exercise the production guard and terminal publication path.

The pinned Qwen codec already emits image start/end markers. The existing P16 renderer expands a bare image placeholder by adding those markers. The product adapter must remove the codec framing before calling the renderer, leaving exactly one pair per image; sizing must equal base codec tokens plus the sum of image rows minus one. The new runtime suite covers this actual composition without changing the accepted P16 renderer contract. The accepted focused runtime suite verifies these corrections.

### Tasks

#### 18.1 Add explicit Qwen-capable CLI options and help

| | |
|---|---|
| Duty | `cli-code` |
| Touches | `Sources/TurboFieldfareCLI/Args.swift` |
| Depends on | `12.6, 15.5, 16.8` |
| Parallel safe | `yes` |
| Deliverable | Expose thinking mode and verified identity reporting while preserving existing prompt/messages/image exclusivity and generation defaults. |

Video remains an explicit unsupported input. Do not silently apply publisher sampling defaults.

**Acceptance detail**
- [x] Help names actual supported text, thinking, tool, and still-image behavior.
- [x] Invalid option combinations fail before tokenizer/model loading.

**How to check**
```sh
Scripts/test.sh --filter CLIQwenArgumentsTests
```

**Evidence:** Final CLI build exit 0; six selectors 49 passes plus one expected hardware skip. Corrected runtime suite 10/10 pass, exit 0. Exact source/test identities and logs are indexed in `execution/codex-accepted-20260922T1329/acceptance.json`.

#### 18.2 Select family runtime and codec in CLI Run

| | |
|---|---|
| Duty | `cli-code` |
| Touches | `Sources/TurboFieldfareCLI/Run.swift`; `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` |
| Depends on | `18.1` |
| Parallel safe | `no` |
| Deliverable | Route through verified family admission, retain the existing Gemma generation path, and use the family session for Qwen while preserving streaming/result formatting. |

Use Qwen image preparation only for a verified Qwen companion and print the loaded model ID/revision/format in diagnostics.

**Acceptance detail**
- [x] Tiny Qwen text, tool, thinking, and image requests reach the right family components.
- [x] Missing/invalid image support fails rather than ignoring accepted images.

**How to check**
```sh
Scripts/test.sh --filter CLIQwenRunTests
```

**Evidence:** Final CLI build exit 0; six selectors 49 passes plus one expected hardware skip. Corrected runtime suite 10/10 pass, exit 0. Exact source/test identities and logs are indexed in `execution/codex-accepted-20260922T1329/acceptance.json`.

#### 18.3 Add CLI argument and help tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/CLI/CLIQwenArgumentsTests.swift` |
| Depends on | `18.1` |
| Parallel safe | `yes` |
| Deliverable | Tests cover help text, thinking options, identity output, unsupported video, sampling preservation, and validation before load. |

Use `TurboFieldfareCLICore`, which is already imported by `TurboFieldfareTestsCore`; no nonexistent CLI test target is introduced.

**Acceptance detail**
- [x] The suite executes with a nonzero count.
- [x] Help and errors describe only implemented options.

**How to check**
```sh
Scripts/test.sh --filter CLIQwenArgumentsTests
```

**Evidence:** Final CLI build exit 0; six selectors 49 passes plus one expected hardware skip. Corrected runtime suite 10/10 pass, exit 0. Exact source/test identities and logs are indexed in `execution/codex-accepted-20260922T1329/acceptance.json`.

#### 18.4 Add tiny CLI run routing tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfare/Core/CLI/CLIQwenRunTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` |
| Depends on | `18.2` |
| Parallel safe | `yes` |
| Deliverable | Inject tiny family runtimes/codecs to verify text, message, tool, thinking, image order, streaming, descriptor reporting, and fail-closed image behavior. |

Do not load the official model in unit tests.

**Acceptance detail**
- [x] The suite executes with a nonzero count.
- [x] Every request uses the verified family codec and preserves output ordering.

**How to check**
```sh
Scripts/test.sh --filter CLIQwenRunTests
```

**Evidence:** Final CLI build exit 0; six selectors 49 passes plus one expected hardware skip. Corrected runtime suite 10/10 pass, exit 0. Exact source/test identities and logs are indexed in `execution/codex-accepted-20260922T1329/acceptance.json`.

#### 18.5 Re-run existing CLI argument and image-order suites

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfare/Core/CLI/CLIArgumentsTests.swift`; `Tests/TurboFieldfare/Core/CLI/CLIImageOrderTests.swift`; `Tests/TurboFieldfare/Core/CLI/CLIPromptInputTests.swift` |
| Depends on | `18.1, 18.2, 18.3, 18.4` |
| Parallel safe | `no` |
| Deliverable | Run all current CLICore tests after family routing. |

Do not change Gemma defaults, image ordering, prompt size refusal, or early hardware/error behavior.

**Acceptance detail**
- [x] All three existing suites execute with nonzero counts.
- [x] Current Gemma CLI behavior remains unchanged.

**How to check**
```sh
Scripts/test.sh --filter CLIArgumentsTests
```

**Evidence:** Final CLI build exit 0; six selectors 49 passes plus one expected hardware skip. Corrected runtime suite 10/10 pass, exit 0. Exact source/test identities and logs are indexed in `execution/codex-accepted-20260922T1329/acceptance.json`.

### Phase 18 coverage plan

Each changed path and required regression suite is linked to an executed test; hardware skip applicability is explicit.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareCLI/Args.swift` | `Tests/TurboFieldfare/Core/CLI/CLIQwenArgumentsTests.swift` | 5/5 pass; final receipt indexes the exact executed log. | `Scripts/test.sh --filter CLIQwenArgumentsTests` |
| `Sources/TurboFieldfareCLI/Run.swift` | `Tests/TurboFieldfare/Core/CLI/CLIQwenRunTests.swift` | 9/9 pass; final receipt indexes the exact executed log. | `Scripts/test.sh --filter CLIQwenRunTests` |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | 10/10 pass; final receipt indexes the exact executed log. | `Scripts/test.sh --filter ModelFamilyGenerationTests` |
| `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | 10/10 pass; final receipt indexes the exact executed log. | `Scripts/test.sh --filter ModelFamilyGenerationTests` |
| `Tests/TurboFieldfare/Core/CLI/CLIQwenArgumentsTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIQwenArgumentsTests.swift` | 5/5 pass; final receipt indexes the exact executed log. | `Scripts/test.sh --filter CLIQwenArgumentsTests` |
| `Tests/TurboFieldfare/Core/CLI/CLIQwenRunTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIQwenRunTests.swift` | 9/9 pass; final receipt indexes the exact executed log. | `Scripts/test.sh --filter CLIQwenRunTests` |
| `Tests/TurboFieldfare/Core/CLI/CLIArgumentsTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIArgumentsTests.swift` | 17/17 pass; final receipt indexes the exact executed log. | `Scripts/test.sh --filter CLIArgumentsTests` |
| `Tests/TurboFieldfare/Core/CLI/CLIImageOrderTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIImageOrderTests.swift` | 1/1 pass; final receipt indexes the exact executed log. | `Scripts/test.sh --filter CLIImageOrderTests` |
| `Tests/TurboFieldfare/Core/CLI/CLIPromptInputTests.swift` | `Tests/TurboFieldfare/Core/CLI/CLIPromptInputTests.swift` | 7/7 pass; 1 expected Apple7-only hardware skip; final receipt indexes the exact executed log. | `Scripts/test.sh --filter CLIPromptInputTests` |

### Phase 18 evidence

<a id="phase-18-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-22 | CLI product build with identity revalidation | exit 0; Sources and Package hashes preserved; later terminal/image corrections still need a fresh build and tests | `execution/sol-cli-build-20260922T122733Z-final/` |
| 2026-09-22 | Main source/test review | Found and requested the image marker correction, deterministic overlap proof, and terminal cancellation checks; no phase acceptance claimed | `coordination/`; production and test candidate files |
| 2026-09-22 | Final production build | exit 0; 11.14 seconds; Sources and Package unchanged | `execution/sol-cli-build-20260922T123449Z-seams/` |
| 2026-09-22 | Final focused selectors and timeout repair rerun | 49 passes, one existing hardware skip; corrected runtime 10/10 pass, exit 0; all 645 inputs unchanged | `execution/luna-unit-tests-20260922T124759Z/`; `execution/luna-timeout-correction-20260922T132152Z/` |
| 2026-09-22 | Main acceptance and snapshot | Nine coverage rows satisfied, seven changed source/test files accounted; source review and independent test finding resolved | `execution/codex-accepted-20260922T1329/` |

---

## Phase 19 - Loopback server serves verified Qwen chat

**When this is done:** loopback server serves verified Qwen chat

Needs: `12, 15, 16`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff` — recorded before P19 edits, with all 28 existing server source/test files snapshotted.
Disk baseline: 77 GiB available on 2026-09-22; existing devices, worktrees and processes are recorded under `scratch/qwen3.6-35b-a3b/evidence/phase-19/baseline/`.
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-19/`.

### Phase-start state

`ServerInference.swift:608-687` directly loads `GFTokenizer`, `Model`, and `RealForwardRunner`; `ServerArguments.swift:27,79` defaults `--model-id` to Gemma. `HTTPServer.swift:86` already binds `127.0.0.1`, which must remain fixed.

### Target state

The loopback server derives or checks API model identity from the verified descriptor and selects family runtime/codec. Chat, streaming, tools, thinking, still images, prompt cache identity, errors, and unsupported video behave consistently without weakening loopback binding.

### Accepted implementation on 2026-09-22

Sol prepared six production files under the repository-owned P19 staging directory while Luna verified P18. After P18 acceptance, the exact source candidate and tests were promoted and verified. The sixth production path is `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift`: its narrow fully validated companion check reports ready only after matching the actual loaded text identity and validating the exact companion binding. A path merely existing cannot advertise image support. Its pre-change snapshot is in `baseline/companion-helper-addendum.json`. The exact production scope adds `Command/main.swift` and `Core/HTTPServer.swift` to the original three server paths because those files pass the command model ID into the listening server and API responses. This is necessary integration within the approved identity behavior.

An omitted model assertion preserves Gemma's existing API ID and uses the verified descriptor ID for Qwen. Legacy Gemma retains explicit aliases; a Qwen assertion must equal its verified descriptor ID. A resolved identity object carries family, revision and verified identity into model listings and completion frames.

Qwen initially uses the stateless P18 session and reports zero cached tokens. No Qwen prompt state is reused, so family, revision, template, thinking and image lineage cannot cross requests. Gemma's existing cache stays intact. Diagnostics describe the effective Qwen cache behavior.

Qwen message, reasoning, tool-schema and historical-argument member order comes from the sanitized request bytes. Staged images retain their existing attachment lease and ordered message IDs. The thinking field accepts `auto`, `on` or `off`, defaulting to `auto`. Common scalar, tool, body-size, depth, attachment, queue and loopback constraints remain in force. Video/audio fail explicitly.


The initial companion guard assumed the text-only admitted identity already carried verified vision. The correction validates the actual companion and exact text binding independently, then compares every declared vision field when one is supplied. Sparse 333-tensor tests prove valid, missing, corrupt, stale and text-identity mismatch behavior. The existing tiny text fixture rejects image overrides earlier, so its single error expectation was corrected without changing production or other runtime tests.

The corrected server build passed in 10.50 seconds. Identity 11, inference 7, HTTP 19, prompt-cache 16, OpenAI-related 44 and streaming-body 6 tests passed with all 646 execution inputs unchanged. The final shared-runtime rerun passed 10/10 in 26.850 seconds. This 113-test total does not count failed attempts or repeats. Authentic model qualification remains later work.

### Tasks

#### 19.1 Derive server model identity from the descriptor

| | |
|---|---|
| Duty | `server-code` |
| Touches | `Sources/TurboFieldfareServer/Core/ServerArguments.swift`; `Sources/TurboFieldfareServer/Command/main.swift` |
| Depends on | `12.6, 15.5, 16.8` |
| Parallel safe | `yes` |
| Deliverable | Remove the unconditional Gemma API-ID default. Treat an optional `--model-id` as an assertion/alias governed by explicit compatibility rules after verified load. |

Keep required model path and existing runtime option validation. Bind remains `127.0.0.1`.

**Acceptance detail**
- [x] No Qwen load can report the Gemma default ID.
- [x] An asserted incompatible ID fails before serving requests.

**How to check**
```sh
Scripts/test.sh --filter ServerQwenIdentityTests
```

**Evidence:** `execution/codex-accepted-20260922T1528/acceptance.json`, the server logs in `execution/20260922T1415Z/`, and the focused corrected runtime log in `execution/runtime-fixture-addendum-20260922T1520Z/`.

#### 19.2 Select family runtime and codec in server inference

| | |
|---|---|
| Duty | `server-code` |
| Touches | `Sources/TurboFieldfareServer/Core/ServerInference.swift`; `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` |
| Depends on | `19.1` |
| Parallel safe | `no` |
| Deliverable | Replace direct Gemma construction with verified family runtime/codec and bind prompt caches to descriptor plus codec/template identity. |

Route text, thinking, tools, still images, and streaming through family components; reject missing vision and video explicitly.

**Acceptance detail**
- [x] Tiny Qwen requests emit correctly framed streaming/non-streaming responses.
- [x] Cache entries cannot cross family, revision, template, thinking, or image lineage.

**How to check**
```sh
Scripts/test.sh --filter ServerQwenInferenceTests
```

**Evidence:** `execution/codex-accepted-20260922T1528/acceptance.json`, the server logs in `execution/20260922T1415Z/`, and the focused corrected runtime log in `execution/runtime-fixture-addendum-20260922T1520Z/`.

#### 19.3 Expose verified model data in API responses

| | |
|---|---|
| Duty | `server-code` |
| Touches | `Sources/TurboFieldfareServer/Core/OpenAIModels.swift`; `Sources/TurboFieldfareServer/Core/HTTPServer.swift` |
| Depends on | `19.2` |
| Parallel safe | `no` |
| Deliverable | Populate model-list and response metadata from the verified descriptor rather than a command-line string. |

Do not expose local paths or weaken existing response validation.

**Acceptance detail**
- [x] Model list and completion responses agree on ID/family/revision policy.
- [x] A stale asserted ID cannot leak into a response.

**How to check**
```sh
Scripts/test.sh --filter ServerQwenIdentityTests
```

**Evidence:** `execution/codex-accepted-20260922T1528/acceptance.json`, the server logs in `execution/20260922T1415Z/`, and the focused corrected runtime log in `execution/runtime-fixture-addendum-20260922T1520Z/`.

#### 19.4 Add server identity and inference tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift`; `Tests/TurboFieldfareServer/ServerQwenInferenceTests.swift` |
| Depends on | `19.1` |
| Parallel safe | `yes` |
| Deliverable | Tests cover identity derivation/assertion, text/tools/thinking/images, streaming chunks, prompt-cache separation, missing vision, unsupported video, and bind host. |

Use NIOEmbedded/tiny runtimes. Assert `127.0.0.1` remains the only host.

**Acceptance detail**
- [x] Both suites execute with nonzero counts.
- [x] No model identity or cached state crosses family boundaries.

**How to check**
```sh
Scripts/test.sh --filter ServerQwen
```

**Evidence:** `execution/codex-accepted-20260922T1528/acceptance.json`, the server logs in `execution/20260922T1415Z/`, and the focused corrected runtime log in `execution/runtime-fixture-addendum-20260922T1520Z/`.

#### 19.5 Re-run HTTP, prompt-cache, and validation suites

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfareServer/HTTPServerTests.swift`; `Tests/TurboFieldfareServer/ServerPromptCacheTests.swift`; `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` |
| Depends on | `19.1, 19.2, 19.3, 19.4` |
| Parallel safe | `no` |
| Deliverable | Run current loopback, cache, request-validation, streaming-stop, Gemma tool, and server-argument behavior after family integration. |

Do not broaden host binding, request sizes, attachment roots, or tool acceptance.

**Acceptance detail**
- [x] All three existing suites execute with nonzero counts.
- [x] Existing Gemma and loopback security behavior remains unchanged.

**How to check**
```sh
Scripts/test.sh --filter OpenAIValidationTests
```

**Evidence:** `execution/codex-accepted-20260922T1528/acceptance.json`, the server logs in `execution/20260922T1415Z/`, and the focused corrected runtime log in `execution/runtime-fixture-addendum-20260922T1520Z/`.

### Phase 19 coverage plan

Every changed source and test has a coverage row. The final candidate passed 113 focused tests; historical failed attempts remain in the evidence.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareServer/Core/ServerArguments.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift; Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | 11/11 identity + 44/44 validation-related pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter OpenAIValidationTests` |
| `Sources/TurboFieldfareServer/Core/ServerInference.swift` | `Tests/TurboFieldfareServer/ServerQwenInferenceTests.swift` | 7/7 inference pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter ServerQwenInferenceTests` |
| `Sources/TurboFieldfare/Runtime/Inference/ModelFamilyGeneration.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift; Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | 11/11 identity + 10/10 runtime pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter ServerQwenIdentityTests` |
| `Sources/TurboFieldfareServer/Core/OpenAIModels.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | 11/11 identity pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter ServerQwenIdentityTests` |
| `Sources/TurboFieldfareServer/Command/main.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | server build + 11/11/11 identity pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter ServerQwenIdentityTests` |
| `Sources/TurboFieldfareServer/Core/HTTPServer.swift` | `Tests/TurboFieldfareServer/HTTPServerTests.swift; Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | 19/19 HTTP + 11/11/11 identity pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter HTTPServerTests` |
| `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | `Tests/TurboFieldfareServer/ServerQwenIdentityTests.swift` | 11/11 pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter ServerQwenIdentityTests` |
| `Tests/TurboFieldfareServer/ServerQwenInferenceTests.swift` | `Tests/TurboFieldfareServer/ServerQwenInferenceTests.swift` | 7/7 pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter ServerQwenInferenceTests` |
| `Tests/TurboFieldfareServer/HTTPServerTests.swift` | `Tests/TurboFieldfareServer/HTTPServerTests.swift` | 19/19 pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter HTTPServerTests` |
| `Tests/TurboFieldfareServer/ServerPromptCacheTests.swift` | `Tests/TurboFieldfareServer/ServerPromptCacheTests.swift` | 16/16 pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter ServerPromptCacheTests` |
| `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | `Tests/TurboFieldfareServer/OpenAIValidationTests.swift` | 44/44 across 4 suites pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter OpenAIValidationTests` |
| `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/ModelFamilyGenerationTests.swift` | 10/10 pass; exact evidence indexed in final acceptance receipt. | `Scripts/test.sh --filter ModelFamilyGenerationTests` |

### Phase 19 evidence

<a id="phase-19-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-22 | Phase baseline and staged design | 28 input snapshots, HEAD, disk, devices, worktrees and process inventory; no live P19 edits or tests yet | `baseline/`; `coordination-brief.md` |
| 2026-09-22 | Six-file staged draft and test seam | Syntax parse passed; patch SHA256 `5cdf830b1f26c6902087c727b359245fc3c0ee90e57a1e5cee278433accc7dd3`. Internal two-method generation driver preserves the real production facade and permits deterministic server tests. Full typecheck/build/tests and live promotion remain pending. | `staging/`; `baseline/companion-helper-addendum.json` |
| 2026-09-22 | Six-file production promotion and server product build | exit 0, 10.51 seconds; exact staged/live hashes match and Sources/Package unchanged during build. Main source review passed; tests still pending. | `execution/sol-server-build-20260922T133326Z/`; `main-production-review.json` |

| 2026-09-22 | Corrected companion product build | exit 0, 10.50 seconds; only authorized facade changed before build; five other server files unchanged | `execution/sol-vision-identity-build-20260922T135527Z/` |
| 2026-09-22 | Final server and shared-runtime selectors | 113 passes. Server 646-input freeze proved; focused runtime rerun changed only the authorized fixture before execution. Main accepted all 12 coverage rows. | `execution/20260922T1415Z/`; `execution/runtime-fixture-addendum-20260922T1520Z/`; `execution/codex-accepted-20260922T1528/` |

---

## Phase 20 - The app selects Qwen without moving Gemma

**When this is done:** the app selects Qwen without moving Gemma

Needs: `6, 12, 15, 16, 17`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff` — captured before new source work with 216 existing source/test snapshots.
Disk baseline: 78 GiB available on 2026-09-22; devices, worktrees and processes are in `scratch/qwen3.6-35b-a3b/evidence/phase-20/baseline/`.
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-20/`.

### Current state

`AppModelLocation.swift:3-14` defaults to `scratch/gemma4.gturbo`; `AppModelInstallDescriptor.swift:13-38` has one Gemma descriptor; `MacAppSettings.swift:47-122` stores no selection; `AppModel.swift:628-689,985-1098` owns load/unload; `RootView.swift:45-100` and `ModelInstallView.swift:42-183` expose one-model UI.

### Target state

A closed app catalog keeps Gemma selected by default and exposes Qwen only as verified or installable. A visible accessible picker performs unload→verified load→new identity/epoch, separate install/import/activation/cancel/resume for Qwen text/vision, correct diagnostics, and family codec routing in Agent Mode. Failure never moves, deletes, or replaces Gemma.

### Required runtime and presentation integration recorded on 2026-09-22

The normal Qwen app request path currently refuses string, tool and image turns; removing its guard is insufficient because the existing prepared-token conversation cannot compose or generate them. Task 20.7 therefore adds retained Qwen generation over the already loaded model and transactional state, then connects `RealInferenceClient`. It must preserve token-boundary Stop, hard-cancel rollback, checkpoint rebuild, image lineage and complete terminal tool validation. It must not load a second model or substitute stateless per-turn sessions.

The catalog/probe must recognize verified v2 receipts. Visible status, thinking controls and transcript labels must name the selected family. These necessary existing paths and new runtime tests are included below before live edits. The decode protocol and service transport already carry the required fields and need no planned source change. Host validation remains in `VisionCaptureToolLoop`; family prompt composition belongs in the runtime and inference client.

The 32 reviewed production files are now in the working tree. Catalog IDs are `gemma4-26b-a4b-it` and `qwen3.6-35b-a3b`; unknown/legacy settings select Gemma. Qwen import/conversion uses the P21 local workflow. Eleven test files are now promoted. The first compile exposed six fixture API mismatches, which have been corrected and preserved in the execution record. The later focused-test outcome and exact execution records are recorded in the app validation update below.

### Build verification on 2026-09-22

The Mac app and decode service debug builds passed with exit 0 in 11.21 and 3.32 seconds, with no warnings. Source, test, and package inputs remained unchanged during both builds. Compiler corrections preserve actor isolation, explicitly convert attachment UUIDs at the Qwen string boundary, and treat diagnostic state publication as best effort. Evidence is `execution/codex-p20-appcore-fix-20260922T153904Z/`; final input manifest SHA-256 is `b47053d623ad9fa3b92d559abf6cb5a037714fd30c56e72f735e682fddb55d44`. At that build checkpoint, focused tests, visible picker behavior, and real model qualification were not yet accepted. The later bounded acceptance is recorded below.

### Tasks

#### 20.1 Add stable app model catalog and separate locations

| | |
|---|---|
| Duty | `app-code` |
| Touches | `Sources/TurboFieldfareApp/Core/Installation/AppModelCatalog.swift` (**PROPOSED**); `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallDescriptor.swift`; `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift` |
| Depends on | `6.6, 12.6, 15.5, 16.8, 17.6` |
| Parallel safe | `yes` |
| Deliverable | Catalog entries bind stable selection ID, display name, family, expected source identity, text path, adjacent vision path, and installability. |

Keep current Gemma descriptor/path byte-for-byte in behavior. Qwen uses `scratch/qwen3.6-35b-a3b.gturbo` and never shares a destination.

**Acceptance detail**
- [x] Legacy/default selection resolves to Gemma.
- [x] No Qwen install/remove operation targets a Gemma path.

**How to check**
```sh
Scripts/test.sh --filter AppModelCatalogTests
```

**Evidence:** Accepted 2026-09-22 from the focused logs mapped in `coordination/p20-acceptance-reconciliation.md`. Bounded fixtures do not claim official-model or live host execution.

#### 20.2 Persist selection with backward-compatible settings

| | |
|---|---|
| Duty | `app-code` |
| Touches | `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift` |
| Depends on | `20.1` |
| Parallel safe | `yes` |
| Deliverable | Persist stable selected model ID and any family-scoped settings with migration from files that lack the field. |

Do not silently change sampling settings during a retained conversation.

**Acceptance detail**
- [x] Old settings decode to Gemma selection.
- [x] Unknown selection fails to Gemma/unloaded safe state without deleting data.

**How to check**
```sh
Scripts/test.sh --filter AppModelSelectionSettingsTests
```

**Evidence:** Accepted 2026-09-22 from the focused logs mapped in `coordination/p20-acceptance-reconciliation.md`. Bounded fixtures do not claim official-model or live host execution.

#### 20.3 Bind app lifecycle to model identity and epoch

| | |
|---|---|
| Duty | `app-code` |
| Touches | `Sources/TurboFieldfareApp/Core/State/AppModel.swift` |
| Depends on | `20.1, 20.2` |
| Parallel safe | `no` |
| Deliverable | Implement selection only while lifecycle permits, unload current runtime, load verified target, bind returned descriptor to service epoch, and archive visible transcript when context lineage changes. |

Failed Qwen load leaves a clear unloaded state and Gemma selectable; never keep two loaded models.

**Acceptance detail**
- [x] Idle switch follows unload→load→new epoch order.
- [x] Running switch is refused/cancelled by explicit UI policy and stale responses are ignored.

**How to check**
```sh
Scripts/test.sh --filter AppModelSelectionTests
```

**Evidence:** Accepted 2026-09-22 from the focused logs mapped in `coordination/p20-acceptance-reconciliation.md`. Bounded fixtures do not claim official-model or live host execution.

#### 20.4 Route installation and vision status by selection

| | |
|---|---|
| Duty | `app-code` |
| Touches | `Sources/TurboFieldfareApp/Core/Installation/RepackModelInstallerClient.swift`; `Sources/TurboFieldfareApp/Core/Installation/RepackVisionPackInstallerClient.swift` |
| Depends on | `20.1, 20.3` |
| Parallel safe | `no` |
| Deliverable | Send catalog source/destination into install/import/activate/cancel/resume and keep text/vision readiness separate per model. |

Installation cannot overwrite completed artifacts and does not authorize the full P21 conversion.

**Acceptance detail**
- [x] Qwen partial/resume/discard affects only Qwen-owned paths.
- [x] Missing/mismatched Qwen vision blocks images but not verified text load.

**How to check**
```sh
Scripts/test.sh --filter AppQwenInstallationTests
```

**Evidence:** Accepted 2026-09-22 from the focused logs mapped in `coordination/p20-acceptance-reconciliation.md`. Bounded fixtures do not claim official-model or live host execution.

#### 20.5 Add a visible accessible model picker

| | |
|---|---|
| Duty | `app-ui` |
| Touches | `Sources/TurboFieldfareApp/Mac/App/RootView.swift`; `Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift`; `Sources/TurboFieldfareApp/Mac/App/TurboFieldfareMacApp.swift` native Model menu |
| Depends on | `20.3` |
| Parallel safe | `no` |
| Deliverable | Place the picker in the actual Mac app hierarchy and show selected name, verified/unverified status, text/vision install state, load transition, and rollback choice. |

Provide stable accessibility labels/values and keyboard focus; do not place production picker code under `Tests/`.

**Acceptance detail**
- [ ] Keyboard and VoiceOver users can identify and choose each available model.
- [ ] The UI never claims Qwen loaded before the service returns its descriptor.

**How to check**
```sh
Scripts/test.sh --filter AppModelPickerPresentationTests
```

**Evidence:** Partial direct keyboard rollback proof is recorded in `execution/luna6-direct-model-shortcuts-20260922T190503Z/`. The bounded VoiceOver attempt in `execution/luna6-voiceover-20260922T192634Z/` reached first-use setup without verifying a model-selection action. Available-Qwen selection, actual VoiceOver identification/selection and verified loaded-state presentation remain pending.

#### 20.6 Report selected identity and family diagnostics

| | |
|---|---|
| Duty | `app-ui` |
| Touches | `Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift`; `Sources/TurboFieldfareApp/MacPresentation/AppModelIdentityPresentation.swift` (**PROPOSED**) |
| Depends on | `20.3` |
| Parallel safe | `yes` |
| Deliverable | Show verified model ID, source revision, format, quantization profile, vision status, state bytes, and expert-cache bytes from runtime diagnostics. |

Do not expose a user-entered label as verified identity. The runtime byte reporting paths in this phase’s coverage table carry committed conversation bytes and currently owned expert-cache Metal buffer bytes through existing service frames. Optional fields preserve legacy decoding; unload and failure clear the reported values. Gemma source revision remains unavailable when its legacy runtime does not report it.

**Acceptance detail**
- [ ] Inspector values equal the loaded descriptor/runtime diagnostics.
- [ ] Unloaded and failed states show no stale prior identity.

**How to check**
```sh
Scripts/test.sh --filter AppModelPickerPresentationTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-20/`.

#### 20.7 Route ordinary Qwen app and Agent turns through retained family generation

| | |
|---|---|
| Duty | `app-code` |
| Touches | `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift`; `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift`; narrow `QwenConversationState.swift`, `QwenChatCodec.swift`, and `QwenStructuredAssistantDecoder.swift` support; existing `VisionCaptureToolLoop.swift` remains the host validation boundary. |
| Depends on | `20.3, 20.4` |
| Parallel safe | `no` |
| Deliverable | Use the selected family codec for prompt/result serialization while preserving VisionCapture tool allowlists, returned-identity validation, verified verdict, target locking, and no-replay rules. |

The model parser proposes calls; host validation remains the source of truth.

**Acceptance detail**
- [x] Gemma and Qwen serialize differently but reach identical host permission checks.
- [x] A codec/model identity mismatch stops before any host request.

**How to check**
```sh
Scripts/test.sh --filter AppAgentCodecRoutingTests
```

**Evidence:** Accepted 2026-09-22 from the focused logs mapped in `coordination/p20-acceptance-reconciliation.md`. Bounded fixtures do not claim official-model or live host execution.

#### 20.8 Add catalog, lifecycle, install, picker, and Agent tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift` (**PROPOSED**); `Tests/TurboFieldfareApp/Core/Configuration/AppModelSelectionSettingsTests.swift` (**PROPOSED**); `Tests/TurboFieldfareApp/Core/State/AppModelSelectionTests.swift` (**PROPOSED**); `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` (**PROPOSED**); `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` (**PROPOSED**); `Tests/TurboFieldfareApp/Core/Tools/AppAgentCodecRoutingTests.swift` (**PROPOSED**) |
| Depends on | `20.1` |
| Parallel safe | `yes` |
| Deliverable | Tests cover migration, accessible labels, idle/running switch, failed load, stale response, separate install paths, cancel/resume, missing vision, diagnostics, codec selection, and Gemma rollback. |

Test owners edit only their assigned coverage paths, including the retained-runtime, inference and installation-probe regressions added after the concrete app-path map.

**Acceptance detail**
- [x] All six suites execute with nonzero counts.
- [x] No switch, install, or tool route crosses selected model identity.

**How to check**
```sh
Scripts/test.sh --filter AppModelSelection
```

**Evidence:** Accepted 2026-09-22 from the focused logs mapped in `coordination/p20-acceptance-reconciliation.md`. Bounded fixtures do not claim official-model or live host execution.

#### 20.9 Re-run existing app lifecycle and VisionCapture truth tests

| | |
|---|---|
| Duty | `compat-test` |
| Touches | `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelLoadPhaseOrderTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppGenerationRunIdentityTests.swift`; `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` |
| Depends on | `20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8` |
| Parallel safe | `no` |
| Deliverable | Run existing settings, phase order, run identity, and host returned-identity tests after app integration. |

Do not weaken Gemma launch/default behavior or VisionCapture verification.

**Acceptance detail**
- [x] All four existing suites execute with nonzero counts.
- [x] Gemma remains default and host truth checks retain current outcomes.

**How to check**
```sh
Scripts/test.sh --filter AppModelLoadPhaseOrderTests
```

**Evidence:** Accepted 2026-09-22 from the focused logs mapped in `coordination/p20-acceptance-reconciliation.md`. Bounded fixtures do not claim official-model or live host execution.

### Phase 20 coverage plan

All 49 intended paths are mapped to actual test files. The tracker records 43 accepted bounded unit rows and six partially verified UI rows; official-model and host workflow qualification remain separate.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareApp/Core/Installation/AppModelCatalog.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift` | AppModelCatalogTests selector 6/6 pass; AppModelLocationTests selector 5/5 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelCatalogTests`; `Scripts/test.sh --filter AppModelLocationTests` |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallDescriptor.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | AppModelCatalogTests selector 6/6 pass; AppQwenInstallationTests selector 8/8 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelCatalogTests`; `Scripts/test.sh --filter AppQwenInstallationTests` |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppModelLocationTests.swift` | AppModelCatalogTests selector 6/6 pass; AppModelLocationTests selector 5/5 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelCatalogTests`; `Scripts/test.sh --filter AppModelLocationTests` |
| `Sources/TurboFieldfareApp/Core/Configuration/MacAppSettings.swift` | `Tests/TurboFieldfareApp/Core/Configuration/AppModelSelectionSettingsTests.swift`; `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift` | AppModelSelectionSettingsTests selector 4/4 pass; MacAppSettingsTests selector 17/17 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelSelectionSettingsTests`; `Scripts/test.sh --filter MacAppSettingsTests` |
| `Sources/TurboFieldfareApp/Core/State/AppModel.swift` | `Tests/TurboFieldfareApp/Core/State/AppModelSelectionTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelLoadPhaseOrderTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelConversationTests.swift` | AppModelSelectionTests selector 9/9 pass; AppModelLoadPhaseOrderTests selector 5/5 pass; AppModelConversationTests selector 12/12 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelSelectionTests`; `Scripts/test.sh --filter AppModelLoadPhaseOrderTests`; `Scripts/test.sh --filter AppModelConversationTests` |
| `Sources/TurboFieldfareApp/Core/Installation/RepackModelInstallerClient.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallTests.swift` | AppQwenInstallationTests selector 8/8 pass; AppModelInstallTests selector 14/14 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppQwenInstallationTests`; `Scripts/test.sh --filter AppModelInstallTests` |
| `Sources/TurboFieldfareApp/Core/Installation/RepackVisionPackInstallerClient.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | AppQwenInstallationTests selector 8/8 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppQwenInstallationTests` |
| `Sources/TurboFieldfareApp/Mac/App/RootView.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | AppModelPickerPresentationTests selector 6/6 pass. Direct UI acceptance remains partial. | `Scripts/test.sh --filter AppModelPickerPresentationTests` |
| `Sources/TurboFieldfareApp/Mac/App/TurboFieldfareMacApp.swift` | native UI evidence; existing AppModelSelectionTests | Clean Mac build; direct Cmd-Option-1 changed fixture selection from Qwen to available Gemma, repeated selection and unavailable-Qwen shortcut were no-ops. Available-Qwen selection and VoiceOver remain unproven. | Isolated native-app keyboard evidence in `execution/luna6-direct-model-shortcuts-20260922T190503Z/`; existing AppModelSelectionTests |
| `Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | AppModelPickerPresentationTests selector 6/6 pass; AppQwenInstallationTests selector 8/8 pass. Direct UI acceptance remains partial. | `Scripts/test.sh --filter AppModelPickerPresentationTests`; `Scripts/test.sh --filter AppQwenInstallationTests` |
| `Sources/TurboFieldfareApp/Mac/Diagnostics/InspectorView.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallationProbeTests.swift` | AppModelPickerPresentationTests selector 6/6 pass; AppModelInstallationProbeTests selector 5/5 pass. Direct UI acceptance remains partial. | `Scripts/test.sh --filter AppModelPickerPresentationTests`; `Scripts/test.sh --filter AppModelInstallationProbeTests` |
| `Sources/TurboFieldfareApp/Core/Tools/VisionCaptureToolLoop.swift` | `Tests/TurboFieldfareApp/Core/Tools/AppAgentCodecRoutingTests.swift`; `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` | AppAgentCodecRoutingTests selector 4/4 pass; VisionCaptureReturnedIdentityTests selector 11/11 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppAgentCodecRoutingTests`; `Scripts/test.sh --filter VisionCaptureReturnedIdentityTests` |
| `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelCatalogTests.swift` | AppModelCatalogTests selector 6/6 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelCatalogTests` |
| `Tests/TurboFieldfareApp/Core/Configuration/AppModelSelectionSettingsTests.swift` | `Tests/TurboFieldfareApp/Core/Configuration/AppModelSelectionSettingsTests.swift` | AppModelSelectionSettingsTests selector 4/4 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelSelectionSettingsTests` |
| `Tests/TurboFieldfareApp/Core/State/AppModelSelectionTests.swift` | `Tests/TurboFieldfareApp/Core/State/AppModelSelectionTests.swift` | AppModelSelectionTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelSelectionTests` |
| `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | AppQwenInstallationTests selector 8/8 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppQwenInstallationTests` |
| `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | AppModelPickerPresentationTests selector 6/6 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelPickerPresentationTests` |
| `Tests/TurboFieldfareApp/Core/Tools/AppAgentCodecRoutingTests.swift` | `Tests/TurboFieldfareApp/Core/Tools/AppAgentCodecRoutingTests.swift` | AppAgentCodecRoutingTests selector 4/4 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppAgentCodecRoutingTests` |
| `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift` | `Tests/TurboFieldfareApp/Core/Configuration/MacAppSettingsTests.swift` | MacAppSettingsTests selector 17/17 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter MacAppSettingsTests` |
| `Tests/TurboFieldfareApp/Core/State/AppModelLoadPhaseOrderTests.swift` | `Tests/TurboFieldfareApp/Core/State/AppModelLoadPhaseOrderTests.swift` | AppModelLoadPhaseOrderTests selector 5/5 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelLoadPhaseOrderTests` |
| `Tests/TurboFieldfareApp/Core/State/AppGenerationRunIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/State/AppGenerationRunIdentityTests.swift` | AppGenerationRunIdentityTests selector 1/1 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppGenerationRunIdentityTests` |
| `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/Tools/VisionCaptureReturnedIdentityTests.swift` | VisionCaptureReturnedIdentityTests selector 11/11 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter VisionCaptureReturnedIdentityTests` |
| `Sources/TurboFieldfare/Runtime/Inference/QwenConversationGeneration.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | QwenConversationGenerationTests selector 13/13 pass; RealInferenceClientQwenRoutingTests selector 27/27 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter QwenConversationGenerationTests`; `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift` | QwenConversationGenerationTests selector 13/13 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter QwenConversationGenerationTests` |
| `Sources/TurboFieldfare/Tokenization/QwenChatCodec.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift`; `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | QwenConversationGenerationTests selector 13/13 pass; QwenChatTemplateTests selector 21/21 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter QwenConversationGenerationTests`; `Scripts/test.sh --filter QwenChatTemplateTests` |
| `Sources/TurboFieldfare/Tokenization/QwenStructuredAssistantDecoder.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift`; `Tests/TurboFieldfare/Core/Tokenization/QwenStructuredAssistantDecoderTests.swift` | QwenConversationGenerationTests selector 13/13 pass; QwenStructuredAssistantDecoderTests selector 16/16 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter QwenConversationGenerationTests`; `Scripts/test.sh --filter QwenStructuredAssistantDecoderTests` |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift` | RealInferenceClientQwenRoutingTests selector 27/27 pass; QwenConversationGenerationTests selector 13/13 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests`; `Scripts/test.sh --filter QwenConversationGenerationTests` |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelInstallationProbe.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallationProbeTests.swift`; `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | AppModelInstallationProbeTests selector 5/5 pass; AppQwenInstallationTests selector 8/8 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelInstallationProbeTests`; `Scripts/test.sh --filter AppQwenInstallationTests` |
| `Sources/TurboFieldfareApp/Mac/Components/ModelStatusBadge.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | AppModelPickerPresentationTests selector 6/6 pass. Direct UI acceptance remains partial. | `Scripts/test.sh --filter AppModelPickerPresentationTests` |
| `Sources/TurboFieldfareApp/Mac/Generation/OutputPaneView.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelTests.swift` | AppModelPickerPresentationTests selector 6/6 pass; AppModelTests selector 26/26 pass. Direct UI acceptance remains partial. | `Scripts/test.sh --filter AppModelPickerPresentationTests`; `Scripts/test.sh --filter AppModelTests` |
| `Sources/TurboFieldfareApp/MacPresentation/InstructionTranscriptDocumentController.swift` | `Tests/TurboFieldfareApp/Core/State/AppModelTests.swift`; `Tests/TurboFieldfareApp/Core/State/AppModelConversationTests.swift` | AppModelTests selector 26/26 pass; AppModelConversationTests selector 12/12 pass. Direct UI acceptance remains partial. | `Scripts/test.sh --filter AppModelTests`; `Scripts/test.sh --filter AppModelConversationTests` |
| `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift` | QwenConversationGenerationTests selector 13/13 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter QwenConversationGenerationTests` |
| `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift` | RealInferenceClientQwenRoutingTests selector 27/27 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests` |
| `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallationProbeTests.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppModelInstallationProbeTests.swift` | AppModelInstallationProbeTests selector 5/5 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelInstallationProbeTests` |
| `Sources/TurboFieldfareApp/Core/Installation/AppVisionPackInstallationProbe.swift` | `Tests/TurboFieldfareApp/Core/Installation/AppQwenInstallationTests.swift` | AppQwenInstallationTests selector 8/8 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppQwenInstallationTests` |
| `Sources/TurboFieldfareApp/MacPresentation/AppModelIdentityPresentation.swift` | `Tests/TurboFieldfareApp/MacPresentation/AppModelPickerPresentationTests.swift` | AppModelPickerPresentationTests selector 6/6 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppModelPickerPresentationTests` |
| `Sources/TurboFieldfare/Runtime/Vision/MultimodalPromptRenderer.swift` | `Tests/TurboFieldfare/Core/Runtime/Inference/QwenConversationGenerationTests.swift`; `Tests/TurboFieldfare/Core/Tokenization/QwenChatTemplateTests.swift` | QwenConversationGenerationTests selector 13/13 pass; QwenChatTemplateTests selector 21/21 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter QwenConversationGenerationTests`; `Scripts/test.sh --filter QwenChatTemplateTests` |
| `Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass; RealInferenceClientQwenRoutingTests selector 27/27 pass; DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests`; `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests`; `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Sources/TurboFieldfare/Runtime/Inference/Model.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass; RealInferenceClientQwenRoutingTests selector 27/27 pass; DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests`; `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests`; `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass; RealInferenceClientQwenRoutingTests selector 27/27 pass; DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests`; `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests`; `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenTextRunner.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass; RealInferenceClientQwenRoutingTests selector 27/27 pass; DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests`; `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests`; `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Sources/TurboFieldfareApp/Core/Inference/AppInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass; RealInferenceClientQwenRoutingTests selector 27/27 pass; DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests`; `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests`; `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Sources/TurboFieldfareApp/Core/Diagnostics/AppDiagnostics.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass; RealInferenceClientQwenRoutingTests selector 27/27 pass; DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests`; `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests`; `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass; DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests`; `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Sources/TurboFieldfareDecodeService/DecodeServiceOutbox.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift` | DecodeServiceOutboxTests selector 9/9 pass; AppRuntimeByteReportingTests selector 5/5 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter DecodeServiceOutboxTests`; `Scripts/test.sh --filter AppRuntimeByteReportingTests` |
| `Sources/TurboFieldfareDecodeService/Entry.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift` | DecodeServiceOutboxTests selector 9/9 pass; AppRuntimeByteReportingTests selector 5/5 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter DecodeServiceOutboxTests`; `Scripts/test.sh --filter AppRuntimeByteReportingTests` |
| `Sources/TurboFieldfareApp/Core/Inference/DecodeServiceInferenceClient.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift`; `Tests/TurboFieldfareApp/Core/Inference/RealInferenceClientStateTests.swift`; `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass; RealInferenceClientQwenRoutingTests selector 27/27 pass; DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests`; `Scripts/test.sh --filter RealInferenceClientQwenRoutingTests`; `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppRuntimeByteReportingTests.swift` | AppRuntimeByteReportingTests selector 5/5 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter AppRuntimeByteReportingTests` |
| `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | `Tests/TurboFieldfareDecodeService/DecodeServiceOutboxTests.swift` | DecodeServiceOutboxTests selector 9/9 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter DecodeServiceOutboxTests` |
| `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/DecodeServiceModelIdentityTests.swift` | DecodeServiceModelIdentityTests selector 26/26 pass. Bounded unit evidence accepted. | `Scripts/test.sh --filter DecodeServiceModelIdentityTests` |

### App validation update on 2026-09-22

The final app selection and decode-cleanup corrections passed all 116 tests across ten focused selectors in `execution/main-selection-rerun-20260922T174125Z/`. Earlier retained runtime, routing, codec, settings and host-validator receipts also passed and are mapped in `coordination/p20-acceptance-reconciliation.md`. All before/after source, test and package hashes stayed equal during each run. Failed earlier fixture attempts remain preserved and are not counted as acceptance.

The earlier Mac and decode-service builds passed in 11.48 and 1.15 seconds. Two isolated app checks verified the visible Qwen selection, both menu choices, missing-Gemma failure, accessible selected-name/status text, and no verified loaded identity. No model or decode service started, both test apps were closed, and the live settings hash stayed unchanged. Temporary app wrappers were needed because computer automation could not bind the bare SwiftPM executable. At that checkpoint keyboard-only selection and actual VoiceOver were unproven; loaded Inspector/transcript rendering still needs direct evidence. Receipts are under `ui-presentation/picker-fixture-20260922T165605Z/` and `ui-presentation/picker-keyboard-20260922T174822Z/`.

Tasks 20.1–20.4 and 20.7–20.9 satisfy their stated bounded code/test criteria. Split receipts satisfy 20.9 because all four required suites ran with nonzero counts and zero failures. Tasks 20.5–20.6 and the whole phase remain open. No official Qwen artifact or full UI-to-host workflow is qualified here.

A further focused check using the final compiled app proved keyboard operation after a direct coordinate click opened the picker: Up moved to Gemma and Return selected it, with the expected missing-install failure and no loaded identity. Tab did not move reported focus, so that check did not establish keyboard-only traversal or actual VoiceOver. Both owned app processes were closed and the live settings hash remained unchanged. Receipt: `execution/main-coordinate-keyboard-20260922T181933Z/result.json`.

### Native keyboard selection correction — 2026-09-22

GPT-6 Sol added checked model controls to the existing native Model menu, with Command-Option-1 for Gemma and Command-Option-2 for Qwen. The controls retain both lifecycle availability guards and call the existing `selectModel` action. Re-selecting the checked model is a no-op. The unsupported/unproven shortcut on the visible Picker was removed; its selected-name, status and accessibility presentation remain. Main reviewed both diffs before promotion. Final source hashes are `1f024d699c75be79eee9de952d6e127943db829d40cb059f978ff83de6cc8187` for RootView and `e57dddf5999d1a1c68e4c2f715b0d87975625be8b0d1aa8f7c0e0b31b79e916e` for TurboFieldfareMacApp. The native-command path was added to coverage before promotion, bringing this phase to 50 paths: 43 accepted bounded unit rows and seven partial UI rows.

GPT-6 Luna built the final Mac product with exit 0 in 4.24 seconds, no warnings, and all 672 source/test/package input hashes unchanged. This declarative wiring reuses the already-tested selection action; no unit-test rerun or new mirrored test is claimed. A first direct-key attempt was non-diagnostic because the fixture started with Gemma selected and no Qwen entry eligible for registration. Its no-ops were expected and are preserved with that corrected interpretation. A separate Control-F2 menu-focus attempt was unavailable through the computer-use session and does not prove a production failure.

The corrected no-model fixture changed only its temporary initial selected ID to Qwen. One direct Command-Option-1 then selected Gemma. The visible picker and accessibility state reported Gemma's expected missing-install load failure, and Inspector loaded identity remained None. Repeating the Gemma shortcut did not deselect it; the now-unavailable Qwen shortcut did nothing. This proves keyboard rollback to an available Gemma entry, not selection of an available Qwen installation, actual VoiceOver announcements, or a loaded-to-unloaded identity transition. Tasks 20.5 and 20.6 remain open.

Evidence is `execution/luna6-direct-model-shortcuts-20260922T190503Z/`. The corrected-attempt receipt is `corrected-qwen-selected-20260922T1909Z/corrected-attempt.json`, SHA-256 `9917d8ed85d0a30127c5ec5c8314a2b6003b090b34d3ac2c7c4897568576b70f`. The owned launch session 34718 exited 0 after Command-Q. Its PID was not captured before closure, so no PID claim is made. Main separately confirmed no prohibited process remained, the final source hashes matched, both user settings hashes were unchanged, and fixture settings were restored byte-for-byte. No model, inference, installation or unit suite was run during this correction.

### VoiceOver setup observation — 2026-09-22

GPT-6 Luna at xhigh reasoning tested the existing isolated app without rebuilding or running a model. Command-F5 and Option-Command-F5 produced no observable VoiceOver activation through computer use. The System Settings switch was then temporarily changed from off to on. The test-started VoiceOver Quickstart process and welcome dialog appeared, but the fixture showed no verifiable VoiceOver menu, focus or caption change after the bounded navigation commands. The agent stopped before entering setup and restored the switch to off. This is an incomplete accessibility observation, not proof of a product defect or successful VoiceOver selection.

The first-use dialog was dismissed and the owned app closed. Main independently observed no matching VoiceOver, app, model or test process and confirmed both user app-settings files were unchanged. Evidence is `execution/luna6-voiceover-20260922T192634Z/`, with Main's review in `execution/main-voiceover-review-20260922T193531Z/`. Tasks 20.5 and 20.6 remain open; no coverage row or acceptance box is newly accepted.

### Phase 20 evidence

<a id="phase-20-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-22 | Phase baseline and bounded runtime/app map | 216 app/protocol/service snapshots plus runtime addendum; ordinary Qwen refusal and required retained generation identified. Staged implementation started; no build/test/model qualification. | `baseline/`; `coordination/p20-existing-path-map.md` |
| 2026-09-22 | Native menu correction and direct keyboard rollback | Clean final Mac build; Qwen-selected fixture switched to available Gemma by Cmd-Option-1, repeat and unavailable target stayed unchanged. Broader UI acceptance remains pending. | `execution/luna6-direct-model-shortcuts-20260922T190503Z/`; `coordination/sol6-picker-keyboard-correction.md` |
| 2026-09-22 | Bounded VoiceOver identification and selection attempt | First-use setup appeared; model identification/selection was not verified. VoiceOver switch restored off and owned UI closed. No new accessibility acceptance. | `execution/luna6-voiceover-20260922T192634Z/`; `execution/main-voiceover-review-20260922T193531Z/` |

---


## Phase 21 - Authorized conversion writes Qwen beside Gemma

**When this is done:** authorized conversion writes Qwen beside Gemma

Needs: `2, 4, 6, 7`
Base commit: `5770260510935cd32b82c4008ff06ae6458539ff` — captured before new source work with 77 existing source/test snapshots.
Disk baseline: 78 GiB available on 2026-09-22; devices, worktrees and processes are in `scratch/qwen3.6-35b-a3b/evidence/phase-21/baseline/`.
Evidence: `scratch/qwen3.6-35b-a3b/evidence/phase-21/`.

### Current state

The official BF16 bundle is prepared but unconverted. Current `Sources/TurboFieldfareRepack/Command/main.swift:4-155` exposes remote Gemma/vision modes and no local official-source conversion mode. Current free disk is unknown and the historic preparation measurement cannot authorize output creation.

### Target state

After separate execution authorization, the repack CLI validates local source, plans/preflights, transforms one tensor stream at a time, resumes safely, writes v2 text plus bound vision companion, audits every output, publishes receipts, and never targets `scratch/gemma4.gturbo`.

### Complete local conversion boundary recorded on 2026-09-22

The existing planner and transformed writer supply bounded conversion and resume. The workflow must also authenticate pinned shard payloads, create v2 text/vision receipts accepted by runtime loading, sum both destinations on a shared volume, validate ownership before partial discard, audit receipt-inclusive inventories, and roll back paired publication if the second rename or final audit fails. These are required to produce a runnable verified installation, and their exact files/tests are recorded below before live promotion.

The prepared `SHA256SUMS` document is 3,571 bytes with 38 entries, including 26 shard hashes; its SHA256 is `1378d1bb153694b13c20641fa5d2485dede401d1ce4c267953ba4a855fc59e7e`. The publisher LFS identities are saved in the preparation API response and manifest. Staged implementation must not open authentic payloads yet. The v2 artifact receipt needs deterministic provenance and final-directory binding; any canonical timestamp must be explicitly marked, with actual verification time recorded separately. Existing Gemma receipt behavior remains.

Sol authors the seven staged production paths and Luna owns CLI/workflow/payload/resume tests. CLI tests exercise the actual local-command parser exposed through the workflow module. A real conversion remains subject to the separate operational authorization and preflight in tasks 21.4 and 21.5.

### Pre-promotion review on 2026-09-22

Independent review found four issues: resume requested fresh full-output disk space, discard trusted a checkpoint without validating the current partial contents, resume did not authenticate the fixed packed-expert layout, and publication ignored checkpoint cleanup failures. Sol corrected the workflow/writer and Main reviewed the changes. All seven production files were promoted, and `swift build --product TurboFieldfareRepack` passed with exit 0 in 8.58 seconds (10 seconds wall time). Sources, tests and Package.swift stayed unchanged during the build. Luna is running the four conversion suites and focused writer/quantizer regressions, including paired text-and-vision resume equality at the same destination. Build evidence is `execution/codex-p21-promotion-20260922T144822Z/`. The static durability audit below identified a checkpoint bottleneck before real conversion. Initial test runs exposed fixture/help expectations that Luna is correcting in the owned test files while preserving failed logs. No real source conversion has run.

### Published receipt path correction

A standalone Foundation reproduction proved that standardizing the physical `/private/var/...` output path changes its spelling to `/var/...` after publication. The repacker and runtime could therefore reject a valid path-bound receipt. The Qwen conversion path will use stable physical directory binding and verify the emitted receipt through the actual runtime reader. Legacy Gemma receipt behavior remains unchanged. The extra runtime source and test path were recorded before editing in `baseline/receipt-path-binding/`; reproduction evidence is `coordination/foundation-path-repro/`. No model or authentic payload was used.

### Corrected converter build on 2026-09-22

Main reviewed and promoted exactly four files for durability batching and Qwen physical receipt binding. `swift build --product TurboFieldfareRepack` passed with exit 0 in 4.45 seconds (4.86 seconds wall time). The post-promotion and post-build input manifests match at SHA-256 `6370456b3b53c21a2773c7125e4b10cbb7b822977192641288f03723040f9f07`. Evidence is `execution/promotion-build-20260922T154443Z/`. Luna now owns focused workflow, resume, generic writer, and actual runtime receipt-reader tests. No real conversion has run.

### Operational reporting required before conversion

The CLI now exposes `--preflight-only`, exact pre-write capacity, durable conversion progress, and maximum observed transform scratch. The scratch value measures transform buffers, not total process memory. The three reviewed source changes are promoted. An initial Mac build caught a trailing-closure compatibility problem; the workflow now preserves its original overloads and requires both callbacks in the new reporting overload. The corrected release repacker build passed in 24.44 seconds and the Mac build passed in 7.07 seconds, both with exit 0. The repacker retained one existing unrelated unused-result warning; the Mac build had no diagnostics. The 308 source and package inputs stayed unchanged across those corrected builds. Tests were excluded from that build comparison because their separate fixture corrections were authorized concurrently. Evidence is `execution/operational-reporting-20260922T162250Z/`; independent review receipt SHA-256 is `937297aec10514b1a828080c746996f16021a1e39407b23dbdc22eda5192e772`.

The prior corrected workflow, resume, generic writer and receipt suites passed 37/37 tests. The added reporting and compatibility cases are now promoted, giving 17 CLI, 13 workflow and 15 resume cases. Their focused rerun passed all 58 tests with no failures. Tasks 21.1–21.3 are accepted. The payload, quantizer and reader sources/tests match the earlier passed snapshots (17 additional unchanged cases). One conditional approval for the actual preflight and conversion has been requested and is pending. The operation brief is `coordination/conversion-operation-brief.md`.

### Conversion durability correction

The metadata-only audit counted 541,801,847 writer units and 1,625,405,543 scoped disk-sync calls for the default full plan. Recovery checkpoint writes after each 64-element affine group make the current conversion impractical. Sol implemented Qwen affine durability batching at 65,536 groups or request end while retaining per-group writes and hash order, the unchanged file/receipt format, bounded scratch memory, and overwrite-on-resume of any uncommitted tail. The promoted writer digest is `7f4032f08265ea6a8c3e347ba6bcc32761c3964532f85b0b15e51efac86fb95d`; focused durability tests are running. Main requested explicit recovery from a saved nonzero prefix with an uncommitted tail. The generic writer default and retained-BF16 tile behavior remain unchanged. Luna owns independent crash-window tests in the existing writer test file. The proposed full-plan schedule is 23,501 commits and 70,505 scoped disk-sync calls; no elapsed-time speed claim or real conversion has been made. Evidence is `coordination/conversion-durability-cost.md` and the pre-edit baseline is `baseline/durability-batch-addendum.json`.

### Tasks

#### 21.1 Add explicit local official-source CLI mode

| | |
|---|---|
| Duty | `repack-cli` |
| Touches | `Sources/TurboFieldfareRepack/Command/main.swift` |
| Depends on | `2.6, 4.5, 6.6, 7.6` |
| Parallel safe | `no` |
| Deliverable | Add `--local-source`, `--output`, resume/discard, and text/vision selection with mutually exclusive remote modes and explicit official-Qwen validation. |

Parse and validate all paths/options before writing. Help must state that conversion is local, deterministic, resumable, and never overwrites a completed destination.

**Acceptance detail**
- [x] Valid local Qwen arguments produce a plan request without network access.
- [x] Conflicts, wrong identity, existing destination, and Gemma destination fail before writing.

**How to check**
```sh
Scripts/test.sh --filter QwenLocalRepackCLITests
```

**Evidence:** Accepted 2026-09-22. Final reporting rerun: CLI 17/17, workflow 13/13, resume 15/15, generic writer 10/10, receipt path 3/3. Exact logs, timings and hashes are in `coordination/final-reporting-focused-results.json`. Unchanged payload verification retains its prior 7/7 pass. Actual conversion remains pending Tasks 21.4–21.6.

#### 21.2 Connect CLI mode to plan, transform, audit, and receipt

| | |
|---|---|
| Duty | `repack-code` |
| Touches | `Sources/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepacker.swift`; `Core/Local/LocalOfficialQwenPayloadVerifier.swift`; `Core/Writing/TransformedTensorWriter.swift`; `Core/Verification/VerifiedInstallReceiptWriter.swift`; `Core/Verification/RepackAudit.swift`; `Core/System/DiskSpaceChecker.swift` |
| Depends on | `21.1` |
| Parallel safe | `no` |
| Deliverable | Execute P5/P6/P7 components sequentially, publish text then compatible vision receipts only after full audit, and preserve resumable partials on cancellation. |

Use one bounded source/output stream at a time. Never load model weights into the inference runtime.

**Acceptance detail**
- [x] Tiny end-to-end local conversion publishes only after verified audit.
- [x] Cancellation/resume and discard affect only Qwen-owned partial paths.

**How to check**
```sh
Scripts/test.sh --filter LocalQwenStreamingRepackerTests
```

**Evidence:** Accepted 2026-09-22. Final reporting rerun: CLI 17/17, workflow 13/13, resume 15/15, generic writer 10/10, receipt path 3/3. Exact logs, timings and hashes are in `coordination/final-reporting-focused-results.json`. Unchanged payload verification retains its prior 7/7 pass. Actual conversion remains pending Tasks 21.4–21.6.

#### 21.3 Add local CLI and workflow unit tests

| | |
|---|---|
| Duty | `test` |
| Touches | `Tests/TurboFieldfareRepack/Core/Command/QwenLocalRepackCLITests.swift` (**PROPOSED**); `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` (**PROPOSED**) |
| Depends on | `21.1` |
| Parallel safe | `yes` |
| Deliverable | Tests cover help, conflicts, source identity, path isolation, existing output, plan/preflight failure, cancellation/resume, text/vision receipt binding, and audit-before-publish using tiny data. |

No official shard or full conversion is read by unit tests.

**Acceptance detail**
- [x] Both suites execute with nonzero counts.
- [x] Every failure leaves Gemma and completed artifacts untouched.

**How to check**
```sh
Scripts/test.sh --filter LocalQwen
```

**Evidence:** Accepted 2026-09-22. Final reporting rerun: CLI 17/17, workflow 13/13, resume 15/15, generic writer 10/10, receipt path 3/3. Exact logs, timings and hashes are in `coordination/final-reporting-focused-results.json`. Unchanged payload verification retains its prior 7/7 pass. Actual conversion remains pending Tasks 21.4–21.6.

#### 21.4 Record current capacity and process preflight

| | |
|---|---|
| Duty | `operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-21/preflight.txt` (**PROPOSED**) |
| Depends on | `21.2, 21.3` |
| Parallel safe | `no` |
| Deliverable | Only after explicit authorization, record macOS/Swift, `memory_pressure -Q`, free bytes, source/destination requirements, source completeness, and absence of prohibited model processes. |

If any AGENTS.md model-run prerequisite fails, stop. Never kill an app, purge caches, duplicate weights, or use the historic free-space value.

**Acceptance detail**
- [ ] Preflight records the exact plan-required and currently available bytes.
- [ ] No conversion command starts when a prerequisite fails.

**How to check**
```sh
sw_vers && swift --version && memory_pressure -Q && df -k scratch && pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-21/`.

#### 21.5 Run one authorized official conversion with resume evidence

| | |
|---|---|
| Duty | `operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-21/conversion.log` (**PROPOSED**) |
| Depends on | `21.4` |
| Parallel safe | `no` |
| Deliverable | Run one foreground release repacker from canonical source to `scratch/qwen3.6-35b-a3b.gturbo`, recording commit, command, exit, progress, resume behavior if interrupted, peak scratch, and final text/vision sizes. |

This task is future operational authorization only. Do not run it without explicit conversion authorization and repository preflight.

**Acceptance detail**
- [ ] The command exits zero once and publishes audited receipts.
- [ ] Source bundle, Gemma artifact, and any pre-existing receipt remain unchanged.

**How to check**
```sh
swift run -c release TurboFieldfareRepack --local-source scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0 --output scratch/qwen3.6-35b-a3b.gturbo
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-21/`.

#### 21.6 Verify receipts, output digests, and no-Gemma mutation

| | |
|---|---|
| Duty | `operator-review` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-21/verification.md` (**PROPOSED**) |
| Depends on | `21.5` |
| Parallel safe | `no` |
| Deliverable | Independently compare source identity, plan/policy/converter digests, every output file size/hash, text/vision binding, and before/after Gemma metadata digests. |

Do not delete partial evidence or source data. A mismatch blocks activation and preserves artifacts for diagnosis.

**Acceptance detail**
- [ ] Published receipts trace every output to official source and deterministic policy.
- [ ] Gemma directory metadata/payload digests match the preflight record.

**How to check**
```sh
Scripts/test.sh --filter LocalQwenStreamingRepackerTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-21/`.

### Phase 21 coverage plan

Every intended file has its own row. Fourteen implementation/test rows now have passing tiny-fixture evidence. The three operational evidence rows remain pending. CLI intent is tested through the production parser, then verified workflow composition is tested separately; this does not claim a successful authentic-source CLI conversion.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Sources/TurboFieldfareRepack/Command/main.swift` | `Tests/TurboFieldfareRepack/Core/Command/QwenLocalRepackCLITests.swift` | 17/17 pass on 2026-09-22. covers CLI parsing/failures. | `Scripts/test.sh --filter QwenLocalRepackCLITests` |
| `Sources/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepacker.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | 13/13 pass on 2026-09-22. covers workflow. | `Scripts/test.sh --filter LocalQwenStreamingRepackerTests` |
| `Tests/TurboFieldfareRepack/Core/Command/QwenLocalRepackCLITests.swift` | `Tests/TurboFieldfareRepack/Core/Command/QwenLocalRepackCLITests.swift` | 17/17 pass on 2026-09-22. The suite executes with a nonzero count and zero failures. | `Scripts/test.sh --filter QwenLocalRepackCLITests` |
| `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | 13/13 pass on 2026-09-22. The suite executes with a nonzero count and zero failures. | `Scripts/test.sh --filter LocalQwenStreamingRepackerTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-21/preflight.txt` (**PROPOSED**) | none yet | Planned proof: authorized preflight records environment, processes, memory, current capacity, and plan requirement. | `Scripts/test.sh --filter LocalQwenStreamingRepackerTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-21/conversion.log` (**PROPOSED**) | none yet | Planned proof: one foreground conversion log records command, exit, progress, and output sizes. | `Scripts/test.sh --filter LocalQwenStreamingRepackerTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-21/verification.md` (**PROPOSED**) | none yet | Planned proof: independent receipt/digest comparison and before/after Gemma hashes. | `Scripts/test.sh --filter LocalQwenStreamingRepackerTests` |
| `Sources/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifier.swift` | `Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift` | 7/7 pass on 2026-09-22. Pinned publisher shard hashes, safe paths, size/hash mismatch, cancellation and source-change rejection using tiny fixtures. | `Scripts/test.sh --filter LocalOfficialQwenPayloadVerifierTests` |
| `Sources/TurboFieldfareRepack/Core/Writing/TransformedTensorWriter.swift` | `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | 15/15 pass on 2026-09-22. Receipt-inclusive text/vision outputs resume identically and paired publication rolls back coherently on failure. | `Scripts/test.sh --filter QwenTransformResumeTests` |
| `Sources/TurboFieldfareRepack/Core/Verification/VerifiedInstallReceiptWriter.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | 13/13 pass on 2026-09-22. V2 provenance and final directory bindings are deterministic and accepted by runtime receipt validation; Gemma behavior preserved. | `Scripts/test.sh --filter LocalQwenStreamingRepackerTests` |
| `Sources/TurboFieldfareRepack/Core/Verification/RepackAudit.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | 13/13 pass on 2026-09-22. Exact manifest/receipt/file inventory is verified before publication. | `Scripts/test.sh --filter LocalQwenStreamingRepackerTests` |
| `Sources/TurboFieldfareRepack/Core/System/DiskSpaceChecker.swift` | `Tests/TurboFieldfareRepack/Core/Workflow/LocalQwenStreamingRepackerTests.swift` | 13/13 pass on 2026-09-22. Same-volume text plus vision capacity is summed with metadata/checkpoint/temp requirements and protected reserve. | `Scripts/test.sh --filter LocalQwenStreamingRepackerTests` |
| `Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift` | `Tests/TurboFieldfareRepack/Core/Local/LocalOfficialQwenPayloadVerifierTests.swift` | 7/7 pass on 2026-09-22. Tiny payload-verification cases execute with nonzero count and zero failures. | `Scripts/test.sh --filter LocalOfficialQwenPayloadVerifierTests` |
| `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | `Tests/TurboFieldfareRepack/Core/Writing/QwenTransformResumeTests.swift` | 15/15 pass on 2026-09-22. Existing resume proofs plus receipt/publication failure cases execute with zero failures. | `Scripts/test.sh --filter QwenTransformResumeTests` |
| `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | `Tests/TurboFieldfareRepack/Core/Writing/TransformedTensorWriterTests.swift` | 10/10 pass on 2026-09-22. Bounded checkpoint batches preserve output bytes and hash order; cancellation, sync failure and commit failure resume from the last durable prefix, with forced final commit. | `Scripts/test.sh --filter TransformedTensorWriterTests` |
| `Sources/TurboFieldfare/Infrastructure/ModelIO/VerifiedInstallReceipt.swift` | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/QwenReceiptPathBindingTests.swift` | 3/3 pass on 2026-09-22. Qwen conversion receipts bind the same physical published directory across Foundation alias changes; legacy Gemma behavior stays unchanged; wrong or missing directory is rejected. | `Scripts/test.sh --filter QwenReceiptPathBindingTests` |
| `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/QwenReceiptPathBindingTests.swift` | `Tests/TurboFieldfare/Core/Infrastructure/ModelIO/QwenReceiptPathBindingTests.swift` | 3/3 pass on 2026-09-22. The actual repacker receipt encoder and runtime receipt reader agree before/after publication, preserve strict wrong-directory refusal, and retain legacy semantics. | `Scripts/test.sh --filter QwenReceiptPathBindingTests` |


### Phase 21 evidence

<a id="phase-21-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-22 | Phase baseline and conversion-path map | 77 source/test snapshots; necessary payload, receipt, shared-volume and publication closures identified. Staged implementation started; no real payload read or conversion. | `baseline/`; `coordination/sol-production-conversion-map.md` |
| 2026-09-22 | Final converter reporting and recovery suites | 58/58 pass across five suites; prior unchanged payload/quantizer/reader 17/17 retained. Tasks 21.1–21.3 accepted, operational conversion pending. | `coordination/final-reporting-focused-results.json`; `../phase-20/execution/luna-p20-rerun-20260922T173246/p21-selector-summary.tsv` |

---


## Phase 22 - Real Qwen text behavior matches the quantized reference

**When this is done:** real Qwen text behavior matches the quantized reference

Needs: `12, 13, 15, 21`
Base commit: `pending — record immediately before the phase starts`
Disk baseline: `pending — record current free space, processes, devices, and worktrees at phase start`
Evidence: `pending — scratch/qwen3.6-35b-a3b/evidence/phase-22/`

### Current state

Tiny fixtures prove component math but not the produced full artifact. Real text qualification requires one model-related process at a time, current environment preflight, the same quantized weights on both implementations, Metal validation, and separate evidence for logits, state recovery, thinking, and tools.

### Target state

The authorized full artifact passes release load, same-quantized-weight logit/token differential, soft Stop/hard cancellation/state rebuild, thinking/tool grammar, and Metal API/shader validation. Each concern has its own recorded task and verdict; no speed claim is made.

### Preparatory reference verification on 2026-09-22

The independent Python reference, Swift parity consumer, state recovery suite, and tool suite passed bounded static review on their frozen candidates. The reference selects the explicit pinned local Transformers checkout before import and verifies executed module paths. Candidate comparison evidence records actual candidate token choices, failures, and measured finite numerical differences. No tolerance was widened.

The pinned Python environment ran 16 standard-library unit tests with exit 0 in 0.039 seconds (0.13 seconds wall time). They verify publication ownership and rollback, cancellation, and source-checkout validation without importing a model library. Source SHA-256 is `9b2463fd8935a17c049b3399f9ced52b62c554defaf3038894f6c21e4815a50d`; test SHA-256 is `a3f7361e953863e76ff6fe902c9e6d2a8f6b122fae7e77f3df08cdda8ce31986`. Both were unchanged during execution. Evidence is `execution/python-unit-20260922T161220Z/`. All five reviewed files are now promoted. `swift build --build-tests --jobs 1` passed with exit 0 in 29.27 seconds and compiled the three Swift suites without running tests. The rejected target-plus-build-tests command and the first compiler failure are preserved. Two bounded corrections split an oversized JSON expression and use asynchronous Metal completion; Main reviewed both without changing assertions, schema or numerical thresholds. Final parity test SHA-256 is `68a44e9aecd1da03fc0bbcb339b3640e8b00d3f0697a0cef36f30a2ba0ad0a6a`; final tool test SHA-256 is `b6fe8068fc1ed91c7bbc3646d864e5c400d11dc79049059deb9e3ab92ce26725`. Compile evidence is `execution/compile-only-20260922T164500Z/`. Python source/unit bytes are unchanged from their 16-test passed candidate. Nine opt-in Swift cases exist (one parity, five recovery, three tool/thinking), require `TURBO_FIELDFARE_REAL_QWEN_ARTIFACT`, and compiled but were not executed. The final receipt records `compile_status=success`, `test_execution=not_run`, `model_execution=not_run`, and `payload_access=none`. Python's 16/16 checks exercised helpers only, not model behavior. This reference targets the superseded quantized design; it does not qualify the paused BF16 requirement. Authentic-weight qualification remains pending.

### Preparatory reference decision on 2026-09-22

The pinned reference environment has no independent v2 pack reader. A staged Python reference and the planned Swift parity test will read the exact converted tensors and execute serially, with the Python process exiting before the Swift runtime creates a model. The reference streams one official CPU Float32 decoder layer at a time to fit the saved 32 GB machine. Existing Phase 3 comparison bounds remain unchanged: maximum absolute logit error must be at most `1e-5 + 1e-5 * scale`, with exact greedy tokens. A failing result will not justify changing that threshold. Preparatory paths and absence were recorded before editing in `baseline/preparatory-scope.json`; model execution still awaits conversion and fresh preflight.

### Tasks

#### 22.1 Add separate real-text integration test cases

| | |
|---|---|
| Duty | `test` |
| Touches | `Scripts/qwen36_quantized_reference.py`; `Tests/Python/test_qwen36_quantized_reference.py`; `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealTextParityTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealStateRecoveryTests.swift`; `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealToolCodecTests.swift` (all present; compile-only for Swift) |
| Depends on | `12.6, 13.6, 15.5, 21.6` |
| Parallel safe | `yes` |
| Deliverable | Three opt-in integration suites cover same-quantized-weight logits/tokens, transactional recovery, and thinking/tool structure without combining their verdicts. |

Test code reads the verified artifact path from an explicit test environment and fails closed when a required real-model run is selected without it. The test owner edits only these files.

**Acceptance detail**
- [ ] All three suites are discovered under `TurboFieldfareTestsCore`.
- [ ] Each suite reports its own executed/failed/skipped count and exact artifact identity.

**How to check**
```sh
Scripts/test.sh --filter QwenReal
```

**Evidence:** Preparatory Python helper checks: `scratch/qwen3.6-35b-a3b/evidence/phase-22/execution/python-unit-20260922T161220Z/test.log` (16/16, not model behavior). Swift compile-only: `scratch/qwen3.6-35b-a3b/evidence/phase-22/execution/compile-only-20260922T164500Z/final-receipt.txt` (nine opt-in cases compiled, none run). Task acceptance and operational evidence remain pending.

#### 22.2 Preflight the authorized real-text session

| | |
|---|---|
| Duty | `operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-22/preflight.txt` (**PROPOSED**) |
| Depends on | `12.6, 13.6, 15.5, 21.6` |
| Parallel safe | `no` |
| Deliverable | Record commit, hardware/RAM, macOS, Swift, free disk, memory pressure, artifact/receipt digests, and prohibited-process check before any model load. |

If the vision companion or any text receipt is invalid, stop. Run one CLI/reference/model process at a time.

**Acceptance detail**
- [ ] Every AGENTS.md prerequisite is recorded and passes.
- [ ] The evidence identifies the exact uncommitted candidate digest.

**How to check**
```sh
sw_vers && swift --version && system_profiler SPHardwareDataType && memory_pressure -Q && pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-22/`.

#### 22.3 Build the affected release products once

| | |
|---|---|
| Duty | `build-owner` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-22/release-build.log` (**PROPOSED**) |
| Depends on | `22.2` |
| Parallel safe | `no` |
| Deliverable | Build release CLI, repacker, service, app, and server from the exact candidate, preserving exit status and compiler/concurrency diagnostics. |

Do not launch a model. A build is not behavioral proof.

**Acceptance detail**
- [ ] Every affected product compiles in Release.
- [ ] No new compiler or concurrency diagnostic is attributable to Qwen changes.

**How to check**
```sh
swift build -c release --product TurboFieldfareCLI && swift build -c release --product TurboFieldfareRepack && swift build -c release --product TurboFieldfareDecodeService && swift build -c release --product TurboFieldfareMac && swift build -c release --product TurboFieldfareServer
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-22/`.

#### 22.4 Compare real logits and greedy tokens to the same weights

| | |
|---|---|
| Duty | `reference-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-22/text-parity.json` (**PROPOSED**) |
| Depends on | `22.3` |
| Parallel safe | `no` |
| Deliverable | Run short fixed prompts through TurboFieldfare and an independent reference consuming the identical quantized tensors; record prompt IDs, logits/tokens, absolute/relative errors, stop reason, and command exits. |

Do not compare official BF16 reference logits to quantized runtime as if it isolated runtime correctness.

**Acceptance detail**
- [ ] Every declared tolerance was fixed by P3/P4 before this run.
- [ ] Greedy tokens and measured logit errors satisfy those criteria.

**How to check**
```sh
Scripts/test.sh --filter QwenRealTextParityTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-22/`.

#### 22.5 Verify soft Stop, hard cancel, suffix, and rebuild

| | |
|---|---|
| Duty | `state-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-22/state-recovery.json` (**PROPOSED**) |
| Depends on | `22.4` |
| Parallel safe | `no` |
| Deliverable | Run separate deterministic cases for cooperative Stop commit, task cancellation rollback, matched hidden suffix removal, capacity checkpoint rebuild, and multi-turn continuation. |

After each interruption compare the next turn and internal state digest with a clean reference conversation.

**Acceptance detail**
- [ ] Soft Stop preserves accepted output/state.
- [ ] Cancel, error, suffix trim, and rebuild produce the expected clean-state digest.

**How to check**
```sh
Scripts/test.sh --filter QwenRealStateRecoveryTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-22/`.

#### 22.6 Verify real thinking and tool output behavior

| | |
|---|---|
| Duty | `tool-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-22/thinking-tools.json` (**PROPOSED**) |
| Depends on | `22.4` |
| Parallel safe | `no` |
| Deliverable | Run thinking on/off and allowed/malformed tool prompts; capture raw tokens, structured events, proposed host calls, and stop reason. |

No external tool action is dispatched in this phase. Malformed output must yield zero executable proposals.

**Acceptance detail**
- [ ] Thinking boundaries match the Qwen codec in both modes.
- [ ] Allowed complete calls parse; malformed/incomplete calls remain non-executable.

**How to check**
```sh
Scripts/test.sh --filter QwenRealToolCodecTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-22/`.

#### 22.7 Run Metal API and shader validation on text kernels

| | |
|---|---|
| Duty | `validation-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-22/metal-validation.log` (**PROPOSED**) |
| Depends on | `22.3` |
| Parallel safe | `no` |
| Deliverable | Read `man MetalValidation` first. Set validation variables before `MTLDevice` creation and capture stderr; run a bounded text case that reaches full attention, DeltaNet, and MoE. |

Validation overhead invalidates performance timing. Record actual supported hardware; no GPU skip counts as a pass.

**Acceptance detail**
- [ ] All three Qwen text kernel families execute under API/shader validation.
- [ ] The log contains no validation error or out-of-bounds report.

**How to check**
```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 swift run -c release TurboFieldfareCLI --model scratch/qwen3.6-35b-a3b.gturbo --prompt "Return OK." --max-new 8 --temperature 0 --seed 20260915
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-22/`.

#### 22.8 Obtain independent text evidence review

| | |
|---|---|
| Duty | `reviewer` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-22/review.txt` (**PROPOSED**) |
| Depends on | `22.1, 22.3, 22.4, 22.5, 22.6, 22.7` |
| Parallel safe | `no` |
| Deliverable | A fresh read-only reviewer checks candidate identity, source/quantized lineage, commands/exits, nonzero cases, tolerance provenance, state semantics, tool non-dispatch, and validation output. |

Exit zero alone is not approval; the reviewer must return PASS for the exact candidate or list a blocking correction.

**Acceptance detail**
- [ ] Review identifies the exact candidate digest.
- [ ] Verdict is PASS with evidence or FAIL/BLOCKED with a concrete missing proof.

**How to check**
```sh
pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-22/`.

### Phase 22 coverage plan

The five Phase 22 preparation files exist at the final receipt's hashes. The three opt-in Swift suites compiled but none ran; the 16/16 Python checks cover helpers only. Operational proof paths below remain proposed and absent. Coverage is still pending for quantized same-pack/model behavior and does not qualify the BF16 requirement.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Scripts/qwen36_quantized_reference.py` | `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealTextParityTests.swift` (exists; compiled, not run) | Still pending: independently validate and decode the exact v2 tensors, stream pinned official CPU layers with bounded memory, and emit identity-bound full-logit evidence. The Python 16/16 helper checks do not prove this; no official BF16 comparison substitutes for same-weight proof. | `Scripts/test.sh --filter QwenRealTextParityTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealTextParityTests.swift` | self (exists; compiled, not run) | The opt-in parity case must execute with the authorized artifact, nonzero count, and no unexpected skip. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealStateRecoveryTests.swift` | self (exists; compiled, not run) | The five opt-in recovery cases must execute with the authorized artifact, nonzero count, and no unexpected skip. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenRealToolCodecTests.swift` | self (exists; compiled, not run) | The three opt-in tool/thinking cases must execute with the authorized artifact, nonzero count, and no unexpected skip. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/preflight.txt` (**PROPOSED**) | none yet | Planned proof: recorded current environment and candidate identity. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/release-build.log` (**PROPOSED**) | none yet | Planned proof: preserved Release build exits/diagnostics. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/text-parity.json` (**PROPOSED**) | none yet | Planned proof: same-weight reference comparison with fixed tolerances. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/state-recovery.json` (**PROPOSED**) | none yet | Planned proof: clean-versus-recovered state cases. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/thinking-tools.json` (**PROPOSED**) | none yet | Planned proof: raw/structured output and zero-dispatch malformed cases. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/metal-validation.log` (**PROPOSED**) | none yet | Planned proof: API/shader validation log on supported hardware. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-22/review.txt` (**PROPOSED**) | none yet | Planned proof: independent exact-candidate verdict. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-22/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `Tests/Python/test_qwen36_quantized_reference.py` | self | 16/16 helper checks passed for the unchanged promoted candidate; not model/parity behavior. Preserve existing output and remove only owned publication after failure/cancellation; reject changed supporting Python modules against pinned tree before any model library import. | `python3 -m unittest discover -s Tests/Python -p test_qwen36_quantized_reference.py` |

### Phase 22 evidence

<a id="phase-22-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-22 | Python standard-library helper checks | 16/16 pass; no model library import or model-behavior proof. | `scratch/qwen3.6-35b-a3b/evidence/phase-22/execution/python-unit-20260922T161220Z/test.log` |
| 2026-09-22 | Swift test compilation only | `swift build --build-tests --jobs 1` succeeded; nine opt-in cases compiled, zero executed; no model or payload access. | `scratch/qwen3.6-35b-a3b/evidence/phase-22/execution/compile-only-20260922T164500Z/final-receipt.txt` |
| - | Operational preflight, Release products, same-pack parity/recovery/tools, Metal validation, and final review | pending; no qualification or task acceptance claimed. | `scratch/qwen3.6-35b-a3b/evidence/phase-22/` |

---

## Phase 23 - Real Qwen screenshots preserve verified workflow truth

**When this is done:** real Qwen screenshots preserve verified workflow truth

Needs: `16, 20, 22`
Base commit: `pending — record immediately before the phase starts`
Disk baseline: `pending — record current free space, processes, devices, and worktrees at phase start`
Evidence: `pending — scratch/qwen3.6-35b-a3b/evidence/phase-23/`

### Current state

Tiny vision fixtures do not establish real screenshot quality, app attachment behavior, or VisionCapture task success. Image support also requires a valid adjacent companion and separately authorized target/tool permissions. Host verified results must remain the workflow ground truth.

### Target state

Authorized still-image runs prove preprocessing/tower/M-RoPE parity, invalid-pack failure, image checkpoint recovery, app attachment behavior, and a scoped VisionCapture QA workflow whose target and permissions are recorded before launch. No video input is accepted.

### Parallel preparation on 2026-09-22

The owner requested more parallel work. Sol wrote the independent local vision reference and minimal app/state changes; Luna wrote the tests. The reference consumes converted quantized packs and the frozen P22 helper, not official model shards. All paths were registered before edits. Reviewed candidates were promoted for compilation and small synthetic tests only. Formal phase dependencies, conversion approval and the workflow target still gate authentic execution.

### Tasks

#### 23.1 Add separate real-image integration test cases

| | |
|---|---|
| Duty | `test` |
| Touches | `Scripts/qwen36_quantized_vision_reference.py` (**PROPOSED**); `Tests/Python/test_qwen36_quantized_vision_reference.py` (**PROPOSED**); `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenRealVisionIntegrationTests.swift` (**PROPOSED**); `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionFailClosedIntegrationTests.swift` (**PROPOSED**); `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenRealImageStateTests.swift` (**PROPOSED**) |
| Depends on | `16.8, 20.9, 22.8` |
| Parallel safe | `yes` |
| Deliverable | Three opt-in suites separately cover real feature/position parity, invalid companion/video failures, and image-state recovery with the verified text/vision artifacts. |

Test code reads explicit artifact and corpus paths from the authorized environment. The test owner edits only these files and records unexpected skips as failures of evidence.

**Acceptance detail**
- [ ] All three suites are discovered under `TurboFieldfareTestsCore`.
- [ ] Each suite reports its own nonzero executed count and exact text/vision/image identities.

**How to check**
```sh
Scripts/test.sh --filter QwenRealVision
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-23/`.

#### 23.2 Authorize and record target plus tool permissions

| | |
|---|---|
| Duty | `operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-23/authorization.md` (**PROPOSED**) |
| Depends on | `23.1` |
| Parallel safe | `no` |
| Deliverable | Record the exact app bundle/simulator target, permitted VisionCapture tools/actions, screenshot corpus digests, and prohibited actions before any workflow run. |

Do not infer permission from model capability. If target identity or permission is ambiguous, stop.

**Acceptance detail**
- [ ] The target and allowed action set are explicit and immutable for the run.
- [ ] No model or VisionCapture process starts before this record exists.

**How to check**
```sh
Scripts/test.sh --filter QwenVision
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-23/`.

#### 23.3 Compare real image features and positions

| | |
|---|---|
| Duty | `vision-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-23/vision-parity.json` (**PROPOSED**) |
| Depends on | `23.2` |
| Parallel safe | `no` |
| Deliverable | Run fixed screenshots through TurboFieldfare and the pinned reference using the same processor/profile; compare grid, pad rows, merged features, M-RoPE positions, decode delta, and declared tolerances. |

Run one model/reference process at a time and record every image digest.

**Acceptance detail**
- [ ] Each image has identical grid and pad-row counts.
- [ ] Feature/position errors meet fixed tolerances and axes are not interchangeable.

**How to check**
```sh
Scripts/test.sh --filter QwenRealVisionIntegrationTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-23/`.

#### 23.4 Verify invalid companion and unsupported video failures

| | |
|---|---|
| Duty | `vision-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-23/fail-closed.json` (**PROPOSED**) |
| Depends on | `23.2` |
| Parallel safe | `no` |
| Deliverable | Exercise absent pack, wrong text digest, wrong revision, wrong processor hash, corrupt receipt/region, and video request without mutating the verified installed pack. |

Use isolated fixture paths; never rename or damage the canonical companion to create a negative case.

**Acceptance detail**
- [ ] Every invalid image capability fails before text-only generation.
- [ ] Video is rejected explicitly at CLI, server, and app boundaries.

**How to check**
```sh
Scripts/test.sh --filter QwenVisionFailClosedIntegrationTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-23/`.

#### 23.5 Verify image checkpoint and app attachment lifecycle

| | |
|---|---|
| Duty | `vision-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-23/image-state.json` (**PROPOSED**) |
| Depends on | `23.3, 23.4` |
| Parallel safe | `no` |
| Deliverable | Run multi-turn images across soft Stop, hard cancel, checkpoint rebuild, reload, and a changed-image digest; compare against clean state and inspect retained attachment lifetime. |

No stale feature buffer or M-RoPE delta may cross lineage changes.

**Acceptance detail**
- [ ] Clean and recovered image turns produce matching state/output.
- [ ] Changed or released images invalidate retained features without leaking files.

**How to check**
```sh
Scripts/test.sh --filter QwenRealImageStateTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-23/`.

#### 23.6 Run the scoped VisionCapture QA workflow once

| | |
|---|---|
| Duty | `qa-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-23/visioncapture-qa.json` (**PROPOSED**) |
| Depends on | `23.3, 23.5` |
| Parallel safe | `no` |
| Deliverable | Use the authorized target and actions for one bounded screenshot-plus-tool workflow; capture model proposals, host requests/results, returned identity, verified verdict, and final user-visible outcome. |

Accepted requests are not proof. Only VisionCapture’s host-owned verified evidence establishes action/outcome; uncertain actions are not replayed.

**Acceptance detail**
- [ ] Every dispatched action was allowed, identity-bound, and host verified.
- [ ] The final claim matches recorded host evidence and names any uncertainty.

**How to check**
```sh
Scripts/test.sh --filter VisionCaptureReturnedIdentityTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-23/`.

#### 23.7 Obtain independent vision and workflow review

| | |
|---|---|
| Duty | `reviewer` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-23/review.txt` (**PROPOSED**) |
| Depends on | `23.1, 23.3, 23.4, 23.5, 23.6` |
| Parallel safe | `no` |
| Deliverable | A fresh reviewer checks image/reference comparability, invalid-pack isolation, state lineage, target authorization, host evidence, non-replay, and candidate identity. |

The reviewer returns PASS only when all required real GPU/workflow cases executed without unexpected skip.

**Acceptance detail**
- [ ] Review names the exact candidate, model, companion, processor, and image digests.
- [ ] Verdict is PASS or a concrete FAIL/BLOCKED.

**How to check**
```sh
pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-23/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-23/`.

### Phase 23 coverage plan

All 17 intended paths are recorded below. Existing promoted test files are identified explicitly. Authentic integration evidence remains pending even when source compiles or synthetic tests pass.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenRealVisionIntegrationTests.swift` | self | The proposed integration suite must execute with authorized artifacts/images, nonzero count, and no unexpected skip. | `Scripts/test.sh --filter QwenRealVisionIntegrationTests` after artifact/corpus authorization. |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionFailClosedIntegrationTests.swift` | self | The proposed integration suite must execute isolated negative artifacts with a nonzero count. | `Scripts/test.sh --filter QwenVisionFailClosedIntegrationTests` after artifact/corpus authorization. |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenRealImageStateTests.swift` | self | The proposed integration suite must execute image-state cases with a nonzero count and no unexpected skip. | `Scripts/test.sh --filter QwenRealImageStateTests` after artifact/corpus authorization. |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/authorization.md` (**PROPOSED**) | none yet | Planned proof: immutable target/tool authorization and corpus digests. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-23/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/vision-parity.json` (**PROPOSED**) | none yet | Planned proof: real grid/feature/position comparison. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-23/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/fail-closed.json` (**PROPOSED**) | none yet | Planned proof: isolated invalid-pack and video refusals. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-23/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/image-state.json` (**PROPOSED**) | none yet | Planned proof: clean-versus-recovered image state. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-23/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/visioncapture-qa.json` (**PROPOSED**) | none yet | Planned proof: authorized host request/result/verdict trail. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-23/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-23/review.txt` (**PROPOSED**) | none yet | Planned proof: independent exact-candidate and exact-artifact verdict. | `pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-23/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."` |
| `Scripts/qwen36_quantized_vision_reference.py` | `Tests/Python/test_qwen36_quantized_vision_reference.py` | Independent pinned processor, companion tensor reader, feature and position evidence with frozen helper provenance and owned atomic output. | `python3 -B Tests/Python/test_qwen36_quantized_vision_reference.py` (16/16 synthetic tests passed; no numerical qualification). |
| `Tests/Python/test_qwen36_quantized_vision_reference.py` | self | Reject mutated identities, unsafe paths, invalid requests and partial publication using bounded synthetic fixtures. | `python3 -B Tests/Python/test_qwen36_quantized_vision_reference.py` (16/16 synthetic tests passed; no numerical qualification). |

| `Sources/TurboFieldfare/Runtime/Vision/Preprocessing/VisionImageSource.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppImageAttachmentStoreTests.swift` | Dedicated unsupported-video error is reported for app file imports. | `Scripts/test.sh --filter AppImageAttachmentStoreTests` |
| `Sources/TurboFieldfareApp/Core/Inference/AppImageAttachment.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppImageAttachmentStoreTests.swift` | Movie file imports fail before staging bytes while still-image and no-follow behavior remains. | `Scripts/test.sh --filter AppImageAttachmentStoreTests` |
| `Tests/TurboFieldfareApp/Core/Inference/AppImageAttachmentStoreTests.swift` | `Tests/TurboFieldfareApp/Core/Inference/AppImageAttachmentStoreTests.swift` | Focused video rejection leaves no staged output and source bytes unchanged. | `Scripts/test.sh --filter AppImageAttachmentStoreTests` |
| `Tests/TurboFieldfareApp/Core/State/AppImageAddConcurrencyTests.swift` | `Tests/TurboFieldfareApp/Core/State/AppImageAddConcurrencyTests.swift` | AppModel.addImages surfaces the video error, resets pending-add state and retains no rejected attachment. | `Scripts/test.sh --filter AppImageAddConcurrencyTests` |
| `Sources/TurboFieldfare/Runtime/Qwen/QwenConversationState.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` | Internal occurrence diagnostics report immutable retained-owner image and processor digests without changing generation behavior. | `Scripts/test.sh --filter QwenVisionConversationStateTests` |
| `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` | `Tests/TurboFieldfare/Core/Runtime/Qwen/Vision/QwenVisionConversationStateTests.swift` | Tiny production-path owner fixtures prove diagnostic digests follow retained occurrences and rollback. | `Scripts/test.sh --filter QwenVisionConversationStateTests` |

### Parallel preparation and numerical limits on 2026-09-22

The independent reference and all three Swift integration suites passed static correction review and were promoted in `execution/main-reviewed-preparation-promotion-20260922T181224Z/`. The reference passed 16/16 standard-library tests in `execution/main-reference-stdlib-20260922T181116Z/` with unchanged hashes. These cover synthetic request, descriptor, tensor-contract, BF16 and publication boundaries. The full Swift test build passed in 25.43 seconds after four files received reviewed mechanical compiler corrections. The tiny retained-image state suite passed 19/19 tests in 1.653 seconds. Exact logs, final hashes and Main review are in `execution/luna-p23-compile-tiny-20260922T181421Z/`. No authentic image, Qwen artifact or host workflow has run; Task 23.2 remains pending.

Before seeing authentic results, Main accepted the unchanged numerical constants with distinct comparison rules: merger features use `abs(candidate-reference) <= 1e-5 + 1e-5*abs(reference)` for every finite element; full logits use `maxAbsDifference <= 1e-5 + 1e-5*max(maxAbsCandidate,maxAbsReference)` plus exact first-argmax and generated-token agreement. Exact image identity, oriented/target geometry, BF16 patch shape/byte count/SHA-256, tokens, pad rows and complete three-axis positions/delta are prerequisite checks. A patch mismatch blocks downstream numerical admission for that case. These are preregistered limits, not evidence that an authentic run passes. No result-based widening or image substitution is permitted. The decision and review are `coordination/main-vision-comparison-preregistration.json` and `coordination/preregistered-vision-comparison-review.md`.

### App video import validation on 2026-09-22

The shared file import boundary now rejects declared movie/video URLs with `Video input is not supported: <name>` before staging bytes, while preserving existing no-follow and size checks. Both focused suites passed 4/4, and related attachment/lifetime/capacity regressions bring the total to 36/36. Source and package inputs were unchanged during execution. Evidence: `execution/main-app-video-promotion-20260922T180123Z/` and `execution/main-app-video-tests-20260922T180234Z/`. This proves the app import boundary. It does not claim arbitrary renamed-video detection, real Qwen image inference, CLI/server video behavior, or completion of 23.4.

### Phase 23 evidence

<a id="phase-23-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-22 | App video import and attachment regressions | 36/36 pass; exact input hashes unchanged | `execution/main-app-video-tests-20260922T180234Z/` |
| 2026-09-22 | Corrected reference synthetic units | 16/16 pass, 0.024 seconds, exit 0 | `execution/main-reference-stdlib-20260922T181116Z/` |
| 2026-09-22 | Reviewed preparation promotion | Seven files promoted for compilation/tiny tests; no authentic run | `execution/main-reviewed-preparation-promotion-20260922T181224Z/` |
| 2026-09-22 | Swift compilation and tiny state test | Build exit 0 in 25.43 seconds; 19/19 tiny tests pass in 1.653 seconds; no authentic tests executed | `execution/luna-p23-compile-tiny-20260922T181421Z/` |

---

## Phase 24 - A controlled Gemma-Qwen comparison is recorded

**When this is done:** a controlled Gemma-Qwen comparison is recorded

Needs: `22, 23`
Base commit: `pending — record immediately before the phase starts`
Disk baseline: `pending — record current free space, processes, devices, and worktrees at phase start`
Evidence: `pending — scratch/qwen3.6-35b-a3b/evidence/phase-24/`

### Current state

README links `docs/COMMUNITY_BENCHMARKS.md`, but the current checkout lacks that guide and its prompt files. The archived architectural report at `/Users/dev-machine/Documents/Idea Home/turboCharge/archive/2026-09-15-document-cleanup/repository/plan/report.md:813-835` identifies a complete prior copy at commit `66508fbe8117b0a307cf521c6f7d01b923c3c0b5`. No Qwen baseline or speed threshold exists.

### Preparation completed before dependencies — 2026-09-22

The historical guide and all three prompts are now preserved under `scratch/qwen3.6-35b-a3b/evidence/phase-24/staging/docs/`, with exact Git blob provenance in `baseline/parallel-benchmark-preparation.json`. The staged guide adds a dated applicability note while preserving all 17,414 original bytes as an unchanged suffix. The three prompt files remain byte-identical. No live `docs/` path was restored and no Phase 24 task is accepted yet.

The staged `protocol.md` fixes twelve serial processes: six discarded warmups followed by six measured observations, alternating family order across cases. It records the current CLI command, family-native thinking/cache/prefill differences, Gemma `endOfTurn` versus Qwen `eos`, Qwen generated-token versus visible-answer accounting, ordinary wall/peak-resident-memory measurement, invalid-run rules, and a per-case quality rubric before any result. Model/candidate identities and the authorized image/tool target still need binding after Phases 22 and 23. This preparation does not authorize conversion or benchmark execution.

GPT-6 Luna passed both bounded reviews: the original schedule, prompt fidelity, quality and reporting checks, then the final CLI integration, token accounting, resource fields and preserved historical bytes. Main reviewed and accepted the final preparation at protocol SHA-256 `b49651ef29ba5a9cae0a73825e434ce4cacbb558c1253dcfc6483e0f533e8fc9` and guide SHA-256 `d3d4bbad946049bb69d0807885d34ea965dfec2dbb2855f847e4139e1850bce6`. The input record is `coordination/cli-integrated-preparation.json`; the independent review is `coordination/benchmark-protocol-review.md`, SHA-256 `0413bbe56a76c6a4d63a39fe3c2c2fbd1c8d9760c5b0198ec89aa0cb86fa0f2f`. This accepts preparation only. No build, unit test, model, reference, or benchmark was run for it.

### Target state

A reviewed change restores or explicitly versions the historical guide/prompts, records exact deviations for Qwen tokenization/thinking/defaults, pre-registers common cases and quality/performance fields, then runs paired same-hardware Gemma/Qwen cases in alternating order with one model process at a time. Results make no universal speed claim.

### Tasks

#### 24.1 Restore and review the historical benchmark guide

| | |
|---|---|
| Duty | `docs-owner` |
| Touches | `docs/COMMUNITY_BENCHMARKS.md` (**PROPOSED**) |
| Depends on | `22.8, 23.7` |
| Parallel safe | `yes` |
| Deliverable | Restore the exact file from commit `66508fbe8117b0a307cf521c6f7d01b923c3c0b5`, review it against current commands, and version any necessary correction visibly. |

Do not copy commands from memory or add experimental profiling controls outside the community protocol.

**Acceptance detail**
- [ ] The restored guide cites its source revision and current applicability.
- [ ] Every changed instruction is explicit rather than silently rewritten.

**How to check**
```sh
git show 66508fbe8117b0a307cf521c6f7d01b923c3c0b5:docs/COMMUNITY_BENCHMARKS.md
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-24/`.

#### 24.2 Restore the three frozen benchmark prompts

| | |
|---|---|
| Duty | `docs-owner` |
| Touches | `docs/benchmark-prompts/real-generation-v1/short-explanation.json` (**PROPOSED**); `docs/benchmark-prompts/real-generation-v1/medium-review.json` (**PROPOSED**); `docs/benchmark-prompts/real-generation-v1/long-synthesis.json` (**PROPOSED**) |
| Depends on | `24.1` |
| Parallel safe | `no` |
| Deliverable | Restore exact prompt bytes from the same historical commit and record their SHA-256 digests in the guide/evidence. |

Do not edit prompts to favor either tokenizer or thinking mode.

**Acceptance detail**
- [ ] All three files match the historical blobs and recorded digests.
- [ ] Both models receive the same prompt bytes per common case.

**How to check**
```sh
git diff --no-index <(git show 66508fbe8117b0a307cf521c6f7d01b923c3c0b5:docs/benchmark-prompts/real-generation-v1/short-explanation.json) docs/benchmark-prompts/real-generation-v1/short-explanation.json
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-24/`.

#### 24.3 Pre-register comparison cases and deviations

| | |
|---|---|
| Duty | `comparison-owner` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-24/protocol.md` (**PROPOSED**) |
| Depends on | `24.2` |
| Parallel safe | `yes` |
| Deliverable | Record commit/product/model/prompt digests, shared settings, seeds, token limits, stop policy, warmups/repetitions, output validity rules, quality rubric, latency/throughput/memory fields, and order alternation. |

Call out different token counts, Qwen thinking, publisher defaults versus common settings, and lack of a Qwen speed threshold.

**Acceptance detail**
- [ ] Decision rules and invalid-run handling are fixed before results exist.
- [ ] No row is called directly comparable when settings/token counts violate the guide.

**How to check**
```sh
Scripts/test.sh --filter QwenTextRunnerTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-24/`.

#### 24.4 Record same-hardware environment and baselines

| | |
|---|---|
| Duty | `comparison-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-24/environment.txt` (**PROPOSED**) |
| Depends on | `24.3` |
| Parallel safe | `no` |
| Deliverable | Record commit, hardware/RAM, macOS, Swift, power/Low Power Mode, memory pressure, model manifests, prompt digests, and process preflight immediately before runs. |

Use the repository community guide. Do not add profiler/thermal controls the guide does not authorize.

**Acceptance detail**
- [ ] Both models use the same recorded machine/toolchain/workload conditions.
- [ ] Preflight satisfies AGENTS.md before each model process.

**How to check**
```sh
sw_vers && swift --version && system_profiler SPHardwareDataType SPPowerDataType && memory_pressure -Q
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-24/`.

#### 24.5 Run paired community text cases in alternating order

| | |
|---|---|
| Duty | `comparison-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-24/text-results.json` (**PROPOSED**) |
| Depends on | `24.4` |
| Parallel safe | `no` |
| Deliverable | Run the restored short/medium/long cases with one discarded warmup and the guide’s measured repetitions, alternating Gemma→Qwen then Qwen→Gemma. |

Run one CLI/model process at a time. Record the complete timing footer/error, prompt/generated token counts, stop reason, settings, validity, and deviations.

**Acceptance detail**
- [ ] Every accepted row follows the pre-registered protocol or names its deviation.
- [ ] No timing from Metal validation or an invalid output enters comparison summaries.

**How to check**
```sh
swift run -c release TurboFieldfareCLI --model scratch/gemma4.gturbo --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-24/`.

#### 24.6 Run paired VisionCapture quality cases

| | |
|---|---|
| Duty | `comparison-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-24/qa-results.json` (**PROPOSED**) |
| Depends on | `24.4, 24.5` |
| Parallel safe | `no` |
| Deliverable | Run the pre-registered image-plus-tool cases for each model using fresh conversations, the same authorized target, and alternating order. |

Score only host-recorded completion, safety, correction, and uncertainty evidence; publisher benchmark claims do not override task evidence.

**Acceptance detail**
- [ ] Each pair uses identical target facts and allowed actions.
- [ ] Scores trace to host evidence and preserve model-specific token/settings deviations.

**How to check**
```sh
Scripts/test.sh --filter VisionCaptureReturnedIdentityTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-24/`.

#### 24.7 Publish comparison with uncertainty and no speed promise

| | |
|---|---|
| Duty | `comparison-review` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-24/comparison.md` (**PROPOSED**) |
| Depends on | `24.5, 24.6` |
| Parallel safe | `no` |
| Deliverable | Summarize sample counts, medians/percentiles requested by the guide, memory, output validity, QA rubric, deviations, and noisy/inconclusive differences. |

Independent review checks source rows and arithmetic. Do not establish a universal throughput target or promote Qwen from one faster run.

**Acceptance detail**
- [ ] Every summary value traces to a valid recorded run.
- [ ] Conclusion distinguishes measured difference, noise, quality, and unsupported claims.

**How to check**
```sh
Scripts/test.sh --filter QwenTextRunnerTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-24/`.

### Phase 24 coverage plan

Every intended file has its own row. New test paths appear as `PROPOSED` in the proof column until created; the tracker correctly says `none yet` rather than pretending the file exists.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `docs/COMMUNITY_BENCHMARKS.md` (**PROPOSED**) | none yet | Planned proof: exact historical-source diff plus reviewed current-command corrections. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `docs/benchmark-prompts/real-generation-v1/short-explanation.json` (**PROPOSED**) | none yet | Planned proof: exact historical blob comparison and SHA-256 record. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `docs/benchmark-prompts/real-generation-v1/medium-review.json` (**PROPOSED**) | none yet | Planned proof: exact historical blob comparison and SHA-256 record. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `docs/benchmark-prompts/real-generation-v1/long-synthesis.json` (**PROPOSED**) | none yet | Planned proof: exact historical blob comparison and SHA-256 record. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/protocol.md` (**PROPOSED**) | none yet | Planned proof: pre-registered protocol and deviations before results. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/environment.txt` (**PROPOSED**) | none yet | Planned proof: same-hardware/toolchain/process environment. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/text-results.json` (**PROPOSED**) | none yet | Planned proof: complete valid text-run records. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/qa-results.json` (**PROPOSED**) | none yet | Planned proof: paired host-evidence QA records. | `Scripts/test.sh --filter QwenTextRunnerTests` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-24/comparison.md` (**PROPOSED**) | none yet | Planned proof: reviewed traceable summary with uncertainty. | `Scripts/test.sh --filter QwenTextRunnerTests` |

### Phase 24 evidence

<a id="phase-24-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-09-22 | Local Git blob preservation, current CLI source mapping and bounded GPT-6 Luna protocol review | Preparation only; frozen prompts and twelve-run design retained. No live restoration, phase acceptance, or benchmark execution. | `baseline/parallel-benchmark-preparation.json`; `coordination/current-cli-benchmark-contract.md`; `coordination/cli-integrated-preparation.json`; `coordination/benchmark-protocol-review.md` |
| - | Authentic paired comparison | pending Phases 22 and 23 and final artifact/target binding | `scratch/qwen3.6-35b-a3b/evidence/phase-24/` |

---

## Phase 25 - Opt-in Qwen can roll back to Gemma

**When this is done:** opt-in Qwen can roll back to Gemma

Needs: `18, 19, 20, 22, 23, 24`
Base commit: `pending — record immediately before the phase starts`
Disk baseline: `pending — record current free space, processes, devices, and worktrees at phase start`
Evidence: `pending — scratch/qwen3.6-35b-a3b/evidence/phase-25/`

### Current state

Unit, real-model, vision, and comparison evidence must converge on one exact candidate before an opt-in catalog entry can be enabled. `script/build_and_run.sh:1-154` stages, installs, stops named app/service processes, and launches the app, so it requires separate install/process authorization and must not be run during planning.

### Target state

The exact candidate receives final independent test/review, an authorized staged/installed app launch proves opt-in selection and one-loaded-model lifecycle, CLI/server cross-surface identity agrees, and a one-action Gemma rollback is exercised. Only then may the catalog entry be enabled. Cleanup removes only dispensable artifacts created by this work item and preserves all evidence, weights, models, receipts, fixtures, caches, devices, and unrelated worktrees.

### Tasks

#### 25.1 Freeze exact candidate and evidence index

| | |
|---|---|
| Duty | `release-owner` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-25/candidate.md` (**PROPOSED**) |
| Depends on | `18.5, 19.5, 20.9, 22.8, 23.7, 24.7` |
| Parallel safe | `no` |
| Deliverable | Record HEAD plus digest of all staged/unstaged/untracked inputs, product digests, model/vision receipts, and links to every gate result. |

Any subsequent code/test/config change invalidates review and requires fresh affected checks.

**Acceptance detail**
- [ ] One identity names the complete candidate under review.
- [ ] Evidence from an older candidate is marked stale and excluded.

**How to check**
```sh
git status --short --untracked-files=all && git diff --stat && git diff --cached --stat
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-25/`.

#### 25.2 Run final independent behavioral verification

| | |
|---|---|
| Duty | `test-reviewer` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-25/test-review.txt` (**PROPOSED**) |
| Depends on | `25.1` |
| Parallel safe | `no` |
| Deliverable | A fresh mac-test process reads original requirements first, checks acceptance mapping and test evidence, and requests targeted missing cases without changing production scope. |

It must confirm nonzero executed counts, no unexpected skips, Release products, real GPU checks, and exact candidate identity.

**Acceptance detail**
- [ ] Verdict covers every phase acceptance family and the exact candidate.
- [ ] Any missing or stale evidence returns CHALLENGE/BLOCKED, not PASS.

**How to check**
```sh
pi -p --no-session --no-extensions --tools read,bash,edit,write,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-25/test-packet.md "ROLE: mac-test. Complete the testing assignment. Do not delegate."
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-25/`.

#### 25.3 Run final independent architecture review

| | |
|---|---|
| Duty | `reviewer` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-25/architecture-review.txt` (**PROPOSED**) |
| Depends on | `25.1, 25.2` |
| Parallel safe | `no` |
| Deliverable | A fresh read-only mac-review process inspects the complete diff, untracked files, tests, format/identity, quantization, concurrency/state, Metal lifetime/layout, product lifecycle, and evidence. |

PASS applies only to the frozen candidate; resolve blockers and rerun invalidated checks before requesting a fresh review.

**Acceptance detail**
- [ ] Review returns PASS for the exact candidate or actionable FAIL/BLOCKED.
- [ ] No preference-only comment is treated as a blocking defect.

**How to check**
```sh
pi -p --no-session --no-extensions --tools read,grep,find,ls @scratch/qwen3.6-35b-a3b/evidence/phase-25/review-packet.md "ROLE: mac-review. Review the candidate and supplied evidence. Do not delegate or edit."
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-25/`.

#### 25.4 Stage and install the authorized app candidate

| | |
|---|---|
| Duty | `install-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-25/install.log` (**PROPOSED**) |
| Depends on | `25.2, 25.3` |
| Parallel safe | `no` |
| Deliverable | Only with explicit install/process authorization and successful process preflight, run the supported `script/build_and_run.sh --verify` once and record its full output/exit and installed product digests. |

The script stops named app/service processes; never invoke it without permission. Do not run a second app/model process.

**Acceptance detail**
- [ ] The staged and installed bundle pass the script’s code-sign/digest checks.
- [ ] One installed app process starts and no second model process is created.

**How to check**
```sh
script/build_and_run.sh --verify
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-25/`.

#### 25.5 Verify opt-in picker and cross-surface identity

| | |
|---|---|
| Duty | `release-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-25/opt-in-launch.json` (**PROPOSED**) |
| Depends on | `25.4` |
| Parallel safe | `no` |
| Deliverable | Launch from Gemma default, select verified Qwen, confirm unload/load/epoch order, run one bounded request, then compare app, CLI, and loopback server reported descriptor fields. |

Run only one app/CLI/server/model process at a time. Qwen remains opt-in until this task and rollback pass.

**Acceptance detail**
- [ ] App, CLI, and server agree on model ID, revision, family, format, and vision status.
- [ ] Selection never loads both models or changes Gemma files.

**How to check**
```sh
Scripts/test.sh --filter AppModelSelectionTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-25/`.

#### 25.6 Exercise one-action Gemma rollback

| | |
|---|---|
| Duty | `release-operator` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-25/rollback.json` (**PROPOSED**) |
| Depends on | `25.5` |
| Parallel safe | `no` |
| Deliverable | From loaded Qwen, select Gemma, verify Qwen unload before Gemma load, new epoch, successful bounded Gemma request, and unchanged Gemma artifact digest. |

If Qwen load or workflow gates fail, leave Qwen disabled/experimental and preserve Gemma as default.

**Acceptance detail**
- [ ] Rollback needs one visible selection action and reaches verified Gemma identity.
- [ ] Gemma generation and v1 compatibility evidence remain passing for the candidate.

**How to check**
```sh
Scripts/test.sh --filter AppModelSelectionTests
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-25/`.

#### 25.7 Clean up only dispensable artifacts this work item created

| | |
|---|---|
| Duty | `cleanup-owner` |
| Touches | `scratch/qwen3.6-35b-a3b/evidence/phase-25/cleanup.txt` (**PROPOSED**) |
| Depends on | `25.6` |
| Parallel safe | `no` |
| Deliverable | Measure before/after, then remove only owned temporary build output, transient logs duplicated into evidence, generated simulator devices/device sets, and clean disposable worktrees created by this work item. |

Never remove official weights, Gemma/Qwen artifacts, receipts, evidence, source, fixtures, caches, pre-existing devices, dirty/unrelated worktrees, or anything under Project-files. Never kill an app/process not started by the authorized P25 run.

**Acceptance detail**
- [ ] Cleanup record names each removed path and bytes freed.
- [ ] Every protected artifact and unrelated resource remains present.

**How to check**
```sh
df -h / && git worktree list && git status --short --untracked-files=all
```

**Evidence:** Pending in `scratch/qwen3.6-35b-a3b/evidence/phase-25/`.

### Phase 25 coverage plan

Every intended file has its own row. New test paths appear as `PROPOSED` in the proof column until created; the tracker correctly says `none yet` rather than pretending the file exists.

| Code this phase changes | Test file that must cover it | What the test or evidence must prove | Command |
|---|---|---|---|
| `none - this phase changes no production or test code` | none yet | Planned proof: frozen candidate, independent test/review, authorized install, opt-in, rollback, and cleanup evidence. | `df -h / && git worktree list && git status --short --untracked-files=all` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/candidate.md` (**PROPOSED**) | none yet | Planned proof: exact candidate and evidence index. | `df -h / && git worktree list && git status --short --untracked-files=all` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/test-review.txt` (**PROPOSED**) | none yet | Planned proof: independent behavioral verification verdict. | `df -h / && git worktree list && git status --short --untracked-files=all` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/architecture-review.txt` (**PROPOSED**) | none yet | Planned proof: independent architecture review verdict. | `df -h / && git worktree list && git status --short --untracked-files=all` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/install.log` (**PROPOSED**) | none yet | Planned proof: authorized supported installer output and exit. | `df -h / && git worktree list && git status --short --untracked-files=all` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/opt-in-launch.json` (**PROPOSED**) | none yet | Planned proof: one-process picker and cross-surface descriptor evidence. | `df -h / && git worktree list && git status --short --untracked-files=all` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/rollback.json` (**PROPOSED**) | none yet | Planned proof: one-action rollback and Gemma digest/generation evidence. | `df -h / && git worktree list && git status --short --untracked-files=all` |
| `scratch/qwen3.6-35b-a3b/evidence/phase-25/cleanup.txt` (**PROPOSED**) | none yet | Planned proof: measured owned cleanup and protected-resource inventory. | `df -h / && git worktree list && git status --short --untracked-files=all` |

### Phase 25 evidence

<a id="phase-25-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| - | Nothing; phase awaits dependencies | pending | `scratch/qwen3.6-35b-a3b/evidence/phase-25/` |

---

## Decisions

Version 1 was approved by Hebert at `2026-09-15T11:53:55Z`, as quoted under `## Approved scope`. The full 25-phase implementation remains approved. At the earlier version-1 approval, Phases 1 through 9 were closed. The earlier Phase 2, Phase 3, and Phase 4 approvals remain recorded above; Phase 5 closed after #51/#53/#57, Phase 6 after #93/#95/#98/#96, Phase 7 after #57/#59/#62/#64/#65, and Phase 8 after #74/#75/#76/#81. Each closeout distinguishes reviewed candidate identity from the later postapproval document-only revision. Later operational model/data actions still require their phase-specific authorization and repository preflight.

Phase 11 is closed on final candidate `ff2f4899b380480f7a58d93e69a729f1deb559a8fdb782ee4b7616654a29cd64` at HEAD `5770260510935cd32b82c4008ff06ae6458539ff`. Five of five tasks, three of three acceptance rows, seven of seven coverage rows, and D1-D5 pass; its candidate, approval, and correction history remain in the Phase 11 evidence section. Phase 12 is closed on the exact final2 candidate: six of six tasks, four of four acceptance items, sixteen of sixteen coverage rows, and D1-D5 pass. P13 is also closed on correction2; P14 is closed on 2026-09-17; P15 through P25 remain pending, and no authentic payload qualification, app installation, conversion, benchmark, performance result, or all-25-phase completion is claimed.

Phase 13 is closed on correction2 candidate `f9f620fd3d60d41bcb1d5becee5e0896c8eabfb18dd3b0a2d10a04d002d4b8c3` at HEAD `5770260510935cd32b82c4008ff06ae6458539ff`: six of six tasks, four of four acceptance items, twelve of twelve coverage rows, and D1-D5 pass. The accepted aggregate spans Qwen runner KV/convolution/recurrent state, transaction and facade state, session/client prepared-token routing, pending-token exact-once handling, full-state and FP16-scratch rollback/replay, hidden suffix removal, nonzero same-epoch failed-checkpoint recovery, capacity policy, and logical committed bytes. P14 is closed on 2026-09-17 with four of four tasks, three of three PHASE acceptance items, five of five coverage rows, and D1–D5 pass; P15 through P25 remain pending. The historical P14 staffing and review trail remains preserved in the earlier document body.

User staffing decision recorded 2026-09-16: make no further Grok assignments because credits are low; Sol remains implementation lead, DeepSeek Pro handles reviews, and Luna xhigh handles less-complex work. Independent tests and required gates remain unchanged. This policy applies to P14–P25; the current read-only P14 launch is recorded separately, historical assignments and prior approvals remain preserved, and no new preset is created.

Main corrected the unsupported P3 statement that unstored weights were unreconstructible. The Phase 3 source initializer and `make_schedule_modules` deterministically reconstruct old-attention and MoE inputs; absent serialization was not a blocker. The actual independent gap is the incomplete graph and non-cached old greedy fixture documented in the Phase 3 and Phase 12 sections. All P3 bytes, results, tolerances, and approvals remain valid as reduced-graph/component evidence.

Main authorized the additive Stage A artifact contract. One JSON carries relative-path/base64 virtual files, byte counts, per-file hashes, and aggregate identity for model weights, writer-v2-like packed-expert layout, four bounded expert blobs, and tensor-region records; actual generation, review, comparison, installation, and audit are recorded in the Phase 12 evidence table. There is no re-quantization, re-encoding, self-derived expected value, ignored-scratch test dependency, separate binary resource, manifest, receipt, or official-install claim.

The legal synthetic seam is deliberate. `GTurboFormatV2.swift:468-505,518-523` continues to pin official 2,048 hidden size, 40 layers, 256 experts, vocabulary, and provenance, so a 4×32 fixture cannot be a verified official Qwen `.gturbo`. Internal `QwenTextFixtureRecords` and `QwenTextModel.loadFixtureRecords` use synthetic geometry, actual BF16/affine bytes, the same private validated mapper, and the concrete runner; they cannot create `InstalledModelDescriptor`, enter `ModelFamilyRuntime`, inject direct Float values, or create a parallel toy path. Positive authentic payload execution is deferred.

The released Stage B scope was implemented and verified on the exact final2 candidate: metadata-only `ModelFamilyAdmission.classify` returns Gemma v1 or official Qwen v2 `LoadedModelManifest` without payload/GPU/model/runner work, and `Runtime.load` validates declared files, hashes, records, and layout before runner construction while v1 delegates unchanged `Model.load`. Package resource registration, focused suites, and the bounded hybrid correction are closed for the synthetic acceptance boundary. P13 is closed on its separate correction2 candidate; P14 is closed on 2026-09-17, P15 through P25 remain pending, and authentic 35B payload qualification remains outside the evidence.

The authoritative `qwen-phase12-text-runner` rev3 decision trail is preserved here: Main #11 corrected the unsupported unreconstructible-weights claim; Main #17 authorized an additive planning oracle; Main #34 required one durable JSON with exact virtual-pack bytes, counts, hashes, and aggregate identity; Main #41 set the synthetic legal seam; plan #42 revised as #45 was approved by Terra #46, Grok full #48, and formal #49; Main #51 forbade unfiltered `Scripts/test.sh`; Sol #53, Terra #52, and Grok #54 corrected and accepted the scoped verification boundary; and Main #56 accepted the P11 documentation and released Stage A only at that time. Subsequent #73 accepted/reviewed/installed artifact evidence and released Stage B; the later CPU-only finding, Main #104 rejection, Main #137 bounded correction release, Main #230 invalidation, and final2 closure are recorded in Phase 12. This historical trail does not claim P13–P25, authentic payload qualification, application installation, or all-plan completion.

A separately authorized prerequisite decision was recorded at `2026-09-15T14:17:13Z` (UTC). Main conveyed the exact user approval selection **"Authorize the pinned reference environment (Recommended)"** in response to **"May I prepare an isolated, repository-local reference environment using the exact official Transformers commit and its required Python dependencies?"**. This authorizes only the repository-local pinned Transformers source checkout and required CPU Python dependencies; it does not authorize model weights, model execution, generation, GPU commands, system-Python changes, or touching running app processes.

The setup completed with Transformers `git rev-parse HEAD` equal to `bd15bc95a89e728bbc1224084eb3b5829428c353`, an isolated venv, `pip check` exit 0, and offline imports of the readable `Qwen3_5Moe` class family. Exact setup evidence is in `scratch/qwen3.6-35b-a3b/evidence/phase-3/reference-environment/setup-summary.md`. Phase 3 later used that environment for bounded synthetic CPU-only construction after fresh preflights. No official model weights, shard payloads, or headers were downloaded or opened, and no GPU or production Qwen implementation ran.

A narrow Phase 11 task 11.2 owner decision was recorded in qwen-phase11-moe #35 on 2026-09-16. Owner said: “Choose lowest expert IDs deterministically (Recommended)”. This applies only to the task 11.2 routing tie policy: finite FP32 routing probabilities sort descending; exact equal probabilities break ties by ascending expert ID; eight unique IDs are selected and their weights are renormalized. This is TurboFieldfare's deterministic tie policy, not a guarantee of universal `torch.topk` tie parity. Frozen P3 reference outputs, fixtures, and tolerances remain unchanged. It does not approve the full plan by itself; Phase 11 completion is separately recorded in the implementation evidence.

## Waivers

No test waiver is recorded for version 1.

## Discovered work

| Found | What | Where it went |
|---|---|---|
| 2026-09-15 | The first assembled tracker used repeated generic tasks and non-path coverage rows despite structural PASS. | Replaced in version 1 before approval; no implementation task was marked complete. |
| 2026-09-15 | A prior tester unnecessarily rehashed the full 38-file `SHA256SUMS` set read-only, violating the no-shard-read instruction. | Disclosed in Phase 1 evidence; no mutations occurred; excluded from acceptance evidence and will not be repeated. |
| 2026-09-15 | The canonical official source moved into repository `scratch/` by same-volume rename; preparation did not rehash 72 GB. | P1/P5 consume recorded preparation evidence; P21 verifies current inputs before conversion. |
| 2026-09-15 | `Package.swift` already defines real `TurboFieldfareDecodeServiceTests`; CLI tests belong to `TurboFieldfareTestsCore` through `TurboFieldfareCLICore`. | P17 and P18 use those actual targets and paths. |
| 2026-09-15 | `script/build_and_run.sh` can stop app/service processes and install `/Applications/TurboFieldfare.app`. | P25 requires separate install/process authorization before invoking it. |
| 2026-09-15 | The first Phase 3 draft proposed position-only vision values, which could not serve the later tower/merger comparisons. | Formal escalation blocked it; revised plan `#91` constructed the actual pinned depth-one CPU/eager vision model and received substantive independent review. The earlier rejected and truncated review records remain disclosed in workflow history. |
| 2026-09-15 | The independent Phase 3 test's first exit-1 log was overwritten by its passing rerun. | Tester reported the four test-only compile errors exactly; the limitation is recorded in `fixture-generation.md`, and both independent and lead final logs are preserved. |
| 2026-09-15 | Task-tracker tooling expects lowercase `project-files`, while this project canonically stores plans under uppercase `Project-files`. | Phase 3 uses the same disclosed repository-scratch lowercase mirror as Phase 2, runs the builder/checker there, then copies only the generated HTML back to canonical `Project-files/human`. |
| 2026-09-16 | Phase 13 closeout has 12 source/test scope paths while the prior plan listed 10 coverage rows. | Added direct mappings for the inherited Qwen runner and app state test path; the authoritative engineering D4 union remains the four-stream receipt, not the lowercase documentation mirror. |

| 2026-09-22 | CLI cannot call the internal Qwen runner and vision implementation through the existing public bundle alone. | Task 18.2 adds the narrow ModelFamilyGeneration session interface; production-path tiny-fixture coverage is added to task 18.4 and both coverage tables. |
| 2026-09-22 | RealInferenceClient currently admits Qwen loads and prepared-token turns but deliberately refuses ordinary Qwen string/tool/image requests. | Phase 17 acceptance remains limited to identity/lifecycle behavior. Phase 20 must connect real family composition to app/service requests as part of the already approved selected-codec behavior; account for its exact source/test paths before edits. |

## Open questions

None.
