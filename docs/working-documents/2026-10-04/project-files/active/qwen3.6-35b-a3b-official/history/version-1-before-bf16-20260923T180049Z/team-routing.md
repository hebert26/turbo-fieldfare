# Qwen staffing routing

Static staffing reference for the proposed Qwen3.6-35B-A3B work. It does not
start work, approve a phase, replace the tracker, or change its dependencies.
Read the [tracker](tracker-qwen3.6-35b-a3b-official-2026-09-15.md) and
[implementation plan](implementation-qwen3.6-35b-a3b-official-2026-09-15.md)
before selecting a new bounded task. At this writing P13 is closed, P14 planning/reference preparation is underway, and
the read-only P14 team is launched. P14 implementation remains held; P15–25 remain
pending. The historical rosters and phase recommendations below are preserved for
prior assignments; future work follows the staffing decision below.

## Historical presets

| Preset | Roster |
| --- | --- |
| `mac-engineering-light` | Luna xhigh implementation/lead; Flash plan review; Flash straightforward test contributor; Grok initial review and verification. Small metadata, local validation, or simple bounded work only. |
| `mac-engineering-standard` | Terra medium implementation/lead; Flash plan review; Flash straightforward test contributor; Grok initial review and verification. Bounded plumbing after contracts freeze. |
| `mac-engineering-complex` | Sol high implementation/lead; Terra plan review and independent verification; Terra test contributor; Grok initial review and independent verification. Both verifiers must approve the same latest evidence. |
| `mac-engineering` | Existing five-member parallel option: Sol lead, Luna bounded contributor, Terra plan reviewer, Terra test contributor, Grok initial review/verification. Use only when disjoint owned paths justify its extra cost. |

## Future staffing decision

Recorded 2026-09-16 after P13 closeout: make no further Grok assignments because
credits are low. Sol remains the implementation lead for complex work. DeepSeek
Pro provides the independent reviews. Luna xhigh handles less-complex work.
Independent tests and required gates remain unchanged.

This is a future staffing policy, not a new preset. It applies to P14–P25 only;
the historical assignments, approvals, and evidence for P1–P13 remain unchanged.
The read-only `qwen-phase14-chat-codec` team is launched with Sol as lead, Luna
owning tests, and two separate DeepSeek Pro plan/initial/final review processes.
P14 production code and tests remain unwritten and unreleased. Main journal 52
releases only Sol's scratch-owned isolated tokenizer/template oracle stage in the
existing pinned offline environment; no model, weights, GPU, inference, or
downloads are involved.

Material math, concurrency, state, Metal, numerical-reference, or contract
changes route to Sol and normally `mac-engineering-complex`; do not disguise
them as Flash work. `mac-plan-review` gains verification authority only when a
roster explicitly assigns that duty. The implementation owner never verifies
its own work.

For one-file documentation, archival, or file moves, use solo Flash or Luna
outside the formal team pipeline. Do not create a two-member normal workflow to
save cost: the normal gate needs separate implementation, plan-review, and
initial-review/verification ownership. Select one preset per new bounded phase
task; do not start duplicate teams for the same goal. P21–P25 remain serial and
one model-related process at a time even when the code roster is parallel.

## Historical phase routing

These recommendations preserve the staffing record for earlier planning. Apply the
future staffing decision above to any phase not yet launched; the current P14 launch
record supersedes its historical recommendation.

| Phase | Recommended preset | Boundary and escalation |
| --- | --- | --- |
| P1 identity/hash fixture | light | Metadata identity and rejection tests only. Its tensor classification must be separately scoped to standard or complex before it starts; do not begin unsupported classification under light. |
| P2 v2 format compatibility | complex | Sol establishes the header/loader contract; later isolated plumbing may be standard after that contract freezes. |
| P3 reproducible execution fixtures | complex | Independent numerical references and generator provenance are not Flash fixture work. |
| P4 BF16 quantization | complex | Policy, rounding, transform, cancellation, and goldens stay Sol/independent Terra test. |
| P5 offline snapshot validation | standard | Terra owns local no-follow/catalog validation; escalate catalog semantics or loader contract changes to Sol. |
| P6 pack sizing | complex | Tensor accounting, hybrid state, and expert planning require Sol. |
| P7 resumable transformed pack | complex | Byte identity, checkpoints, disk audit, and transforms require Sol. |
| P8 Metal binding contract | complex | One host/MSL contract owner; do not split shared declarations without a Sol handoff. |
| P9 full attention | complex | Numerical oracle, KV lifetime, and Metal kernels require Sol and two verifiers. |
| P10 linear attention | complex | Hybrid/recurrent state, chunk carry, and kernels require Sol and two verifiers. |
| P11 MoE | complex | Routing, shared experts, paging, and numerical proof require Sol and two verifiers. |
| P12 tiny runner | complex | Runtime factory and weight mapping consume the prior kernel/state contracts. |
| P13 atomic turns | complex | Commit/rollback and replay state remain Sol work. |
| P14 chat template | standard | Terra owns pinned-template integration with Grok review; escalate tokenizer/BPE or runtime-state coupling to Sol. |
| P15 partial tool parser | complex | Action dispatch and the no-call safety boundary require Sol. |
| P16 still-image rows | complex | Vision state, Metal paths, and token-row numerical proof require Sol. |
| P17 decode service identity | standard | Terra can bind frozen identity/epoch contracts; move protocol or transaction-state changes to Sol. |
| P18 CLI verified requests | standard | Terra integrates frozen contracts; state or vision handoff changes route to Sol. |
| P19 loopback server | standard | Terra integrates frozen contracts; server lifecycle or state changes route to Sol. |
| P20 app selection | standard | Terra integrates only after contracts freeze; AppModel/state/vision handoff changes route to Sol. |
| P21 authorized conversion | standard | Existing verified converter operation only, with fresh authorization, disk and model-process preflight; converter-code changes are complex. |
| P22 real text qualification | complex | Preserve independent tests and two verifiers; execute serially with one model process. |
| P23 real screenshot qualification | complex | Preserve independent tests and two verifiers; execute serially with one model process. |
| P24 controlled comparison | standard | Flash/Luna may format captured evidence; Terra interprets metrics. Never benchmark in parallel. |
| P25 opt-in rollback | standard | Terra handles bounded bookkeeping while preserving prior Sol evidence and final code-approval gates; route rollback semantics to Sol. |

Use the tracker’s declared topological waves and shared-file handoffs, not this
table, to determine whether a phase may begin.
