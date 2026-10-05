# Continuation handoff

Folder index: [[turboCharge/Index]].

User wants Main to resume work after restarting, using this handoff; no need to ask again whether to continue.

## Program state

The overall goal is 25 approved Qwen3.6-35B-A3B integration phases, with Gemma unchanged as
the default and rollback. P1–P16 are accepted: 16/25 phases, 85/145 tasks, and 138/226
coverage rows satisfied; 88 remain pending.

P17 remains BLOCKED and NOT ACCEPTED after three unsuccessful correction cycles. P18–P25 have
not started. No P17 source, test, canonical byte, build, or test was changed or run in this
reassessment/repair segment.

The repository is `/Users/dev-machine/dev/turbo-fieldfare-personal`. Historical HEAD is
`5770260510935cd32b82c4008ff06ae6458539ff` and was not freshly verified.

Canonical root:
`/Users/dev-machine/dev/personal-project-documents/turboCharge/Project-files/active/qwen3.6-35b-a3b-official/`

The canonical files are `implementation-qwen3.6-35b-a3b-official-2026-09-15.md` (P17 lines
3064–3242), `tracker-qwen3.6-35b-a3b-official-2026-09-15.md` (P17 lines 750–791), and the
older `handoff.md` containing the P16 accepted addendum. This document records continuation
state; it is not a canonical acceptance update. The obsolete Idea Home path is historical.

Current stopped workers are logistics, not a prohibition: `qwen-p17-recovery-sol`,
`qwen-p17-plan-review-terra`, `qwen-p17-review-terra`, `qwen-p17-tests-luna`, and
`qwen-p17-reassessment-escalation-terra` confirmed stopped. `qwen-p17-review-luna` had already
stopped gracefully. Inspect their saved attempts rather than assuming they can resume.

## P17 decision record

The team is `workflow-turbo-high-mu6mcb0p-ivdcyd`, revision 7. Its journal is
`/Users/dev-machine/.pi/agent/team-chat/workflow-turbo-high-mu6mcb0p-ivdcyd.jsonl`.
The last live state was escalation after 304 events, before Main #304 was appended. Main #304
requested a read-only extension-path and input diagnosis; its completion was never received.

The current design is plan #276 with only observer section 7 replaced by #288. Initial reviews
are Terra #278 and Luna #279. Terra-high #285 blocks unamended #276. Luna renewal #293 covers
#276 plus #288, and Terra comparison #295 accepts #293. Luna still lacks the current comparison
of peer #278 assessed alongside own #293 (never self #293). Old #282 and escalation #283 do not
cover the current [#278, #293] round. A separate current Terra-high adjudication is still due.

Proposal #288 gives one observer OS thread ownership of kqueue create/register, both
`EV_RECEIPT` receipts, report, wait, last-use, and close. Setup sees the registration result.
Pre-registration cancellation and post-wait serialized USER-trigger-versus-close behavior are
part of the design. Only `NOTE_EXIT` proves process death; `NOTE_EXEC` revokes admission while retaining the exit watcher.
Those native production-observer checks have not run.

Remember the core lifecycle design while continuing: overlapping load/install must be REFUSED before detaching or constructing. Nonthrowing unload must JOIN actual cleanup, including ordinary bound unload and multiple or cancelled waiters; it must never be refused or return early. All strong holders release before successor construction or unload completion. Same-key runtimes must remain reusable; admission covers stale mutation and stream yield; bounded writer enqueue is one short atomic section with encoding and I/O outside it; and no lock crosses await or socket I/O. The complete requirements remain in plan #276 and the canonical P17 documents.

## Repair and evidence state

The separate workflow repair was accepted by Main #66 in `pi-review-round-recovery` with
same-byte approvals #56 and #57. Closeout was accepted by Main #19 in
`pi-review-recovery-closeout`. It added current-author/peer comparison provenance and
`author_review_index`; its frozen composite is
`022bce962943d6fb0fbb3ab70d865b47647673215633a11353cda1caf31ae6fd`.

Frozen inputs are `/Users/dev-machine/.pi/agent/extensions/team-chat/workflow.ts` SHA
`623530c707b285d1837d660d8bd5156bcf6431f79d9eb65befd0f9a576109873`, `index.ts` SHA
`eee01a6879ce6e34593ef80bcaa3955db7ec60065eae53818b1f5d12bd40c72a`, and
`Tests/PiWorkflowRecovery/ReviewRecovery.test.mjs` SHA
`af59a4437de2b3d9f9ded5b3c3b558563fc2cc936a9a70df8b55db2a91bb240e`.

Receipts are under `/Users/dev-machine/dev/turbo-fieldfare-personal/scratch/workflow-review-recovery/`
and closeout `closeout-305e50c0-e357-4a70-be42-84fe60806f2a/`. The candidate focused script
passed after the baseline orphan-escalation assertion failed. One script ran with 31 static
assertion sites, 34 authored invocations, and nine scenario groups; this is not 31 discovered
tests. Eleven existing scripts and the source-focused TypeScript check passed. Full TypeScript
`noEmit` still failed with the same 40 baseline dependency errors. The repair does not accept P17.

Source replay expected `initial_review_peer_challenge`, while live Main `team_status` remained
escalation after the reported `/reload` and the exact-same-Main restart attempt (“try now”).
Repaired-module activation is therefore unproved; the cause is not established. Sol diagnosis
#301 describes reload invalidating the runner and resource loader, clearing factories, and Jiti
using `moduleCache: false`, but source mechanics are not live proof.

Luna attempt `0d1cc817` was saved at roster revision 4; current roster revision is 7.
`verifySavedResume` rejects that exact mismatch. Establish supported fresh reviewer staffing
rather than assuming old attempts are resumable. Main #304's unfinished diagnostic concerned
project/global extension resolution, duplicates, and the actual `team_status` derive route.

## Continuation sequence

1. After Main restarts, inspect the current runtime, team journal, saved attempts, assignments,
   and the unfinished #304 diagnosis.
2. Resolve why live state disagrees with the repaired source, and establish that the repaired
   module is actually active in the processes that will review the work.
3. Establish supported fresh reviewer staffing, then finish the current #276 + #288 comparison
   and the separate Terra-high adjudication. Treat missing comparisons as missing approvals.
4. Carry the implementation, independent test, verification, evidence, and review gates through
   P17. The proposed packet has nine production files for Sol and five existing test files for
   Luna; its detailed file list is in plan #276 and the stopped P17 packet.
5. Run each individual `Scripts/test.sh --filter <suite>` serially; the wrapper already supplies `--no-parallel`. Follow `scratch/qwen3.6-35b-a3b/evidence/phase-17/proposed-execution-commands.txt` for the exact nine-build/thirteen-filter packet, reporting real counts and unique evidence paths. Historical P17 evidence is under `scratch/qwen3.6-35b-a3b/evidence/phase-17/`; nine historical builds passed, but later selectors were not all executed and Layer B failed before install.
6. Once P17 earns its gates and acceptance, proceed through P18–P25 with separate evidence for
   metadata/tokenizer/wire behavior, production installation, fake-runtime coordination, and
   authentic-model success. Preserve Gemma, `.gturbo` v1, and all P1–P16 evidence.

Known unrelated package failures remain `MacAppSettingsTests.aNewerSettingsFileSurvivesAKeyThisBuildDoesNotKnow`
and `VisionResourceLimitTests.privateStoragePositionsAreRefusedRatherThanDereferenced`; no
blanket-green claim has been established. This documentation update itself only rewrote this
handoff and did not restart engineering or execute commands.
