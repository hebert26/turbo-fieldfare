# Worked example

A made-up work item, small enough to read in one go. It shows a phase that passes its gate and a phase that fails it.

Work item: **offline notes** - a notes app that keeps working with no network.

Three phases. Read the names first:

- *A note saves and reopens with no network.*
- *Queued notes reach the server when the network returns.*
- *The user can see which notes are still waiting.*

Each one finishes the sentence "when this phase is done, ______ works". None of them needs the word "and". None of them is called "backend", "phase 2 of sync", or "polish".

---

## The tracker

File: `project-files/active/offline-notes/tracker-offline-notes-2026-08-15.md`

````markdown
---
title: "Offline notes"
slug: offline-notes
implementation: ./implementation-offline-notes-2026-08-15.md
status: in-progress
version: 1
approved_version: 1
approved_by: Hebert
approved_at: "2026-08-12"
created_at: "2026-08-12"
updated_at: "2026-08-15"
current: "2.3"
next_action: "Write OfflineQueueTests before task 2.4 can start."
---

# Tracker: Offline notes

Goal: the app keeps saving, reading and queueing notes while the device has no network, and catches up on its own when the network returns.

Details, scope, code, evidence: [implementation-offline-notes-2026-08-15.md](./implementation-offline-notes-2026-08-15.md)

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

- Phase `2` - task `2.3` - `Luna`
- Blocked: `0` - coverage rows not passing: `3`
- Next: write `OfflineQueueTests` so phase 2 can close.

## Phases

| # | Phase | Needs | Tasks | Unit tests | Status | Done |
|---|---|---|---|---|---|---|
| 1 | A note saves and reopens with no network | - | 3/3 | all pass | `[x]` | 2026-08-13 |
| 2 | Queued notes reach the server when network returns | 1 | 2/4 | 2 of 3 not passing | `[~]` | - |
| 3 | The user can see which notes are still waiting | 2 | 0/3 | 1 of 1 not passing | `[ ]` | - |

## Done-when rules

Every phase closes on the same five lines. Written in full once here. Repeated short inside each phase. Copy them word for word.

- **D1 tasks** - every task in the phase is `[x]` or `[-]` with a reason in the implementation document. `[~]`, `[!]` and `[s]` do not pass.
- **D2 acceptance** - every acceptance box in the phase is ticked.
- **D3 covered** - every row of the phase coverage table reads `<n>/<n> pass`, `no-unit-test - <proof>`, or `waived - <link>`. `missing`, `stale`, and any failing count block the phase.
- **D4 no blind spot** - no file this phase changed is missing from its coverage table. Run the command in `coverage-gate.md`. Every path it prints must appear in the table, as its own row or under a directory row that is a prefix of it.
- **D5 evidence** - the phase evidence path is written in the implementation document.

**A phase cannot be `[x]` while D3 or D4 fails.** Code this phase added, with no unit test covering it, blocks this phase.

D3 proves a named test file exists and really ran and passed. It does not prove the test exercises that code well. The `What the test must prove` column in the implementation document is what a reviewer reads to judge that.

## Phase 1 - A note saves and reopens with no network

Status: `[x]` - Owner: `Terra` - Needs: `-` - Done: `2026-08-13` - [details](./implementation-offline-notes-2026-08-15.md#phase-1)

- [x] 1.1 Add a local note store backed by SQLite            2026-08-12
- [x] 1.2 Route reads and writes through the local store     2026-08-13
- [x] 1.3 Register the new test file in Package.swift         2026-08-13

**Acceptance**

- [x] A note written with airplane mode on is readable after a cold app restart.
- [x] No write path calls the network directly.

**Unit test coverage**

Base: `a3f19c2`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/Notes/LocalNoteStore.swift` | `Tests/LocalNoteStoreTests.swift` | `11/11 pass` | `2026-08-13` |
| `Sources/Notes/NoteRepository.swift` | `Tests/LocalNoteStoreTests.swift` | `11/11 pass` | `2026-08-13` |
| `Package.swift` | none | `no-unit-test - build plus test run, evidence/2026-08-13_phase1/` | `2026-08-13` |

**Done when**

- [x] D1 every task above is `[x]` or `[-]`
- [x] D2 every acceptance box above is ticked
- [x] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [x] D4 no file this phase changed is missing from the table
- [x] D5 evidence path is in the implementation document

## Phase 2 - Queued notes reach the server when network returns

Status: `[~]` - Owner: `Luna` - Needs: `1` - Done: `-` - [details](./implementation-offline-notes-2026-08-15.md#phase-2)

- [x] 2.1 Add a durable write queue                          2026-08-14
- [x] 2.2 Drain the queue on a network-reachable event        2026-08-15
- [~] 2.3 Resolve a note edited on two devices
- [ ] 2.4 Register the queue test files in Package.swift

**Acceptance**

- [ ] A note written offline appears on the server within 30 seconds of reconnecting.
- [ ] A note edited on two devices resolves to the newer edit, with no data lost.

**Unit test coverage**

Base: `c81b407`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/Notes/OfflineQueue.swift` | none yet | `missing - task 2.4 not started` | - |
| `Sources/Notes/ReachabilityWatcher.swift` | `Tests/ReachabilityWatcherTests.swift` | `6/6 pass` | `2026-08-15` |
| `Sources/Notes/ConflictResolver.swift` | none yet | `missing - task 2.3 in progress` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

Blocked by: D3 fails. `OfflineQueue.swift` and `ConflictResolver.swift` have no test file.

## Phase 3 - The user can see which notes are still waiting

Status: `[ ]` - Owner: `Terra` - Needs: `2` - Done: `-` - [details](./implementation-offline-notes-2026-08-15.md#phase-3)

- [ ] 3.1 Show a pending badge on any note not yet on the server
- [ ] 3.2 Show the queue count in the toolbar
- [ ] 3.3 Clean up: build output, logs, devices and worktrees this work item created

**Acceptance**

- [ ] A note that has not reached the server shows a pending badge.
- [ ] The badge clears within one second of the note syncing.

**Unit test coverage**

Base: `unknown - D4 fallback`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/Notes/PendingBadgeModel.swift` | none yet | `missing - phase not started` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Changelog

| Date | Phase | What happened |
|---|---|---|
| 2026-08-15 | 2 | 2.2 done. Phase 2 held open by D3 - two files have no tests. |
| 2026-08-13 | 1 | Phase 1 closed. All five gates passed. |
| 2026-08-12 | - | Approved at version 1 by Hebert. |
````

That is 3 phases and 10 tasks in 161 lines. The budget in `self-check.md` predicts 160.

---

## What the tracker is doing here

**Phase 1 closed properly.** Three tasks done, both acceptance boxes ticked, every coverage row passing, and `Package.swift` handled honestly with `no-unit-test` plus a real evidence path.

**Phase 2 is stuck, and the tracker says so out loud.** Two of its three source files have no test file. D3 fails. The `Blocked by:` line names exactly which files.

Nothing in phase 2 is hidden. The old broken tracker in this repo would have shown two green acceptance ticks and never mentioned that no test existed.

**Task 1.3 and task 2.4 exist on purpose.** On this repo a test file does not run until it is listed in `Package.swift`. Registering it is real work, so it is a real task. See `coverage-gate.md`.

**Cleanup is one task, at the very end** - 3.3, the last task of the last phase. Phases 1 and 2 do not carry one. They each recorded a disk baseline when they started, and 3.3 uses those baselines to tell what this work item created from what was already on the machine. The work item cannot close until 3.3 is `[x]`. See `cleanup.md`.

**Phase 3 uses the D4 fallback** because it has not started and has no base commit yet. The Base line says so instead of hiding it.

---

## The matching implementation document

Only the parts a tracker reader would follow a link to.

````markdown
## Phase 2 - Queued notes reach the server when network returns

**When this is done:** a note written with no network reaches the server on its own, and a note edited in two places does not lose an edit.

Needs: `Phase 1`
Base commit: `c81b407`
Evidence: `evidence/2026-08-15_phase2/`

### Current state

`NoteRepository.swift:88` writes straight to the local store and returns. Nothing ever retries. A note written offline stays local forever.

### Target state

A durable queue table sits beside the note table. Every local write appends a queue row. A reachability watcher drains it in order, oldest first, with one retry backoff.

### Tasks

#### 2.3 Resolve a note edited on two devices

| | |
|---|---|
| Touches | `Sources/Notes/ConflictResolver.swift` |
| Depends on | 2.1 |
| Parallel safe | no |

Last-writer-wins on `updated_at`, with the losing version kept in a `note_conflicts` table so nothing is destroyed.

**Acceptance detail**

- [ ] Two edits to one note resolve to the one with the later `updated_at`.
- [ ] The losing edit is readable from `note_conflicts` after resolution.
- [ ] Equal timestamps resolve by device ID, deterministically, never at random.

**How to check**

```
cd VisionCapture && swift test --filter ConflictResolverTests 2>&1 | tail -20
```

**Evidence**

Pending. `ConflictResolver.swift` exists. `ConflictResolverTests.swift` does not.

### Phase 2 coverage plan

| Code this phase changes | Test file that must cover it | What the test must prove | Command |
|---|---|---|---|
| `Sources/Notes/OfflineQueue.swift` | `Tests/OfflineQueueTests.swift` | A queued write survives a cold restart and drains in the order it was written. | `--filter OfflineQueueTests` |
| `Sources/Notes/ReachabilityWatcher.swift` | `Tests/ReachabilityWatcherTests.swift` | The drain fires once per reconnect, not once per interface event. | `--filter ReachabilityWatcherTests` |
| `Sources/Notes/ConflictResolver.swift` | `Tests/ConflictResolverTests.swift` | The later edit wins, the earlier edit is kept, equal timestamps resolve deterministically. | `--filter ConflictResolverTests` |

### Phase 2 evidence

<a id="phase-2-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| 2026-08-15 | `swift test --filter ReachabilityWatcherTests` | Executed 6 tests, 0 failures | `evidence/2026-08-15_phase2/reachability.txt` |
````

Note the `What the test must prove` column. It says **behaviour** - "drains in the order it was written" - not "tests OfflineQueue.swift". That column is the only defence against a test that passes while proving nothing, because this repo has no coverage tooling.

---

## The HTML page

Copy `implementation-plan-template.html` into the same folder as `implementation-offline-notes-2026-08-15.html`.

For this example it would show:

- **scope card** - the approved scope paragraph, word for word.
- **stats** - `3` phases, `6/12` tasks done, `3` coverage rows failing, `0` need your decision.
- **timeline** - three `t-plan` entries. Phase 1 gets `class="chosen"` on its Unit tests row because everything passes. Phase 2 does not, and its Done-when row reads *"D3 fails. Two files have no test file."*
- **open block** - "Nothing open."

The page tells the owner in five seconds what the tracker takes a minute to read.

---

## What to copy from this example

- Phase names that finish the sentence.
- One line per task, with a date when it is done.
- A `Blocked by:` line that names files, not feelings.
- `no-unit-test` used honestly for `Package.swift`, with a real evidence path.
- `unknown - D4 fallback` written down rather than hidden.
- Two acceptance criteria where there are honestly two. Not padded to four.

## What not to copy

Do not copy the phase names or task names into real work. They are invented. Write your own from the actual code.
