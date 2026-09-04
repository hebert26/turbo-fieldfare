# Tracker template

Copy everything between the two `=== COPY ===` markers into `tracker-<slug>-<YYYY-MM-DD>.md`.

Replace every `<...>`. Delete what you do not need. Add one phase block per phase.

Rules for the file you write:

- **Tasks only.** No scope, no reasoning, no risks, no code. Those live in the implementation document.
- One task is one line. A task line never holds a sentence about *why*.
- A phase has exactly four blocks, this order: tasks, acceptance, unit test coverage, done when.
- Never add a block. Never add a heading deeper than `###`.
- Never reword D1 to D5. They are fixed text.
- Cell values: `field-values.md`. Coverage rules: `coverage-gate.md`.
- Check it before handing over: `self-check.md`.

=== COPY ===
---
title: "<work item name>"
slug: <work-item-slug>
implementation: ./implementation-<slug>-<YYYY-MM-DD>.md
status: awaiting-approval
version: 1
approved_version: null
approved_by: null
approved_at: null
created_at: "<YYYY-MM-DD>"
updated_at: "<YYYY-MM-DD>"
current: null
next_action: "Review and approve version 1."
---

# Tracker: <work item name>

Goal: <one plain sentence. What is true when all of this is finished.>

Details, scope, code, evidence: [implementation-<slug>-<YYYY-MM-DD>.md](./implementation-<slug>-<YYYY-MM-DD>.md)

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

- Phase `<n>` - task `<n.n>` - `<owner>`
- Blocked: `<n>` - coverage rows not passing: `<n>`
- Next: `<one short sentence>`

## Phases

| # | Phase | Needs | Tasks | Unit tests | Status | Done |
|---|---|---|---|---|---|---|
| 1 | <what works when this is done> | - | 0/<n> | <n> of <m> not passing | `[ ]` | - |
| 2 | <what works when this is done> | 1 | 0/<n> | <n> of <m> not passing | `[ ]` | - |

## Done-when rules

Every phase closes on the same five lines. Written in full once here. Repeated short inside each phase. Copy them word for word.

- **D1 tasks** - every task in the phase is `[x]` or `[-]` with a reason in the implementation document. `[~]`, `[!]` and `[s]` do not pass.
- **D2 acceptance** - every acceptance box in the phase is ticked.
- **D3 covered** - every row of the phase coverage table reads `<n>/<n> pass`, `no-unit-test - <proof>`, or `waived - <link>`. `missing`, `stale`, and any failing count block the phase.
- **D4 no blind spot** - no file this phase changed is missing from its coverage table. Run the command in `coverage-gate.md`. Every path it prints must appear in the table, as its own row or under a directory row that is a prefix of it.
- **D5 evidence** - the phase evidence path is written in the implementation document.

**A phase cannot be `[x]` while D3 or D4 fails.** Code this phase added, with no unit test covering it, blocks this phase.

D3 proves a named test file exists and really ran and passed. It does not prove the test exercises that code well. The `What the test must prove` column in the implementation document is what a reviewer reads to judge that.

## Phase 1 - <what works when this is done>

Status: `[ ]` - Owner: `<name>` - Needs: `<phase numbers, or ->` - Done: `-` - [details](./implementation-<slug>-<YYYY-MM-DD>.md#phase-1)

- [ ] 1.1 <task, plain words, under 70 characters>
- [ ] 1.2 <task, plain words, under 70 characters>
- [ ] 1.3 <task, plain words, under 70 characters>

**Acceptance**

- [ ] <observably true when this phase works>
- [ ] <observably true when this phase works>

**Unit test coverage**

Base: `<commit sha, or "unknown - D4 fallback">`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `<source path>` | `<test path>` | `<n>/<n> pass` | `<YYYY-MM-DD>` |
| `<source path>` | none yet | `missing - <short reason>` | - |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

Blocked by: `<what is stuck, one short line. Delete this line when nothing is stuck.>`

## Phase 2 - <what works when this is done>

Status: `[ ]` - Owner: `<name>` - Needs: `1` - Done: `-` - [details](./implementation-<slug>-<YYYY-MM-DD>.md#phase-2)

- [ ] 2.1 <task>
- [ ] 2.2 Clean up: build output, logs, devices and worktrees this work item created

**Acceptance**

- [ ] <observably true when this phase works>

**Unit test coverage**

Base: `<commit sha, or "unknown - D4 fallback">`

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `<source path>` | `<test path>` | `<n>/<n> pass` | `<YYYY-MM-DD>` |

**Done when**

- [ ] D1 every task above is `[x]` or `[-]`
- [ ] D2 every acceptance box above is ticked
- [ ] D3 every coverage row passes, or is `no-unit-test` or `waived`
- [ ] D4 no file this phase changed is missing from the table
- [ ] D5 evidence path is in the implementation document

## Changelog

| Date | Phase | What happened |
|---|---|---|
| `<YYYY-MM-DD>` | - | Tracker created at version 1. Awaiting approval. |
=== COPY ===

## The cleanup task

The last task of the **last** phase, always, no exception:

```
- [ ] <n>.<last> Clean up: build output, logs, devices and worktrees this work item created
```

It counts for D1 on that phase, so the work item does not close until it is `[x]`. What to clean, what never to delete, and how to record what was freed: `cleanup.md`.

Earlier phases do not get a cleanup task. One clean at the end covers the whole work item.

They still record a disk baseline the moment they start - recipe 0 in `update-recipes.md` - because the final cleanup needs it to tell what this work item created from what was already there. A phase that writes no code still creates something: a booted simulator, a log, a worktree. Its baseline is what makes that visible at the end.

If the disk gets tight before the last phase, clean early anyway. That is housekeeping, not the cleanup task - it ticks nothing and removes nothing from the tracker. See "When the disk is already tight" in `cleanup.md`.

## A task line

One line. Three parts at most: the ID, what to do, and the date it was finished.

```
- [x] 2.3 Add responsive breakpoints to the column layout   2026-03-14
- [~] 2.4 Remove the fixed 560pt width cap
- [!] 2.5 Wire the redeploy button to each row
```

No owner on the line - the owner is on the phase. No "why" - that is the implementation document. No sub-bullets.

If a task needs explaining, it gets a section in the implementation document and the phase links there once. It does not get three more lines here.

## A phase with no code

Some phases change no code: a live proof run, a manual check, a written document.

Do not delete the coverage block. Write exactly one row:

```
| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| none - this phase changes no code | none | `no-unit-test - <what proves it instead>` | `<YYYY-MM-DD>` |
```

This row is only legal when the D4 command prints nothing. If D4 prints a file, the phase did change code and this row is a lie. See `coverage-gate.md`.

## Size

The old broken tracker in this repo hit 988 lines for 23 tasks - about 25 lines per task - because it repeated three near-identical gate lists under every single task.

In this format one task is one line, and the gate lives on the phase. Expect roughly:

```
60 + (30 x number of phases) + (1 x number of tasks)
```

Five phases and twenty tasks lands near 230 lines. Seven phases and forty-five tasks lands near 315.

This is the same formula as `SKILL.md` and check 1 in `self-check.md`. **There is one budget. If you find a second one written anywhere, it is wrong - fix it, do not average them.** An earlier copy of this file said `40 + 18 x phases + tasks`, which predicted 211 lines for a real 338-line tracker and would have told an agent to go and delete gate blocks from a correct file.

If you are far above the budget, you are putting prose in the tracker. Move it.
