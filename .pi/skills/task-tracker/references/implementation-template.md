# Implementation document template

Copy everything between the two `=== COPY ===` markers into `implementation-<slug>-<YYYY-MM-DD>.md`.

This file holds **everything the tracker is not allowed to hold**: why, scope, risks, where the code is, how to build it, how to check it, what was proved.

Rules for the file you write:

- Write this file **first**. The tracker is built from it, not the other way round.
- No line limit. Be as detailed as the work needs.
- It holds acceptance detail and evidence. It does **not** hold live task state - no `[x]` task lists, no owner-of-record, no completion dates. That is the tracker's job.
- Every phase heading gets an anchor the tracker links to: `## Phase 1 - <name>` -> `#phase-1`.
- Cite real code as `path/to/File.swift:120`. Never describe code you have not opened.

=== COPY ===
# Implementation: <work item name>

Tracker: [tracker-<slug>-<YYYY-MM-DD>.md](./tracker-<slug>-<YYYY-MM-DD>.md)
Date: <YYYY-MM-DD>

---

## Approved scope - version 1

**Changing anything in this section needs the owner's approval again.** Bump `version` in the tracker and set it back to `awaiting-approval`.

<One paragraph. What we are doing, and the boundary. Be blunt about what is out of bounds.>

### What changes

1. <exact change>
2. <exact change>

### What does not change

- <thing that stays exactly as it is>
- <thing that stays exactly as it is>

### Files

| Action | File | What |
|---|---|---|
| Create | `<path>` | <why> |
| Modify | `<path>` | <what changes in it> |
| Delete | `<path>` | <why> |
| Do not touch | everything else | - |

---

## Problem

<Why this work exists. What is wrong or missing today. Two or three short paragraphs.>

## Phase order

```
Phase 1 <name>
   |
   v
Phase 2 <name>
   |
   +--> Phase 3 <name>
   +--> Phase 4 <name>
```

<One line saying which phases can run at the same time.>

## Risks

| Risk | Impact | What we do about it |
|---|---|---|
| <what could go wrong> | low / medium / high | <the mitigation> |

## Owners

| Short name | Full role and any constraint |
|---|---|
| `<name>` | <full role. Include any safety rule, for example "only in the approved session".> |

---

## Phase 1 - <what works when this is done>

**When this is done:** <finish the sentence "______ works">

Needs: `<phase numbers, or nothing>`
Base commit: `<sha written the moment this phase starts>`
Disk baseline: `<free space, plus the simulator devices and worktrees that already existed when this phase started>`
Evidence: `<path to the evidence folder, or "pending">`

### Current state

<Where the relevant code lives now. Real paths and line numbers.>

### Target state

<What it looks like after. Diagrams, code sketches, contracts - whatever makes it unambiguous.>

### Tasks

#### 1.1 <task name>

| | |
|---|---|
| Touches | `<paths>` |
| Depends on | `<task IDs, or nothing>` |
| Parallel safe | yes / no |

<What to do. As long as it needs to be.>

**Acceptance detail**

- [ ] <precise, checkable statement>
- [ ] <precise, checkable statement>

**How to check**

```
<the exact command>
```

**Evidence**

<Pending, or what was actually observed. Never write "done" without saying how you know.>

#### 1.2 <task name>

<same shape>

### Phase 1 coverage plan

Write this **before** the code, not after. The tracker's coverage rows are copied from the first two columns of this table.

| Code this phase changes | Test file that must cover it | What the test must prove | Command |
|---|---|---|---|
| `<source path>` | `<test path>` | <the behaviour, not the file name> | `<test command>` |

If a phase changes no code, write one row saying so and name the proof used instead.

### Phase 1 evidence

<a id="phase-1-evidence"></a>

| Date | What was run | Result | Where the output is |
|---|---|---|---|
| `<YYYY-MM-DD>` | `<command>` | <pass or fail, with the real numbers> | `<path>` |

---

## Phase 2 - <what works when this is done>

<same shape as Phase 1>

---

## Decisions

| Date | Decision | Who | Why |
|---|---|---|---|
| `<YYYY-MM-DD>` | <what was decided> | <who> | <the reason> |

## Waivers

Only the owner puts a row here. One row per `waived` coverage cell, each with his exact words. Recipe 0b.

| Date | Coverage row waived | Owner quote |
|---|---|---|
| `<YYYY-MM-DD>` | `<the source path from the tracker>` | Owner said: "<their exact sentence>" |

Delete the example row. If there are no waivers, write `None.`

## Discovered work

| Found | What | Where it went |
|---|---|---|
| `<YYYY-MM-DD>` | <the thing nobody planned for> | <new task ID, or "needs re-approval"> |

## Open questions

<None, or the list. An open question that blocks a phase must also appear as "Blocked by" in the tracker.>
=== COPY ===

## Where the old three-file split went

If you are used to a separate `plan-<slug>.md`, its content now lives here:

| Old plan section | New home |
|---|---|
| Scope | `## Approved scope` |
| Problem overview | `## Problem` |
| Task breakdown | `### Tasks` inside each phase |
| Dependency graph | `## Phase order` |
| Risk register | `## Risks` |
| Files summary | `### Files` inside the approved scope |
| Open questions | `## Open questions` |

One file instead of two means one copy of the step list. Two copies always drift, and then one of them lies.

## The frozen part

Everything under `## Approved scope` is what the owner said yes to. Everything below it is working detail that agents update freely.

Change the working detail whenever the work demands it. Change the approved scope only by asking again.
