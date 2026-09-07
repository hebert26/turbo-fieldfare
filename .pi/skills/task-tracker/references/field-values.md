# Field values

Every cell in the tracker comes from this file. If a value you want is not here, it is not allowed - fix the work or ask, do not invent a word.

## Task and phase marks

Same five marks for both, plus one extra for tasks.

| Mark | Means | Closes a phase? |
|---|---|---|
| `[ ]` | not started | no |
| `[~]` | in progress | no |
| `[x]` | done, with evidence | yes |
| `[!]` | blocked - something outside this task stops it | no |
| `[-]` | skipped - reason written in the implementation document | yes |
| `[s]` | **source-done** - the code was written and accepted, but its live or manual proof has not passed | **no** |

### About `[s]`

Use it when code review said yes but the thing was never proven working. The old broken tracker in this repo marked a task `done` while its own notes said the live run failed. `[s]` is the honest word for that state.

`[s]` does not pass D1. A phase holding an `[s]` task stays open.

## Document status

`draft` -> `awaiting-approval` -> `approved` -> `in-progress` <-> `blocked` -> `completed`

Also `cancelled` and `superseded` as endings.

- Use `blocked` on the document only when **no** task can move. If one task is stuck and others can run, the document stays `in-progress` and the task gets `[!]`.
- Never set `approved` yourself. Only the owner.

## IDs

- Phases: `1`, `2`, `3`. Plain numbers, in dependency order.
- Tasks: `1.1`, `1.2`, `2.1`. Phase number, dot, task number.
- Anchors in the implementation document: `#phase-1`, `#phase-1-evidence`.

Never renumber a phase or task after approval. If work is dropped, mark it `[-]`. Renumbering breaks every link and every past log line.

## Dates

`YYYY-MM-DD`. Always. No times, no "yesterday", no relative words.

A done date is the date the task actually finished. If nobody recorded it, write `unknown` - never guess it from a folder name or a file timestamp.

## Numbers

**Every number in the tracker must come from a command that was actually run.** Put that command next to the number in the implementation document, so anyone can rerun it.

If you cannot name the command that produced a number, do not write the number. Write what you do know instead - "the module does not compile" is true and useful; "11 errors" that nobody can reproduce is worse than saying nothing.

### Counting compiler errors

Swift prints each error **twice** - once as `/path:line:col: error: message`, and once again as an inline caret annotation under the source. A failed test run adds a trailing `error: fatalError` on top.

So `grep -c 'error:'` over-counts by more than double. Anchor it:

```sh
swift build 2>&1 | grep -cE '^/.*error:'          # real error count
swift build 2>&1 | grep -E '^/.*error:' | cut -d: -f1-2 | sort -u | wc -l   # distinct sites
```

Measured 2026-08-15 on this repo: unanchored grep said **11**. The real count was **5**, at 3 sites. Both numbers reached the tracker and the HTML page, and they disagreed with each other.

### Counting tests

Read `Executed N tests, with K failures` from the run output. Never derive a count by reading the test file - on this repo a suite registered in three targets runs three times. See `coverage-gate.md`.

## Coverage Result

The closed list lives in `coverage-gate.md`. Repeated here so this file is complete:

`<n>/<n> pass` · `<n>/<m> pass, <k> fail` · `missing - <reason>` · `stale - rerun` · `no-unit-test - <proof>` · `waived - <link>`

## Board cells

| Column | What goes in it |
|---|---|
| `#` | the phase number |
| `Phase` | the phase name - what works when it is done |
| `Needs` | phase numbers this one waits on, comma separated, or `-` |
| `Tasks` | `<done>/<total>`, counting `[x]` and `[-]` as done |
| `Unit tests` | `<n> of <m> not passing`, or `all pass`, or `none - no code`. Append ` · D4 fallback` when the phase has no base commit. |
| `Status` | one of the five marks |
| `Done` | the date the phase closed, or `-` |

## Now block

Three lines, always these three.

```
- Phase <n> - task <n.n> - <owner>
- Blocked: <n> - coverage rows not passing: <n>
- Next: <one short sentence>
```

`Blocked` counts tasks marked `[!]` across the whole tracker. `coverage rows not passing` counts every row in every phase whose Result does not pass D3.

If nothing is running, write `Phase - - task - - -` and put the reason in `Next`.

## Owners

Short name in the tracker. Full role in the implementation document's `## Owners` table.

Keep the short name to one word so rows stay thin. Any constraint that matters for safety - who is allowed to run a thing, in which session - is written **once** in the implementation document, never dropped.

## Lengths

| Thing | Cap |
|---|---|
| Task line | 70 characters after the ID |
| Phase name | 60 characters |
| Acceptance line | 100 characters |
| Acceptance boxes per phase | 2 to 6 |
| Coverage rows per phase | about 8. More means the phase is too big - split it. |
| `Blocked by:` line | 100 characters |

**Copy the real number of acceptance criteria.** If a phase honestly has two, write two. Never pad to reach a minimum - an invented acceptance criterion is worse than a short list.

A task with no acceptance detail in the implementation document is a task that has not been written yet. Go write it there first, then add the row here.

## Banned words

Do not write these anywhere in the tracker:

- `exit criteria` - deleted concept, it always repeated phase acceptance
- `Definition of Done` under a task - it belongs to the phase
- `works correctly`, `handle the backend`, `as expected` - not checkable
- `TBD`, `...`, `etc` - either you know it or the row does not exist yet
