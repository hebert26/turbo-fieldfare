# The unit test gate

The owner's rule:

> Each phase should have code covered by unit tests, and the definition of done for that phase should include that.

Two of the five done-when lines carry it.

- **D3** - every row of the phase coverage table passes.
- **D4** - no file this phase changed is missing from the table.

D3 alone is not enough. Someone can list two files they tested and quietly leave out the three they did not. D4 is what closes that hole.

**A phase cannot be marked done while D3 or D4 fails.**

## The coverage table

Lives in the tracker, under the phase. Four columns, no more.

| Code this phase changed | Test file | Result | Checked |
|---|---|---|---|
| `Sources/Notes/OfflineQueue.swift` | `Tests/OfflineQueueTests.swift` | `9/9 pass` | `2026-08-15` |
| `Sources/Notes/SyncResolver.swift` | none yet | `missing - task 2.4 not started` | - |

The first two columns are **copied** from the coverage plan in the implementation document. Never invent a mapping here. If nobody has said which test covers a file, that is a question for the implementation document, not a guess in the tracker.

## Result values - the only ones allowed

| Value | Means | Passes D3 |
|---|---|---|
| `<n>/<n> pass` | The named test file ran and every test passed. | yes |
| `<n>/<m> pass, <k> fail` | It ran and something failed. | no |
| `missing - <reason>` | No test file exists yet. | no |
| `stale - rerun` | It passed once, then the code changed. Nobody has rerun it. | no |
| `no-unit-test - <proof>` | This row genuinely cannot have a unit test. Name what proves it instead. | yes |
| `waived - <link>` | The owner accepted no test here. Link the decision. | yes |

Nothing else. Not `ok`. Not `done`. Not `pass` on its own - a count with no number hides whether anything ran.

## Directory rows

A row may name a directory instead of a file:

```
| `Sources/Notes/` | `Tests/NotesTests.swift` | `14/14 pass` | `2026-08-15` |
```

A directory row covers every file under it. Use one when a phase adds a whole module. Use file rows when a phase touches scattered files in a module it does not own.

## D4 - the command

**Do not type this by hand. Run the script:**

```sh
$REPO/.codex/skills/task-tracker/scripts/check-tracker.sh <work-item-folder> --base <sha>
```

It runs D4 with the shape below, plus the empty-output guard, plus every other check in `self-check.md`.

If you must run D4 alone, **run it from the repository root** and use exactly this shape:

```sh
BASE=<the phase base commit>
PATHS=(VisionCapture/Sources VisionCapture/Resources VisionCapture/scripts VisionCapture/Package.swift)

{ git diff --name-only "$BASE"..HEAD -- "${PATHS[@]}"
  git diff --name-only            -- "${PATHS[@]}"
  git diff --name-only --cached   -- "${PATHS[@]}"
  git ls-files --others --exclude-standard -- "${PATHS[@]}"
} | sort -u
```

Every path it prints must appear in the coverage table - as its own row, or under a directory row that is a prefix of it.

### The guard - run it every time

An empty result means one of two things, and they are opposites: the phase truly changed nothing, or the command is broken. Never assume the first.

```sh
git status --porcelain -- "${PATHS[@]}" | head
```

**If D4 printed nothing and this prints something, D4 is broken. Fix it before you tick anything.**

### Never write the path list into a plain string

This is not style. A string breaks the gate.

```sh
P="VisionCapture/Sources VisionCapture/Resources ..."   # WRONG
git diff --name-only -- $P
```

Bash splits `$P` on spaces. **zsh does not.** zsh passes the whole string as one pathspec, git finds no such path, and prints nothing.

The shell on this machine is zsh (`/bin/zsh`, 5.9). Measured 2026-08-15, same repository, same moment:

| Shape | Files printed |
|---|---|
| `P="..."` then `-- $P` | **0** |
| `PATHS=(...)` then `-- "${PATHS[@]}"` | **5** |

Five real untested source files, and the string version reported a clean phase. An array works in both shells - verified on zsh 5.9 and bash 3.2.

This is the same failure as "A gate that checks nothing" in `anti-patterns.md`, reintroduced by one pair of quotes. That is why the script exists: so nobody retypes it.

### Why all four lines

`git diff` alone **does not work on this project.**

Commits belong to Hebert alone - no agent ever commits. So work in progress sits untracked, often for days. `git diff` never lists untracked files. A single-command version of D4 prints nothing and ticks green on a phase with zero tests - the exact failure the gate exists to catch.

**Never commit something to make this gate pass.** That is not allowed and it is not your call. If the gate fails, write the missing test.

Verified on 2026-08-15: `git diff --name-only HEAD~3..HEAD -- VisionCapture/Sources` printed nothing while seven new untracked source modules sat in the tree.

The four lines cover committed, unstaged, staged, and untracked. Keep all four.

### The base commit

Write it into the implementation document under the phase heading, **the moment the phase starts**, before the first line of code. See recipe 0 in `update-recipes.md`.

### If there is no base commit

Fallback: every path in the implementation document's `### Files` table for this phase must appear in the coverage table.

It is weaker - it trusts a list a person wrote instead of asking git. So make the weakness visible: write `unknown - D4 fallback` in the Base line, and put `D4 fallback` in the phase's `Unit tests` cell on the board.

## D3 - what counts as a pass

Three things, all of them:

1. The test file **exists** at the path written in the table.
2. The test run printed **`Executed N tests`** with **N above zero**.
3. Zero failures.

### Why the count matters on this repo

`VisionCapture/Package.swift` declares three test targets - `CaptureCoreTests` (line 126), `CLITests` (line 404), `VisionCaptureTests` (line 683). Each one lists its test files **by hand** in an explicit `sources:` array.

A new file in `VisionCapture/Tests/` **does not run** until it is added to those arrays. And `swift test --filter Foo` still **exits 0** when it matches nothing.

So an agent can write a test, run it, match nothing, see a clean exit, and record `pass`. Nothing ran.

Two consequences:

- Adding a test file is not done until it is registered in `Package.swift`. Make that an explicit task, not an afterthought.
- A file registered in all three targets runs **three times**. A printed count of 27 for a 9-method class is correct, not a bug. Read the count from the run - never derive it by counting methods in the file.

### The commands

From `VisionCapture/`:

```sh
swift test --filter <SuiteName> 2>&1 | tail -20
```

Read the `Executed N tests, with K failures` line. If N is 0, the row is `missing`, not `pass`.

## The escape hatch

Some phases genuinely change no code: a live proof run, a manual check, a decision written down.

They still keep the coverage block. One row:

```
| none - this phase changes no code | none | `no-unit-test - <what proves it instead>` | `<date>` |
```

**This row is only legal when the D4 command prints nothing.** If D4 prints a file, the phase did change code, and this row is a lie. Fix the row, not the command.

The `<proof>` must name something real - an evidence folder, a recorded run, a signed-off review. Not "checked manually".

### Non-Swift files

Shell scripts, `Package.swift`, resources and plists have no unit test. They still appear in D4's output, so they still need a row:

```
| `VisionCapture/scripts/deploy.sh` | none | `no-unit-test - run recorded in evidence/2026-08-15_deploy/` | `2026-08-15` |
```

`Package.swift` is a special case. Every new test file forces an edit to it, so it shows up in almost every phase. Its proof is **not** `swift build` alone - a package can build and still run zero tests. Its proof is a build **and** a test run with a non-zero executed count.

## What this gate does not prove

It proves a named test file exists, ran, and passed.

It does **not** prove the test exercises that code well. A test that asserts `true == true` passes D3.

There is no coverage tooling wired into this repo, so there is no number to check. The defence is the `What the test must prove` column in the implementation document's coverage plan. A reviewer reads that column, then reads the test, and judges. Write that column as a behaviour - "a queued note survives a cold restart" - never as a restatement of the file name.

If real coverage percentages are ever wanted, that is tooling work. It is not a change to this template.
