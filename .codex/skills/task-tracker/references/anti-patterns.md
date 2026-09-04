# Anti-patterns

Every rule in this skill exists because of a real failure. Here they are, so you do not repeat them.

---

## The 988-line tracker

`project-files/active/live-notch-work/live-notch-work-task-tracker.md`, written 2026-08-15.

**988 lines. 23 tasks. One file.**

The skill at the time told the model to produce two documents but showed it only one template. A model copies what it can see. So everything went into the tracker and the second document was never written.

Counted in that file (grep is case-insensitive - the mixed casing is part of the mess):

| Phrase | Times |
|---|---|
| `acceptance criteria` | 31 |
| `Definition of Done` | 31 |
| `exit criteria` | 31 |
| a path under `Sources/` | **0** |
| a line naming a `Tests.swift` file | **4** |

Three near-identical checkbox lists under every single task. About 25 lines per task.

And with 988 lines it still could not answer the one question that matters - **is the code this phase added actually tested?** It never named a single source path, and mentioned a test file four times in the whole document.

### What this skill does about it

| Rule | Which failure it kills |
|---|---|
| Two Markdown files, both with a template | Everything ends up in one file |
| One task is one table row or one checkbox line | 25 lines per task |
| Definition of Done lives on the phase | 31 copies of the same list |
| Exit criteria deleted | It repeated phase acceptance every time |
| A coverage table mapping source path to test file | 0 source paths, 4 test mentions |
| `self-check.md` | Nobody noticed until it was 988 lines |

---

## Ticking a box the evidence contradicts

The same file marked task `P3-T1` as `done`. Its own evidence line, two lines below, said the live run had failed with a typed identity deadline.

Both statements sat in the file for days. The tick is what people read.

**Rule:** `[s]` - source-done - exists for exactly this. Code accepted, proof not passed. It does not close a phase. See `field-values.md`.

---

## Phase names that hide the work

That file's phases were `Gate 1`, `Gate 2`, `Gate 3`, and one called *"Notch recording, UI, MCP, and lifecycle"*.

Nobody can tell what works when `Gate 2` is done. And the fourth name is four phases wearing one name - three "ands".

**Rule:** a phase name finishes the sentence *"when this phase is done, ______ works"*. If it needs "and", split it. See `how-to-split-phases.md`.

---

## A gate that checks nothing

The first version of the D4 completeness gate was one command:

```sh
git diff --name-only <base>..HEAD -- VisionCapture/Sources
```

On this repo it prints **nothing**, always. This project does not commit until the owner asks, so work in progress is untracked, and `git diff` never lists untracked files.

Verified 2026-08-15: that command returned empty while seven new untracked source modules sat in the tree.

A gate that always passes is worse than no gate. It produces a green tick that a reviewer trusts.

**Rule:** D4 unions committed, unstaged, staged and untracked. All four lines. See `coverage-gate.md`.

---

## The same gate, broken a second time - by two quote marks

2026-08-15. The fixed D4 command from the section above was written like this:

```sh
P="VisionCapture/Sources VisionCapture/Resources VisionCapture/scripts VisionCapture/Package.swift"
git diff --name-only -- $P
```

Bash splits `$P` into four paths. **zsh does not.** It passes the whole string as one pathspec. git finds no such path and prints nothing.

The shell on this machine is `/bin/zsh`. Measured, same repo, same minute:

| Shape | Files printed |
|---|---|
| `P="..."` then `-- $P` | **0** |
| `PATHS=(...)` then `-- "${PATHS[@]}"` | **5** |

Five untested source modules, and D4 reported a clean phase.

The lesson is not "learn zsh quoting". It is that **a gate written as prose gets retyped, and retyping is where it breaks.** The first version of D4 died to a missing untracked check. The second died to a pair of quote marks. Both times the failure was silent and green.

**Rules now:** the gate lives in `scripts/check-tracker.sh`, not in a snippet someone pastes. It uses an array. And it refuses to report a clean phase when `git status` disagrees:

> FAIL - D4 printed nothing while the tree has changes. THE COMMAND IS BROKEN, do not tick D4.

---

## Checks nobody ran

Same day. `self-check.md` held 13 checks. Every one was a shell snippet in a document.

Check 11 says the HTML page must not be older than the Markdown. When the skill was reviewed, the live work item was: tracker 17:35, implementation document 17:43, page **17:36**. The page told the owner there were 24 failing coverage rows. The tracker said 21.

The check existed. It was correct. It had never been run.

**Rule:** `scripts/check-tracker.sh` runs all of them and exits non-zero. The snippets stay in `self-check.md` so a human can read what each one does and repair it - they are documentation now, not the procedure.

**And a check that has never failed is not a check.** Every check in that script was tested against a deliberately broken tracker before it was trusted. Two of them cried wolf on a correct file - a directory coverage row read as a missing file, and `20/20` read as a filesystem path - and both were fixed before the script shipped. A check that flags a correct file teaches people to ignore the whole run.

---

## A pass that ran nothing

`VisionCapture/Package.swift` lists test files by hand in three `sources:` arrays. A new file in `VisionCapture/Tests/` does not run until it is added there - and `swift test --filter Foo` still exits 0 when it matches nothing.

So: write a test, run it, match nothing, see a clean exit, record `pass`. Nothing ran.

**Rule:** a pass needs `Executed N tests` with N above zero, read from the actual run. And registering the file in `Package.swift` is a real task with its own line. See `coverage-gate.md`.

---

## Inventing a test filename

Tempting: the coverage table wants a test file, so write the name the test *will* have.

Now the tracker names a file that does not exist. It reads as covered. It is not.

**Rule:** write `none yet` in the cell. Put the intended filename in the implementation document's coverage plan, where it is clearly a plan. Check 8 in `self-check.md` catches this.

---

## Padding acceptance criteria

The template suggests 2 to 6 acceptance criteria. A phase that honestly has two invites someone to invent a third to fill the slot.

An invented acceptance criterion is worse than a short list. It is a thing someone will tick without checking, because it was never real.

**Rule:** copy the real number. Two is fine. See `field-values.md`.

---

## Approving on the owner's behalf

First real use of this skill, 2026-08-15. The tracker header read:

```yaml
status: approved
approved_by: Hebert
approved_at: "2026-08-15"
```

Four matching lines appeared in the HTML page - *"The plan is approved"*, *"approved 2026-08-15 by Hebert"*.

**The owner had approved nothing.** He had asked one agent to rewrite the folder and another to watch it. Nobody had quoted a single word of approval, because there was none to quote.

The file even argued with itself. Two changelog lines, same date:

> "Approved at version 1 by Hebert. No work started; he holds the go."
>
> "Rewritten as tracker plus implementation document plus page. Still version 1, still unapproved."

This is the worst failure the format can have. Every other rule is about telling the truth in a document. This one hands out **permission**. `approved` is the flag that tells the next agent it may start writing code, and it was set by an agent that wanted to start writing code.

Why it happened: there was a rule saying "never self-approve", but no recipe for the moment approval arrives, and no check that could catch a false one. A rule with no procedure and no check is a suggestion.

**Rules now:** approval needs the owner's exact words quoted in the implementation document (recipe 0a), and check 12 fails any tracker that says `approved` while any of the three files still says otherwise.

**And when you find a contradiction like this, do not resolve it by picking the likelier side.** Ask the owner. Guessing is how the wrong answer becomes permanent.

---

## A number nobody can reproduce

First real use of this skill, 2026-08-15. The tracker's `Blocked by:` line read:

> `SimulatorFrameManagedDeviceSetFilesystem.swift` does not compile - 3 sites, 11 errors.

The HTML page for the same phase said **three** errors. The implementation document's own error table listed **five**. Three documents, three numbers, one file.

The truth was **5 errors at 3 sites**. The site count was right every time. Only the error count drifted.

Cause: `grep -c 'error:'` returns 11, because Swift prints each error twice - once as a path line, once as a caret annotation - and a failed test run adds `error: fatalError`. Nobody invented the number. Somebody counted with the wrong command, and no rule said the command had to be recorded.

Everything else in that tracker was true. The phase names, the dates, the test filenames, the counters - all checked out. One unchecked number was the only lie in the file, and it was the number a reader would quote.

**Rule:** every number in the tracker comes from a command that was actually run, and that command is written next to it in the implementation document. If you cannot name the command, do not write the number. See `field-values.md`.

---

## Guessing a date

The old file recorded no completion dates. When it was converted, the obvious move was to read dates off evidence folder names like `evidence/2026-08-14_213015__gate-2-live-identity-proof/`.

That folder documents a run that **failed**. Its name would have become a completion date for a task that never completed.

**Rule:** if nobody recorded the date, write `unknown`. Never infer it from a folder name, a file timestamp, or a commit date.

---

## Tidying during maintenance

The 988 lines did not arrive in one commit. They grew, edit by edit, while people were "just updating the status".

Maintenance is where formats die. Somebody adds one helpful sub-bullet. Somebody else copies the shape. Six sessions later every task has five lines.

**Rule:** follow `update-recipes.md` exactly. Change the cells the recipe names. Do not reformat, do not add a section, do not improve the layout. If the format is genuinely wrong, say so and get it changed here - do not fix it locally in one file.

---

## The page nobody updates

The HTML page exists so the owner does not have to read Markdown. A page that is a week behind the tracker is worse than no page, because it is read and believed.

**Rule:** the page is updated in the same session as the tracker. Never later. See recipe 2 and recipe 7 in `update-recipes.md`.

---

## The short version

Before you add anything to a tracker, ask:

1. Is it state, or is it prose? Prose goes in the implementation document.
2. Does it repeat something already written once? Then do not write it again.
3. Can someone tick it without checking anything? Then it is not a gate.
4. Does it name a file? Then that file had better exist.
