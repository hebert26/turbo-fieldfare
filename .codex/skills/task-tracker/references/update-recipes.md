# Update recipes

Exact edits for the moments that keep happening. Follow the recipe. Do not reformat, do not "tidy up", do not add a section.

The tracker grew to 988 lines in this repo during maintenance, not during creation. Maintenance is where formats die.

Set `$T` = the tracker, `$I` = the implementation document, `$H` = the owner page at `project-files/human/<slug>.html`.

`$T` and `$I` sit in the work-item folder. **`$H` does not, and nothing in `$T` or `$I` may ever link to it** - that is what keeps agents from spending context on a page written for the owner. ADR 0052.

---

## 0a. The owner approves

**Only the owner can move a tracker to `approved`. No agent may do it, ever, for any reason.**

Not because the plan looks finished. Not because the owner said something that sounded close. Not because work is obviously wanted. If you are reading this, you are not the owner.

When the owner has actually approved:

1. `$I` - under `## Approved scope`, add a line quoting **the owner's exact words** and where they said them:
   `Approved 2026-08-15. Owner said: "<their exact sentence>" (this session).`
2. `$T` header - `status: approved`, `approved_by`, `approved_at`, `approved_version` equal to `version`.
3. `$T` - changelog: one line.
4. `$T` - `next_action`: what happens first.
5. `$H` - update the scope card label and the stat row.

**If you cannot quote the words, you cannot set the field.** Write `status: awaiting-approval` and put the question to the owner instead. A tracker that falsely reads `approved` tells every other agent it has permission to write code.

Approval does not carry over. A new `version` needs a new approval, and the old `approved_version` stays behind so the header shows which version was last signed off.

Never leave the file self-contradicting - if the header says `approved`, no line anywhere in the three files may still say the work is unapproved. See check 12 in `self-check.md`.

---

## 0b. The owner waives a test

`waived - <link>` passes D3. It is the only Result value that closes a gate on authority instead of evidence, so it needs the same proof of authority that `approved` does.

**No agent may write `waived` on its own.** Not because the file is trivial. Not because a test would be awkward. Not because the phase is otherwise finished.

When the owner has actually waived it:

1. `$I` - under `## Waivers`, add one row: the date, the exact coverage row being waived, and **the owner's exact words**.
   `| 2026-08-15 | \`Sources/App/DebugProbe.swift\` | Owner said: "don't bother testing the debug probe, it never ships" |`
2. `$T` - that coverage row's Result becomes `waived - see ## Waivers in the implementation document`.
3. `$T` - changelog: one line.

**One waiver, one quote.** The approval quote at the top of the document does not authorise a waiver. If it did, one sentence from the owner would silently excuse every missing test in the work item.

If you cannot quote the words, the row is not `waived`. It is `missing`, and the phase stays open. Check 16 in `self-check.md`.

---

## 0. A phase starts

The most-skipped recipe, and the one D4 depends on.

1. `$I` - under the phase heading, write `Base commit:` and paste the current `git rev-parse HEAD`.
2. `$I` - record the disk baseline: `df -h /`, and the list of simulator devices and worktrees that already exist. The cleanup task at the end of the **work item** needs every phase's baseline to tell what this work created from what was already there. Every phase records one, including phases that write no code. See `cleanup.md`.
3. `$T` - phase strip: `Status: [~]`, set `Owner`.
4. `$T` - board row: `Status` to `[~]`.
5. `$T` - `Now` block: set phase, task, owner, and `Next`.
6. `$T` header - if this is the first phase to start, `status: in-progress`.
7. `$T` - changelog: one line.

Write the base commit **before the first line of code**. After code exists you cannot recover it honestly, and the phase falls back to the weak D4 check for the rest of its life.

---

## 1. Claim a task

1. `$I` - open the task section. Check `Depends on` and `Parallel safe`. If another agent holds a task that touches the same files and either is `Parallel safe: no`, stop and say so.
2. `$T` - task line: `[ ]` to `[~]`.
3. `$T` - `Now` block: task and owner.

Do not claim two tasks at once.

---

## 2. Finish a task

Implementation document first. The tracker copies from it.

1. `$I` - tick the task's acceptance detail boxes. Fill `Evidence` with what was actually observed, including real numbers.
2. Run the tests. Read the `Executed N tests` line - see `coverage-gate.md`.
3. `$T` - coverage table: update the `Result` and `Checked` cells for every row this task touched.
4. `$T` - task line: `[~]` to `[x]`, and add the date.
5. `$T` - board row: bump the `Tasks` count and the `Unit tests` cell.
6. `$T` - `Now` block: move to the next task.
7. `$T` - changelog: one line.
8. `$H` - if anything surprising happened, add a `t-dev` or `t-disc` entry by hand, then rebuild: `scripts/build-page.py <folder>`. Never hand-edit a phase entry or a prompt - the next build overwrites it.

If the code was accepted but the live or manual proof did not pass, the mark is `[s]`, not `[x]`. See `field-values.md`.

---

## 3. Block a task

1. `$T` - task line: `[!]`.
2. `$T` - under the phase, add or update the single `Blocked by:` line. Name what is stuck and who or what unblocks it. One line, 100 characters.
3. `$T` - `Now` block: bump the blocked count, and set `Next` to the next task that can actually run.
4. `$I` - if the block is a question, add it to `## Open questions`.
5. `$H` - if the owner has to decide, add a `t-human` entry and bump the "need your decision" stat.

The document status only becomes `blocked` when **no** task anywhere can move.

---

## 4. Unblock a task

1. `$T` - task line: `[!]` back to `[ ]` or `[~]`.
2. `$T` - delete the `Blocked by:` line if nothing else in the phase is stuck. Do not leave a stale one.
3. `$T` - `Now` block: drop the blocked count.
4. `$I` - if it was an open question, move it to `## Decisions` with the date and who decided.
5. `$H` - remove the `t-human` entry, or turn it into a `t-disc` recording what was decided.

---

## 5. Hand a task to someone else

1. `$T` - phase strip: change `Owner`.
2. `$T` - `Now` block: change the owner.
3. `$T` - changelog: one line naming both people.

If the new owner has a safety constraint - a session they must run in, a device they must use - it is already in `$I`'s `## Owners` table. Do not copy it into the tracker.

---

## 6. Work nobody planned for

1. Decide: is it inside the approved scope?
2. **Inside** - `$I` - add a task section under the right phase. `$T` - add a task line with the next free number. `$I` - add a row to `## Discovered work`. `$T` - changelog.
3. **Outside** - stop. `$I` - add it to `## Discovered work` marked `needs re-approval`. `$T` - bump `version`, set `status: awaiting-approval`, set `next_action`. Do not do the work.
4. If it changes what code the phase touches, add its files to the phase coverage table now, as `missing`. A phase that grows quietly is how D4 gets bypassed.

Never renumber existing tasks to make room. Append.

---

## 7. Close a phase

Run the checks before you tick anything.

1. D1 - every task in the phase is `[x]` or `[-]`. `[~]`, `[!]` and `[s]` fail. On the **last** phase this includes the cleanup task - run it now if it is still open, and read the "never delete" list in `cleanup.md` before you remove anything.
2. D2 - every acceptance box in the phase is ticked.
3. D3 - every coverage row passes, or is `no-unit-test` or `waived`.
4. D4 - run the command in `coverage-gate.md`. Every path it prints appears in the table.
5. D5 - the evidence path is written in `$I`.
6. Only when all five pass: `$T` - tick D1 to D5, set the phase strip `Status: [x]` and the `Done` date, update the board row.
7. `$T` - `Now` block: move to the next phase.
8. `$T` - changelog.
9. `$H` - rebuild: `scripts/build-page.py <folder>`. The badge, the gate marks, the counters and the prompts all follow the tracker. A closed task loses its prompt automatically.

If a check fails, stop. Write what failed on the `Blocked by:` line. A phase with a failing D3 or D4 is not done, no matter how finished the code looks.

---

## 8. Close the work item

1. Every phase is `[x]` or `[-]`. That already means the cleanup task in the last phase ran - it is the one task that gives the disk back, and D1 on that phase blocks without it. If you got here and the disk was never measured, the last phase was closed wrongly. Go back and do it.
2. `$T` header - `status: completed`, set `completed_at`.
3. `$T` - `Now` block: `Next: nothing. Complete.`
4. `$I` - `## Open questions` reads `None`, or every remaining one is recorded as accepted.
5. `$H` - the "need your decision" stat reads 0, or says what was accepted.
6. Move the folder from `project-files/active/` to `project-files/archive/`.
7. Move the owner page too: `project-files/human/<slug>.html` to `project-files/human/archive/<slug>.html`, so `human/` only ever shows live work. Then rebuild once so its links point at the archived Markdown.

---

## Never

- Never rewrite a changelog line. Append a correction that points at the wrong one.
- Never delete a task. Mark it `[-]` with a reason in `$I`.
- Never move prose into the tracker "just this once".
- Never tick a done-when box you did not check.
- Never update the tracker without rebuilding the HTML page in the same session. A page the owner reads that is a week behind is worse than no page. The rebuild is one command; there is no excuse.
- Never hand-edit a generated part of the page. Phase entries, gate marks, counters and prompts are rewritten on every build. Fix the Markdown and rebuild.
