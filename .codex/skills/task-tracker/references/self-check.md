# Self-check

Run these before you hand the pair over. They take seconds and they catch the drift that turned the last tracker in this repo into 988 lines.

## Run the script, not the snippets

```sh
$(git rev-parse --show-toplevel)/.codex/skills/task-tracker/scripts/check-tracker.sh <work-item-folder>
```

Add `--base <sha>` to include the phase base commit in check 13.

It runs every check below and prints `PASS`, `FAIL` or `WARN` per line. It exits non-zero if anything failed. **A non-zero exit means you may not hand over.**

Copy-pasting the snippets by hand is how checks get skipped. Measured 2026-08-15 on `live-notch-work`: check 11 was in this file, nobody ran it, and the HTML page sat seven minutes behind the implementation document telling the owner a wrong number.

The snippets below stay so you can read what each check does and fix it when it is wrong. The script is what you run.

Run from the work-item folder. Set:

```sh
T=tracker-<slug>-<date>.md
REPO=$(git rev-parse --show-toplevel)
```

---

## 1. Size

```sh
wc -l "$T"
```

Compare against the budget, not against a fixed number:

```
60 + (30 x phases) + (1 x task)
```

Measured: 3 phases and 9 tasks is 160 lines. 5 phases and 20 tasks is about 230. 7 phases and 23 tasks is about 290.

**Expect** within about 15% of the budget. Far over means prose leaked in - move it to the implementation document. Do not fix it by deleting gate blocks.

Far **under** is also a warning: it usually means a phase is missing a block.

---

## 2. No per-task subsections

```sh
grep -n '^####' "$T"
```

**Expect nothing.**

A `####` heading under a task is how 25 lines per task happened last time. One task is one line.

---

## 3. Deleted concepts stay deleted

```sh
grep -ni 'exit criteria' "$T"
grep -n 'Definition of Done' "$T"
```

**Expect nothing from both.**

Exit criteria is gone - it always repeated phase acceptance. Per-task Definition of Done is gone - it belongs to the phase, and the phase block is called `Done when`.

---

## 4. The gate still exists

This is the important one. Checks 2 and 3 catch structure being **added**. This catches a gate block being quietly **dropped**.

```sh
grep -c '^## Phase '                "$T"
grep -c '^\*\*Unit test coverage\*\*' "$T"
grep -c '^\*\*Done when\*\*'          "$T"
grep -c '^\*\*Acceptance\*\*'         "$T"
```

**Expect all four numbers to be equal.** Every phase has all four blocks. No exceptions, including phases that change no code.

---

## 5. Five done-when lines per phase

```sh
grep -c '^- \[.\] D[1-5] ' "$T"
```

**Expect** five times the phase count.

If it is lower, someone deleted a line - almost always D3 or D4, the two that block. If it is higher, someone added a sixth. The five are fixed.

---

## 6. Nobody reworded the gate

```sh
grep -n '^- \[.\] D3 ' "$T" | sed 's/.*D3 //' | sort -u
grep -n '^- \[.\] D4 ' "$T" | sed 's/.*D4 //' | sort -u
```

**Expect exactly one distinct line from each.** D1 to D5 are copied word for word. A softened gate should show up as a git diff, not slip through as a paraphrase.

---

## 7. No paragraph smuggled into a row

```sh
grep -nE '^\|' "$T" | awk -F: 'length($0) > 200 { print $1": "substr($0,1,80)"..." }'
```

**Expect nothing.**

A 300-character table row is a paragraph wearing a table's clothes. Measured: the longest honest row in the worked example is 114 characters.

This checks **table rows only**. Do not run it over the whole file - the fixed D3 and D4 rule lines are legitimately long prose, and a whole-file version flags them every time. A check that cries wolf on a correct file teaches people to ignore checks.

---

## 8. Every named test file exists

```sh
grep -E '^\|' "$T" \
| awk -F'|' 'NF==6 { gsub(/[` ]/,"",$3); if ($3 ~ /\.swift$/) print $3 }' \
| sort -u \
| while read -r f; do
    [ -e "$REPO/VisionCapture/$f" ] || [ -e "$REPO/$f" ] || echo "MISSING FILE: $f"
  done
```

**Expect nothing.**

A coverage row naming a test file that does not exist is the worst failure this format can have - it looks covered and is not. If the test is planned but unwritten, the cell reads `none yet`, and the intended filename lives in the implementation document's coverage plan. Never write a filename you have not created.

---

## 9. The Now counters are true

```sh
grep -E '^\|' "$T" \
| awk -F'|' 'NF==6 {
    gsub(/[` ]/,"",$4)
    if ($4 ~ /^-+$/ || $4 == "Result") next
    if ($4 !~ /^[0-9]+\/[0-9]+pass$/ && $4 !~ /^no-unit-test/ && $4 !~ /^waived/) c++
  } END { print "coverage rows not passing: " c+0 }'

grep -c '^- \[!\] ' "$T"
```

**Expect** both numbers to match what the `Now` block says.

The awk reads only 4-column tables, which are the coverage tables. It deliberately does not grep the file for the words `missing` or `stale`, because those words also appear in the fixed D3 rules line - a plain grep over-counts and tells you a correct file is broken.

---

## 10. Every number is reproducible

Not a shell check - a read. Find every number in the tracker that is not a task ID, a date, or a coverage count.

For each one, ask: **which command produced this, and is that command written next to it in the implementation document?**

If the answer is "I do not know", delete the number. Say what you know instead.

The most common source of a wrong number is counting compiler errors with an unanchored grep - see `field-values.md`. It over-counts by more than double.

A quick cross-file check for the same claim written twice with different numbers:

```sh
grep -ohE '[0-9]+ (errors?|tests?|files?|sites?)' \
  tracker-*.md implementation-*.md implementation-*.html | sort | uniq -c | sort -rn
```

**Expect** each claim to appear with one number, not two. Two spellings of the same count means one document is lying.

---

## 11. The files agree

```sh
ls tracker-*.md implementation-*.md
ls implementation-*.html          # expect NOTHING - the page is not kept here
ls ../../human/<slug>.html        # the owner page
```

**Expect exactly two files in the work-item folder**, sharing the same slug and date, and **no `.html` at all**.

The owner page lives in `project-files/human/<slug>.html`. A page found inside the work-item folder is a failure, not a convenience: an agent listing the folder sees it, opens it, and spends about 8,000 tokens on a file written for a human. ADR 0052.

### The HTML page must not be stale

The page carries a stamp of the two Markdown files it was built from. Line 2 of the template:

```html
<!-- generated-from: tracker=<12 hex> implementation=<12 hex> - DO NOT DELETE -->
```

`scripts/build-page.py` writes it. If the stamp is missing or stale, the page was not rebuilt - run:

```sh
$REPO/.codex/skills/task-tracker/scripts/build-page.py <work-item-folder>
```

**Expect the stamp to match both files.** If it does not, the page was built from older Markdown - regenerate it.

The stamp is exact. Modification times are not: a `git checkout`, a `cp`, or a branch switch rewrites them and the page looks fresh when it is not. When a page has no stamp the script falls back to mtime:

```sh
stat -f '%Sm %N' -t '%H:%M:%S' tracker-*.md implementation-*.md implementation-*.html
```

**Expect the `.html` to be the newest of the three, or within a minute of them.**

If the HTML is older than either Markdown file, it is lying to the owner right now - it is the only one of the three they read. Update it before you hand over. This is not a nice-to-have: a stale page states blockers that are cleared, asks for decisions already made, and shows counts that no longer hold.

Seen on the first real use of this skill: the tracker and implementation document were updated to unblock a task, and the HTML sat untouched for two and a half hours still telling the owner the task was blocked and still asking him to choose between two options he had already resolved.

### Then, by eye

- Every phase heading in the tracker has a matching `## Phase <n>` anchor in the implementation document.
- Every coverage row's first two columns appear in that phase's coverage plan.
- The HTML page's stats match the tracker's board - phases, tasks done, coverage rows failing, needs-your-decision.
- Every task the tracker shows as unblocked is not described as blocked anywhere in the HTML.

---

## 12. The approval is real and not self-contradicting

The highest-stakes field in the whole format. `approved` is what tells every other agent it may start writing code.

```sh
grep -n '^status:\|^approved' "$T"
grep -in 'unapproved\|not approved\|awaiting.approval' tracker-*.md implementation-*.md implementation-*.html
```

**If the header says `status: approved`, the second command must print nothing.**

A file that says `approved` at the top and `still unapproved` in its changelog is lying in one of those two places, and a reader will believe whichever they see first.

A populated approval field is a claim even when the status is not `approved`:

```sh
grep -n '^approved_by:\|^approved_at:\|^approved_version:' "$T"
grep -n 'Owner said:' implementation-*.md
```

**If any of the three approval fields is not `null`, the second command must print a quote.** Recipe 0a requires it.

This matters after a version bump. `status: awaiting-approval` with `version: 2` and `approved_version: 1` is the correct shape - it records that version 1 was signed off and version 2 is not. But it only tells the truth if version 1 really was approved. With no quote, those three fields are an unverifiable claim that the owner said yes once, and the next reader has no way to test it.

Then, by hand:

- Does `approved_version` equal `version`? If not, work must stop until the new version is approved.
And the page must not argue with itself:

```sh
grep -m1 '^status:' "$T"
grep -oin 'approved\|awaiting approval' implementation-*.html | sort -t: -k2 | uniq -c -f1
```

**If the status is not `approved`, no line of the HTML may read "the plan is approved" or similar.**

Seen on the first real use: the page's header was corrected to *"awaiting approval"* while its closing block still read *"The plan is approved"* and *"approved at version 1 by Hebert"*. A reader gets a different answer depending on where they look, and the closing block is the part that tells them what to do next.

When you change an approval word, change **every** instance in the page - header label, meta line, stat row, and the closing block. Grep it, do not scroll it.

Seen on the first real use of this skill: a tracker carried `status: approved` / `approved_by: Hebert` in its header and four matching claims in the HTML page, while its own changelog still read *"still version 1, still unapproved"* - and the owner had not approved anything. Nobody had quoted a single word from him. This check takes one second and catches it.

**Never fix a contradiction here by guessing which side is true.** Ask the owner.

---

## 13. D4, per phase

Not a shape check - the real gate. Run it for any phase you are about to close. Command and reasoning: `coverage-gate.md`.

```sh
BASE=<the phase base commit>
PATHS=(VisionCapture/Sources VisionCapture/Resources VisionCapture/scripts VisionCapture/Package.swift)
cd "$REPO" && { git diff --name-only "$BASE"..HEAD -- "${PATHS[@]}"
                git diff --name-only            -- "${PATHS[@]}"
                git diff --name-only --cached   -- "${PATHS[@]}"
                git ls-files --others --exclude-standard -- "${PATHS[@]}"
              } | sort -u
```

**Expect** every path printed to appear in that phase's coverage table.

Two ways to break this command, both of which produce a silent green gate:

- **Shortening it to a single `git diff`.** This project leaves work uncommitted until the owner asks, so a lone `git diff` prints nothing and the gate passes on a phase with no tests at all.
- **Putting the paths in a plain string** (`P="a b c"` then `-- $P`). zsh does not split it. Measured 2026-08-15: the string version printed 0 files while the array version printed 5. Use the array. See `coverage-gate.md`.

Always run the guard afterwards:

```sh
cd "$REPO" && git status --porcelain -- "${PATHS[@]}" | head
```

**If D4 printed nothing and the guard prints something, D4 is broken.** Do not tick D4.

---

## 14. No stale `Blocked by:` line

```sh
grep -n '^Blocked by:' "$T"
```

For each one, look at the phase it sits in.

**FAIL** if the phase is `[x]`. A closed phase cannot still be blocked.

**WARN** if the phase has no `[!]` task *and* every coverage row in it passes. Nothing is stuck, so the line is stale - delete it (recipe 4).

`Blocked by:` on a phase held open by a failing gate is correct and stays. The worked example uses it that way.

Seen 2026-08-15: the page told the owner a task was blocked and asked him to choose between two options he had already resolved, for two and a half hours.

---

## 15. D1 is true - a closed phase holds no open task

```sh
grep -n '^Status: `\[x\]`' "$T"     # then read the tasks under each
```

**Expect** every task under a `[x]` phase to be `[x]` or `[-]`.

A `[~]`, `[!]` or `[s]` task under a closed phase means D1 was ticked without being checked.

`[s]` deserves its own look. It means the code was accepted but its live or manual proof never passed. The `Now` block does not count it - only `[!]` - so a phase can hold four `[s]` tasks and still read "Blocked: 0". The script prints the `[s]` count as a WARN so it is never invisible.

---

## 16. Every `waived` row is the owner's decision

`waived - <link>` passes D3. It is the one Result value that closes a gate on the owner's authority rather than on evidence, so it needs the same procedure `approved` got.

```sh
grep -E '^\|' "$T" | awk -F'|' 'NF==6 { gsub(/^[ `]+|[ `]+$/,"",$4); if ($4 ~ /^waived/) print $4 }'
awk '/^## Waivers/{f=1;next} /^## /{f=0} f' implementation-*.md
```

**Expect** one row in `## Waivers` per waived cell, each carrying a quoted `Owner said:` line.

**The approval quote does not count.** An approved tracker already has one `Owner said:` line at the top. Reusing it to justify a waiver means one sentence from the owner silently authorises every skipped test in the work item. Each waiver needs its own quote. Recipe 0b.

---

## 17. Every `no-unit-test` names a real proof

```sh
grep -E '^\|' "$T" | awk -F'|' 'NF==6 { gsub(/^[ `]+|[ `]+$/,"",$4); if ($4 ~ /^no-unit-test/) print $4 }'
```

**FAIL** if the proof names a path that does not exist.

**WARN** if the proof names no path at all. `no-unit-test - developer launch script, not shipped` is a *reason*, not a proof. The worked example shows the right shape: `no-unit-test - build plus test run, evidence/2026-08-13_phase1/`.

A count like `20/20` is not a path. The script only treats a token as a path when it starts with a letter.

---

## If a check fails

Fix the file. Never fix the check.

A command that keeps flagging a correct file is a broken command - report it and get it corrected here. A command that flags a real problem is doing its job, and the answer is never to loosen it.
