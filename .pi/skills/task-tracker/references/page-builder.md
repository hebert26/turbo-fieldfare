# The page builder

The owner's HTML page is **generated**, not written, and it does **not** live in the work-item folder.

```
project-files/active/<slug>/tracker-<slug>-<date>.md          agents read this
project-files/active/<slug>/implementation-<slug>-<date>.md   agents read this
project-files/human/<slug>.html                               the owner reads this
```

Nothing in the two Markdown files points at the page. The builder finds it from the `slug:` field in the tracker header, and so can you. That one-way arrangement is the point: an agent that lists the work-item folder never learns the page exists. **ADR 0052.**

The page keeps links *back* to the two Markdown files. That direction is safe - a human clicks them, and an agent is never in there to follow them.

```sh
$REPO/.codex/skills/task-tracker/scripts/build-page.py <work-item-folder>
```

Rewrites `project-files/human/<slug>.html` in place, creating it from the template on the first run. Add `--out FILE` to write somewhere else - use that when you want to show a change before replacing what the owner is reading.

Run it every time either Markdown file changes. Recipe 2 and recipe 7 in `update-recipes.md`.

## Why a script and not a template to fill in

The page used to be filled in by hand from the same two files the tracker holds. Every hand-copy is a chance to drift, and drift is what the whole format exists to stop.

Measured on the first real work item: the page's Tasks box for phase 1 was a 49-word paragraph re-telling six task lines that the tracker already stored one line each. The seven boxes averaged 27 words. Re-telling is work, and it is work that goes stale.

So the builder copies. A task line on the page is the task line from the tracker. A gate mark is computed from the tracker. Nothing is typed twice.

## What it generates

| Part | Where it comes from |
|---|---|
| Task lines, with marks and dates | the tracker, copied |
| Acceptance boxes | the tracker, copied |
| `n of m coverage rows pass` | counted from the tracker |
| D1&ndash;D5 marks | computed: D1 from open tasks, D2 from unticked acceptance, D3 from coverage rows. D4 and D5 show `?` because they need a command nobody has run |
| Waits-on line | the `Needs` column, plus what depends on this phase |
| Cleanup rows | the one cleanup task, found by searching - not assumed to be the last number |
| Prompts | one per open task, for phases that can start today |
| Stats and chip counts | counted, never typed |
| `generated-from` stamp | sha of both Markdown files, for self-check 11 |
| The stop banner | the first thing in the file, telling any agent that opened it to close it again |

## What it never touches

These are judgment, and a script has none:

- the header, the goal sentence and the approved-scope card;
- each phase's one-line lede - the sentence that says what the phase is *for*;
- every `t-dev`, `t-disc` and `t-human` entry.

Write those by hand, once. The builder reads them out of the current page and puts them back unchanged.

## It is safe to run twice

Everything generated sits between `<!-- BUILD:phases -->` and `<!-- BUILD:prompts -->` markers, so a second run replaces its own output instead of stacking another copy on it. Verified: runs 1, 2, 3 and 4 produce byte-identical files.

A page written before the builder existed has no markers. The first run finds the phase entries structurally, replaces them, and writes the markers in. From then on it uses the markers.

**A bug worth remembering.** The first version pulled each phase's lede with a regex that searched the whole document. On the second build it ran past the end of the phase entry and took a paragraph out of a discovery entry instead - so all seven phases got the same wrong sentence, and the last one went blank. The fix is to cut the phase's own entry block out first, then look inside it. **Never let a pattern that reads one entry span into the next.**

## The prompts

One runnable prompt per **open** task. A finished task gets none, so the section shrinks as the work item closes.

Only for phases that can start today: the phase in progress, plus any phase whose `Needs` are all `[x]`. A phase still waiting on another cannot be worked on, so a prompt for it is noise. The page says which phases are waiting and why.

Each prompt carries the task detail **inlined**. That is deliberate: the implementation document on the first real work item was 78 KB, about 19,600 tokens. Pointing a model at it to find one 753-byte task section wastes context that the model then does not have for the work. The prompt is 220 to 580 tokens and does not grow when the tracker does.

The prompt shape, all of it copied from the two Markdown files:

```
GOAL       what this task delivers
SOURCE     the file and the "#### <id>" heading it came from
TOUCHES    the files, from "Touches" - change nothing else
DEPENDS    task IDs, and whether it is parallel safe
CONTEXT    the task's own detail, inlined
DONE WHEN  the acceptance detail, verbatim
VERIFY     the exact command, plus what a pass looks like
EVIDENCE   where to write the result
RULES      follow AGENTS.md; do not commit; markdown wins
```

**Shell commands are never wrapped.** A wrapped command is a broken command the moment somebody pastes it. Everything else wraps at 76 characters.

Two rules for prompts:

- **Regenerate, never edit.** A hand-edited prompt drifts from the task it came from, which is the same failure as a stale page, one level down.
- **Durable rules do not belong in a prompt.** The git policy, the test-target rule and the app-agnostic rule live in `AGENTS.md`. Repeating them in forty prompts buries them. The prompt points at `AGENTS.md` and restates only the one line that is worth the space: do not commit.

## The stop banner is the weakest layer, on purpose

Every generated page opens with a comment telling agents not to read it, naming the two files they should read instead.

**Be honest about what this buys.** By the time an agent sees that banner, the file is already in its context and the tokens are already spent. A banner cannot undo a read.

It earns its place for three narrower things:

- a partial read (`head`, an offset read) stops early;
- someone about to hand-edit the page learns that the next build overwrites it;
- a human opening the file understands where it came from.

The layer that actually works is the **location**. The banner is a backstop, not the defence.

## What still leaks

A repository-wide grep still finds the page:

```sh
grep -r "task 1.4" project-files/
```

Nothing short of a tool-level block stops that, and the owner has accepted it. Keep the page out of the work-item folder and the common case - listing or globbing the folder you are working in - never surfaces it.

## If it refuses to run

| Message | What to do |
|---|---|
| `expected exactly one tracker` | the folder has two, or none. Fix the folder. |
| `no <div class="timeline" id="timeline">` | the page is not from the template. Delete it and rerun - the builder reseeds from the template. |
| `cannot find where the phase entries stop` | a hand-written page with no non-phase entries. Add the two `BUILD:phases` markers by hand, once, then rerun. |
| `WARNING: no cleanup task found` | the last phase does not end with a cleanup task. See `cleanup.md`. |
| `a page still sits in the work-item folder` | there are two pages and one of them lies. Delete the one inside the work item; the real one is in `project-files/human/`. |
| `not inside a project-files/ tree` | the work item is somewhere unexpected, so the builder cannot work out where `human/` is. |
