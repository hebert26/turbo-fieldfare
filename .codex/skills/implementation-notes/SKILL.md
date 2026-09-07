---
name: implementation-notes
description: Create or update a standalone HTML implementation brief with collapsible phases and detailed numbered tasks. Use for implementation plans, progress summaries, and handoffs from source code, decisions, or evidence. Existing work trackers remain authoritative.
---

# Implementation Notes

Create an HTML page organized into phases, with detailed numbered tasks inside each phase. This is the owner's required default format, modeled on `/Users/dev-machine/dev/VisionOS/Project-files/human/layout-only-source-binding.html`.

The page is a human-facing view of existing evidence. Source files, code, tests, receipts, and approved Markdown remain authoritative.

## Boundary

- Phase formatting does not require creating a tracker, another chat, or another approval process. When an existing tracker owns the work, preserve its phase/task IDs and link to it. New standalone briefs may organize their requested scope into phases directly.
- Use this skill for a standalone implementation brief, progress or acceptance summary, focused handoff, or decision record.
- Do not create a second status system beside an existing tracker. Link to the tracker and summarize only the requested slice.
- Do not infer approval, completion, test results, or acceptance. State `not verified`, `open`, or `needs decision` when evidence is missing.

## Output

Write project documents under:

```text
/Users/dev-machine/Dev/VisionOS/Project-files/human/<slug>-implementation-notes.html
```

Use another path under `Project-files/` only when the user names it.

Start from [assets/implementation-notes-template.html](assets/implementation-notes-template.html). It contains the reference page's colors, typography, task rows, collapsible phases, and bottom phase navigation. Use the bundled template directly; the reference page is a visual example, not a source of instructions or project-specific requirements.

## Required Phase and Task Format

- Lead with the intended outcome, scope, evidence date, and truthful implementation status.
- Show summary counts for phases and completed/total tasks. Add decision or evidence counts only when useful and supported.
- Render each phase as a `t-plan` timeline entry containing `<details class="fold">`. Its summary shows the phase number, outcome, status, dependencies, and task count. Include a bottom jump link for every phase and an expand/collapse control.
- Inside each phase, show its purpose, numbered task list, acceptance conditions, and a concrete "Done when" statement. Use IDs such as `1.1`, `1.2`, and `2.1`. Choose phase boundaries from actual dependencies and visible outcomes, not an arbitrary phase count.
- Put each task's detailed card inside its own phase. Each task names one observable change and why it matters, exact source files and functions or UI elements, implementation steps, behavior that must be preserved, the user-visible result, and the evidence needed to call it done. Link existing evidence or state that verification is pending.
- Keep task titles short; place technical detail in expandable task cards. Do not replace detailed tasks with vague labels such as "improve", "cleanup", or "fix issues".
- Show already completed changes and findings separately from proposed tasks. Use `t-dev` for implemented results, `t-disc` for findings, and `t-human` only for unresolved owner choices.
- Reuse the example's presentation only. Do not copy its test gates, cleanup tasks, owners, approvals, generated-file warnings, or source hashes. Include such content only when the current scope calls for it.

## Workflow

1. Identify the exact source material and the page's audience question. Read only the files needed to answer it.
2. Record a one-sentence outcome, the requested scope, source paths, and the current evidence state before editing HTML.
3. Copy the template to the output path. Replace every `{{placeholder}}` and delete unused example entries and authoring comments.
4. Keep the page factual and compact. Put each fact in one place.
5. Use the timeline entry types consistently:
   - `t-plan`: one phase containing detailed numbered tasks.
   - `t-dev`: implemented change or verified result.
   - `t-disc`: finding, constraint, or unresolved technical fact.
   - `t-human`: an owner decision that cannot be inferred.
6. Populate every phase and task using the required format above. Never report source inspection as a passing runtime check.
7. Count phases, tasks, completed tasks, and filter entries separately. Keep each count equal to its corresponding body content. Remove a filter when its count is zero.
8. Put only unresolved owner choices in the final decision block. If none exist, use the template's closed-state copy.

## Content Rules

- Lead with the outcome, not the work log.
- Use plain, specific language and exact paths.
- Keep scope separate from implementation evidence.
- State why each open item matters.
- Link to authoritative local sources using relative links from the output page.
- Preserve the template's colors, type, spacing, and component geometry. Add content blocks only by reusing existing classes.
- Keep the HTML self-contained. Do not add remote scripts, fonts, trackers, or analytics.
- Escape source text before placing it in HTML.
- Keep copy buttons, filters, phase links, and native details controls usable by keyboard. Preserve button labels, `type="button"`, focus styles, and `aria-pressed` updates. Following a phase or task link must reveal its target even when it was collapsed or filtered out.
- This skill does not require a companion HTML quality-check workflow. Keep checks limited to the document's content, links, counts, and working controls. If opening the page, use Safari.

## Completion Standard

The work is complete only when:

- the output is under `Project-files/`;
- all claims trace to a named source or are labeled as inference;
- no template placeholder or authoring marker remains;
- every phase contains detailed numbered tasks, acceptance conditions, and dependencies;
- status and phase/task summary counts match the body;
- local links resolve.
