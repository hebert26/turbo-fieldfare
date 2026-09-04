---
name: implementation-notes
description: Create or update a standalone, human-readable HTML implementation-notes page from code, specifications, decisions, test evidence, or existing Markdown. Use for implementation briefs, progress summaries, acceptance handoffs, and decision records that are easier to review visually. Do not use it as the source of truth for a tracked multi-phase work item; use task-tracker for that.
---

# Implementation Notes

Create a concise HTML page that tells the owner what was requested, what changed, what evidence supports it, what remains open, and what needs a decision.

The page is a human-facing view of existing evidence. Source files, code, tests, receipts, and approved Markdown remain authoritative.

## Boundary

- Use `task-tracker` when the request needs phases, task state, coverage gates, or work across sessions. Its generated owner page already uses this visual system.
- Use this skill for a standalone implementation brief, progress or acceptance summary, focused handoff, or decision record.
- Do not create a second status system beside an existing tracker. Link to the tracker and summarize only the requested slice.
- Do not infer approval, completion, test results, or acceptance. State `not verified`, `open`, or `needs decision` when evidence is missing.

## Output

Write project documents under:

```text
/Users/dev-machine/Dev/VisionOS/Project-files/human/<slug>-implementation-notes.html
```

Use another path under `Project-files/` only when the user names it.

Start from [assets/implementation-notes-template.html](assets/implementation-notes-template.html). The template preserves the visual system of `Project-files/human/live-notch-acceptance.html` without copying that generated page's project-specific content.

## Workflow

1. Identify the exact source material and the page's audience question. Read only the files needed to answer it.
2. Record a one-sentence outcome, the requested scope, source paths, and the current evidence state before editing HTML.
3. Copy the template to the output path. Replace every `{{placeholder}}` and delete unused example entries and authoring comments.
4. Keep the page factual and compact. Put each fact in one place.
5. Use the timeline entry types consistently:
   - `t-plan`: intended work, scope, or acceptance contract.
   - `t-dev`: implemented change or verified result.
   - `t-disc`: finding, constraint, or unresolved technical fact.
   - `t-human`: an owner decision that cannot be inferred.
6. Every implementation entry names the visible outcome, exact files or components changed, and verification evidence. Never report source inspection as a passing runtime check.
7. Keep the summary counts equal to the visible entries. Remove a filter when its count is zero.
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
- Keep copy buttons and filters usable by keyboard. Preserve button labels, `type="button"`, focus styles, and `aria-pressed` updates.

## Completion Standard

The work is complete only when:

- the output is under `Project-files/`;
- all claims trace to a named source or are labeled as inference;
- no template placeholder or authoring marker remains;
- status and summary counts match the body;
- local links resolve.
