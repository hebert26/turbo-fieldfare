---
name: ui-polish-html-qa-workbench
description: Use for HTML/CSS UI polish on the VisionCapture marketing site, product UI prototypes, viewer surfaces, and visual artifacts — visual matching, screenshot-based implementation, spacing/alignment repair, selected/hover/focus-state geometry, color consistency, list/row rhythm, component state review, and design-system adherence. Trigger when a UI looks off, should match another page or screen, needs the same padding/alignment/rhythm, has wrong selected or hover states, or when agents keep missing small visual details despite clear instructions.
---

# UI Polish HTML QA Workbench

## Purpose

Use this skill to make UI work evidence-based instead of guess-based.

Agents often fail at small UI polish because they translate the request into code too early. They hear "add padding" and edit one CSS value, while the user actually means "make this part of the interface belong to the same visual system as the rest of the site or app."

For non-trivial UI polish, create a temporary self-contained HTML QA workbench that makes the visual comparison inspectable.

Template:

```text
assets/ui-qa-workbench-template.html
```

Deeper artifact patterns:

```text
references/html-effectiveness-ui-patterns.md
```

Read the reference when the work is broad UI improvement, design-system cleanup, component-state review, interaction prototyping, or "HTML effectiveness" style output.

Recommended output path:

```text
project-files/ui-qa/<screen-or-feature>-visual-qa.html
```

## HTML Effectiveness Rule

Use HTML when the problem is easier to see, compare, try, or decide than to explain in prose.

HTML artifacts are most useful for:

- visual QA workbenches
- living design-system sheets
- component variant contact sheets
- design direction fan-outs
- interaction sandboxes
- feature flow explainers
- UI PR or review artifacts
- custom decision editors with export buttons

Do not make HTML for its own sake. Make it when it helps the user stay in the loop and helps the agent avoid vague visual guesses.

## Core Rule

Solve the picture before solving the code.

Before editing, identify:

- the visible object that is wrong
- the existing UI that is the correct reference
- the full visual object that must move or change together
- the layout owner that controls that object
- the states that must be checked after the fix

Do not call a UI task complete from source inspection alone when a rendered check is possible.

## Workflow

### 1. Translate The Request Into Visual Acceptance Criteria

State the visual contract in plain product language.

For list rows, navigation items, tabs, buttons, cards, tables, and form rows, include the relevant checks:

- row/container inset
- icon left edge
- text start
- selected background inset
- hover background inset
- focus ring geometry
- row height
- vertical rhythm
- section label hierarchy
- hit area or click target
- badge/count alignment
- disclosure indicator alignment
- child indentation, if nested

### 2. Find The Reference Pattern

Inspect both the broken target and the correct existing pattern.

Required comparison targets:

- target broken view
- nearest correct existing view
- default state
- selected state
- hover state, when possible
- focus/keyboard state, when relevant
- empty/loading/disabled states, when relevant

Do not assume similar-looking UI uses the same component or stylesheet. Confirm whether the target and reference share a component, duplicate styles, or use separate code paths.

### 3. Build Or Update The HTML QA Workbench

For non-trivial UI work, copy the template into an artifact file and fill it with the current task's evidence:

```bash
mkdir -p project-files/ui-qa
cp .claude/skills/ui-polish-html-qa-workbench/assets/ui-qa-workbench-template.html project-files/ui-qa/<screen-or-feature>-visual-qa.html
```

The workbench should include:

- problem statement in product language
- reference vs target comparison
- screenshots when available, or clearly labeled schematics when not
- source evidence: screenshots, DOM measurements, computed styles, token files, or component files
- state matrix for default, hover, selected, focus, disabled/error as applicable
- measurement table for x-position, row height, spacing, radius, and color token
- design tokens actually used by the site or app, or inferred from nearby code
- fix hypothesis naming the layout owner to change
- patch summary after implementation
- verification checklist
- copyable acceptance criteria
- copyable follow-up prompt, issue summary, or review checklist when useful

The workbench is a review artifact, not production UI. Keep it calm, readable, and based on the real visual language of the surface you are fixing.

Label evidence clearly:

- `measured`
- `from token`
- `from source`
- `inferred`
- `schematic`

### 4. Change The Layout Owner

Change the element that owns the whole visual object, not cosmetic children.

Usually correct:

- row container padding/inset
- shared list-row component or class
- selected/hover background wrapper
- focus-ring wrapper
- design token used by comparable rows

Usually wrong:

- padding only the text label
- margin only the icon
- padding only the section wrapper
- moving the selected background separately from the hit area
- hardcoding an arbitrary value without checking the reference
- changing the correct reference component instead of the broken target

### 5. Move Complete Objects Together

For row/list/navigation fixes, these must move as one object when relevant:

- selected background
- hover background
- focus ring
- icon
- label
- badge/count
- disclosure indicator
- hit area/content shape
- drag/drop affordance

A fix is incomplete if the text aligns but the selected pill, hover state, focus ring, or click target still starts from the wrong place.

### 6. Verify The Rendered UI

Open the affected HTML file in a browser, or run the surface that renders it, and navigate to the exact affected screen.

Check:

- before/after screenshot or direct visual inspection
- reference screen against target screen
- default state
- selected state
- hover state, when possible
- focus/keyboard state, when relevant
- responsive behavior if the affected area can resize

If rendered verification is impossible, say that clearly and list the exact manual checks still needed.

## Why Agents Miss These Issues

Use this diagnosis when a user is frustrated that a clear UI instruction was missed:

- The agent solved the wording instead of the picture.
- The agent edited code before comparing the visible target with the visual reference.
- The agent treated "padding" or "alignment" as a single numeric value instead of a relationship between UI objects.
- The agent moved a child element but not the row background, focus ring, or hit area.
- The agent inspected the wrong component because two screens looked similar but used different implementations.
- The agent skipped rendered verification and trusted that the CSS change worked.
- The agent treated tiny spacing issues as cosmetic, even though repeated UI patterns make small mistakes highly visible.

Correct the failure by making the visual comparison explicit and verifiable.

## Where This Applies In This Repo

Use this skill on VisionCapture's real HTML surfaces, for example:

- the marketing site under `Website/` (home, pricing, features, docs, support, transparency, internal pages, and the product/mcp and product/architecture pages)
- product UI prototypes and muck-ups under `project-files/archive/muck-ups/` and deliverable visuals under `project-files/visuals/`
- the view-describe surface at `VisionCapture/scripts/describe-viewer.html`

For multi-page work, preserve the shared header/footer, nav, and responsive behavior that already exist across a surface. Visual polish must not make navigation, layout, or cross-page consistency worse.

## Example: Matching Card Rhythm Across Two Pages

User request:

```text
The pricing cards look tighter than the feature cards on the Features page. They should share the same vertical rhythm and the same selected/active border treatment.
```

Correct interpretation:

```text
The pricing cards do not visually belong to the same card system as the feature cards. The complete card object must align with the reference pattern: padding, radius, border, title/price block, and any active or highlighted state.
```

Required checks:

- Compare Pricing cards against Features cards.
- Identify whether the cards share the same component/class or use separate CSS.
- Move the shared card token or container, not only the title or price text.
- Verify the active/selected border starts at the same inset as comparable cards.
- Verify hover/focus geometry and click target still match the card.

## Final Response Shape

When using this skill, finish with:

1. **Visual mismatch** - what was wrong on screen.
2. **Reference pattern** - what existing UI defined the right result.
3. **Implementation** - what layout owner changed.
4. **States covered** - selected, hover, focus, hit area, and responsive checks where relevant.
5. **HTML QA workbench** - path to the artifact, if created.
6. **Verification** - what was visually checked and what remains uncertain.

## Completion Standard

Do not mark the task complete until the work has:

- identified the visible mismatch
- found the correct reference pattern
- defined visual acceptance criteria
- changed the layout owner that controls the full visual object
- checked relevant states
- reused or matched existing design tokens/components
- visually verified the rendered result or stated why it could not
- updated the HTML QA workbench or final checklist with evidence
