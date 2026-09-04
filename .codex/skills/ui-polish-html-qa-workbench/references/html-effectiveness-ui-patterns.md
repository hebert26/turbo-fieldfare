# HTML Effectiveness Patterns For UI Work

Source inspiration: https://thariqs.github.io/html-effectiveness/

Use this reference when a UI task needs more than a code patch or a prose explanation. The point is not to make a pretty report. The point is to give the human and the agent a small browser-native surface where visual comparison, state review, and decisions are easier than reading a long markdown answer.

## Core Principle

Use HTML when the work is spatial, visual, comparative, interactive, or decision-heavy.

Markdown is fine for short notes. HTML is better when the user needs to see relationships:

- before vs after
- reference vs target
- one design direction vs another
- default vs hover vs selected vs focus
- component states across density, theme, and data size
- a flow, timeline, dependency map, or rollout path
- an interaction that must be felt, such as scroll, drag, hover, collapse, transition, or resize

Every useful HTML artifact should tighten the loop: the user points at what is wrong, the agent adjusts, and the artifact exports the decision back into the implementation or review.

## Artifact Types

### 1. Visual QA Workbench

Use for polish fixes, visual mismatches, spacing, alignment, navigation rhythm, row geometry, selected/hover/focus state bugs, and screenshot matching.

Required sections:

- problem statement in product language
- reference vs target comparison
- state matrix
- measurement table
- source evidence and file links
- fix hypothesis
- after screenshot or verification notes
- copyable acceptance criteria

Best for:

- "These cards should have the same padding as the cards on the other page"
- "This selected item looks wrong"
- "This page feels less polished than the reference page"
- "The modal spacing does not match the rest of the site"

### 2. Living Design System Sheet

Use when the agent needs to understand or clean up the site's or app's visual language before changing UI.

Render:

- color tokens as swatches
- type scale with actual text samples
- spacing tokens as visual bars
- radius and shadow samples
- icon sizes and button sizes
- density rules for navigation, tables, cards, and forms
- notes showing where each token came from in code

Best for:

- before a broad UI polish pass
- when agents keep guessing arbitrary pixel values
- when several components use similar but not identical colors or spacing
- when a new component should match existing surfaces

### 3. Component Variant Contact Sheet

Use when one component has many states or variants. Put all states on one sheet so wrong details become obvious.

Render:

- default
- hover
- selected/active
- focus-visible
- disabled
- loading
- error
- empty
- long text
- badge/count
- keyboard/drag state, when relevant
- compact and roomy density, if the component supports both

Best for:

- nav and list rows
- cards
- tabs/segmented controls
- table rows
- form rows
- buttons and icon buttons

### 4. Design Direction Fan-Out

Use before committing to a visual direction when the user is still deciding.

Render 2-4 real alternatives side by side. Each option should use the same real content so the comparison is honest.

Show:

- option name
- live mockup
- best use case
- tradeoffs
- what would change in code

Keep the options close enough to the real design system that any selected option can actually ship.

### 5. Interaction Sandbox

Use when the issue depends on feel, timing, or input behavior.

Render the smallest working prototype that lets the user try the interaction.

Good controls:

- duration slider
- easing selector
- density selector
- state toggle
- reduced-motion toggle
- sample data size selector

Best for:

- scroll behavior
- drag/reorder
- expand/collapse rows
- hover/focus timing
- state transitions
- resize behavior

### 6. Feature Flow Explainer

Use when UI polish depends on understanding the flow behind the screen.

Render:

- TL;DR
- flow diagram
- involved files
- event/state path
- failure or loading paths
- gotchas
- user-facing checkpoints

Best for:

- a multi-step page flow
- form submission and validation states
- auth or onboarding flows
- loading, error, and empty-state paths

### 7. UI PR / Review Artifact

Use after a meaningful UI change so reviewers know what to inspect.

Render:

- motivation
- before/after screenshots or schematics
- file-by-file tour with the reason for each change
- states covered
- accessibility notes
- performance notes
- exact review focus areas
- manual QA checklist

Best for:

- PR handoff
- review gates
- changes to shared UI primitives
- changes to navigation or layout behavior

### 8. Custom UI Decision Editor

Use when the user needs to sort, tune, or choose among many items.

The page should let the user interact, then export the result as markdown, JSON, CSS variables, a prompt, or an implementation checklist.

Best for:

- choosing navigation density
- sorting UI polish tasks
- tuning templates
- selecting design tokens
- grouping component variants
- ranking UX issues by severity

## Required Artifact Qualities

Every HTML artifact should be:

- self-contained
- readable by opening the file directly in a browser
- based on real app evidence where possible
- honest about schematic vs measured data
- useful at desktop and narrow widths
- restrained, calm, and aligned with the real styling of the surface
- built for scanning, comparison, and decision-making
- equipped with copy/export buttons when the artifact changes the next step

Avoid:

- generic dashboard decoration
- large hero sections
- gradients, glass, or visual effects that are not part of the real design
- fake metrics
- screenshots or measurements without labels
- walls of prose inside cards
- changing the design direction without showing alternatives

## Evidence Rules

Do not let the HTML artifact become fiction.

Prefer this evidence order:

1. Real screenshot from the app or site.
2. DOM measurements or computed styles from the rendered app.
3. Existing design tokens or CSS variables from source.
4. Nearby component implementation.
5. Clearly labeled schematic when the app cannot be run.

Always label measurements as:

- `measured`
- `from token`
- `from source`
- `inferred`
- `schematic`

## VisionCapture HTML Surfaces To Watch

Use extra care on these repo surfaces:

- the marketing site under `Website/` (home, pricing, features, docs, support, transparency, internal, product/mcp, product/architecture)
- product UI prototypes and muck-ups under `project-files/archive/muck-ups/`
- deliverable visuals under `project-files/visuals/`
- the view-describe surface at `VisionCapture/scripts/describe-viewer.html`

Small spacing errors matter most on repeated UI patterns because the user's eye learns the rhythm. If one section breaks that rhythm, the whole site feels less intentional.

## Export Blocks

Every non-trivial artifact should include at least one copyable export block:

- acceptance criteria
- implementation plan
- review checklist
- issue summary
- token changes
- selected option summary
- manual QA steps

This keeps the human in the loop and lets the artifact feed the next agent turn instead of becoming a dead-end document.

## Decision Rule

Create an HTML artifact when one of these is true:

- the user gave screenshots
- the issue is visual, spatial, or stateful
- there are multiple possible directions
- the component has several states
- the change touches shared UI primitives
- the task involves scrolling, dragging, animation, keyboard focus, or resize behavior
- the user is frustrated because previous agents missed the visual detail

Skip the artifact when the task is a tiny, unambiguous code-only fix and the rendered UI can be checked directly with a short note.
