# Docs And Visuals Reference

Use this for durable docs, plans, visual explainers, diagrams, dashboards, slide-style pages, and knowledge work.

## Durable Docs

- Plans, specs, architecture notes, and durable docs belong under:
  `/Users/dev-machine/Dev/VisionOS/Project-files/`
- Use the active work folder when the task belongs to a current initiative.
- Keep docs grounded in real VisionCapture files, modules, actors, and data flow.
- Do not create durable project docs outside `Project-files/` unless the user explicitly asks.

## Knowledge Lookup

Start from:

- `/Users/dev-machine/Dev/VisionOS/Project-files/knowledge/index.md`
- fallback: `/Users/dev-machine/Dev/VisionOS/docs/knowledge/index.md`

Do not bulk-read the knowledge base. Use the index to find the specific doc needed.

## Diagrams

Project diagram rule:

- standalone HTML files with embedded Mermaid JS;
- dark theme `#0d1117`;
- color-coded nodes;
- never plain markdown mermaid blocks for durable project diagrams.

## Visual Explainers

Use this style when the user asks for a visual page or explainer:

- self-contained HTML under the requested path, or under `Project-files/` when no path is given;
- distinctive typography and deliberate light/dark theme;
- readable on desktop and narrow widths;
- zoom controls for Mermaid diagrams when useful;
- no invented app-specific automation examples.

## Research

For external APIs and package behaviour:

- prefer primary sources and official docs;
- state version/date when it matters;
- separate known facts from inference;
- say what needs local proof when docs are unclear.
