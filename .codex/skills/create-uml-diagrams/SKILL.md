---
name: create-uml-diagrams
description: "Create accurate diagrams for all 14 UML types from requirements, code, architecture notes, workflows, or observed behavior. Use when Codex must choose, create, explain, compare, or revise Activity, Use Case, Sequence, State Machine, Communication, Interaction Overview, Timing, Class, Object, Component, Composite Structure, Deployment, Package, or Profile diagrams in Mermaid, PlantUML, or another requested format without inventing system behavior."
---

# Create UML Diagrams

Create the smallest diagram that makes the requested idea clear. Treat a diagram as an evidence-backed model, not as decoration: inspect the supplied requirements, code, documents, or runtime evidence before naming system elements or relationships.

## Workflow

1. **Clarify the modeling question.** Identify what the reader needs to understand: scope and responsibilities, workflow, message order, lifecycle, domain structure, module boundaries, deployment, or another concern. If the task does not say, infer only from the supplied material and state the inference.
2. **Collect evidence.** Read the relevant source files and local project instructions. For repository work, inspect the actual code and relevant architecture decisions before describing behavior. Distinguish implemented, planned, unsupported, and assumed behavior.
3. **Choose the diagram type.** Use the decision guide in [references/uml-types.md](references/uml-types.md). All 14 types are supported; use [references/uml-templates.md](references/uml-templates.md) for the type-specific construction checklist and starter pattern. Prefer one focused diagram unless the user asks for a complete set or separate views answer genuinely different questions. Do not create a class diagram merely because the task mentions “architecture.”
4. **Set scope and level.** Define the system boundary, audience, abstraction level, and whether the view is conceptual, logical, runtime, or deployment-oriented. Omit unrelated detail that makes the main relationship harder to see.
5. **Model only supported facts.** Use names, nodes, actors, messages, states, operations, and dependencies found in the evidence. Mark unresolved items as assumptions or open questions. Never add a fake provider, button, service, state, integration, or success path to make the drawing look complete.
6. **Select the notation.** Use Mermaid for supported, lightweight documentation diagrams and quick iteration. Use PlantUML or another explicitly requested format when UML semantics such as lifelines, guards, stereotypes, ports, deployment nodes, or profile extensions need to be expressed precisely. Do not label a non-UML flowchart as a UML diagram.
7. **Generate the artifact.** Provide the diagram source in a fenced code block with the correct language tag. If the user asks for a saved artifact, write it to the requested location and follow local repository conventions. If a rendered image or HTML page is requested, render or build it only through a real supported path.
8. **Validate before presenting.** Check syntax when a renderer is available, then check semantic completeness, direction of arrows, decision and merge behavior, state transition triggers, message ordering, relationship multiplicity, and legibility. Confirm that every important node has a purpose and every important edge has a reason.

## Creating all 14 diagrams

When the user asks for “all diagrams,” “the complete UML set,” or an equivalent request:

1. Build one shared inventory of actors, goals, actions, participants, events, states, types, instances, components, interfaces, packages, nodes, artifacts, stereotypes, constraints, and relationships from the evidence.
2. Produce one named view for each of the 14 types in the reference taxonomy. Reuse the shared inventory so names and relationships do not drift between views.
3. Use the type-specific template and checklist in [references/uml-templates.md](references/uml-templates.md). Do not silently replace a requested type with a generic flowchart, architecture diagram, or another UML type.
4. Mark a view **not applicable**, **insufficient evidence**, or **planned, not implemented** when the source material cannot support that view. Explain exactly what is missing. Do not populate a view with fictional components, states, messages, infrastructure, or constraints.
5. Run a cross-view consistency review: actors and use cases must agree with scope; sequence and communication messages must agree with participants; states and transitions must agree with lifecycle evidence; classes, objects, components, packages, nodes, artifacts, and stereotypes must agree with the inspected design.
6. Keep each view readable. A complete UML set may be 14 separate files or clearly separated sections, depending on the requested output format. Give every view a stable title and type label.

## Output contract

Return, in this order:

- **Purpose:** one sentence describing the question the diagram answers.
- **Selected type:** the UML type and a short reason it fits better than nearby alternatives.
- **Evidence boundary:** what was inspected and which elements are inferred, if any.
- **Diagram source:** Mermaid, PlantUML, or the format requested by the user.
- **Notes:** assumptions, omitted detail, unresolved decisions, and validation status.

When the task is underspecified and different diagram types would materially change the result, ask what the real behavior or audience should be. If a useful first pass is still safe, make the smallest evidence-backed assumption and label it clearly.

## Type selection rules

Use these practical defaults, then confirm them against the full taxonomy in the reference file:

- Requirements and system scope → **Use Case**.
- A business or technical workflow with branching, parallel work, or roles → **Activity**.
- Time-ordered calls, events, API requests, or service interactions → **Sequence**.
- Entity or object lifecycle with event-triggered transitions → **State Machine**.
- Domain entities, interfaces, attributes, operations, and relationships → **Class**.
- Concrete instances and values at one moment → **Object**.
- Replaceable modules, services, interfaces, and dependencies → **Component**.
- Runtime hosts, devices, containers, artifacts, and communication paths → **Deployment**.
- Logical code organization and dependency direction → **Package**.
- Timing constraints, durations, deadlines, or protocol state over time → **Timing**.
- A high-level map that embeds or references several detailed interactions → **Interaction Overview**.
- A relationship-focused interaction view where spatial connectivity matters more than chronology → **Communication**; prefer Sequence when both fit.
- Internal parts, ports, and connectors of one classifier or component → **Composite Structure**.
- Domain-specific UML extensions, stereotypes, tagged values, and constraints → **Profile**.

## Modeling and readability rules

- Keep the system boundary explicit when actors or external systems are involved.
- Use verb phrases for use cases and actions; use nouns for classes, objects, components, packages, and states.
- Put time direction and participant order where the notation expects them; do not encode chronology only in prose.
- Label important conditions, guards, protocols, cardinalities, and failure paths instead of relying on implication.
- Use stable, concise identifiers in diagram source. Put explanatory prose in notes or a legend.
- Prefer semantically meaningful relationships over dense crossing arrows. Split an overloaded diagram into named views when a single page becomes unreadable.
- Keep optional and alternative paths visibly distinct from the normal path.
- Treat examples as examples. Do not copy example labels or architecture from the reference page into the user's system.

## Tool and repository boundaries

Follow the current repository's `AGENTS.md`, `CONTEXT.md`, architecture decisions, and document-location rules when creating diagrams inside a project. Do not modify deprecated or out-of-scope tools just to produce a diagram. If the requested output format, renderer, or destination is unavailable, report that plainly and provide the verified source format instead of pretending that it rendered.

## Reference

Read [references/uml-types.md](references/uml-types.md) when selecting among less-common UML types, explaining notation, or checking that a diagram matches the requested purpose. The taxonomy is based on Visual Paradigm's guide to the 14 UML diagram types: https://www.visual-paradigm.com/guide/the-complete-guide-to-uml-diagrams-all-14-types-explained-with-practical-examples/

Read [references/uml-templates.md](references/uml-templates.md) when creating any diagram, creating the complete 14-type set, or deciding whether the requested renderer can express a UML concept without approximation.
