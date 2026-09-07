---
name: knowledge-doc
description: Generate knowledge base documents for VisionCapture. Creates guides, specs, investigations, and diagram docs in `docs/knowledge/`, following established conventions (emoji H1, See-also blockquote, Related footer, MOC registration). Use when a user wants to document a system, add a knowledge article, write a guide, create a diagram doc, or add an investigation report. Triggers on "document this", "add to knowledge base", "write a guide for", "create a knowledge doc", "/knowledge-doc".
---

# Knowledge Doc

Generate standardized knowledge base documents for VisionCapture. Every document follows the conventions established in `docs/knowledge/` — emoji-prefixed titles, See-also cross-links, Related footers, and MOC registration.

## When This Skill Activates

**Creating Knowledge Articles**:
- "Document this system..."
- "Add a knowledge doc for..."
- "Write a guide for..."
- "Create a spec for..."

**Adding Diagrams**:
- "Create a diagram for..."
- "Add a Mermaid diagram showing..."
- "Draw the flow for..."

**Investigation Reports**:
- "Write up the investigation for..."
- "Document what we found about..."

**MOC Updates**:
- "Add a new MOC for..."
- "Update the modes MOC..."

**Manual Activation**:
- User explicitly says "use knowledge-doc skill"
- User invokes `/knowledge-doc`

---

## Knowledge Base Location

All documents go in:

```
docs/knowledge/
```

The entry point is `knowledge/index.md`. MOCs live in `knowledge/mocs/`. Topic documents live in topic subdirectories (`architecture/`, `workflows/`, `mcp-modes/`, `diagrams/`, etc.).

---

## Document Types

| Type | Directory | Naming Pattern | Example |
|------|-----------|----------------|---------|
| Guide | `<topic>/` | `<topic-name>.md` | `mcp-modes/mcp-modes-guide.md` |
| Spec | `architecture/` | `<system>-<version>-spec.md` | `architecture/mcp-modes-0.2-spec.md` |
| Investigation | `architecture/` | `INVESTIGATION-<topic>-<YYYY-MM-DD>.md` | `architecture/INVESTIGATION-execute-modes-2026-02-17.md` |
| Diagram (Mermaid) | `diagrams/` | `<topic>-mermaid.md` | `diagrams/scout-mode-full-process-mermaid.md` |
| Diagram (ASCII) | `diagrams/` | `<topic>-ascii.md` | `diagrams/request-flow-ascii.md` |
| MOC | `mocs/` | `<topic>.md` | `mocs/workflows.md` |
| Standalone | `knowledge/` (root) | `<topic>.md` | `codex-config-practical-guide.md` |

**Naming rules:**
- All filenames are **lowercase**, words separated by hyphens
- Diagram files **must** append `-mermaid.md` or `-ascii.md`
- Investigation files are prefixed with `INVESTIGATION-` (uppercase) and date-stamped: `INVESTIGATION-<topic>-<YYYY-MM-DD>.md`
- MOC files use simple topic nouns: `workflows.md`, `modes.md`
- No undated investigation files — all others are undated unless tracking a delivery

---

## Reference Files

For templates (Guide, Investigation, Diagram, MOC), see `references/templates.md`.

For formatting rules (visual conventions 1-11, scene composition diagrams, writing voice, MOC emoji reference), see `references/formatting-rules.md`.

---

## Workflow

### 1. Gather Information

Determine from the user's request:
- **Topic** — what system/feature/concept to document
- **Document type** — guide, spec, investigation, diagram, or MOC
- **Subdirectory** — where it belongs (`architecture/`, `diagrams/`, `mcp-modes/`, `workflows/`, or a new topic folder)

If the user has already provided this in their message, skip asking.

### 2. Read Existing Context

Before writing:
1. **Read `knowledge/index.md`** — understand the current document inventory and link resolution table
2. **Read the relevant MOC** — understand what already exists for this topic
3. **Read related documents** — avoid duplicating content that already exists

### 3. Create the Document

Write the document following the templates in `references/templates.md` and the formatting rules in `references/formatting-rules.md`. Place it in the correct subdirectory.

### 4. Update the MOC

Add an entry for the new document in the appropriate MOC file (`mocs/<topic>.md`). If no suitable MOC exists and the document warrants one, create a new MOC.

### 5. Update the Index

Add the new document to `knowledge/index.md`:
- Add to the **Document Inventory** table
- Add to the **Link Resolution** table (if wiki-links are used)
- Update the **Quick Navigation** section if the new doc changes entry points

### 6. Confirm Output

Summarize what was created, where it lives, and which MOC/index entries were updated.

---

## Validation Checklist

Before delivering, verify:

- [ ] Document follows the correct template for its type
- [ ] **H1 has emoji prefix** matching document type
- [ ] **Bold subtitle** immediately after H1
- [ ] **See-also blockquote** present (non-MOC documents)
- [ ] **Horizontal rules** (`---`) between all H2 sections
- [ ] **H2 headers have section-level icons** from the icon table (rule 10)
- [ ] **Related section** at bottom with MOC links table
- [ ] **Footer line** present: `*Part of the [VisionCapture Knowledge Base](../index.md)*`
- [ ] **All internal links use relative Markdown paths** — no `[[wiki-links]]`
- [ ] **All file paths and code in backticks** or fenced code blocks
- [ ] **Tables used for structured data** — no bullet-list alternatives
- [ ] **Filename follows naming convention** — lowercase, hyphenated, correct suffix
- [ ] **Placed in correct subdirectory** for document type
- [ ] **Parent MOC updated** with entry for new document
- [ ] **index.md updated** — document inventory table entry added
- [ ] **No external metaphors or analogies** — all explanations grounded in project domain
- [ ] **Investigation docs are date-stamped** in filename: `INVESTIGATION-<topic>-<YYYY-MM-DD>.md`
- [ ] **Diagram docs append format suffix**: `-mermaid.md` or `-ascii.md`
- [ ] **Mermaid diagrams use standard color coding** (green/red/purple/blue/yellow)
- [ ] **Key terms bolded**, code in backticks, paths in backticks
- [ ] **Inline Mermaid diagrams** included where they help explain scene/view connections (rule 11)
- [ ] **Inline diagrams have a "Reading this diagram" blockquote** after the code fence

If any check fails, fix the document before delivering.
