# Document archive manifest

Folder index: [[turboCharge/Index]].

Batch: `/Users/dev-machine/Documents/Idea Home/turboCharge/archive/2026-09-15-document-cleanup`

Scope: authorized items under `/Users/dev-machine/Documents/Idea Home/turboCharge`, excluding the existing `archive` subtree. Follow-up authorizations explicitly included the complete mixed `cache-work` subtree and the repository `plan/` documentation folder. Files were moved by same-filesystem rename without deletion, duplication, or overwrite. Repository source, models, apps, and processes were untouched.

## Archived

| Original path | Archived path | Reason |
|---|---|---|
| `Project-files/human/qwen-ios-integration.html` | `Project-files/human/qwen-ios-integration.html` | Superseded 2026-09-06 MLX/no-requant Qwen plan; the current official-BF16 plan replaces it. |
| `Project-files/qwen-ios-integration.html` | `Project-files/qwen-ios-integration.html` | Compatibility symlink for the superseded Qwen HTML; still resolves within this batch. |
| `Project-files/qwen-ios-integration` | `Project-files/qwen-ios-integration` | Old plan-directory symlink. Its target was already absent before cleanup and remains a preserved baseline issue. |
| `Project-files/human/turbo-new-chat-memory-20260905.md` | `Project-files/human/turbo-new-chat-memory-20260905.md` | Completed, unreferenced New Chat memory check. |
| `Project-files/human/cache-work/` | `Project-files/human/cache-work/` | Entire mixed evidence/build/runtime tree archived by explicit follow-up authorization: 1.3 GB, 14,105 regular files, 2,705 directories, and 8 symlinks. |
| `Project-files/human/gemma-visioncapture-mcp-implementation-notes.html` | `Project-files/human/gemma-visioncapture-mcp-implementation-notes.html` | Explicitly authorized for archive in the follow-up. |
| `Project-files/human/mcp-concise-responses-and-turbo-context-implementation-notes.html` | `Project-files/human/mcp-concise-responses-and-turbo-context-implementation-notes.html` | Explicitly authorized for archive in the follow-up. |
| `Project-files/human/turbocharge-evaluation-goal.md` | `Project-files/human/turbocharge-evaluation-goal.md` | Explicitly authorized for archive in the follow-up. |
| `docs/expert-cache-implementation-notes.html` | `docs/expert-cache-implementation-notes.html` | Completed expert-cache experiment/retention decision. |
| `docs/assets/expert-cache-current-vs-proposed.png` | `docs/assets/expert-cache-current-vs-proposed.png` | Archived with its HTML dependency. |
| `docs/assets/expert-cache-lookup-planner.png` | `docs/assets/expert-cache-lookup-planner.png` | Unreferenced historical document asset in the self-contained docs bundle. |
| `docs/assets/turbofieldfare-metal-architecture-v2.png` | `docs/assets/turbofieldfare-metal-architecture-v2.png` | Unreferenced historical document asset in the self-contained docs bundle. |
| `docs/assets/turbofieldfare-metal-architecture.png` | `docs/assets/turbofieldfare-metal-architecture.png` | Unreferenced historical document asset in the self-contained docs bundle. |
| `/Users/dev-machine/dev/turbo-fieldfare-personal/plan/` | `repository/plan/` | Repository planning folder retired after its sole report became background to the current Qwen tracker. The complete folder was atomically renamed; the report's pre-move SHA-256 was `c32f05eaa84ee03ebb4d31bbcf204f58455b4f8efd1e98b36d791a9c9a03b740`. |

The archived expert-cache page and its raw evidence now share the batch-relative layout expected by its original `../Project-files/...` link. Two duplicate links in the archived concise-responses page were location-adjusted to the retained `Project-files/active/report.md`. Historical content and status were not changed.

## Retained current documents

- `Project-files/active/qwen3.6-35b-a3b-official/tracker-qwen3.6-35b-a3b-official-2026-09-15.md` — awaiting approval.
- `Project-files/active/qwen3.6-35b-a3b-official/implementation-qwen3.6-35b-a3b-official-2026-09-15.md` — proposed scope, not approved.
- `Project-files/human/qwen3.6-35b-a3b-official-implementation-notes.html` — current human-readable Qwen plan.

## Retained unresolved document

- `Project-files/active/report.md` — substantive Gemma investigation; report is complete but five measurements remain open.

## Baseline and link checks

- Current Qwen closure: 33 local links, 0 broken before cleanup and 0 broken after all archive passes. Its two live report references now target `repository/plan/report.md` in this archive.
- The three follow-up standalone documents had 204 local links and 189 baseline-broken links; moving their preserved relative tree introduced no current-Qwen link issue. The full `cache-work` subtree was renamed as one unit without following its 8 symlinks.
- Retained `Project-files/active/report.md`: 29 local links, 15 baseline-broken links, unchanged after cleanup.
- Other archived source documents: old Qwen had 3/3 broken local links at baseline because its paired Markdown folder was already absent; expert-cache had 13/15 broken local links at baseline; New Chat had no local links. The expert-cache link to its now-co-located raw evidence resolves.
- The archived repository report has 24 links: 10 external and 14 local repository source links. Its 14 formerly relative source links were updated after the move and all resolve.

## Read-only model-comparison finding

`Project-files/model-comparison/` contains only the empty directory `qwen3.6-35b-a3b/` (no regular files, hidden files, or symlinks). Its name and prior task provenance identify it as the former Idea Home staging location for Qwen analysis/download material; the official bundle now lives in repository scratch. Current Qwen documents contain no `model-comparison` reference. The directory was inspected only and was not moved or deleted.

## Missing at inventory time

- `Project-files/agent-planning/implementation-plan.html` was not present anywhere in the authorized tree, so the requested current Gemma public-plan file could not be retained or moved.
- `gemma-prompt.md` existed only inside `Project-files/human/cache-work/baseline-d40af7e/`; it moved unchanged with the explicitly authorized complete subtree.
- The old `Project-files/active/qwen-ios-integration/` Markdown folder was already absent; only its dangling compatibility symlink existed.

## Counts

- Archived selections: 6 standalone documents, 4 standalone document images, 2 top-level symlink aliases, 1 complete mixed subtree (1.3 GB; 14,105 regular files, 2,705 directories, 8 symlinks), and 1 complete repository documentation folder containing 1 report.
- Retained as current: 3 Qwen documents.
- Retained as unresolved: 1 report.
- Deleted: 0.
