# Overview — MuckJournal

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.journal` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckJournalApp.swift`, `Sources/MuckJournalRoot.swift`, `Sources/MuckJournalState.swift` (338 lines) | file listing |

## Purpose

A mock journaling app with a day-selector timeline of entries and an entry editor with mood and tag pickers.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Timeline | day buttons filtering a list of entries with an empty-state placeholder | MuckJournalRoot, MuckJournalEntry |
| Entry editor | a title field, a body text editor, a mood grid, and tag toggles with validation on save | MuckJournalEditor |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `editor,mood-grid,tags`; navigation
`timeline-calendar-editor`; motion `insertion-fade`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckJournalState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

12 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.muckjournal.calendar.root` | MuckJournalRoot.swift:56 |
| `screen.muckjournal.editor.root` | MuckJournalRoot.swift:122 |
| `screen.muckjournal.empty.root` | MuckJournalRoot.swift:20 |
| `screen.muckjournal.error.root` | MuckJournalRoot.swift:113 |
| `screen.muckjournal.success.root` | MuckJournalRoot.swift:32 |
| `screen.muckjournal.timeline.root` | MuckJournalRoot.swift:47 |
| `ui.muckjournal.bodyEditor` | MuckJournalRoot.swift:91 |
| `ui.muckjournal.cancelButton` | MuckJournalRoot.swift:118 |
| `ui.muckjournal.newEntryButton` | MuckJournalRoot.swift:41 |
| `ui.muckjournal.resetButton` | MuckJournalRoot.swift:43 |
| `ui.muckjournal.saveButton` | MuckJournalRoot.swift:119 |
| `ui.muckjournal.titleField` | MuckJournalRoot.swift:87 |

Unstable — composed from runtime values, not reliable landmarks: `entry.buttonAccessibilityID`, `entry.rootAccessibilityID`, `identifier`, `moodAccessibilityID(mood`, `tagAccessibilityID(tag`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
