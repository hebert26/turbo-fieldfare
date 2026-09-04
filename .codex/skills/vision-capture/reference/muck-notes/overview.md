# Overview — MuckNotes

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.notes` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckNotesApp.swift`, `Sources/MuckNotesScreens.swift`, `Sources/MuckNotesState.swift` (253 lines) | file listing |

## Purpose

A mock notes app with a folder outline, a note editor with tagging, and an inspector panel in a three-column split view.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Folder and note list | an outline of folders plus a flat list of notes | MuckNotesRoot, NoteFolder |
| Note editor | a body text editor and an add-tag menu | NoteEditor |
| Inspector | a read-only tags and details panel for the selected note | NoteInspector |
| Fixture state menu | load, empty, error, and reset controls | NotesFixtureControls |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `outline,editor,menu`; navigation
`split-folders-note-inspector`; motion `card-expansion-crossfade`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckNotesState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

17 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.notes.editor.root` | MuckNotesScreens.swift:111 |
| `screen.notes.folders.root` | MuckNotesScreens.swift:47 |
| `screen.notes.inspector.root` | MuckNotesScreens.swift:129 |
| `ui.notes.addImportantTagButton` | MuckNotesScreens.swift:99 |
| `ui.notes.addPersonalTagButton` | MuckNotesScreens.swift:97 |
| `ui.notes.bodyEditor` | MuckNotesScreens.swift:94 |
| `ui.notes.emptyButton` | MuckNotesScreens.swift:138 |
| `ui.notes.errorButton` | MuckNotesScreens.swift:139 |
| `ui.notes.fixtureMenu` | MuckNotesScreens.swift:142 |
| `ui.notes.folderOutline` | MuckNotesScreens.swift:24 |
| `ui.notes.loadButton` | MuckNotesScreens.swift:137 |
| `ui.notes.loading` | MuckNotesScreens.swift:11 |
| `ui.notes.resetButton` | MuckNotesScreens.swift:140 |
| `ui.notes.retryButton` | MuckNotesScreens.swift:15 |
| `ui.notes.successBanner` | MuckNotesScreens.swift:52 |
| `ui.notes.tagMenu` | MuckNotesScreens.swift:101 |
| `ui.notes.validationText` | MuckNotesScreens.swift:103 |

Unstable — composed from runtime values, not reliable landmarks: `note.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
