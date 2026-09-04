# Overview — MuckFiles

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.files` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/FilesScreens.swift`, `Sources/MuckFilesApp.swift`, `Sources/MuckFilesState.swift` (277 lines) | file listing |

## Purpose

A mock file-browser app with a folder sidebar, a file list supporting rename and move, and a detail pane.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Folder browsing | a sidebar of folders and a breadcrumb-labeled file list | FilesRootView, FixtureFile |
| Rename and move | context menu and toolbar actions to rename or move a file | RenameSheet, FileDetailView |
| Reorder and load states | edit-mode reordering plus content, empty, loading, and error states | FilesState |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `selection,menu,rename`; navigation
`split-outline-breadcrumbs`; motion `drag-reorder`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable FilesState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

17 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.files.browser.root` | FilesScreens.swift:24 |
| `screen.files.detail.root` | FilesScreens.swift:117 |
| `screen.files.rename.root` | FilesScreens.swift:130 |
| `screen.files.sidebar.root` | FilesScreens.swift:20 |
| `ui.files.actionsMenu` | FilesScreens.swift:99 |
| `ui.files.breadcrumbs` | FilesScreens.swift:59 |
| `ui.files.moveButton` | FilesScreens.swift:92 |
| `ui.files.moveMenuButton` | FilesScreens.swift:79 |
| `ui.files.refreshButton` | FilesScreens.swift:93 |
| `ui.files.renameButton` | FilesScreens.swift:90 |
| `ui.files.renameCancelButton` | FilesScreens.swift:134 |
| `ui.files.renameField` | FilesScreens.swift:128 |
| `ui.files.renameMenuButton` | FilesScreens.swift:77 |
| `ui.files.renameSaveButton` | FilesScreens.swift:138 |
| `ui.files.reorderButton` | FilesScreens.swift:102 |
| `ui.files.showEmptyButton` | FilesScreens.swift:95 |
| `ui.files.showErrorButton` | FilesScreens.swift:97 |

Unstable — composed from runtime values, not reliable landmarks: `file.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
