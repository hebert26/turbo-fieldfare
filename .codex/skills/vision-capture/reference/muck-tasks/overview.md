# Overview — MuckTasks

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.tasks` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckTasksApp.swift` (262 lines) | file listing |

## Purpose

A mock to-do app with a filterable, searchable task list with swipe actions, a task detail editor, and a new-task composer sheet.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Task list | a filter picker and a searchable list with swipe-to-complete and swipe-to-delete | TasksRootView, TaskRow |
| Task detail | an editable title, project, priority, due date, notes, and a completed toggle | TaskDetailView |
| Task composer | a sheet to create a task with a required title | TaskComposerView |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `list,search,swipe`; navigation
`navigation-stack`; motion `standard`; accessibility profile
`P`; reset mode `relaunch`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

Plain SwiftUI @State array of TaskItem in TasksRootView, no reset hook found in this file. Reset mode is `relaunch` (`apps.tsv`).

## Source-declared accessibility landmarks

11 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.tasks.composer.root` | MuckTasksApp.swift:243 |
| `screen.tasks.list.root` | MuckTasksApp.swift:114 |
| `ui.tasks.addButton` | MuckTasksApp.swift:124 |
| `ui.tasks.composer.cancelButton` | MuckTasksApp.swift:248 |
| `ui.tasks.composer.dueDatePicker` | MuckTasksApp.swift:234 |
| `ui.tasks.composer.notesEditor` | MuckTasksApp.swift:240 |
| `ui.tasks.composer.priorityPicker` | MuckTasksApp.swift:232 |
| `ui.tasks.composer.projectField` | MuckTasksApp.swift:226 |
| `ui.tasks.composer.saveButton` | MuckTasksApp.swift:256 |
| `ui.tasks.composer.titleField` | MuckTasksApp.swift:224 |
| `ui.tasks.filterPicker` | MuckTasksApp.swift:91 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
