# Overview — MuckGarden

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.garden` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckGardenApp.swift`, `Sources/MuckGardenRoot.swift`, `Sources/MuckGardenState.swift` (299 lines) | file listing |

## Purpose

A mock garden-tracking app with a plant grid showing growth progress and a reorderable care-schedule list.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Plant grid | grid of plant cards with growth-progress bars navigating to a care schedule | MuckGardenRoot, MuckGardenPlant |
| Care schedule | a date picker and a reorderable list of care tasks that updates plant progress on completion | MuckGardenCareSchedule, MuckGardenCareTask |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `drag,progress,date`; navigation
`plant-grid-care-schedule`; motion `growth-spring`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckGardenState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

10 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.muckgarden.emptyCare.root` | MuckGardenRoot.swift:122 |
| `screen.muckgarden.error.root` | MuckGardenRoot.swift:84 |
| `screen.muckgarden.grid.root` | MuckGardenRoot.swift:36 |
| `screen.muckgarden.loading.root` | MuckGardenRoot.swift:76 |
| `screen.muckgarden.schedule.root` | MuckGardenRoot.swift:128 |
| `screen.muckgarden.success.root` | MuckGardenRoot.swift:80 |
| `ui.muckgarden.careDatePicker` | MuckGardenRoot.swift:97 |
| `ui.muckgarden.editOrderButton` | MuckGardenRoot.swift:127 |
| `ui.muckgarden.resetButton` | MuckGardenRoot.swift:33 |
| `ui.muckgarden.scheduleButton` | MuckGardenRoot.swift:67 |

Unstable — composed from runtime values, not reliable landmarks: `plant.buttonAccessibilityID`, `plant.rootAccessibilityID`, `task.buttonAccessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
