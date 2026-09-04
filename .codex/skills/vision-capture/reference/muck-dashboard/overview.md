# Overview — MuckDashboard

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.dashboard` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckDashboardApp.swift`, `Sources/MuckDashboardModels.swift`, `Sources/MuckDashboardRoot.swift` (322 lines) | file listing |

## Purpose

A mock analytics dashboard with a filterable record list, a detail panel with a bar-style chart, and refresh actions that can surface empty or error states.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Filtered record list | a sidebar of views filtering a content list of records | MuckDashboardRoot, DashboardFilter |
| Record detail | a detail panel with a bar-style chart and a refresh action | DashboardDetail |
| Empty states | separate empty-content and empty-detail placeholders with a refresh error message | DashboardEmptyContent, DashboardEmptyDetail |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `tables,charts,filters,toolbars`; navigation
`adaptive-split-dashboard`; motion `controlled-refresh`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckDashboardState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

10 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.dashboard.root` | MuckDashboardRoot.swift:58 |
| `state.dashboard.empty` | MuckDashboardRoot.swift:90 |
| `state.dashboard.emptyDetail` | MuckDashboardRoot.swift:97 |
| `state.dashboard.emptyMessage` | MuckDashboardRoot.swift:82 |
| `state.dashboard.refresh` | MuckDashboardRoot.swift:138 |
| `ui.dashboard.detailRefreshButton` | MuckDashboardRoot.swift:147 |
| `ui.dashboard.emptyShowOverviewButton` | MuckDashboardRoot.swift:87 |
| `ui.dashboard.ownerColumnToggle` | MuckDashboardRoot.swift:21 |
| `ui.dashboard.refreshButton` | MuckDashboardRoot.swift:118 |
| `ui.dashboard.toolbarRefreshButton` | MuckDashboardRoot.swift:63 |

Unstable — composed from runtime values, not reliable landmarks: `filter.accessibilityID`, `record.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
