# Overview — MuckGarage

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.garage` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/GarageModels.swift`, `Sources/GarageScreens.swift`, `Sources/GarageStore.swift`, `Sources/MuckGarageApp.swift` (204 lines) | file listing |

## Purpose

A mock vehicle-service app with a vehicle picker, a service-health gauge, a service checklist/timeline, and alert toggles.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Vehicle and health | a vehicle picker and a service-health gauge | GarageRootView |
| Service checklist | a timeline plus a list of service items with an async complete action | GarageTimeline, GarageStatusPane |
| Alerts | a local-alerts toggle and a dismissible service note alert | GarageStore |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `gauge,toggle,checklist,alert`; navigation
`vehicle-tabs-service-timeline`; motion `dial-status-motion`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable GarageStore, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

17 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `garage.alertsToggle` | GarageScreens.swift:19 |
| `garage.cancelButton` | GarageScreens.swift:89 |
| `garage.clearButton` | GarageScreens.swift:38 |
| `garage.empty` | GarageScreens.swift:34 |
| `garage.error` | GarageScreens.swift:93 |
| `garage.finishButton` | GarageScreens.swift:88 |
| `garage.fixtureNotice` | GarageScreens.swift:14 |
| `garage.gauge` | GarageScreens.swift:17 |
| `garage.loading` | GarageScreens.swift:91 |
| `garage.resetButton` | GarageScreens.swift:46 |
| `garage.root` | GarageScreens.swift:48 |
| `garage.serviceAlertButton` | GarageScreens.swift:43 |
| `garage.serviceAlertDismissButton` | GarageScreens.swift:52 |
| `garage.status.empty` | GarageScreens.swift:94 |
| `garage.success` | GarageScreens.swift:92 |
| `garage.timeline` | GarageScreens.swift:76 |
| `garage.vehiclePicker` | GarageScreens.swift:12 |

Unstable — composed from runtime values, not reliable landmarks: `item.completeAccessibilityID`, `item.rowAccessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
