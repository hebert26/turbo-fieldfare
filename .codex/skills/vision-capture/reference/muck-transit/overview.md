# Overview — MuckTransit

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.transit` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckTransitApp.swift`, `Sources/TransitAppState.swift`, `Sources/TransitModels.swift`, `Sources/TransitScreens.swift` (241 lines) | file listing |

## Purpose

A mock public-transit app with a route list, a station timeline showing direction and progress, and a service-notice alert.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Route list | a split list of fixture routes navigating to a timeline | TransitRootScreen, TransitRoute |
| Station timeline | a direction picker and a stop list with a progress bar and a service disclosure | TransitTimelineScreen |
| Start route | an async start action with a loading/success/error status and a service-notice alert | TransitStatusPanel |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `picker,disclosure,alert`; navigation
`route-list-station-timeline`; motion `route-progress`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable TransitAppState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

12 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.transit.empty.root` | TransitScreens.swift:28 |
| `screen.transit.routes.root` | TransitScreens.swift:20 |
| `screen.transit.timeline.root` | TransitScreens.swift:88 |
| `ui.transit.alert.okButton` | TransitScreens.swift:91 |
| `ui.transit.timeline.directionPicker` | TransitScreens.swift:56 |
| `ui.transit.timeline.error` | TransitScreens.swift:109 |
| `ui.transit.timeline.loading` | TransitScreens.swift:105 |
| `ui.transit.timeline.progress` | TransitScreens.swift:66 |
| `ui.transit.timeline.serviceDisclosure` | TransitScreens.swift:72 |
| `ui.transit.timeline.serviceNoticeButton` | TransitScreens.swift:83 |
| `ui.transit.timeline.startRouteButton` | TransitScreens.swift:79 |
| `ui.transit.timeline.success` | TransitScreens.swift:113 |

Unstable — composed from runtime values, not reliable landmarks: `route.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
