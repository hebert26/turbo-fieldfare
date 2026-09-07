# Overview — MuckRide

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.ride` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckRideApp.swift`, `Sources/RideAppState.swift`, `Sources/RideModels.swift`, `Sources/RideScreens.swift` (309 lines) | file listing |

## Purpose

A mock ride-hailing app with a draggable offline pickup map, a vehicle carousel and picker, a ride request, and a ride-status sheet.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Pickup map | a drag-to-move pin on an offline canvas map with move-west/east button alternatives | RideMapCard |
| Vehicle selection | a horizontal carousel plus a menu picker of vehicles | RideRootScreen, VehicleCard, RideVehicle |
| Request and status | an async ride request and a status sheet to view or complete the ride | RideStatusPanel, RideStatusSheet |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `drag,picker,status-sheet`; navigation
`vehicle-carousel-status-sheet`; motion `ride-state-keyframes`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable RideAppState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

15 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.ride.root` | RideScreens.swift:45 |
| `screen.ride.status.root` | RideScreens.swift:179 |
| `ui.ride.empty` | RideScreens.swift:144 |
| `ui.ride.error` | RideScreens.swift:140 |
| `ui.ride.loading` | RideScreens.swift:136 |
| `ui.ride.pickup.dragCanvas` | RideScreens.swift:87 |
| `ui.ride.pickup.moveEastButton` | RideScreens.swift:95 |
| `ui.ride.pickup.moveWestButton` | RideScreens.swift:91 |
| `ui.ride.requestButton` | RideScreens.swift:37 |
| `ui.ride.status.completeButton` | RideScreens.swift:167 |
| `ui.ride.status.doneButton` | RideScreens.swift:176 |
| `ui.ride.statusButton` | RideScreens.swift:40 |
| `ui.ride.success` | RideScreens.swift:148 |
| `ui.ride.vehicleCarousel` | RideScreens.swift:24 |
| `ui.ride.vehiclePicker` | RideScreens.swift:32 |

Unstable — composed from runtime values, not reliable landmarks: `vehicle.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
