# Overview — MuckHotel

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.hotel` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckHotelApp.swift`, `Sources/MuckHotelState.swift`, `Sources/MuckHotelViews.swift` (298 lines) | file listing |

## Purpose

A mock hotel-booking app with filterable room search, a room list supporting comparison, and a booking sheet.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Search filters | check-in/out date pickers, guest and minimum-bed steppers, and a grid/list layout picker | HotelFilters |
| Room list and compare | a room list with a two-room compare toggle and a choose action | HotelRooms, HotelRoomCard |
| Booking | a sheet to enter a guest name and confirm the booking with validation | HotelBookingSheet |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `dates,guests,filters`; navigation
`grid-room-booking-wizard`; motion `gallery-hero-transition`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckHotelState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

13 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.hotel.booking.root` | MuckHotelViews.swift:121 |
| `screen.hotel.rooms.root` | MuckHotelViews.swift:53 |
| `screen.hotel.root` | MuckHotelViews.swift:16 |
| `ui.hotel.bedsStepper` | MuckHotelViews.swift:34 |
| `ui.hotel.bookButton` | MuckHotelViews.swift:118 |
| `ui.hotel.cancelButton` | MuckHotelViews.swift:114 |
| `ui.hotel.checkInPicker` | MuckHotelViews.swift:26 |
| `ui.hotel.checkOutPicker` | MuckHotelViews.swift:28 |
| `ui.hotel.displayPicker` | MuckHotelViews.swift:38 |
| `ui.hotel.guestNameField` | MuckHotelViews.swift:107 |
| `ui.hotel.guestsStepper` | MuckHotelViews.swift:30 |
| `ui.hotel.refreshButton` | MuckHotelViews.swift:51 |
| `ui.hotel.statusBanner` | MuckHotelViews.swift:13 |

Unstable — composed from runtime values, not reliable landmarks: `room.chooseAccessibilityID`, `room.compareAccessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
