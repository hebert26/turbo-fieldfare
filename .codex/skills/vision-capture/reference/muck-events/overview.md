# Overview — MuckEvents

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.events` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckEventsApp.swift`, `Sources/MuckEventsState.swift`, `Sources/MuckEventsViews.swift` (365 lines) | file listing |

## Purpose

A mock event-program app with a schedule list, a seat-map picker, and a ticket-issuing screen with a share action.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Program schedule | a list of program slots with an info popover | EventProgram, EventDetailsPopover |
| Seat selection | a seat grid over a stage canvas plus a contact form | EventSeats |
| Ticket issuing | issues a ticket after validation and offers a share action | EventTicket, TicketKeyframe |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `canvas,share,popover,form`; navigation
`tabs-seat-grid-ticket`; motion `ticket-keyframe`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckEventsState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

22 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.events.program.root` | MuckEventsViews.swift:74 |
| `screen.events.root` | MuckEventsViews.swift:47 |
| `screen.events.seats.root` | MuckEventsViews.swift:134 |
| `screen.events.ticket.root` | MuckEventsViews.swift:174 |
| `ui.events.contactEmailField` | MuckEventsViews.swift:123 |
| `ui.events.contactNameField` | MuckEventsViews.swift:120 |
| `ui.events.infoButton` | MuckEventsViews.swift:70 |
| `ui.events.issueTicketButton` | MuckEventsViews.swift:128 |
| `ui.events.refreshButton` | MuckEventsViews.swift:36 |
| `ui.events.reserveSeatButton` | MuckEventsViews.swift:58 |
| `ui.events.schedulePicker` | MuckEventsViews.swift:16 |
| `ui.events.seatA2Button` | MuckEventsViews.swift:99 |
| `ui.events.seatA3Button` | MuckEventsViews.swift:101 |
| `ui.events.seatB1Button` | MuckEventsViews.swift:103 |
| `ui.events.seatB2Button` | MuckEventsViews.swift:105 |
| `ui.events.seatB3Button` | MuckEventsViews.swift:107 |
| `ui.events.seatButton` | MuckEventsViews.swift:97 |
| `ui.events.seatC1Button` | MuckEventsViews.swift:109 |
| `ui.events.seatC2Button` | MuckEventsViews.swift:111 |
| `ui.events.seatC3Button` | MuckEventsViews.swift:113 |
| `ui.events.shareButton` | MuckEventsViews.swift:169 |
| `ui.events.statusBanner` | MuckEventsViews.swift:225 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
