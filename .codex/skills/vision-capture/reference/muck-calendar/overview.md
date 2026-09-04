# Overview — MuckCalendar

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.calendar` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckCalendarApp.swift`, `Sources/MuckCalendarScreens.swift`, `Sources/MuckCalendarState.swift` (319 lines) | file listing |

## Purpose

A mock calendar app with an agenda tab, a multi-date picker tab, and a sheet for creating or editing events.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `TabView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Agenda | list of events for the selected day navigating to a detail screen | CalendarAgenda, CalendarEventDetail |
| Dates | a multi-date picker tab | MuckCalendarRoot |
| Event editor | title, date, and notes fields for a new or edited event, with validation | CalendarEventSheet |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `calendar,dates,form`; navigation
`tabs-stack-event-sheet`; motion `matched-day-selection`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckCalendarState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

23 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.calendar.agenda.root` | MuckCalendarScreens.swift:35 |
| `screen.calendar.dates.root` | MuckCalendarScreens.swift:30 |
| `screen.calendar.event.root` | MuckCalendarScreens.swift:109 |
| `screen.calendar.eventSheet.root` | MuckCalendarScreens.swift:156 |
| `ui.calendar.agendaTab` | MuckCalendarScreens.swift:21 |
| `ui.calendar.cancelEventButton` | MuckCalendarScreens.swift:138 |
| `ui.calendar.datesTab` | MuckCalendarScreens.swift:33 |
| `ui.calendar.dayPicker` | MuckCalendarScreens.swift:67 |
| `ui.calendar.editEventButton` | MuckCalendarScreens.swift:102 |
| `ui.calendar.emptyButton` | MuckCalendarScreens.swift:165 |
| `ui.calendar.errorButton` | MuckCalendarScreens.swift:166 |
| `ui.calendar.eventDatePicker` | MuckCalendarScreens.swift:128 |
| `ui.calendar.eventNotesEditor` | MuckCalendarScreens.swift:130 |
| `ui.calendar.eventTitleField` | MuckCalendarScreens.swift:125 |
| `ui.calendar.fixtureMenu` | MuckCalendarScreens.swift:169 |
| `ui.calendar.loadButton` | MuckCalendarScreens.swift:164 |
| `ui.calendar.loading` | MuckCalendarScreens.swift:56 |
| `ui.calendar.multiDatePicker` | MuckCalendarScreens.swift:26 |
| `ui.calendar.newEventButton` | MuckCalendarScreens.swift:16 |
| `ui.calendar.resetButton` | MuckCalendarScreens.swift:167 |
| `ui.calendar.retryButton` | MuckCalendarScreens.swift:60 |
| `ui.calendar.successBanner` | MuckCalendarScreens.swift:41 |
| `ui.calendar.validationText` | MuckCalendarScreens.swift:132 |

Unstable — composed from runtime values, not reliable landmarks: `event.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
