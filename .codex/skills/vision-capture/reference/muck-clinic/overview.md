# Overview — MuckClinic

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.clinic` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/ClinicAppState.swift`, `Sources/ClinicModels.swift`, `Sources/ClinicScreens.swift`, `Sources/MuckClinicApp.swift` (279 lines) | file listing |

## Purpose

A mock clinic-appointment app with a split agenda list, an appointment detail screen, and a reschedule form that ends in a confirmation screen.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Agenda | split list of appointments navigating to a detail screen | ClinicAgendaScreen, ClinicAppointmentScreen |
| Reschedule | a new-time date picker, a visit-kind picker, and a next-slot shortcut | ClinicIntakeScreen, ClinicStatusPanel |
| Confirmation | a dedicated confirmation screen shown after a reschedule completes | ClinicConfirmationScreen |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `date,segments,form`; navigation
`split-agenda-intake-confirmation`; motion `content-transition`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable ClinicAppState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

14 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.clinic.agenda.root` | ClinicScreens.swift:20 |
| `screen.clinic.appointment.root` | ClinicScreens.swift:63 |
| `screen.clinic.confirmation.root` | ClinicScreens.swift:117 |
| `screen.clinic.empty.root` | ClinicScreens.swift:30 |
| `screen.clinic.intake.root` | ClinicScreens.swift:94 |
| `ui.clinic.agenda.rescheduleButton` | ClinicScreens.swift:58 |
| `ui.clinic.confirmation.doneButton` | ClinicScreens.swift:114 |
| `ui.clinic.intake.datePicker` | ClinicScreens.swift:74 |
| `ui.clinic.intake.error` | ClinicScreens.swift:132 |
| `ui.clinic.intake.loading` | ClinicScreens.swift:128 |
| `ui.clinic.intake.nextSlotButton` | ClinicScreens.swift:76 |
| `ui.clinic.intake.saveButton` | ClinicScreens.swift:90 |
| `ui.clinic.intake.success` | ClinicScreens.swift:136 |
| `ui.clinic.intake.visitKindPicker` | ClinicScreens.swift:84 |

Unstable — composed from runtime values, not reliable landmarks: `appointment.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
