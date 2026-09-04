# Overview — MuckPharmacy

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.pharmacy` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckPharmacyApp.swift`, `Sources/PharmacyModels.swift`, `Sources/PharmacyScreens.swift`, `Sources/PharmacyStore.swift` (184 lines) | file listing |

## Purpose

A mock medication-reminder app with an agenda list, a dose-progress bar, and a reminder-scheduling form gated by a confirmation alert.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Reminder agenda | a list of reminders with a dose-progress bar | PharmacyRootView, Reminder |
| Schedule reminder | a multi-date picker, an enabled toggle, a quantity stepper, and a confirmation alert before an async finish | PharmacyStore, PharmacyStatusPane |
| Clear agenda | a destructive clear-all action that produces an empty state | PharmacyStore |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `dates,toggle,stepper,alert`; navigation
`agenda-schedule-form`; motion `dose-progress`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable PharmacyStore, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

18 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `pharmacy.alertCancelButton` | PharmacyScreens.swift:54 |
| `pharmacy.cancelButton` | PharmacyScreens.swift:73 |
| `pharmacy.clearButton` | PharmacyScreens.swift:46 |
| `pharmacy.confirmButton` | PharmacyScreens.swift:53 |
| `pharmacy.datePicker` | PharmacyScreens.swift:32 |
| `pharmacy.doseProgress` | PharmacyScreens.swift:26 |
| `pharmacy.empty` | PharmacyScreens.swift:18 |
| `pharmacy.enabledToggle` | PharmacyScreens.swift:34 |
| `pharmacy.error` | PharmacyScreens.swift:77 |
| `pharmacy.finishButton` | PharmacyScreens.swift:72 |
| `pharmacy.fixtureNotice` | PharmacyScreens.swift:30 |
| `pharmacy.loading` | PharmacyScreens.swift:75 |
| `pharmacy.quantityStepper` | PharmacyScreens.swift:36 |
| `pharmacy.resetButton` | PharmacyScreens.swift:49 |
| `pharmacy.root` | PharmacyScreens.swift:58 |
| `pharmacy.scheduleButton` | PharmacyScreens.swift:38 |
| `pharmacy.status.empty` | PharmacyScreens.swift:78 |
| `pharmacy.success` | PharmacyScreens.swift:76 |

Unstable — composed from runtime values, not reliable landmarks: `reminder.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
