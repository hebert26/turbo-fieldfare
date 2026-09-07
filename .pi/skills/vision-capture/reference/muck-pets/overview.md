# Overview — MuckPets

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.pets` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckPetsApp.swift`, `Sources/MuckPetsRoot.swift`, `Sources/MuckPetsState.swift` (268 lines) | file listing |

## Purpose

A mock pet-care app with a swipeable pet-profile carousel, a daily care checklist, and a vet-visit scheduling form.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Pet carousel | a paged carousel of pet-profile cards | MuckPetsRoot, PetCard |
| Care checklist | today's care task for the selected pet with a link to plan a visit | CareStack |
| Visit planner | a date picker, a reminder toggle, and an async schedule action with a status message | VisitPlanner |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `segments,date,swipe,toggle`; navigation
`profile-carousel-care-stack`; motion `badge-bounce`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckPetsState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

10 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.pets.carousel` | MuckPetsRoot.swift:25 |
| `screen.pets.root` | MuckPetsRoot.swift:35 |
| `ui.pets.errorButton` | MuckPetsRoot.swift:120 |
| `ui.pets.planVisitButton` | MuckPetsRoot.swift:87 |
| `ui.pets.reminderToggle` | MuckPetsRoot.swift:102 |
| `ui.pets.scheduleButton` | MuckPetsRoot.swift:116 |
| `ui.pets.sectionPicker` | MuckPetsRoot.swift:14 |
| `ui.pets.status` | MuckPetsRoot.swift:134 |
| `ui.pets.visitBadge` | MuckPetsRoot.swift:131 |
| `ui.pets.visitDatePicker` | MuckPetsRoot.swift:100 |

Unstable — composed from runtime values, not reliable landmarks: `pet.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
