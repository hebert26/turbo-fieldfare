# Overview — MuckMeditation

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.meditation` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckMeditationApp.swift`, `Sources/MuckMeditationRoot.swift`, `Sources/MuckMeditationState.swift` (278 lines) | file listing |

## Purpose

A mock meditation app with a session carousel, an animated breathing ring and timer, and duration/sound/reminder settings.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Session picker | a paged carousel of session fixtures | MuckMeditationRoot, MuckMeditationSession |
| Breathing ring and timer | an animated ring showing remaining minutes with start, advance, and complete controls | MuckMeditationState |
| Session settings | a duration picker and sounds/reminder toggles | MuckMeditationRoot.swift:80 |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `picker,toggles,controls`; navigation
`session-carousel-timer`; motion `breathing-ring`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckMeditationState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

13 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.muckmeditation.breathingRing.root` | MuckMeditationRoot.swift:74 |
| `screen.muckmeditation.configuration.root` | MuckMeditationRoot.swift:98 |
| `screen.muckmeditation.error.root` | MuckMeditationRoot.swift:129 |
| `screen.muckmeditation.loading.root` | MuckMeditationRoot.swift:121 |
| `screen.muckmeditation.session.root` | MuckMeditationRoot.swift:35 |
| `screen.muckmeditation.success.root` | MuckMeditationRoot.swift:125 |
| `ui.muckmeditation.completeButton` | MuckMeditationRoot.swift:110 |
| `ui.muckmeditation.durationPicker` | MuckMeditationRoot.swift:89 |
| `ui.muckmeditation.primaryButton` | MuckMeditationRoot.swift:107 |
| `ui.muckmeditation.reminderToggle` | MuckMeditationRoot.swift:94 |
| `ui.muckmeditation.resetButton` | MuckMeditationRoot.swift:32 |
| `ui.muckmeditation.sessionCarousel` | MuckMeditationRoot.swift:56 |
| `ui.muckmeditation.soundsToggle` | MuckMeditationRoot.swift:92 |

Unstable — composed from runtime values, not reliable landmarks: `session.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
