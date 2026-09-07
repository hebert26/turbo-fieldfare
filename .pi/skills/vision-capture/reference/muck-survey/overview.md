# Overview — MuckSurvey

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.survey` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckSurveyApp.swift`, `Sources/MuckSurveyState.swift`, `Sources/SurveyScreens.swift` (234 lines) | file listing |

## Purpose

A mock three-page survey wizard covering role selection, a fixture-notice and clarity slider, and an optional note, ending in a success screen.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Role question | a single-select list of role buttons | SurveyScreens.swift:53 |
| Detail question | a required fixture-notice toggle, an optional updates toggle, and a clarity slider | SurveyScreens.swift:72 |
| Note and submit | an optional note field and an async submit with validation | SurveyScreens.swift:84, SurveyState |
| Success | a completion screen with a restart action | SurveyScreens.swift:123 |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `radio,check,slider,fields`; navigation
`branching-wizard`; motion `progress-transition`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable SurveyState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

14 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.survey.success.root` | SurveyScreens.swift:129 |
| `screen.survey.wizard.root` | SurveyScreens.swift:9 |
| `ui.survey.backButton` | SurveyScreens.swift:98 |
| `ui.survey.claritySlider` | SurveyScreens.swift:80 |
| `ui.survey.continueButton` | SurveyScreens.swift:107 |
| `ui.survey.fixtureToggle` | SurveyScreens.swift:76 |
| `ui.survey.noteField` | SurveyScreens.swift:90 |
| `ui.survey.progress` | SurveyScreens.swift:34 |
| `ui.survey.restartButton` | SurveyScreens.swift:132 |
| `ui.survey.showEmptyButton` | SurveyScreens.swift:116 |
| `ui.survey.showErrorButton` | SurveyScreens.swift:118 |
| `ui.survey.submitButton` | SurveyScreens.swift:103 |
| `ui.survey.updatesToggle` | SurveyScreens.swift:78 |
| `ui.survey.validationMessage` | SurveyScreens.swift:37 |

Unstable — composed from runtime values, not reliable landmarks: `identifier`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
