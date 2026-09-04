# Overview — MuckLocalization

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.localization` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckLocalizationApp.swift`, `Sources/MuckLocalizationModels.swift`, `Sources/MuckLocalizationRoot.swift` (202 lines) | file listing |

## Purpose

A fixture app for language, reading-direction, and type-scale rendering with English, Spanish, and Arabic sample text, with no translation service involved.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Language selection | a row of language buttons plus a menu picker for English, Spanish, and Arabic | MuckLocalizationRoot, LocalizationLanguageButton, FixtureLocale |
| Direction and scale | a right-to-left reading-order toggle and a type-scale slider | MuckLocalizationRoot.swift:46, MuckLocalizationRoot.swift:51 |
| Mirrored preview | a preview panel that reflows text alignment and order to match the settings | MuckLocalizationRoot.swift:60 |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `language-picker,long-text,scale`; navigation
`language-tabs-mirrored-stack`; motion `rtl-ltr-transition`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckLocalizationState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

6 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.localization.root` | MuckLocalizationRoot.swift:26 |
| `state.localization.direction` | MuckLocalizationRoot.swift:72 |
| `state.localization.preview` | MuckLocalizationRoot.swift:76 |
| `ui.localization.directionToggle` | MuckLocalizationRoot.swift:48 |
| `ui.localization.languagePicker` | MuckLocalizationRoot.swift:45 |
| `ui.localization.typeScaleSlider` | MuckLocalizationRoot.swift:52 |

Unstable — composed from runtime values, not reliable landmarks: `locale.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
