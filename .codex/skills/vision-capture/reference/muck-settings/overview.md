# Overview — MuckSettings

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.settings` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckSettingsApp.swift`, `Sources/MuckSettingsState.swift`, `Sources/SettingsScreens.swift` (176 lines) | file listing |

## Purpose

A mock system-settings app with appearance, accent, and text-scale controls, and an advanced-toggle-gated diagnostics section.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Appearance | an appearance picker, a text-scale slider, and an accent color picker | SettingsScreens.swift:67 |
| Advanced controls | a toggle that gates a dependent diagnostics toggle | SettingsScreens.swift:39 |
| Save and fixture states | an async save with a success message and empty/error/reset controls | SettingsState, SettingsRootView |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `toggle,picker,slider,menu`; navigation
`nested-forms`; motion `disabled-state-transition`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable SettingsState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

12 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.settings.form.root` | SettingsScreens.swift:24 |
| `ui.settings.accentColorPicker` | SettingsScreens.swift:86 |
| `ui.settings.accentMenu` | SettingsScreens.swift:81 |
| `ui.settings.advancedToggle` | SettingsScreens.swift:44 |
| `ui.settings.appearancePicker` | SettingsScreens.swift:74 |
| `ui.settings.detailToggle` | SettingsScreens.swift:50 |
| `ui.settings.resetButton` | SettingsScreens.swift:62 |
| `ui.settings.saveButton` | SettingsScreens.swift:56 |
| `ui.settings.showEmptyButton` | SettingsScreens.swift:58 |
| `ui.settings.showErrorButton` | SettingsScreens.swift:60 |
| `ui.settings.successMessage` | SettingsScreens.swift:35 |
| `ui.settings.textScaleSlider` | SettingsScreens.swift:78 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
