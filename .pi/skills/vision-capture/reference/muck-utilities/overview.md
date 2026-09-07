# Overview — MuckUtilities

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.utilities` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckUtilitiesApp.swift`, `Sources/MuckUtilitiesModels.swift`, `Sources/MuckUtilitiesRoot.swift` (314 lines) | file listing |

## Purpose

A mock utility app bundling a basic calculator, a unit converter, and an accent-color appearance picker behind a custom tab switch.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a custom root view.

| Area | What it covers | Source anchors |
|---|---|---|
| Calculator | a numeric keypad with basic operators and an updating display | MuckUtilitiesRoot.swift:32 |
| Converter | a value field, from/to unit wheel pickers, and a copy-to-clipboard result | MuckUtilitiesRoot.swift:57 |
| Appearance | a color picker with a live preview swatch | MuckUtilitiesRoot.swift:96 |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `keypad,picker,color,copy`; navigation
`calculator-converter-tabs`; motion `numeric-transition`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckUtilitiesState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

11 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.utilities.root` | MuckUtilitiesRoot.swift:29 |
| `state.utilities.calculatorDisplay` | MuckUtilitiesRoot.swift:41 |
| `state.utilities.colorPreview` | MuckUtilitiesRoot.swift:107 |
| `state.utilities.conversion` | MuckUtilitiesRoot.swift:83 |
| `state.utilities.copy` | MuckUtilitiesRoot.swift:89 |
| `ui.utilities.colorPicker` | MuckUtilitiesRoot.swift:102 |
| `ui.utilities.convertButton` | MuckUtilitiesRoot.swift:77 |
| `ui.utilities.converterValueField` | MuckUtilitiesRoot.swift:63 |
| `ui.utilities.copyButton` | MuckUtilitiesRoot.swift:86 |
| `ui.utilities.fromUnitPicker` | MuckUtilitiesRoot.swift:68 |
| `ui.utilities.toUnitPicker` | MuckUtilitiesRoot.swift:73 |

Unstable — composed from runtime values, not reliable landmarks: `keyAccessibilityID(key`, `tab.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
