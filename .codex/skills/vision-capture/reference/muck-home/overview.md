# Overview — MuckHome

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.home` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckHomeApp.swift`, `Sources/MuckHomeRoot.swift`, `Sources/MuckHomeState.swift` (276 lines) | file listing |

## Purpose

A mock smart-home app with a room picker, lamp and temperature device cards, and a scene-selection menu.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Room and devices | a room picker with lamp and temperature device cards | MuckHomeRoot, MuckHomeRoom |
| Scene selection | a scene menu plus a button that reveals the same detail as a long press | MuckHomeRoot.swift:102, MuckHomeRoot.swift:132 |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `switch,slider,menu,long-press`; navigation
`room-tabs-dashboard`; motion `symbol-tile-effects`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckHomeState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

14 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.muckhome.dashboard.root` | MuckHomeRoot.swift:35 |
| `screen.muckhome.error.root` | MuckHomeRoot.swift:156 |
| `screen.muckhome.header.root` | MuckHomeRoot.swift:53 |
| `screen.muckhome.lampCard.root` | MuckHomeRoot.swift:79 |
| `screen.muckhome.loading.root` | MuckHomeRoot.swift:148 |
| `screen.muckhome.longPressDetail.root` | MuckHomeRoot.swift:125 |
| `screen.muckhome.success.root` | MuckHomeRoot.swift:152 |
| `screen.muckhome.temperatureCard.root` | MuckHomeRoot.swift:99 |
| `ui.muckhome.lampToggle` | MuckHomeRoot.swift:72 |
| `ui.muckhome.resetButton` | MuckHomeRoot.swift:32 |
| `ui.muckhome.roomDetailsButton` | MuckHomeRoot.swift:137 |
| `ui.muckhome.roomPicker` | MuckHomeRoot.swift:17 |
| `ui.muckhome.sceneMenu` | MuckHomeRoot.swift:116 |
| `ui.muckhome.temperatureSlider` | MuckHomeRoot.swift:94 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
