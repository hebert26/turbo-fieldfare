# Overview — MuckGameHub

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.gamehub` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckGameHubApp.swift`, `Sources/MuckGameHubRoot.swift`, `Sources/MuckGameHubState.swift` (266 lines) | file listing |

## Purpose

A mock game-hub app with an arcade tab of game tiles and an achievement unlock/claim flow, plus a profile and leaderboard tab.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `TabView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Arcade | a grid of game tiles with a featured-game picker | ArcadeScreen, GameFixture |
| Achievements | an unlock-then-claim flow with an async claim and status message | AchievementPanel |
| Profile | rank and points plus a static leaderboard list | ProfileScreen |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `picker,buttons,progress`; navigation
`tabs-game-grid-profile`; motion `achievement-burst`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckGameHubState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

8 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.gamehub.arcade` | MuckGameHubRoot.swift:63 |
| `screen.gamehub.profile` | MuckGameHubRoot.swift:152 |
| `screen.gamehub.root` | MuckGameHubRoot.swift:17 |
| `ui.gamehub.achievementPanel` | MuckGameHubRoot.swift:102 |
| `ui.gamehub.claimButton` | MuckGameHubRoot.swift:96 |
| `ui.gamehub.gamePicker` | MuckGameHubRoot.swift:35 |
| `ui.gamehub.status` | MuckGameHubRoot.swift:78 |
| `ui.gamehub.unlockButton` | MuckGameHubRoot.swift:82 |

Unstable — composed from runtime values, not reliable landmarks: `game.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
