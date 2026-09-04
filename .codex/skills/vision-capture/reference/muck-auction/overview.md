# Overview — MuckAuction

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.auction` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/AuctionAppState.swift`, `Sources/AuctionModels.swift`, `Sources/AuctionScreens.swift`, `Sources/MuckAuctionApp.swift` (357 lines) | file listing |

## Purpose

A mock local auction app with a grid of lots, a lot detail screen with a live countdown, and a bid sheet with a confirmation alert.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Lot browsing | grid of lots navigating to a detail screen | AuctionLotsScreen, AuctionLotCard, AuctionDetailScreen |
| Bidding | bid sheet with an amount field and stepper plus a confirmation alert | AuctionBidSheet |
| Countdown | local countdown with a manual advance control and a closed-bidding state | AuctionCountdown, AuctionStatusPanel |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `field,stepper,countdown,alert`; navigation
`lot-grid-bid-sheet`; motion `timer-pulse`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable AuctionAppState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

18 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.auction.bid.root` | AuctionScreens.swift:141 |
| `screen.auction.detail.root` | AuctionScreens.swift:79 |
| `screen.auction.empty.root` | AuctionScreens.swift:83 |
| `screen.auction.lots.root` | AuctionScreens.swift:22 |
| `ui.auction.alert.cancelButton` | AuctionScreens.swift:146 |
| `ui.auction.alert.placeBidButton` | AuctionScreens.swift:144 |
| `ui.auction.bid.amountField` | AuctionScreens.swift:119 |
| `ui.auction.bid.cancelButton` | AuctionScreens.swift:134 |
| `ui.auction.bid.error` | AuctionScreens.swift:128 |
| `ui.auction.bid.reviewButton` | AuctionScreens.swift:138 |
| `ui.auction.bid.stepper` | AuctionScreens.swift:125 |
| `ui.auction.detail.advanceCountdownButton` | AuctionScreens.swift:104 |
| `ui.auction.detail.closedNotice` | AuctionScreens.swift:65 |
| `ui.auction.detail.countdownLabel` | AuctionScreens.swift:101 |
| `ui.auction.detail.error` | AuctionScreens.swift:165 |
| `ui.auction.detail.loading` | AuctionScreens.swift:161 |
| `ui.auction.detail.placeBidButton` | AuctionScreens.swift:74 |
| `ui.auction.detail.success` | AuctionScreens.swift:169 |

Unstable — composed from runtime values, not reliable landmarks: `lot.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
