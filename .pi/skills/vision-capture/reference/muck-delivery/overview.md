# Overview — MuckDelivery

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.delivery` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/DeliveryAppState.swift`, `Sources/DeliveryModels.swift`, `Sources/DeliveryScreens.swift`, `Sources/MuckDeliveryApp.swift` (301 lines) | file listing |

## Purpose

A mock parcel-delivery app with a parcel list, a tracking screen showing delivery progress and a tip/rating form, and an address-update sheet.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Parcel list and tracking | a split list of parcels and a tracking screen with a progress bar | DeliveryRootScreen, DeliveryTrackingScreen |
| Delivery preferences | a tip slider and a star-rating picker | DeliveryStatusPanel |
| Address update | a sheet to edit and save a new delivery address | DeliveryAddressSheet |
| Support chat placeholder | a sheet with a static no-live-chat-service message | DeliveryChatSheet |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `fields,slider,rating`; navigation
`parcel-list-tracking-chat`; motion `step-progress`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable DeliveryAppState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

19 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.delivery.address.root` | DeliveryScreens.swift:122 |
| `screen.delivery.chat.root` | DeliveryScreens.swift:143 |
| `screen.delivery.empty.root` | DeliveryScreens.swift:29 |
| `screen.delivery.parcels.root` | DeliveryScreens.swift:19 |
| `screen.delivery.tracking.root` | DeliveryScreens.swift:90 |
| `ui.delivery.address.addressField` | DeliveryScreens.swift:106 |
| `ui.delivery.address.cancelButton` | DeliveryScreens.swift:115 |
| `ui.delivery.address.error` | DeliveryScreens.swift:109 |
| `ui.delivery.address.saveButton` | DeliveryScreens.swift:119 |
| `ui.delivery.chat.doneButton` | DeliveryScreens.swift:140 |
| `ui.delivery.tracking.chatButton` | DeliveryScreens.swift:85 |
| `ui.delivery.tracking.error` | DeliveryScreens.swift:159 |
| `ui.delivery.tracking.finishButton` | DeliveryScreens.swift:82 |
| `ui.delivery.tracking.loading` | DeliveryScreens.swift:155 |
| `ui.delivery.tracking.progress` | DeliveryScreens.swift:56 |
| `ui.delivery.tracking.ratingPicker` | DeliveryScreens.swift:68 |
| `ui.delivery.tracking.success` | DeliveryScreens.swift:163 |
| `ui.delivery.tracking.tipSlider` | DeliveryScreens.swift:62 |
| `ui.delivery.tracking.updateAddressButton` | DeliveryScreens.swift:78 |

Unstable — composed from runtime values, not reliable landmarks: `parcel.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
