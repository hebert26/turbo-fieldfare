# Overview — MuckRealty

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.realty` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckRealtyApp.swift`, `Sources/RealtyModels.swift`, `Sources/RealtyScreens.swift`, `Sources/RealtyStore.swift` (257 lines) | file listing |

## Purpose

A mock real-estate listing app with a list/detail split view, swipe-to-favorite and reorderable listings, and an illustrative mortgage-years slider.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Listing list | a swipe-to-favorite and reorderable list of listings | RealtyRootView |
| Listing detail | a price display, an illustrative mortgage-years slider, and an async favorite action | ListingDetail, RealtyStatusPane |
| Gallery | a sheet with an expandable local gallery placeholder | RealtyGallery |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `drag,favorite,slider`; navigation
`list-map-gallery`; motion `card-expansion-swipe`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable RealtyStore, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

20 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `realty.cancelButton` | RealtyScreens.swift:126 |
| `realty.clearButton` | RealtyScreens.swift:82 |
| `realty.closeGalleryButton` | RealtyScreens.swift:109 |
| `realty.editListingsButton` | RealtyScreens.swift:87 |
| `realty.empty` | RealtyScreens.swift:39 |
| `realty.error` | RealtyScreens.swift:130 |
| `realty.expandGalleryButton` | RealtyScreens.swift:66 |
| `realty.favoriteButton` | RealtyScreens.swift:79 |
| `realty.finishButton` | RealtyScreens.swift:125 |
| `realty.fixtureNotice` | RealtyScreens.swift:58 |
| `realty.gallery` | RealtyScreens.swift:61 |
| `realty.gallerySheet.root` | RealtyScreens.swift:112 |
| `realty.loading` | RealtyScreens.swift:128 |
| `realty.map` | RealtyScreens.swift:70 |
| `realty.mortgageSlider` | RealtyScreens.swift:72 |
| `realty.price` | RealtyScreens.swift:69 |
| `realty.resetButton` | RealtyScreens.swift:88 |
| `realty.root` | RealtyScreens.swift:42 |
| `realty.status.empty` | RealtyScreens.swift:131 |
| `realty.success` | RealtyScreens.swift:129 |

Unstable — composed from runtime values, not reliable landmarks: `listing.accessibilityID`, `listing.favoriteAccessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
