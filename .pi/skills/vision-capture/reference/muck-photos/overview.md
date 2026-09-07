# Overview — MuckPhotos

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.photos` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckPhotosApp.swift`, `Sources/MuckPhotosRoot.swift`, `Sources/MuckPhotosState.swift` (327 lines) | file listing |

## Purpose

A mock photo-gallery app with a selectable grid, a magnifiable full-screen pager with favorites, and a share-selection action.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Gallery grid | a grid of photo tiles with a multi-select mode | MuckPhotosRoot, MuckPhotoFixture |
| Full-screen detail | a pager with a magnification gesture, zoom controls, a favorite toggle, and a share link | MuckPhotoPager, MuckPhotoDetail |
| Share selection | a share action that requires at least one selected photo | MuckPhotosState |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `selection,magnify,share`; navigation
`grid-fullscreen-paging`; motion `matched-geometry`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckPhotosState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

15 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.muckphotos.detail.root` | MuckPhotosRoot.swift:170 |
| `screen.muckphotos.error.root` | MuckPhotosRoot.swift:100 |
| `screen.muckphotos.gallery.root` | MuckPhotosRoot.swift:49 |
| `screen.muckphotos.loading.root` | MuckPhotosRoot.swift:92 |
| `screen.muckphotos.success.root` | MuckPhotosRoot.swift:96 |
| `ui.muckphotos.closeButton` | MuckPhotosRoot.swift:168 |
| `ui.muckphotos.favoriteButton` | MuckPhotosRoot.swift:159 |
| `ui.muckphotos.fullScreenPager` | MuckPhotosRoot.swift:199 |
| `ui.muckphotos.magnifyImage` | MuckPhotosRoot.swift:147 |
| `ui.muckphotos.resetButton` | MuckPhotosRoot.swift:27 |
| `ui.muckphotos.resetZoomButton` | MuckPhotosRoot.swift:153 |
| `ui.muckphotos.selectButton` | MuckPhotosRoot.swift:25 |
| `ui.muckphotos.shareLink` | MuckPhotosRoot.swift:163 |
| `ui.muckphotos.shareSelectionButton` | MuckPhotosRoot.swift:29 |
| `ui.muckphotos.zoomButton` | MuckPhotosRoot.swift:150 |

Unstable — composed from runtime values, not reliable landmarks: `photo.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
