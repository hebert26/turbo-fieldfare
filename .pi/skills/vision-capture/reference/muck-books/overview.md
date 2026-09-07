# Overview — MuckBooks

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.books` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckBooksApp.swift`, `Sources/MuckBooksScreens.swift`, `Sources/MuckBooksState.swift` (235 lines) | file listing |

## Purpose

A mock e-reader app with a searchable book shelf, a paged reader with a reading-progress slider, and a type-size popover.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Shelf browsing | searchable grid of books navigating to a reader | MuckBooksRoot, Book |
| Reader | paged book pages with a progress slider and a save action | ReaderView |
| Fixture state controls | load, empty, error, and reset menu | BooksFixtureControls, FixtureState |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `search,slider,popover`; navigation
`shelf-reader`; motion `page-transition`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckBooksState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

18 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.books.reader.root` | MuckBooksScreens.swift:115 |
| `screen.books.shelf.root` | MuckBooksScreens.swift:57 |
| `screen.books.typePopover.root` | MuckBooksScreens.swift:113 |
| `ui.books.emptyButton` | MuckBooksScreens.swift:124 |
| `ui.books.errorButton` | MuckBooksScreens.swift:125 |
| `ui.books.fixtureMenu` | MuckBooksScreens.swift:128 |
| `ui.books.largeTypeButton` | MuckBooksScreens.swift:111 |
| `ui.books.loadButton` | MuckBooksScreens.swift:123 |
| `ui.books.loading` | MuckBooksScreens.swift:11 |
| `ui.books.progressSlider` | MuckBooksScreens.swift:97 |
| `ui.books.readerPage` | MuckBooksScreens.swift:92 |
| `ui.books.resetButton` | MuckBooksScreens.swift:126 |
| `ui.books.retryButton` | MuckBooksScreens.swift:15 |
| `ui.books.saveProgressButton` | MuckBooksScreens.swift:99 |
| `ui.books.searchField` | MuckBooksScreens.swift:25 |
| `ui.books.smallTypeButton` | MuckBooksScreens.swift:110 |
| `ui.books.successBanner` | MuckBooksScreens.swift:62 |
| `ui.books.typeButton` | MuckBooksScreens.swift:105 |

Unstable — composed from runtime values, not reliable landmarks: `book.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
