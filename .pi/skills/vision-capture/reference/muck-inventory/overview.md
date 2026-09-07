# Overview — MuckInventory

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.inventory` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckInventoryApp.swift`, `Sources/MuckInventoryState.swift`, `Sources/MuckInventoryViews.swift` (331 lines) | file listing |

## Purpose

A mock warehouse-stock app with a searchable and sortable item list, a receive/adjust-quantity sheet, and an async refresh with status banners.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Stock list | a search field and a sort menu over a list of stock items | MuckInventoryRoot, InventoryRow |
| Adjust and receive quantity | a sheet with a quantity field for receiving or reducing stock, validated against zero | InventoryAdjustmentSheet |
| Refresh and summary | an async refresh with a status banner and a summary workspace placeholder | InventorySummary |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `sort,search,selection,field`; navigation
`split-list-adjustment-sheet`; motion `numeric-row-transition`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckInventoryState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

12 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.inventory.adjustment.root` | MuckInventoryViews.swift:138 |
| `screen.inventory.root` | MuckInventoryViews.swift:53 |
| `ui.inventory.applyButton` | MuckInventoryViews.swift:134 |
| `ui.inventory.cancelButton` | MuckInventoryViews.swift:130 |
| `ui.inventory.quantityField` | MuckInventoryViews.swift:123 |
| `ui.inventory.receiveButton` | MuckInventoryViews.swift:40 |
| `ui.inventory.refreshButton` | MuckInventoryViews.swift:100 |
| `ui.inventory.searchField` | MuckInventoryViews.swift:13 |
| `ui.inventory.selectButton` | MuckInventoryViews.swift:103 |
| `ui.inventory.sortMenu` | MuckInventoryViews.swift:33 |
| `ui.inventory.statusBanner` | MuckInventoryViews.swift:147 |
| `ui.inventory.stockRow` | MuckInventoryViews.swift:21 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
