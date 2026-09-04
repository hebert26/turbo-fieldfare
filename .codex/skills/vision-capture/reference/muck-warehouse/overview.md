# Overview — MuckWarehouse

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.warehouse` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckWarehouseApp.swift`, `Sources/MuckWarehouseState.swift`, `Sources/MuckWarehouseViews.swift` (259 lines) | file listing |

## Purpose

A mock warehouse bin-management app with a zone-grouped bin list, a scanner field to receive stock, and bin-to-bin relocation.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Bin browsing | a zone-grouped sidebar of bins with a context menu to select | WarehouseSidebar, WarehouseBinButton |
| Receive stock | a scanner text field that increments quantity on entry | WarehouseBinDetail |
| Quantity and relocate | a quantity stepper and a destination picker to move one unit between bins | WarehouseBinDetail, WarehouseBin |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `scanner,stepper,menu,drag`; navigation
`hierarchical-zones-bins`; motion `row-relocation`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckWarehouseState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

11 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.warehouse.bin.root` | MuckWarehouseViews.swift:111 |
| `screen.warehouse.root` | MuckWarehouseViews.swift:13 |
| `ui.warehouse.binRow` | MuckWarehouseViews.swift:46 |
| `ui.warehouse.destinationPicker` | MuckWarehouseViews.swift:104 |
| `ui.warehouse.moveButton` | MuckWarehouseViews.swift:106 |
| `ui.warehouse.quantityStepper` | MuckWarehouseViews.swift:98 |
| `ui.warehouse.receiveButton` | MuckWarehouseViews.swift:90 |
| `ui.warehouse.refreshButton` | MuckWarehouseViews.swift:33 |
| `ui.warehouse.scannerField` | MuckWarehouseViews.swift:88 |
| `ui.warehouse.selectBinButton` | MuckWarehouseViews.swift:49 |
| `ui.warehouse.statusBanner` | MuckWarehouseViews.swift:122 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
