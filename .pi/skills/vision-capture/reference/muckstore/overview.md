# Overview — MuckStore

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.store` | built `Info.plist`; `project.yml` |
| Display name | `MuckStore` | built `Info.plist` → `CFBundleDisplayName` |
| Version | `1.0` | built `Info.plist` |
| Minimum OS | `17.0` | built `Info.plist` → `MinimumOSVersion` |
| Device families | iPhone and iPad | `project.yml` → `TARGETED_DEVICE_FAMILY: "1,2"` |

## Purpose

A mock iOS shopping app, one of the MuckApps fixtures built for VisionOS automation
and UI interaction testing (`/Users/dev-machine/Dev/MuckApps/README.md`). It is a
fixture, not a customer product.

All of it lives in one file: `Sources/MuckStoreApp.swift` (353 lines). It uses only
in-memory mock data and makes no network calls (`MuckApps/README.md`).

## Capability areas

Root is a `NavigationStack` catalog (`MuckStoreApp.swift:68`), not a tab bar.

| Area | What it covers | Verified from |
|---|---|---|
| Catalog | Product grid, category picker, search field | `StoreRootView:44`, `Picker("Category"):70`, `.searchable(prompt: "Search products"):101`, `ProductCard:144` |
| Product detail | Quantity stepper, add-to-cart, add confirmation | `ProductDetailView:177`, `:209`, `:221`, `:227` |
| Cart and checkout | Per-item quantity steppers, receipt-email field, shipping picker, save-preference toggle, checkout, success state | `CartView:236`, `:285`, `:295`, `:297`, `:302`, `:326`, `:336` |

Declared product categories (`ProductCategory:3`): `All`, `Devices`, `Audio`,
`Accessories`.

Declared navigation titles: `MuckStore` (`:99`), `Product` (`:231`), `Cart` (`:315`).

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `grid,search,form`;
navigation `tabs-detail-checkout`; motion `standard`; accessibility profile `N`;
reset mode `relaunch`.

## System permission prompts to expect

None. The built `Info.plist` contains zero usage-description keys.

## State and reset behaviour

In-memory only. Reset mode is `relaunch` (`apps.tsv`) — relaunching the app restores
initial state. No persistence, no network.

## Source-declared accessibility landmarks

All 10 `accessibilityIdentifier` sites. Use them to **recognise** a screen in observed
evidence and to plan coverage. They are not selectors to send and not action
authority.

| Identifier | Declared at |
|---|---|
| `screen.store.catalog.root` | `:100` |
| `ui.store.primaryProductButton` | `:86` |
| `screen.store.productDetail.root` | `:232` |
| `ui.store.quantityStepper` | `:212` |
| `ui.store.addButton` | `:221` |
| `ui.store.addConfirmation` | `:227` |
| `ui.store.cartButton` | `:120` |
| `screen.store.cart.root` | `:316` |
| `ui.store.checkoutButton` | `:326` |
| `ui.store.checkoutSuccess` | `:336` |

No runtime-composed identifiers are declared in this app. The cart's per-item
steppers (`:285`), receipt-email field (`:295`), shipping picker (`:297`), and
save-preference toggle (`:302`) carry **no** identifier — reach them through generic
accessibility roles and observed structure.

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
