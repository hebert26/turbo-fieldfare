# Overview — MuckRestaurant

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.restaurant` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckRestaurantApp.swift`, `Sources/MuckRestaurantState.swift`, `Sources/MuckRestaurantViews.swift` (359 lines) | file listing |

## Purpose

A mock restaurant-ordering app with a categorized dish grid, a dish-modifier sheet, a cart with quantity steppers, and order placement.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Menu browsing | a category picker over a grid of dishes | MuckRestaurantRoot, MenuDish |
| Dish modifiers | a spicy toggle and a special-note field before adding to the cart | DishModifierSheet |
| Cart and order | quantity steppers, a reservation date picker, and order placement with validation | RestaurantCartSheet, RestaurantCartLine |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `stepper,dates,alerts`; navigation
`grid-modifiers-cart`; motion `matched-cart-transition`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckRestaurantState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

16 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.restaurant.cart.root` | MuckRestaurantViews.swift:142 |
| `screen.restaurant.modifiers.root` | MuckRestaurantViews.swift:103 |
| `screen.restaurant.root` | MuckRestaurantViews.swift:60 |
| `ui.restaurant.addButton` | MuckRestaurantViews.swift:99 |
| `ui.restaurant.cancelButton` | MuckRestaurantViews.swift:95 |
| `ui.restaurant.cartButton` | MuckRestaurantViews.swift:47 |
| `ui.restaurant.categoryPicker` | MuckRestaurantViews.swift:13 |
| `ui.restaurant.closeCartButton` | MuckRestaurantViews.swift:133 |
| `ui.restaurant.dishButton` | MuckRestaurantViews.swift:31 |
| `ui.restaurant.noteField` | MuckRestaurantViews.swift:89 |
| `ui.restaurant.placeOrderButton` | MuckRestaurantViews.swift:138 |
| `ui.restaurant.quantityStepper` | MuckRestaurantViews.swift:121 |
| `ui.restaurant.refreshButton` | MuckRestaurantViews.swift:43 |
| `ui.restaurant.serviceDatePicker` | MuckRestaurantViews.swift:126 |
| `ui.restaurant.spicyToggle` | MuckRestaurantViews.swift:87 |
| `ui.restaurant.statusBanner` | MuckRestaurantViews.swift:57 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
