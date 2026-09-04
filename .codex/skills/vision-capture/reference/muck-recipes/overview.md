# Overview — MuckRecipes

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.recipes` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckRecipesApp.swift`, `Sources/MuckRecipesScreens.swift`, `Sources/MuckRecipesState.swift` (248 lines) | file listing |

## Purpose

A mock recipe app with a recipe grid, a detail screen with a servings stepper and ingredient checklist, and a cooking-timer sheet.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Recipe grid | a grid of recipes navigating to a detail screen | MuckRecipesRoot, Recipe |
| Recipe detail | a servings stepper and an ingredient toggle checklist | RecipeDetail, RecipeIngredient |
| Cooking timer | a sheet with a minutes stepper finished by a manual action | RecipeTimerSheet |
| Fixture state menu | load, empty, error, and reset controls | RecipesFixtureControls |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `toggles,stepper,timer`; navigation
`grid-detail-sheet`; motion `spring-progress`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckRecipesState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

16 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.recipes.detail.root` | MuckRecipesScreens.swift:89 |
| `screen.recipes.grid.root` | MuckRecipesScreens.swift:44 |
| `screen.recipes.timer.root` | MuckRecipesScreens.swift:129 |
| `ui.recipes.closeTimerButton` | MuckRecipesScreens.swift:126 |
| `ui.recipes.emptyButton` | MuckRecipesScreens.swift:138 |
| `ui.recipes.errorButton` | MuckRecipesScreens.swift:139 |
| `ui.recipes.finishTimerButton` | MuckRecipesScreens.swift:123 |
| `ui.recipes.fixtureMenu` | MuckRecipesScreens.swift:142 |
| `ui.recipes.loadButton` | MuckRecipesScreens.swift:137 |
| `ui.recipes.loading` | MuckRecipesScreens.swift:11 |
| `ui.recipes.openTimerButton` | MuckRecipesScreens.swift:85 |
| `ui.recipes.resetButton` | MuckRecipesScreens.swift:140 |
| `ui.recipes.retryButton` | MuckRecipesScreens.swift:15 |
| `ui.recipes.servingsStepper` | MuckRecipesScreens.swift:68 |
| `ui.recipes.successBanner` | MuckRecipesScreens.swift:49 |
| `ui.recipes.timerStepper` | MuckRecipesScreens.swift:120 |

Unstable — composed from runtime values, not reliable landmarks: `ingredient.accessibilityID`, `recipe.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
