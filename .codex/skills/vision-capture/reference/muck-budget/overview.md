# Overview — MuckBudget

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.budget` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/BudgetModels.swift`, `Sources/BudgetScreens.swift`, `Sources/BudgetStore.swift`, `Sources/MuckBudgetApp.swift` (259 lines) | file listing |

## Purpose

A mock personal-budgeting app with a spending overview (gauge and bar chart), an expense-entry form, and a category-rule toggle across three tabs.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Overview | total spend, a spending gauge, a bar chart, and a monthly-limit slider | BudgetRootView, BudgetBarChart |
| Expenses | amount and category entry that classifies a new expense with async status | BudgetTabContent, BudgetStatusPane |
| Rules | a toggle for a category rule with a status label | BudgetStore |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `fields,slider,charts`; navigation
`tabs-category-rule-builder`; motion `bar-numeric-transition`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable BudgetStore, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

21 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `budget.amountField` | BudgetScreens.swift:46 |
| `budget.cancelButton` | BudgetScreens.swift:134 |
| `budget.categoryPicker` | BudgetScreens.swift:50 |
| `budget.chart` | BudgetScreens.swift:105 |
| `budget.classifyButton` | BudgetScreens.swift:52 |
| `budget.clearButton` | BudgetScreens.swift:74 |
| `budget.empty` | BudgetScreens.swift:139 |
| `budget.error` | BudgetScreens.swift:138 |
| `budget.finishButton` | BudgetScreens.swift:133 |
| `budget.fixtureNotice` | BudgetScreens.swift:14 |
| `budget.limitSlider` | BudgetScreens.swift:38 |
| `budget.limitValue` | BudgetScreens.swift:40 |
| `budget.loading` | BudgetScreens.swift:136 |
| `budget.resetButton` | BudgetScreens.swift:19 |
| `budget.root` | BudgetScreens.swift:20 |
| `budget.ruleStatus` | BudgetScreens.swift:81 |
| `budget.ruleToggle` | BudgetScreens.swift:79 |
| `budget.spendingGauge` | BudgetScreens.swift:35 |
| `budget.success` | BudgetScreens.swift:137 |
| `budget.tabPicker` | BudgetScreens.swift:12 |
| `budget.total` | BudgetScreens.swift:33 |

Unstable — composed from runtime values, not reliable landmarks: `categoryAccessibilityID(category`, `chartAccessibilityID(category`, `tab.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
