# Overview — MuckHealth

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.health` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/HealthModels.swift`, `Sources/HealthScreens.swift`, `Sources/HealthStore.swift`, `Sources/MuckHealthApp.swift` (280 lines) | file listing |

## Purpose

A mock health-metrics app with a metric picker (pulse, hydration, rest), a gauge summary, a bar-chart detail screen, and a reading-log form.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Metric summary | a metric picker and a gauge summary card | HealthRootView, HealthSummaryCard |
| Chart detail | a bar-chart screen filtered by a day-range slider | HealthChartDetail |
| Log reading | a slider and date picker to log a new reading with async save/clear actions | HealthStore, HealthStatusView |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `gauge,slider,dates,forms`; navigation
`metric-tabs-chart-detail`; motion `numeric-transition`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable HealthStore, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

21 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `health.cancelButton` | HealthScreens.swift:143 |
| `health.chart` | HealthScreens.swift:100 |
| `health.chartDetail.root` | HealthScreens.swift:106 |
| `health.chartDetailButton` | HealthScreens.swift:33 |
| `health.chartEmpty` | HealthScreens.swift:86 |
| `health.clearButton` | HealthScreens.swift:54 |
| `health.datePicker` | HealthScreens.swift:42 |
| `health.empty` | HealthScreens.swift:148 |
| `health.error` | HealthScreens.swift:147 |
| `health.finishButton` | HealthScreens.swift:142 |
| `health.fixtureNotice` | HealthScreens.swift:13 |
| `health.loading` | HealthScreens.swift:145 |
| `health.metricPicker` | HealthScreens.swift:21 |
| `health.rangeSlider` | HealthScreens.swift:46 |
| `health.readingSlider` | HealthScreens.swift:37 |
| `health.readingValue` | HealthScreens.swift:40 |
| `health.resetButton` | HealthScreens.swift:61 |
| `health.root` | HealthScreens.swift:62 |
| `health.saveButton` | HealthScreens.swift:51 |
| `health.success` | HealthScreens.swift:146 |
| `health.summaryCard` | HealthScreens.swift:128 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
