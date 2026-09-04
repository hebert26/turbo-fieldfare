# Overview — MuckWeather

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.weather` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckWeatherApp.swift`, `Sources/MuckWeatherState.swift`, `Sources/WeatherScreens.swift` (189 lines) | file listing |

## Purpose

A mock weather-forecast app with a location picker and pager, an hourly temperature strip, a rain-chance gauge, and unit toggling.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Location forecast | a menu picker plus a paged view of location cards with a rain-chance gauge | WeatherRootView |
| Units and hourly | a Celsius/Fahrenheit toggle and a horizontal hourly-temperature strip | WeatherScreens.swift:57, WeatherScreens.swift:60 |
| Fixture states | show-empty/show-error controls and an async refresh | WeatherState |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `gauge,picker,toggle`; navigation
`paged-locations-hourly-strip`; motion `timeline-symbol`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable WeatherState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

8 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.weather.forecast.root` | WeatherScreens.swift:28 |
| `screen.weather.locationPages.root` | WeatherScreens.swift:56 |
| `ui.weather.locationPicker` | WeatherScreens.swift:45 |
| `ui.weather.nextLocationButton` | WeatherScreens.swift:48 |
| `ui.weather.refreshButton` | WeatherScreens.swift:33 |
| `ui.weather.showEmptyButton` | WeatherScreens.swift:78 |
| `ui.weather.showErrorButton` | WeatherScreens.swift:80 |
| `ui.weather.unitsToggle` | WeatherScreens.swift:58 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
