# Overview — MuckNews

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.news` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckNewsApp.swift`, `Sources/MuckNewsState.swift`, `Sources/NewsScreens.swift` (247 lines) | file listing |

## Purpose

A mock news-reader app with a sectioned, pull-to-refresh feed, an article detail with a save-story action, and load-state fixture controls.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Feed | a section picker over a pull-to-refresh list with content, empty, loading, and error states | NewsRootView, NewsStory |
| Article and save | an article view with a save-or-remove-saved-story toggle | NewsArticleView |
| Fixture menu | show-empty, show-error, and reload controls | NewsScreens.swift:61 |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `refresh,disclosure,menu`; navigation
`tabs-feed-article`; motion `row-insertion-fade`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable NewsState; resetMuckFixture() also clears a UserDefaults key. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

8 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.news.article.root` | NewsScreens.swift:103 |
| `screen.news.feed.root` | NewsScreens.swift:29 |
| `ui.news.moreMenu` | NewsScreens.swift:68 |
| `ui.news.reloadButton` | NewsScreens.swift:67 |
| `ui.news.saveStoryButton` | NewsScreens.swift:98 |
| `ui.news.sectionPicker` | NewsScreens.swift:46 |
| `ui.news.showEmptyButton` | NewsScreens.swift:64 |
| `ui.news.showErrorButton` | NewsScreens.swift:65 |

Unstable — composed from runtime values, not reliable landmarks: `story.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
