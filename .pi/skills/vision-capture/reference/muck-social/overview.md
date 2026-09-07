# Overview — MuckSocial

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.social` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckSocialApp.swift` (1196 lines) | file listing |

## Purpose

A mock social and microblogging app with a feed, post composer, comments, profiles with follow, a discover/search tab, and notifications.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `TabView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Feed and posting | a pull-to-refresh post feed, like/comment/report actions, and a compose sheet with a character counter | FeedView, PostCardView, ComposeView |
| Comments | a per-post comments sheet with a reply field | CommentsView |
| Profiles and follow | a profile screen with follower/following stats and a follow toggle | ProfileView |
| Discover | search across posts and people, trending hashtag topics, and suggested accounts | DiscoverView, TopicView |
| Notifications | an unread-only filter, mark-all-read, and a reset action | NotificationsView, NotificationRow |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `feed,compose,menus`; navigation
`tabs-profile-sheet`; motion `animated-likes`; accessibility profile
`B`; reset mode `relaunch`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory ObservableObject MuckSocialStore, no reset hook found in this file. Reset mode is `relaunch` (`apps.tsv`).

## Source-declared accessibility landmarks

24 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `comments.done` | MuckSocialApp.swift:787 |
| `comments.empty` | MuckSocialApp.swift:745 |
| `comments.field` | MuckSocialApp.swift:765 |
| `comments.header.counts` | MuckSocialApp.swift:809 |
| `comments.scroll` | MuckSocialApp.swift:756 |
| `comments.send` | MuckSocialApp.swift:770 |
| `compose.cancel` | MuckSocialApp.swift:692 |
| `compose.counter` | MuckSocialApp.swift:672 |
| `compose.editor` | MuckSocialApp.swift:649 |
| `compose.post` | MuckSocialApp.swift:701 |
| `discover.noresults` | MuckSocialApp.swift:969 |
| `feed.compose.button` | MuckSocialApp.swift:391 |
| `feed.count.label` | MuckSocialApp.swift:383 |
| `feed.scroll` | MuckSocialApp.swift:375 |
| `notifications.list` | MuckSocialApp.swift:1106 |
| `notifications.markall` | MuckSocialApp.swift:1141 |
| `notifications.reset` | MuckSocialApp.swift:1134 |
| `notifications.status` | MuckSocialApp.swift:1123 |
| `notifications.unreadcount` | MuckSocialApp.swift:1115 |
| `notifications.unreadfilter` | MuckSocialApp.swift:1084 |
| `root.tabview` | MuckSocialApp.swift:344 |
| `tab.discover` | MuckSocialApp.swift:335 |
| `tab.feed` | MuckSocialApp.swift:330 |
| `tab.notifications` | MuckSocialApp.swift:341 |

Unstable — composed from runtime values, not reliable landmarks: `"avatar.\(user.id`, `"comment.\(comment.id`, `"compose.tag.\(tag.dropFirst(`, `"discover.follow.\(user.id`, `"discover.result.\(post.id`, `"discover.topic.\(topic.tag`, `"discover.user.\(user.id`, `"notification.\(notification.id`, `"post.\(post.id`, `"profile.\(userID`, `"topic.\(tag`, `identifier`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
