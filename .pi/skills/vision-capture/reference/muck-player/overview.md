# Overview — MuckPlayer

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.player` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckPlayerApp.swift` (1304 lines) | file listing |

## Purpose

A mock music-player app with playlist, album, and song browsing, a simulated playback engine, a persistent mini-player, and a full Now Playing screen with a queue.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `TabView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Library browsing | tabs for playlists, a grid of albums, and a searchable song list with a play/queue context menu | PlaylistsView, AlbumsView, SongsView, TrackContextMenu |
| Playback transport | a timer-driven playback engine with play/pause, rewind, advance, seek, shuffle, and repeat modes | PlayerEngine |
| Now Playing | a full-screen player with artwork, a scrubbable progress bar, and a volume slider | NowPlayingView |
| Queue | a reorderable and deletable up-next queue with a session log | QueueListView, QueueTabView |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `player,queue,drag`; navigation
`tabs-now-playing`; motion `playback-progress`; accessibility profile
`B`; reset mode `relaunch`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory ObservableObject PlayerEngine driven by a repeating Timer, no reset hook found in this file. Reset mode is `relaunch` (`apps.tsv`).

## Source-declared accessibility landmarks

29 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `albumsTab` | MuckPlayerApp.swift:752 |
| `clearQueueButton` | MuckPlayerApp.swift:924 |
| `closeNowPlayingButton` | MuckPlayerApp.swift:1072 |
| `elapsedLabel` | MuckPlayerApp.swift:1191 |
| `miniNextButton` | MuckPlayerApp.swift:1004 |
| `miniPlayPauseButton` | MuckPlayerApp.swift:994 |
| `miniPlayerBar` | MuckPlayerApp.swift:1023 |
| `nextButton` | MuckPlayerApp.swift:1227 |
| `nowPlayingArtist` | MuckPlayerApp.swift:1134 |
| `nowPlayingArtwork` | MuckPlayerApp.swift:1121 |
| `nowPlayingIndicator` | MuckPlayerApp.swift:543 |
| `nowPlayingProgressBar` | MuckPlayerApp.swift:1177 |
| `nowPlayingProgressSlider` | MuckPlayerApp.swift:1186 |
| `nowPlayingTitle` | MuckPlayerApp.swift:1129 |
| `openQueueSheetButton` | MuckPlayerApp.swift:1081 |
| `playCollectionButton` | MuckPlayerApp.swift:621 |
| `playPauseButton` | MuckPlayerApp.swift:1219 |
| `playlistsTab` | MuckPlayerApp.swift:672 |
| `previousButton` | MuckPlayerApp.swift:1209 |
| `queueEditButton` | MuckPlayerApp.swift:941 |
| `queueEmptyLabel` | MuckPlayerApp.swift:889 |
| `queueSkipButton` | MuckPlayerApp.swift:951 |
| `queueTab` | MuckPlayerApp.swift:855 |
| `remainingLabel` | MuckPlayerApp.swift:1194 |
| `screen.player.root` | MuckPlayerApp.swift:469 |
| `shuffleCollectionButton` | MuckPlayerApp.swift:630 |
| `songsTab` | MuckPlayerApp.swift:838 |
| `volumeSlider` | MuckPlayerApp.swift:1290 |
| `volumeValueLabel` | MuckPlayerApp.swift:1300 |

Unstable — composed from runtime values, not reliable landmarks: `"albumCard.\(album.title`, `"playlistRow.\(playlist.name`, `"queueRow.\(item.track.title`, `"trackRow.\(track.title`, `identifier`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
