# Overview — MuckStudio

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.studio` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckStudioApp.swift`, `Sources/MuckStudioRoot.swift`, `Sources/MuckStudioState.swift` (266 lines) | file listing |

## Purpose

A mock video/design-editor app with a draggable-reorderable layer panel, a timeline playhead, and an inspector to move layers or render a preview.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Layer panel | a selectable, drag-and-drop reorderable list of layers | LayerPanel, StudioLayer |
| Timeline | a playhead slider with a preview area for the selected layer | TimelinePanel |
| Inspector | move-forward/backward controls and an async render-preview action | InspectorPanel |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `drag,menu,scrubber`; navigation
`split-timeline-layers-inspector`; motion `matched-selection`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckStudioState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

9 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.studio.root` | MuckStudioRoot.swift:24 |
| `ui.studio.inspectorPanel` | MuckStudioRoot.swift:123 |
| `ui.studio.layerMenu` | MuckStudioRoot.swift:119 |
| `ui.studio.layersPanel` | MuckStudioRoot.swift:59 |
| `ui.studio.moveBackwardButton` | MuckStudioRoot.swift:114 |
| `ui.studio.moveForwardButton` | MuckStudioRoot.swift:109 |
| `ui.studio.playheadSlider` | MuckStudioRoot.swift:71 |
| `ui.studio.status` | MuckStudioRoot.swift:89 |
| `ui.studio.timelinePanel` | MuckStudioRoot.swift:94 |

Unstable — composed from runtime values, not reliable landmarks: `layer.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
