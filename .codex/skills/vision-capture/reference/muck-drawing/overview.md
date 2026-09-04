# Overview — MuckDrawing

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.drawing` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckDrawingApp.swift`, `Sources/MuckDrawingRoot.swift`, `Sources/MuckDrawingState.swift` (220 lines) | file listing |

## Purpose

A mock drawing app with a freehand canvas, ink color and stroke-width controls, canvas zoom, undo, and a save action.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Canvas drawing | freehand drag-to-draw canvas plus a fixture-stroke accessible alternative | MuckDrawingRoot, FixtureStroke |
| Tool inspector | ink color, stroke width, and canvas zoom controls | MuckDrawingRoot.swift:71, MuckDrawingRoot.swift:77 |
| Undo and save | undo of the last stroke and an async save with a status message | MuckDrawingState |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `drag,color,slider,undo`; navigation
`canvas-tool-inspector`; motion `interactive-stroke`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckDrawingState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

11 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.drawing.root` | MuckDrawingRoot.swift:17 |
| `ui.drawing.addStrokeButton` | MuckDrawingRoot.swift:87 |
| `ui.drawing.canvas` | MuckDrawingRoot.swift:65 |
| `ui.drawing.canvasZoomSlider` | MuckDrawingRoot.swift:79 |
| `ui.drawing.inkColorPicker` | MuckDrawingRoot.swift:72 |
| `ui.drawing.inspector` | MuckDrawingRoot.swift:110 |
| `ui.drawing.resetZoomButton` | MuckDrawingRoot.swift:83 |
| `ui.drawing.saveButton` | MuckDrawingRoot.swift:104 |
| `ui.drawing.status` | MuckDrawingRoot.swift:106 |
| `ui.drawing.strokeWidthSlider` | MuckDrawingRoot.swift:75 |
| `ui.drawing.undoButton` | MuckDrawingRoot.swift:92 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
