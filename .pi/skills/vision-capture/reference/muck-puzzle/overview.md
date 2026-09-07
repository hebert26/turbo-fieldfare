# Overview — MuckPuzzle

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.puzzle` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckPuzzleApp.swift`, `Sources/MuckPuzzleRoot.swift`, `Sources/MuckPuzzleState.swift` (258 lines) | file listing |

## Purpose

A mock eight-tile sliding puzzle with a fixture timer, undo/redo, and a restart confirmation dialog.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Puzzle board | a three-by-three sliding-tile grid moved by tap or drag | MuckPuzzleRoot, PuzzleTile |
| Timer | a fixture timer with start/stop and a manual advance control | MuckPuzzleState |
| Undo, redo, and restart | an undo/redo stack and a destructive restart confirmation | MuckPuzzleState |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `drag,undo,redo,timer`; navigation
`board-menus-dialog`; motion `tile-spring`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckPuzzleState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

10 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.puzzle.root` | MuckPuzzleRoot.swift:68 |
| `ui.puzzle.advanceTimerButton` | MuckPuzzleRoot.swift:16 |
| `ui.puzzle.board` | MuckPuzzleRoot.swift:45 |
| `ui.puzzle.boardMenu` | MuckPuzzleRoot.swift:61 |
| `ui.puzzle.redoButton` | MuckPuzzleRoot.swift:56 |
| `ui.puzzle.startTimerButton` | MuckPuzzleRoot.swift:23 |
| `ui.puzzle.status` | MuckPuzzleRoot.swift:49 |
| `ui.puzzle.stopTimerButton` | MuckPuzzleRoot.swift:26 |
| `ui.puzzle.timerStatus` | MuckPuzzleRoot.swift:13 |
| `ui.puzzle.undoButton` | MuckPuzzleRoot.swift:53 |

Unstable — composed from runtime values, not reliable landmarks: `accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
