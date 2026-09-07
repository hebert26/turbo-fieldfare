# Overview — MuckAccessibilityLab

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.accessibilitylab` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckAccessibilityLabApp.swift`, `Sources/MuckAccessibilityLabModels.swift`, `Sources/MuckAccessibilityLabRoot.swift` (244 lines) | file listing |

## Purpose

A mock accessibility fixture app with duplicate-label controls, a numbered focus-order screen, and a modal sheet route.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationStack`.

| Area | What it covers | Source anchors |
|---|---|---|
| Duplicate label controls | two identically-labeled buttons distinguished only by stable identifiers, a disabled button, and a hidden/visible toggle | MuckAccessibilityLabRoot, MuckAccessibilityLabRoot.swift:63, MuckAccessibilityLabRoot.swift:69 |
| Focus order | two numbered focus-stop buttons with distinct accessibility values | MuckAccessibilityLabRoot.swift:111, MuckAccessibilityLabRoot.swift:116 |
| Modal sheet | opens and closes a sheet containing its own navigation stack | MuckAccessibilityLabRoot.swift:37, MuckAccessibilityLabRoot.swift:132 |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `labels,states,traits`; navigation
`sections-tabs-modals`; motion `minimal-reduce-motion`; accessibility profile
`B`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckAccessibilityLabState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

18 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.accessibilityLab.controls` | MuckAccessibilityLabRoot.swift:22 |
| `screen.accessibilityLab.focus` | MuckAccessibilityLabRoot.swift:25 |
| `screen.accessibilityLab.modal` | MuckAccessibilityLabRoot.swift:50 |
| `screen.accessibilityLab.modalRoute` | MuckAccessibilityLabRoot.swift:28 |
| `screen.accessibilityLab.root` | MuckAccessibilityLabRoot.swift:36 |
| `state.accessibilityLab.dynamicType` | MuckAccessibilityLabRoot.swift:110 |
| `state.accessibilityLab.message` | MuckAccessibilityLabRoot.swift:96 |
| `ui.accessibilityLab.closeModalButton` | MuckAccessibilityLabRoot.swift:46 |
| `ui.accessibilityLab.declineAmbiguousButton` | MuckAccessibilityLabRoot.swift:77 |
| `ui.accessibilityLab.disabledButton` | MuckAccessibilityLabRoot.swift:84 |
| `ui.accessibilityLab.firstChooseButton` | MuckAccessibilityLabRoot.swift:68 |
| `ui.accessibilityLab.firstFocusButton` | MuckAccessibilityLabRoot.swift:115 |
| `ui.accessibilityLab.hiddenToggle` | MuckAccessibilityLabRoot.swift:87 |
| `ui.accessibilityLab.openModalButton` | MuckAccessibilityLabRoot.swift:135 |
| `ui.accessibilityLab.secondChooseButton` | MuckAccessibilityLabRoot.swift:74 |
| `ui.accessibilityLab.secondFocusButton` | MuckAccessibilityLabRoot.swift:120 |
| `ui.accessibilityLab.sectionPicker` | MuckAccessibilityLabRoot.swift:17 |
| `ui.accessibilityLab.visibleButton` | MuckAccessibilityLabRoot.swift:91 |

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
