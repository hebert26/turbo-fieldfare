# Overview — MuckSupport

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.support` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckSupportApp.swift` (267 lines) | file listing |

## Purpose

A mock customer-support app with a status-filtered ticket list and detail, a live chat with simulated agent replies, and a profile/workflow settings form.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `TabView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Tickets | a status-filtered list navigating to a detail form with assign/comment/resolve buttons | TicketsView, TicketDetailView |
| Live chat | a scrolling message thread with a draft field and a simulated agent reply | LiveChatView, ChatMessage |
| Profile and workflow settings | notification and auto-assign toggles, a team picker, and workflow sliders/steppers | ProfileSettingsView |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `list,chat,settings`; navigation
`tabs-detail-sheet`; motion `standard`; accessibility profile
`N`; reset mode `relaunch`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

Plain SwiftUI @State per view, no reset hook found in this file. Reset mode is `relaunch` (`apps.tsv`).

## Source-declared accessibility landmarks

5 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.support.chat.root` | MuckSupportApp.swift:205 |
| `screen.support.root` | MuckSupportApp.swift:71 |
| `ui.support.chatDraftField` | MuckSupportApp.swift:193 |
| `ui.support.chatSendButton` | MuckSupportApp.swift:200 |
| `ui.support.chatTab` | MuckSupportApp.swift:64 |

## Not verified

The following could not be confirmed in source and is deliberately not claimed:

- The ticket-detail Assign/Comment/Resolve buttons have empty action closures in this file, so it is unclear whether any state change is intended there

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
