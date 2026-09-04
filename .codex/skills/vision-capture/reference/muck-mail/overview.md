# Overview — MuckMail

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.mail` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckMailApp.swift`, `Sources/MuckMailScreens.swift`, `Sources/MuckMailState.swift` (332 lines) | file listing |

## Purpose

A mock email app with a searchable inbox, a message detail pane, swipe-to-toggle read state, and a compose sheet.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Inbox | a searchable list with edit-mode multi-select and a swipe action to toggle read state | MuckMailRoot, MailRow |
| Message detail | a reading pane for the selected message | MailDetail |
| Compose | a new-message sheet with recipient, subject, and body fields and validation | MailComposer |
| Fixture state menu | load, empty, error, and reset controls | MailFixtureControls |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `edit,selection,search,swipe`; navigation
`split-inbox-thread-compose`; motion `row-insertion-sheet`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable MuckMailState, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

20 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.mail.compose.root` | MuckMailScreens.swift:163 |
| `screen.mail.inbox.root` | MuckMailScreens.swift:62 |
| `screen.mail.thread.root` | MuckMailScreens.swift:124 |
| `ui.mail.bodyEditor` | MuckMailScreens.swift:146 |
| `ui.mail.cancelComposeButton` | MuckMailScreens.swift:155 |
| `ui.mail.composeButton` | MuckMailScreens.swift:56 |
| `ui.mail.editButton` | MuckMailScreens.swift:51 |
| `ui.mail.emptyButton` | MuckMailScreens.swift:173 |
| `ui.mail.errorButton` | MuckMailScreens.swift:174 |
| `ui.mail.fixtureMenu` | MuckMailScreens.swift:178 |
| `ui.mail.loadButton` | MuckMailScreens.swift:172 |
| `ui.mail.loading` | MuckMailScreens.swift:18 |
| `ui.mail.recipientField` | MuckMailScreens.swift:138 |
| `ui.mail.resetButton` | MuckMailScreens.swift:175 |
| `ui.mail.retryButton` | MuckMailScreens.swift:23 |
| `ui.mail.searchField` | MuckMailScreens.swift:14 |
| `ui.mail.sendButton` | MuckMailScreens.swift:159 |
| `ui.mail.subjectField` | MuckMailScreens.swift:142 |
| `ui.mail.successBanner` | MuckMailScreens.swift:72 |
| `ui.mail.validationText` | MuckMailScreens.swift:149 |

Unstable — composed from runtime values, not reliable landmarks: `message.readAccessibilityID`, `message.rowAccessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
