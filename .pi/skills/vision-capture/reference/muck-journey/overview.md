# Overview — MuckJourney

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.journey` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckJourneyApp.swift` (1275 lines) | file listing |

## Purpose

An intentionally complex mock project/issue-tracking and inbox app covering onboarding, real system permission prompts, deep navigation, sheets, full-screen covers, and confirmation dialogs.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a custom root view.

| Area | What it covers | Source anchors |
|---|---|---|
| Onboarding and permissions | a four-step paged flow (welcome, goals, permissions, profile) gated by an agreement toggle, requesting real notification and location-when-in-use authorization along the way | OnboardingFlowView, TermsSheetView, PermissionCenter |
| Dashboard | quick actions opening a release-notes sheet, a full-screen daily brief, and an escalation alert | DashboardView, ReleaseNotesSheet, DailyBriefView |
| Projects | a searchable project list, milestone toggles, a milestone-planner sheet, and a full-screen task composer | ProjectsView, ProjectDetailView, MilestonePlannerSheet, TaskComposerFullScreen |
| Inbox | a searchable thread list with swipe-to-read/archive, a reply composer, and an escalation sheet | InboxView, ThreadDetailView, ComposeThreadSheet, EscalationSheet |
| Settings | automation-preference toggles, permission re-request buttons, and a destructive reset-mock-data alert | SettingsView |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `forms,alerts,permissions`; navigation
`onboarding-navigation`; motion `standard`; accessibility profile
`N`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

Declared usage-description keys: `NSLocationWhenInUseUsageDescription`.

Permission requests found in source. Some of these raise a system prompt
without needing any `Info.plist` key, so this list is not the same as the one
above:

- notification authorisation — `MuckJourneyApp.swift:152`
- location when-in-use authorisation — `MuckJourneyApp.swift:177`

## State and reset behaviour

In-memory ObservableObject MuckJourneyStore and PermissionCenter; onboarding-complete flag stored via @AppStorage. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

11 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `screen.journey.onboarding.automation.root` | MuckJourneyApp.swift:420 |
| `screen.journey.onboarding.permissions.root` | MuckJourneyApp.swift:453 |
| `screen.journey.onboarding.profile.root` | MuckJourneyApp.swift:476 |
| `screen.journey.onboarding.root` | MuckJourneyApp.swift:294 |
| `screen.journey.onboarding.welcome.root` | MuckJourneyApp.swift:387 |
| `ui.journey.agreementToggle` | MuckJourneyApp.swift:470 |
| `ui.journey.backButton` | MuckJourneyApp.swift:328 |
| `ui.journey.finishButton` | MuckJourneyApp.swift:345 |
| `ui.journey.nameField` | MuckJourneyApp.swift:460 |
| `ui.journey.nextButton` | MuckJourneyApp.swift:337 |
| `ui.journey.termsButton` | MuckJourneyApp.swift:414 |

## Not verified

The following could not be confirmed in source and is deliberately not claimed:

- resetMockState() (which restores project and inbox data) is only invoked from a button inside SettingsView in this file; no other call site was found

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
