# Overview — NestMind (debug)

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.hebertgo.nestmind.debug` | built `Info.plist` |
| Display name | `NestMind Debug` | built `Info.plist` → `CFBundleDisplayName` |
| Bundle name | `NestMind` | built `Info.plist` → `CFBundleName` |
| Version | `1.0.10` (build `1`) | built `Info.plist` |
| Minimum OS | `26.0` | built `Info.plist` → `MinimumOSVersion` |

## Purpose

A privacy-first iOS app for saving, organising, and finding personal knowledge —
bookmarks, videos, and todos — using on-device AI and semantic search
(`/Users/dev-machine/Dev/NestMind/README.md`,
`/Users/dev-machine/Dev/NestMind/project-files/docs/reference/app-description.md`).

## Capability areas

Source directories under `/Users/dev-machine/Dev/NestMind/NestMind/Features/`:

| Area | What it covers | Verified from |
|---|---|---|
| Onboarding | First-run flow before the main app is reachable | `Features/Onboarding/` |
| Library | Bookmarks, Playlists, Todos | `Features/Library/{Bookmarks,Playlist,Todos}/` |
| Assistant | On-device chat assistant | `Features/Assistant/` |
| Search | Search surface | `Features/Search/` |
| Account | Profile and account surface | `Features/Account/` |
| Knowledge hub | Dashboard entry surface | `Features/KnowledgeHubDashboard.swift` |

Router-declared destinations (`App/Router.swift:12-15, 25-27`): `knowledge`
(titled `Knowledge`, and `Home` in the second enum), `assistant`, `profile`,
`search`.

## System permission prompts to expect

Two usage-description keys in the built `Info.plist`:

- `NSSpeechRecognitionUsageDescription`
- `NSMicrophoneUsageDescription`

Both are requested in source:

- `Core/Platform/Speech/SpeechRecognitionService.swift:46` (`SFSpeechRecognizer.requestAuthorization`)
  and `:56` (`AVAudioApplication.requestRecordPermission`)
- `Features/Onboarding/PermissionSheet.swift:272, :280`
- `Features/Onboarding/PermissionSheetView.swift:235, :248`

No other permission usage keys are present. Do not assume any other prompt.

## State and first-run gates

`App/NestMindApp.swift` selects the root view in this order:

1. `@AppStorage("hasCompletedOnboarding")` is `false` → `OnboardingFlowView` (`:12, :47`).
   It is force-set to `true` when existing profile data is detected (`:32-33`).
2. Otherwise `@AppStorage("activeProfileId")` is empty → `ProfilesView` (`:50`).
3. Otherwise → `ContentView` (`:53`).

Onboarding pages present in `Features/Onboarding/Pages/`: `WelcomePageView`,
`FeaturesPageView`, `PrivacyPageView`, `PermissionsPageView`, `DemoReadyPageView`.

## Source-declared accessibility landmarks

Identifiers declared in source — use them to **recognise** a screen in observed
evidence and to plan coverage. They are not selectors to send and not action
authority.

| Identifier | Declared in |
|---|---|
| `screen.onboarding.flow.root` | `Features/Onboarding/OnboardingFlowView.swift:120` |
| `ui.onboarding.continueButton` | `Features/Onboarding/OnboardingFlowView.swift:319` |
| `ui.onboarding.backButton` | `Features/Onboarding/OnboardingFlowView.swift:341` |
| `screen.library.bookmarks.root` | `Features/Library/Bookmarks/BookmarksHubView.swift:68` |
| `screen.library.playlists.root` | `Features/Library/Playlist/PlaylistsHubView.swift:52` |
| `screen.library.todos.root` | `Features/Library/Todos/TodosHubView.swift:169` |
| `bookmarks-view-mode-switcher` | `Features/Library/Bookmarks/Views/BookmarksView.swift:366` |
| `ui.todos.titleField` | `Features/Library/Todos/Views/AddTodosView.swift:77` |
| `ui.todos.tagPickerButton` | `Features/Library/Todos/Views/AddTodosView.swift:227` |
| `ui.todos.newTagNameField` | `Features/Library/Todos/Views/TodoTagPickerSheet.swift:159` |
| `ui.todos.createTagButton` | `Features/Library/Todos/Views/TodoTagPickerSheet.swift:178` |

Unstable: `library-row-\(title)`
(`Features/Library/Playlist/component/PlaylistButtonView.swift:59`) is composed from a
runtime title. It is not a reliable landmark.

That is every `accessibilityIdentifier` site in the tree — 12 in total. Most screens
carry none, so expect to work from generic accessibility roles and observed
structure.

## Known documentation conflict

`project-files/docs/reference/app-description.md` states iOS 18+. The built artifact
declares `MinimumOSVersion 26.0`, and `README.md` states iOS 26+. Trust the built
artifact.

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
