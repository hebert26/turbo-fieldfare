# Overview — MuckEducation

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.education` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/EducationModels.swift`, `Sources/EducationScreens.swift`, `Sources/EducationStore.swift`, `Sources/MuckEducationApp.swift` (267 lines) | file listing |

## Purpose

A mock online-course app with a unit list, a reorderable lesson outline, and a single-question quiz with async grading.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `NavigationSplitView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Course units | a sidebar list of units with completion checkmarks | EducationRootView, CourseUnit |
| Lesson outline | reorderable outline plus a drag-and-drop target and a disclosure group | LessonPane, OutlineDropTarget |
| Quiz | a single multiple-choice question with submit and async grading status | QuizPane, EducationStatusPane |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `disclosure,radio,drag,progress`; navigation
`split-course-quiz`; motion `completion-burst`; accessibility profile
`P`; reset mode `muck_reset`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory @Observable EducationStore, resetMuckFixture(), no persistence. Reset mode is `muck_reset` (`apps.tsv`).

## Source-declared accessibility landmarks

15 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `education.cancelButton` | EducationScreens.swift:139 |
| `education.completeLessonButton` | EducationScreens.swift:69 |
| `education.editOutlineButton` | EducationScreens.swift:38 |
| `education.empty` | EducationScreens.swift:145 |
| `education.error` | EducationScreens.swift:144 |
| `education.finishButton` | EducationScreens.swift:138 |
| `education.lessonDisclosure` | EducationScreens.swift:67 |
| `education.loading` | EducationScreens.swift:141 |
| `education.moveExplainButton` | EducationScreens.swift:60 |
| `education.outlineDropTarget` | EducationScreens.swift:95 |
| `education.progress` | EducationScreens.swift:25 |
| `education.resetButton` | EducationScreens.swift:39 |
| `education.root` | EducationScreens.swift:41 |
| `education.submitButton` | EducationScreens.swift:115 |
| `education.success` | EducationScreens.swift:143 |

Unstable — composed from runtime values, not reliable landmarks: `answerAccessibilityID(choice`, `outlineAccessibilityID(item`, `unit.accessibilityID`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
