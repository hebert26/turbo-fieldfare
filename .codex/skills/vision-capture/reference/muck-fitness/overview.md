# Overview — MuckFitness

Catalogued reference only. Not scheduled — see [`../index.md`](../index.md).

Context only. Never execution authority, never proof.

## Identity

| Fact | Value | Verified from |
|---|---|---|
| Bundle ID | `com.visionos.muckapps.fitness` | `project.pbxproj`, `project.yml`, `apps.tsv` (all three agree) |
| Deployment target | iOS 17.0 | `project.yml` |
| Device families | `1,2` (1 = iPhone, 2 = iPad) | `project.yml` |
| Source | `Sources/MuckFitnessApp.swift` (1230 lines) | file listing |

## Purpose

A mock fitness-tracking app with daily activity rings, a workout logger with a live timer, a workout history, and adjustable goals, across four tabs.

It is a MuckApps fixture built for VisionOS automation and UI interaction testing
(`/Users/dev-machine/Dev/MuckApps/README.md`), not a customer product.

## Capability areas

Root is a `TabView`.

| Area | What it covers | Source anchors |
|---|---|---|
| Today dashboard | activity rings, stat tiles, and quick-log actions | TodayView, ActivityRingsCluster, StatTile |
| Workout logging | an exercise picker, set/rep/weight steppers, and a start/pause/resume timer | WorkoutView, LogActivitySheet |
| History | a seven-day bar chart plus a swipe-to-delete workout list with a detail screen | HistoryView, HistoryBarChart, WorkoutDetailView |
| Goals | sliders and steppers for daily targets, presets, and a restore-mock-data alert | GoalsView |

Catalog row (`/Users/dev-machine/Dev/MuckApps/apps.tsv`): controls `tabs,timer,stepper`; navigation
`tabs-sheet`; motion `animated-progress`; accessibility profile
`B`; reset mode `relaunch`;
implementation state `implemented`.

## System permission prompts to expect

None. The project `Info.plist` declares zero usage-description keys, and no
permission-requesting API appears in source.

## State and reset behaviour

In-memory ObservableObject FitnessStore (@Published), no reset hook found in this file. Reset mode is `relaunch` (`apps.tsv`).

## Source-declared accessibility landmarks

49 static identifier value(s). Use them to **recognise** a
screen in observed evidence and to plan coverage. They are not selectors to send and
not action authority.

| Identifier | Declared at |
|---|---|
| `detail.deleteButton` | MuckFitnessApp.swift:1094 |
| `detail.notes` | MuckFitnessApp.swift:1080 |
| `detail.repeatButton` | MuckFitnessApp.swift:1089 |
| `goals.exerciseProgress` | MuckFitnessApp.swift:1185 |
| `goals.exerciseStepper` | MuckFitnessApp.swift:1134 |
| `goals.intensitySlider` | MuckFitnessApp.swift:1146 |
| `goals.metricToggle` | MuckFitnessApp.swift:1150 |
| `goals.moveProgress` | MuckFitnessApp.swift:1180 |
| `goals.moveSlider` | MuckFitnessApp.swift:1128 |
| `goals.presetHard` | MuckFitnessApp.swift:1173 |
| `goals.presetLight` | MuckFitnessApp.swift:1163 |
| `goals.presetStandard` | MuckFitnessApp.swift:1168 |
| `goals.remindersToggle` | MuckFitnessApp.swift:1156 |
| `goals.resetTodayButton` | MuckFitnessApp.swift:1199 |
| `goals.restoreButton` | MuckFitnessApp.swift:1204 |
| `goals.sessionProgress` | MuckFitnessApp.swift:1190 |
| `goals.sessionRingToggle` | MuckFitnessApp.swift:1153 |
| `goals.sessionsStepper` | MuckFitnessApp.swift:1139 |
| `history.addSampleButton` | MuckFitnessApp.swift:1009 |
| `history.categoryPicker` | MuckFitnessApp.swift:961 |
| `history.chart` | MuckFitnessApp.swift:945 |
| `logSheet.cancelButton` | MuckFitnessApp.swift:596 |
| `logSheet.confirmButton` | MuckFitnessApp.swift:589 |
| `logSheet.minutesSlider` | MuckFitnessApp.swift:578 |
| `logSheet.typePicker` | MuckFitnessApp.swift:571 |
| `root.tabview` | MuckFitnessApp.swift:265 |
| `tab.goals` | MuckFitnessApp.swift:263 |
| `tab.history` | MuckFitnessApp.swift:258 |
| `tab.today` | MuckFitnessApp.swift:248 |
| `tab.workout` | MuckFitnessApp.swift:253 |
| `today.addMinutesButton` | MuckFitnessApp.swift:486 |
| `today.confirmationBanner` | MuckFitnessApp.swift:427 |
| `today.logActivityButton` | MuckFitnessApp.swift:465 |
| `today.movePercent` | MuckFitnessApp.swift:323 |
| `today.quickBurnButton` | MuckFitnessApp.swift:476 |
| `today.resetButton` | MuckFitnessApp.swift:495 |
| `workout.categoryPicker` | MuckFitnessApp.swift:725 |
| `workout.exercisePicker` | MuckFitnessApp.swift:733 |
| `workout.finishButton` | MuckFitnessApp.swift:684 |
| `workout.notesField` | MuckFitnessApp.swift:669 |
| `workout.repsStepper` | MuckFitnessApp.swift:763 |
| `workout.savedBanner` | MuckFitnessApp.swift:657 |
| `workout.selectedExerciseName` | MuckFitnessApp.swift:740 |
| `workout.setsStepper` | MuckFitnessApp.swift:758 |
| `workout.timerPhase` | MuckFitnessApp.swift:789 |
| `workout.timerPrimaryButton` | MuckFitnessApp.swift:806 |
| `workout.timerStopButton` | MuckFitnessApp.swift:818 |
| `workout.timerValue` | MuckFitnessApp.swift:797 |
| `workout.weightStepper` | MuckFitnessApp.swift:769 |

Unstable — composed from runtime values, not reliable landmarks: `"history.bar.\(sample.label`, `identifier`

## Evidence rule

Live state can differ from this file. Observe before every action. Where this file
and a current VisionCapture response disagree, the response is correct and this file
is stale.
