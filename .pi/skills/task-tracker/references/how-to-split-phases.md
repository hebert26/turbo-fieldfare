# How to split phases

Read this before you write any phase. Getting the split wrong is the mistake that ruins the whole tracker.

## The rule

**A phase separates work. One phase is one piece of work. Never mix two.**

## The test

Finish this sentence in plain words:

> "When this phase is done, ______ works."

If you can finish it in one short clause, it is one phase.

If you need the word **and**, it is two phases. Split it.

If you cannot finish it at all, it is not a phase. See "Things that are not phases" below.

## Name it after what works

The name says what a person can do, or what the product can now do. Plain words. No jargon.

| Bad name | Why it is bad | Good name |
|---|---|---|
| `Gate 2 exact identity and managed device sets` | "Gate 2" means nothing. Two things joined by "and". | `The app finds one exact device` / `The app runs in its own device set` |
| `Phase 3: Implementation` | A stage, not a piece of work. Nothing "works" when implementation is done. | Name the thing being implemented. |
| `Backend work` | A layer, not a piece of work. | `Saved notes survive a restart` |
| `Notch recording, UI, MCP, and lifecycle` | Four things. Three "ands". | Four phases. |
| `Polish and regression testing` | Two things, and neither is a piece of work. | Fold the checks into each phase's done-when. |
| `Team A tasks` | A team, not a piece of work. | Name what they build. |

## Size

A phase must be small enough that **one coverage table can honestly cover it**.

If the coverage table would need more than about eight rows, the phase is doing too much. Split it.

A phase with one task is fine. A phase with fifteen tasks is a warning.

## Order

Order phases by what depends on what, not by team, not by layer, not by difficulty.

Each phase names what it depends on. A phase can only start when everything it depends on is done.

Two phases that depend on nothing from each other can run at the same time. Say so.

## Things that are not phases

Nothing "works" when these finish, so they are not phases. They already have a home.

| Thing | Where it lives |
|---|---|
| Getting the tracker approved | The `status` field in the tracker header. |
| Choosing an architecture | The Decisions table in the implementation document. |
| Writing a document | A task inside the phase whose work it describes. |
| Research or discovery | A phase **only** if the answer is the deliverable. Otherwise a task. |
| "Final review" / "sign-off" | The `done when` block of the last phase. |
| Releasing | A phase if real work happens. Otherwise the document status. |

## Discovery is allowed, once

If you genuinely cannot name the later phases until something unknown is answered, make one discovery phase and stop there.

> "When this phase is done, we know whether the private framework exposes a frame callback."

Then write the rest of the phases after it answers. Do not guess five phases you cannot name yet.

## Worked split

Work item: *put a live device preview in the app and record it.*

**Wrong** - layers and gates:

```
P1 Architecture
P2 Gate 1 helper process
P3 Gate 2 identity
P4 Gate 3 transport
P5 Recording, UI, API, and lifecycle
P6 Testing and release
```

Nothing "works" at the end of P1, P2, P3 or P4 - they are slices of one capability. P5 is four phases wearing one name. P6 is a stage.

**Right** - pieces of work:

```
P1 The app shows one exact device, live
      tasks: helper process, identity lookup, frame transport
P2 The live device appears inside the notch
P3 The notch records to one video file
P4 The user can change preview size and frame rate
P5 The API can start, stop and cancel a recording
```

Each name finishes the sentence. Each has its own tasks, its own acceptance, its own coverage table, its own done-when. The old P2, P3 and P4 became **tasks** inside P1, because on their own nothing works.

## Check yourself

Before you write the tracker, read your phase names out loud in order. If any of them:

- contains "and" -> split it
- contains a number that is not an ID (`Gate 2`, `Phase 3 of transport`) -> rename it
- names a stage (`design`, `build`, `test`, `polish`, `release`) -> rename it or fold it in
- names a layer (`backend`, `UI`, `infra`) -> rename it
- you cannot finish the sentence for -> it is not a phase

Fix it now. Fixing it after tasks are written costs ten times more.
