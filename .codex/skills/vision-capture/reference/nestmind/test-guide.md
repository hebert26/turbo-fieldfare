# Test guide — NestMind (debug)

## Order

Position **1** of the order in [`../index.md`](../index.md). NestMind is the active
app and must reach 100% journey coverage before MuckBank or MuckStore starts.

Target bundle: `com.hebertgo.nestmind.debug`. The release bundle
`com.hebertgo.nestmind` is never the target.

## What this file is not

- Not action authority. It never authorises a tap, coordinate, selector, cache
  handle, or claimed outcome.
- Not a coordinate list.
- Not a label contract. A label is observed data, never permission to act.
- Not proof. Only `payload.proof.verdict` from the current response settles an
  outcome.

## Runtime workflow is unchanged

Follow [`../../SKILL.md`](../../SKILL.md) exactly:

1. `launch app` first, with the caller's exact `bundle_id` and `udid`.
2. Read the newest cache block before every app-owned mutation.
3. `ready` / `revalidation` / `mixed` → using the exact returned capability is
   mandatory. There is no caller `use_cache` flag.
4. `cold` → use the exact `observation_grant` flow.
5. Never bypass a usable published cache action with Computer Use merely because this
   guide names a control.
6. Read `payload.proof.verdict` before continuing.

## Preconditions

- The build under test is installed on the caller's target Simulator — see
  [`installation.md`](installation.md).
- For a first-run journey, the app must actually be at first run. The gate is
  `@AppStorage("hasCompletedOnboarding")` (`App/NestMindApp.swift:12`), and it is
  force-set to `true` when profile data already exists (`:32-33`). A previously
  driven Simulator will skip onboarding.
- Speech Recognition and Microphone permission state decides whether the system
  prompts appear at all. Record the state you started from.

## Coverage goals

| # | Goal | Done when |
|---|---|---|
| 1 | Cold launch reaches a proven foreground app | `launch app` returns the app as proven foreground with a cache block |
| 2 | First-run onboarding flow is traversed | Onboarding root observed, then a later root observed after forward progress, each mutation carrying a verdict |
| 3 | Both system permission prompts are handled | Each prompt observed and resolved, with the app proven resumed afterwards |
| 4 | Profile gate is passed | `ProfilesView` state cleared — `activeProfileId` no longer empty, evidenced by reaching the main shell |
| 5 | Every router destination is reached | `knowledge`/`Home`, `assistant`, `profile`, `search` each observed as foreground content |
| 6 | Every Library area is reached | Bookmarks, Playlists, and Todos roots each observed |
| 7 | A real mutation succeeds with a truthful verdict | A todo is created and its creation is proven by `payload.proof.verdict` |
| 8 | A second, structurally different mutation succeeds | A tag is created through the tag picker, proven the same way |
| 9 | Cache lifecycle is walked | cold recording → revisit shows `ready` → cached action with `cache_used: true` verified → rebuild → revalidation manifest → `revalidate cached action` verified with `fresh_authority_recorded: true` |
| 10 | App-resume proof after interruption | The app is proven foreground again after each interruption |

## Domain journeys

### First run

Reach the app from a cold, never-driven install: get through onboarding, resolve the
permission prompts the app raises, choose or create a profile, and land in the main
shell. Steps are intents — the runtime picks each target from observed evidence and
the cache manifest.

### Library round trip

From the main shell, reach each Library area in turn (bookmarks, playlists, todos)
and confirm each root is observed before moving on.

### Todo mutation

Reach the todos area, open the add-todo surface, enter a title, and commit. Then open
the tag picker, create a tag, and commit. Both are real mutations and both need a
verdict.

### Assistant and search

Reach the assistant destination and the search destination, and exercise each with
one input. Text entry uses the typed Driver `type`/`enter` request.

## Current execution source

Use the exact task ID, coverage request, bundle ID, device, and receipt directory
supplied by the coordinator. Confirm every runtime capability against current
VisionCapture responses; never use a historical receipt as current truth.

## Exit criteria

- Every coverage goal above is met on the build under test.
- Zero false `verified` and zero false `failed` across the run.
- Every refusal encountered carried a next step that was actually followed to
  recovery.
- Evidence is durable and stamped with the build fingerprint under test.

## Drift rule

Live state can differ from this guide. Observe before every action. If the app has
changed, trust the observed evidence and update this file — never force the app to
match the guide.
