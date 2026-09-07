# Test guide — MuckIdentity

## Not scheduled

**Catalogued reference only.** This app is not in the execution queue. The queue is
strict and one app at a time — NestMind first, then MuckBank, then MuckStore — as
recorded in [`../index.md`](../index.md). Do not start this app's journey. This file
exists so the context is ready if and when the owner schedules it.

Target bundle: `com.visionos.muckapps.identity`.

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
- Reset mode is `muck_reset` (`/Users/dev-machine/Dev/MuckApps/apps.tsv`).
- No system permission prompts are expected — zero usage-description keys and
  no permission-requesting API in source.

## Fixture baseline

`/Users/dev-machine/Dev/MuckApps/journeys/MuckIdentity.yaml` declares the fixture's own minimum coverage — `observe`, `primary`, `continue`, `reset_repeat`.
It asserts `screen.identity.root`, `state.identity.complete`, `ui.identity.accountNameField`, `ui.identity.continueButton`, `ui.identity.passcodeField`, `ui.identity.verificationCodeField`, `ui.identity.verifyButton`.
That file is the MuckApps harness format, not VisionCapture input. Treat it as
the floor, not the goal.

## Coverage goals

| # | Goal | Done when |
|---|---|---|
| 1 | Cold launch reaches a proven foreground app | `launch app` returns the app as proven foreground with a cache block |
| 2 | Reach and exercise **Credentials step** | the area is observed, and one mutation there carries a truthful `payload.proof.verdict` |
| 3 | Reach and exercise **Code verification** | the area is observed, and one mutation there carries a truthful `payload.proof.verdict` |
| 4 | Reach and exercise **Recovery** | the area is observed, and one mutation there carries a truthful `payload.proof.verdict` |
| 5 | Reach and exercise **Completion** | the area is observed, and one mutation there carries a truthful `payload.proof.verdict` |
| 6 | Cache lifecycle walked | cold → `ready` → cached action with `cache_used: true` verified → rebuild → revalidation → `revalidate cached action` verified with `fresh_authority_recorded: true` |
| 7 | Reset restores initial state | relaunch observed to return the app to its starting state |

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
