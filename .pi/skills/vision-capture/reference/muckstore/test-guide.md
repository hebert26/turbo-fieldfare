# Test guide — MuckStore

## Order

Position **3** of the order in [`../index.md`](../index.md). MuckStore starts only
after MuckBank is complete, which itself starts only after NestMind reaches 100%
journey coverage. Do not begin it early.

Target bundle: `com.visionos.muckapps.store`.

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
  [`installation.md`](installation.md). No build path is recorded; if the app is not
  installed, ask the customer for the exact one.
- State is in-memory only and reset mode is `relaunch`
  (`/Users/dev-machine/Dev/MuckApps/apps.tsv`), so a relaunch restores initial state.
- No system permission prompts are expected — the built `Info.plist` has zero
  usage-description keys.

## Fixture baseline

`/Users/dev-machine/Dev/MuckApps/journeys/MuckStore.yaml` declares the fixture's own
minimum coverage — `observe`, `primary` (reach the cart), and `reset_repeat`. That
file is the MuckApps harness format, not VisionCapture input. Treat it as the floor,
not the goal.

## Coverage goals

| # | Goal | Done when |
|---|---|---|
| 1 | Cold launch reaches a proven foreground app | `launch app` returns the app as proven foreground with a cache block |
| 2 | Catalog browsed across categories | Each of the four declared categories selected and the grid observed to change |
| 3 | Search exercised | A query typed into the search field and the filtered result observed |
| 4 | Product detail reached | Product detail root observed |
| 5 | Quantity stepper exercised | Stepper value changed and the new value observed |
| 6 | Add to cart succeeds | Add mutation carries a truthful verdict and the confirmation state is observed |
| 7 | Cart reached | Cart root observed with the added item present |
| 8 | Cart form exercised | Per-item quantity, receipt email, shipping option, and save-preference toggle each changed — none of these carry an identifier, so work from observed roles |
| 9 | Checkout completes | Checkout mutation carries a truthful verdict and the success state is observed |
| 10 | Cache lifecycle walked | cold → `ready` → cached action with `cache_used: true` verified → rebuild → revalidation → `revalidate cached action` verified with `fresh_authority_recorded: true` |
| 11 | Grant-expiry recovery proven | Deliberately idle past the observation-grant TTL mid-journey, then prove the public responses alone bring the caller back to a verified action |
| 12 | Reset restores initial state | Relaunch observed to return the app to its starting state |

## Domain journeys

### Browse and search

From the catalog, switch category, then search, and confirm the visible product set
changes each time.

### Purchase

Open a product, set a quantity, add it to the cart, open the cart, fill the checkout
form, and complete checkout. Every step is an intent — the runtime picks each target
from observed evidence and the cache manifest.

### Grant expiry

Mid-journey, stop acting until the observation grant expires. Then recover using only
the refusal code and its next step, and finish with a verified action. This is the
journey that previously died at grant expiry with no usable next step; closing it is
a named goal, not an optional extra.

## Generality target

Like MuckBank, MuckStore should complete with **zero new production code**. If a code
change turns out to be needed, that is a generic gap: stop the journey, take the gap
through the normal decision and review path, then restart the journey clean.

## Exit criteria

- Every coverage goal above is met on the build under test, including goal 11.
- Zero false `verified` and zero false `failed` across the run.
- Every refusal encountered carried a next step that was actually followed to
  recovery.
- Evidence is durable and stamped with the build fingerprint under test.

## Drift rule

Live state can differ from this guide. Observe before every action. If the app has
changed, trust the observed evidence and update this file — never force the app to
match the guide.
