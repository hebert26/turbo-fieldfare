# Test guide — MuckBank

## Order

Position **2** of the order in [`../index.md`](../index.md). MuckBank starts only
after NestMind reaches 100% journey coverage. Do not begin it early.

Target bundle: `com.visionos.muckapps.bank`.

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
- State is in-memory only and reset mode is `relaunch`
  (`/Users/dev-machine/Dev/MuckApps/apps.tsv`), so a relaunch restores initial state.
- No system permission prompts are expected — the built `Info.plist` has zero
  usage-description keys.

## Fixture baseline

`/Users/dev-machine/Dev/MuckApps/journeys/MuckBank.yaml` declares the fixture's own
minimum coverage — `observe`, `primary` (balance-visibility toggle), and
`reset_repeat`. That file is the MuckApps harness format, not VisionCapture input.
Treat it as the floor, not the goal.

## Coverage goals

| # | Goal | Done when |
|---|---|---|
| 1 | Cold launch reaches a proven foreground app | `launch app` returns the app as proven foreground with a cache block |
| 2 | All three tabs reached | Accounts, Cards, and Activity each observed as foreground content |
| 3 | Balance visibility toggled | Toggle mutation carries a truthful verdict and the balance display state is observed to change |
| 4 | Account detail and transaction detail reached | Each detail surface observed |
| 5 | Transfer wizard completed end to end | Source, destination, amount, review, and confirm all traversed; success state observed with a verdict |
| 6 | Transfer validation exercised | At least one invalid state reached and its error observed — same-account and amount validation both exist |
| 7 | A card toggle mutation succeeds | One of the freeze / online-payments / ATM / contactless toggles changed via `execute cached action` with a verdict |
| 8 | A confirmation dialog is handled | Dispute, report-lost, order-replacement, or activity-reset dialog reached and resolved |
| 9 | Slider input exercised | Transfer amount slider or card limit slider moved, with the resulting value observed |
| 10 | Cache lifecycle walked | cold → `ready` → cached action with `cache_used: true` verified → rebuild → revalidation → `revalidate cached action` verified with `fresh_authority_recorded: true` |
| 11 | Reset restores initial state | Relaunch observed to return the app to its starting state |

## Domain journeys

### Tab sweep

Reach each of the three tabs in turn and confirm each root is observed before moving
on.

### Transfer

Start a transfer, pick two different accounts, set an amount, add a memo, review, and
confirm. Deliberately pick the same account once to reach the validation error, then
correct it. Every step is an intent — the runtime picks each target from observed
evidence and the cache manifest.

### Card management

Open a card, change one toggle, adjust the limit, then reach a destructive dialog and
resolve it.

### Dispute

From a transaction detail, raise a dispute and observe its confirmation.

## Generality target

MuckBank is the proof that NestMind's path was generic. The target is a complete
journey with **zero new production code**. If a code change turns out to be needed,
that is a generic gap: stop the journey, take the gap through the normal decision and
review path, then restart the journey clean.

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
