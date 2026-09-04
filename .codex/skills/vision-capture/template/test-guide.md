# Test guide — `<App Name>`

Authoring template. Copy to `reference/<app-folder>/test-guide.md`. Define coverage
goals and domain journeys only. Delete anything you could not verify from source.

**Link note.** The two links below point at the real files from *this* template's
location. After copying into `reference/<app-folder>/`, rewrite them one level
deeper: `../reference/index.md` becomes `../index.md`, and `../SKILL.md` becomes
`../../SKILL.md`.

## Order

Keep one of the two headings below and delete the other.

**If the app is scheduled:** one app at a time. This app is position `<n>` in the
queue recorded in [`../reference/index.md`](../reference/index.md). Do not start it
while an earlier app is unfinished.

**If the app is catalogued only:** say so first and plainly — *Catalogued reference
only. This app is not in the execution queue. Do not start its journey.* Adding a
reference folder never schedules an app; only the owner does that.

## What this file is not

- Not action authority. It never authorises a tap, coordinate, selector, cache
  handle, or claimed outcome.
- Not a coordinate list. Never hardcode coordinates here.
- Not a label contract. A label is observed data, never permission to act.
- Not proof. Only `payload.proof.verdict` from the current response settles an
  outcome.

## Runtime workflow is unchanged

Follow the generic workflow in [`../SKILL.md`](../SKILL.md) exactly:

1. `launch app` first, with the caller's exact `bundle_id` and `udid`.
2. Read the newest cache block before every app-owned mutation.
3. If `cache.state` is `ready`, `revalidation`, or `mixed`, using the exact returned
   capability is mandatory. There is no caller `use_cache` flag.
4. If `cold`, use the exact `observation_grant` flow.
5. Never bypass a usable published cache action with Computer Use merely because this
   guide names a control.
6. Read `payload.proof.verdict` before continuing.

## Preconditions

`<Verified setup this journey needs: installed build, first-run state, permission
state, data state. Cite what proves each.>`

## Coverage goals

| # | Goal | Done when |
|---|---|---|
| 1 | `<capability area reached or exercised>` | `<observable condition, verdict-based>` |

Write goals as capabilities to reach and exercise, not as tap sequences. Each "done
when" must be something a VisionCapture response can prove.

## Domain journeys

### `<journey name>`

`<Ordered intent steps in domain language — "reach the cart", "submit a transfer".
Each step says what to accomplish, not which pixel to hit. The runtime chooses the
target from observed evidence and the cache manifest.>`

## Known blockers

`<Verified blockers only, with the file or receipt that proves each. Delete if none
are proven.>`

## Exit criteria

- Every coverage goal above is met.
- Every mutation carries a truthful `payload.proof.verdict`; zero false `verified`
  and zero false `failed`.
- Every refusal encountered carried a next step that was actually followed to
  recovery.
- Evidence is durable and stamped with the build under test.

## Drift rule

Live state can differ from this guide. Observe before every action. If the app has
changed, trust the observed evidence and update this file — never force the app to
match the guide.
