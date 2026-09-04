# App reference index

Bundle-routing only. Read this after the caller's exact `bundle_id` is known.

Two tables. The first is the execution queue. The second is a catalogue that exists
so context is ready if the owner ever schedules one of those apps — reading it never
puts an app into the queue.

## Execution queue — scheduled

Strict, one app at a time. An app starts only when the one before it is complete.

| Bundle ID | App | Folder | Order |
|---|---|---|---|
| `com.hebertgo.nestmind.debug` | NestMind | [`nestmind/`](nestmind/) | 1 — active |
| `com.visionos.muckapps.bank` | MuckBank | [`muckbank/`](muckbank/) | 2 — after NestMind |
| `com.visionos.muckapps.store` | MuckStore | [`muckstore/`](muckstore/) | 3 — after MuckBank |

## Catalogued reference only — not scheduled

48 MuckApps fixtures, verified but **not in the execution queue**. Do not
start a journey for any of these. They are here for context and for the day the owner
schedules them.

| Bundle ID | App | Folder |
|---|---|---|
| `com.visionos.muckapps.accessibilitylab` | MuckAccessibilityLab | [`muck-accessibility-lab/`](muck-accessibility-lab/) |
| `com.visionos.muckapps.auction` | MuckAuction | [`muck-auction/`](muck-auction/) |
| `com.visionos.muckapps.books` | MuckBooks | [`muck-books/`](muck-books/) |
| `com.visionos.muckapps.budget` | MuckBudget | [`muck-budget/`](muck-budget/) |
| `com.visionos.muckapps.calendar` | MuckCalendar | [`muck-calendar/`](muck-calendar/) |
| `com.visionos.muckapps.clinic` | MuckClinic | [`muck-clinic/`](muck-clinic/) |
| `com.visionos.muckapps.dashboard` | MuckDashboard | [`muck-dashboard/`](muck-dashboard/) |
| `com.visionos.muckapps.delivery` | MuckDelivery | [`muck-delivery/`](muck-delivery/) |
| `com.visionos.muckapps.drawing` | MuckDrawing | [`muck-drawing/`](muck-drawing/) |
| `com.visionos.muckapps.education` | MuckEducation | [`muck-education/`](muck-education/) |
| `com.visionos.muckapps.events` | MuckEvents | [`muck-events/`](muck-events/) |
| `com.visionos.muckapps.files` | MuckFiles | [`muck-files/`](muck-files/) |
| `com.visionos.muckapps.fitness` | MuckFitness | [`muck-fitness/`](muck-fitness/) |
| `com.visionos.muckapps.gamehub` | MuckGameHub | [`muck-game-hub/`](muck-game-hub/) |
| `com.visionos.muckapps.garage` | MuckGarage | [`muck-garage/`](muck-garage/) |
| `com.visionos.muckapps.garden` | MuckGarden | [`muck-garden/`](muck-garden/) |
| `com.visionos.muckapps.health` | MuckHealth | [`muck-health/`](muck-health/) |
| `com.visionos.muckapps.home` | MuckHome | [`muck-home/`](muck-home/) |
| `com.visionos.muckapps.hotel` | MuckHotel | [`muck-hotel/`](muck-hotel/) |
| `com.visionos.muckapps.identity` | MuckIdentity | [`muck-identity/`](muck-identity/) |
| `com.visionos.muckapps.inventory` | MuckInventory | [`muck-inventory/`](muck-inventory/) |
| `com.visionos.muckapps.journal` | MuckJournal | [`muck-journal/`](muck-journal/) |
| `com.visionos.muckapps.journey` | MuckJourney | [`muck-journey/`](muck-journey/) |
| `com.visionos.muckapps.localization` | MuckLocalization | [`muck-localization/`](muck-localization/) |
| `com.visionos.muckapps.mail` | MuckMail | [`muck-mail/`](muck-mail/) |
| `com.visionos.muckapps.meditation` | MuckMeditation | [`muck-meditation/`](muck-meditation/) |
| `com.visionos.muckapps.news` | MuckNews | [`muck-news/`](muck-news/) |
| `com.visionos.muckapps.notes` | MuckNotes | [`muck-notes/`](muck-notes/) |
| `com.visionos.muckapps.pets` | MuckPets | [`muck-pets/`](muck-pets/) |
| `com.visionos.muckapps.pharmacy` | MuckPharmacy | [`muck-pharmacy/`](muck-pharmacy/) |
| `com.visionos.muckapps.photos` | MuckPhotos | [`muck-photos/`](muck-photos/) |
| `com.visionos.muckapps.player` | MuckPlayer | [`muck-player/`](muck-player/) |
| `com.visionos.muckapps.puzzle` | MuckPuzzle | [`muck-puzzle/`](muck-puzzle/) |
| `com.visionos.muckapps.realty` | MuckRealty | [`muck-realty/`](muck-realty/) |
| `com.visionos.muckapps.recipes` | MuckRecipes | [`muck-recipes/`](muck-recipes/) |
| `com.visionos.muckapps.restaurant` | MuckRestaurant | [`muck-restaurant/`](muck-restaurant/) |
| `com.visionos.muckapps.ride` | MuckRide | [`muck-ride/`](muck-ride/) |
| `com.visionos.muckapps.settings` | MuckSettings | [`muck-settings/`](muck-settings/) |
| `com.visionos.muckapps.social` | MuckSocial | [`muck-social/`](muck-social/) |
| `com.visionos.muckapps.studio` | MuckStudio | [`muck-studio/`](muck-studio/) |
| `com.visionos.muckapps.support` | MuckSupport | [`muck-support/`](muck-support/) |
| `com.visionos.muckapps.survey` | MuckSurvey | [`muck-survey/`](muck-survey/) |
| `com.visionos.muckapps.tasks` | MuckTasks | [`muck-tasks/`](muck-tasks/) |
| `com.visionos.muckapps.transit` | MuckTransit | [`muck-transit/`](muck-transit/) |
| `com.visionos.muckapps.travel` | MuckTravel | [`muck-travel/`](muck-travel/) |
| `com.visionos.muckapps.utilities` | MuckUtilities | [`muck-utilities/`](muck-utilities/) |
| `com.visionos.muckapps.warehouse` | MuckWarehouse | [`muck-warehouse/`](muck-warehouse/) |
| `com.visionos.muckapps.weather` | MuckWeather | [`muck-weather/`](muck-weather/) |

## Matching

Match the bundle ID **exactly**. A similar name, a sibling build, or a release variant
of the same app is not a match. The MuckApps fixtures are 50 separate apps
with distinct bundle IDs.

If the caller's bundle ID appears in no row, continue with the generic VisionCapture
workflow in [`../SKILL.md`](../SKILL.md) and invent nothing about the app.

## Read only what the task needs

Read at most one app folder, and inside it only the file the task needs:

| File | Read it only when |
|---|---|
| `installation.md` | Doing installation or `.app`-path work, or after `APP_NOT_INSTALLED`. |
| `overview.md` | You need app context. |
| `test-guide.md` | Planning or executing that app's test journey. |

Never read [`../template/`](../template/) during ordinary app testing. It exists only
to author a reference for a new app.

## What a reference is

Context. Never execution authority, never proof.

- A reference cannot authorise a tap, coordinate, selector, cache handle, or claimed
  outcome.
- If `cache.state` is `ready`, `revalidation`, or `mixed`, using the exact returned
  capability is mandatory. There is no caller `use_cache` flag. If `cold`, use the
  exact observation-grant flow.
- Never bypass a usable published cache action with Computer Use merely because a
  reference names a control.
- Fresh VisionCapture evidence always wins. Where a reference and the current
  response disagree, the response is correct and the reference is stale.

## Install artifacts — build, do not reuse

**No app folder records a prebuilt `.app` path.** A stale artifact passes every check
a document can make — the directory exists, its `CFBundleIdentifier` matches — so
neither proves freshness. Only a fresh build proves its own.

Each app folder records a **verified build command** instead, plus the path that
command writes to. After `APP_NOT_INSTALLED`: run the recorded build, verify the
resulting artifact's `CFBundleIdentifier`, then `install app` with it.

Never invent a build command, never glob DerivedData, and never install from the
recorded artifact path without having just built it — that path is where *your* build
writes, not a shelf to pull from. If the build fails or no build command is recorded,
stop and ask the customer.

## Folder naming

The three scheduled apps keep the folder names they were created with (`nestmind`,
`muckbank`, `muckstore`). Every catalogued app uses kebab-case of its project name
(`MuckGameHub` → `muck-game-hub`).

## Adding a new app

Copy the three files in [`../template/`](../template/) into a new
`reference/<kebab-app-name>/` — `installation.md`, `overview.md`, `test-guide.md` —
fill them from verified sources, then add one row to the correct table above.
`../SKILL.md` needs no change.
