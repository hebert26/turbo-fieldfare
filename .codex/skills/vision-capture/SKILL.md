---
name: vision-capture
description: "Drive any iOS Simulator app through VisionCapture's single public MCP tool mcp__vision-capture__execute. Use for driver-only, preview-only, and notch-recording modes; external evidence capture; app lifecycle; Discovery-cache actions; native alerts; Computer Use; proof verdicts; and technical refusals."
---

# VisionCapture MCP Orchestration

Operate any iOS Simulator app — including apps never seen before — through the
one public tool `mcp__vision-capture__execute`. Carry the caller-selected
`bundle_id` and exact `parameters.udid` on every normal app operation.
Preview-only control requests are the deliberate exception: they carry neither
field. Never hardcode an app name, bundle ID, screen name, label, journey,
permission type, locale, or target coordinate: use placeholders such as
`<bundle-id>` and `<udid>`, and facts observed at runtime.

## Authority

The caller — the model or operator using this skill — owns test intent, the
meaning and risk of every screen and control, and the selection of each action.
VisionCapture owns exact execution and truthful evidence (ADR 0044).

- Select actions from your own reading of observed evidence. Any real observed
  control is a valid target; add no safety, risk, or meaning filter, and expect
  none from VisionCapture.
- Expect refusals to be technical only — identity, binding, staleness,
  capability, delivery — and never mistake one for a safety verdict or success.
  Recover only through the named contract path; otherwise preserve the response
  and fail closed.
- `askQuestion` appears in every response and is informational only. It never
  pauses, gates, or stops the run.

## Efficient exploration companion

When exploring an unfamiliar app, testing without a predefined screen path, or
deciding what to do when the next action is unknown, read and follow
[`vision-capture-efficient-exploration`](../vision-capture-efficient-exploration/SKILL.md)
before the first MCP request. That companion controls only the observation
order: newest cache block, then accessibility description, then a viewed
screenshot when the earlier evidence is not enough. This skill remains
authoritative for every MCP request, capability, grant, lane, refusal, proof,
and stop rule.

## Choose one public product mode

Choose one mode before the first MCP request and state it in the task brief. These are the only public product modes.
They are mutually exclusive on the live-notch admission surface: a preview-only session cannot start while Recording
V2 owns that surface, and Recording V2 owns its own live preview. Do not silently add, remove, or switch modes.

| Mode | Visible result | Product video | Route | Lifecycle |
|---|---|---|---|---|
| Driver-only | Ordinary app observation and actions; no notch preview | None | Default route | Normal MCP operations; no mode start/status/continue/stop/cancel block |
| Preview-only | One exact managed, already-booted Simulator appears in the notch | None. It opens no recording writer and creates no MP4. | Exact managed route | Control-only `manage preview`: start, status, stop, or cancel by `preview_id` |
| Notch recording | Recording V2 owns the live notch preview for one run | One completed `kind:"notch"` MP4 only after successful finalization | Recording's managed route | Normal-operation start and continue; control-only status, one stop, or cancel by `recording_id` |

### Common prerequisites

- VisionCapture.app must be running. Use only `mcp__vision-capture__execute`, never raw HTTP.
- Preserve any caller-selected app identity for normal app work and the exact device identity for every mode. Use only
  runtime-observed values or placeholders such as `<bundle-id>` and `<udid>` in examples. Never substitute another
  device, a default route, or an app-specific label.
- For a fresh app journey, use the fresh-cycle proof below before app work begins. That applies to driver-only and
  notch-recording app operations. Preview-only has no app operation, `bundle_id`, or `parameters`; its own lifecycle
  begins with `manage preview`.
- Each mode has its own lifecycle below. A common prerequisite is not permission to send another mode's control request.

### Driver-only

Use ordinary public MCP operations with neither a `preview` nor `recording` object. The default route drives the
caller-selected app and device. It creates neither a notch preview nor product video.

Copyable example:

```json
{"request":"launch app","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>"}}
```

On a healthy installed app, read `launch.launch_outcome` in the returned top-level `launch` object before claiming
app progress and retain the newest `cache` block for a later app action. A driver-only response has no `preview` or
`recording` result. Continue with the cache, alert, proof, and refusal rules below. There is no driver-only control
stop or cancel: finish the caller-selected normal operations deliberately.

### Preview-only

Preview-only shows one exact managed, already-booted Simulator in the notch. It is a closed, control-only schema
version 1 request: start with exactly one managed UDID and no `bundle_id`, `parameters`, `mode`, `session_id`, or
`recording`. Retain the returned `preview_id`. Preview-only never opens a recording writer or publishes an MP4,
`recording_id`, `run_directory`, or `videos` field.

1. Start one preview:

   ```json
   {"request":"manage preview","preview":{"schema_version":1,"command":"start","udids":["<managed-udid>"]}}
   ```

   A successful response contains `preview.accepted:true`, a `preview_id`, the requested identity, and a lifecycle
   state such as `starting` or `streaming`. Keep that exact `preview_id`; it is the only bearer for later controls.

2. Read current lifecycle state when needed:

   ```json
   {"request":"manage preview","preview":{"schema_version":1,"command":"status","preview_id":"<preview-id>"}}
   ```

3. End a successful preview once:

   ```json
   {"request":"manage preview","preview":{"schema_version":1,"command":"stop","preview_id":"<preview-id>"}}
   ```

   The stop response is terminal preview truth. Confirm its accepted terminal state and that it still contains no
   recording or artifact fields. Do not use `cancel` as a successful stop. `cancel` is the failure or abandonment
   path and publishes a cancelled terminal state; preserve that result rather than recasting it as normal completion.

### Notch recording

Use Recording V2 when the caller needs a live notch preview and a completed product recording. It uses schema version
2, starts on a normal app operation, accepts one exact selected UDID, and produces a `kind:"notch"` MP4 only after a
successful terminal stop. Do not start preview-only alongside it: Recording V2 owns the live preview surface.

1. Start on a normal operation, normally `launch app`, with the same exact UDID in `parameters.udid` and
   `recording.udids`:

   ```json
   {"request":"launch app","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>"},"recording":{"schema_version":2,"command":"start","udids":["<udid>"]}}
   ```

   A successful start returns `recording.accepted:true`, `state:"preparing"`, a `recording_id`,
   `paired_operation_started:false`, and `paired_operation_outcome:"not_started"`. The supplied normal operation did
   not run. Save the ID and do not claim launch or app progress.

2. Poll only this control request until `state:"recording"` or a terminal state. It forbids `bundle_id`,
   `parameters`, `mode`, and `session_id`:

   ```json
   {"request":"manage recording","recording":{"schema_version":2,"command":"status","recording_id":"<recording-id>"}}
   ```

3. When state is `recording`, resend the original normal operation with `continue`:

   ```json
   {"request":"launch app","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>"},"recording":{"schema_version":2,"command":"continue","recording_id":"<recording-id>"}}
   ```

   Carry that same `continue` block on every later paired normal app, cache, alert, screenshot, and lifecycle
   operation. Keep `parameters.udid` on the accepted recording identity. If the response explicitly discloses a bound
   managed UDID for later operations, use that exact value. Otherwise keep the selected UDID. Never guess or switch
   routes. Read recording truth (`accepted`, `state`, `has_known_failure`) separately from paired operation truth
   (`isError`, `launch_outcome`, `paired_operation_started`, and `paired_operation_outcome`).

4. Stop exactly once with the control-only request:

   ```json
   {"request":"manage recording","recording":{"schema_version":2,"command":"stop","recording_id":"<recording-id>"}}
   ```

   This terminal stop response is final recording truth. Pass only when it returns `accepted:true`,
   `state:"completed"`, no known failure, a `run_directory`, and one complete `videos` entry with `kind:"notch"` and
   a real path. Verify that path exists and is non-empty. Do not send a second status lookup or a second stop.
   `cancel` is abandonment or failure, not normal completion.

### External single-device capture is evidence workflow, not a product mode

Use coordinator-owned `xcrun simctl io <exact-udid> recordVideo` only when the caller requests an external,
phone-viewable record of a journey. The coordinator starts it before the live tester, stops it with SIGINT after the
tester, retains the full file in the receipt directory, and creates
`Project-files/videos/<YYYY-MM-DD_HHMMSS>__<App>__<Device>/journey-phone.mp4`.

The live tester receives no capture command. Its MCP requests stay in the already selected product mode. This video
proves what appeared on that Simulator; it is not a VisionCapture recording receipt and is not a fourth public mode.

## Driver-only and notch app work: start and observe

This section applies to normal app operations in driver-only and notch-recording modes. It does not add a launch,
bundle ID, parameters, cache action, or other app work to the preview-only control lifecycle.

### Fresh live-test cycle

Ordinary observation may attach to an already running proven app. A live test,
post-fix validation, or acceptance run is stricter: it must start from a fresh
installation and the first app screen.

1. Send `launch app` first. The fresh-cycle precondition is proven only when it
   returns `APP_NOT_INSTALLED`.
2. If launch succeeds because the app is installed, stop. VisionCapture has no
   public `uninstall app` request. Ask the customer/operator to delete that app
   from the selected Simulator. Never substitute raw `simctl`, raw HTTP, another
   automation tool, or an invented MCP request.
   A newly disclosed or auto-created managed clone is not fresh-cycle proof by
   itself: clone creation may preserve the target app and its data. Require the
   same `APP_NOT_INSTALLED` proof on the exact device that will run the journey.
3. Once `APP_NOT_INSTALLED` is proven — whether from the first `launch app` or
   from a fresh one after the operator deleted the app; do not repeat a launch
   that already proved it — **build the app from its project** using the build command recorded in the matching
   reference. No reference records a prebuilt `.app`: a stale artifact passes
   every check a document can make, and only a fresh build proves its own
   freshness. Verify the built artifact's `CFBundleIdentifier` equals the
   requested bundle ID, install that artifact through VisionCapture, then launch
   once. If the build fails, or no reference records a build command, stop and
   ask the customer — never invent a build command, never glob DerivedData,
   never fall back to an older artifact.
4. If a live cycle exposes a VisionCapture defect, preserve it and end the
   cycle. After the accepted VisionCapture fix is deployed, delete/reinstall the
   target app and restart the whole journey from the first screen. Never resume
   from the failed screen.

This fresh-cycle rule applies to every app. App-specific references cannot
weaken it.

1. VisionCapture.app must be running (health: `http://localhost:8766/health`).
   Automate only through the registered MCP tool, never raw HTTP.
2. Get the booted Simulator UDID and target bundle ID from the caller or the
   runtime (`xcrun simctl list devices booted`). Never guess either.
3. The first MCP operation for every app run is `launch app` with that exact
   `bundle_id` and `udid`. This validates installation on that Simulator. If
   the app is already the proven foreground app, `launch app` attaches without
   restarting it. On success, keep the returned session and newest cache block.
4. If `launch app` returns `APP_NOT_INSTALLED`, stop navigation because no app
   action can run. Then take exactly one path:
   - If the exact existing `.app` bundle path is known from the caller or a
     proven build result, send `install app` with that absolute `app_path` and
     the same `udid`. Never guess a path or install a merely similar app.
     Require the returned `bundle_id` to equal the requested bundle ID, then
     send one new `launch app` request.
   - If no exact `.app` path is known, ask the customer to provide the path or
     install the app on that Simulator. Do not continue until installation is
     confirmed, then send one new `launch app` request.
5. When you only need to look after launch, send a plain `take a screenshot` or
   `describe screen` with `bundle_id` and `udid`. Do not call `inspect cache`
   and do not attach an `observation_grant`. A plain read gives evidence; it
   never gives authority to act.
6. Before choosing an app action, read the newest valid `cache` block returned
   by `launch app`, a terminal action response, or `inspect cache`. If no
   current block exists, call `inspect cache`. It also accepts a typed
   `desired_action` — exactly `{action:"tap",selector,role}` or
   `{action:"set_boolean",selector,role:"switch",desired_state:<bool>}` — to
   ask about one intended action.
7. **View every screenshot you rely on.** Open the PNG in the runtime image
   viewer before deriving anything from it. A file path, OCR text, or a prose
   description is not a viewed screenshot.

## App reference — progressive disclosure

Optional local context about specific apps. It changes nothing else in this
skill: launch-first, `APP_NOT_INSTALLED` handling, mandatory cache use, lane
selection, Computer Use rules, and the proof checklist all stay exactly as
written, and the runtime workflow stays app-agnostic.

Load in this order, and load nothing else:

1. Read [`reference/index.md`](reference/index.md) as soon as you have either
   the caller's exact `bundle_id` or the app's name. The index maps both, so a
   caller who named only the app resolves the bundle ID there. Never use a bundle
   ID the index does not list unless the caller gave it to you directly.
2. If that bundle ID maps to an app, read **only** that app's folder.
3. `reference/<app>/installation.md` — only for installation or `.app`-path
   work, or after `APP_NOT_INSTALLED`.
4. `reference/<app>/overview.md` — only when app context is needed.
5. `reference/<app>/test-guide.md` — only when planning or executing that app's
   test journey.
6. Never read [`template/`](template/) during ordinary app testing. It exists
   only to author a reference for a new app.
7. If the bundle ID matches no entry, continue with the generic workflow in this
   skill and invent nothing about the app.

**Read references before you hold live authority.** Reading takes time, and a
handle can go stale while you read — a grant returned by `launch app` has been
observed dead on first use after nothing but local file reads. So do the reference
reading up front: before `launch app`, or immediately after a terminal response and
before you request the next block. Never leave a live handle idle while you open a
file. If you must read mid-run, discard the handle first and `inspect cache` again
afterwards.

A reference is context, never execution authority and never proof:

- It cannot authorise a tap, coordinate, selector, cache handle, or claimed
  outcome. Only the current public VisionCapture response and its
  `payload.proof.verdict` can.
- `cache.state` `ready`, `revalidation`, or `mixed` still means the exact
  returned capability is mandatory; `cold` still means the exact
  observation-grant flow. There is no caller `use_cache` flag.
- A reference naming a control is never a reason to bypass cache with Computer
  Use.
- Fresh VisionCapture evidence always wins. Where a reference and the current
  response disagree, the response is correct and the reference is stale.

## Cache is mandatory for app actions

Cache use is not an optional optimisation. There is no `use_cache` flag. The
request sequence below selects the cache path.

“Cache exists for the selected action” means the newest cache block contains a
usable handle for that exact action, or its `cold` grant publishes that exact
action. An old database row, an unrelated action, or a consumed, expired, or
stale handle is not usable cache authority.

Before every app-owned mutation:

1. Obtain the newest cache block. Reuse a terminal block only when no read,
   screen change, lifecycle change, window change, or other mutation occurred
   after it; otherwise call `inspect cache` once.
2. Match the caller-selected action exactly. Never substitute a similar label,
   role, selector, or coordinate.
3. If that exact action has an `action_capability` or
   `revalidation_capability`, using that handle is required. Do not replace it
   with an ordinary tap, typed Driver request, coordinate action,
   `cache_policy: "visual_bypass"`, or Computer Use.
4. If the cache state is `cold`, use its `observation_grant` for exactly one
   bound `describe screen` or `take a screenshot`, then use the same grant for
   the exact published action. This bound read is part of preparing the action;
   it is different from a plain read made only to look.
5. Only when the newest bound observation does not publish the selected action
   may the caller use a typed Driver request, if Driver can express it, or
   VisionCapture Computer Use, if it cannot.

A plain read made after cache inspection invalidates the plan to use those
handles. Discard them and inspect again before acting. Never keep a handle from
an older cache block.

## Lane selection

Both lanes live inside the same `mcp__vision-capture__execute` tool. Decide the
lane per selected action, before any input submission:

1. Observe. Use a plain read when only looking. Use `inspect cache` followed by
   the required bound read when preparing an app action.
2. Select exactly one action: your intent plus one exact target.
3. If the exact action maps to a published cache capability or cold-grant
   published action, using it is mandatory → **Driver/Discovery-cache lane**.
   If no cache action is published but a typed Driver request can express it
   (`type`/`enter`, swipe, direct `go back`), also use the Driver lane.
4. If the Driver/WDA lane cannot express or perform that exact **app-owned** action — the
   selected target exists only as a point in the current viewed screenshot,
   such as an unpublished control or a purely visual element with no published
   handle — first prove that VisionCapture can bind one visible Simulator window
   to the exact UDID. Only then use the **VisionCapture Computer Use lane**
   through the same tool. This is VisionCapture's own pointer lane, not
   the Codex/macOS Computer Use plugin or any other app in this Machine. Route the already selected action;
   do not reselect a different target while routing.
   A managed or headless clone with no published visible-window binding is
   ineligible for Computer Use. A streamed device frame does not prove a Mac
   Simulator window exists. Stop that selected action and report the missing
   headless app-control capability; do not call `activate computer use` and do
   not invent a request.
5. Submit once on the chosen lane. Never move the **same** action to another
   lane after submission begins, after a refusal, or on unknown delivery. A
   different lane requires a fresh observation and a new caller decision — that
   is escalation, not a retry (see "When the Driver lane cannot express the
   action" and Refusals).

## Native system alerts use their dedicated route

An iOS permission alert or other native system alert is never a Computer Use
action. Use exactly this sequence:

1. Send `describe system alert`.
2. From that one response, choose exactly one returned button `label` and its
   `content_digest`.
3. Send `press system alert button` once with that exact label and digest.
4. Re-observe with a read.

Do not activate, show, move, click, or hide a Computer Use pointer before,
after, or as fallback for a native system alert. Any `SYSTEM_ALERT_*` refusal is
terminal: end that alert attempt without a pointer task or input. The one
exception is `SYSTEM_ALERT_TAP_DELIVERY_UNKNOWN`: the press may have landed, so
only re-observe read-only; never resubmit it.

## When the Driver lane cannot express an app action

The Driver/Discovery-cache lane can only reach the requested app's own
accessibility tree. When the selected control is app-owned but has no published
handle or typed Driver expression, it may be eligible for Computer Use.

### The tool will tell you to stop. Do not obey it as written.

`CACHE_LIVE_VALIDATION_FAILED` returns this sentence, in second person, at the
exact moment you are deciding what to do next:

> *"If Apple system UI is visible, a person must complete or dismiss it;
> VisionCapture will not act on it."*

**That sentence is scoped to the Driver/Discovery-cache lane.** It describes what
*that lane* will not do. It is not a statement about the product, and it is not an
instruction to end the run or hand off to a human.

Read that refusal as: *"the cache lane is out; decide whether the selected
app-owned control is eligible for Computer Use."* Native system alerts are the
dedicated route above, never a Computer Use escalation.

**Escalation test.** The Driver lane is *proven unable* when all four hold on the
current screen:

1. You took a plain `take a screenshot` and viewed it.
2. The control you selected is visible in that image.
3. The newest evidence publishes no action for it — `inspect cache` cannot
   validate a complete foreground accessibility screen for the requested app, or
   `describe screen` returns only elements the requested app does not own, or it
   returns nothing matching the visible control.
4. You re-observed once and got the same result.

Before escalating, apply the **visible-window gate**. Computer Use is eligible
only when current VisionCapture evidence binds one visible Simulator window to
the exact UDID. A managed recording or preview clone whose response discloses a
headless route is ineligible unless a later public response explicitly publishes
that visible-window binding. Never infer it from a screenshot, framebuffer or
notch tile.

When all four escalation conditions and the visible-window gate hold, use the
Computer Use lane:

- Discard every earlier capability, observation grant, and task tuple. None survive.
- Select a **new** action from the freshly viewed image.
- Derive `x_norm`/`y_norm` from that image and run the Computer Use lifecycle.

When the four escalation conditions hold but the visible-window gate does not,
stop that app action. Preserve the screenshot and cache/description evidence,
report that the Driver lane could not publish the control and that Computer Use
was ineligible, and make no input request. Native system alerts still use their
dedicated route above. There is no public dedicated route for other headless
app-owned controls yet; never imitate the alert verbs or invent one.

**Escalation is not lane-switching.** Only one of these is forbidden:

| Forbidden | Required |
|---|---|
| Re-submitting the **same** action on another lane after it was refused, failed, or had unknown delivery. | Selecting a **new** action on the Computer Use lane after fresh observation proves the Driver lane cannot express it. |

Two identical refusals on the same visible screen are proof that the Driver lane
cannot publish the selected control. They do not prove a Simulator window exists.
Use Computer Use only after the visible-window gate passes; otherwise preserve the
gap and stop that action without asking the operator to bypass it.

This rule is app-agnostic by construction. It keys on what the evidence shows, not
on knowing in advance which screens are hard.

## Driver/Discovery-cache lane

Cache handles are opaque single-use authority. `session_id` only continues
Flow/Learning state and never authorizes cache actions.

| `cache.state` | Do exactly this |
|---|---|
| `ready` | Pick one returned action. Send `tap cached action` for a tap capability, or `execute cached action` for a typed `set_boolean` capability, with only that `action_capability` and `udid`. Never re-send selector, role, desired state, coordinates, or text. |
| `revalidation` | Send `revalidate cached action` with only the returned `revalidation_capability` and `udid`. No selector, role, coordinates, desired state, expected outcome, or historical record data. |
| `mixed` | Use each action's own handle: `action_capability` as `ready`, `revalidation_capability` as `revalidation`. Never combine them. |
| `cold` | The one `observation_grant` authorizes exactly one `describe screen` or `take a screenshot`, then the exact action that observation publishes, with the same grant. If a `cold` block arrives with **no** `observation_grant`, you hold no authority to prepare an action: call `inspect cache` once for a fresh block. If that still returns no grant, run the escalation test. |

**Handles are perishable.** Treat every `observation_grant` and
`action_capability` as valid only for the request you send next. Only
`revalidation_capability` documents a life (fixed 120 seconds); the others state
none and have been observed expiring sooner. Anything that puts time between
receiving a handle and using it — reading a file, a long deliberation, an
unrelated request — risks a `CACHE_ACTION_CAPABILITY_STALE` that proves nothing
about the app. Get the handle, use it, then read.

**Terminal replacement manifests.** Each terminal response replaces the cache
manifest with the next `ready`, `cold`, or unavailable state. Consuming any
capability retires the whole previous manifest, including unconsumed sibling
handles; continue only from the newest returned cache block. Do not insert
screenshot, describe-screen, name-tap, or selector requests between warm
cached actions — a visual bypass invalidates outstanding cache authority.

**Cold continuation.** A published cold tap is `tap <published selector>` with
only `udid` plus the same `observation_grant`; `expected_outcome` is forbidden
there — the tap learns its built-in structural transition instead. A published
cold switch is `execute observed action` with the published typed
`desired_action`, never one invented from prose. Ordinary name taps without a
cold grant or a returned capability are rejected; inspect first. Never target
a control merely because it appeared in OCR or a description.

**Revalidation truth.** A `revalidation_capability` is single-use with a fixed
120-second life. Its terminal payload always reports
`cache_revalidation_used`, `revalidation_verified`, `fresh_authority_recorded`,
and `dispatch_attempted`, with `cache_used=false`. A verified action whose
authority promotion fails is still a success — it reports
`fresh_authority_error_code: "CACHE_REVALIDATION_PROMOTION_FAILED"` — and must
not be replayed. `dispatch_attempted=true` consumes the capability even when
delivery is unproven (`CACHE_REVALIDATION_DISPATCH_UNPROVEN`); keep its
evidence and never retry it.

**Typed Driver requests.**

- `type <text>` / `enter <text>`: when the field is not already focused, pass
  exactly one of `identifier` or `label`. VisionCapture resolves it,
  guarded-taps it, proves focus, then types. When a fresh screenshot or
  description shows that the intended field is already focused, omit both
  selectors. VisionCapture then proves that one current focused element is
  enabled, editable, and owned by the requested app before it types; otherwise
  it refuses before submission. Do not pass a redacted or hashed description
  token as though it were the app's real accessibility identifier. Without a
  typed `expected_outcome`,
  `LEGACY_FIELD_VALUE_CHANGED` may verify the action while
  `interaction_evidence.outcome.status` stays inconclusive
  (`EXPECTED_OUTCOME_MISSING`); `payload.proof.verdict` alone controls
  continuation.
- Direct `go back`: needs a valid flow `session_id`, matching `bundle_id`, and
  exact `udid` — the UDID is canonical, with no retry on another Simulator.
  Proof scope is `navigation_bar_transition`; `verified` requires a structural
  navigation-bar change bound to the same app process.
- Optional typed `expected_outcome` (ordinary driver label taps and
  `click pointer`): `schema_version: 1`; kind `accessibility_exposed`,
  `accessibility_value_equals`, or `accessibility_enabled_equals`; selector is
  exactly one identifier or label. Invalid shapes are rejected with JSON-RPC
  -32602 before any action, and it cannot combine with `verify: false`.
- `verify: false` opts out of outcome verification: the action still runs and
  its verdict is `inconclusive`, never `verified`.

Examples — every value comes from the caller or a runtime response:

```json
{"request":"launch app","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>"}}
{"request":"install app","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>","app_path":"<absolute-path-to-app-bundle.app>"}}
{"request":"inspect cache","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>"}}
{"request":"tap cached action","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>","action_capability":"<returned-capability>"}}
{"request":"revalidate cached action","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>","revalidation_capability":"<returned-capability>"}}
{"request":"describe screen","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>","observation_grant":"<returned-grant>"}}
{"request":"tap <published selector>","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>","observation_grant":"<same-grant>"}}
```

## VisionCapture Computer Use lane

Use this lane for a caller-selected action the Driver/WDA lane cannot express or
perform, decided before any input submission — never as a retry of an action the
Driver lane already submitted. Arriving here through the escalation test is not a
retry: the action is new and the evidence is fresh.

It may act only on app-owned surfaces. Native system alerts, permission alerts,
and system dialogs use the dedicated `describe system alert` / `press system
alert button` route and never enter this lifecycle.

It also requires one visible Simulator window already bound by VisionCapture to
the exact UDID. A headless managed clone, framebuffer stream or notch preview is
not eligible. If current evidence does not publish that window binding, do not
send `activate computer use` as a probe; stop the selected action before input.

Lifecycle, in order:

1. Take a fresh screenshot through VisionCapture and view it. After any
   screen, window, rotation, or geometry change, the previous image is stale.
2. Derive the target point from that viewed image as integer `x_norm`/`y_norm`
   in inclusive 0–1000 device-screen space, origin top-left. Desktop-point
   fields are removed (`POINTER_DESKTOP_COORDINATES_REMOVED`).
3. Send `activate computer use` and keep the returned `computer_use_task_id`
   and `computer_use_generation` (`show pointer` also creates a task;
   `move pointer` repositions with the same tuple). The task carries a
   renewable 120-second idle lease; valid activity renews it. A stale or
   closed task is rejected and never recreated.
4. Send one `click pointer` with the same `udid`, the task tuple, the
   normalized point, a non-empty caller `intent`, and
   `cache_policy: "visual_bypass"`. VisionCapture freshly binds the one exact
   visible Simulator window and current geometry before submitting; the
   human's Mac cursor never moves. If an exact accessibility `target` or typed
   `expected_outcome` is genuinely known, pass it exactly — with a typed
   `expected_outcome`, the pre-click snapshot must match exactly one `target`
   whose frame contains the point or the click refuses before input; without
   one, `target` is recorded, not validated. Never invent a `target` or
   outcome for an unpublished or system-owned control. Optional non-empty
   `expected_text` adds a strict OCR presence check.
5. Inspect the returned evidence and view the before/after images. An after
   image exists only when `stability_status` is `stable` and the image
   encoded; `timed_out`, `superseded`, or `observation_unavailable` never
   rewrites the truthful input-submission fact.
6. While the task is live, bind plain `take a screenshot` / `describe screen`
   reads to the same task tuple; bare reads report
   `BARE_READ_WITH_COMPUTER_USE_OWNER_DEPRECATED`.
7. Send `hide pointer` with the same task tuple when pointer work is done.
   Hide is coordinate-free and closes the task.

```json
{"request":"activate computer use","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>","x_norm":<0-1000>,"y_norm":<0-1000>}}
{"request":"click pointer","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>","x_norm":<0-1000>,"y_norm":<0-1000>,"intent":"<caller-selected action>","cache_policy":"visual_bypass","computer_use_task_id":"<task-id>","computer_use_generation":<n>}}
{"request":"hide pointer","bundle_id":"<bundle-id>","parameters":{"udid":"<udid>","computer_use_task_id":"<task-id>","computer_use_generation":<n>}}
```

Computer Use evidence limits: a click is native submission with **no delivery
acknowledgement**. A click with a valid typed `expected_outcome` can earn a
real `verified` or `failed` through the accessibility evidence gates; OCR
`expected_text` alone never can — it reports `preexisting` or `inconclusive`
and cannot prove absence. `presentation_timing`, when present, is a partial
pointer-surface snapshot, not delivery or outcome proof. With neither a typed
`expected_outcome` nor `expected_text`, the click's semantic proof stays
`inconclusive` by design: judge the viewed before/after evidence yourself. The
window follower handles only pure translation of the same window; after
resize, rotation, or window-identity change, the next explicit action resolves
fresh geometry.

## Proof, for every mutation

No mutating action is complete until this checklist is done:

1. **Before** — viewed the current screenshot, or read the bound observation.
   The newest returned `cache` block is that observation for the actions it
   publishes: do not insert a plain read before consuming a warm capability,
   because that read invalidates the handle.
2. **Bound** — exact `bundle_id`, `udid`, presentation, target (capability /
   selector / window + normalized point), and lane.
3. **Submitted once** — exactly one submission of the selected action.
4. **After** — inspected the returned post-state evidence and opened every
   returned image used as evidence.
5. **Verdict** — read `payload.proof.verdict`, with its `reason_code` and
   `verdict_source`, before continuing.

Keep these facts separate; never flatten them into one claim:

- target identity — what was bound;
- submission — input was submitted, once;
- delivery — acknowledged or unknown (a Computer Use click is always
  unacknowledged);
- stability — the after frame's `stability_status`;
- screen change — pixels or structure differ;
- semantic outcome — the expected result is proven.

`verified` is allowed only when the expected outcome is proven. `failed` is
allowed only when evidence proves the expected outcome contradicted. Otherwise
report `inconclusive` — success prose, a returned receipt, acknowledged input,
stable pixels, a changed screen, OCR non-detection, or timing data never
proves outcome, alone or together. Lack of proof is never proof of failure.
`payload.proof.verdict` is the one authoritative continuation signal; a
retained inconclusive diagnostic envelope never overrides a verified proof.

## Refusals

A refusal is a named technical code — never an outcome and never a safety
judgment. On any refusal: stop that attempt, then report the code plus the
four facts: target binding, submission status, delivery status, and outcome
verdict.

Codes the live contract names: `CACHE_ACTION_CAPABILITY_STALE`,
`CACHE_REVALIDATION_CAPABILITY_REQUIRED`,
`CACHE_REVALIDATION_CAPABILITY_INVALID`, `CACHE_REVALIDATION_CAPABILITY_STALE`,
`CACHE_REVALIDATION_ALREADY_CLAIMED`,
`CACHE_REVALIDATION_CLAIM_PERSISTENCE_FAILED`,
`CACHE_REVALIDATION_DISPATCH_UNPROVEN`, `CACHE_CONFLICTING_TARGET`,
`CACHE_CLEAR_IN_PROGRESS`, `CACHE_LIVE_VALIDATION_FAILED`,
`COMPUTER_USE_TASK_STALE`, `DRIVER_REQUEST_OWNER_MISMATCH`,
`POINTER_DESKTOP_COORDINATES_REMOVED`, and
`BARE_READ_WITH_COMPUTER_USE_OWNER_DEPRECATED`. Owner-mismatch and stale-task
codes reject input submission instead of reporting success — they prove no
dispatch happened.

### Preview-only and Recording V2 refusals

- **Preview wire refusal.** `PREVIEW_INVALID_SHAPE`, `PREVIEW_SCHEMA_UNSUPPORTED`,
  `PREVIEW_COMMAND_INVALID`, the `PREVIEW_*_ID_*` and `PREVIEW_*_UDIDS_*` codes,
  and `PREVIEW_CONTROL_FIELDS_FORBIDDEN` reject the control request before preview
  work. Correct the schema-version-1, control-only request from the contract:
  start has one canonical managed UDID and no ID; status, stop, and cancel have
  the returned canonical preview ID and no UDIDs. Do not add normal-operation
  fields, recording, or a route fallback.
- **Preview unavailable or conflicted.** `PREVIEW_CONFLICT` means another live
  notch owner holds the admission surface. `PREVIEW_ROUTE_UNAVAILABLE` means the
  requested managed route is not usable. Do not substitute a default route, a
  different device, or Recording V2. Preserve the refusal and require a new
  caller decision after the stated condition is corrected.
- **Preview terminal failure.** `PREVIEW_START_FAILED`,
  `PREVIEW_SOURCE_UNAVAILABLE`, and `PREVIEW_ROUTE_LOST` leave no successful
  preview to continue. Preserve the terminal response and fail closed; do not
  start another preview automatically. `PREVIEW_NOT_FOUND` means the ID is not
  active or retained, so do not issue another stop or cancel against it.
- **Recording wire or pairing refusal.** `INVALID_RECORDING_SHAPE`,
  `RECORDING_SCHEMA_UNSUPPORTED`, `RECORDING_COMMAND_INVALID`, the
  `RECORDING_*_ID_*` and `RECORDING_*_UDIDS_*` codes,
  `RECORDING_CONTROL_ONLY_REQUIRED`, `RECORDING_CONTROL_FIELDS_FORBIDDEN`, and
  `RECORDING_ACTION_UDID_MISMATCH` require a corrected schema-version-2 request
  before any new submission. Keep lifecycle controls control-only; keep start
  and continue on a normal operation; preserve the exact accepted recording
  identity. Never remove `continue` or change route to make an app action run.
- **Recording not ready, unavailable, or conflicted.** `RECORDING_NOT_READY`
  means preparation is still active: use the existing control-only status request
  until it is `recording` or terminal, then continue only as documented.
  `RECORDING_CONFLICT`, `RECORDING_DEVICE_UNAVAILABLE`, stream/display/settings
  failures, or a pre-acceptance start failure are not permission to select a
  replacement device or mode. Preserve the reason and obtain a new caller
  decision.
- **Recording terminal failure.** `RECORDING_CHANNEL_FAILED`, a terminal
  incomplete state, `CANCELLED`, `CLIENT_LEASE_EXPIRED`, or
  `RECORDING_NOT_FOUND` is not completed recording proof. If a final stop returns
  one of these, that stop response is final truth: do not send a second stop, a
  second status lookup, or a replacement run to hide the result.

This list is not exhaustive. An unlisted code is still technical, never a safety
verdict or success. It grants no retry, route switch, or other input: preserve
the response and fail closed unless the current public response names a safe
recovery path that is known to occur before submission.

Then take exactly one of these paths:

- **Stale cache authority.** `CACHE_ACTION_CAPABILITY_STALE` proves that no
  input was submitted. Discard the grant and the whole manifest, then follow this
  exactly. The attempt budget is a number, not a judgement call:

  1. Take one plain read and view it.
  2. **Screen changed** → this is a new screen, not a repeat. Make a new action
     decision, `inspect cache`, continue normally. No budget is consumed.
  3. **Screen unchanged** → you get **exactly one** confirming attempt: one
     `inspect cache`, one bound read, one action. Not two. Not "a few".
  4. That confirming attempt refuses the same way → **stop that action and end
     the live cycle.** Report a cache-binding product bug, keeping both refusal
     payloads and the unchanged screenshot as evidence.

  Two stale refusals on the same unchanged screen is the entire budget. A third
  `inspect cache` → bound read on that screen is the forbidden loop. Nothing in
  the tool will stop you — the refusals are byte-identical and carry no attempt
  counter — so this budget is yours to enforce.

  Do not bypass it with coordinates or Computer Use. The escalation test does not
  fire here: `inspect cache` itself succeeded, so the Driver lane is working and
  only the binding failed.

- **Foreign foreground, or the app's screen cannot be validated.**
  `CACHE_LIVE_VALIDATION_FAILED` proves no input was submitted and that the cache
  lane could not bind a complete screen for the requested app. Discard every
  handle. Take one plain read and view it. If another process owns the foreground,
  or nothing published matches the visible control, first determine ownership.
  A native system alert uses its dedicated route; an eligible app-owned control
  may use the Computer Use lane only when the visible-window gate passes. A
  headless app-owned control stops without input until a dedicated public route
  exists. Do not loop `inspect cache`.

- **Proven no dispatch, with a recovery path.** When the response proves input
  was rejected before submission and names a safe next step, that attempt is
  over. Take one fresh observation and, as a new caller decision, select and
  bind a new exact action on either lane — with fresh handles, image,
  coordinates, and task. That is a new action, never a replay of the old
  attempt's authority.
- **Submission began, or delivery or outcome is unknown.** Do not retry, do
  not switch lanes, do not resubmit anything. Re-observe the live screen, keep
  the evidence, and report the uncertainty. The same action is never submitted
  twice.

## Never

- Never route Simulator-app actions through the Codex/macOS Computer Use
  plugin, raw desktop points, or raw HTTP.
- Never use a coordinate click, a prose label, or a semantically similar
  control as automatic recovery for **the same** failed or refused Driver action.
  Escalating to a new action on Computer Use after a fresh observation is a
  different thing, and it is required — see "When the Driver lane cannot express
  the action".
- Never bypass a **usable published** cache action. When nothing is published for
  the visible control there is nothing to bypass. A stale-capability refusal by
  itself is not permission to switch lanes; the escalation test is.
- Never reuse a consumed capability, observation grant, revalidation claim,
  task tuple, screenshot, or derived coordinate after a refusal or terminal
  response.
- Never convert cache prose, success flags, or screen changes into a verdict;
  only `payload.proof.verdict` rules.
- Never use Computer Use for native system alerts, including permission alerts
  and system dialogs. Their dedicated alert route is the only allowed input
  path; a `SYSTEM_ALERT_*` refusal ends that alert attempt.
- Never call `activate computer use` for a headless managed clone or use it to
  probe whether a hidden window might exist. Require an explicit exact-UDID
  visible-window binding before entering the lane.
- Never cite an image as evidence without opening and viewing it.
