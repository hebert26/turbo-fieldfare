---
name: vision-capture-efficient-exploration
description: Customer-facing companion workflow for exploring an unfamiliar iOS Simulator app through VisionCapture while avoiding unnecessary screenshots. Use together with the vision-capture skill when the next action is not known, when a live test should prefer current cache actions and accessibility descriptions, or when image-token use should be kept low without weakening action authority, evidence, or proof rules.
---

# Efficient VisionCapture Exploration

Explore an unfamiliar app with the smallest useful observation.

Prefer this order:

```text
newest cache block → accessibility description → viewed screenshot
```

A screenshot is allowed. Use one when text and cache evidence are not enough.

## Keep the main skill in charge

1. Read and follow the `vision-capture` skill before the first VisionCapture MCP request.
2. Treat this skill only as an observation-order companion.
3. Let the main skill own MCP request shapes, capabilities, grants, action lanes, refusals, proof, and stop rules.
4. If this skill and the main skill disagree, follow the main skill and report the mismatch.

Do not copy internal cache records into the request. Use only the public fields VisionCapture returns.

## Start from the launch result

Send `launch app` as required by the main skill. Read its newest `cache` block before requesting another observation.

VisionCapture may already have inspected the live accessibility screen and returned actions for that exact app,
device, build, and screen. Do not request a screenshot merely to confirm a usable action that is already clear.

## Choose the smallest useful observation

### `ready`, `revalidation`, or `mixed`

Read each returned action's `action`, `selector`, and `role`.

- If the user's intent and one returned action identify the same exact control, consume that action's returned handle
  immediately as the main skill requires.
- If several actions are returned, choose without another observation only when the user's intent or the returned
  selectors make one exact action clear.
- If the meaning is unclear, do not guess. Discard the current handles, call a plain `describe screen`, select one
  exact action from that live description, then call `inspect cache` for fresh authority before acting.

Do not insert `describe screen` or `take a screenshot` between a usable cache handle and its action. A read can make
the handle stale.

### `cold`

Use the returned `observation_grant` with `describe screen` first.

- If the description gives enough information, select one exact published action and use the same grant to submit it
  as required by the main skill.
- If the description is not enough, do not act. Request a fresh cache block and use its new grant with
  `take a screenshot`. View that image before selecting an action.

Never spend one grant on a description and then reuse it for a second observation.

### `unavailable`, no cache block, or no understandable action

Use a plain `describe screen` first to learn the current screen.

Then:

1. Select one exact action only if the description is sufficient.
2. Call `inspect cache` to obtain current authority for that selected action.
3. Use a screenshot only when the description cannot supply the facts required by the main skill or cannot identify
   the intended control safely.

## Use `desired_action` only after selection

`desired_action` is a filter. It is not a discovery request.

Do not send it while asking, “What can I do here?” First learn enough from the newest cache block or a live
description. Then select one action. Only then may `desired_action` narrow a fresh `inspect cache` result to that exact
action.

## Continue from terminal evidence

After every submitted action:

1. Read `payload.proof.verdict` and its reason code.
2. Read the newest returned cache block.
3. If the result and next action are clear, continue from that cache block without an extra description or screenshot.
4. If the proof is inconclusive, follow the main skill's re-observation and no-retry rules. Never repeat the same
   action merely because a screenshot was skipped.

## Take a screenshot when it adds required facts

Take and view a fresh screenshot when any of these is true:

- the main `vision-capture` skill requires it for the selected lane;
- the control is visual and has no useful accessibility name;
- geometry, layout, colour, animation, or other pixel state matters;
- the accessibility description is partial, ambiguous, or contradicts the visible state;
- a system-owned screen must be handled and the main skill requires visual target selection;
- the action outcome cannot be judged from the returned proof and accessibility evidence.

Do not take a screenshot only for reassurance after a verified action with a clear next cache block.

## Examples

### One action is already clear

```text
launch app
→ cache.state: revalidation
→ one tap action with selector and revalidation_capability
→ revalidate cached action
```

No separate screen read is needed.

### A new screen is cold

```text
terminal response
→ cache.state: cold + observation_grant
→ describe screen with that grant
→ choose one published action
→ submit that action once with the same grant
```

No screenshot is needed when the description is sufficient.

### Returned actions are unclear

```text
cache actions are not enough to understand the screen
→ discard their handles
→ plain describe screen
→ select one exact action
→ inspect cache for fresh authority
→ submit once through the main skill's required lane
```

## Report the observation cost

When reporting an exploration run, include:

- cache-only decisions;
- `describe screen` calls;
- screenshots taken;
- the reason each screenshot was necessary;
- actions that remained unclear and were not submitted.

Success means using enough evidence to act correctly. It does not mean reaching zero screenshots.
