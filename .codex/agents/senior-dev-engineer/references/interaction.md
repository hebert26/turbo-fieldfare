# Interaction Reference

Use this for WebDriverAgent, tap, type, swipe, slider, readiness, retries, timeouts, and per-device behaviour.

## Owned Areas

- `VisionCapture/Sources/Interaction/`
- Core or HTTPServer call sites that invoke interaction actions.

You own:

- WDA readiness and health checks;
- tap, type, swipe, switch, and slider actions;
- coordinate conversion and safe target selection;
- per-device WDA state and ports;
- retries, timeouts, and fail-fast behaviour;
- action error classification;
- interaction logs needed for diagnosis.

## Boundaries

- Do not own UI layout, LLM strategy, SwiftData schemas, or general MCP design unless the interaction contract is
  directly affected.
- Keep device state per UDID. Do not add single-device global state.
- Keep UI updates off interaction actors.

## Working Rules

- Avoid app-specific labels and screen assumptions.
- Make retries bounded and explainable.
- Preserve cancellation for long waits.
- Keep target selection safe and traceable.
- Keep coordinate conversion rules explicit.

## Verification

Run `swift build` from `VisionCapture/`. Run focused WDA or interaction tests when present. For behaviour depending
on Simulator state, report whether manual verification was run and what device or app was used.

## Handoff Notes

Report:

- action path changed;
- per-device state impact;
- retry and timeout behaviour;
- cancellation story;
- tests or manual checks run;
- remaining device or environment risks.
