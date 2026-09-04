# Workflow Engine Reference

Use this for Core workflow execution, learning, scout, recovery, confidence, cache, and step logic.

## Owned Areas

- `VisionCapture/Sources/Core/Engine/`
- workflow step semantics;
- learning artifacts and replay behaviour;
- scout and exploration flow where it affects workflow execution;
- recovery and confidence decisions;
- cache use during workflow execution;
- shared Core entry points for UI and MCP.

## Boundaries

- Core owns business and workflow logic.
- UI and HTTPServer should call Core rather than duplicating workflow policy.
- Interaction owns low-level WDA actions.
- Persistence schema changes belong to the persistence reference unless the workflow contract requires them.

## Working Rules

- Keep business logic in Core, not in SwiftUI or HTTP handlers.
- Preserve app-agnostic automation. Labels are observations, not routing rules.
- Treat workflow execution as cancellable.
- Watch actor reentrancy around step lists, cache state, and session state.
- Prefer per-device or per-session ownership.
- Keep long-running workflow execution off the main actor.
- Keep retry and recovery logic explainable.

## Verification

Run `swift build` from `VisionCapture/`. Run focused workflow, scout, learning, cache, or recovery tests based on the
changed path. For hot paths, state expected complexity or why the change is not hot-path sensitive.

## Handoff Notes

Report:

- workflow behaviour changed;
- Core entry points affected;
- state and actor ownership;
- cancellation and retry behaviour;
- tests run and results;
- app-agnostic risks.
