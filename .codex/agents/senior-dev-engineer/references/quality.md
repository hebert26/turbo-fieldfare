# Quality Reference

Use this for validation, review, QA, test strategy, build failures, concurrency risk, and handoff gates.

## Quality Gate Ownership

- `swift build` from `/Users/dev-machine/Dev/VisionOS/VisionCapture`.
- Focused `swift test` commands for touched areas.
- Broader `swift test` when the change is wide enough.
- SwiftFormat and SwiftLint checks when available and warranted.
- `code-quality-enforcement` skill checks when validating code quality or review readiness.
- Strict concurrency risk review.
- App-agnostic review.
- Missing coverage review.
- Hot-path and main-actor blocking checks.
- Files over 500 lines need review or justification. Files over 1,000 lines need explicit acceptance.

## Review Focus

Find defects that could break the product:

- Swift concurrency correctness;
- actor isolation and reentrancy;
- cancellation paths;
- main-actor blocking;
- Sendable and cross-actor transport;
- SwiftData context safety;
- automation hot-path complexity;
- security and secret handling;
- process, timer, stream, and notification lifetimes;
- app-agnostic automation failures.

## QA Questions

- What should work for a normal user?
- What could easily break?
- What happens when input is missing, wrong, slow, denied, or unavailable?
- Does the app explain failures clearly?
- Does the change work without hardcoded app-specific labels, bundle ids, or screen assumptions?
- Are there tests or manual checks that prove the important behaviour?
- Is anything serious enough to block release?

## Validation Choices

- Role/config/reference changes: TOML validation and line-length sanity check.
- Focused source changes: `swift build` plus focused tests.
- Broad source changes: focused tests first, then `swift test` when practical.
- UI behaviour: relevant tests plus manual app verification where practical.
- MCP changes: focused server/JSON-RPC tests plus sample request/response shape.
- Interaction/capture/OCR changes: focused unit tests first; Simulator/WDA checks when necessary.
- Persistence changes: tied build/tests plus migration/backwards-compatibility checks.

Always report what ran, what passed, and what could not run. Never claim tests passed unless they actually ran.

## Failure Reporting

Separate:

- blocker failures that prevent acceptance;
- warnings or risks that can ship with awareness;
- coverage gaps;
- exact next fix when the gate fails.
