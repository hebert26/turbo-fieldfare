# Codex Agent Briefs

These files are a lightweight Codex-facing agent library derived from the richer prompts in `.claude/agents/`.

This repo does not currently have a native Codex agent registry format under `.codex/environments/`, so these briefs are intended as reusable task prompts you can paste or adapt when spawning Codex sub-agents.

## Included Agents

- `senior-ios-engineer.md`
  - Staff-level orchestrator/reviewer for architecture, code review, and high-risk changes.
  - Best mapped to Codex `default` or `explorer` for analysis-heavy work.
- `ios-ui-worker.md`
  - SwiftUI-focused implementation worker for views, composition, bindings, and UI polish.
  - Best mapped to Codex `worker`.
- `ios-services-worker.md`
  - Services/infrastructure worker for managers, data flow, integration points, and concurrency-safe implementation.
  - Best mapped to Codex `worker`.
- `ios-tests-worker.md`
  - Test-focused worker for Swift Testing and XCTest coverage.
  - Best mapped to Codex `worker`.

## How To Use

1. Open the brief that matches the task.
2. Spawn a Codex sub-agent with a narrow ownership scope.
3. Paste the relevant brief into the sub-agent prompt, followed by the concrete assignment.
4. Keep architectural decisions and final review in the main agent.

## Source Mapping

- `senior-ios-engineer.md` ← `.claude/agents/senior-ios-engineer.md`
- `ios-ui-worker.md` ← `.claude/agents/ios-dev-ui.md`
- `ios-services-worker.md` ← `.claude/agents/ios-dev-services.md`
- `ios-tests-worker.md` ← `.claude/agents/ios-dev-tests.md`
