# Codebase Reference

Use this when changing source, answering project-shape questions, or deciding where work belongs.

## Required Reading

- `/Users/dev-machine/Dev/VisionOS/AGENTS.md`
- `/Users/dev-machine/Dev/VisionOS/CONTEXT.md` for project shape, vocabulary, and architecture orientation
- `/Users/dev-machine/Dev/VisionOS/VisionCapture/AGENTS.md` for source work under `VisionCapture/`
- Nearest module `CLAUDE.md` or `AGENTS.md` for touched files
- Knowledge index when product/system context is needed:
  - preferred: `/Users/dev-machine/Dev/VisionOS/Project-files/knowledge/index.md`
  - fallback: `/Users/dev-machine/Dev/VisionOS/docs/knowledge/index.md`

## Module Ownership

- `VisionCapture/Sources/App` - app entry point, app shell, settings, scenes, commands, windows.
- `VisionCapture/Sources/UI` - shared SwiftUI components and top-level reusable presentation.
- `VisionCapture/Sources/Features` - workflow studio, flow runner, annotation, capture, export, timeline UI.
- `VisionCapture/Sources/Core` - workflow engine, learning, recovery, persistence, services, LLM integration.
- `VisionCapture/Sources/Interaction` - WDA taps, typing, swipes, sliders, readiness, retries.
- `VisionCapture/Sources/HTTPServer` - MCP bridge, Hummingbird HTTP, JSON-RPC handlers.
- `VisionCapture/Sources/CaptureCore` - zero-dependency capture primitives and PNG handling.
- `VisionCapture/Sources/OCRCore` - zero-dependency Vision framework OCR.
- `VisionCapture/Tests` - XCTest coverage.

## Boundaries

- Do not modify deprecated `VisionCapture/Sources/CLI/` unless the user explicitly asks.
- Do not look for `Tools/` or `Website/` under `VisionCapture/`; they are repo-root folders.
- Keep plans, specs, and durable docs under `/Users/dev-machine/Dev/VisionOS/Project-files/`.
- Keep ADRs under `/Users/dev-machine/Dev/VisionOS/Project-files/adr/`.
- Keep AgentOS role memory under `/Users/dev-machine/Dev/VisionOS/.agentOS/agents/<role-id>/`.
- Keep shared AgentOS memory under `/Users/dev-machine/Dev/VisionOS/.agentOS/memory/`.
- Use `/Users/dev-machine/Dev/VisionOS/.claude/agent-memory/` only for legacy Claude-specific tooling.

## First Questions

- What module owns the behaviour?
- Is this UI, Core policy, HTTP/MCP transport, WDA interaction, capture/OCR, persistence, or docs/config?
- What callers already exist?
- What tests already describe the behaviour?
- Does a module-level `CLAUDE.md` describe a rule that changes the plan?
