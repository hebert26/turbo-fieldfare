# LLM Providers Reference

Use this for AI provider integration used by VisionCapture exploration and workflow features.

## Owned Areas

- `VisionCapture/Sources/Core/LLM/`
- provider adapters for OpenAI, Gemini, local models, or other configured providers;
- model catalogues and capability flags;
- prompt construction contracts;
- streaming and cancellation;
- structured response and tool-call parsing;
- keychain and local configuration handling;
- provider errors presented to UI and MCP callers.

## Boundaries

- Do not own UI layout, WDA actions, or workflow step semantics except where provider output contracts require a
  coordinated Core change.
- Keep provider-specific details inside provider adapters.
- Keep Core-facing contracts structured and stable.
- Keep provider errors actionable for UI and MCP callers.

## Working Rules

- Use official provider docs for API behaviour that may have changed.
- Keep prompts app-agnostic and role-based.
- Do not hardcode one app's screens or labels.
- Keep secrets out of logs, prompts, test fixtures, and source files.
- Make streaming and long requests cancellable.
- Return structured errors callers can act on.
- Preserve model capability flags when adding or changing models.

## Verification

Run `swift build` from `VisionCapture/`. Run focused LLM provider, parser, catalog, key store, or prompt tests. For
live provider checks, report exactly what was run and never expose secrets.

## Handoff Notes

Report:

- provider or prompt contract changed;
- model capability impact;
- privacy and secret-handling notes;
- cancellation behaviour;
- tests run and results;
- live-check limits or risks.
