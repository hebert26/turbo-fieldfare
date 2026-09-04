# Architecture Reference

Use this for refactors, hard-to-change areas, coupling, broad design questions, and architecture reviews.

## Architecture Language

- Module: anything with an interface and implementation.
- Interface: everything a caller must know to use a module correctly.
- Implementation: the code hidden behind the interface.
- Depth: useful behaviour behind a small interface.
- Seam: where behaviour can change or be tested without spreading knowledge across callers.
- Adapter: concrete implementation at a seam.
- Leverage: what callers gain when a small interface does a lot of work.
- Locality: what maintainers gain when knowledge, bugs, and changes live in one place.

## Dependency Direction

Preferred direction:

```text
UI / HTTP / MCP / AppKit / SwiftUI
    -> Core application use cases
    -> Domain policies and workflow rules
    -> Interfaces / ports
    -> Infrastructure adapters: WDA, OCR, filesystem, SwiftData, network, LLM providers
```

Rules:

- Keep workflow and automation policy in Core.
- Keep UI thin.
- Keep MCP and UI paths connected through Core.
- Do not put business or workflow policy in SwiftUI views or HTTP handlers.
- Do not let Core depend on UI.
- Wrap external systems behind narrow adapters.
- Prefer per-device and per-session actors over global mutable state.

## Deepening Checks

Look for:

- caller knowledge about ordering rules, retries, state transitions, or configuration;
- duplicated logic across UI, Core, HTTPServer, Interaction, CaptureCore, or OCRCore;
- tests forced to know private setup;
- one-adapter protocols with no real variation;
- app-specific routing rules hidden inside generic automation;
- actor isolation, cancellation, or persistence rules spread across callers.

Use the deletion test:

- If deleting a module makes complexity disappear, the module may be shallow.
- If deleting it spreads complexity across many callers, it is probably earning its place.

## Refactoring Rules

1. Establish current behaviour.
2. Add characterisation tests if behaviour is risky or under-tested.
3. Make the smallest safe structural change.
4. Preserve public behaviour.
5. Keep the diff reviewable.
6. Explain why the new design is deeper, simpler, or safer.

Avoid a giant rewrite when one vertical slice can remove the pain.
