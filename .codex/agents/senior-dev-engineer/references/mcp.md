# MCP Reference

Use this for HTTPServer, MCP tools, JSON-RPC, Hummingbird, SSE, health, request/response contracts, and Core
delegation.

## Owned Areas

- `VisionCapture/Sources/HTTPServer/`
- Hummingbird route and lifecycle code;
- JSON-RPC 2.0 request, response, and error contracts;
- MCP tool schemas and argument validation;
- server health and SSE behaviour;
- transport-safe DTOs;
- machine-readable errors that help agents recover;
- delegation from HTTPServer into Core.

## Boundaries

- Keep HTTPServer thin.
- Business logic belongs in Core.
- Preserve the same underlying behaviour for UI and MCP paths.
- Do not leak WDA, OCR, SwiftData, or filesystem details into the transport layer unless they are part of a stable
  response contract.

## Working Rules

- Validate inputs before calling Core.
- Keep responses stable and machine-readable.
- Keep outputs app-agnostic.
- Do not bake in customer app labels or flows.
- When changing behaviour, include a sample JSON-RPC request and expected response shape in the handoff.

## Local Quick Reference

Default local MCP endpoint:

```json
{ "mcpServers": { "vision-capture": { "type": "sse", "url": "http://localhost:8766/mcp" } } }
```

Useful diagnostics:

```bash
pgrep -l VisionCapture
curl http://localhost:8766/health
xcrun simctl list devices booted
```

## Verification

Run `swift build` from `VisionCapture/`. Run focused MCP or HTTP tests for touched handlers. Validate request shape,
response shape, and error codes when contracts change.
