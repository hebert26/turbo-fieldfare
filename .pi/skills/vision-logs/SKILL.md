---
name: vision-logs
description: Use the installed `vision-logs` CLI to monitor VisionCapture logs, inspect local process health, collect bounded log history, and generate diagnostic bundles from any working directory.
---

# Vision Logs

Use this skill when a user wants to inspect VisionCapture logs, monitor tool calls, gather history for debugging, or capture a diagnostic bundle.

## Start

1. Verify the command exists:

```bash
command -v vision-logs
```

2. Check local setup first:

```bash
vision-logs --json doctor
```

3. If config is missing or stale, point the CLI at the repo:

```bash
vision-logs init --repo /Users/dev-machine/Dev/VisionOS
```

## Safe order of use

- Discovery:

```bash
vision-logs --json categories list
vision-logs --json processes list
```

- Safe bounded reads:

```bash
vision-logs --json history 10m --category mcp
vision-logs --json history 5m --tool-calls
```

- Human live monitoring:

```bash
vision-logs stream --tool-calls
vision-logs stream --category workflow
```

- Diagnostic collection:

```bash
vision-logs --json bundle ./visioncapture-diagnostics
```

## Config and auth

- This CLI does not require auth.
- Config lives at `~/.vision-logs/config.toml`.
- Environment overrides take precedence:
  `VISION_LOGS_REPO`, `VISION_LOGS_SCRIPT`, `VISION_LOGS_PROCESS_NAME`

## Raw escape hatch

Use this only when the high-level commands are missing an option you need:

```bash
vision-logs request raw -- --tool-calls -h 15m
```

## Safety

- Prefer `history --json` or `bundle --json` when you need machine-readable output.
- Do not use `request raw` for unbounded live streaming in `--json` mode.
- Do not modify the backend shell script unless the user explicitly asks for backend changes.

## Examples

```bash
vision-logs --json doctor
vision-logs --json history 15m --category mcp --contains "tools/call"
vision-logs stream --tool-calls
```
