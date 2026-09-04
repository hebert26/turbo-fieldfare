# Logs And Permissions Reference

Use this for local runtime diagnosis, `vision-logs`, app logs, permissions, TCC, Screen Recording, Accessibility,
Simulator access, and WebDriverAgent permission-adjacent failures.

## Logs

The app log helper is:

`/Users/dev-machine/Dev/VisionOS/VisionCapture/scripts/visionlogs.sh`

The `vision-logs` CLI flow:

1. Start with `command -v vision-logs`.
2. Run `vision-logs --json doctor` before deeper reads.
3. If config is missing, initialise with:
   `vision-logs init --repo /Users/dev-machine/Dev/VisionOS`
4. Prefer bounded reads such as `history 10m` or category-limited history.
5. Do not use unbounded raw streaming in JSON mode.

Report commands run, important health state, bounded log evidence, diagnostic bundle path if created, and likely
owner if logs point to a code issue.

## Permissions

Use this for failures that stop VisionCapture from seeing, controlling, recording, or analysing Simulator state.

Check:

- Accessibility permission for UI automation.
- Screen Recording permission for screenshots and window capture.
- Microphone permission for speech or recording features.
- Apple Intelligence availability and local model status when relevant.
- TCC state when symptoms point there.
- Simulator and WDA permission-adjacent failures.

## Working Rules

- Prefer non-mutating diagnostics first: status checks, `sqlite3` reads, `log show`, `simctl`, and process checks.
- Name the exact permission that is missing or stale.
- Explain fixes in plain English.
- Do not reset TCC or run destructive commands unless the user explicitly asks.
- Do not assume a specific customer app.

## Handoff Notes

Report:

- permission or health state found;
- commands run and important output;
- likely cause;
- safest user-facing fix;
- anything that needs app restart, Simulator restart, or System Settings action.
