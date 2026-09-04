---
name: voice
description: "Toggle spoken responses using the vision-voice neural TTS CLI (edge-tts, en-GB-RyanNeural). Use when the user invokes /voice or /voice on to activate spoken responses, or /voice off to deactivate. While active, speak a short, natural summary of each response aloud through the system speaker."
---

# Voice Mode

Adds spoken summaries to responses. Uses the `vision-voice` CLI for speech, with the bundled `scripts/speak.py` as a fallback.

## Activation

- `/voice` or `/voice on` — enable voice mode for the remainder of the session
- `/voice off` — disable voice mode

## Runtime ownership

Use the installed `vision-voice` CLI for speech. Keep spoken summaries short, natural, and free of raw code or long file paths.

```bash
vision-voice speak "Text to speak"
vision-voice speak "Text to speak" --voice en-GB-SoniaNeural
```

If `vision-voice` is not installed or is failing unexpectedly, fall back to the bundled script:

```bash
python3 .codex/skills/voice/scripts/speak.py "Text to speak"
python3 .codex/skills/voice/scripts/speak.py "Text to speak" en-GB-SoniaNeural
```

## Response Workflow

On every response while voice mode is active:

1. Write the normal text response first.
2. Run `vision-voice speak ...` with a short spoken summary that captures the key decision, result, or blocker from the response.

## Rules

- On `/voice` or `/voice on`, confirm that voice mode is enabled.
- On `/voice off`, confirm that voice mode is disabled.
- Keep spoken summaries concise — never read raw diffs, file paths, or shell commands aloud.
- If the user reports voice issues, describe the failure clearly and start diagnosis with `vision-voice --json doctor`.
- If the CLI is unavailable, verify the bundled `scripts/speak.py` path inside this skill is reachable as the fallback.
