# Working with Hebert

Hebert has dyslexia. Use short, clear paragraphs and everyday language. Explain technical terms when needed. Use images for complex explanations when helpful or requested.

- State each fact once. Describe the outcome, why it matters, and the next step.
- End each response with the most important conclusion or next action.
- Use lists only when they make information easier to scan.
- Avoid jargon, decorative headings, canned phrases, analogies, and contrastive slogans.
- Write agent messages as clearly as user-facing responses.

## Scope and execution

- Infer the task from the current request and relevant conversation. Complete authorized work without repeatedly asking for permission.
- Keep changes within scope. Preserve others' edits and avoid unrelated cleanup or tests.
- Follow explicit user instructions over skill guidelines. If a skill or project rule blocks requested work, link to it, quote the relevant instruction, and explain the blocker.
- Use verification proportionate to the change. Do not claim completion without evidence.
- Open HTML and other browser-viewed documents in Safari.

## Delegated work

- Coordinate concrete, independent assignments through native Codex subagents when useful. Create separate sidebar tasks only when explicitly requested.
- Give each agent a bounded task, exact file ownership, relevant context, and an observable completion condition. Remind agents that the workspace is shared and others' edits must be preserved.
- Reuse existing agents for related work. Review their changes and evidence before reporting completion. Do not leave required delegated work outstanding.
- Coordinate model runs across the team so only one model process runs at a time.
- Create persistent goals only on explicit request. Use a thread heartbeat for requested scheduled follow-ups, preserving deadlines and stop conditions. Notify only on meaningful progress, completion, failure, or required user action.

## Project file location

The destination for all new project-related files is:

`/Users/dev-machine/Documents/Idea Home/turboCharge/`

- Save every new project-related file here unless Hebert explicitly specifies another path. This includes Markdown, HTML, images, plans, reports, trackers, briefs, and diagrams. Reuse existing subject folders and documents.
- The full `Project-files` folder belongs at `/Users/dev-machine/Documents/Idea Home/turboCharge/Project-files/`, preserving its internal structure.
- Do not default new document deliverables to this repository's `docs/`, `Project-files/`, a worktree, or a temporary folder.
- Existing source code, repository instructions/configuration, and app assets may be read and updated at their current paths. This rule does not request moving the existing source checkout.
- Honor explicit user-specified paths. Do not create a worktree for document work.
- Include the exact destination in delegated assignments that create files. Verify the saved location and link to that file when reporting completion.
- If library access is blocked, report the actual error rather than silently saving elsewhere.

# TurboFieldfare

Swift and Metal inference for Gemma 4 26B-A4B on Apple Silicon.

## Scope

This checkout is for running and reporting existing behavior. Do not edit source, change runtime defaults, or start optimization work unless the user asks.

## Layout and commands

`Sources/TurboFieldfareFormat/` owns the Foundation-only `.gturbo` v1 wire
contract. `Sources/TurboFieldfare/` is the runtime; `Sources/TurboFieldfareRepack/`,
`Sources/TurboFieldfareCLI/`, `Sources/TurboFieldfareServer/`, and
`Sources/TurboFieldfareApp/` contain the installer, CLI, loopback server, and
Mac app.
`Tests/` contains focused public tests. Existing `docs/` files provide repository references; new user-facing documents belong in the library above.

```bash
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo --resume
swift build -c release
.build/release/TurboFieldfareMac
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams the pinned model without staging the full source checkpoint. Set `HF_TOKEN` only if requested. The download is about 15 GB. Cancellation preserves verified completed ranges; continue them with `--resume` or remove them with `--discard-partial --output scratch/gemma4.gturbo`.

## Local server

Follow the [server guide](docs/OPENAI_SERVER.md) for launch commands, health
checks, client setup, prompt reuse, tool loops, and supported API behavior.
Apply the model-process checks below first; never start a second model process
or terminate an existing one.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do
not proxy, tunnel, or expose it. A tool call from the local model never bypasses
the client's normal permission policy. Keep the execution session alive while
the server is needed, and stop only a server you launched.

## Test rules

Before a model run, require macOS 26+, Swift 6.2+, enough disk, acceptable `memory_pressure -Q`, a completed `scratch/gemma4.gturbo`, and no process from `pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. If a check fails, inform the user and stop; do not terminate apps or delete or reinstall the model.

Run package tests through `Scripts/test.sh`. Run only one app, CLI, or model-using test at a time.

For performance results, build release once and follow the [community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) exactly. Do not enable experimental controls or profiling.

Do not download a full checkpoint, duplicate the `.gturbo` model, create a worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code, complete timing footer or error, and every protocol deviation. Treat results as measurements, not performance ceilings.

## App controls

The Mac app sends prompts through the pinned Gemma 4 IT chat format. It
exposes context length, temperature, Top-K, Top-P, expert-cache slots, prefill,
and RDADVISE. The defaults are temperature `0.2`, Top-K `64`, and Top-P `0.95`.
The app retains one in-memory conversation and reuses its FP16 KV state across
turns. The HUD shows generation rate, context use, decode-service memory, and,
on hover, cached-token reuse; Last run also shows time to first token and I/O.
Use **New Chat** to clear the transcript, KV lineage, gauge, and retained
images. Reloading or unloading keeps the transcript but marks it outside the
new model context. Build the app with its sibling
`TurboFieldfareDecodeService`; it never loads a second in-process model. See
[README](README.md) and [Runtime controls](docs/RUNTIME_CONTROLS.md).

## Images

Image support is an optional `<name>.vision.gturbo` companion pack that sits
beside the text model. Without it the text runtime behaves exactly as before.

```bash
swift run -c release TurboFieldfareRepack \
  --vision-output scratch/gemma4.vision.gturbo \
  --text-model scratch/gemma4.gturbo
```

The model-free suites cover preprocessing, the tower kernels, the companion
format, the installer transaction, prompt rendering, and the server ingress.
Cases that need a real installed pack skip themselves when
`scratch/gemma4.gturbo` and its companion are missing, so a checkout without a
model still runs green.

Keep the image path fail-closed. If the pack is missing or invalid, say image
support is unavailable. Never accept an image and then answer as though it had
not been sent.
