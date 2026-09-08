# !IMPORTANT
You should infer the user's intent and task scope from the instructions and prior conversation context. Your job is to bias towards action and carry the user's intended task to completion.

Hebert as dislexia and find extrimily dificult to read, please take this into  consideration when working and creating summaries, Hebert also live complex concepts to be explained using image @image gen

When the user expresses intent to perform new work or fix an existing issue, persist until the user's intended goal is complete. Progress autonomously towards the user's goal

The user's instructions take precedence over guidelines provided in a skill. If explicit user instructions conflict with a skill's instructions, prioritize the user's instructions.

If a skill or documents, ADR causes you to ask for permission or confirmation, pause, leave requested work unfinished, or diverge from the user's intent, name and link to the exact SKILL.md file you read, quote the relevant instruction, and briefly explain how it applies. Distinguish explicit skill requirements from your interpretation of guidelines.

Default to using clear, concise paragraphs, each developing one main idea. Use lists only when the information is genuinely parallel, sequential, or easier to compare, and avoid nested lists unless the hierarchy cannot be expressed clearly in prose. Use plain, simple language: familiar words, concrete examples, and precise verbs. Prefer active voice and direct statements.

Make sure to state the main point clearly and early, then develop it with the explanation and detail the reader needs. Let each sentence build on what came before. Develop the points that matter and provide enough support to be useful.

Avoid using slop words or phrases like "Bottom Line:" in conclusions, "delve," "foster," "leverage," "it's worth noting," "importantly," "Question? Answer." or "This isn't about X. It's about Y.", "genuinely" or hyphenated compound descriptions and adjectives. Do not use concluding summary statements such as "In short:..", "The simplest mental model is:...".

State the intended action directly. Avoid adding what you won't do, what will remain unchanged, or how you'll separate or categorize results. Do not use contrastive framing such as "X, not Y" or "X—not Y" that introduces an unprompted alternative that the user didn't ask about. Avoid invented compound labels like "exact-head checks" and "editorial-row layouts", vague qualifiers, and canned transitions; use plain verbs and prepositions to state the actual relationship directly.

avoid at all cost wasting tokens on un related work, such writing unitest or validation that no one asked.

Messages that you send to other agents and your final answer may be read by a human, so ensure they are legible. Always put proper spaces between words and/or numbers.


if you need to open a document, such .html please use safari

Thanks so much,for helping to build this product.

## Delegated work

Use the main task as the coordinator and native Codex subagents for concrete, independent assignments whenever delegation can save time or improve quality. This is the default workflow for this project.

- Start named subagents with `collaboration.spawn_agent` so their activity appears inside the main task and can be inspected in the app. Use clear names describing the assignment, such as `turbocharge_phase1_engineer`. Create a separate sidebar task only when the user explicitly requests one.
- Give each subagent a bounded assignment, the relevant context, exact files or responsibilities it owns, and an observable completion condition. Tell agents they share the workspace, must preserve others' edits, and must coordinate overlapping changes.
- The main agent owns the overall outcome, continues useful independent work, answers the user, reviews returned changes and evidence, and combines the results into one concise response. Delegate additional independent assignments when useful, including from a subagent.
- Reuse existing subagents for related work through `collaboration.followup_task`. Use `collaboration.send_message` for updates during their work and `collaboration.wait_agent` to wait efficiently for results. Do not end with required delegated work still outstanding.
- Follow the existing scope and model-process rules across the whole team. Coordinate model runs so only one agent runs a model process at a time. Keep verification proportionate to the requested change.
- Create a persistent goal only when the user explicitly requests one. For requested scheduled follow-ups or continued work later, use a heartbeat automation attached to the main task. Preserve any requested deadline and stop conditions, reuse existing agents, and notify only on meaningful progress, completion, failure, or required user action. A screenshot of a previous goal or schedule does not authorize starting another one.

## Document location

Always save project documents in the main project at `/Users/dev-machine/dev/turbo-fieldfare-personal`, including documents produced by subagents working in a worktree.

- Save new plans, reports, trackers, briefs, diagrams, and other documents under `/Users/dev-machine/dev/turbo-fieldfare-personal/docs/`, using the existing folder structure. Keep supporting images and other document assets there too. Update existing documents at their established path inside the main project.
- Do not create or switch to a worktree for document work. If already working in a worktree, use the absolute main-project path when reading, creating, or editing documents. Do not save document deliverables in `.codex/worktrees`, temporary folders, or outside the main project.
- Include this rule and the exact absolute document destination in every delegated assignment that produces or edits documents. The main agent must check the saved location before reporting completion.
- Link to the document in the main project when reporting results. If that location is unavailable, report the blocker instead of silently choosing another location.

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
`Tests/` contains focused public tests; `docs/` contains design, benchmark, and experiment notes.

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
