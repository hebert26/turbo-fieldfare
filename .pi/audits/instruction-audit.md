# Pi instruction audit — 2026-09-15

**Status: PARTIAL.** The project instruction file is smaller and its remaining routes are valid. Two project instruction references point at resources not present in this checkout. They need an owner decision because fixing them would change the declared Metal-work process.

## Scope and loading map

Pi 0.85.1 documentation and `pi --help` confirm these paths for this checkout:

| Resource | Scope and condition | Loading result |
| --- | --- | --- |
| `AGENTS.md` | Context file from the current directory; layered with ancestor/global files unless disabled by `--no-context-files` | Present. No ancestor or global `AGENTS.md`; no `AGENTS.override.md` or `CLAUDE.md` found. |
| `/Users/dev-machine/.pi/agent/APPEND_SYSTEM.md` | User-wide appended system instructions | Present. Applies beyond this project; not edited. |
| `.pi/APPEND_SYSTEM.md` | Project appended system instructions after project trust | Present. No replacing `SYSTEM.md` exists globally or locally. |
| `.pi/skills/*/SKILL.md` | Project skills after trust | Four valid skills. Descriptions are always advertised; bodies and linked references load on demand. |
| `/Users/dev-machine/.pi/agent/skills/*/SKILL.md` | Global skills | Eight valid skills. Same progressive disclosure model. |
| `.pi/agents/*.md`, `/Users/dev-machine/.pi/agent/agents/*.md` | Separate agent profiles, selected by a task/team; not inherited automatically as main-session instructions | 29 project and 8 global profiles inspected. `.pi/teams.yaml` selects project profiles for declared teams. |
| `.pi/prompts/`, `/Users/dev-machine/.pi/agent/prompts/` | On-demand prompt templates | Neither directory exists. No configured prompt paths. |

`settings.json` only sets provider, model, thinking, theme, and TUI values. No configured skills, prompts, or system files were found. Session, history, and credential paths were excluded.

## Changes

- Rewrote `AGENTS.md` from **1,073** to **306 words**.
- Removed repeated app-control/tutorial detail and stale links from the always-loaded project context.
- Kept the project-specific safety controls: scope limits, document location, model-run gate, one-process rule, loopback-only server, benchmark reporting, and fail-closed image handling.
- Saved the original outside Pi discovery paths at `/Users/dev-machine/Documents/Idea Home/turboCharge/Project-files/instruction-audit-backups/turbo-fieldfare-personal-2026-09-15/AGENTS.md.pre-audit` (SHA-256 `aec5db0f31aab37fa4dd3b48ba295feecf5253c451dc43a775002e9699bebba9`).

## Requirement mapping

| Removed or condensed requirement | Surviving home and route | Reason |
| --- | --- | --- |
| Detailed module, app HUD, chat-state, and installer behaviour | `README.md`; loaded only when a task needs product behaviour | Reference material, not an always-needed constraint. |
| Exact example build/repack/CLI commands | `README.md`; task-specific read | Keeps commands maintained beside user documentation. |
| Detailed server-client workflow | `README.md` and server source/docs when requested | The old `docs/OPENAI_SERVER.md` target is absent. The safety boundary remains in `AGENTS.md`. |
| Detailed runtime-control descriptions | `README.md` and relevant source when requested | The old `docs/RUNTIME_CONTROLS.md` target is absent. |
| Benchmark procedure narrative | Repository benchmark guidance when available; safety/reporting rule remains in `AGENTS.md` | The old `docs/COMMUNITY_BENCHMARKS.md` target is absent. |
| Generic Swift/Metal quality, concurrency, verification, and role rules | `.pi/APPEND_SYSTEM.md`, loaded as project system append; role-specific application is in `mac-*` agent profiles | Project-wide engineering contract already owns these rules. |
| Clear, short communication style | Global `APPEND_SYSTEM.md`; dyslexia-specific instructions remain at the top of `AGENTS.md` | Avoids duplicating the user-wide style policy while retaining the project user need. |

## Findings and unresolved items

- **F1 — missing Metal skill index.** `.pi/APPEND_SYSTEM.md` requires `.pi/apple-official-skills/game-porting-skills/README.md`, but it is absent. It also names `managing-metal4-resources`, `managing-metal4-synchronization`, and `creating-metal4-shader-pipelines`, which are not discoverable project skills. The available `translating-to-metal4-api` skill refers to those missing companion skills. Do not silently replace this route. **Q1:** Should the missing official skill collection be restored, or should the engineering contract be revised to use the checked-in `docs/` references?
- **F2 — stale documentation targets.** The prior `AGENTS.md` links to `docs/OPENAI_SERVER.md`, `docs/COMMUNITY_BENCHMARKS.md`, and `docs/RUNTIME_CONTROLS.md`; none exists. `README.md` contains the same missing links. The project instruction now routes to `README.md` without claiming those files exist. **Q2:** Should the missing guides be restored, or should the README links be updated in a separate documentation task?
- **F3 — unrelated in-progress work.** `.pi/agents/mac-implementer.md`, `.pi/agents/mac-test.md`, `.pi/agents/mac-review.md`, and `plan/` were already modified/untracked. They were inspected but not changed.

## Counts (token estimates are approximate: words × 1.33)

| Set | Before | After |
| --- | ---: | ---: |
| `AGENTS.md` | 1,073 words, ~1,427 tokens | 306 words, ~407 tokens |
| Project always-loaded set: `AGENTS.md` + `.pi/APPEND_SYSTEM.md` | 4,123 words, ~5,483 tokens | 3,356 words, ~4,463 tokens |
| Same set including the user-wide `APPEND_SYSTEM.md` | 5,497 words, ~7,311 tokens | 4,730 words, ~6,291 tokens |
| Project skill descriptions (always advertised) | 367 words, ~488 tokens | unchanged |
| Project skill bodies (on demand) | 4,573 words, ~6,082 tokens | unchanged |
| Global skill descriptions (always advertised) | 459 words, ~610 tokens | unchanged |
| Global skill bodies (on demand) | 6,078 words, ~8,084 tokens | unchanged |

## Validation

- Read Pi context, system-prompt, skill, prompt-template, and trust documentation; confirmed relevant CLI flags with `pi --help`.
- Confirmed no ancestor/global context file, no override, no replacing system prompt, no prompt templates, and no configured extra resource paths.
- Read project/global appended instructions, discovered agent profiles, project/global skill bodies, and the three referenced project Metal reference files.
- Validated all four project skill frontmatter blocks, all project skill relative Markdown links, and 36 of 37 agent filename/name pairs. The one mismatch is F3.
- Re-scanned `AGENTS.md`: it has no Markdown links, so it contains no broken outbound instruction link.
- This was an instruction-only audit. No build, test, model run, or runtime load was claimed or run.

## Agent-profile cleanup — 2026-09-15

**Status: PASS.** This bounded pass covered the 26 non-`ma*` project profiles present before editing. The three `ma*` profiles were deliberately skipped and not edited or audited. Their existing worktree changes remain unrelated.

### Scope and findings

- Agent profiles are separate contexts. `turbo-high`, `smoke-test`, and `observability` reference only the listed profile names in `.pi/teams.yaml`; there are no references to `deepseek-smoke` outside its old profile.
- Removed unreferenced `.pi/agents/deepseek-smoke.md`. It duplicated `smoke-luna` exactly while declaring the same name. `smoke-luna.md` remains the unique selected profile for all three `smoke-test` roles.
- Removed TurboFieldfare module and model-run-preflight copies from `senior-dev-engineer.md`; it already requires reading `AGENTS.md`, the surviving project route for the `.gturbo` owner, loopback boundary, preflight, one-process rule, and test command.
- Condensed the corresponding `luna-test-engineer.md` test/run instructions to the same explicit `AGENTS.md` route.
- Rewrote `worker.md` to remove an unsupported `subagent` API, fictional named-agent availability, and repeated generic editing guidance. Its listed tools remain authoritative.
- Corrected `deepseek-pi-engineer.md`'s stale `.pi/agents/teams.yaml` path to `.pi/teams.yaml`.
- Removed references to the absent `/Users/dev-machine/Dev/VisionOS/VisionCapture/AGENTS.md` from the three legacy VisionCapture profiles that required it. Their distinct external-project roles and `/Users/dev-machine/Dev/VisionOS/AGENTS.md` route remain intact.
- `scout.md` and `scout-luna.md` have intentionally equivalent bodies but distinct names and model selections. There is no verified shared-include mechanism for profile bodies, so merging them would remove a standalone loading route; they remain unchanged.

### Files changed and requirement mapping

| Changed path | Removed or corrected requirement | Surviving route / reason |
| --- | --- | --- |
| `.pi/agents/deepseek-smoke.md` | Duplicate `smoke-luna` profile and mismatched duplicate name | Deleted; `.pi/agents/smoke-luna.md` is the sole profile selected by `.pi/teams.yaml`. |
| `.pi/agents/senior-dev-engineer.md` | Repeated module map and model preflight | `AGENTS.md`, explicitly read by this profile. |
| `.pi/agents/luna-test-engineer.md` | Repeated model-run/process rule wording | `AGENTS.md`, explicitly read by this profile. |
| `.pi/agents/worker.md` | Unsupported subagent workflow and generic repeated instructions | Listed tools and task packet are authoritative; focused worker rules remain. |
| `.pi/agents/deepseek-pi-engineer.md` | Stale team configuration path | `.pi/teams.yaml`, verified present. |
| `.pi/agents/app-agnostic-check.md`, `.pi/agents/code-quality-check.md`, `.pi/agents/team-lead.md` | Absent nested VisionCapture `AGENTS.md` route | Existing external root `AGENTS.md` and `CONTEXT.md` routes remain. |

Pre-edit copies of every in-scope profile are at `/Users/dev-machine/Documents/Idea Home/turboCharge/Project-files/instruction-audit-backups/turbo-fieldfare-personal-2026-09-15/agents-dedup/` outside Pi discovery paths.

### Counts (token estimates are approximate: words × 1.33)

| Edited profile | Before | After |
| --- | ---: | ---: |
| `app-agnostic-check.md` | 704 words, ~936 tokens | 696 words, ~926 tokens |
| `code-quality-check.md` | 838 words, ~1,115 tokens | 826 words, ~1,099 tokens |
| `deepseek-pi-engineer.md` | 508 words, ~676 tokens | 508 words, ~676 tokens |
| `deepseek-smoke.md` (removed) | 236 words, ~314 tokens | 0 words, 0 tokens |
| `luna-test-engineer.md` | 666 words, ~886 tokens | 649 words, ~863 tokens |
| `senior-dev-engineer.md` | 973 words, ~1,294 tokens | 857 words, ~1,140 tokens |
| `team-lead.md` | 605 words, ~805 tokens | 535 words, ~712 tokens |
| `worker.md` | 751 words, ~999 tokens | 107 words, ~142 tokens |
| All non-`ma*` profiles | 12,098 words, ~16,090 tokens | 10,995 words, ~14,623 tokens |

### Validation

- Parsed every remaining non-`ma*` frontmatter block for required keys and unique `name`; every filename/name pair now matches.
- Checked every non-`ma*` `.pi/teams.yaml` agent reference against the profiles; all resolve. The `mac-engineering` names remain outside this task's explicit `ma*` exclusion.
- Confirmed the corrected `.pi/teams.yaml` path exists and the removed nested VisionCapture instruction path does not.
- Re-scanned profiles and project references for `deepseek-smoke`, the absent nested VisionCapture `AGENTS.md`, and `.pi/agents/teams.yaml`; no active reference remains.
- Re-scanned semantic duplicates. The standalone `scout` pair remains by design; other repeated workflow rules have different duties, permissions, or loading conditions.
- This was an instruction-only change. No application build, test, model run, or runtime profile-load execution was applicable or claimed.
